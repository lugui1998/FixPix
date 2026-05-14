use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::detection::{PixelWidthDetection, analyze_pixel_width};
use crate::image::{
    RawImage, center_transparent_content, clear_fully_transparent_pixels, crop_transparent_padding,
    decode_image, downscale_ignoring_transparent, encode_image, fit_image_inside_dimensions,
    make_boundary_background_transparent_with_edge_closing, scale_nearest,
};
use crate::mesh::{
    DEFAULT_WARP_SUBDIVISION_DEPTH, DEFAULT_WARP_SUBDIVISION_EDGE_THRESHOLD, DebugSheetOptions,
    MAX_WARP_SUBDIVISION_DEPTH, Mesh, MeshResult, WarpSubdivisionOptions,
    create_debug_sheet_with_options, refine_mesh_to_local_edges_with_boundary_signals,
};
use crate::palette::{PaletteResult, create_palette_image, quantize_image};

const MAX_ANCHORED_ASPECT_DRIFT: f32 = 0.18;
pub const DEFAULT_EDGE_CLOSE_KERNEL_SIZE: u32 = 3;
pub const DEFAULT_MIN_INPUT_WIDTH: u32 = 512;
pub const DEFAULT_MIN_INPUT_HEIGHT: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Png,
    Jpeg,
    Webp,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Auto,
    Full,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteStrategy {
    Global,
    Sampled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownscaleSampleFrom {
    Pixelated,
    Original,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelWidthDetector {
    Projection,
    Hough,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelWidthSource {
    Manual,
    Projection,
    Hybrid,
    Hough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct ArtifactOptions {
    pub palette_path: Option<PathBuf>,
    pub palette_scale: u32,
    pub unscaled_path: Option<PathBuf>,
    pub debug_sheet_path: Option<PathBuf>,
    pub debug_scale: u32,
}

impl Default for ArtifactOptions {
    fn default() -> Self {
        Self {
            palette_path: None,
            palette_scale: 6,
            unscaled_path: None,
            debug_sheet_path: None,
            debug_scale: 6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformOptions {
    pub colors: ColorMode,
    pub palette_merge_threshold: f32,
    pub color_sample_grid_size: u32,
    pub palette_strategy: PaletteStrategy,
    pub scale: Option<u32>,
    pub auto_scale_target: Option<Size>,
    pub downscale: Option<Size>,
    pub downscale_sample_from: DownscaleSampleFrom,
    pub transparent_background: bool,
    pub crop: bool,
    pub crop_size: Option<Size>,
    pub pixel_width: Option<u32>,
    pub pixel_width_detector: PixelWidthDetector,
    pub initial_upscale: u32,
    pub min_input_width: u32,
    pub min_input_height: u32,
    pub edge_close_kernel_size: u32,
    pub warp_subdivision_depth: u32,
    pub warp_subdivision_edge_threshold: f32,
    pub artifacts: ArtifactOptions,
    pub format: OutputFormat,
    pub quality: Option<u8>,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            colors: ColorMode::Auto,
            palette_merge_threshold: 1.0,
            color_sample_grid_size: 5,
            palette_strategy: PaletteStrategy::Global,
            scale: None,
            auto_scale_target: None,
            downscale: None,
            downscale_sample_from: DownscaleSampleFrom::Pixelated,
            transparent_background: false,
            crop: false,
            crop_size: None,
            pixel_width: None,
            pixel_width_detector: PixelWidthDetector::Hybrid,
            initial_upscale: 2,
            min_input_width: DEFAULT_MIN_INPUT_WIDTH,
            min_input_height: DEFAULT_MIN_INPUT_HEIGHT,
            edge_close_kernel_size: DEFAULT_EDGE_CLOSE_KERNEL_SIZE,
            warp_subdivision_depth: DEFAULT_WARP_SUBDIVISION_DEPTH,
            warp_subdivision_edge_threshold: DEFAULT_WARP_SUBDIVISION_EDGE_THRESHOLD,
            artifacts: ArtifactOptions::default(),
            format: OutputFormat::Png,
            quality: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformMetadata {
    pub edge_close_kernel_size: u32,
    pub palette_color_count: usize,
    pub pixel_width_source: PixelWidthSource,
}

#[derive(Debug, Clone)]
pub struct TransformResult {
    pub output: RawImage,
    pub unscaled: RawImage,
    pub detected_pixel_width: u32,
    pub resolved_colors: Option<usize>,
    pub scale_used: u32,
    pub metadata: TransformMetadata,
    pub artifacts: TransformArtifactPaths,
}

#[derive(Debug, Clone, Default)]
pub struct TransformArtifactPaths {
    pub palette_path: Option<PathBuf>,
    pub unscaled_path: Option<PathBuf>,
    pub debug_sheet_path: Option<PathBuf>,
}

pub fn transform_bytes(input: &[u8], options: &TransformOptions) -> Result<TransformResult> {
    validate_options(options)?;
    let decoded = decode_image(input).context("failed to decode input image")?;
    transform_image(decoded, options)
}

pub fn transform_image(decoded: RawImage, options: &TransformOptions) -> Result<TransformResult> {
    validate_options(options)?;
    let prepared = prepare_image(decoded, options)?;
    let sampled = sample_prepared(&prepared, options);
    let PaletteResult {
        image: downsampled,
        resolved_colors,
        palette_colors: downsampled_palette_colors,
    } = quantize_image(
        &sampled,
        options.colors,
        options.palette_merge_threshold,
        options.palette_strategy,
    );
    let base = if let Some(size) = options.crop_size {
        center_transparent_content(&downsampled, size.width, size.height)?
    } else if options.crop {
        crop_transparent_padding(&downsampled)
    } else {
        downsampled
    };
    let palette_colors = if options.crop || options.crop_size.is_some() {
        crate::palette::extract_palette_colors(&base)
    } else {
        downsampled_palette_colors
    };
    let scale = options.scale.unwrap_or_else(|| {
        let target = options.auto_scale_target.unwrap_or(Size {
            width: prepared.decoded.width.max(base.width),
            height: prepared.decoded.height.max(base.height),
        });
        choose_output_scale(
            &base,
            target.width.max(prepared.decoded.width),
            target.height.max(prepared.decoded.height),
        )
    });
    let output = if scale > 1 {
        scale_nearest(&base, scale)
    } else {
        base.clone()
    };
    let artifacts = write_artifacts(
        prepared.debug_image.as_ref().unwrap_or(&prepared.decoded),
        &base,
        &prepared.debug_mesh,
        &palette_colors,
        options,
    )?;

    Ok(TransformResult {
        output,
        unscaled: base,
        detected_pixel_width: prepared.mesh.detected_pixel_width,
        resolved_colors,
        scale_used: prepared.mesh.scale_used,
        metadata: TransformMetadata {
            edge_close_kernel_size: options.edge_close_kernel_size,
            palette_color_count: palette_colors.len(),
            pixel_width_source: prepared.mesh.pixel_width_source,
        },
        artifacts,
    })
}

struct PreparedImage {
    decoded: RawImage,
    debug_image: Option<RawImage>,
    mesh: MeshResult,
    debug_mesh: MeshResult,
    downscale_source: Option<RawImage>,
}

fn prepare_image(decoded: RawImage, options: &TransformOptions) -> Result<PreparedImage> {
    if let Some(size) = options.downscale {
        let transparent = make_boundary_background_transparent_with_edge_closing(
            &decoded,
            options.edge_close_kernel_size,
        );
        let cropped = crop_transparent_padding(&clear_fully_transparent_pixels(&transparent));
        let resized = clear_fully_transparent_pixels(&downscale_ignoring_transparent(
            &cropped,
            size.width,
            size.height,
        ));
        let mesh = MeshResult {
            mesh: Mesh::regular(resized.width, resized.height, 1),
            detected_pixel_width: 1,
            pixel_width_source: PixelWidthSource::Manual,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };
        let debug_mesh = create_downscaled_source_mesh_result(&cropped, size.width, size.height);
        return Ok(PreparedImage {
            decoded: resized,
            debug_image: Some(cropped.clone()),
            mesh,
            debug_mesh,
            downscale_source: Some(cropped),
        });
    }

    let minimum_input_scale = minimum_input_scale(
        decoded.width,
        decoded.height,
        options.min_input_width,
        options.min_input_height,
    )?;
    let decoded = upscale_by_integer_scale(decoded, minimum_input_scale);

    let (detection, boundary_signals) = if let Some(width) = options.pixel_width {
        let scaled_width = width
            .checked_mul(minimum_input_scale)
            .context("minimum input scaling would overflow manual pixel width")?;
        (
            PixelWidthDetection {
                width: scaled_width,
                width_x: scaled_width,
                width_y: scaled_width,
                offset_x: 0,
                offset_y: 0,
                source: PixelWidthSource::Manual,
                anchor_lines_x: None,
                anchor_lines_y: None,
            },
            None,
        )
    } else {
        let analysis = analyze_pixel_width(&decoded, options.pixel_width_detector);
        (
            analysis.detection,
            Some((analysis.edge_projection_x, analysis.edge_projection_y)),
        )
    };
    let (mesh, detected_pixel_width) = if detection.source == PixelWidthSource::Hough {
        match (&detection.anchor_lines_x, &detection.anchor_lines_y) {
            (Some(lines_x), Some(lines_y)) if lines_x.len() > 3 && lines_y.len() > 3 => {
                choose_hough_anchor_mesh(
                    decoded.width,
                    decoded.height,
                    lines_x,
                    lines_y,
                    &detection,
                )
            }
            _ => (
                Mesh::regular_with_offset(
                    decoded.width,
                    decoded.height,
                    detection.width,
                    detection.offset_x,
                    detection.offset_y,
                ),
                detection.width,
            ),
        }
    } else {
        (
            Mesh::regular_with_offset(
                decoded.width,
                decoded.height,
                detection.width,
                detection.offset_x,
                detection.offset_y,
            ),
            detection.width,
        )
    };
    let mesh = MeshResult {
        mesh,
        detected_pixel_width,
        pixel_width_source: detection.source,
        scale_used: options.initial_upscale.max(1),
        debug_crop_offset: (0, 0),
        debug_anchor_lines_x: detection.anchor_lines_x,
        debug_anchor_lines_y: detection.anchor_lines_y,
    };
    let mesh = refine_mesh_to_local_edges_with_boundary_signals(
        &decoded,
        &mesh,
        boundary_signals
            .as_ref()
            .map(|(horizontal, vertical)| (horizontal.as_slice(), vertical.as_slice())),
    );

    Ok(PreparedImage {
        decoded,
        debug_image: None,
        mesh: mesh.clone(),
        debug_mesh: mesh,
        downscale_source: None,
    })
}

fn upscale_by_integer_scale(image: RawImage, scale: u32) -> RawImage {
    if scale > 1 {
        scale_nearest(&image, scale)
    } else {
        image
    }
}

fn minimum_input_scale(width: u32, height: u32, min_width: u32, min_height: u32) -> Result<u32> {
    if width == 0 || height == 0 || (min_width == 0 && min_height == 0) {
        return Ok(1);
    }

    let mut scale = 1u32;
    loop {
        let scaled_width = width
            .checked_mul(scale)
            .context("minimum input scaling would overflow image width")?;
        let scaled_height = height
            .checked_mul(scale)
            .context("minimum input scaling would overflow image height")?;
        let width_ok = min_width == 0 || scaled_width >= min_width;
        let height_ok = min_height == 0 || scaled_height >= min_height;
        if width_ok && height_ok {
            return Ok(scale);
        }
        scale = scale
            .checked_mul(2)
            .context("minimum input scale is too large")?;
    }
}

fn choose_hough_anchor_mesh(
    source_width: u32,
    source_height: u32,
    lines_x: &[u32],
    lines_y: &[u32],
    detection: &PixelWidthDetection,
) -> (Mesh, u32) {
    let resolved_width = detection.width.max(1);

    let anchored = Mesh::complete_from_anchors(
        source_width,
        source_height,
        lines_x,
        lines_y,
        resolved_width,
        resolved_width,
    );
    if anchored_mesh_preserves_aspect(&anchored, source_width, source_height) {
        return (anchored, resolved_width);
    }

    (
        Mesh::regular_from_anchor_phase(
            source_width,
            source_height,
            lines_x,
            lines_y,
            resolved_width,
            detection.offset_x,
            detection.offset_y,
        ),
        resolved_width,
    )
}

fn anchored_mesh_preserves_aspect(mesh: &Mesh, source_width: u32, source_height: u32) -> bool {
    let output_width = mesh.lines_x.len().saturating_sub(1);
    let output_height = mesh.lines_y.len().saturating_sub(1);
    if output_width == 0 || output_height == 0 || source_width == 0 || source_height == 0 {
        return true;
    }
    let source_aspect = source_width as f32 / source_height as f32;
    let mesh_aspect = output_width as f32 / output_height as f32;
    let ratio = source_aspect.max(mesh_aspect) / source_aspect.min(mesh_aspect);
    ratio <= 1.0 + MAX_ANCHORED_ASPECT_DRIFT
}

fn create_downscaled_source_mesh_result(
    source: &RawImage,
    target_width: u32,
    target_height: u32,
) -> MeshResult {
    let fit = fit_image_inside_dimensions(source, target_width, target_height);
    let lines_x = (0..=fit.width)
        .map(|index| {
            ((index as f32 * source.width as f32) / fit.width.max(1) as f32).round() as u32
        })
        .collect::<Vec<_>>();
    let lines_y = (0..=fit.height)
        .map(|index| {
            ((index as f32 * source.height as f32) / fit.height.max(1) as f32).round() as u32
        })
        .collect::<Vec<_>>();
    let average_pixel_width = ((source.width as f32 / fit.width.max(1) as f32)
        .max(source.height as f32 / fit.height.max(1) as f32))
    .round()
    .max(1.0) as u32;

    MeshResult {
        mesh: Mesh {
            lines_x,
            lines_y,
            warp: None,
        },
        detected_pixel_width: average_pixel_width,
        pixel_width_source: PixelWidthSource::Manual,
        scale_used: 1,
        debug_crop_offset: (0, 0),
        debug_anchor_lines_x: None,
        debug_anchor_lines_y: None,
    }
}

fn sample_prepared(prepared: &PreparedImage, options: &TransformOptions) -> RawImage {
    if let (Some(source), DownscaleSampleFrom::Original) =
        (&prepared.downscale_source, options.downscale_sample_from)
    {
        return sample_downscale_from_original(
            source,
            prepared.decoded.width,
            prepared.decoded.height,
            options.color_sample_grid_size,
        );
    }
    crate::mesh::sample_cells_with_warp_and_edge_close_options(
        &prepared.decoded,
        &prepared.mesh.mesh,
        options.color_sample_grid_size,
        options.transparent_background,
        WarpSubdivisionOptions {
            max_depth: options.warp_subdivision_depth,
            edge_threshold: options.warp_subdivision_edge_threshold,
        },
        options.edge_close_kernel_size,
    )
}

fn sample_downscale_from_original(
    image: &RawImage,
    width: u32,
    height: u32,
    sample_grid: u32,
) -> RawImage {
    let fit = fit_image_inside_dimensions(image, width, height);
    let mut out = RawImage::transparent(width, height);
    let scale_x = image.width as f32 / fit.width.max(1) as f32;
    let scale_y = image.height as f32 / fit.height.max(1) as f32;
    for target_y in fit.offset_y..fit.offset_y + fit.height {
        let y0 = ((target_y - fit.offset_y) as f32 * scale_y).floor() as u32;
        let y1 = (((target_y - fit.offset_y + 1) as f32 * scale_y).ceil() as u32).min(image.height);
        for target_x in fit.offset_x..fit.offset_x + fit.width {
            let x0 = ((target_x - fit.offset_x) as f32 * scale_x).floor() as u32;
            let x1 =
                (((target_x - fit.offset_x + 1) as f32 * scale_x).ceil() as u32).min(image.width);
            let color = crate::palette::sample_cell_color(image, x0, x1, y0, y1, sample_grid);
            out.set_pixel(target_x, target_y, color);
        }
    }
    clear_fully_transparent_pixels(&out)
}

fn choose_output_scale(image: &RawImage, target_width: u32, target_height: u32) -> u32 {
    if image.width == 0 || image.height == 0 {
        return 1;
    }
    let ratio_x = target_width as f32 / image.width as f32;
    let ratio_y = target_height as f32 / image.height as f32;
    let ideal = ratio_x.min(ratio_y).max(1.0);
    let lower = ideal.floor().max(1.0) as u32;
    let upper = ideal.ceil().max(1.0) as u32;
    let lower_delta = (ideal - lower as f32).abs();
    let upper_delta = (upper as f32 - ideal).abs();
    if upper_delta <= lower_delta {
        upper
    } else {
        lower
    }
}

fn write_artifacts(
    debug_image: &RawImage,
    unscaled: &RawImage,
    mesh: &MeshResult,
    palette_colors: &[[u8; 3]],
    options: &TransformOptions,
) -> Result<TransformArtifactPaths> {
    let mut paths = TransformArtifactPaths::default();
    if let Some(path) = &options.artifacts.unscaled_path {
        write_image_file(unscaled, path, OutputFormat::Png, options.quality)?;
        paths.unscaled_path = Some(path.clone());
    }
    if let Some(path) = &options.artifacts.palette_path {
        let palette = create_palette_image(palette_colors);
        let palette = if options.artifacts.palette_scale > 1 {
            scale_nearest(&palette, options.artifacts.palette_scale)
        } else {
            palette
        };
        write_image_file(&palette, path, OutputFormat::Png, None)?;
        paths.palette_path = Some(path.clone());
    }
    if let Some(path) = &options.artifacts.debug_sheet_path {
        let debug = create_debug_sheet_with_options(
            debug_image,
            unscaled,
            mesh,
            palette_colors,
            DebugSheetOptions {
                debug_scale: options.artifacts.debug_scale,
                palette_merge_threshold: options.palette_merge_threshold,
                transparent_background: options.transparent_background,
                edge_close_kernel_size: options.edge_close_kernel_size,
                sample_grid: options.color_sample_grid_size,
                warp_subdivision: WarpSubdivisionOptions {
                    max_depth: options.warp_subdivision_depth,
                    edge_threshold: options.warp_subdivision_edge_threshold,
                },
            },
        );
        write_image_file(&debug, path, OutputFormat::Png, None)?;
        paths.debug_sheet_path = Some(path.clone());
    }
    Ok(paths)
}

pub fn write_image_file(
    image: &RawImage,
    path: &std::path::Path,
    format: OutputFormat,
    quality: Option<u8>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = encode_image(image, format, quality)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn validate_options(options: &TransformOptions) -> Result<()> {
    if options.color_sample_grid_size == 0 {
        bail!("color-sample-grid-size must be a positive integer");
    }
    if options.initial_upscale == 0 {
        bail!("initial-upscale must be a positive integer");
    }
    if options.edge_close_kernel_size > 0 && options.edge_close_kernel_size.is_multiple_of(2) {
        bail!("edge-close-kernel-size must be 0 or an odd positive integer");
    }
    if options.warp_subdivision_depth > MAX_WARP_SUBDIVISION_DEPTH {
        bail!("warp-subdivision-depth must be between 0 and {MAX_WARP_SUBDIVISION_DEPTH}");
    }
    if !options.warp_subdivision_edge_threshold.is_finite()
        || options.warp_subdivision_edge_threshold < 0.0
    {
        bail!("warp-subdivision-edge-threshold must be a non-negative finite number");
    }
    if options.scale == Some(0) {
        bail!("scale must be a positive integer");
    }
    if options
        .quality
        .is_some_and(|quality| quality == 0 || quality > 100)
    {
        bail!("quality must be between 1 and 100");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_debug_mesh_uses_source_cell_boundaries() {
        let source = RawImage::transparent(100, 50);
        let mesh = create_downscaled_source_mesh_result(&source, 10, 10);

        assert_eq!(
            mesh.mesh.lines_x,
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        );
        assert_eq!(mesh.mesh.lines_y, vec![0, 10, 20, 30, 40, 50]);
        assert_eq!(mesh.detected_pixel_width, 10);
    }

    #[test]
    fn minimum_input_scale_doubles_until_active_dimensions_are_met() {
        assert_eq!(minimum_input_scale(300, 300, 512, 512).unwrap(), 2);
        assert_eq!(minimum_input_scale(128, 512, 512, 512).unwrap(), 4);
        assert_eq!(minimum_input_scale(64, 20, 0, 512).unwrap(), 32);
        assert_eq!(minimum_input_scale(64, 20, 0, 0).unwrap(), 1);
    }

    #[test]
    fn minimum_input_size_upscales_with_nearest_neighbor_before_detection() {
        let mut image = RawImage::transparent(2, 3);
        image.set_pixel(1, 2, [10, 20, 30, 255]);
        let mut options = TransformOptions {
            pixel_width: Some(1),
            min_input_width: 3,
            min_input_height: 5,
            ..TransformOptions::default()
        };

        let prepared = prepare_image(image.clone(), &options).unwrap();

        assert_eq!((prepared.decoded.width, prepared.decoded.height), (4, 6));
        assert_eq!(prepared.decoded.pixel(2, 4), [10, 20, 30, 255]);

        options.min_input_width = 0;
        options.min_input_height = 0;
        let prepared = prepare_image(image, &options).unwrap();

        assert_eq!((prepared.decoded.width, prepared.decoded.height), (2, 3));
    }

    #[test]
    fn minimum_input_size_scales_manual_pixel_width() {
        let image = RawImage::transparent(4, 4);
        let options = TransformOptions {
            pixel_width: Some(2),
            min_input_width: 8,
            min_input_height: 8,
            ..TransformOptions::default()
        };

        let prepared = prepare_image(image, &options).unwrap();

        assert_eq!((prepared.decoded.width, prepared.decoded.height), (8, 8));
        assert_eq!(prepared.mesh.detected_pixel_width, 4);
        assert_eq!(prepared.mesh.mesh.lines_x, vec![0, 4, 7]);
        assert_eq!(prepared.mesh.mesh.lines_y, vec![0, 4, 7]);
    }

    #[test]
    fn rejects_invalid_warp_subdivision_options() {
        let mut options = TransformOptions::default();
        options.edge_close_kernel_size = 4;
        assert!(
            validate_options(&options)
                .unwrap_err()
                .to_string()
                .contains("edge-close-kernel-size")
        );

        options.edge_close_kernel_size = DEFAULT_EDGE_CLOSE_KERNEL_SIZE;
        options.warp_subdivision_depth = MAX_WARP_SUBDIVISION_DEPTH + 1;
        assert!(
            validate_options(&options)
                .unwrap_err()
                .to_string()
                .contains("warp-subdivision-depth")
        );

        options.warp_subdivision_depth = DEFAULT_WARP_SUBDIVISION_DEPTH;
        options.warp_subdivision_edge_threshold = f32::NAN;
        assert!(
            validate_options(&options)
                .unwrap_err()
                .to_string()
                .contains("warp-subdivision-edge-threshold")
        );
    }

    #[test]
    fn rejects_completed_anchor_meshes_that_distort_source_aspect() {
        let stretched = Mesh {
            lines_x: (0..=180).collect(),
            lines_y: (0..=100).collect(),
            warp: None,
        };
        let square = Mesh {
            lines_x: (0..=100).collect(),
            lines_y: (0..=100).collect(),
            warp: None,
        };

        assert!(!anchored_mesh_preserves_aspect(&stretched, 100, 100));
        assert!(anchored_mesh_preserves_aspect(&square, 100, 100));
    }

    #[test]
    fn high_confidence_anchors_keep_text_fixture_at_anchor_period_without_stretching() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/sources/dragon_tavern_3.png");
        let bytes = std::fs::read(path).unwrap();
        let result = transform_bytes(&bytes, &TransformOptions::default()).unwrap();

        assert!(
            result.detected_pixel_width >= 7,
            "pixel width was {}",
            result.detected_pixel_width
        );
        assert!(
            result.unscaled.width >= 96 && result.unscaled.width <= 140,
            "width was {}",
            result.unscaled.width
        );
        assert!(
            result.unscaled.height >= 96 && result.unscaled.height <= 140,
            "height was {}",
            result.unscaled.height
        );
        assert!(anchored_mesh_preserves_aspect(
            &Mesh {
                lines_x: (0..=result.unscaled.width).collect(),
                lines_y: (0..=result.unscaled.height).collect(),
                warp: None,
            },
            816,
            816,
        ));
    }
}
