use rayon::prelude::*;

use crate::core::{PixelWidthDetector, PixelWidthSource};
use crate::image::{
    ALPHA_THRESHOLD, BackgroundMask, RawImage, apply_background_mask, boundary_background_color,
    boundary_background_mask_with_color_and_edge_closing, choose_closest_integer_scale,
    closed_color_edge_mask, color_distance_sq, color_edge_mask, limit_scale_for_max_dimension,
    scale_nearest,
};
use crate::palette::{has_min_opaque_coverage, sample_cell_color_with_min_opaque_coverage};

const MIN_DEBUG_SEGMENT_LENGTH: u32 = 6;
const MAX_DEBUG_SEGMENTS_PER_FAMILY: usize = 250;
const DEBUG_PALETTE_MAX_SWATCH_SCALE: u32 = 64;
const DEBUG_PALETTE_MAX_WIDTH_RATIO: f32 = 0.36;
const DEBUG_PALETTE_MAX_HEIGHT_RATIO: f32 = 0.24;
const DEBUG_GRID_COLOR: [u8; 4] = [255, 0, 0, 255];
const DEBUG_UNWARPED_GRID_COLOR: [u8; 4] = [80, 200, 255, 255];
const DEBUG_UNUSED_WARPED_GRID_COLOR: [u8; 4] = [255, 80, 220, 120];
const DEBUG_SUBDIVIDED_EDGE_COLOR_A: [u8; 4] = [0, 255, 80, 255];
const DEBUG_SUBDIVIDED_EDGE_COLOR_B: [u8; 4] = [0, 170, 80, 255];
const DEBUG_CLOSED_EDGE_COLOR: [u8; 4] = [40, 130, 255, 255];
const DEBUG_RAW_EDGE_COLOR: [u8; 4] = [255, 210, 0, 255];
const DEBUG_BACKGROUND_MASK_COLOR: [u8; 4] = [0, 190, 255, 255];
const DEBUG_BACKGROUND_COVERAGE_STRONG_COLOR: [u8; 4] = [0, 210, 255, 255];
const DEBUG_BACKGROUND_COVERAGE_MEDIUM_COLOR: [u8; 4] = [255, 170, 0, 255];
const DEBUG_BACKGROUND_COVERAGE_WEAK_COLOR: [u8; 4] = [255, 0, 90, 255];
const BACKGROUND_TRANSPARENT_COVERAGE: u8 = 153;
const BACKGROUND_FRINGE_MIN_COVERAGE: u8 = 64;
const TRANSPARENT_BACKGROUND_MIN_OPAQUE_COVERAGE: f32 = 0.06;
const SAMPLED_BACKGROUND_FRINGE_DISTANCE_LIMIT: i32 = 2500;
const VIVID_SAMPLED_BACKGROUND_FRINGE_DISTANCE_LIMIT: i32 = 18000;
const SAMPLED_LOCAL_OPPOSING_FOREGROUND_DISTANCE_LIMIT: i32 = 8000;
const DARK_SAMPLED_BACKGROUND_FRINGE_DISTANCE_LIMIT: i32 = 8000;
const SAMPLED_BACKGROUND_FRINGE_PASSES: usize = 2;
const DARK_BACKGROUND_FRINGE_LUMA_MAX: u32 = 48;
const VIVID_BACKGROUND_CHROMA_MIN: u8 = 48;
const VIVID_BACKGROUND_VALUE_MIN: u8 = 96;
const SAMPLED_LOCAL_SUPPORT_RADIUS: u32 = 2;
const SAMPLED_LOCAL_BACKGROUND_SUPPORT_REJECT_DISTANCE_LIMIT: i32 = 400;
const SAMPLED_LOCAL_SIMILAR_COLOR_DISTANCE_LIMIT: i32 = 3500;
const SAMPLED_LOCAL_SIMILAR_SUPPORT_MIN: u32 = 3;
const SAMPLED_LOCAL_OPPOSING_SUPPORT_MARGIN: u32 = 3;
const SAMPLED_LOCAL_STRONG_FOREGROUND_SUPPORT_MIN: u32 = 3;
const LOCAL_EDGE_REFINEMENT_RADIUS_RATIO: f32 = 0.14;
const LOCAL_EDGE_REFINEMENT_MAX_RADIUS: u32 = 4;
const LOCAL_EDGE_REFINEMENT_GAP_PENALTY: f32 = 1.15;
const LOCAL_EDGE_REFINEMENT_SHIFT_PENALTY: f32 = 0.2;
const LOCAL_EDGE_REFINEMENT_BAND_RADIUS: usize = 0;
const LOCAL_EDGE_REFINEMENT_SMOOTHING_PASSES: usize = 2;
const LOCAL_EDGE_MIN_PROMINENCE: f32 = 0.08;
const LOCAL_EDGE_MIN_IMPROVEMENT: f32 = 0.03;
const LOCAL_EDGE_STRONG_CONFIDENCE: f32 = 0.42;
const LOCAL_EDGE_WEAK_CONFIDENCE: f32 = 0.18;
const LOCAL_EDGE_CORNER_BONUS: f32 = 0.75;
const LOCAL_EDGE_WARP_MIN_ERROR_GAIN: f32 = 0.1;
pub const DEFAULT_WARP_SUBDIVISION_DEPTH: u32 = 2;
pub const DEFAULT_WARP_SUBDIVISION_EDGE_THRESHOLD: f32 = 18.0;
pub const MAX_WARP_SUBDIVISION_DEPTH: u32 = 4;
const WARP_EDGE_RELIABILITY_MIN_THRESHOLD: f32 = 48.0;
const WARP_EDGE_UNRELIABLE_LOCAL_DENSITY: f32 = 0.035;
const WARP_EDGE_RELIABLE_MIN_COHERENCE: f32 = 0.34;
const WARP_EDGE_RELIABILITY_MIN_TRANSITIONS: u32 = 24;
const WARP_CELL_MAX_CORNER_SHIFT_RATIO: f32 = 0.25;
const WARP_CELL_MAX_CORNER_SHIFT_MIN: f32 = 1.0;
const WARP_CELL_MAX_CORNER_SHIFT_MAX: f32 = 3.0;
const WARP_CELL_MAX_EDGE_CROSS_AXIS_RATIO: f32 = 0.2;
const WARP_CELL_MAX_EDGE_CROSS_AXIS_MIN: f32 = 1.0;
const WARP_CELL_MAX_EDGE_CROSS_AXIS_MAX: f32 = 3.0;
const WARP_CELL_GEOMETRY_LIMIT_STEPS: usize = 16;
const WARP_SUBDIVISION_MIN_EDGE_GAIN: f32 = 1.12;
const WARP_SUBDIVISION_MIN_PROMINENCE: f32 = 0.16;
const WARP_SUBDIVISION_BASELINE_SHIFT_RATIO: f32 = 0.25;
const WARP_SUBDIVISION_BASELINE_SHIFT_MIN: f32 = 1.0;
const WARP_SUBDIVISION_MAX_ADJACENT_SHIFT_DELTA: f32 = 0.5;
const WARP_SUBDIVISION_MAX_UNSUPPORTED_SHIFT: f32 = 2.0;
const WARP_SUBDIVISION_MAX_RADIUS: u32 = 2;
const WARP_SUBDIVISION_CONTOUR_RANK_WEIGHT: f32 = 0.45;
const WARP_SUBDIVISION_SHIFT_PENALTY: f32 = 3.0;
const WARP_CORNER_SNAP_MAX_RADIUS: u32 = 3;
const WARP_CORNER_SHIFT_PENALTY: f32 = 3.0;
const WARP_SAMPLE_COLOR_CAPACITY: usize = 1024;
const ANCHOR_PHASE_TOLERANCE_MIN: u32 = 2;
const ANCHOR_PHASE_TOLERANCE_MAX: u32 = 4;

#[derive(Debug, Clone)]
pub struct Mesh {
    pub lines_x: Vec<u32>,
    pub lines_y: Vec<u32>,
    pub warp: Option<MeshWarp>,
}

#[derive(Debug, Clone)]
pub struct MeshWarp {
    pub lines_x_by_row: Vec<Vec<u32>>,
    pub lines_y_by_column: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy)]
pub struct WarpSubdivisionOptions {
    pub max_depth: u32,
    pub edge_threshold: f32,
}

impl Default for WarpSubdivisionOptions {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_WARP_SUBDIVISION_DEPTH,
            edge_threshold: DEFAULT_WARP_SUBDIVISION_EDGE_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeshResult {
    pub mesh: Mesh,
    pub detected_pixel_width: u32,
    pub pixel_width_source: PixelWidthSource,
    pub scale_used: u32,
    pub debug_crop_offset: (u32, u32),
    pub debug_anchor_lines_x: Option<Vec<u32>>,
    pub debug_anchor_lines_y: Option<Vec<u32>>,
}

impl Mesh {
    pub fn regular(width: u32, height: u32, pixel_width: u32) -> Self {
        Self::regular_with_offset(width, height, pixel_width, 0, 0)
    }

    pub fn regular_with_offset(
        width: u32,
        height: u32,
        pixel_width: u32,
        offset_x: u32,
        offset_y: u32,
    ) -> Self {
        let pixel_width = pixel_width.max(1);
        let lines_x = build_lines(width, pixel_width, offset_x);
        let lines_y = build_lines(height, pixel_width, offset_y);
        Self {
            lines_x,
            lines_y,
            warp: None,
        }
    }

    pub fn complete_from_anchors(
        width: u32,
        height: u32,
        anchor_lines_x: &[u32],
        anchor_lines_y: &[u32],
        width_x: u32,
        width_y: u32,
    ) -> Self {
        Self {
            lines_x: homogenize_lines(
                cluster_line_positions(
                    [vec![0, width.saturating_sub(1)], anchor_lines_x.to_vec()].concat(),
                    4,
                ),
                width_x.max(1),
            ),
            lines_y: homogenize_lines(
                cluster_line_positions(
                    [vec![0, height.saturating_sub(1)], anchor_lines_y.to_vec()].concat(),
                    4,
                ),
                width_y.max(1),
            ),
            warp: None,
        }
    }

    pub fn regular_from_anchor_phase(
        width: u32,
        height: u32,
        anchor_lines_x: &[u32],
        anchor_lines_y: &[u32],
        pixel_width: u32,
        fallback_offset_x: u32,
        fallback_offset_y: u32,
    ) -> Self {
        let pixel_width = pixel_width.max(1);
        Self {
            lines_x: build_lines(
                width,
                pixel_width,
                estimate_phase_from_anchors(
                    anchor_lines_x,
                    pixel_width,
                    fallback_offset_x,
                    width.saturating_sub(1),
                ),
            ),
            lines_y: build_lines(
                height,
                pixel_width,
                estimate_phase_from_anchors(
                    anchor_lines_y,
                    pixel_width,
                    fallback_offset_y,
                    height.saturating_sub(1),
                ),
            ),
            warp: None,
        }
    }
}

fn build_lines(size: u32, pixel_width: u32, offset: u32) -> Vec<u32> {
    let max_value = size.saturating_sub(1);
    let mut lines = vec![0];
    let mut position = offset as i32;
    while position > 0 {
        position -= pixel_width as i32;
    }
    while position <= max_value as i32 {
        if position > 0 && position < max_value as i32 {
            lines.push(position as u32);
        }
        position += pixel_width as i32;
    }
    lines.push(max_value);
    lines.sort_unstable();
    lines.dedup();
    if lines.len() < 2 {
        vec![0, max_value]
    } else {
        lines
    }
}

fn cluster_line_positions(mut positions: Vec<u32>, tolerance: u32) -> Vec<u32> {
    if positions.is_empty() {
        return Vec::new();
    }

    positions.sort_unstable();
    let mut clusters = Vec::<(u64, u32)>::new();
    for position in positions {
        if let Some((sum, count)) = clusters.last_mut() {
            let previous = (*sum / *count as u64) as u32;
            if position.abs_diff(previous) <= tolerance {
                *sum += position as u64;
                *count += 1;
                continue;
            }
        }
        clusters.push((position as u64, 1));
    }

    clusters
        .into_iter()
        .map(|(sum, count)| ((sum as f32 / count as f32).round()) as u32)
        .collect()
}

fn homogenize_lines(lines: Vec<u32>, pixel_width: u32) -> Vec<u32> {
    if lines.len() < 2 || pixel_width == 0 {
        return lines;
    }

    let mut out = Vec::new();
    for pair in lines.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let section_width = end.saturating_sub(start);
        let pixel_count = ((section_width as f32 / pixel_width as f32).round() as u32).max(1);
        let section_pixel_width = section_width as f32 / pixel_count as f32;
        for pixel_index in 0..pixel_count {
            out.push(start + (pixel_index as f32 * section_pixel_width).floor() as u32);
        }
    }
    if let Some(last) = lines.last() {
        out.push(*last);
    }
    cluster_line_positions(out, 0)
}

fn estimate_phase_from_anchors(
    anchors: &[u32],
    pixel_width: u32,
    fallback_phase: u32,
    max_value: u32,
) -> u32 {
    if anchors.is_empty() || pixel_width <= 1 {
        return fallback_phase;
    }

    let interior = anchors
        .iter()
        .copied()
        .filter(|line| *line > 0 && *line < max_value)
        .collect::<Vec<_>>();
    if interior.is_empty() {
        return fallback_phase;
    }

    let fallback_phase = fallback_phase % pixel_width;
    let mut candidates = vec![fallback_phase];
    candidates.extend(interior.iter().map(|line| line % pixel_width));
    candidates.sort_unstable();
    candidates.dedup();

    let tolerance = anchor_phase_tolerance(pixel_width);
    let mut best_phase = fallback_phase;
    let mut best_score = f32::NEG_INFINITY;
    for candidate in candidates {
        let mut score = 0.0f32;
        for anchor in &interior {
            let distance = circular_phase_distance(*anchor % pixel_width, candidate, pixel_width);
            if distance <= tolerance {
                score += 1.0 - distance as f32 / (tolerance + 1) as f32;
            }
        }
        score -= circular_phase_distance(candidate, fallback_phase, pixel_width) as f32 * 0.01;
        if score > best_score {
            best_score = score;
            best_phase = candidate;
        }
    }
    best_phase
}

fn anchor_phase_tolerance(pixel_width: u32) -> u32 {
    ((pixel_width as f32 * 0.18).round() as u32)
        .clamp(ANCHOR_PHASE_TOLERANCE_MIN, ANCHOR_PHASE_TOLERANCE_MAX)
}

fn circular_phase_distance(left: u32, right: u32, period: u32) -> u32 {
    if period <= 1 {
        return 0;
    }
    let delta = left.abs_diff(right);
    delta.min(period - delta)
}

pub fn sample_cells(
    image: &RawImage,
    mesh: &Mesh,
    sample_grid: u32,
    transparent_background: bool,
) -> RawImage {
    sample_cells_with_warp_options(
        image,
        mesh,
        sample_grid,
        transparent_background,
        WarpSubdivisionOptions::default(),
    )
}

pub fn sample_cells_with_warp_options(
    image: &RawImage,
    mesh: &Mesh,
    sample_grid: u32,
    transparent_background: bool,
    warp_options: WarpSubdivisionOptions,
) -> RawImage {
    sample_cells_with_warp_and_edge_close_options(
        image,
        mesh,
        sample_grid,
        transparent_background,
        warp_options,
        0,
    )
}

pub fn sample_cells_with_warp_and_edge_close_options(
    image: &RawImage,
    mesh: &Mesh,
    sample_grid: u32,
    transparent_background: bool,
    warp_options: WarpSubdivisionOptions,
    edge_close_kernel_size: u32,
) -> RawImage {
    let background = transparent_background
        .then(|| boundary_background_color(image))
        .flatten();
    let background_mask = background.map(|background| {
        boundary_background_mask_with_color_and_edge_closing(
            image,
            &background,
            edge_close_kernel_size,
        )
    });
    let transparent_image;
    let sampling_image = if let Some(mask) = &background_mask {
        transparent_image = apply_background_mask(image, mask);
        &transparent_image
    } else {
        image
    };
    let width = mesh.lines_x.len().saturating_sub(1) as u32;
    let height = mesh.lines_y.len().saturating_sub(1) as u32;
    let min_opaque_coverage = if transparent_background {
        TRANSPARENT_BACKGROUND_MIN_OPAQUE_COVERAGE
    } else {
        0.5
    };
    let cells = (0..width as usize * height as usize)
        .into_par_iter()
        .map(|index| {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let mut cell = if mesh.warp.is_some() {
                sample_warped_cell(
                    sampling_image,
                    background_mask.as_ref(),
                    mesh,
                    x as usize,
                    y as usize,
                    sample_grid,
                    warp_options,
                    min_opaque_coverage,
                )
            } else {
                let (x0, x1, y0, y1) =
                    mesh_cell_bounds(mesh, x as usize, y as usize, sampling_image);
                SampledCell {
                    color: sample_cell_color_with_min_opaque_coverage(
                        sampling_image,
                        x0,
                        x1,
                        y0,
                        y1,
                        sample_grid,
                        min_opaque_coverage,
                    ),
                    background_coverage: background_mask
                        .as_ref()
                        .map(|mask| background_coverage_for_bounds(mask, x0, x1, y0, y1))
                        .unwrap_or(0),
                }
            };
            if transparent_background && cell.color[3] < 160 {
                cell.color = [0, 0, 0, 0];
            }
            cell
        })
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(cells.len() * 4);
    let mut coverages = Vec::with_capacity(cells.len());
    for cell in cells {
        out.extend_from_slice(&cell.color);
        coverages.push(cell.background_coverage);
    }
    let mut sampled = RawImage::new(width, height, out);
    if let Some(background) = background {
        remove_sampled_background_fringe(&mut sampled, &coverages, &background);
    }
    sampled
}

#[derive(Debug, Clone, Copy)]
struct SampledCell {
    color: [u8; 4],
    background_coverage: u8,
}

fn sample_warped_cell(
    image: &RawImage,
    background_mask: Option<&BackgroundMask>,
    mesh: &Mesh,
    cell_x: usize,
    cell_y: usize,
    sample_grid: u32,
    warp_options: WarpSubdivisionOptions,
    min_opaque_coverage: f32,
) -> SampledCell {
    let Some(corners) = warped_cell_corners(mesh, cell_x, cell_y) else {
        let (x0, x1, y0, y1) = mesh_cell_bounds(mesh, cell_x, cell_y, image);
        return SampledCell {
            color: sample_cell_color_with_min_opaque_coverage(
                image,
                x0,
                x1,
                y0,
                y1,
                sample_grid,
                min_opaque_coverage,
            ),
            background_coverage: background_mask
                .map(|mask| background_coverage_for_bounds(mask, x0, x1, y0, y1))
                .unwrap_or(0),
        };
    };

    let resolved = resolve_warped_cell(image, mesh, cell_x, cell_y, warp_options, corners);
    let subdivision =
        subdivided_warped_cell_grid_with_options(image, resolved.corners, resolved.options);
    let grid = sample_grid.max(1);
    let center = subdivided_warp_point(&subdivision, 0.5, 0.5);
    let mut keys = [0u32; WARP_SAMPLE_COLOR_CAPACITY];
    let mut counts = [0u32; WARP_SAMPLE_COLOR_CAPACITY];
    let mut distances = [0.0f32; WARP_SAMPLE_COLOR_CAPACITY];
    let mut color_count = 0usize;
    let mut opaque_samples = 0u32;
    let mut background_samples = 0u32;
    let mut total_samples = 0u32;
    let mut best_key = None;
    let mut best_count = 0u32;
    let mut best_distance = f32::INFINITY;

    for sample_y in 0..grid {
        let v = (sample_y as f32 + 0.5) / grid as f32;
        for sample_x in 0..grid {
            let u = (sample_x as f32 + 0.5) / grid as f32;
            let point = subdivided_warp_point(&subdivision, u, v);
            let x = point
                .0
                .round()
                .clamp(0.0, image.width.saturating_sub(1) as f32) as u32;
            let y = point
                .1
                .round()
                .clamp(0.0, image.height.saturating_sub(1) as f32) as u32;
            total_samples += 1;
            if background_mask.is_some_and(|mask| mask.is_background(x, y)) {
                background_samples += 1;
            }
            let sampled = image.pixel(x, y);
            if sampled[3] < ALPHA_THRESHOLD {
                continue;
            }
            opaque_samples += 1;
            let key = pack_rgb([sampled[0], sampled[1], sampled[2]]);
            let distance = (point.0 - center.0).abs() + (point.1 - center.1).abs();

            let slot = keys[..color_count]
                .iter()
                .position(|existing| *existing == key);
            let Some(slot) = slot.or_else(|| {
                if color_count < WARP_SAMPLE_COLOR_CAPACITY {
                    let slot = color_count;
                    keys[slot] = key;
                    color_count += 1;
                    Some(slot)
                } else {
                    None
                }
            }) else {
                continue;
            };

            counts[slot] += 1;
            distances[slot] += distance;
            if counts[slot] > best_count
                || (counts[slot] == best_count && distances[slot] < best_distance)
            {
                best_key = Some(key);
                best_count = counts[slot];
                best_distance = distances[slot];
            }
        }
    }

    if !has_min_opaque_coverage(opaque_samples, total_samples, min_opaque_coverage) {
        return SampledCell {
            color: [0, 0, 0, 0],
            background_coverage: coverage_to_u8(background_samples, total_samples),
        };
    }
    let color = best_key
        .map(unpack_rgb)
        .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
        .unwrap_or([0, 0, 0, 0]);
    SampledCell {
        color,
        background_coverage: coverage_to_u8(background_samples, total_samples),
    }
}

fn background_coverage_for_bounds(mask: &BackgroundMask, x0: u32, x1: u32, y0: u32, y1: u32) -> u8 {
    if x1 <= x0 || y1 <= y0 {
        return 0;
    }
    let mut background = 0u32;
    let mut total = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            total += 1;
            background += u32::from(mask.is_background(x, y));
        }
    }
    coverage_to_u8(background, total)
}

fn coverage_to_u8(background: u32, total: u32) -> u8 {
    if total == 0 {
        return 0;
    }
    ((background * 255 + total / 2) / total).min(255) as u8
}

fn remove_sampled_background_fringe(
    image: &mut RawImage,
    background_coverages: &[u8],
    background: &[u8; 4],
) {
    debug_assert_eq!(
        background_coverages.len(),
        image.width as usize * image.height as usize
    );
    let distance_limit = sampled_background_fringe_distance_limit(background);
    let background_is_dark = background_luma(background) <= DARK_BACKGROUND_FRINGE_LUMA_MAX;
    let background_is_vivid = has_vivid_background_chroma(background);
    for _ in 0..SAMPLED_BACKGROUND_FRINGE_PASSES {
        let mut fringe = Vec::new();
        for y in 0..image.height {
            for x in 0..image.width {
                let index = (y * image.width + x) as usize;
                let pixel = image.pixel(x, y);
                if pixel[3] < ALPHA_THRESHOLD
                    || background_coverages[index] < BACKGROUND_FRINGE_MIN_COVERAGE
                    || color_distance_sq(&pixel, background) > distance_limit
                    || (!touches_transparent_output_neighbor(image, x, y)
                        && !sampled_local_context_rejects_background_candidate(
                            image,
                            background,
                            background_is_vivid,
                            pixel,
                            x,
                            y,
                        ))
                    || sampled_local_support_keeps_background_candidate(
                        image,
                        background,
                        background_is_dark,
                        background_is_vivid,
                        pixel,
                        x,
                        y,
                    )
                {
                    continue;
                }
                fringe.push((x, y));
            }
        }

        if fringe.is_empty() {
            break;
        }

        for (x, y) in fringe {
            image.set_pixel(x, y, [0, 0, 0, 0]);
        }
    }
}

fn sampled_local_context_rejects_background_candidate(
    image: &RawImage,
    background: &[u8; 4],
    background_is_vivid: bool,
    candidate: [u8; 4],
    x: u32,
    y: u32,
) -> bool {
    if !background_is_vivid
        || !shares_background_dominant_channel(candidate, *background)
        || color_distance_sq(&candidate, background) == 0
    {
        return false;
    }
    let support = sampled_local_foreground_support(image, background, candidate, x, y);
    support.strong >= SAMPLED_LOCAL_STRONG_FOREGROUND_SUPPORT_MIN
        && support.opposing >= support.similar + SAMPLED_LOCAL_OPPOSING_SUPPORT_MARGIN
}

fn sampled_background_fringe_distance_limit(background: &[u8; 4]) -> i32 {
    if background_luma(background) <= DARK_BACKGROUND_FRINGE_LUMA_MAX {
        DARK_SAMPLED_BACKGROUND_FRINGE_DISTANCE_LIMIT
    } else if has_vivid_background_chroma(background) {
        VIVID_SAMPLED_BACKGROUND_FRINGE_DISTANCE_LIMIT
    } else {
        SAMPLED_BACKGROUND_FRINGE_DISTANCE_LIMIT
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SampledLocalForegroundSupport {
    similar: u32,
    strong: u32,
    opposing: u32,
}

fn sampled_local_support_keeps_background_candidate(
    image: &RawImage,
    background: &[u8; 4],
    background_is_dark: bool,
    background_is_vivid: bool,
    candidate: [u8; 4],
    x: u32,
    y: u32,
) -> bool {
    if color_distance_sq(&candidate, background) == 0 {
        return false;
    }
    let support = sampled_local_foreground_support(image, background, candidate, x, y);
    let has_similar = support.similar >= SAMPLED_LOCAL_SIMILAR_SUPPORT_MIN;
    let has_strong = support.strong >= SAMPLED_LOCAL_STRONG_FOREGROUND_SUPPORT_MIN;
    if background_is_vivid && shares_background_dominant_channel(candidate, *background) {
        let opposing_blocks =
            support.opposing >= support.similar + SAMPLED_LOCAL_OPPOSING_SUPPORT_MARGIN;
        (has_similar && !opposing_blocks) || (!has_strong && !opposing_blocks)
    } else if background_is_dark {
        has_similar || has_strong
    } else {
        false
    }
}

fn sampled_local_foreground_support(
    image: &RawImage,
    background: &[u8; 4],
    candidate: [u8; 4],
    x: u32,
    y: u32,
) -> SampledLocalForegroundSupport {
    let x0 = x.saturating_sub(SAMPLED_LOCAL_SUPPORT_RADIUS);
    let x1 = x
        .saturating_add(SAMPLED_LOCAL_SUPPORT_RADIUS)
        .min(image.width.saturating_sub(1));
    let y0 = y.saturating_sub(SAMPLED_LOCAL_SUPPORT_RADIUS);
    let y1 = y
        .saturating_add(SAMPLED_LOCAL_SUPPORT_RADIUS)
        .min(image.height.saturating_sub(1));
    let mut support = SampledLocalForegroundSupport::default();
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            if xx == x && yy == y {
                continue;
            }
            let pixel = image.pixel(xx, yy);
            if pixel[3] < ALPHA_THRESHOLD {
                continue;
            }
            if color_distance_sq(&pixel, background)
                <= SAMPLED_LOCAL_BACKGROUND_SUPPORT_REJECT_DISTANCE_LIMIT
            {
                continue;
            }
            let weight = sampled_local_support_weight(x.abs_diff(xx), y.abs_diff(yy));
            if color_distance_sq(&pixel, &candidate) <= SAMPLED_LOCAL_SIMILAR_COLOR_DISTANCE_LIMIT {
                support.similar += weight;
            }
            if !shares_background_dominant_channel(pixel, *background)
                && color_distance_sq(&pixel, background)
                    > SAMPLED_LOCAL_OPPOSING_FOREGROUND_DISTANCE_LIMIT
            {
                support.opposing += weight;
            }
            if color_distance_sq(&pixel, background)
                > SAMPLED_LOCAL_OPPOSING_FOREGROUND_DISTANCE_LIMIT
            {
                support.strong += weight;
            }
        }
    }
    support
}

fn sampled_local_support_weight(dx: u32, dy: u32) -> u32 {
    if dx <= 1 && dy <= 1 { 3 } else { 1 }
}

fn has_vivid_background_chroma(pixel: &[u8; 4]) -> bool {
    let max = pixel[0].max(pixel[1]).max(pixel[2]);
    let min = pixel[0].min(pixel[1]).min(pixel[2]);
    max >= VIVID_BACKGROUND_VALUE_MIN && max - min >= VIVID_BACKGROUND_CHROMA_MIN
}

fn background_luma(pixel: &[u8; 4]) -> u32 {
    (299 * pixel[0] as u32 + 587 * pixel[1] as u32 + 114 * pixel[2] as u32) / 1000
}

fn shares_background_dominant_channel(pixel: [u8; 4], background: [u8; 4]) -> bool {
    let pixel_max = pixel[0].max(pixel[1]).max(pixel[2]);
    let background_max = background[0].max(background[1]).max(background[2]);
    (0..3).any(|channel| {
        pixel[channel].saturating_add(8) >= pixel_max
            && background[channel].saturating_add(8) >= background_max
    })
}

fn touches_transparent_output_neighbor(image: &RawImage, x: u32, y: u32) -> bool {
    (x > 0 && image.pixel(x - 1, y)[3] < ALPHA_THRESHOLD)
        || (y > 0 && image.pixel(x, y - 1)[3] < ALPHA_THRESHOLD)
        || (x + 1 < image.width && image.pixel(x + 1, y)[3] < ALPHA_THRESHOLD)
        || (y + 1 < image.height && image.pixel(x, y + 1)[3] < ALPHA_THRESHOLD)
}

#[derive(Debug, Clone, Copy)]
struct WarpedCellCorners {
    top_left: (f32, f32),
    top_right: (f32, f32),
    bottom_left: (f32, f32),
    bottom_right: (f32, f32),
}

fn warped_cell_corners(mesh: &Mesh, cell_x: usize, cell_y: usize) -> Option<WarpedCellCorners> {
    let warp = mesh.warp.as_ref()?;
    let row_count = mesh.lines_y.len().checked_sub(1)?;
    let column_count = mesh.lines_x.len().checked_sub(1)?;
    if cell_x >= column_count || cell_y >= row_count {
        return None;
    }

    Some(WarpedCellCorners {
        top_left: (
            warped_x_at_boundary(mesh, warp, cell_y, cell_x) as f32,
            warped_y_at_boundary(mesh, warp, cell_x, cell_y) as f32,
        ),
        top_right: (
            warped_x_at_boundary(mesh, warp, cell_y, cell_x + 1) as f32,
            warped_y_at_boundary(mesh, warp, cell_x + 1, cell_y) as f32,
        ),
        bottom_left: (
            warped_x_at_boundary(mesh, warp, cell_y + 1, cell_x) as f32,
            warped_y_at_boundary(mesh, warp, cell_x, cell_y + 1) as f32,
        ),
        bottom_right: (
            warped_x_at_boundary(mesh, warp, cell_y + 1, cell_x + 1) as f32,
            warped_y_at_boundary(mesh, warp, cell_x + 1, cell_y + 1) as f32,
        ),
    })
}

#[derive(Debug, Clone, Copy)]
struct ResolvedWarpedCell {
    corners: WarpedCellCorners,
    options: WarpSubdivisionOptions,
    uses_warped_corners: bool,
}

fn resolve_warped_cell(
    image: &RawImage,
    mesh: &Mesh,
    cell_x: usize,
    cell_y: usize,
    options: WarpSubdivisionOptions,
    corners: WarpedCellCorners,
) -> ResolvedWarpedCell {
    let unwarped_corners = unwarped_cell_corners(mesh, cell_x, cell_y);
    let (corners, uses_warped_corners) = limit_warped_cell_geometry(unwarped_corners, corners);
    if !uses_warped_corners
        || !warped_cell_edge_region_reliable(image, corners, options.edge_threshold)
    {
        return ResolvedWarpedCell {
            corners: unwarped_corners,
            options: WarpSubdivisionOptions {
                max_depth: 0,
                ..options
            },
            uses_warped_corners: false,
        };
    }

    let corners = if options.max_depth > 0 {
        refined_warped_cell_corners(image, corners, options.edge_threshold)
    } else {
        corners
    };
    let (corners, uses_refined_corners) = limit_warped_cell_geometry(unwarped_corners, corners);
    if !uses_refined_corners {
        return ResolvedWarpedCell {
            corners: unwarped_corners,
            options: WarpSubdivisionOptions {
                max_depth: 0,
                ..options
            },
            uses_warped_corners: false,
        };
    }
    ResolvedWarpedCell {
        corners,
        options,
        uses_warped_corners: true,
    }
}

fn refined_warped_cell_corners(
    image: &RawImage,
    corners: WarpedCellCorners,
    edge_threshold: f32,
) -> WarpedCellCorners {
    if !warped_cell_edge_region_reliable(image, corners, edge_threshold) {
        return corners;
    }

    WarpedCellCorners {
        top_left: refine_warped_corner_point(image, corners.top_left, edge_threshold),
        top_right: refine_warped_corner_point(image, corners.top_right, edge_threshold),
        bottom_left: refine_warped_corner_point(image, corners.bottom_left, edge_threshold),
        bottom_right: refine_warped_corner_point(image, corners.bottom_right, edge_threshold),
    }
}

fn warped_cell_geometry_reliable(unwarped: WarpedCellCorners, warped: WarpedCellCorners) -> bool {
    let width = point_distance(unwarped.top_left, unwarped.top_right)
        .max(point_distance(unwarped.bottom_left, unwarped.bottom_right));
    let height = point_distance(unwarped.top_left, unwarped.bottom_left)
        .max(point_distance(unwarped.top_right, unwarped.bottom_right));
    let min_span = width.min(height).max(1.0);
    let max_corner_shift = (min_span * WARP_CELL_MAX_CORNER_SHIFT_RATIO).clamp(
        WARP_CELL_MAX_CORNER_SHIFT_MIN,
        WARP_CELL_MAX_CORNER_SHIFT_MAX,
    );
    let max_cross_axis_shift = (min_span * WARP_CELL_MAX_EDGE_CROSS_AXIS_RATIO).clamp(
        WARP_CELL_MAX_EDGE_CROSS_AXIS_MIN,
        WARP_CELL_MAX_EDGE_CROSS_AXIS_MAX,
    );

    if point_distance(unwarped.top_left, warped.top_left) > max_corner_shift
        || point_distance(unwarped.top_right, warped.top_right) > max_corner_shift
        || point_distance(unwarped.bottom_left, warped.bottom_left) > max_corner_shift
        || point_distance(unwarped.bottom_right, warped.bottom_right) > max_corner_shift
    {
        return false;
    }

    (warped.top_left.1 - warped.top_right.1).abs() <= max_cross_axis_shift
        && (warped.bottom_left.1 - warped.bottom_right.1).abs() <= max_cross_axis_shift
        && (warped.top_left.0 - warped.bottom_left.0).abs() <= max_cross_axis_shift
        && (warped.top_right.0 - warped.bottom_right.0).abs() <= max_cross_axis_shift
}

fn limit_warped_cell_geometry(
    unwarped: WarpedCellCorners,
    target: WarpedCellCorners,
) -> (WarpedCellCorners, bool) {
    if warped_cell_geometry_reliable(unwarped, target) {
        return (target, true);
    }

    let mut best = unwarped;
    let mut moved = false;
    for step in 1..=WARP_CELL_GEOMETRY_LIMIT_STEPS {
        let t = step as f32 / WARP_CELL_GEOMETRY_LIMIT_STEPS as f32;
        let candidate = interpolate_warped_cell_corners(unwarped, target, t);
        if !warped_cell_geometry_reliable(unwarped, candidate) {
            break;
        }
        best = candidate;
        moved = true;
    }
    (best, moved)
}

fn interpolate_warped_cell_corners(
    start: WarpedCellCorners,
    end: WarpedCellCorners,
    t: f32,
) -> WarpedCellCorners {
    WarpedCellCorners {
        top_left: interpolate_point(start.top_left, end.top_left, t),
        top_right: interpolate_point(start.top_right, end.top_right, t),
        bottom_left: interpolate_point(start.bottom_left, end.bottom_left, t),
        bottom_right: interpolate_point(start.bottom_right, end.bottom_right, t),
    }
}

fn interpolate_point(start: (f32, f32), end: (f32, f32), t: f32) -> (f32, f32) {
    (
        start.0 * (1.0 - t) + end.0 * t,
        start.1 * (1.0 - t) + end.1 * t,
    )
}

fn refine_warped_corner_point(
    image: &RawImage,
    point: (f32, f32),
    edge_threshold: f32,
) -> (f32, f32) {
    if image.width < 2 || image.height < 2 {
        return point;
    }

    let radius = WARP_CORNER_SNAP_MAX_RADIUS as i32;
    let original_x = point.0.round() as i32;
    let original_y = point.1.round() as i32;
    let min_x = (original_x - radius).max(1);
    let max_x = (original_x + radius).min(image.width.saturating_sub(1) as i32);
    let min_y = (original_y - radius).max(1);
    let max_y = (original_y + radius).min(image.height.saturating_sub(1) as i32);
    if min_x > max_x || min_y > max_y {
        return point;
    }

    let threshold = edge_threshold.max(0.0);
    let base_score = color_corner_score(image, original_x as u32, original_y as u32);
    let mut best_x = original_x;
    let mut best_y = original_y;
    let mut best_score = base_score;
    let mut best_rank = base_score;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let score = color_corner_score(image, x as u32, y as u32);
            let shift = (x.abs_diff(original_x) + y.abs_diff(original_y)) as f32;
            let rank = score - shift * WARP_CORNER_SHIFT_PENALTY;
            if rank > best_rank {
                best_rank = rank;
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }

    if subdivision_snap_weight(base_score, best_score, threshold).is_none() {
        return point;
    }
    (best_x as f32, best_y as f32)
}

fn color_corner_score(image: &RawImage, x: u32, y: u32) -> f32 {
    color_vertical_point_score(image, x, y).min(color_horizontal_point_score(image, x, y))
}

fn color_vertical_point_score(image: &RawImage, x: u32, y: u32) -> f32 {
    if x == 0 || x >= image.width {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut weight_total = 0.0f32;
    for yy in y.saturating_sub(1)..=(y + 1).min(image.height.saturating_sub(1)) {
        let weight = if yy == y { 1.0 } else { 0.5 };
        total += color_edge_delta(image.pixel(x - 1, yy), image.pixel(x, yy)) as f32 * weight;
        weight_total += weight;
    }
    total / weight_total.max(1.0)
}

fn color_horizontal_point_score(image: &RawImage, x: u32, y: u32) -> f32 {
    if y == 0 || y >= image.height {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut weight_total = 0.0f32;
    for xx in x.saturating_sub(1)..=(x + 1).min(image.width.saturating_sub(1)) {
        let weight = if xx == x { 1.0 } else { 0.5 };
        total += color_edge_delta(image.pixel(xx, y - 1), image.pixel(xx, y)) as f32 * weight;
        weight_total += weight;
    }
    total / weight_total.max(1.0)
}

fn warped_cell_edge_region_reliable(
    image: &RawImage,
    corners: WarpedCellCorners,
    edge_threshold: f32,
) -> bool {
    let Some((x0, x1, y0, y1)) = warped_cell_edge_region_bounds(image, corners) else {
        return true;
    };
    edge_region_reliable(image, x0, x1, y0, y1, edge_threshold)
}

fn warped_cell_edge_region_bounds(
    image: &RawImage,
    corners: WarpedCellCorners,
) -> Option<(u32, u32, u32, u32)> {
    if image.width == 0 || image.height == 0 {
        return None;
    }
    let radius = WARP_SUBDIVISION_MAX_RADIUS as f32;
    let min_x = corners
        .top_left
        .0
        .min(corners.top_right.0)
        .min(corners.bottom_left.0)
        .min(corners.bottom_right.0)
        .floor()
        - radius;
    let max_x = corners
        .top_left
        .0
        .max(corners.top_right.0)
        .max(corners.bottom_left.0)
        .max(corners.bottom_right.0)
        .ceil()
        + radius;
    let min_y = corners
        .top_left
        .1
        .min(corners.top_right.1)
        .min(corners.bottom_left.1)
        .min(corners.bottom_right.1)
        .floor()
        - radius;
    let max_y = corners
        .top_left
        .1
        .max(corners.top_right.1)
        .max(corners.bottom_left.1)
        .max(corners.bottom_right.1)
        .ceil()
        + radius;
    let x0 = min_x.max(0.0) as u32;
    let x1 = max_x.min(image.width as f32).ceil() as u32;
    let y0 = min_y.max(0.0) as u32;
    let y1 = max_y.min(image.height as f32).ceil() as u32;
    (x0 < x1 && y0 < y1).then_some((x0, x1, y0, y1))
}

fn edge_region_reliable(
    image: &RawImage,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    edge_threshold: f32,
) -> bool {
    let stats = edge_region_stats(image, x0, x1, y0, y1, edge_threshold);
    stats.total < WARP_EDGE_RELIABILITY_MIN_TRANSITIONS
        || stats.density <= WARP_EDGE_UNRELIABLE_LOCAL_DENSITY
        || stats.coherence >= WARP_EDGE_RELIABLE_MIN_COHERENCE
}

#[derive(Debug, Clone, Copy)]
struct EdgeRegionStats {
    total: u32,
    density: f32,
    coherence: f32,
}

fn edge_region_stats(
    image: &RawImage,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    edge_threshold: f32,
) -> EdgeRegionStats {
    let x0 = x0.min(image.width);
    let x1 = x1.min(image.width).max(x0);
    let y0 = y0.min(image.height);
    let y1 = y1.min(image.height).max(y0);
    let threshold = edge_threshold
        .max(WARP_EDGE_RELIABILITY_MIN_THRESHOLD)
        .round() as u32;
    let mut strong = 0u32;
    let mut total = 0u32;
    let mut vertical_strong = 0u32;
    let mut horizontal_strong = 0u32;
    let mut vertical_by_x = vec![0u32; x1.saturating_sub(x0) as usize];
    let mut horizontal_by_y = vec![0u32; y1.saturating_sub(y0) as usize];

    for y in y0..y1 {
        for x in x0..x1 {
            let current = image.pixel(x, y);
            if x + 1 < x1 {
                total += 1;
                if color_edge_delta(current, image.pixel(x + 1, y)) >= threshold {
                    strong += 1;
                    vertical_strong += 1;
                    if let Some(count) = vertical_by_x.get_mut((x - x0) as usize) {
                        *count += 1;
                    }
                }
            }
            if y + 1 < y1 {
                total += 1;
                if color_edge_delta(current, image.pixel(x, y + 1)) >= threshold {
                    strong += 1;
                    horizontal_strong += 1;
                    if let Some(count) = horizontal_by_y.get_mut((y - y0) as usize) {
                        *count += 1;
                    }
                }
            }
        }
    }

    let density = if total == 0 {
        0.0
    } else {
        strong as f32 / total as f32
    };
    let vertical_coherence = vertical_by_x
        .into_iter()
        .max()
        .filter(|_| vertical_strong > 0)
        .map(|max| max as f32 / vertical_strong as f32)
        .unwrap_or(0.0);
    let horizontal_coherence = horizontal_by_y
        .into_iter()
        .max()
        .filter(|_| horizontal_strong > 0)
        .map(|max| max as f32 / horizontal_strong as f32)
        .unwrap_or(0.0);
    EdgeRegionStats {
        total,
        density,
        coherence: vertical_coherence.max(horizontal_coherence),
    }
}

fn warped_x_at_boundary(
    mesh: &Mesh,
    warp: &MeshWarp,
    row_boundary: usize,
    line_index: usize,
) -> u32 {
    let row_count = mesh.lines_y.len().saturating_sub(1);
    if row_count == 0 {
        return mesh.lines_x[line_index];
    }
    if row_boundary == 0 {
        return warped_row_x(mesh, warp, 0, line_index);
    }
    if row_boundary >= row_count {
        return warped_row_x(mesh, warp, row_count - 1, line_index);
    }
    let top = warped_row_x(mesh, warp, row_boundary - 1, line_index);
    let bottom = warped_row_x(mesh, warp, row_boundary, line_index);
    ((top + bottom) as f32 / 2.0).round() as u32
}

fn warped_y_at_boundary(
    mesh: &Mesh,
    warp: &MeshWarp,
    column_boundary: usize,
    line_index: usize,
) -> u32 {
    let column_count = mesh.lines_x.len().saturating_sub(1);
    if column_count == 0 {
        return mesh.lines_y[line_index];
    }
    if column_boundary == 0 {
        return warped_column_y(mesh, warp, 0, line_index);
    }
    if column_boundary >= column_count {
        return warped_column_y(mesh, warp, column_count - 1, line_index);
    }
    let left = warped_column_y(mesh, warp, column_boundary - 1, line_index);
    let right = warped_column_y(mesh, warp, column_boundary, line_index);
    ((left + right) as f32 / 2.0).round() as u32
}

fn warped_row_x(mesh: &Mesh, warp: &MeshWarp, row: usize, line_index: usize) -> u32 {
    warp.lines_x_by_row
        .get(row)
        .and_then(|lines| lines.get(line_index))
        .copied()
        .unwrap_or(mesh.lines_x[line_index])
}

fn warped_column_y(mesh: &Mesh, warp: &MeshWarp, column: usize, line_index: usize) -> u32 {
    warp.lines_y_by_column
        .get(column)
        .and_then(|lines| lines.get(line_index))
        .copied()
        .unwrap_or(mesh.lines_y[line_index])
}

fn bilinear_point(corners: WarpedCellCorners, u: f32, v: f32) -> (f32, f32) {
    let top_x = corners.top_left.0 * (1.0 - u) + corners.top_right.0 * u;
    let top_y = corners.top_left.1 * (1.0 - u) + corners.top_right.1 * u;
    let bottom_x = corners.bottom_left.0 * (1.0 - u) + corners.bottom_right.0 * u;
    let bottom_y = corners.bottom_left.1 * (1.0 - u) + corners.bottom_right.1 * u;
    (
        top_x * (1.0 - v) + bottom_x * v,
        top_y * (1.0 - v) + bottom_y * v,
    )
}

#[derive(Debug, Clone)]
struct WarpedCellSubgrid {
    points: Vec<Vec<(f32, f32)>>,
    edge_refinements: Vec<EdgeRefinement>,
}

#[derive(Debug, Clone)]
struct EdgeRefinement {
    points: Vec<(f32, f32)>,
    subdivided_segments: Vec<((f32, f32), (f32, f32))>,
}

#[cfg(test)]
fn subdivided_warped_cell_grid(image: &RawImage, corners: WarpedCellCorners) -> WarpedCellSubgrid {
    subdivided_warped_cell_grid_with_options(image, corners, WarpSubdivisionOptions::default())
}

fn subdivided_warped_cell_grid_with_options(
    image: &RawImage,
    corners: WarpedCellCorners,
    options: WarpSubdivisionOptions,
) -> WarpedCellSubgrid {
    let depth = if warped_cell_edge_region_reliable(image, corners, options.edge_threshold) {
        options.max_depth.min(MAX_WARP_SUBDIVISION_DEPTH)
    } else {
        0
    };
    let segments = 1usize << depth;
    let threshold = options.edge_threshold.max(0.0);
    let top = refined_horizontal_edge_points(
        image,
        corners.top_left,
        corners.top_right,
        segments,
        threshold,
    );
    let bottom = refined_horizontal_edge_points(
        image,
        corners.bottom_left,
        corners.bottom_right,
        segments,
        threshold,
    );
    let left = refined_vertical_edge_points(
        image,
        corners.top_left,
        corners.bottom_left,
        segments,
        threshold,
    );
    let right = refined_vertical_edge_points(
        image,
        corners.top_right,
        corners.bottom_right,
        segments,
        threshold,
    );

    let mut points = vec![vec![(0.0, 0.0); segments + 1]; segments + 1];
    for y in 0..=segments {
        let v = y as f32 / segments as f32;
        for x in 0..=segments {
            let u = x as f32 / segments as f32;
            points[y][x] = if y == 0 {
                top.points[x]
            } else if y == segments {
                bottom.points[x]
            } else if x == 0 {
                left.points[y]
            } else if x == segments {
                right.points[y]
            } else {
                coons_patch_point(
                    corners,
                    top.points[x],
                    bottom.points[x],
                    left.points[y],
                    right.points[y],
                    u,
                    v,
                )
            };
        }
    }
    WarpedCellSubgrid {
        points,
        edge_refinements: vec![top, bottom, left, right],
    }
}

fn subdivided_warp_point(subdivision: &WarpedCellSubgrid, u: f32, v: f32) -> (f32, f32) {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let segments = subdivision.points.len().saturating_sub(1).max(1);
    let scaled_u = u * segments as f32;
    let scaled_v = v * segments as f32;
    let sub_x = (scaled_u.floor() as usize).min(segments - 1);
    let sub_y = (scaled_v.floor() as usize).min(segments - 1);
    let local_u = scaled_u - sub_x as f32;
    let local_v = scaled_v - sub_y as f32;
    bilinear_tuple_point(
        subdivision.points[sub_y][sub_x],
        subdivision.points[sub_y][sub_x + 1],
        subdivision.points[sub_y + 1][sub_x],
        subdivision.points[sub_y + 1][sub_x + 1],
        local_u,
        local_v,
    )
}

fn bilinear_tuple_point(
    top_left: (f32, f32),
    top_right: (f32, f32),
    bottom_left: (f32, f32),
    bottom_right: (f32, f32),
    u: f32,
    v: f32,
) -> (f32, f32) {
    let top_x = top_left.0 * (1.0 - u) + top_right.0 * u;
    let top_y = top_left.1 * (1.0 - u) + top_right.1 * u;
    let bottom_x = bottom_left.0 * (1.0 - u) + bottom_right.0 * u;
    let bottom_y = bottom_left.1 * (1.0 - u) + bottom_right.1 * u;
    (
        top_x * (1.0 - v) + bottom_x * v,
        top_y * (1.0 - v) + bottom_y * v,
    )
}

fn coons_patch_point(
    corners: WarpedCellCorners,
    top: (f32, f32),
    bottom: (f32, f32),
    left: (f32, f32),
    right: (f32, f32),
    u: f32,
    v: f32,
) -> (f32, f32) {
    let horizontal = (
        left.0 * (1.0 - u) + right.0 * u,
        left.1 * (1.0 - u) + right.1 * u,
    );
    let vertical = (
        top.0 * (1.0 - v) + bottom.0 * v,
        top.1 * (1.0 - v) + bottom.1 * v,
    );
    let corner_blend = bilinear_point(corners, u, v);
    (
        horizontal.0 + vertical.0 - corner_blend.0,
        horizontal.1 + vertical.1 - corner_blend.1,
    )
}

fn refined_horizontal_edge_points(
    image: &RawImage,
    start: (f32, f32),
    end: (f32, f32),
    segments: usize,
    edge_threshold: f32,
) -> EdgeRefinement {
    let mut points = (0..=segments)
        .map(|index| {
            let u = index as f32 / segments as f32;
            (
                start.0 * (1.0 - u) + end.0 * u,
                start.1 * (1.0 - u) + end.1 * u,
            )
        })
        .collect::<Vec<_>>();
    let baseline_points = points.clone();
    let mut subdivided_segments = Vec::new();
    refine_horizontal_edge_segment(
        image,
        &mut points,
        &baseline_points,
        &mut subdivided_segments,
        0,
        segments,
        edge_threshold,
    );
    limit_edge_refinement_displacement(&mut points, &baseline_points);
    subdivided_segments = changed_edge_segments(&points, &baseline_points);
    EdgeRefinement {
        points,
        subdivided_segments,
    }
}

fn refine_horizontal_edge_segment(
    image: &RawImage,
    points: &mut [(f32, f32)],
    baseline_points: &[(f32, f32)],
    subdivided_segments: &mut Vec<((f32, f32), (f32, f32))>,
    start_index: usize,
    end_index: usize,
    edge_threshold: f32,
) -> bool {
    if end_index.saturating_sub(start_index) <= 1 {
        return false;
    }
    let mid_index = (start_index + end_index) / 2;
    let start = points[start_index];
    let end = points[end_index];
    let midpoint = midpoint(start, end);
    let refined = refine_horizontal_subdivision_point_with_baseline_limit(
        image,
        midpoint,
        start,
        end,
        edge_threshold,
        baseline_points[mid_index],
        max_subdivision_baseline_shift(baseline_points[start_index], baseline_points[end_index]),
    );
    if !point_moved(midpoint, refined) {
        return false;
    }

    points[mid_index] = refined;
    let left_subdivided = refine_horizontal_edge_segment(
        image,
        points,
        baseline_points,
        subdivided_segments,
        start_index,
        mid_index,
        edge_threshold,
    );
    if !left_subdivided {
        fill_linear_edge_points(points, start_index, mid_index);
        subdivided_segments.push((points[start_index], points[mid_index]));
    }
    let right_subdivided = refine_horizontal_edge_segment(
        image,
        points,
        baseline_points,
        subdivided_segments,
        mid_index,
        end_index,
        edge_threshold,
    );
    if !right_subdivided {
        fill_linear_edge_points(points, mid_index, end_index);
        subdivided_segments.push((points[mid_index], points[end_index]));
    }
    true
}

fn refined_vertical_edge_points(
    image: &RawImage,
    start: (f32, f32),
    end: (f32, f32),
    segments: usize,
    edge_threshold: f32,
) -> EdgeRefinement {
    let mut points = (0..=segments)
        .map(|index| {
            let v = index as f32 / segments as f32;
            (
                start.0 * (1.0 - v) + end.0 * v,
                start.1 * (1.0 - v) + end.1 * v,
            )
        })
        .collect::<Vec<_>>();
    let baseline_points = points.clone();
    let mut subdivided_segments = Vec::new();
    refine_vertical_edge_segment(
        image,
        &mut points,
        &baseline_points,
        &mut subdivided_segments,
        0,
        segments,
        edge_threshold,
    );
    limit_edge_refinement_displacement(&mut points, &baseline_points);
    subdivided_segments = changed_edge_segments(&points, &baseline_points);
    EdgeRefinement {
        points,
        subdivided_segments,
    }
}

fn refine_vertical_edge_segment(
    image: &RawImage,
    points: &mut [(f32, f32)],
    baseline_points: &[(f32, f32)],
    subdivided_segments: &mut Vec<((f32, f32), (f32, f32))>,
    start_index: usize,
    end_index: usize,
    edge_threshold: f32,
) -> bool {
    if end_index.saturating_sub(start_index) <= 1 {
        return false;
    }
    let mid_index = (start_index + end_index) / 2;
    let start = points[start_index];
    let end = points[end_index];
    let midpoint = midpoint(start, end);
    let refined = refine_vertical_subdivision_point_with_baseline_limit(
        image,
        midpoint,
        start,
        end,
        edge_threshold,
        baseline_points[mid_index],
        max_subdivision_baseline_shift(baseline_points[start_index], baseline_points[end_index]),
    );
    if !point_moved(midpoint, refined) {
        return false;
    }

    points[mid_index] = refined;
    let top_subdivided = refine_vertical_edge_segment(
        image,
        points,
        baseline_points,
        subdivided_segments,
        start_index,
        mid_index,
        edge_threshold,
    );
    if !top_subdivided {
        fill_linear_edge_points(points, start_index, mid_index);
        subdivided_segments.push((points[start_index], points[mid_index]));
    }
    let bottom_subdivided = refine_vertical_edge_segment(
        image,
        points,
        baseline_points,
        subdivided_segments,
        mid_index,
        end_index,
        edge_threshold,
    );
    if !bottom_subdivided {
        fill_linear_edge_points(points, mid_index, end_index);
        subdivided_segments.push((points[mid_index], points[end_index]));
    }
    true
}

fn point_moved(before: (f32, f32), after: (f32, f32)) -> bool {
    point_distance(before, after) >= 0.25
}

fn limit_edge_refinement_displacement(points: &mut [(f32, f32)], baseline_points: &[(f32, f32)]) {
    if points.len() != baseline_points.len() || points.len() < 3 {
        return;
    }

    for index in 1..points.len() - 1 {
        clamp_displacement_to_neighbor(points, baseline_points, index, index - 1);
    }
    for index in (1..points.len() - 1).rev() {
        clamp_displacement_to_neighbor(points, baseline_points, index, index + 1);
    }
}

fn clamp_displacement_to_neighbor(
    points: &mut [(f32, f32)],
    baseline_points: &[(f32, f32)],
    index: usize,
    neighbor_index: usize,
) {
    let neighbor_displacement = (
        points[neighbor_index].0 - baseline_points[neighbor_index].0,
        points[neighbor_index].1 - baseline_points[neighbor_index].1,
    );
    let displacement = (
        points[index].0 - baseline_points[index].0,
        points[index].1 - baseline_points[index].1,
    );
    let clamped_displacement = (
        neighbor_displacement.0
            + (displacement.0 - neighbor_displacement.0).clamp(
                -WARP_SUBDIVISION_MAX_ADJACENT_SHIFT_DELTA,
                WARP_SUBDIVISION_MAX_ADJACENT_SHIFT_DELTA,
            ),
        neighbor_displacement.1
            + (displacement.1 - neighbor_displacement.1).clamp(
                -WARP_SUBDIVISION_MAX_ADJACENT_SHIFT_DELTA,
                WARP_SUBDIVISION_MAX_ADJACENT_SHIFT_DELTA,
            ),
    );
    points[index] = (
        baseline_points[index].0 + clamped_displacement.0,
        baseline_points[index].1 + clamped_displacement.1,
    );
}

fn changed_edge_segments(
    points: &[(f32, f32)],
    baseline_points: &[(f32, f32)],
) -> Vec<((f32, f32), (f32, f32))> {
    if points.len() != baseline_points.len() {
        return Vec::new();
    }
    points
        .windows(2)
        .zip(baseline_points.windows(2))
        .filter_map(|(segment, baseline_segment)| {
            (point_moved(segment[0], baseline_segment[0])
                || point_moved(segment[1], baseline_segment[1]))
            .then_some((segment[0], segment[1]))
        })
        .collect()
}

fn point_distance(left: (f32, f32), right: (f32, f32)) -> f32 {
    (left.0 - right.0).abs().max((left.1 - right.1).abs())
}

fn max_subdivision_baseline_shift(start: (f32, f32), end: (f32, f32)) -> f32 {
    let span = point_distance(start, end);
    (span * WARP_SUBDIVISION_BASELINE_SHIFT_RATIO)
        .max(WARP_SUBDIVISION_BASELINE_SHIFT_MIN)
        .min(WARP_SUBDIVISION_MAX_RADIUS as f32)
}

fn fill_linear_edge_points(points: &mut [(f32, f32)], start_index: usize, end_index: usize) {
    let span = end_index.saturating_sub(start_index);
    if span <= 1 {
        return;
    }
    let start = points[start_index];
    let end = points[end_index];
    for index in start_index + 1..end_index {
        let t = (index - start_index) as f32 / span as f32;
        points[index] = (
            start.0 * (1.0 - t) + end.0 * t,
            start.1 * (1.0 - t) + end.1 * t,
        );
    }
}

#[cfg(test)]
fn refine_horizontal_subdivision_point(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
) -> (f32, f32) {
    refine_subdivision_point(
        image,
        point,
        start,
        end,
        edge_threshold,
        WarpSubdivisionAxis::Horizontal,
        None,
    )
}

fn refine_horizontal_subdivision_point_with_baseline_limit(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
    baseline_point: (f32, f32),
    max_baseline_shift: f32,
) -> (f32, f32) {
    refine_subdivision_point(
        image,
        point,
        start,
        end,
        edge_threshold,
        WarpSubdivisionAxis::Horizontal,
        Some((baseline_point, max_baseline_shift)),
    )
}

#[cfg(test)]
fn refine_vertical_subdivision_point(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
) -> (f32, f32) {
    refine_subdivision_point(
        image,
        point,
        start,
        end,
        edge_threshold,
        WarpSubdivisionAxis::Vertical,
        None,
    )
}

fn refine_vertical_subdivision_point_with_baseline_limit(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
    baseline_point: (f32, f32),
    max_baseline_shift: f32,
) -> (f32, f32) {
    refine_subdivision_point(
        image,
        point,
        start,
        end,
        edge_threshold,
        WarpSubdivisionAxis::Vertical,
        Some((baseline_point, max_baseline_shift)),
    )
}

#[derive(Debug, Clone, Copy)]
enum WarpSubdivisionAxis {
    Horizontal,
    Vertical,
}

fn refine_subdivision_point(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
    axis: WarpSubdivisionAxis,
    baseline_limit: Option<((f32, f32), f32)>,
) -> (f32, f32) {
    if image.width < 2 || image.height < 2 {
        return point;
    }

    let radius = subdivision_search_radius(start, end);
    let original_x = point.0.round() as i32;
    let original_y = point.1.round() as i32;
    let base_score = subdivision_point_score(image, original_x, original_y, axis);
    let Some(local_bounds) = subdivision_local_candidate_bounds(image, point, radius) else {
        return point;
    };

    let mut best = search_subdivision_candidate(
        image,
        point,
        start,
        end,
        axis,
        base_score,
        edge_threshold,
        local_bounds,
        false,
        baseline_limit,
    );

    if segment_needs_contour_fallback(start, end) {
        if let Some(envelope_bounds) =
            subdivision_candidate_bounds(image, point, start, end, radius)
        {
            if let Some(envelope_candidate) = search_subdivision_candidate(
                image,
                point,
                start,
                end,
                axis,
                base_score,
                edge_threshold,
                envelope_bounds,
                true,
                baseline_limit,
            ) {
                if best.is_none_or(|candidate| envelope_candidate.rank > candidate.rank) {
                    best = Some(envelope_candidate);
                }
            }
        }
    }

    best.map(|candidate| candidate.point).unwrap_or(point)
}

#[derive(Debug, Clone, Copy)]
struct SubdivisionCandidate {
    point: (f32, f32),
    score: f32,
    rank: f32,
}

fn search_subdivision_candidate(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    axis: WarpSubdivisionAxis,
    base_score: f32,
    edge_threshold: f32,
    bounds: (i32, i32, i32, i32),
    require_contour: bool,
    baseline_limit: Option<((f32, f32), f32)>,
) -> Option<SubdivisionCandidate> {
    let original_x = point.0.round() as i32;
    let original_y = point.1.round() as i32;
    let (min_x, max_x, min_y, max_y) = bounds;
    let mut best = None;
    let mut accepted = Vec::new();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let candidate = (x as f32, y as f32);
            if point_overlaps(candidate, start) || point_overlaps(candidate, end) {
                continue;
            }
            if let Some((baseline_point, max_baseline_shift)) = baseline_limit {
                if point_distance(candidate, baseline_point) > max_baseline_shift {
                    continue;
                }
            }
            let score = subdivision_point_score(image, x, y, axis);
            if subdivision_snap_weight(base_score, score, edge_threshold).is_none() {
                continue;
            }
            let contour_score =
                subdivision_candidate_contour_score(image, start, candidate, end, edge_threshold);
            let unsupported_shift = (candidate.0 - point.0)
                .abs()
                .max((candidate.1 - point.1).abs());
            if (require_contour || unsupported_shift > WARP_SUBDIVISION_MAX_UNSUPPORTED_SHIFT)
                && contour_score < edge_threshold
            {
                continue;
            }
            let shift = (x.abs_diff(original_x) + y.abs_diff(original_y)) as f32;
            let rank = score + contour_score * WARP_SUBDIVISION_CONTOUR_RANK_WEIGHT
                - shift * WARP_SUBDIVISION_SHIFT_PENALTY;
            if best.is_none_or(|candidate: SubdivisionCandidate| rank > candidate.rank) {
                best = Some(SubdivisionCandidate {
                    point: candidate,
                    score,
                    rank,
                });
            }
            accepted.push(SubdivisionCandidate {
                point: candidate,
                score,
                rank,
            });
        }
    }
    let best = best?;
    let alternative_score = accepted
        .iter()
        .filter(|candidate| {
            !subdivision_candidates_share_primary_edge(axis, candidate.point, best.point)
        })
        .map(|candidate| candidate.score)
        .fold(0.0f32, f32::max);
    let prominence = if best.score <= 1.0 {
        1.0
    } else {
        (best.score - alternative_score).max(0.0) / best.score
    };
    (prominence >= WARP_SUBDIVISION_MIN_PROMINENCE).then_some(best)
}

fn subdivision_candidates_share_primary_edge(
    axis: WarpSubdivisionAxis,
    first: (f32, f32),
    second: (f32, f32),
) -> bool {
    match axis {
        WarpSubdivisionAxis::Horizontal => first.1.round() == second.1.round(),
        WarpSubdivisionAxis::Vertical => first.0.round() == second.0.round(),
    }
}

fn subdivision_point_score(image: &RawImage, x: i32, y: i32, axis: WarpSubdivisionAxis) -> f32 {
    if x < 0 || y < 0 || x >= image.width as i32 || y >= image.height as i32 {
        return 0.0;
    }
    let x = x as u32;
    let y = y as u32;
    let primary = match axis {
        WarpSubdivisionAxis::Horizontal => color_horizontal_point_score(image, x, y),
        WarpSubdivisionAxis::Vertical => color_vertical_point_score(image, x, y),
    };
    let perpendicular = match axis {
        WarpSubdivisionAxis::Horizontal => color_vertical_point_score(image, x, y),
        WarpSubdivisionAxis::Vertical => color_horizontal_point_score(image, x, y),
    };
    primary + primary.min(perpendicular) * LOCAL_EDGE_CORNER_BONUS
}

fn subdivision_local_candidate_bounds(
    image: &RawImage,
    point: (f32, f32),
    radius: u32,
) -> Option<(i32, i32, i32, i32)> {
    let radius = radius as i32;
    let original_x = point.0.round() as i32;
    let original_y = point.1.round() as i32;
    let min_x = (original_x - radius).max(0);
    let max_x = (original_x + radius).min(image.width.saturating_sub(1) as i32);
    let min_y = (original_y - radius).max(0);
    let max_y = (original_y + radius).min(image.height.saturating_sub(1) as i32);
    (min_x <= max_x && min_y <= max_y).then_some((min_x, max_x, min_y, max_y))
}

fn subdivision_candidate_bounds(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
    radius: u32,
) -> Option<(i32, i32, i32, i32)> {
    let radius = radius as f32;
    let min_x = point.0.min(start.0).min(end.0).floor() - radius;
    let max_x = point.0.max(start.0).max(end.0).ceil() + radius;
    let min_y = point.1.min(start.1).min(end.1).floor() - radius;
    let max_y = point.1.max(start.1).max(end.1).ceil() + radius;
    let min_x = min_x.max(0.0) as i32;
    let max_x = max_x.min(image.width.saturating_sub(1) as f32) as i32;
    let min_y = min_y.max(0.0) as i32;
    let max_y = max_y.min(image.height.saturating_sub(1) as f32) as i32;
    (min_x <= max_x && min_y <= max_y).then_some((min_x, max_x, min_y, max_y))
}

fn segment_needs_contour_fallback(start: (f32, f32), end: (f32, f32)) -> bool {
    (start.0 - end.0).abs() > 1.0 && (start.1 - end.1).abs() > 1.0
}

fn subdivision_candidate_contour_score(
    image: &RawImage,
    start: (f32, f32),
    candidate: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
) -> f32 {
    let before = subdivision_contour_segment_score(image, start, candidate, edge_threshold);
    let after = subdivision_contour_segment_score(image, candidate, end, edge_threshold);
    before.min(after)
}

fn subdivision_contour_segment_score(
    image: &RawImage,
    start: (f32, f32),
    end: (f32, f32),
    edge_threshold: f32,
) -> f32 {
    let span = (start.0 - end.0).abs().max((start.1 - end.1).abs());
    let steps = span.ceil().max(1.0) as u32;
    if steps <= 1 {
        return edge_threshold;
    }

    let mut total = 0.0f32;
    let mut covered = 0u32;
    let mut samples = 0u32;
    for index in 1..steps {
        let t = index as f32 / steps as f32;
        let x = (start.0 * (1.0 - t) + end.0 * t)
            .round()
            .clamp(0.0, image.width.saturating_sub(1) as f32) as u32;
        let y = (start.1 * (1.0 - t) + end.1 * t)
            .round()
            .clamp(0.0, image.height.saturating_sub(1) as f32) as u32;
        let score = contour_point_score(image, x, y);
        total += score;
        covered += u32::from(score >= edge_threshold);
        samples += 1;
    }
    if samples == 0 {
        return edge_threshold;
    }

    let average = total / samples as f32;
    let coverage = covered as f32 / samples as f32;
    average * coverage
}

fn contour_point_score(image: &RawImage, x: u32, y: u32) -> f32 {
    let horizontal = color_horizontal_point_score(image, x, y);
    let vertical = color_vertical_point_score(image, x, y);
    horizontal.max(vertical) + horizontal.min(vertical) * LOCAL_EDGE_CORNER_BONUS
}

fn point_overlaps(left: (f32, f32), right: (f32, f32)) -> bool {
    (left.0 - right.0).abs().max((left.1 - right.1).abs()) < 0.5
}

fn subdivision_search_radius(start: (f32, f32), end: (f32, f32)) -> u32 {
    let span = (start.0 - end.0).abs().max((start.1 - end.1).abs());
    ((span * 0.25).round() as u32).clamp(2, WARP_SUBDIVISION_MAX_RADIUS)
}

fn subdivision_snap_weight(base_score: f32, best_score: f32, edge_threshold: f32) -> Option<f32> {
    if best_score < edge_threshold {
        return None;
    }
    if best_score < (base_score * WARP_SUBDIVISION_MIN_EDGE_GAIN).max(base_score + 2.0) {
        return None;
    }
    let gain = (best_score - base_score).max(0.0) / best_score.max(1.0);
    Some((0.35 + gain * 0.55).clamp(0.0, 0.9))
}

fn midpoint(start: (f32, f32), end: (f32, f32)) -> (f32, f32) {
    ((start.0 + end.0) / 2.0, (start.1 + end.1) / 2.0)
}

fn pack_rgb(rgb: [u8; 3]) -> u32 {
    ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32
}

fn unpack_rgb(value: u32) -> [u8; 3] {
    [
        ((value >> 16) & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        (value & 0xff) as u8,
    ]
}

pub fn refine_mesh_to_local_edges(image: &RawImage, result: &MeshResult) -> MeshResult {
    refine_mesh_to_local_edges_with_boundary_signals(image, result, None)
}

pub fn refine_mesh_to_local_edges_with_boundary_signals(
    image: &RawImage,
    result: &MeshResult,
    boundary_signals: Option<(&[u32], &[u32])>,
) -> MeshResult {
    if !matches!(
        result.pixel_width_source,
        PixelWidthSource::Hough | PixelWidthSource::Hybrid
    ) || is_trivial_mesh(&result.mesh)
    {
        return result.clone();
    }

    let color_signals = color_boundary_signals(image);
    let merged_signals;
    let (horizontal, vertical) = match boundary_signals {
        Some((horizontal, vertical)) => {
            merged_signals = (
                merge_boundary_signals(horizontal, &color_signals.0),
                merge_boundary_signals(vertical, &color_signals.1),
            );
            (merged_signals.0.as_slice(), merged_signals.1.as_slice())
        }
        None => (color_signals.0.as_slice(), color_signals.1.as_slice()),
    };
    let integral = build_color_integral_image(image);
    let integral_stride = image.width as usize + 1;
    let mut best_mesh = result.mesh.clone();
    let mut best_error = reconstruction_error(image, &integral, integral_stride, &best_mesh);
    let refined = Mesh {
        lines_x: refine_axis_lines_to_local_edges(
            &result.mesh.lines_x,
            horizontal,
            result.detected_pixel_width,
        ),
        lines_y: refine_axis_lines_to_local_edges(
            &result.mesh.lines_y,
            vertical,
            result.detected_pixel_width,
        ),
        warp: None,
    };
    let refined_error = reconstruction_error(image, &integral, integral_stride, &refined);
    if refined_error < best_error {
        best_mesh = refined;
        best_error = refined_error;
    }

    if let Some(warp) = create_warped_mesh_from_local_edges(image, &best_mesh, result) {
        let warped_mesh = Mesh {
            lines_x: best_mesh.lines_x.clone(),
            lines_y: best_mesh.lines_y.clone(),
            warp: Some(warp),
        };
        let warped_error = reconstruction_error(image, &integral, integral_stride, &warped_mesh);
        if best_error - warped_error >= LOCAL_EDGE_WARP_MIN_ERROR_GAIN {
            best_mesh = warped_mesh;
        }
    }

    MeshResult {
        mesh: best_mesh,
        ..result.clone()
    }
}

fn mesh_cell_bounds(
    mesh: &Mesh,
    cell_x: usize,
    cell_y: usize,
    image: &RawImage,
) -> (u32, u32, u32, u32) {
    let warped_row = mesh
        .warp
        .as_ref()
        .and_then(|warp| warp.lines_x_by_row.get(cell_y));
    let warped_column = mesh
        .warp
        .as_ref()
        .and_then(|warp| warp.lines_y_by_column.get(cell_x));
    let raw_x0 = warped_row
        .and_then(|row| row.get(cell_x))
        .copied()
        .unwrap_or(mesh.lines_x[cell_x]);
    let raw_x1 = warped_row
        .and_then(|row| row.get(cell_x + 1))
        .copied()
        .unwrap_or(mesh.lines_x[cell_x + 1]);
    let raw_y0 = warped_column
        .and_then(|column| column.get(cell_y))
        .copied()
        .unwrap_or(mesh.lines_y[cell_y]);
    let raw_y1 = warped_column
        .and_then(|column| column.get(cell_y + 1))
        .copied()
        .unwrap_or(mesh.lines_y[cell_y + 1]);
    let x0 = raw_x0.min(raw_x1).min(image.width.saturating_sub(1));
    let x1 = raw_x0.max(raw_x1).max(x0 + 1).min(image.width);
    let y0 = raw_y0.min(raw_y1).min(image.height.saturating_sub(1));
    let y1 = raw_y0.max(raw_y1).max(y0 + 1).min(image.height);
    (x0, x1, y0, y1)
}

fn create_warped_mesh_from_local_edges(
    image: &RawImage,
    mesh: &Mesh,
    result: &MeshResult,
) -> Option<MeshWarp> {
    let row_count = mesh.lines_y.len().saturating_sub(1);
    let column_count = mesh.lines_x.len().saturating_sub(1);
    if row_count < 2 || column_count < 2 {
        return None;
    }

    let energy = LocalEdgeEnergy::new(image);
    let lines_x_by_row = smooth_warped_line_sets(
        &mesh.lines_x,
        (0..row_count)
            .into_par_iter()
            .map(|row_index| {
                let start_row = row_index.saturating_sub(LOCAL_EDGE_REFINEMENT_BAND_RADIUS);
                let end_row = (row_index + LOCAL_EDGE_REFINEMENT_BAND_RADIUS + 1).min(row_count);
                let y0 = mesh.lines_y[start_row];
                let y1 = mesh.lines_y[end_row];
                refine_axis_lines_by_score_with_confidence(
                    &mesh.lines_x,
                    energy.width as usize,
                    result.detected_pixel_width,
                    |position| energy.vertical_band_corner_score(position, y0, y1),
                )
            })
            .collect(),
    );
    let lines_y_by_column = smooth_warped_line_sets(
        &mesh.lines_y,
        (0..column_count)
            .into_par_iter()
            .map(|column_index| {
                let start_column = column_index.saturating_sub(LOCAL_EDGE_REFINEMENT_BAND_RADIUS);
                let end_column =
                    (column_index + LOCAL_EDGE_REFINEMENT_BAND_RADIUS + 1).min(column_count);
                let x0 = mesh.lines_x[start_column];
                let x1 = mesh.lines_x[end_column];
                refine_axis_lines_by_score_with_confidence(
                    &mesh.lines_y,
                    energy.height as usize,
                    result.detected_pixel_width,
                    |position| energy.horizontal_band_corner_score(position, x0, x1),
                )
            })
            .collect(),
    );

    Some(MeshWarp {
        lines_x_by_row,
        lines_y_by_column,
    })
}

struct LocalEdgeEnergy {
    width: u32,
    height: u32,
    vertical: Vec<f32>,
    horizontal: Vec<f32>,
    vertical_column_prefix: Vec<f64>,
    horizontal_row_prefix: Vec<f64>,
}

impl LocalEdgeEnergy {
    fn new(image: &RawImage) -> Self {
        let width = image.width;
        let height = image.height;
        let len = width as usize * height as usize;
        let mut vertical = vec![0.0f32; len];
        let mut horizontal = vec![0.0f32; len];
        for y in 0..height {
            for x in 1..width {
                let index = (y * width + x) as usize;
                vertical[index] = color_edge_delta(image.pixel(x - 1, y), image.pixel(x, y)) as f32;
            }
        }
        for y in 1..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                horizontal[index] =
                    color_edge_delta(image.pixel(x, y - 1), image.pixel(x, y)) as f32;
            }
        }

        let vertical_column_prefix = build_vertical_column_prefix(&vertical, width, height);
        let horizontal_row_prefix = build_horizontal_row_prefix(&horizontal, width, height);

        Self {
            width,
            height,
            vertical,
            horizontal,
            vertical_column_prefix,
            horizontal_row_prefix,
        }
    }

    fn vertical_column_sum(&self, x: u32, start_y: u32, end_y: u32) -> f64 {
        let column = x.min(self.width.saturating_sub(1));
        self.vertical_column_prefix[(end_y * self.width + column) as usize]
            - self.vertical_column_prefix[(start_y * self.width + column) as usize]
    }

    fn horizontal_row_sum(&self, y: u32, start_x: u32, end_x: u32) -> f64 {
        let row = y.min(self.height.saturating_sub(1));
        let stride = self.width + 1;
        self.horizontal_row_prefix[(row * stride + end_x) as usize]
            - self.horizontal_row_prefix[(row * stride + start_x) as usize]
    }

    fn vertical_band_edge_score(&self, position: usize, y0: u32, y1: u32) -> f32 {
        let x = (position as u32).clamp(1, self.width.saturating_sub(2));
        let start_y = y0.min(self.height.saturating_sub(1));
        let end_y = y1.clamp(start_y + 1, self.height);
        let total = self.vertical_column_sum(x - 1, start_y, end_y) * 0.25
            + self.vertical_column_sum(x, start_y, end_y)
            + self.vertical_column_sum(x + 1, start_y, end_y) * 0.25;
        (total / (end_y - start_y).max(1) as f64) as f32
    }

    fn vertical_band_corner_score(&self, position: usize, y0: u32, y1: u32) -> f32 {
        let edge = self.vertical_band_edge_score(position, y0, y1);
        let x = (position as u32).clamp(1, self.width.saturating_sub(1));
        let top = self.corner_score(x, y0);
        let bottom = self.corner_score(x, y1.saturating_sub(1));
        edge + ((top + bottom) / 2.0) * LOCAL_EDGE_CORNER_BONUS
    }

    fn horizontal_band_edge_score(&self, position: usize, x0: u32, x1: u32) -> f32 {
        let y = (position as u32).clamp(1, self.height.saturating_sub(2));
        let start_x = x0.min(self.width.saturating_sub(1));
        let end_x = x1.clamp(start_x + 1, self.width);
        let total = self.horizontal_row_sum(y - 1, start_x, end_x) * 0.25
            + self.horizontal_row_sum(y, start_x, end_x)
            + self.horizontal_row_sum(y + 1, start_x, end_x) * 0.25;
        (total / (end_x - start_x).max(1) as f64) as f32
    }

    fn horizontal_band_corner_score(&self, position: usize, x0: u32, x1: u32) -> f32 {
        let edge = self.horizontal_band_edge_score(position, x0, x1);
        let y = (position as u32).clamp(1, self.height.saturating_sub(1));
        let left = self.corner_score(x0, y);
        let right = self.corner_score(x1.saturating_sub(1), y);
        edge + ((left + right) / 2.0) * LOCAL_EDGE_CORNER_BONUS
    }

    fn corner_score(&self, x: u32, y: u32) -> f32 {
        if self.width < 2 || self.height < 2 {
            return 0.0;
        }
        let x = x.clamp(1, self.width.saturating_sub(1));
        let y = y.clamp(1, self.height.saturating_sub(1));
        let mut best = 0.0f32;
        for yy in y.saturating_sub(1)..=(y + 1).min(self.height.saturating_sub(1)) {
            for xx in x.saturating_sub(1)..=(x + 1).min(self.width.saturating_sub(1)) {
                let index = (yy * self.width + xx) as usize;
                let vertical = self.vertical.get(index).copied().unwrap_or(0.0);
                let horizontal = self.horizontal.get(index).copied().unwrap_or(0.0);
                best = best.max(vertical.min(horizontal));
            }
        }
        best
    }
}

fn build_vertical_column_prefix(values: &[f32], width: u32, height: u32) -> Vec<f64> {
    let mut prefix = vec![0.0; width as usize * (height as usize + 1)];
    for y in 0..height {
        let source_row = (y * width) as usize;
        let previous_prefix_row = (y * width) as usize;
        let prefix_row = ((y + 1) * width) as usize;
        for x in 0..width as usize {
            prefix[prefix_row + x] =
                prefix[previous_prefix_row + x] + values[source_row + x] as f64;
        }
    }
    prefix
}

fn build_horizontal_row_prefix(values: &[f32], width: u32, height: u32) -> Vec<f64> {
    let stride = width as usize + 1;
    let mut prefix = vec![0.0; stride * height as usize];
    for y in 0..height as usize {
        let mut row_sum = 0.0;
        let source_row = y * width as usize;
        let prefix_row = y * stride;
        for x in 0..width as usize {
            row_sum += values[source_row + x] as f64;
            prefix[prefix_row + x + 1] = row_sum;
        }
    }
    prefix
}

#[derive(Debug, Clone)]
struct AxisLineRefinement {
    lines: Vec<u32>,
    confidences: Vec<f32>,
}

fn smooth_warped_line_sets(
    base_lines: &[u32],
    refinements: Vec<AxisLineRefinement>,
) -> Vec<Vec<u32>> {
    let mut out = refinements
        .iter()
        .map(|refinement| stabilize_refined_lines(base_lines, refinement))
        .collect::<Vec<_>>();
    if out.len() < 3 {
        return out;
    }
    let confidences = refinements
        .iter()
        .map(|refinement| refinement.confidences.clone())
        .collect::<Vec<_>>();
    for _ in 0..LOCAL_EDGE_REFINEMENT_SMOOTHING_PASSES {
        let previous = out.clone();
        out = previous
            .iter()
            .enumerate()
            .map(|(set_index, lines)| {
                if set_index == 0 || set_index + 1 == previous.len() {
                    return lines.clone();
                }

                enforce_line_order(
                    lines
                        .iter()
                        .enumerate()
                        .map(|(line_index, line)| {
                            if line_index == 0 || line_index + 1 == lines.len() {
                                return *line;
                            }
                            let confidence = confidences
                                .get(set_index)
                                .and_then(|set| set.get(line_index))
                                .copied()
                                .unwrap_or(0.0);
                            let local_weight = local_warp_weight(confidence);
                            let neighbor_mean = (previous[set_index - 1][line_index]
                                + previous[set_index + 1][line_index])
                                as f32
                                / 2.0;
                            let base_line = base_lines.get(line_index).copied().unwrap_or(*line);
                            let regularized = (neighbor_mean + base_line as f32) / 2.0;
                            (*line as f32 * local_weight + regularized * (1.0 - local_weight))
                                .round() as u32
                        })
                        .collect(),
                )
            })
            .collect();
    }
    out
}

fn stabilize_refined_lines(base_lines: &[u32], refinement: &AxisLineRefinement) -> Vec<u32> {
    enforce_line_order(
        refinement
            .lines
            .iter()
            .enumerate()
            .map(|(line_index, line)| {
                let Some(base_line) = base_lines.get(line_index).copied() else {
                    return *line;
                };
                if line_index == 0 || line_index + 1 == refinement.lines.len() {
                    return *line;
                }
                let confidence = refinement
                    .confidences
                    .get(line_index)
                    .copied()
                    .unwrap_or(0.0);
                let weight = local_warp_weight(confidence);
                (*line as f32 * weight + base_line as f32 * (1.0 - weight)).round() as u32
            })
            .collect(),
    )
}

fn local_warp_weight(confidence: f32) -> f32 {
    if confidence <= LOCAL_EDGE_WEAK_CONFIDENCE {
        return 0.0;
    }
    if confidence >= LOCAL_EDGE_STRONG_CONFIDENCE {
        return 0.65;
    }
    let t = (confidence - LOCAL_EDGE_WEAK_CONFIDENCE)
        / (LOCAL_EDGE_STRONG_CONFIDENCE - LOCAL_EDGE_WEAK_CONFIDENCE);
    0.15 + t * 0.5
}

fn enforce_line_order(mut lines: Vec<u32>) -> Vec<u32> {
    if lines.len() < 2 {
        return lines;
    }

    for index in 1..lines.len() {
        lines[index] = lines[index].max(lines[index - 1] + 1);
    }
    for index in (0..lines.len() - 1).rev() {
        lines[index] = lines[index].min(lines[index + 1] - 1);
    }
    lines
}

struct ColorIntegralImage {
    red: Vec<f64>,
    green: Vec<f64>,
    blue: Vec<f64>,
}

fn reconstruction_error(
    image: &RawImage,
    integral: &ColorIntegralImage,
    stride: usize,
    mesh: &Mesh,
) -> f32 {
    let mut total_error = 0.0f64;
    let mut total_pixels = 0usize;

    for cell_y in 0..mesh.lines_y.len().saturating_sub(1) {
        for cell_x in 0..mesh.lines_x.len().saturating_sub(1) {
            let (x0, x1, y0, y1) =
                mesh_cell_bounds_for_size(mesh, cell_x, cell_y, image.width, image.height);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }

            let pixel_count = (x1 - x0) as usize * (y1 - y0) as usize;
            let mean = rect_color_sum(integral, stride, x0, x1, y0, y1);
            let mean = [
                mean[0] / pixel_count as f64,
                mean[1] / pixel_count as f64,
                mean[2] / pixel_count as f64,
            ];
            for y in y0..y1 {
                for x in x0..x1 {
                    let pixel = reconstruction_pixel(image.pixel(x, y));
                    total_error += (pixel[0] as f64 - mean[0]).abs();
                    total_error += (pixel[1] as f64 - mean[1]).abs();
                    total_error += (pixel[2] as f64 - mean[2]).abs();
                    total_pixels += 1;
                }
            }
        }
    }

    if total_pixels == 0 {
        f32::INFINITY
    } else {
        (total_error / (total_pixels * 3) as f64) as f32
    }
}

fn mesh_cell_bounds_for_size(
    mesh: &Mesh,
    cell_x: usize,
    cell_y: usize,
    width: u32,
    height: u32,
) -> (u32, u32, u32, u32) {
    let warped_row = mesh
        .warp
        .as_ref()
        .and_then(|warp| warp.lines_x_by_row.get(cell_y));
    let warped_column = mesh
        .warp
        .as_ref()
        .and_then(|warp| warp.lines_y_by_column.get(cell_x));
    let raw_x0 = warped_row
        .and_then(|row| row.get(cell_x))
        .copied()
        .unwrap_or(mesh.lines_x[cell_x]);
    let raw_x1 = warped_row
        .and_then(|row| row.get(cell_x + 1))
        .copied()
        .unwrap_or(mesh.lines_x[cell_x + 1]);
    let raw_y0 = warped_column
        .and_then(|column| column.get(cell_y))
        .copied()
        .unwrap_or(mesh.lines_y[cell_y]);
    let raw_y1 = warped_column
        .and_then(|column| column.get(cell_y + 1))
        .copied()
        .unwrap_or(mesh.lines_y[cell_y + 1]);
    let x0 = raw_x0.min(raw_x1).min(width);
    let x1 = raw_x0.max(raw_x1).min(width);
    let y0 = raw_y0.min(raw_y1).min(height);
    let y1 = raw_y0.max(raw_y1).min(height);
    (x0, x1, y0, y1)
}

fn build_color_integral_image(image: &RawImage) -> ColorIntegralImage {
    let stride = image.width as usize + 1;
    let len = stride * (image.height as usize + 1);
    let mut red = vec![0.0; len];
    let mut green = vec![0.0; len];
    let mut blue = vec![0.0; len];
    for y in 0..image.height as usize {
        let mut row_sum = [0.0, 0.0, 0.0];
        let integral_row = (y + 1) * stride;
        let previous_integral_row = y * stride;
        for x in 0..image.width as usize {
            let pixel = reconstruction_pixel(image.pixel(x as u32, y as u32));
            row_sum[0] += pixel[0] as f64;
            row_sum[1] += pixel[1] as f64;
            row_sum[2] += pixel[2] as f64;
            red[integral_row + x + 1] = red[previous_integral_row + x + 1] + row_sum[0];
            green[integral_row + x + 1] = green[previous_integral_row + x + 1] + row_sum[1];
            blue[integral_row + x + 1] = blue[previous_integral_row + x + 1] + row_sum[2];
        }
    }
    ColorIntegralImage { red, green, blue }
}

fn reconstruction_pixel(pixel: [u8; 4]) -> [u8; 3] {
    if pixel[3] < ALPHA_THRESHOLD {
        [0, 0, 0]
    } else {
        [pixel[0], pixel[1], pixel[2]]
    }
}

fn rect_color_sum(
    integral: &ColorIntegralImage,
    stride: usize,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
) -> [f64; 3] {
    [
        rect_sum(&integral.red, stride, x0, x1, y0, y1),
        rect_sum(&integral.green, stride, x0, x1, y0, y1),
        rect_sum(&integral.blue, stride, x0, x1, y0, y1),
    ]
}

fn rect_sum(integral: &[f64], stride: usize, x0: u32, x1: u32, y0: u32, y1: u32) -> f64 {
    let x0 = x0 as usize;
    let x1 = x1 as usize;
    let y0 = y0 as usize;
    let y1 = y1 as usize;
    integral[y1 * stride + x1] - integral[y0 * stride + x1] - integral[y1 * stride + x0]
        + integral[y0 * stride + x0]
}

pub fn create_debug_sheet(
    original: &RawImage,
    unscaled: &RawImage,
    mesh: &MeshResult,
    palette_colors: &[[u8; 3]],
    debug_scale: u32,
    _palette_merge_threshold: f32,
) -> RawImage {
    create_debug_sheet_with_options(
        original,
        unscaled,
        mesh,
        palette_colors,
        DebugSheetOptions {
            debug_scale,
            palette_merge_threshold: _palette_merge_threshold,
            transparent_background: false,
            edge_close_kernel_size: 0,
            sample_grid: 5,
            warp_subdivision: WarpSubdivisionOptions::default(),
        },
    )
}

#[derive(Debug, Clone, Copy)]
pub struct DebugSheetOptions {
    pub debug_scale: u32,
    pub palette_merge_threshold: f32,
    pub transparent_background: bool,
    pub edge_close_kernel_size: u32,
    pub sample_grid: u32,
    pub warp_subdivision: WarpSubdivisionOptions,
}

pub fn create_debug_sheet_with_options(
    original: &RawImage,
    unscaled: &RawImage,
    mesh: &MeshResult,
    palette_colors: &[[u8; 3]],
    options: DebugSheetOptions,
) -> RawImage {
    let detection_image = if mesh.scale_used > 1 {
        scale_nearest(original, mesh.scale_used)
    } else {
        original.clone()
    };
    let preview_scale = limit_scale_for_max_dimension(&detection_image, options.debug_scale.max(1));
    let debug_display_multiplier = mesh.scale_used.max(1) * preview_scale;
    let edge_source = crop_debug_detection_image(&detection_image, mesh.debug_crop_offset);

    let original_preview = scale_nearest(original, debug_display_multiplier);
    let enlarged_detection_image = scale_nearest(&detection_image, preview_scale);
    let transparency_mask_preview = create_background_mask_preview(
        &detection_image,
        options.transparent_background,
        options.edge_close_kernel_size,
    )
    .map(|preview| scale_nearest(&preview, preview_scale));
    let canny_mask = create_edge_preview(&edge_source, options.edge_close_kernel_size);
    let canny_preview = scale_nearest(&canny_mask, preview_scale);
    let hough_preview = draw_detection_line_overlay(
        &enlarged_detection_image,
        &canny_mask,
        mesh.debug_crop_offset,
        preview_scale,
    );
    let hough_preview = draw_anchor_overlay(&hough_preview, mesh, preview_scale);
    let grid_preview = draw_grid_overlay_with_options(
        &enlarged_detection_image,
        original,
        mesh,
        preview_scale,
        options.warp_subdivision,
    );
    let mut final_preview = scale_nearest(unscaled, debug_display_multiplier);
    let sampled_background_coverage_preview = create_sampled_background_coverage_preview(
        original,
        &mesh.mesh,
        options.transparent_background,
        options.sample_grid,
        options.warp_subdivision,
        options.edge_close_kernel_size,
    );

    let final_preview_scale =
        choose_closest_integer_scale(&final_preview, grid_preview.width, grid_preview.height);
    if final_preview_scale > 1 {
        final_preview = scale_nearest(&final_preview, final_preview_scale);
    }
    let sampled_background_coverage_preview = sampled_background_coverage_preview.map(|preview| {
        let preview = scale_nearest(&preview, debug_display_multiplier);
        if final_preview_scale > 1 {
            scale_nearest(&preview, final_preview_scale)
        } else {
            preview
        }
    });
    let palette_preview =
        create_debug_palette_preview(palette_colors, final_preview.width, final_preview.height);

    let top_row = vec![original_preview, canny_preview, hough_preview, grid_preview];
    let mut mask_row = Vec::new();
    if let Some(preview) = transparency_mask_preview {
        mask_row.push(preview);
    }
    if let Some(preview) = sampled_background_coverage_preview {
        mask_row.push(preview);
    }

    let output_row = vec![final_preview, palette_preview];
    let mut rows = vec![top_row.as_slice()];
    if !mask_row.is_empty() {
        rows.push(mask_row.as_slice());
    }
    rows.push(output_row.as_slice());
    compose_debug_rows(&rows)
}

fn create_background_mask_preview(
    image: &RawImage,
    enabled: bool,
    edge_close_kernel_size: u32,
) -> Option<RawImage> {
    if !enabled {
        return None;
    }
    let background = boundary_background_color(image)?;
    let mask = boundary_background_mask_with_color_and_edge_closing(
        image,
        &background,
        edge_close_kernel_size,
    );
    Some(render_background_mask_preview(image, &mask))
}

fn render_background_mask_preview(image: &RawImage, mask: &BackgroundMask) -> RawImage {
    debug_assert_eq!((image.width, image.height), (mask.width, mask.height));
    let mut out = RawImage::transparent(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.pixel(x, y);
            let color = if mask.is_background(x, y) {
                DEBUG_BACKGROUND_MASK_COLOR
            } else if pixel[3] < ALPHA_THRESHOLD {
                [32, 32, 32, 255]
            } else {
                [
                    ((pixel[0] as u16 * 2) / 5) as u8,
                    ((pixel[1] as u16 * 2) / 5) as u8,
                    ((pixel[2] as u16 * 2) / 5) as u8,
                    255,
                ]
            };
            out.set_pixel(x, y, color);
        }
    }
    out
}

fn create_sampled_background_coverage_preview(
    image: &RawImage,
    mesh: &Mesh,
    enabled: bool,
    sample_grid: u32,
    warp_options: WarpSubdivisionOptions,
    edge_close_kernel_size: u32,
) -> Option<RawImage> {
    if !enabled {
        return None;
    }
    let background = boundary_background_color(image)?;
    let mask = boundary_background_mask_with_color_and_edge_closing(
        image,
        &background,
        edge_close_kernel_size,
    );
    let width = mesh.lines_x.len().saturating_sub(1) as u32;
    let height = mesh.lines_y.len().saturating_sub(1) as u32;
    let colors = (0..width as usize * height as usize)
        .into_par_iter()
        .map(|index| {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let coverage = sampled_background_coverage_for_cell(
                image,
                &mask,
                mesh,
                x as usize,
                y as usize,
                sample_grid,
                warp_options,
            );
            debug_background_coverage_color(coverage)
        })
        .collect::<Vec<_>>();
    let mut pixels = Vec::with_capacity(colors.len() * 4);
    for color in colors {
        pixels.extend_from_slice(&color);
    }
    Some(RawImage::new(width, height, pixels))
}

fn sampled_background_coverage_for_cell(
    image: &RawImage,
    mask: &BackgroundMask,
    mesh: &Mesh,
    cell_x: usize,
    cell_y: usize,
    sample_grid: u32,
    warp_options: WarpSubdivisionOptions,
) -> u8 {
    let Some(corners) = warped_cell_corners(mesh, cell_x, cell_y) else {
        let (x0, x1, y0, y1) = mesh_cell_bounds(mesh, cell_x, cell_y, image);
        return background_coverage_for_bounds(mask, x0, x1, y0, y1);
    };

    let corners = refined_warped_cell_corners(image, corners, warp_options.edge_threshold);
    let subdivision = subdivided_warped_cell_grid_with_options(image, corners, warp_options);
    let grid = sample_grid.max(1);
    let mut background = 0u32;
    let mut total = 0u32;
    for sample_y in 0..grid {
        let v = (sample_y as f32 + 0.5) / grid as f32;
        for sample_x in 0..grid {
            let u = (sample_x as f32 + 0.5) / grid as f32;
            let point = subdivided_warp_point(&subdivision, u, v);
            let x = point
                .0
                .round()
                .clamp(0.0, image.width.saturating_sub(1) as f32) as u32;
            let y = point
                .1
                .round()
                .clamp(0.0, image.height.saturating_sub(1) as f32) as u32;
            total += 1;
            background += u32::from(mask.is_background(x, y));
        }
    }
    coverage_to_u8(background, total)
}

fn debug_background_coverage_color(coverage: u8) -> [u8; 4] {
    if coverage >= BACKGROUND_TRANSPARENT_COVERAGE {
        DEBUG_BACKGROUND_COVERAGE_STRONG_COLOR
    } else if coverage >= BACKGROUND_FRINGE_MIN_COVERAGE {
        DEBUG_BACKGROUND_COVERAGE_MEDIUM_COLOR
    } else if coverage > 0 {
        DEBUG_BACKGROUND_COVERAGE_WEAK_COLOR
    } else {
        [0, 0, 0, 255]
    }
}

fn create_debug_palette_preview(
    palette_colors: &[[u8; 3]],
    panel_width: u32,
    panel_height: u32,
) -> RawImage {
    let palette = crate::palette::create_palette_image(palette_colors);
    let max_width = ((panel_width as f32) * DEBUG_PALETTE_MAX_WIDTH_RATIO)
        .round()
        .max(palette.width as f32) as u32;
    let max_height = ((panel_height as f32) * DEBUG_PALETTE_MAX_HEIGHT_RATIO)
        .round()
        .max(palette.height as f32) as u32;
    let scale = (max_width / palette.width.max(1))
        .min(max_height / palette.height.max(1))
        .clamp(1, DEBUG_PALETTE_MAX_SWATCH_SCALE);

    if scale > 1 {
        scale_nearest(&palette, scale)
    } else {
        palette
    }
}

fn compose_debug_rows(rows: &[&[RawImage]]) -> RawImage {
    const GAP: u32 = 12;

    let non_empty_rows = rows
        .iter()
        .filter(|row| !row.is_empty())
        .copied()
        .collect::<Vec<_>>();
    if non_empty_rows.is_empty() {
        return RawImage::transparent(1, 1);
    }

    let row_sizes = non_empty_rows
        .iter()
        .map(|row| {
            let width = row.iter().map(|image| image.width).sum::<u32>()
                + GAP * (row.len().saturating_sub(1) as u32);
            let height = row.iter().map(|image| image.height).max().unwrap_or(1);
            (width, height)
        })
        .collect::<Vec<_>>();
    let content_width = row_sizes.iter().map(|(width, _)| *width).max().unwrap_or(1);
    let content_height = row_sizes.iter().map(|(_, height)| *height).sum::<u32>()
        + GAP * (row_sizes.len().saturating_sub(1) as u32);
    let mut out = RawImage::new(
        content_width + GAP * 2,
        content_height + GAP * 2,
        vec![245; ((content_width + GAP * 2) * (content_height + GAP * 2) * 4) as usize],
    );

    let mut y = GAP;
    for (row, (row_width, row_height)) in non_empty_rows.iter().zip(row_sizes) {
        let mut x = GAP + (content_width - row_width) / 2;
        for image in *row {
            let target_y = y + (row_height - image.height) / 2;
            out = crate::image::blit_image(&out, image, x as i32, target_y as i32);
            x += image.width + GAP;
        }
        y += row_height + GAP;
    }
    out
}

fn is_trivial_mesh(mesh: &Mesh) -> bool {
    let x_count = mesh.lines_x.len();
    let y_count = mesh.lines_y.len();
    (x_count == 2 || x_count == 3) && (y_count == 2 || y_count == 3)
}

fn color_boundary_signals(image: &RawImage) -> (Vec<u32>, Vec<u32>) {
    let horizontal = (0..image.width)
        .into_par_iter()
        .map(|x| {
            let mut sum = 0;
            for y in 0..image.height {
                if x > 0 {
                    sum += color_edge_delta(image.pixel(x - 1, y), image.pixel(x, y));
                }
            }
            sum
        })
        .collect();
    let vertical = (0..image.height)
        .into_par_iter()
        .map(|y| {
            let mut sum = 0;
            for x in 0..image.width {
                if y > 0 {
                    sum += color_edge_delta(image.pixel(x, y - 1), image.pixel(x, y));
                }
            }
            sum
        })
        .collect();
    (horizontal, vertical)
}

fn merge_boundary_signals(existing: &[u32], color: &[u32]) -> Vec<u32> {
    if existing.len() != color.len() {
        return color.to_vec();
    }
    existing
        .iter()
        .zip(color)
        .map(|(left, right)| (*left).max(*right))
        .collect()
}

fn refine_axis_lines_to_local_edges(lines: &[u32], signal: &[u32], pixel_width: u32) -> Vec<u32> {
    refine_axis_lines_by_score(lines, signal.len(), pixel_width, |position| {
        local_edge_signal_score(signal, position)
    })
}

fn refine_axis_lines_by_score<F>(
    lines: &[u32],
    signal_length: usize,
    pixel_width: u32,
    score_at: F,
) -> Vec<u32>
where
    F: Fn(usize) -> f32,
{
    refine_axis_lines_by_score_with_confidence(lines, signal_length, pixel_width, score_at).lines
}

fn refine_axis_lines_by_score_with_confidence<F>(
    lines: &[u32],
    signal_length: usize,
    pixel_width: u32,
    score_at: F,
) -> AxisLineRefinement
where
    F: Fn(usize) -> f32,
{
    if lines.len() < 3 || signal_length < 3 || pixel_width <= 1 {
        return AxisLineRefinement {
            lines: lines.to_vec(),
            confidences: vec![1.0; lines.len()],
        };
    }

    let radius = (pixel_width as f32 * LOCAL_EDGE_REFINEMENT_RADIUS_RATIO)
        .round()
        .max(1.0)
        .min(LOCAL_EDGE_REFINEMENT_MAX_RADIUS as f32) as u32;
    let candidate_sets = lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 || index + 1 == lines.len() {
                vec![*line]
            } else {
                axis_line_candidates(signal_length, *line, radius)
            }
        })
        .collect::<Vec<_>>();
    let max_edge_score = candidate_sets
        .iter()
        .flatten()
        .map(|position| score_at(*position as usize))
        .fold(1.0f32, f32::max);
    let candidate_scores = candidate_sets
        .iter()
        .enumerate()
        .map(|(line_index, candidates)| {
            candidates
                .iter()
                .map(|position| {
                    let edge_score = score_at(*position as usize) / max_edge_score;
                    let shift_penalty = LOCAL_EDGE_REFINEMENT_SHIFT_PENALTY
                        * position.abs_diff(lines[line_index]) as f32
                        / radius.max(1) as f32;
                    edge_score - shift_penalty
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let candidate_edge_scores = candidate_sets
        .iter()
        .map(|candidates| {
            candidates
                .iter()
                .map(|position| score_at(*position as usize) / max_edge_score)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut scores = candidate_sets
        .iter()
        .map(|candidates| vec![f32::NEG_INFINITY; candidates.len()])
        .collect::<Vec<_>>();
    let mut previous_indexes = candidate_sets
        .iter()
        .map(|candidates| vec![usize::MAX; candidates.len()])
        .collect::<Vec<_>>();
    scores[0][0] = candidate_scores[0][0];

    for line_index in 1..candidate_sets.len() {
        let expected_gap = lines[line_index]
            .saturating_sub(lines[line_index - 1])
            .max(1);
        let min_gap = ((expected_gap as f32) * 0.58).floor().max(1.0) as u32;
        let max_gap = ((expected_gap as f32) * 1.42).ceil() as u32;

        for (candidate_index, candidate) in candidate_sets[line_index].iter().enumerate() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_previous_index = usize::MAX;
            for (previous_index, previous) in candidate_sets[line_index - 1].iter().enumerate() {
                let gap = candidate.saturating_sub(*previous);
                if gap < min_gap || gap > max_gap {
                    continue;
                }
                let gap_delta = (gap as f32 - expected_gap as f32) / expected_gap as f32;
                let gap_penalty = LOCAL_EDGE_REFINEMENT_GAP_PENALTY * gap_delta * gap_delta;
                let score = scores[line_index - 1][previous_index]
                    + candidate_scores[line_index][candidate_index]
                    - gap_penalty;
                if score > best_score {
                    best_score = score;
                    best_previous_index = previous_index;
                }
            }
            scores[line_index][candidate_index] = best_score;
            previous_indexes[line_index][candidate_index] = best_previous_index;
        }
    }

    let Some((mut best_index, best_score)) = scores
        .last()
        .and_then(|last| {
            last.iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
        })
        .map(|(index, score)| (index, *score))
    else {
        return AxisLineRefinement {
            lines: lines.to_vec(),
            confidences: vec![1.0; lines.len()],
        };
    };
    if !best_score.is_finite() {
        return AxisLineRefinement {
            lines: lines.to_vec(),
            confidences: vec![1.0; lines.len()],
        };
    }

    let mut refined = vec![0; lines.len()];
    for line_index in (0..lines.len()).rev() {
        refined[line_index] = candidate_sets[line_index][best_index];
        best_index = previous_indexes[line_index][best_index];
        if line_index > 0 && best_index == usize::MAX {
            return AxisLineRefinement {
                lines: lines.to_vec(),
                confidences: vec![1.0; lines.len()],
            };
        }
    }
    let confidences = refined
        .iter()
        .enumerate()
        .map(|(line_index, line)| {
            if line_index == 0 || line_index + 1 == refined.len() {
                return 1.0;
            }
            let selected_score = score_at(*line as usize) / max_edge_score;
            let base_score = score_at(lines[line_index] as usize) / max_edge_score;
            let selected_index = candidate_sets[line_index]
                .iter()
                .position(|candidate| candidate == line);
            let alternative_score = selected_index
                .and_then(|selected_index| {
                    candidate_edge_scores[line_index]
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != selected_index)
                        .map(|(_, score)| *score)
                        .max_by(f32::total_cmp)
                })
                .unwrap_or(0.0);
            let prominence = (selected_score - alternative_score).max(0.0);
            let improvement = (selected_score - base_score).max(0.0);
            let moved = *line != lines[line_index];
            if moved
                && (prominence < LOCAL_EDGE_MIN_PROMINENCE
                    || improvement < LOCAL_EDGE_MIN_IMPROVEMENT)
            {
                return 0.0;
            }
            (selected_score * 0.65 + improvement * 0.45 + prominence * 0.4).clamp(0.0, 1.0)
        })
        .collect();
    AxisLineRefinement {
        lines: refined,
        confidences,
    }
}

fn axis_line_candidates(signal_length: usize, line: u32, radius: u32) -> Vec<u32> {
    let start = line.saturating_sub(radius).max(1);
    let end = (line + radius).min(signal_length.saturating_sub(2) as u32);
    if start > end {
        return vec![line];
    }
    (start..=end).collect()
}

fn local_edge_signal_score(signal: &[u32], position: usize) -> f32 {
    let clamped = position.min(signal.len().saturating_sub(1));
    (signal.get(clamped.saturating_sub(1)).copied().unwrap_or(0) as f32 * 0.25)
        + signal.get(clamped).copied().unwrap_or(0) as f32
        + signal.get(clamped + 1).copied().unwrap_or(0) as f32 * 0.25
}

fn luma_delta(left: [u8; 4], right: [u8; 4]) -> u32 {
    luma(left).abs_diff(luma(right)) as u32
}

fn color_edge_delta(left: [u8; 4], right: [u8; 4]) -> u32 {
    let channel_delta = left[0]
        .abs_diff(right[0])
        .max(left[1].abs_diff(right[1]))
        .max(left[2].abs_diff(right[2])) as u32;
    let alpha_delta = left[3].abs_diff(right[3]) as u32;
    channel_delta.max(alpha_delta).max(luma_delta(left, right))
}

fn luma(pixel: [u8; 4]) -> u16 {
    if pixel[3] < 128 {
        return 0;
    }
    ((pixel[0] as u32 * 299 + pixel[1] as u32 * 587 + pixel[2] as u32 * 114) / 1000) as u16
}

fn crop_debug_detection_image(image: &RawImage, offset: (u32, u32)) -> RawImage {
    let left = offset.0.min(image.width.saturating_sub(1));
    let top = offset.1.min(image.height.saturating_sub(1));
    let width = image.width.saturating_sub(left * 2).max(1);
    let height = image.height.saturating_sub(top * 2).max(1);
    let mut out = RawImage::transparent(width, height);
    for y in 0..height {
        for x in 0..width {
            out.set_pixel(x, y, image.pixel(x + left, y + top));
        }
    }
    out
}

fn create_edge_preview(image: &RawImage, edge_close_kernel_size: u32) -> RawImage {
    let raw_edge_mask = color_edge_mask(image);
    let closed_edge_mask = closed_color_edge_mask(image, edge_close_kernel_size);
    let mut out = RawImage::transparent(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let index = (y * image.width + x) as usize;
            let color = if raw_edge_mask[index] != 0 {
                DEBUG_RAW_EDGE_COLOR
            } else if closed_edge_mask[index] != 0 {
                DEBUG_CLOSED_EDGE_COLOR
            } else {
                [0, 0, 0, 255]
            };
            out.set_pixel(x, y, color);
        }
    }
    out
}

fn draw_detection_line_overlay(
    base: &RawImage,
    edge_mask: &RawImage,
    crop_offset: (u32, u32),
    preview_scale: u32,
) -> RawImage {
    let mut out = base.clone();
    let min_length = MIN_DEBUG_SEGMENT_LENGTH
        .max(edge_mask.width.max(edge_mask.height) / 12)
        .max(1);
    let mut horizontal_count = 0usize;
    for y in 0..edge_mask.height {
        let mut x = 0;
        while x < edge_mask.width {
            while x < edge_mask.width && edge_mask.pixel(x, y)[0] == 0 {
                x += 1;
            }
            let start = x;
            while x < edge_mask.width && edge_mask.pixel(x, y)[0] > 0 {
                x += 1;
            }
            if x.saturating_sub(start) >= min_length {
                draw_line(
                    &mut out,
                    (start + crop_offset.0) * preview_scale,
                    (y + crop_offset.1) * preview_scale,
                    (x.saturating_sub(1) + crop_offset.0) * preview_scale,
                    (y + crop_offset.1) * preview_scale,
                    [255, 140, 0, 255],
                );
                horizontal_count += 1;
                if horizontal_count >= MAX_DEBUG_SEGMENTS_PER_FAMILY {
                    break;
                }
            }
        }
        if horizontal_count >= MAX_DEBUG_SEGMENTS_PER_FAMILY {
            break;
        }
    }

    let mut vertical_count = 0usize;
    for x in 0..edge_mask.width {
        let mut y = 0;
        while y < edge_mask.height {
            while y < edge_mask.height && edge_mask.pixel(x, y)[0] == 0 {
                y += 1;
            }
            let start = y;
            while y < edge_mask.height && edge_mask.pixel(x, y)[0] > 0 {
                y += 1;
            }
            if y.saturating_sub(start) >= min_length {
                draw_line(
                    &mut out,
                    (x + crop_offset.0) * preview_scale,
                    (start + crop_offset.1) * preview_scale,
                    (x + crop_offset.0) * preview_scale,
                    (y.saturating_sub(1) + crop_offset.1) * preview_scale,
                    [0, 160, 255, 255],
                );
                vertical_count += 1;
                if vertical_count >= MAX_DEBUG_SEGMENTS_PER_FAMILY {
                    break;
                }
            }
        }
        if vertical_count >= MAX_DEBUG_SEGMENTS_PER_FAMILY {
            break;
        }
    }
    out
}

fn draw_anchor_overlay(base: &RawImage, mesh: &MeshResult, preview_scale: u32) -> RawImage {
    let mut out = base.clone();
    let Some(lines_x) = &mesh.debug_anchor_lines_x else {
        return out;
    };
    let Some(lines_y) = &mesh.debug_anchor_lines_y else {
        return out;
    };

    let mesh_scale = mesh.scale_used.max(1);
    let out_width = out.width;
    let out_height = out.height;
    for x in lines_x {
        let scaled_x = debug_grid_coordinate(
            *x,
            mesh.debug_crop_offset.0,
            mesh_scale,
            preview_scale,
            out_width,
        );
        draw_line(
            &mut out,
            scaled_x,
            0,
            scaled_x,
            out_height - 1,
            [0, 255, 255, 255],
        );
    }
    for y in lines_y {
        let scaled_y = debug_grid_coordinate(
            *y,
            mesh.debug_crop_offset.1,
            mesh_scale,
            preview_scale,
            out_height,
        );
        draw_line(
            &mut out,
            0,
            scaled_y,
            out_width - 1,
            scaled_y,
            [255, 190, 0, 255],
        );
    }
    for x in lines_x {
        let scaled_x = debug_grid_coordinate(
            *x,
            mesh.debug_crop_offset.0,
            mesh_scale,
            preview_scale,
            out_width,
        );
        for y in lines_y {
            let scaled_y = debug_grid_coordinate(
                *y,
                mesh.debug_crop_offset.1,
                mesh_scale,
                preview_scale,
                out_height,
            );
            draw_anchor_point(&mut out, scaled_x, scaled_y, [255, 0, 255, 255]);
        }
    }
    out
}

fn draw_anchor_point(image: &mut RawImage, x: u32, y: u32, color: [u8; 4]) {
    const RADIUS: i32 = 2;
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            if dx.abs() + dy.abs() > RADIUS {
                continue;
            }
            let px = x as i32 + dx;
            let py = y as i32 + dy;
            if px >= 0 && py >= 0 && px < image.width as i32 && py < image.height as i32 {
                image.set_pixel(px as u32, py as u32, color);
            }
        }
    }
}

fn draw_grid_overlay_with_options(
    base: &RawImage,
    source: &RawImage,
    mesh: &MeshResult,
    preview_scale: u32,
    warp_options: WarpSubdivisionOptions,
) -> RawImage {
    let mut out = base.clone();
    let mesh_scale = mesh.scale_used.max(1);
    let out_width = out.width;
    let out_height = out.height;
    if mesh.mesh.warp.is_some() {
        let column_count = mesh.mesh.lines_x.len().saturating_sub(1);
        let row_count = mesh.mesh.lines_y.len().saturating_sub(1);
        for cell_y in 0..row_count {
            for cell_x in 0..column_count {
                draw_unwarped_cell_grid_edges(&mut out, mesh, cell_x, cell_y, preview_scale);
            }
        }
        for cell_y in 0..row_count {
            for cell_x in 0..column_count {
                if let Some(corners) = warped_cell_corners(&mesh.mesh, cell_x, cell_y) {
                    draw_warped_cell_grid_edges(
                        &mut out,
                        source,
                        corners,
                        mesh,
                        cell_x,
                        cell_y,
                        preview_scale,
                        warp_options,
                        false,
                    );
                }
            }
        }
        for cell_y in 0..row_count {
            for cell_x in 0..column_count {
                if let Some(corners) = warped_cell_corners(&mesh.mesh, cell_x, cell_y) {
                    draw_warped_cell_grid_edges(
                        &mut out,
                        source,
                        corners,
                        mesh,
                        cell_x,
                        cell_y,
                        preview_scale,
                        warp_options,
                        true,
                    );
                }
            }
        }
    } else {
        for x in &mesh.mesh.lines_x {
            let scaled_x = debug_grid_coordinate(
                *x,
                mesh.debug_crop_offset.0,
                mesh_scale,
                preview_scale,
                out_width,
            );
            for y in 0..out.height {
                out.set_pixel(scaled_x, y, DEBUG_GRID_COLOR);
            }
        }
        for y in &mesh.mesh.lines_y {
            let scaled_y = debug_grid_coordinate(
                *y,
                mesh.debug_crop_offset.1,
                mesh_scale,
                preview_scale,
                out_height,
            );
            for x in 0..out.width {
                out.set_pixel(x, scaled_y, DEBUG_GRID_COLOR);
            }
        }
    }
    out
}

#[cfg(test)]
fn draw_grid_overlay(base: &RawImage, mesh: &MeshResult, preview_scale: u32) -> RawImage {
    draw_grid_overlay_with_options(
        base,
        base,
        mesh,
        preview_scale,
        WarpSubdivisionOptions::default(),
    )
}

fn draw_unwarped_cell_grid_edges(
    image: &mut RawImage,
    mesh: &MeshResult,
    cell_x: usize,
    cell_y: usize,
    preview_scale: u32,
) {
    let corners = unwarped_cell_corners(&mesh.mesh, cell_x, cell_y);
    draw_cell_grid_edges(
        image,
        corners,
        mesh,
        preview_scale,
        DEBUG_UNWARPED_GRID_COLOR,
    );
}

fn draw_warped_cell_grid_edges(
    image: &mut RawImage,
    source: &RawImage,
    corners: WarpedCellCorners,
    mesh: &MeshResult,
    cell_x: usize,
    cell_y: usize,
    preview_scale: u32,
    warp_options: WarpSubdivisionOptions,
    draw_subdivided: bool,
) {
    let resolved = resolve_warped_cell(source, &mesh.mesh, cell_x, cell_y, warp_options, corners);
    if !draw_subdivided && !resolved.uses_warped_corners {
        draw_cell_grid_edges(
            image,
            corners,
            mesh,
            preview_scale,
            DEBUG_UNUSED_WARPED_GRID_COLOR,
        );
    }
    let subdivision =
        subdivided_warped_cell_grid_with_options(source, resolved.corners, resolved.options);
    if draw_subdivided {
        draw_subdivided_edge_segments(image, &subdivision, mesh, preview_scale);
    } else {
        draw_unsubdivided_edge_segments(image, &subdivision, mesh, preview_scale);
    }
}

fn unwarped_cell_corners(mesh: &Mesh, cell_x: usize, cell_y: usize) -> WarpedCellCorners {
    WarpedCellCorners {
        top_left: (mesh.lines_x[cell_x] as f32, mesh.lines_y[cell_y] as f32),
        top_right: (mesh.lines_x[cell_x + 1] as f32, mesh.lines_y[cell_y] as f32),
        bottom_left: (mesh.lines_x[cell_x] as f32, mesh.lines_y[cell_y + 1] as f32),
        bottom_right: (
            mesh.lines_x[cell_x + 1] as f32,
            mesh.lines_y[cell_y + 1] as f32,
        ),
    }
}

fn draw_subdivided_edge_segments(
    image: &mut RawImage,
    subdivision: &WarpedCellSubgrid,
    mesh: &MeshResult,
    preview_scale: u32,
) {
    for edge in &subdivision.edge_refinements {
        let mut visible_index = 0usize;
        for (start, end) in &edge.subdivided_segments {
            let color = if visible_index % 2 == 0 {
                DEBUG_SUBDIVIDED_EDGE_COLOR_A
            } else {
                DEBUG_SUBDIVIDED_EDGE_COLOR_B
            };
            draw_warped_grid_edge(image, *start, *end, mesh, preview_scale, color);
            visible_index += 1;
        }
    }
}

fn draw_unsubdivided_edge_segments(
    image: &mut RawImage,
    subdivision: &WarpedCellSubgrid,
    mesh: &MeshResult,
    preview_scale: u32,
) {
    for edge in &subdivision.edge_refinements {
        if !edge.subdivided_segments.is_empty() {
            continue;
        }
        let Some(start) = edge.points.first().copied() else {
            continue;
        };
        let Some(end) = edge.points.last().copied() else {
            continue;
        };
        draw_warped_grid_edge(image, start, end, mesh, preview_scale, DEBUG_GRID_COLOR);
    }
}

fn draw_cell_grid_edges(
    image: &mut RawImage,
    corners: WarpedCellCorners,
    mesh: &MeshResult,
    preview_scale: u32,
    color: [u8; 4],
) {
    draw_warped_grid_edge(
        image,
        corners.top_left,
        corners.top_right,
        mesh,
        preview_scale,
        color,
    );
    draw_warped_grid_edge(
        image,
        corners.bottom_left,
        corners.bottom_right,
        mesh,
        preview_scale,
        color,
    );
    draw_warped_grid_edge(
        image,
        corners.top_left,
        corners.bottom_left,
        mesh,
        preview_scale,
        color,
    );
    draw_warped_grid_edge(
        image,
        corners.top_right,
        corners.bottom_right,
        mesh,
        preview_scale,
        color,
    );
}

fn draw_warped_grid_edge(
    image: &mut RawImage,
    start: (f32, f32),
    end: (f32, f32),
    mesh: &MeshResult,
    preview_scale: u32,
    color: [u8; 4],
) {
    let mesh_scale = mesh.scale_used.max(1);
    draw_line(
        image,
        debug_grid_coordinate(
            start.0.round().max(0.0) as u32,
            mesh.debug_crop_offset.0,
            mesh_scale,
            preview_scale,
            image.width,
        ),
        debug_grid_coordinate(
            start.1.round().max(0.0) as u32,
            mesh.debug_crop_offset.1,
            mesh_scale,
            preview_scale,
            image.height,
        ),
        debug_grid_coordinate(
            end.0.round().max(0.0) as u32,
            mesh.debug_crop_offset.0,
            mesh_scale,
            preview_scale,
            image.width,
        ),
        debug_grid_coordinate(
            end.1.round().max(0.0) as u32,
            mesh.debug_crop_offset.1,
            mesh_scale,
            preview_scale,
            image.height,
        ),
        color,
    );
}

fn debug_grid_coordinate(
    line: u32,
    offset: u32,
    mesh_scale: u32,
    preview_scale: u32,
    max: u32,
) -> u32 {
    line.saturating_mul(mesh_scale)
        .saturating_add(offset)
        .saturating_mul(preview_scale)
        .min(max.saturating_sub(1))
}

fn draw_line(image: &mut RawImage, x1: u32, y1: u32, x2: u32, y2: u32, color: [u8; 4]) {
    let mut current_x = x1 as i32;
    let mut current_y = y1 as i32;
    let target_x = x2 as i32;
    let target_y = y2 as i32;
    let delta_x = (target_x - current_x).abs();
    let step_x = if current_x < target_x { 1 } else { -1 };
    let delta_y = -(target_y - current_y).abs();
    let step_y = if current_y < target_y { 1 } else { -1 };
    let mut error = delta_x + delta_y;

    loop {
        if current_x >= 0
            && current_y >= 0
            && current_x < image.width as i32
            && current_y < image.height as i32
        {
            blend_debug_pixel(image, current_x as u32, current_y as u32, color);
        }
        if current_x == target_x && current_y == target_y {
            break;
        }
        let doubled_error = error * 2;
        if doubled_error >= delta_y {
            error += delta_y;
            current_x += step_x;
        }
        if doubled_error <= delta_x {
            error += delta_x;
            current_y += step_y;
        }
    }
}

fn blend_debug_pixel(image: &mut RawImage, x: u32, y: u32, color: [u8; 4]) {
    if color[3] == 255 {
        image.set_pixel(x, y, color);
        return;
    }
    if color[3] == 0 {
        return;
    }

    let base = image.pixel(x, y);
    let src_alpha = color[3] as u32;
    let base_alpha = base[3] as u32;
    let inv_alpha = 255 - src_alpha;
    let out_alpha = src_alpha + base_alpha * inv_alpha / 255;
    if out_alpha == 0 {
        image.set_pixel(x, y, [0, 0, 0, 0]);
        return;
    }

    let mut blended = [0u8; 4];
    for channel in 0..3 {
        let src = color[channel] as u32;
        let dst = base[channel] as u32;
        blended[channel] =
            ((src * src_alpha * 255 + dst * base_alpha * inv_alpha) / (out_alpha * 255)) as u8;
    }
    blended[3] = out_alpha as u8;
    image.set_pixel(x, y, blended);
}

#[allow(dead_code)]
pub fn detector_name(detector: PixelWidthDetector) -> &'static str {
    match detector {
        PixelWidthDetector::Projection => "projection",
        PixelWidthDetector::Hough => "hough",
        PixelWidthDetector::Hybrid => "hybrid",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_mesh() -> MeshResult {
        MeshResult {
            mesh: Mesh::regular(2, 2, 1),
            detected_pixel_width: 1,
            pixel_width_source: PixelWidthSource::Manual,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        }
    }

    fn tiny_original() -> RawImage {
        RawImage::new(
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
        )
    }

    fn color_bounds(image: &RawImage, color: [u8; 4]) -> Option<(u32, u32, u32, u32)> {
        let mut left = image.width;
        let mut top = image.height;
        let mut right = 0;
        let mut bottom = 0;
        let mut found = false;
        for y in 0..image.height {
            for x in 0..image.width {
                if image.pixel(x, y) == color {
                    left = left.min(x);
                    top = top.min(y);
                    right = right.max(x);
                    bottom = bottom.max(y);
                    found = true;
                }
            }
        }
        found.then_some((left, top, right, bottom))
    }

    #[test]
    fn debug_sheet_places_final_and_palette_on_second_row() {
        let debug = create_debug_sheet(
            &tiny_original(),
            &RawImage::new(1, 1, vec![255, 0, 0, 255]),
            &tiny_mesh(),
            &[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            1,
            1.0,
        );

        assert_eq!(debug.width, 68);
        assert_eq!(debug.height, 40);
    }

    #[test]
    fn debug_sheet_layout_is_stable_when_palette_merge_is_disabled() {
        let debug = create_debug_sheet(
            &tiny_original(),
            &RawImage::new(1, 1, vec![255, 0, 0, 255]),
            &tiny_mesh(),
            &[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            1,
            0.0,
        );

        assert_eq!(debug.width, 68);
        assert_eq!(debug.height, 40);
    }

    #[test]
    fn transparency_debug_sheet_includes_background_mask_panels() {
        let black = [0, 0, 0, 255];
        let red = [255, 0, 0, 255];
        let final_color = [17, 33, 51, 255];
        let mut data = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                data.extend_from_slice(if x == 1 && y == 1 { &red } else { &black });
            }
        }
        let original = RawImage::new(3, 3, data);
        let mesh = MeshResult {
            mesh: Mesh::regular(3, 3, 1),
            detected_pixel_width: 1,
            pixel_width_source: PixelWidthSource::Manual,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };

        let debug = create_debug_sheet_with_options(
            &original,
            &RawImage::new(
                2,
                2,
                [final_color, final_color, final_color, final_color].concat(),
            ),
            &mesh,
            &[[255, 0, 0], [0, 255, 0]],
            DebugSheetOptions {
                debug_scale: 1,
                palette_merge_threshold: 1.0,
                transparent_background: true,
                edge_close_kernel_size: 0,
                sample_grid: 1,
                warp_subdivision: WarpSubdivisionOptions::default(),
            },
        );

        assert!(
            debug
                .data
                .chunks_exact(4)
                .any(|pixel| pixel == &DEBUG_BACKGROUND_MASK_COLOR[..])
        );
        assert!(
            debug
                .data
                .chunks_exact(4)
                .any(|pixel| pixel == &DEBUG_BACKGROUND_COVERAGE_STRONG_COLOR[..])
        );

        let mask_bounds = color_bounds(&debug, DEBUG_BACKGROUND_MASK_COLOR).unwrap();
        let coverage_bounds = color_bounds(&debug, DEBUG_BACKGROUND_COVERAGE_STRONG_COLOR).unwrap();
        let final_bounds = color_bounds(&debug, final_color).unwrap();
        let palette_bounds = color_bounds(&debug, [0, 255, 0, 255]).unwrap();
        assert!(final_bounds.1 > mask_bounds.3);
        assert!(final_bounds.1 > coverage_bounds.3);
        assert!(palette_bounds.0 > final_bounds.2);
    }

    #[test]
    fn debug_sheet_includes_final_output_without_unscaled_duplicate() {
        let output_color = [17, 33, 51, 255];
        let debug = create_debug_sheet(
            &tiny_original(),
            &RawImage::new(1, 1, output_color.to_vec()),
            &tiny_mesh(),
            &[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            1,
            1.0,
        );

        let output_pixels = debug
            .data
            .chunks_exact(4)
            .filter(|pixel| *pixel == output_color)
            .count();
        assert_eq!(output_pixels, 4);
    }

    #[test]
    fn debug_palette_preview_is_large_enough_without_filling_panel() {
        let palette = create_debug_palette_preview(
            &[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            240,
            160,
        );

        assert!(palette.width >= 32);
        assert!(palette.height >= 32);
        assert!(palette.width <= 240 * 36 / 100);
        assert!(palette.height <= 160 * 24 / 100);
    }

    #[test]
    fn debug_palette_preview_caps_large_palettes_to_panel_fraction() {
        let colors = (0..64)
            .map(|value| [value * 3, value * 2, value])
            .collect::<Vec<_>>();
        let palette = create_debug_palette_preview(&colors, 240, 160);

        assert!(palette.width <= 240 * 36 / 100);
        assert!(palette.height <= 160 * 24 / 100);
    }

    #[test]
    fn anchor_debug_overlay_uses_mesh_coordinate_frame_without_extra_offset() {
        let base = RawImage::transparent(7, 7);
        let mut mesh = tiny_mesh();
        mesh.debug_anchor_lines_x = Some(vec![3]);
        mesh.debug_anchor_lines_y = Some(vec![4]);

        let overlay = draw_anchor_overlay(&base, &mesh, 1);

        assert_eq!(overlay.pixel(3, 0), [0, 255, 255, 255]);
        assert_eq!(overlay.pixel(6, 4), [255, 190, 0, 255]);
        assert_eq!(overlay.pixel(3, 4), [255, 0, 255, 255]);
        assert_eq!(overlay.pixel(6, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn sample_cells_uses_warped_boundaries() {
        let image = RawImage::new(
            4,
            2,
            vec![
                255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255,
                255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255,
            ],
        );
        let mesh = Mesh {
            lines_x: vec![0, 2, 4],
            lines_y: vec![0, 2],
            warp: Some(MeshWarp {
                lines_x_by_row: vec![vec![2, 3, 4]],
                lines_y_by_column: vec![vec![0, 2], vec![0, 2]],
            }),
        };

        let sampled = sample_cells(&image, &mesh, 1, false);

        assert_eq!(sampled.pixel(0, 0), [0, 0, 255, 255]);
    }

    #[test]
    fn sample_cells_preserves_same_color_subject_pixels_disconnected_from_background() {
        let background = [103, 214, 79, 255];
        let object = [255, 0, 0, 255];
        let cells = [
            background, background, background, background, background, background, object, object,
            object, background, background, object, background, object, background, background,
            object, object, object, background, background, background, background, background,
            background,
        ];
        let mut data = Vec::new();
        for cell_y in 0..5 {
            for _ in 0..2 {
                for cell_x in 0..5 {
                    for _ in 0..2 {
                        data.extend_from_slice(&cells[cell_y * 5 + cell_x]);
                    }
                }
            }
        }
        let image = RawImage::new(10, 10, data);
        let mesh = Mesh {
            lines_x: vec![0, 2, 4, 6, 8, 10],
            lines_y: vec![0, 2, 4, 6, 8, 10],
            warp: None,
        };

        let sampled = sample_cells(&image, &mesh, 5, true);

        assert_eq!(sampled.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(sampled.pixel(1, 1), object);
        assert_eq!(sampled.pixel(2, 2), background);
    }

    #[test]
    fn transparent_sample_cells_preserve_sparse_subject_pixels() {
        let background = [0, 1, 21, 255];
        let object = [255, 0, 0, 255];
        let mut image = RawImage::new(5, 5, background.repeat(25));
        image.set_pixel(2, 2, object);
        image.set_pixel(2, 3, object);
        let mesh = Mesh {
            lines_x: vec![0, 5],
            lines_y: vec![0, 5],
            warp: None,
        };

        let opaque = sample_cells(&image, &mesh, 5, false);
        let transparent = sample_cells(&image, &mesh, 5, true);

        assert_eq!(opaque.pixel(0, 0), background);
        assert_eq!(transparent.pixel(0, 0), object);
    }

    #[test]
    fn sampled_fringe_removes_sparse_vivid_background_colors() {
        let background = [103, 214, 79, 255];
        let mut image = RawImage::new(3, 1, vec![0, 0, 0, 0, 105, 172, 76, 255, 230, 80, 0, 255]);

        remove_sampled_background_fringe(&mut image, &[255, 200, 0], &background);

        assert_eq!(image.pixel(1, 0), [0, 0, 0, 0]);
        assert_eq!(image.pixel(2, 0), [230, 80, 0, 255]);
    }

    #[test]
    fn sampled_fringe_removes_vivid_background_color_in_warm_context() {
        let background = [103, 214, 79, 255];
        let mut image = RawImage::new(3, 3, [230, 80, 0, 255].repeat(9));
        image.set_pixel(0, 1, [0, 0, 0, 0]);
        image.set_pixel(1, 1, [109, 118, 48, 255]);
        image.set_pixel(2, 1, [92, 136, 57, 255]);

        remove_sampled_background_fringe(
            &mut image,
            &[0, 200, 200, 0, 255, 200, 0, 0, 0],
            &background,
        );

        assert_eq!(image.pixel(1, 1), [0, 0, 0, 0]);
        assert_eq!(image.pixel(2, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn sampled_spill_removes_vivid_background_color_inside_warm_context() {
        let background = [103, 214, 79, 255];
        let mut image = RawImage::new(3, 3, [230, 80, 0, 255].repeat(9));
        image.set_pixel(1, 1, [109, 118, 48, 255]);

        remove_sampled_background_fringe(&mut image, &[0, 0, 0, 0, 255, 0, 0, 0, 0], &background);

        assert_eq!(image.pixel(1, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn sampled_spill_keeps_vivid_background_colored_subject_cluster() {
        let background = [103, 214, 79, 255];
        let mut image = RawImage::new(3, 3, [40, 90, 20, 255].repeat(9));
        image.set_pixel(1, 1, [109, 118, 48, 255]);

        remove_sampled_background_fringe(&mut image, &[0, 0, 0, 0, 255, 0, 0, 0, 0], &background);

        assert_eq!(image.pixel(1, 1), [109, 118, 48, 255]);
    }

    #[test]
    fn sampled_fringe_keeps_vivid_background_colored_subject_cluster() {
        let background = [103, 214, 79, 255];
        let mut image = RawImage::new(
            4,
            1,
            vec![
                0, 0, 0, 0, 105, 172, 76, 255, 92, 135, 54, 255, 230, 80, 0, 255,
            ],
        );

        remove_sampled_background_fringe(&mut image, &[255, 200, 0, 0], &background);

        assert_eq!(image.pixel(1, 0), [105, 172, 76, 255]);
        assert_eq!(image.pixel(2, 0), [92, 135, 54, 255]);
    }

    #[test]
    fn sampled_fringe_keeps_dark_background_subject_outlines() {
        let background = [0, 1, 21, 255];
        let mut image = RawImage::new(3, 1, vec![0, 0, 0, 0, 0, 0, 0, 255, 220, 0, 0, 255]);

        remove_sampled_background_fringe(&mut image, &[255, 200, 0], &background);

        assert_eq!(image.pixel(1, 0), [0, 0, 0, 255]);
        assert_eq!(image.pixel(2, 0), [220, 0, 0, 255]);
    }

    #[test]
    fn sampled_fringe_removes_dark_background_pixels_without_local_support() {
        let background = [0, 1, 21, 255];
        let mut image = RawImage::new(3, 1, vec![0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0]);

        remove_sampled_background_fringe(&mut image, &[255, 200, 255], &background);

        assert_eq!(image.pixel(1, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn warped_grid_overlay_draws_connected_merged_nodes() {
        let base = RawImage::transparent(7, 7);
        let mesh = MeshResult {
            mesh: Mesh {
                lines_x: vec![0, 3, 6],
                lines_y: vec![0, 3, 6],
                warp: Some(MeshWarp {
                    lines_x_by_row: vec![vec![0, 3, 6], vec![2, 4, 6]],
                    lines_y_by_column: vec![vec![0, 3, 6], vec![0, 4, 6]],
                }),
            },
            detected_pixel_width: 3,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };

        let corners = warped_cell_corners(&mesh.mesh, 0, 0).unwrap();
        let overlay = draw_grid_overlay(&base, &mesh, 1);

        assert_eq!(corners.bottom_left, (1.0, 3.0));
        assert_eq!(overlay.pixel(1, 3), DEBUG_GRID_COLOR);
    }

    #[test]
    fn warped_grid_overlay_draws_unwarped_grid_in_light_blue_under_red_grid() {
        let base = RawImage::transparent(12, 12);
        let mesh = MeshResult {
            mesh: Mesh {
                lines_x: vec![0, 8],
                lines_y: vec![0, 8],
                warp: Some(MeshWarp {
                    lines_x_by_row: vec![vec![0, 10]],
                    lines_y_by_column: vec![vec![0, 10]],
                }),
            },
            detected_pixel_width: 8,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };

        let overlay = draw_grid_overlay(&base, &mesh, 1);

        assert_eq!(overlay.pixel(8, 4), DEBUG_UNWARPED_GRID_COLOR);
        assert_eq!(overlay.pixel(0, 0), DEBUG_GRID_COLOR);
        assert_eq!(overlay.pixel(5, 5), [0, 0, 0, 0]);
    }

    #[test]
    fn warped_grid_overlay_draws_adaptive_subdivided_edges_in_green() {
        let mut source = RawImage::transparent(7, 7);
        for y in 0..7 {
            for x in 0..7 {
                let value = if y < 2 { 0 } else { 255 };
                source.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let mesh = MeshResult {
            mesh: Mesh {
                lines_x: vec![0, 6],
                lines_y: vec![3, 6],
                warp: Some(MeshWarp {
                    lines_x_by_row: vec![vec![0, 6]],
                    lines_y_by_column: vec![vec![3, 6]],
                }),
            },
            detected_pixel_width: 3,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };
        let corners = warped_cell_corners(&mesh.mesh, 0, 0).unwrap();
        let subdivision = subdivided_warped_cell_grid(&source, corners);
        let midpoint_index = subdivision.points.len() / 2;
        let top_midpoint = subdivision.points[0][midpoint_index];

        let overlay = draw_grid_overlay(&source, &mesh, 1);

        assert!(top_midpoint.1 < 3.0);
        assert!(matches!(
            overlay.pixel(top_midpoint.0.round() as u32, top_midpoint.1.round() as u32),
            DEBUG_SUBDIVIDED_EDGE_COLOR_A | DEBUG_SUBDIVIDED_EDGE_COLOR_B
        ));
        assert_eq!(overlay.pixel(3, 3), DEBUG_UNWARPED_GRID_COLOR);
        assert!(
            subdivision
                .edge_refinements
                .iter()
                .any(|edge| !edge.subdivided_segments.is_empty())
        );
    }

    #[test]
    fn warped_grid_overlay_does_not_subdivide_flat_edges_at_default_depth() {
        let source = RawImage::new(7, 7, [80, 80, 80, 255].repeat(49));
        let mesh = MeshResult {
            mesh: Mesh {
                lines_x: vec![0, 6],
                lines_y: vec![0, 6],
                warp: Some(MeshWarp {
                    lines_x_by_row: vec![vec![0, 6]],
                    lines_y_by_column: vec![vec![0, 6]],
                }),
            },
            detected_pixel_width: 6,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };
        let corners = warped_cell_corners(&mesh.mesh, 0, 0).unwrap();
        let subdivision = subdivided_warped_cell_grid(&source, corners);

        let overlay = draw_grid_overlay(&source, &mesh, 1);

        assert!(
            subdivision
                .edge_refinements
                .iter()
                .all(|edge| edge.subdivided_segments.is_empty())
        );
        assert_eq!(overlay.pixel(3, 3), [80, 80, 80, 255]);
        assert_eq!(overlay.pixel(3, 0), DEBUG_GRID_COLOR);
    }

    #[test]
    fn warp_subdivision_skips_unreliable_dense_edge_regions() {
        let mut image = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (8.0, 4.0),
            bottom_left: (0.0, 8.0),
            bottom_right: (8.0, 8.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 3,
                edge_threshold: 18.0,
            },
        );

        assert_eq!(subdivision.points.len(), 2);
        assert!(
            subdivision
                .edge_refinements
                .iter()
                .all(|edge| edge.subdivided_segments.is_empty())
        );
    }

    #[test]
    fn local_warp_relaxes_noisy_edge_fields_back_to_base_lines() {
        let mut image = RawImage::transparent(12, 12);
        for y in 0..12 {
            for x in 0..12 {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let mesh = Mesh {
            lines_x: vec![0, 4, 8, 11],
            lines_y: vec![0, 4, 8, 11],
            warp: None,
        };
        let result = MeshResult {
            mesh: mesh.clone(),
            detected_pixel_width: 4,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };

        let warp = create_warped_mesh_from_local_edges(&image, &mesh, &result);

        let warp = warp.unwrap();
        assert!(warp.lines_x_by_row.iter().all(|row| row == &mesh.lines_x));
        assert!(
            warp.lines_y_by_column
                .iter()
                .all(|column| column == &mesh.lines_y)
        );
    }

    #[test]
    fn hybrid_warp_falls_back_only_for_unreliable_cells() {
        let mut noisy = RawImage::transparent(80, 80);
        for y in 0..80 {
            for x in 0..80 {
                let value = if x < 40 && (x + y) % 2 == 0 {
                    0
                } else if x < 40 {
                    255
                } else if y < 40 {
                    0
                } else {
                    255
                };
                noisy.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let mesh = Mesh {
            lines_x: vec![0, 40, 79],
            lines_y: vec![0, 40, 79],
            warp: Some(MeshWarp {
                lines_x_by_row: vec![vec![0, 45, 79], vec![0, 45, 79]],
                lines_y_by_column: vec![vec![0, 40, 79], vec![0, 40, 79]],
            }),
        };
        let options = WarpSubdivisionOptions {
            max_depth: 3,
            edge_threshold: 18.0,
        };

        let noisy_cell = resolve_warped_cell(
            &noisy,
            &mesh,
            0,
            0,
            options,
            warped_cell_corners(&mesh, 0, 0).unwrap(),
        );
        let coherent_cell = resolve_warped_cell(
            &noisy,
            &mesh,
            1,
            0,
            options,
            warped_cell_corners(&mesh, 1, 0).unwrap(),
        );

        assert_eq!(noisy_cell.corners.top_right, (40.0, 0.0));
        assert_eq!(noisy_cell.options.max_depth, 0);
        assert!(coherent_cell.corners.top_left.0 >= 40.0);
        assert_eq!(coherent_cell.options.max_depth, 3);
    }

    #[test]
    fn hybrid_warp_limits_over_sheared_cells_instead_of_discarding_them() {
        let source = RawImage::new(20, 12, [80, 80, 80, 255].repeat(240));
        let mesh = Mesh {
            lines_x: vec![0, 8, 16],
            lines_y: vec![0, 8],
            warp: Some(MeshWarp {
                lines_x_by_row: vec![vec![0, 8, 16]],
                lines_y_by_column: vec![vec![0, 8], vec![8, 8]],
            }),
        };
        let corners = warped_cell_corners(&mesh, 0, 0).unwrap();

        let resolved = resolve_warped_cell(
            &source,
            &mesh,
            0,
            0,
            WarpSubdivisionOptions::default(),
            corners,
        );

        assert_eq!(corners.top_right, (8.0, 4.0));
        assert!(resolved.uses_warped_corners);
        assert!(resolved.corners.top_right.1 > 0.0);
        assert!(resolved.corners.top_right.1 < corners.top_right.1);
        assert!(warped_cell_geometry_reliable(
            unwarped_cell_corners(&mesh, 0, 0),
            resolved.corners
        ));
        assert_eq!(resolved.options.max_depth, DEFAULT_WARP_SUBDIVISION_DEPTH);
    }

    #[test]
    fn warped_grid_overlay_marks_unused_warp_candidates_transparently() {
        let mut source = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                source.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let mesh = MeshResult {
            mesh: Mesh {
                lines_x: vec![0, 8],
                lines_y: vec![0, 8],
                warp: Some(MeshWarp {
                    lines_x_by_row: vec![vec![0, 6]],
                    lines_y_by_column: vec![vec![0, 8]],
                }),
            },
            detected_pixel_width: 8,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };

        let overlay = draw_grid_overlay(&source, &mesh, 1);
        let mut expected = source.clone();
        blend_debug_pixel(&mut expected, 6, 4, DEBUG_UNUSED_WARPED_GRID_COLOR);

        assert_eq!(overlay.pixel(6, 4), expected.pixel(6, 4));
        assert_eq!(overlay.pixel(8, 4), DEBUG_GRID_COLOR);
    }

    #[test]
    fn warp_subdivision_depth_zero_uses_unsubdivided_cell() {
        let mut image = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let value = if y < 2 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (8.0, 4.0),
            bottom_left: (0.0, 8.0),
            bottom_right: (8.0, 8.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 0,
                edge_threshold: 1.0,
            },
        );

        assert_eq!(subdivision.points.len(), 2);
        assert_eq!(subdivided_warp_point(&subdivision, 0.5, 0.0), (4.0, 4.0));
    }

    #[test]
    fn warp_subdivision_depth_two_refines_quarter_edge_points() {
        let mut image = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let value = if y < 2 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (8.0, 4.0),
            bottom_left: (0.0, 8.0),
            bottom_right: (8.0, 8.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 2,
                edge_threshold: 1.0,
            },
        );

        assert_eq!(subdivision.points.len(), 5);
        assert!(subdivision.points[0][1].1 < 4.0);
        assert!(subdivision.points[0][3].1 < 4.0);
    }

    #[test]
    fn warp_subdivision_limits_child_points_relative_to_original_edge() {
        let mut image = RawImage::transparent(17, 17);
        for y in 0..17 {
            for x in 0..17 {
                let value = if y < 2 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (16.0, 4.0),
            bottom_left: (0.0, 14.0),
            bottom_right: (16.0, 14.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 3,
                edge_threshold: 18.0,
            },
        );

        assert!(subdivision.points[0][4].1 < 4.0);
        for index in 1..subdivision.points[0].len() {
            let previous_displacement = subdivision.points[0][index - 1].1 - 4.0;
            let displacement = subdivision.points[0][index].1 - 4.0;
            assert!(
                (displacement - previous_displacement).abs()
                    <= WARP_SUBDIVISION_MAX_ADJACENT_SHIFT_DELTA
            );
        }
    }

    #[test]
    fn warp_subdivision_edge_threshold_rejects_weak_texture_edges() {
        let mut image = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let value = if y < 2 { 0 } else { 24 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (8.0, 4.0),
            bottom_left: (0.0, 8.0),
            bottom_right: (8.0, 8.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 2,
                edge_threshold: 60.0,
            },
        );

        assert_eq!(subdivision.points[0][2], (4.0, 4.0));
    }

    #[test]
    fn warp_subdivision_requires_local_edge_evidence_at_candidate_point() {
        let mut image = RawImage::new(9, 9, [0, 0, 0, 255].repeat(81));
        for y in 2..9 {
            image.set_pixel(0, y, [255, 255, 255, 255]);
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (8.0, 4.0),
            bottom_left: (0.0, 8.0),
            bottom_right: (8.0, 8.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 1,
                edge_threshold: 18.0,
            },
        );

        assert_eq!(subdivision.points[0][1], (4.0, 4.0));
        assert!(
            subdivision.edge_refinements[0]
                .subdivided_segments
                .is_empty()
        );
    }

    #[test]
    fn warp_subdivision_rejects_parallel_edge_without_contour_to_endpoints() {
        let mut image = RawImage::transparent(17, 17);
        for y in 0..17 {
            for x in 0..17 {
                let value = if y < 8 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (16.0, 4.0),
            bottom_left: (0.0, 14.0),
            bottom_right: (16.0, 14.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 2,
                edge_threshold: 18.0,
            },
        );

        assert_eq!(subdivision.points[0][2], (8.0, 4.0));
        assert!(
            subdivision.edge_refinements[0]
                .subdivided_segments
                .is_empty()
        );
    }

    #[test]
    fn warp_subdivision_prefers_corner_candidate_over_plain_edge_candidate() {
        let mut image = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let dark = (x < 5 && y < 2) || (x >= 5 && y >= 2);
                let value = if dark { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 4.0),
            top_right: (8.0, 4.0),
            bottom_left: (0.0, 8.0),
            bottom_right: (8.0, 8.0),
        };

        let subdivision = subdivided_warped_cell_grid_with_options(
            &image,
            corners,
            WarpSubdivisionOptions {
                max_depth: 1,
                edge_threshold: 18.0,
            },
        );

        assert!(subdivision.points[0][1].0 > 4.0);
        assert!(subdivision.points[0][1].1 < 4.0);
    }

    #[test]
    fn warp_subdivision_searches_segment_envelope_for_missed_corner() {
        let mut image = RawImage::transparent(9, 9);
        for y in 0..9 {
            for x in 0..9 {
                let value = if x < 5 && y >= 2 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }

        let refined =
            refine_vertical_subdivision_point(&image, (2.5, 5.0), (0.0, 2.0), (5.0, 8.0), 18.0);

        assert_eq!(refined, (5.0, 2.0));
    }

    #[test]
    fn contour_score_penalizes_segments_that_cut_across_flat_pixels() {
        let mut image = RawImage::new(9, 9, [0, 0, 0, 255].repeat(81));
        for y in 0..9 {
            for x in 5..9 {
                image.set_pixel(x, y, [255, 255, 255, 255]);
            }
        }

        let supported =
            subdivision_candidate_contour_score(&image, (5.0, 1.0), (5.0, 4.0), (5.0, 7.0), 18.0);
        let unsupported =
            subdivision_candidate_contour_score(&image, (0.0, 1.0), (5.0, 4.0), (5.0, 7.0), 18.0);

        assert!(supported > unsupported);
        assert!(unsupported < 18.0);
    }

    #[test]
    fn warp_subdivision_does_not_snap_midpoint_onto_segment_endpoint() {
        let mut image = RawImage::new(9, 9, [0, 0, 0, 255].repeat(81));
        for y in 4..9 {
            image.set_pixel(0, y, [255, 255, 255, 255]);
        }

        let refined =
            refine_horizontal_subdivision_point(&image, (2.0, 4.0), (0.0, 4.0), (4.0, 4.0), 100.0);

        assert_ne!(refined, (0.0, 4.0));
        assert_ne!(refined, (4.0, 4.0));
    }

    #[test]
    fn local_warp_tracks_shifted_edges_per_column_band() {
        let mut image = RawImage::transparent(12, 6);
        for y in 0..6 {
            for x in 0..12 {
                let edge_y = if x < 4 {
                    2
                } else if x < 8 {
                    3
                } else {
                    4
                };
                let value = if y < edge_y { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let mesh = Mesh {
            lines_x: vec![0, 4, 8, 11],
            lines_y: vec![0, 3, 5],
            warp: None,
        };
        let result = MeshResult {
            mesh: mesh.clone(),
            detected_pixel_width: 3,
            pixel_width_source: PixelWidthSource::Hough,
            scale_used: 1,
            debug_crop_offset: (0, 0),
            debug_anchor_lines_x: None,
            debug_anchor_lines_y: None,
        };
        let warp = create_warped_mesh_from_local_edges(&image, &mesh, &result).unwrap();

        assert_eq!(warp.lines_y_by_column[0][1], 2);
        assert_eq!(warp.lines_y_by_column[1][1], 3);
        assert_eq!(warp.lines_y_by_column[2][1], 4);
    }

    #[test]
    fn local_edge_energy_rewards_perpendicular_corner_nodes() {
        let mut image = RawImage::transparent(7, 7);
        for y in 0..7 {
            for x in 0..7 {
                let dark = (x < 3 && y < 3) || (x >= 3 && y >= 3);
                let value = if dark { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let energy = LocalEdgeEnergy::new(&image);

        let edge_only = energy.vertical_band_edge_score(3, 3, 6);
        let with_corner = energy.vertical_band_corner_score(3, 3, 6);

        assert!(with_corner > edge_only);
        assert!(energy.corner_score(3, 3) > 0.0);
    }

    #[test]
    fn corner_snap_uses_color_edges_when_luma_is_flat() {
        let red = [255, 0, 0, 255];
        let green_same_luma = [0, 130, 0, 255];
        let mut image = RawImage::transparent(7, 7);
        for y in 0..7 {
            for x in 0..7 {
                let red_quadrant = (x < 3 && y < 3) || (x >= 3 && y >= 3);
                image.set_pixel(x, y, if red_quadrant { red } else { green_same_luma });
            }
        }

        let snapped = refine_warped_corner_point(&image, (1.0, 1.0), 1.0);

        assert_eq!(snapped, (3.0, 3.0));
        assert!(color_edge_delta(red, green_same_luma) > luma_delta(red, green_same_luma));
    }

    #[test]
    fn weak_local_warp_snaps_relax_back_to_regular_grid() {
        let refined = AxisLineRefinement {
            lines: vec![0, 1, 5],
            confidences: vec![1.0, 0.05, 1.0],
        };

        let stabilized = smooth_warped_line_sets(&[0, 3, 5], vec![refined]);

        assert_eq!(stabilized[0], vec![0, 3, 5]);
    }

    #[test]
    fn subdivided_warp_grid_moves_edge_midpoint_to_local_edge() {
        let mut image = RawImage::transparent(7, 7);
        for y in 0..7 {
            for x in 0..7 {
                let value = if y < 2 { 0 } else { 255 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        let corners = WarpedCellCorners {
            top_left: (0.0, 3.0),
            top_right: (6.0, 3.0),
            bottom_left: (0.0, 6.0),
            bottom_right: (6.0, 6.0),
        };

        let subdivision = subdivided_warped_cell_grid(&image, corners);

        assert!(subdivision.points[0][1].1 < 3.0);
    }
}
