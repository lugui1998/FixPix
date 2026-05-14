use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rayon::prelude::*;

use crate::core::{
    ArtifactOptions, ColorMode, DownscaleSampleFrom, OutputFormat, PaletteStrategy,
    PixelWidthDetector, Size, TransformOptions, transform_bytes, write_image_file,
};
use crate::mesh::{DEFAULT_WARP_SUBDIVISION_DEPTH, DEFAULT_WARP_SUBDIVISION_EDGE_THRESHOLD};
use crate::threading;

const DEFAULT_URL_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_URL_MAX_BYTES: u64 = 50 * 1024 * 1024;
const DEFAULT_URL_CONTENT_TYPES: &str = "image/*,application/octet-stream";

#[derive(Debug, Parser)]
#[command(name = "fixpix")]
#[command(about = "Convert noisy pixel-art-style images into clean pixel-resolution sprites.")]
pub struct CliArgs {
    pub input_path_or_url: Option<String>,
    pub output_path_or_dir: Option<PathBuf>,

    #[arg(short = 'i', long = "input")]
    pub input: Option<String>,
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,
    #[arg(short = 'j', long = "jobs")]
    pub jobs: Option<String>,
    #[arg(long = "threads")]
    pub threads: Option<String>,
    #[arg(
        short = 'c',
        long = "colors",
        default_value = "auto",
        allow_hyphen_values = true
    )]
    pub colors: String,
    #[arg(long = "palette-merge-threshold", default_value = "1")]
    pub palette_merge_threshold: f32,
    #[arg(long = "color-sample-grid-size", default_value = "5")]
    pub color_sample_grid_size: u32,
    #[arg(long = "palette-strategy", default_value = "global")]
    pub palette_strategy: String,
    #[arg(short = 's', long = "scale")]
    pub scale: Option<u32>,
    #[arg(long = "auto-scale-width")]
    pub auto_scale_width: Option<u32>,
    #[arg(long = "auto-scale-height")]
    pub auto_scale_height: Option<u32>,
    #[arg(long = "downscale")]
    pub downscale: Option<String>,
    #[arg(long = "downscale-sample-from", default_value = "pixelated")]
    pub downscale_sample_from: String,
    #[arg(short = 't', long = "transparent")]
    pub transparent: bool,
    #[arg(long = "crop")]
    pub crop: bool,
    #[arg(long = "crop-size")]
    pub crop_size: Option<String>,
    #[arg(short = 'w', long = "pixel-width")]
    pub pixel_width: Option<u32>,
    #[arg(long = "pixel-width-detector", default_value = "hybrid")]
    pub pixel_width_detector: String,
    #[arg(short = 'u', long = "initial-upscale", default_value = "2")]
    pub initial_upscale: u32,
    #[arg(
        long = "warp-subdivision-depth",
        default_value_t = DEFAULT_WARP_SUBDIVISION_DEPTH
    )]
    pub warp_subdivision_depth: u32,
    #[arg(
        long = "warp-subdivision-edge-threshold",
        default_value_t = DEFAULT_WARP_SUBDIVISION_EDGE_THRESHOLD
    )]
    pub warp_subdivision_edge_threshold: f32,
    #[arg(short = 'f', long = "format")]
    pub format: Option<String>,
    #[arg(short = 'q', long = "quality")]
    pub quality: Option<u8>,
    #[arg(long = "url-timeout-ms", default_value_t = DEFAULT_URL_TIMEOUT_MS)]
    pub url_timeout_ms: u64,
    #[arg(long = "url-max-bytes", default_value_t = DEFAULT_URL_MAX_BYTES)]
    pub url_max_bytes: u64,
    #[arg(long = "url-content-types", default_value = DEFAULT_URL_CONTENT_TYPES)]
    pub url_content_types: String,
    #[arg(long = "debug-out")]
    pub debug_out: Option<PathBuf>,
    #[arg(long = "debug-scale", default_value = "6")]
    pub debug_scale: u32,
    #[arg(long = "unscaled-out")]
    pub unscaled_out: Option<PathBuf>,
    #[arg(long = "palette-out")]
    pub palette_out: Option<PathBuf>,
    #[arg(long = "palette-scale", default_value = "6")]
    pub palette_scale: u32,
}

#[derive(Debug, Clone)]
pub struct ParsedCli {
    pub input: String,
    pub input_path: Option<PathBuf>,
    pub output_path: PathBuf,
    pub raw_output_path: Option<PathBuf>,
    pub is_url: bool,
    pub implicit_url_output: bool,
    pub jobs: Option<usize>,
    pub network: NetworkOptions,
    pub transform: TransformOptions,
}

#[derive(Debug, Clone)]
pub struct NetworkOptions {
    pub timeout_ms: u64,
    pub max_bytes: u64,
    pub content_types: Vec<String>,
}

#[derive(Debug)]
pub enum CliExecutionResult {
    File(PathBuf),
    Batch {
        outputs: Vec<PathBuf>,
        job_count: usize,
    },
}

pub fn run_from_env() -> Result<()> {
    let args = CliArgs::parse();
    match execute(args)? {
        CliExecutionResult::File(path) => {
            println!("{}", path.display());
        }
        CliExecutionResult::Batch { outputs, .. } => {
            for output in outputs {
                println!("{}", output.display());
            }
        }
    }
    Ok(())
}

pub fn execute(args: CliArgs) -> Result<CliExecutionResult> {
    let parsed = parse_cli(args)?;
    let threads = threading::configure_global_pool(parsed.jobs);
    if let Some(input_path) = &parsed.input_path
        && input_path.is_dir()
    {
        let output_dir =
            resolve_batch_output_directory(input_path, parsed.raw_output_path.as_deref())?;
        if same_path(input_path, &output_dir)? {
            bail!("Batch output directory must be different from input directory");
        }
        let inputs = collect_batch_inputs(input_path, &output_dir)?;
        if inputs.is_empty() {
            bail!(
                "No supported image files found in input directory: {}",
                input_path.display()
            );
        }
        let mut outputs = inputs
            .par_iter()
            .map(|input| {
                let output = resolve_batch_file_output_path(
                    input_path,
                    input,
                    &output_dir,
                    parsed.transform.format,
                );
                let options = create_batch_transform_options(&parsed.transform, input_path, input);
                run_one_file(input, &output, &options, &parsed.network)?;
                Ok(output)
            })
            .collect::<Result<Vec<_>>>()?;
        outputs.sort();
        return Ok(CliExecutionResult::Batch {
            outputs,
            job_count: parsed.jobs.unwrap_or(threads).min(inputs.len()).max(1),
        });
    }

    if parsed.is_url && parsed.implicit_url_output && parsed.output_path.exists() {
        bail!(
            "Output file already exists: {}. Use -o to specify an output name.",
            parsed.output_path.display()
        );
    }

    let bytes = load_input(&parsed.input, &parsed.network)?;
    let result = transform_bytes(&bytes, &parsed.transform)?;
    write_image_file(
        &result.output,
        &parsed.output_path,
        parsed.transform.format,
        parsed.transform.quality,
    )?;
    Ok(CliExecutionResult::File(parsed.output_path))
}

pub fn parse_cli(args: CliArgs) -> Result<ParsedCli> {
    if args.input_path_or_url.is_some() && args.input.is_some() {
        bail!("Use either a positional input or --input, not both");
    }
    if args.output_path_or_dir.is_some() && args.output.is_some() {
        bail!("Use either a positional output path or --output, not both");
    }
    let input = args
        .input_path_or_url
        .or(args.input)
        .context("An input path is required")?;
    let is_url = is_web_url(&input);
    let input_path = (!is_url)
        .then(|| PathBuf::from(&input).canonicalize())
        .transpose()?;
    let raw_output_path = args.output.or(args.output_path_or_dir);
    let uses_implicit_output_path = is_url && raw_output_path.is_none();
    let explicit_format = args.format.as_deref().map(parse_format).transpose()?;
    let format = explicit_format
        .or_else(|| raw_output_path.as_deref().and_then(format_from_path))
        .or_else(|| is_url.then(|| format_from_url(&input)).flatten())
        .unwrap_or(OutputFormat::Png);
    let output_path = if is_url {
        resolve_url_output_path(&input, raw_output_path.as_deref(), format)?
    } else {
        resolve_file_output_path(
            input_path
                .as_deref()
                .context("missing resolved input path")?,
            raw_output_path.as_deref(),
            format,
        )
    };
    let auto_scale_target = match (args.auto_scale_width, args.auto_scale_height) {
        (Some(width), Some(height)) => Some(Size { width, height }),
        (None, None) => None,
        _ => bail!("auto-scale-width and auto-scale-height must be provided together"),
    };

    let transform = TransformOptions {
        colors: parse_colors(&args.colors)?,
        palette_merge_threshold: args.palette_merge_threshold,
        color_sample_grid_size: positive(args.color_sample_grid_size, "color-sample-grid-size")?,
        palette_strategy: parse_palette_strategy(&args.palette_strategy)?,
        scale: args
            .scale
            .map(|scale| positive(scale, "scale"))
            .transpose()?,
        auto_scale_target,
        downscale: args
            .downscale
            .as_deref()
            .map(|value| parse_size(value, "downscale"))
            .transpose()?,
        downscale_sample_from: parse_downscale_sample_from(&args.downscale_sample_from)?,
        transparent_background: args.transparent,
        crop: args.crop,
        crop_size: args
            .crop_size
            .as_deref()
            .map(|value| parse_size(value, "crop-size"))
            .transpose()?,
        pixel_width: args
            .pixel_width
            .map(|width| positive(width, "pixel-width"))
            .transpose()?,
        pixel_width_detector: parse_pixel_width_detector(&args.pixel_width_detector)?,
        initial_upscale: positive(args.initial_upscale, "initial-upscale")?,
        warp_subdivision_depth: args.warp_subdivision_depth,
        warp_subdivision_edge_threshold: args.warp_subdivision_edge_threshold,
        artifacts: ArtifactOptions {
            palette_path: args.palette_out,
            palette_scale: positive(args.palette_scale, "palette-scale")?,
            unscaled_path: args.unscaled_out,
            debug_sheet_path: args.debug_out,
            debug_scale: positive(args.debug_scale, "debug-scale")?,
        },
        format,
        quality: args.quality,
        ..TransformOptions::default()
    };

    Ok(ParsedCli {
        input,
        input_path,
        output_path,
        raw_output_path,
        is_url,
        implicit_url_output: uses_implicit_output_path,
        jobs: parse_batch_jobs(args.jobs.as_deref(), args.threads.as_deref())?,
        network: NetworkOptions {
            timeout_ms: positive_u64(args.url_timeout_ms, "url-timeout-ms")?,
            max_bytes: positive_u64(args.url_max_bytes, "url-max-bytes")?,
            content_types: parse_content_types(&args.url_content_types)?,
        },
        transform,
    })
}

fn run_one_file(
    input: &Path,
    output: &Path,
    options: &TransformOptions,
    network: &NetworkOptions,
) -> Result<()> {
    let bytes = load_input(&input.to_string_lossy(), network)?;
    let result = transform_bytes(&bytes, options)?;
    write_image_file(&result.output, output, options.format, options.quality)
}

fn create_batch_transform_options(
    base: &TransformOptions,
    input_dir: &Path,
    input_file: &Path,
) -> TransformOptions {
    let mut options = base.clone();
    options.artifacts.debug_sheet_path = resolve_batch_artifact_path(
        base.artifacts.debug_sheet_path.as_deref(),
        input_dir,
        input_file,
        "debug",
    );
    options.artifacts.unscaled_path = resolve_batch_artifact_path(
        base.artifacts.unscaled_path.as_deref(),
        input_dir,
        input_file,
        "unscaled",
    );
    options.artifacts.palette_path = resolve_batch_artifact_path(
        base.artifacts.palette_path.as_deref(),
        input_dir,
        input_file,
        "palette",
    );
    options
}

fn resolve_batch_artifact_path(
    artifact_dir: Option<&Path>,
    input_dir: &Path,
    input_file: &Path,
    suffix: &str,
) -> Option<PathBuf> {
    let artifact_dir = artifact_dir?;
    let relative = input_file.strip_prefix(input_dir).unwrap_or(input_file);
    let parent = relative
        .parent()
        .filter(|path| !path.as_os_str().is_empty());
    let stem = input_file
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("output");
    let mut path = artifact_dir.to_path_buf();
    if let Some(parent) = parent {
        path = path.join(parent);
    }
    Some(path.join(format!("{stem}-{suffix}.png")))
}

fn load_input(input: &str, network: &NetworkOptions) -> Result<Vec<u8>> {
    if is_web_url(input) {
        load_url(input, network)
    } else {
        Ok(std::fs::read(input)?)
    }
}

fn load_url(input: &str, network: &NetworkOptions) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(network.timeout_ms))
        .build()?;
    let mut response = client.get(input).send()?;
    if !response.status().is_success() {
        bail!("Failed to fetch image from {input}: {}", response.status());
    }
    if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
        let content_type = content_type
            .to_str()
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if !is_allowed_content_type(&content_type, &network.content_types) {
            bail!("Unsupported content type for {input}: {content_type}");
        }
    }
    if let Some(length) = response.content_length()
        && length > network.max_bytes
    {
        bail!(
            "Image from {input} is too large: {length} bytes exceeds {} bytes",
            network.max_bytes
        );
    }
    let mut out = Vec::new();
    response.copy_to(&mut out)?;
    if out.len() as u64 > network.max_bytes {
        bail!(
            "Image from {input} is too large: received more than {} bytes",
            network.max_bytes
        );
    }
    Ok(out)
}

fn resolve_batch_output_directory(input: &Path, raw: Option<&Path>) -> Result<PathBuf> {
    if let Some(raw) = raw {
        if raw.extension().is_some() {
            bail!("Batch output must be a directory path");
        }
        return absolutize(raw);
    }
    let name = input
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("output");
    Ok(input.with_file_name(format!("{name}_fixpix")))
}

fn same_path(left: &Path, right: &Path) -> Result<bool> {
    Ok(normalize_existing_path(left)? == normalize_existing_path(right)?)
}

fn absolutize(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn normalize_existing_path(path: &Path) -> Result<PathBuf> {
    path.canonicalize().or_else(|_| absolutize(path))
}

fn collect_batch_inputs(input: &Path, output: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    fn walk(dir: &Path, output: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if dir.starts_with(output) {
            return Ok(());
        }
        let mut entries = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, output, files)?;
            } else if is_supported_input(&path) {
                files.push(path);
            }
        }
        Ok(())
    }
    walk(input, output, &mut files)?;
    files.sort();
    Ok(files)
}

fn resolve_batch_file_output_path(
    input_dir: &Path,
    input: &Path,
    output_dir: &Path,
    format: OutputFormat,
) -> PathBuf {
    let relative = input.strip_prefix(input_dir).unwrap_or(input);
    let mut output = output_dir.join(relative);
    output.set_file_name(format!(
        "{}_fixpix.{}",
        input
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("output"),
        format.extension()
    ));
    output
}

fn resolve_file_output_path(input: &Path, raw: Option<&Path>, format: OutputFormat) -> PathBuf {
    if let Some(raw) = raw {
        if raw.extension().is_some() {
            return absolutize_lossy(raw);
        }
        return absolutize_lossy(&raw.join(format!(
            "{}_fixpix.{}",
            input
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("output"),
            format.extension()
        )));
    }
    input.with_file_name(format!(
        "{}_fixpix.{}",
        input
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("output"),
        format.extension()
    ))
}

fn resolve_url_output_path(
    input: &str,
    raw: Option<&Path>,
    format: OutputFormat,
) -> Result<PathBuf> {
    let url = reqwest::Url::parse(input)?;
    let name = url
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .filter(|name| !name.is_empty())
        .unwrap_or("output");
    let stem = Path::new(name)
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("output");
    let file_name = format!("{stem}.{}", format.extension());
    Ok(match raw {
        Some(path) if path.extension().is_some() => absolutize(path)?,
        Some(path) => absolutize(&path.join(file_name))?,
        None => std::env::current_dir()?.join(file_name),
    })
}

fn absolutize_lossy(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn parse_colors(value: &str) -> Result<ColorMode> {
    match value {
        "auto" => Ok(ColorMode::Auto),
        "full" => Ok(ColorMode::Full),
        _ => {
            let parsed: i64 = value
                .parse()
                .context("colors must be an integer, 'auto', or 'full'")?;
            Ok(if parsed < 0 {
                ColorMode::Full
            } else if parsed == 0 {
                ColorMode::Auto
            } else {
                ColorMode::Fixed(parsed as usize)
            })
        }
    }
}

fn parse_format(value: &str) -> Result<OutputFormat> {
    match value {
        "png" => Ok(OutputFormat::Png),
        "jpeg" => Ok(OutputFormat::Jpeg),
        "webp" => Ok(OutputFormat::Webp),
        _ => bail!("format must be one of: png, jpeg, webp"),
    }
}

fn format_from_path(path: &Path) -> Option<OutputFormat> {
    match path
        .extension()
        .and_then(OsStr::to_str)?
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => Some(OutputFormat::Png),
        "jpg" | "jpeg" => Some(OutputFormat::Jpeg),
        "webp" => Some(OutputFormat::Webp),
        _ => None,
    }
}

fn format_from_url(input: &str) -> Option<OutputFormat> {
    reqwest::Url::parse(input)
        .ok()
        .and_then(|url| format_from_path(Path::new(url.path())))
}

fn parse_palette_strategy(value: &str) -> Result<PaletteStrategy> {
    match value {
        "global" => Ok(PaletteStrategy::Global),
        "sampled" => Ok(PaletteStrategy::Sampled),
        _ => bail!("palette-strategy must be one of: global, sampled"),
    }
}

fn parse_downscale_sample_from(value: &str) -> Result<DownscaleSampleFrom> {
    match value {
        "pixelated" => Ok(DownscaleSampleFrom::Pixelated),
        "original" => Ok(DownscaleSampleFrom::Original),
        _ => bail!("downscale-sample-from must be one of: pixelated, original"),
    }
}

fn parse_pixel_width_detector(value: &str) -> Result<PixelWidthDetector> {
    match value {
        "projection" => Ok(PixelWidthDetector::Projection),
        "hough" => Ok(PixelWidthDetector::Hough),
        "hybrid" => Ok(PixelWidthDetector::Hybrid),
        _ => bail!("pixel-width-detector must be one of: projection, hough, hybrid"),
    }
}

fn parse_size(value: &str, name: &str) -> Result<Size> {
    let mut parts = value.split('x');
    let width = parts
        .next()
        .context("missing width")?
        .parse::<u32>()
        .with_context(|| format!("{name} must be a positive integer or WxH value"))?;
    let height = parts
        .next()
        .map(str::parse::<u32>)
        .transpose()
        .with_context(|| format!("{name} must be a positive integer or WxH value"))?
        .unwrap_or(width);
    if parts.next().is_some() || width == 0 || height == 0 {
        bail!("{name} must use positive integers");
    }
    Ok(Size { width, height })
}

fn parse_batch_jobs(jobs: Option<&str>, threads: Option<&str>) -> Result<Option<usize>> {
    if jobs.is_some() && threads.is_some() && jobs != threads {
        bail!("Use either --jobs or --threads, not both");
    }
    (jobs.or(threads))
        .map(|value| parse_strict_positive_usize(value, "jobs"))
        .transpose()
}

fn parse_strict_positive_usize(value: &str, name: &str) -> Result<usize> {
    let trimmed = value.trim();
    if trimmed.is_empty() || !trimmed.chars().all(|character| character.is_ascii_digit()) {
        bail!("{name} must be a positive integer");
    }
    let parsed = trimmed
        .parse::<usize>()
        .with_context(|| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        bail!("{name} must be a positive integer");
    }
    Ok(parsed)
}

fn positive(value: u32, name: &str) -> Result<u32> {
    if value == 0 {
        bail!("{name} must be a positive integer");
    }
    Ok(value)
}

fn positive_u64(value: u64, name: &str) -> Result<u64> {
    if value == 0 {
        bail!("{name} must be a positive integer");
    }
    Ok(value)
}

fn parse_content_types(value: &str) -> Result<Vec<String>> {
    let values = value
        .split(',')
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        bail!("url-content-types must include at least one MIME type");
    }
    Ok(values)
}

fn is_allowed_content_type(content_type: &str, allowed: &[String]) -> bool {
    allowed.iter().any(|allowed| {
        allowed == "*/*"
            || allowed == content_type
            || allowed
                .strip_suffix("/*")
                .is_some_and(|prefix| content_type.starts_with(prefix))
    })
}

fn is_web_url(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

fn is_supported_input(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(OsStr::to_str)
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args(input: &str) -> CliArgs {
        CliArgs::parse_from(["fixpix", input])
    }

    #[test]
    fn parses_default_colors_as_auto() {
        let args = base_args("tests/sources/fish.png");
        let parsed = parse_cli(args).unwrap();
        assert_eq!(parsed.transform.colors, ColorMode::Auto);
    }

    #[test]
    fn parses_file_output_and_artifact_options() {
        let args = CliArgs::parse_from([
            "fixpix",
            "tests/sources/dragon_coffee_2.png",
            "--output",
            "out",
            "--colors",
            "auto",
            "--auto-scale-width",
            "1024",
            "--auto-scale-height",
            "1024",
            "--palette-merge-threshold",
            "0",
            "--color-sample-grid-size",
            "5",
            "--palette-strategy",
            "sampled",
            "--downscale",
            "32x32",
            "--downscale-sample-from",
            "original",
            "--crop",
            "--crop-size",
            "32x128",
            "--warp-subdivision-depth",
            "2",
            "--warp-subdivision-edge-threshold",
            "24",
            "--debug-out",
            "debug.png",
            "--unscaled-out",
            "unscaled.png",
            "--palette-out",
            "palette.png",
        ]);
        let parsed = parse_cli(args).unwrap();

        assert!(parsed.output_path.ends_with("dragon_coffee_2_fixpix.png"));
        assert_eq!(parsed.transform.colors, ColorMode::Auto);
        assert_eq!(
            parsed.transform.auto_scale_target,
            Some(Size {
                width: 1024,
                height: 1024
            })
        );
        assert_eq!(parsed.transform.palette_merge_threshold, 0.0);
        assert_eq!(parsed.transform.color_sample_grid_size, 5);
        assert_eq!(parsed.transform.palette_strategy, PaletteStrategy::Sampled);
        assert_eq!(
            parsed.transform.downscale,
            Some(Size {
                width: 32,
                height: 32
            })
        );
        assert_eq!(
            parsed.transform.downscale_sample_from,
            DownscaleSampleFrom::Original
        );
        assert!(parsed.transform.crop);
        assert_eq!(
            parsed.transform.crop_size,
            Some(Size {
                width: 32,
                height: 128
            })
        );
        assert_eq!(parsed.transform.warp_subdivision_depth, 2);
        assert_eq!(parsed.transform.warp_subdivision_edge_threshold, 24.0);
        assert_eq!(
            parsed.transform.artifacts.debug_sheet_path,
            Some(PathBuf::from("debug.png"))
        );
        assert_eq!(
            parsed.transform.artifacts.unscaled_path,
            Some(PathBuf::from("unscaled.png"))
        );
        assert_eq!(
            parsed.transform.artifacts.palette_path,
            Some(PathBuf::from("palette.png"))
        );
    }

    #[test]
    fn requires_both_auto_scale_dimensions_together() {
        let width_only = CliArgs::parse_from([
            "fixpix",
            "tests/sources/fish.png",
            "--auto-scale-width",
            "1024",
        ]);
        assert!(
            parse_cli(width_only)
                .unwrap_err()
                .to_string()
                .contains("auto-scale")
        );

        let height_only = CliArgs::parse_from([
            "fixpix",
            "tests/sources/fish.png",
            "--auto-scale-height",
            "1024",
        ]);
        assert!(
            parse_cli(height_only)
                .unwrap_err()
                .to_string()
                .contains("auto-scale")
        );
    }

    #[test]
    fn maps_numeric_color_shortcuts() {
        let full = CliArgs::parse_from(["fixpix", "tests/sources/fish.png", "--colors", "-1"]);
        assert_eq!(parse_cli(full).unwrap().transform.colors, ColorMode::Full);

        let auto = CliArgs::parse_from(["fixpix", "tests/sources/fish.png", "--colors", "0"]);
        assert_eq!(parse_cli(auto).unwrap().transform.colors, ColorMode::Auto);

        let fixed = CliArgs::parse_from(["fixpix", "tests/sources/fish.png", "--colors", "64"]);
        assert_eq!(
            parse_cli(fixed).unwrap().transform.colors,
            ColorMode::Fixed(64)
        );
    }

    #[test]
    fn parses_square_size() {
        let args = CliArgs::parse_from(["fixpix", "tests/sources/fish.png", "--downscale", "32"]);
        let parsed = parse_cli(args).unwrap();
        assert_eq!(
            parsed.transform.downscale,
            Some(Size {
                width: 32,
                height: 32
            })
        );
    }

    #[test]
    fn rejects_invalid_sizes_and_enums() {
        for args in [
            vec!["fixpix", "tests/sources/fish.png", "--crop-size", "32x"],
            vec!["fixpix", "tests/sources/fish.png", "--crop-size", "0x32"],
            vec!["fixpix", "tests/sources/fish.png", "--downscale", "32x"],
            vec!["fixpix", "tests/sources/fish.png", "--downscale", "0x32"],
            vec![
                "fixpix",
                "tests/sources/fish.png",
                "--palette-strategy",
                "weird",
            ],
            vec![
                "fixpix",
                "tests/sources/fish.png",
                "--downscale-sample-from",
                "source",
            ],
            vec![
                "fixpix",
                "tests/sources/fish.png",
                "--pixel-width-detector",
                "weird",
            ],
        ] {
            let parsed = CliArgs::parse_from(args);
            assert!(parse_cli(parsed).is_err());
        }
    }

    #[test]
    fn rejects_partial_numeric_values() {
        for args in [
            vec!["fixpix", "tests/sources/fish.png", "--colors", "32abc"],
            vec!["fixpix", "tests/sources/fish.png", "--colors", "1.5"],
            vec!["fixpix", "tests/sources/fish.png", "--colors", "none"],
            vec!["fixpix", "tests/sources/fish.png", "--jobs", "2.5"],
        ] {
            let parsed = CliArgs::parse_from(args);
            assert!(parse_cli(parsed).is_err());
        }

        for args in [
            vec!["fixpix", "tests/sources/fish.png", "--scale", "1.5"],
            vec!["fixpix", "tests/sources/fish.png", "--quality", "90abc"],
        ] {
            assert!(CliArgs::try_parse_from(args).is_err());
        }
    }

    #[test]
    fn rejects_input_output_conflicts_and_missing_input() {
        let missing = CliArgs::parse_from(["fixpix"]);
        assert!(
            parse_cli(missing)
                .unwrap_err()
                .to_string()
                .contains("input")
        );

        let input_conflict = CliArgs::parse_from([
            "fixpix",
            "tests/sources/fish.png",
            "--input",
            "tests/sources/fish_2.png",
        ]);
        assert!(
            parse_cli(input_conflict)
                .unwrap_err()
                .to_string()
                .contains("positional input")
        );

        let output_conflict = CliArgs::parse_from([
            "fixpix",
            "tests/sources/fish.png",
            "out.png",
            "--output",
            "other.png",
        ]);
        assert!(
            parse_cli(output_conflict)
                .unwrap_err()
                .to_string()
                .contains("output")
        );

        assert!(
            CliArgs::try_parse_from(["fixpix", "tests/sources/fish.png", "out.png", "extra"])
                .is_err()
        );
    }

    #[test]
    fn resolves_default_output_name() {
        let args = base_args("tests/sources/fish.png");
        let parsed = parse_cli(args).unwrap();
        assert!(parsed.output_path.ends_with("fish_fixpix.png"));
    }

    #[test]
    fn rejects_jobs_threads_mismatch_and_zero() {
        let mismatch = CliArgs::parse_from([
            "fixpix",
            "tests/sources/fish.png",
            "--jobs",
            "4",
            "--threads",
            "3",
        ]);
        assert!(
            parse_cli(mismatch)
                .unwrap_err()
                .to_string()
                .contains("jobs")
        );

        let zero = CliArgs::parse_from(["fixpix", "tests/sources/fish.png", "--jobs", "0"]);
        assert!(
            parse_cli(zero)
                .unwrap_err()
                .to_string()
                .contains("positive integer")
        );

        let same = CliArgs::parse_from([
            "fixpix",
            "tests/sources/fish.png",
            "--jobs",
            "2",
            "--threads",
            "2",
        ]);
        assert_eq!(parse_cli(same).unwrap().jobs, Some(2));
    }

    #[test]
    fn url_implicit_output_is_only_based_on_missing_output_path() {
        let implicit = CliArgs::parse_from([
            "fixpix",
            "https://example.com/images/fish.png?cache=1",
            "--debug-out",
            "debug.png",
        ]);
        assert!(parse_cli(implicit).unwrap().implicit_url_output);

        let explicit = CliArgs::parse_from([
            "fixpix",
            "https://example.com/images/fish.png?cache=1",
            "--output",
            "out.png",
        ]);
        assert!(!parse_cli(explicit).unwrap().implicit_url_output);
    }

    #[test]
    fn parses_url_guardrail_options_and_url_output_names() {
        let parsed = parse_cli(CliArgs::parse_from([
            "fixpix",
            "https://example.com/images/dragon_coffee_2.png?token=abc",
            "--url-timeout-ms",
            "5000",
            "--url-max-bytes",
            "1048576",
            "--url-content-types",
            "image/png, application/octet-stream",
        ]))
        .unwrap();

        assert!(parsed.is_url);
        assert!(parsed.output_path.ends_with("dragon_coffee_2.png"));
        assert_eq!(parsed.network.timeout_ms, 5000);
        assert_eq!(parsed.network.max_bytes, 1_048_576);
        assert_eq!(
            parsed.network.content_types,
            vec!["image/png", "application/octet-stream"]
        );

        let explicit_dir = parse_cli(CliArgs::parse_from([
            "fixpix",
            "https://example.com/images/dragon_coffee_2.png?token=abc",
            "--output",
            "out-dir",
        ]))
        .unwrap();
        assert!(
            explicit_dir
                .output_path
                .ends_with("out-dir/dragon_coffee_2.png")
        );
    }

    #[test]
    fn rejects_jpg_format_flag_for_ts_cli_parity() {
        let args = CliArgs::parse_from(["fixpix", "tests/sources/fish.png", "--format", "jpg"]);
        assert!(parse_cli(args).unwrap_err().to_string().contains("format"));
    }

    #[test]
    fn expands_batch_artifact_paths_per_input() {
        let input_dir = Path::new("fixtures");
        let input_file = Path::new("fixtures/nested/fish.png");
        let mut options = TransformOptions::default();
        options.artifacts.debug_sheet_path = Some(PathBuf::from("debug-artifacts"));
        options.artifacts.unscaled_path = Some(PathBuf::from("unscaled-artifacts"));
        options.artifacts.palette_path = Some(PathBuf::from("palette-artifacts"));

        let expanded = create_batch_transform_options(&options, input_dir, input_file);

        assert_eq!(
            expanded.artifacts.debug_sheet_path.unwrap(),
            PathBuf::from("debug-artifacts/nested/fish-debug.png")
        );
        assert_eq!(
            expanded.artifacts.unscaled_path.unwrap(),
            PathBuf::from("unscaled-artifacts/nested/fish-unscaled.png")
        );
        assert_eq!(
            expanded.artifacts.palette_path.unwrap(),
            PathBuf::from("palette-artifacts/nested/fish-palette.png")
        );
    }
}
