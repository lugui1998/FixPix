use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::Deserialize;

use fixpix::core::{
    ArtifactOptions, ColorMode, DownscaleSampleFrom, OutputFormat, PaletteStrategy,
    PixelWidthDetector, Size, TransformOptions, transform_bytes, write_image_file,
};
use fixpix::threading;

const SOURCE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];
const DEFAULT_DEBUG_SCALE: u32 = 6;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    default_categories: Vec<String>,
    categories: Vec<Category>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Category {
    #[serde(default)]
    aliases: Vec<String>,
    fixtures: FixtureSelection,
    name: String,
    output_dir: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum FixtureSelection {
    All(String),
    Files(Vec<String>),
}

#[derive(Debug, Deserialize, Clone)]
struct Scenario {
    name: String,
    #[serde(default)]
    options: ManifestOptions,
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct ManifestOptions {
    colors: Option<ManifestColorMode>,
    palette_merge_threshold: Option<f32>,
    color_sample_grid_size: Option<u32>,
    palette_strategy: Option<String>,
    scale: Option<u32>,
    auto_scale_target: Option<ManifestSize>,
    downscale: Option<ManifestSize>,
    downscale_sample_from: Option<String>,
    transparent_background: Option<bool>,
    crop: Option<bool>,
    crop_size: Option<ManifestSize>,
    pixel_width: Option<u32>,
    pixel_width_detector: Option<String>,
    initial_upscale: Option<u32>,
    warp_subdivision_depth: Option<u32>,
    warp_subdivision_edge_threshold: Option<f32>,
    artifacts: Option<ManifestArtifacts>,
    output: Option<ManifestOutput>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum ManifestColorMode {
    Text(String),
    Number(i64),
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ManifestSize {
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct ManifestArtifacts {
    palette_scale: Option<u32>,
    debug_scale: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct ManifestOutput {
    format: Option<String>,
    quality: Option<u8>,
}

#[derive(Debug, Clone)]
struct Fixture {
    name: String,
    file: String,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct RenderJob {
    category: Category,
    fixture: Fixture,
    scenario: Scenario,
}

fn main() -> Result<()> {
    let manifest = read_manifest()?;
    let selected_categories = resolve_categories(&manifest)?;
    let fixtures = read_fixtures()?;
    let jobs = build_jobs(&selected_categories, &fixtures)?;
    if jobs.is_empty() {
        bail!("No visual artifact jobs selected");
    }

    let threads = std::env::var("VISUALS_MAX_PROCESSES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let configured_threads = threading::configure_global_pool(threads);

    clean_category_outputs(&selected_categories)?;
    println!(
        "[visuals] running {} renders into output/ across {} threads",
        jobs.len(),
        configured_threads
    );

    jobs.par_iter().try_for_each(render_job)?;
    Ok(())
}

fn read_manifest() -> Result<Manifest> {
    let path = repo_root().join("tests/visual-artifacts-manifest.json");
    let bytes = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&bytes).context("failed to parse visual artifact manifest")
}

fn read_fixtures() -> Result<Vec<Fixture>> {
    let source_dir = source_dir();
    let mut fixtures = Vec::new();
    for entry in std::fs::read_dir(&source_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || !is_source_image(&path) {
            continue;
        }
        let file = path
            .file_name()
            .and_then(OsStr::to_str)
            .context("fixture file name is not utf-8")?
            .to_string();
        let name = path
            .file_stem()
            .and_then(OsStr::to_str)
            .context("fixture stem is not utf-8")?
            .to_string();
        fixtures.push(Fixture { name, file, path });
    }
    fixtures.sort_by(|left, right| left.file.cmp(&right.file));
    Ok(fixtures)
}

fn resolve_categories(manifest: &Manifest) -> Result<Vec<Category>> {
    let requested = requested_categories(manifest);
    let mut selected = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let all = requested.iter().any(|name| name == "all");
    let names = if all {
        manifest
            .categories
            .iter()
            .map(|category| category.name.clone())
            .collect::<Vec<_>>()
    } else {
        requested
    };

    for name in names {
        let category = manifest
            .categories
            .iter()
            .find(|category| {
                category.name == name || category.aliases.iter().any(|alias| alias == &name)
            })
            .with_context(|| format!("Unknown visual category: {name}"))?;
        if seen.insert(category.name.clone()) {
            selected.push(category.clone());
        }
    }
    Ok(selected)
}

fn requested_categories(manifest: &Manifest) -> Vec<String> {
    let args = split_list(std::env::args().skip(1));
    if !args.is_empty() {
        return args;
    }
    let env = std::env::var("VISUALS_CATEGORIES")
        .ok()
        .map(|value| split_list([value]))
        .unwrap_or_default();
    if !env.is_empty() {
        return env;
    }
    manifest.default_categories.clone()
}

fn build_jobs(categories: &[Category], fixtures: &[Fixture]) -> Result<Vec<RenderJob>> {
    let fixture_filter = std::env::var("VISUALS_FIXTURES")
        .ok()
        .map(|value| split_list([value]))
        .filter(|items| !items.is_empty() && items[0] != "all");
    let mut jobs = Vec::new();
    for category in categories {
        for fixture in fixtures_for_category(category, fixtures)? {
            if let Some(filter) = &fixture_filter
                && !filter
                    .iter()
                    .any(|name| name == &fixture.name || name == &fixture.file)
            {
                continue;
            }
            for scenario in &category.scenarios {
                jobs.push(RenderJob {
                    category: category.clone(),
                    fixture: fixture.clone(),
                    scenario: scenario.clone(),
                });
            }
        }
    }
    Ok(jobs)
}

fn fixtures_for_category(category: &Category, fixtures: &[Fixture]) -> Result<Vec<Fixture>> {
    match &category.fixtures {
        FixtureSelection::All(value) if value == "all" => Ok(fixtures.to_vec()),
        FixtureSelection::All(value) => bail!("unsupported fixture selector: {value}"),
        FixtureSelection::Files(files) => files
            .iter()
            .map(|file| {
                fixtures
                    .iter()
                    .find(|fixture| fixture.file == *file || fixture.name == *file)
                    .cloned()
                    .with_context(|| format!("unknown fixture in {}: {file}", category.name))
            })
            .collect(),
    }
}

fn render_job(job: &RenderJob) -> Result<()> {
    let input = std::fs::read(&job.fixture.path)
        .with_context(|| format!("failed to read {}", job.fixture.path.display()))?;
    let output_dir = output_dir().join(&job.category.output_dir);
    let base_name = format!("{}-{}", job.fixture.name, job.scenario.name);
    let scaled_path = output_dir.join(format!("{base_name}-scaled.png"));
    let unscaled_path = output_dir.join(format!("{base_name}-unscaled.png"));
    let debug_path = output_dir.join(format!("{base_name}-debug.png"));
    let mut options = manifest_options_to_transform(&job.scenario.options)?;
    options.format = OutputFormat::Png;
    options.artifacts.unscaled_path = Some(unscaled_path);
    options.artifacts.debug_sheet_path = Some(debug_path);

    let result = transform_bytes(&input, &options)?;
    write_image_file(&result.output, &scaled_path, OutputFormat::Png, None)?;
    println!(
        "[visuals] {}/{}/{} -> {}",
        job.category.name,
        job.fixture.name,
        job.scenario.name,
        scaled_path.display()
    );
    Ok(())
}

fn manifest_options_to_transform(options: &ManifestOptions) -> Result<TransformOptions> {
    let mut transform = TransformOptions::default();
    if let Some(colors) = &options.colors {
        transform.colors = match colors {
            ManifestColorMode::Text(value) if value == "auto" => ColorMode::Auto,
            ManifestColorMode::Text(value) if value == "full" => ColorMode::Full,
            ManifestColorMode::Text(value) => bail!("unsupported color mode: {value}"),
            ManifestColorMode::Number(value) if *value < 0 => ColorMode::Full,
            ManifestColorMode::Number(0) => ColorMode::Auto,
            ManifestColorMode::Number(value) => ColorMode::Fixed(*value as usize),
        };
    }
    if let Some(value) = options.palette_merge_threshold {
        transform.palette_merge_threshold = value;
    }
    if let Some(value) = options.color_sample_grid_size {
        transform.color_sample_grid_size = value;
    }
    if let Some(value) = &options.palette_strategy {
        transform.palette_strategy = match value.as_str() {
            "global" => PaletteStrategy::Global,
            "sampled" => PaletteStrategy::Sampled,
            _ => bail!("unsupported palette strategy: {value}"),
        };
    }
    transform.scale = options.scale;
    transform.auto_scale_target = options.auto_scale_target.map(size);
    transform.downscale = options.downscale.map(size);
    if let Some(value) = &options.downscale_sample_from {
        transform.downscale_sample_from = match value.as_str() {
            "pixelated" => DownscaleSampleFrom::Pixelated,
            "original" => DownscaleSampleFrom::Original,
            _ => bail!("unsupported downscale sample source: {value}"),
        };
    }
    transform.transparent_background = options.transparent_background.unwrap_or(false);
    transform.crop = options.crop.unwrap_or(false);
    transform.crop_size = options.crop_size.map(size);
    transform.pixel_width = options.pixel_width;
    if let Some(value) = &options.pixel_width_detector {
        transform.pixel_width_detector = match value.as_str() {
            "projection" => PixelWidthDetector::Projection,
            "hough" => PixelWidthDetector::Hough,
            "hybrid" => PixelWidthDetector::Hybrid,
            _ => bail!("unsupported pixel width detector: {value}"),
        };
    }
    if let Some(value) = options.initial_upscale {
        transform.initial_upscale = value;
    }
    if let Some(value) = options.warp_subdivision_depth {
        transform.warp_subdivision_depth = value;
    }
    if let Some(value) = options.warp_subdivision_edge_threshold {
        transform.warp_subdivision_edge_threshold = value;
    }
    if let Some(artifacts) = &options.artifacts {
        transform.artifacts = ArtifactOptions {
            palette_scale: artifacts.palette_scale.unwrap_or(6),
            debug_scale: artifacts.debug_scale.unwrap_or(DEFAULT_DEBUG_SCALE),
            ..ArtifactOptions::default()
        };
    }
    if let Some(output) = &options.output {
        if let Some(format) = &output.format {
            transform.format = match format.as_str() {
                "png" => OutputFormat::Png,
                "jpeg" => OutputFormat::Jpeg,
                "webp" => OutputFormat::Webp,
                _ => bail!("unsupported output format: {format}"),
            };
        }
        transform.quality = output.quality;
    }
    Ok(transform)
}

fn size(value: ManifestSize) -> Size {
    Size {
        width: value.width,
        height: value.height,
    }
}

fn clean_category_outputs(categories: &[Category]) -> Result<()> {
    for category in categories {
        let path = output_dir().join(&category.output_dir);
        if path.exists() {
            clear_dir_contents_with_retry(&path)
                .with_context(|| format!("failed to clean {}", path.display()))?;
            remove_empty_dir_if_possible(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
    }
    Ok(())
}

fn clear_dir_contents_with_retry(path: &Path) -> Result<()> {
    let mut last_error = None;
    for _ in 0..8 {
        match clear_dir_contents(path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    match last_error {
        Some(error) => Err(error).with_context(|| format!("failed to clean {}", path.display())),
        None => Ok(()),
    }
}

fn clear_dir_contents(path: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            clear_dir_contents_with_retry(&entry_path)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            remove_empty_dir_if_possible(&entry_path)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
        } else {
            std::fs::remove_file(entry_path)?;
        }
    }
    Ok(())
}

fn remove_empty_dir_if_possible(path: &Path) -> Result<()> {
    let mut last_error = None;
    for _ in 0..8 {
        match std::fs::remove_dir(path) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) if is_busy_directory_error(&error) && is_empty_dir(path)? => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
    match last_error {
        Some(error) if is_busy_directory_error(&error) && is_empty_dir(path)? => Ok(()),
        Some(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
        None => Ok(()),
    }
}

fn is_empty_dir(path: &Path) -> Result<bool> {
    match std::fs::read_dir(path) {
        Ok(mut entries) => Ok(entries.next().is_none()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn is_busy_directory_error(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(32) | Some(33) | Some(145) | Some(1224)
    ) || error.kind() == std::io::ErrorKind::PermissionDenied
}

fn split_list(values: impl IntoIterator<Item = String>) -> Vec<String> {
    values
        .into_iter()
        .flat_map(|value| value.split(',').map(str::to_string).collect::<Vec<_>>())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

fn is_source_image(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|extension| SOURCE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn source_dir() -> PathBuf {
    repo_root().join("tests/sources")
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("output")
}
