use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use fixpix::core::{TransformOptions, transform_bytes};
use fixpix::threading;

fn main() -> Result<()> {
    let threads = std::env::var("PERF_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let configured = threading::configure_global_pool(threads);
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/sources");
    let fixtures = ["tiles.png", "smw.jpg", "dragon_cave_2.png"];
    println!("threads\tfixture\tms\toutput");
    for fixture in fixtures {
        let path = fixture_dir.join(fixture);
        let bytes =
            std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let started = Instant::now();
        let result = transform_bytes(&bytes, &TransformOptions::default())?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{configured}\t{fixture}\t{elapsed:.1}\t{}x{}, pixelWidth={}, colors={}",
            result.unscaled.width,
            result.unscaled.height,
            result.detected_pixel_width,
            result
                .resolved_colors
                .map(|value| value.to_string())
                .unwrap_or_else(|| "full".to_string())
        );
    }
    Ok(())
}
