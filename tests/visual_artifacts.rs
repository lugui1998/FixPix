use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

const GENERATED_FILES_PER_RENDER: usize = 3;
const SOURCE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];
const TEXT_CLARITY_FIXTURES: &[&str] = &["dragon_tavern_1", "dragon_tavern_2", "dragon_tavern_3"];

#[test]
fn cargo_test_generates_default_visual_artifacts() {
    let output = Command::new(env!("CARGO_BIN_EXE_generate-visual-artifacts"))
        .current_dir(manifest_dir())
        .env_remove("VISUALS_CATEGORIES")
        .env_remove("VISUALS_FIXTURES")
        .env_remove("VISUALS_MAX_PROCESSES")
        .output()
        .unwrap();

    if !output.status.success() {
        panic!(
            "visual artifact generation failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let manifest = read_manifest();
    let source_count = source_fixture_count();
    for category_name in default_categories(&manifest) {
        let category = category_by_name(&manifest, &category_name);
        let output_dir = category
            .get("outputDir")
            .and_then(Value::as_str)
            .expect("category outputDir must be a string");
        let scenario_count = category
            .get("scenarios")
            .and_then(Value::as_array)
            .expect("category scenarios must be an array")
            .len();
        let fixture_count = fixture_count_for_category(category, source_count);
        let expected = fixture_count * scenario_count * GENERATED_FILES_PER_RENDER;
        let actual = wait_for_file_count(manifest_dir().join("output").join(output_dir), expected);

        assert_eq!(
            actual, expected,
            "{category_name} should generate every scaled, unscaled, and debug artifact"
        );
    }

    assert_text_clarity_artifacts_exist();
}

fn read_manifest() -> Value {
    let path = manifest_dir()
        .join("tests/visual-artifacts-manifest.json")
        .canonicalize()
        .unwrap();
    let content = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&content).unwrap()
}

fn default_categories(manifest: &Value) -> Vec<String> {
    manifest
        .get("defaultCategories")
        .and_then(Value::as_array)
        .expect("manifest defaultCategories must be an array")
        .iter()
        .map(|value| {
            value
                .as_str()
                .expect("default category must be a string")
                .to_string()
        })
        .collect()
}

fn category_by_name<'a>(manifest: &'a Value, name: &str) -> &'a Value {
    manifest
        .get("categories")
        .and_then(Value::as_array)
        .expect("manifest categories must be an array")
        .iter()
        .find(|category| category.get("name").and_then(Value::as_str) == Some(name))
        .unwrap_or_else(|| panic!("missing visual category: {name}"))
}

fn fixture_count_for_category(category: &Value, source_count: usize) -> usize {
    match category
        .get("fixtures")
        .expect("category fixtures must exist")
    {
        Value::String(value) if value == "all" => source_count,
        Value::Array(files) => files.len(),
        value => panic!("unsupported fixture selector: {value:?}"),
    }
}

fn source_fixture_count() -> usize {
    let source_dir = manifest_dir().join("tests/sources");
    std::fs::read_dir(source_dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file() && is_source_image(&entry.path()))
        .count()
}

fn is_source_image(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|extension| SOURCE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn file_count(path: PathBuf) -> usize {
    std::fs::read_dir(path)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .count()
}

fn wait_for_file_count(path: PathBuf, expected: usize) -> usize {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut actual = file_count(path.clone());
    while actual != expected && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(50));
        actual = file_count(path.clone());
    }
    actual
}

fn assert_text_clarity_artifacts_exist() {
    let output_dir = manifest_dir().join("output").join("text-clarity");
    for fixture in TEXT_CLARITY_FIXTURES {
        for suffix in ["scaled", "unscaled", "debug"] {
            let path = output_dir.join(format!("{fixture}-default-{suffix}.png"));
            assert!(
                path.exists(),
                "text clarity fixture should generate {}",
                path.display()
            );
        }
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
