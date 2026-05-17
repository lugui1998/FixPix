use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::thread;

use clap::Parser;
use fixpix::cli::{CliArgs, CliExecutionResult, execute};
use fixpix::core::{TransformOptions, transform_bytes};
use fixpix::threading;

#[test]
fn transforms_existing_fish_fixture() {
    threading::configure_global_pool(Some(2));
    let path = fixture_path("fish.png");
    let bytes = std::fs::read(path).unwrap();
    let result = transform_bytes(&bytes, &TransformOptions::default()).unwrap();
    assert!(result.output.width > 0);
    assert!(result.output.height > 0);
    assert!(result.detected_pixel_width >= 1);
}

#[test]
fn batch_execution_expands_outputs_and_artifacts() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    let input_dir = temp.path().join("input");
    let nested_dir = input_dir.join("nested");
    let output_dir = temp.path().join("output");
    let debug_dir = temp.path().join("debug");
    let unscaled_dir = temp.path().join("unscaled");
    let palette_dir = temp.path().join("palette");
    std::fs::create_dir_all(&nested_dir).unwrap();
    std::fs::copy(fixture_path("fish.png"), input_dir.join("fish.png")).unwrap();
    std::fs::copy(fixture_path("fish_2.png"), nested_dir.join("fish_2.png")).unwrap();

    let args = CliArgs::parse_from([
        "fixpix",
        input_dir.to_str().unwrap(),
        output_dir.to_str().unwrap(),
        "--jobs",
        "1",
        "--debug-out",
        debug_dir.to_str().unwrap(),
        "--unscaled-out",
        unscaled_dir.to_str().unwrap(),
        "--palette-out",
        palette_dir.to_str().unwrap(),
    ]);
    let result = execute(args).unwrap();
    let CliExecutionResult::Batch { outputs, job_count } = result else {
        panic!("expected batch result");
    };

    assert_eq!(job_count, 1);
    assert!(outputs.contains(&output_dir.join("fish_fixpix.png")));
    assert!(outputs.contains(&output_dir.join("nested/fish_2_fixpix.png")));
    assert!(debug_dir.join("fish-debug.png").exists());
    assert!(debug_dir.join("nested/fish_2-debug.png").exists());
    assert!(unscaled_dir.join("fish-unscaled.png").exists());
    assert!(palette_dir.join("nested/fish_2-palette.png").exists());
}

#[test]
fn single_file_execution_writes_output_and_debug_artifact() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("result.png");
    let debug = temp.path().join("debug.png");
    let unscaled = temp.path().join("unscaled.png");

    let args = CliArgs::parse_from([
        "fixpix",
        fixture_path("dragon_coffee_2.png").to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--debug-out",
        debug.to_str().unwrap(),
        "--unscaled-out",
        unscaled.to_str().unwrap(),
        "--colors",
        "auto",
    ]);
    let result = execute(args).unwrap();
    let CliExecutionResult::File(path) = result else {
        panic!("expected file result");
    };

    assert_eq!(path, output);
    assert!(output.exists());
    assert!(debug.exists());
    assert!(unscaled.exists());
}

#[test]
fn metadata_flag_prints_transform_metadata_as_json() {
    let temp = tempfile::tempdir().unwrap();
    let input = fixture_path("fish.png");
    let output_path = temp.path().join("fish.png");

    let output = Command::new(env!("CARGO_BIN_EXE_fixpix"))
        .arg(&input)
        .arg("--output")
        .arg(&output_path)
        .arg("--metadata")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists());

    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let input_string = input.display().to_string();
    let output_string = output_path.display().to_string();
    assert_eq!(metadata["input"].as_str(), Some(input_string.as_str()));
    assert_eq!(
        metadata["output_path"].as_str(),
        Some(output_string.as_str())
    );
    assert!(metadata["output_width"].as_u64().unwrap() > 0);
    assert!(metadata["output_height"].as_u64().unwrap() > 0);
    assert!(metadata["detected_pixel_width"].as_u64().unwrap() >= 1);
    assert!(metadata["palette_color_count"].as_u64().unwrap() > 0);
    assert!(metadata["pixel_width_source"].as_str().is_some());
}

#[test]
fn url_execution_downloads_image_and_writes_output() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("downloaded.png");
    let image = std::fs::read(fixture_path("dragon_coffee_2.png")).unwrap();
    let url = serve_once(200, "image/png", &image, None);

    let args = CliArgs::parse_from(["fixpix", &url, "--output", output.to_str().unwrap()]);
    let result = execute(args).unwrap();
    let CliExecutionResult::File(path) = result else {
        panic!("expected file result");
    };

    assert_eq!(path, output);
    assert!(output.exists());
}

#[test]
fn url_default_output_refuses_to_overwrite_existing_file() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("dragon_coffee_2.png"), b"taken").unwrap();
    let url = "http://127.0.0.1:9/dragon_coffee_2.png";

    let output = Command::new(env!("CARGO_BIN_EXE_fixpix"))
        .arg(url)
        .current_dir(temp.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success());
    assert!(stderr.contains("Use -o"));
}

#[test]
fn url_guardrails_reject_content_type_and_size() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("downloaded.png");
    let image = std::fs::read(fixture_path("dragon_coffee_2.png")).unwrap();

    let bad_type_url = serve_once(200, "text/plain", &image, None);
    let bad_type = execute(CliArgs::parse_from([
        "fixpix",
        &bad_type_url,
        "--output",
        output.to_str().unwrap(),
    ]))
    .unwrap_err()
    .to_string();
    assert!(bad_type.contains("Unsupported content type"));

    let large_url = serve_once(200, "image/png", &image, Some(image.len() as u64 + 10));
    let too_large = execute(CliArgs::parse_from([
        "fixpix",
        &large_url,
        "--output",
        output.to_str().unwrap(),
        "--url-max-bytes",
        "4",
    ]))
    .unwrap_err()
    .to_string();
    assert!(too_large.contains("too large"));
}

#[test]
fn webp_file_output_is_written_with_lossless_encoder() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("result.webp");

    let args = CliArgs::parse_from([
        "fixpix",
        fixture_path("fish.png").to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--format",
        "webp",
    ]);
    execute(args).unwrap();

    assert!(output.exists());
    let decoded = image::load_from_memory_with_format(
        &std::fs::read(output).unwrap(),
        image::ImageFormat::WebP,
    )
    .unwrap();
    assert!(decoded.width() > 0);
}

#[test]
fn batch_rejects_same_input_and_output_directory() {
    threading::configure_global_pool(Some(2));
    let temp = tempfile::tempdir().unwrap();
    let input_dir = temp.path().join("input");
    std::fs::create_dir_all(&input_dir).unwrap();
    std::fs::copy(fixture_path("fish.png"), input_dir.join("fish.png")).unwrap();

    let error = execute(CliArgs::parse_from([
        "fixpix",
        input_dir.to_str().unwrap(),
        input_dir.to_str().unwrap(),
    ]))
    .unwrap_err()
    .to_string();

    assert!(error.contains("different from input directory"));
}

fn serve_once(status: u16, content_type: &str, body: &[u8], content_length: Option<u64>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let content_type = content_type.to_string();
    let body = body.to_vec();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buffer = [0u8; 1024];
        let _ = stream.read(&mut request_buffer);
        let reason = if status == 200 { "OK" } else { "ERROR" };
        let content_length = content_length.unwrap_or(body.len() as u64);
        write!(
            stream,
            "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        stream.write_all(&body).unwrap();
    });
    format!("http://{address}/dragon_coffee_2.png?cache=1")
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/sources")
        .join(name)
}
