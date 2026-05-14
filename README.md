# FixPix

CPU-only native CLI for converting noisy pixel-art-style images into clean
pixel-resolution sprites.

## Usage

Basic conversion:

```bashP
fixpix ./input.png --output ./output.png
```

With the binary built from this repository:

```bash
./target/release/fixpix ./input.png --output ./output.png
```

On Windows:

```powershell
.\target\release\fixpix.exe .\input.png --output .\output.png
```


## Input And Output Rules

- Input can be a local file path, local directory, or a URL.
- Input can be positional or passed with `--input`.
- Output can be positional or passed with `--output`.
- File input defaults to `<input-name>_fixpix.<format>` beside the input.
- Directory input is processed recursively for `.png`, `.jpg`, `.jpeg`, and
  `.webp` files.
- Directory input defaults to a sibling `<input-directory>_fixpix/`
  output directory.
- URL input defaults to `<url-file-name>.<format>` in the current directory.
- If a file output path has no extension, it is treated as an output directory.

## Examples

Convert one image with defaults:

```bash
fixpix ./tests/sources/dragon_coffee_2.png
```

Limit the color palette to a max of 8 colors:

```bash
fixpix ./tests/sources/dragon_coffee_2.png --colors 8
```

Write debug artifacts:

```bash
fixpix ./tests/sources/dragon_coffee_2.png \
  --output ./art/dragon.png \
  --debug-out ./art/dragon-debug.png \
  --unscaled-out ./art/dragon-unscaled.png \
  --palette-out ./art/dragon-palette.png \
  --palette-scale 8
```

Downscale a high-resolution image into a 32x32 sprite:

```bash
fixpix ./tests/sources/armor2.png ./art/armor2.png \
  --downscale 32x32 \
  --colors auto \
  --scale 16
```

Batch convert a directory using eight CPU workers:

```bash
fixpix ./tests/sources ./art/restored --jobs 8
```

Convert an image from a URL:

```bash
fixpix "https://example.com/sprites/pixel_art.png" --output ./sprite.png
```

Use Hough-only pixel-width detection:

```bash
fixpix ./input.png --pixel-width-detector hough
```

Force a known source pixel width:

```bash
fixpix ./input.png --pixel-width 8
```

## CLI Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `<input-path-or-url>` | path or URL | required unless `--input` is used | Input image path, input directory, or `http://` / `https://` URL. |
| `[output-path-or-dir]` | path | derived automatically | Optional positional output file or directory. |
| `-i, --input <value>` | path or URL | none | Alternative to positional input. Do not use both forms at the same time. |
| `-o, --output <path>` | path | derived automatically | Output file path or output directory. |
| `-j, --jobs <n>` | positive integer | available CPU count | CPU worker count for batch work and the global Rayon pool. |
| `--threads <n>` | positive integer | available CPU count | Alias for `--jobs`. |
| `-c, --colors <integer\|auto\|full>` | integer, `auto`, or `full` | `auto` | Palette mode. Use `auto` or `0` to estimate a palette size, a positive integer to force that many colors, or `full` / any negative integer to skip color clustering. |
| `--palette-merge-threshold <n>` | number | `1` | Merge threshold used by auto palette selection. `0` keeps more exact color distinctions. |
| `--color-sample-grid-size <n>` | positive integer | `5` | Cell sampling control. `1` samples only the center; values above `1` use dominant cell color sampling. |
| `--palette-strategy <global\|sampled>` | enum | `global` | `global` builds the palette from image-wide color stats. `sampled` builds it from sampled cell colors. |
| `-s, --scale <n>` | positive integer | automatic | Final integer output scale. Overrides automatic scale selection. |
| `--auto-scale-width <n>` | positive integer | none | Target width for automatic output scaling. Must be used with `--auto-scale-height`. |
| `--auto-scale-height <n>` | positive integer | none | Target height for automatic output scaling. Must be used with `--auto-scale-width`. |
| `--downscale <n\|WxH>` | positive integer or size | none | Removes detected boundary background, crops transparent padding, and fits the source into the requested size before pixel processing. A single number means square size. |
| `--downscale-sample-from <pixelated\|original>` | enum | `pixelated` | `pixelated` keeps the resized downscale result. `original` samples dominant colors from the cleaned original image. |
| `-t, --transparent` | flag | `false` | Masks detected boundary-connected background before sampling and removes adjacent sampled background fringes. |
| `--crop` | flag | `false` | Crops transparent padding from the final unscaled sprite. |
| `--crop-size <n\|WxH>` | positive integer or size | none | Crops transparent padding and centers content in the requested canvas size. A single number means square size. |
| `-w, --pixel-width <n>` | positive integer | automatic detection | Forces a known source pixel width. |
| `--pixel-width-detector <projection\|hough\|hybrid>` | enum | `hybrid` | Selects automatic pixel-width detection strategy. |
| `-u, --initial-upscale <n>` | positive integer | `2` | Upscale factor used before mesh detection. Higher values can help small inputs but cost more CPU time. |
| `-f, --format <png\|jpeg\|webp>` | enum | inferred or `png` | Output format. If omitted, inferred from output path, URL extension, or defaults to PNG. |
| `-q, --quality <1-100>` | integer | encoder default | JPEG quality. PNG ignores this option. WebP quality is currently rejected because WebP output uses the lossless encoder. |
| `--url-timeout-ms <n>` | positive integer | `30000` | Timeout for URL input downloads, in milliseconds. |
| `--url-max-bytes <n>` | positive integer | `52428800` | Maximum URL input size in bytes. Checked against `Content-Length` when available and while reading the response. |
| `--url-content-types <list>` | comma-separated MIME list | `image/*,application/octet-stream` | Allowed URL response content types. Exact types and wildcards such as `image/*` are supported. |
| `--debug-out <path>` | path | none | Writes a combined debug sheet with source preview, edge preview, line overlays, grid overlay, magnified final image, natural-size unscaled image, and palette. |
| `--debug-scale <n>` | positive integer | `6` | Scale used for debug sheet previews. |
| `--unscaled-out <path>` | path | none | Writes the unscaled cleaned sprite before final output scaling. |
| `--palette-out <path>` | path | none | Writes a palette swatch image. |
| `--palette-scale <n>` | positive integer | `6` | Scale used for the palette swatch artifact. |
| `-h, --help` | flag | `false` | Prints command help. |

## Pixel-Width Detectors

- `projection`: Fast projection/autocorrelation detector. Good for clear regular
  grids.
- `hough`: Hough-style line detector with anchor-line mesh reconstruction. Useful
  for text-heavy or line-rich pixel art.
- `hybrid`: Default. Uses projection and Hough evidence together, with fallbacks
  for noisy inputs.

## Output Formats

- PNG: default output.
- JPEG: supported, with optional `--quality`.
- WebP: supported with lossless encoding. `--quality` is rejected for WebP until
  a self-contained lossy WebP encoder is selected.

## Threading

By default, the CLI uses the available CPU count. Use either `--jobs` or
`--threads` to set a fixed global CPU budget:

```bash
fixpix ./input-dir ./output-dir --jobs 16
```

The same thread pool is used for batch work and per-image CPU work.

## Release Builds

### Windows

```powershell
cargo build --release --target x86_64-pc-windows-msvc --bin fixpix
```

The repository config enables static CRT for the MSVC target.

### Linux

Native Linux release build:

```bash
cargo build --release --bin fixpix
```

For a more self-contained Linux binary, build for musl:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl --bin fixpix
```

When cross-compiling from Windows, musl may require a C toolchain such as
`x86_64-linux-musl-gcc`. Building from Linux or WSL with the musl toolchain
installed is the recommended path.

From WSL:

```bash
cargo build --release --bin fixpix
```

## Development

Run formatting, linting, and tests:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`cargo test` also regenerates the default visual inspection outputs under
`rust/output/`.

Run the benchmark smoke test:

```bash
cargo run --release --bin fixpix-bench
```

Generate visual artifacts manually:

```bash
cargo run --release --bin generate-visual-artifacts -- default,transparency
```

With no category argument, the generator reads defaults from
`tests/visual-artifacts-manifest.json`. Useful environment filters:

```bash
VISUALS_CATEGORIES=detector,palette \
VISUALS_FIXTURES=fish,tiles \
VISUALS_MAX_PROCESSES=8 \
cargo run --release --bin generate-visual-artifacts
```

On PowerShell:

```powershell
$env:VISUALS_CATEGORIES="detector,palette"
$env:VISUALS_FIXTURES="fish,tiles"
$env:VISUALS_MAX_PROCESSES="8"
cargo run --release --bin generate-visual-artifacts
```

## License

FixPix is licensed under the GNU General Public License v3.0 only. See
`LICENSE` for the full license text.

### Disclosure

AI tools where used in the process of creating this project.
This isn't however a _vibecoded_ project. Parts that include AI generated code are reviewed by me, the author of the project.
