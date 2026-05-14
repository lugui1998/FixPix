use std::collections::HashMap;

use rayon::prelude::*;

use crate::core::{ColorMode, PaletteStrategy};
use crate::image::{ALPHA_THRESHOLD, RawImage};

const AUTO_COLOR_COUNT_KMEANS_ITERATIONS: usize = 6;
const AUTO_BYPASS_UNIQUE_COLOR_LIMIT: usize = 128;
const CENTER_SAMPLE_SPREAD: f32 = 0.6;
const DOMINANT_BIN_SIZE: u8 = 52;
const DOMINANT_BIN_SHIFTED_OFFSET: u8 = DOMINANT_BIN_SIZE / 2;
const DOMINANT_BIN_COUNT: usize = 5;
const DOMINANT_BIN_TOTAL: usize = DOMINANT_BIN_COUNT * DOMINANT_BIN_COUNT * DOMINANT_BIN_COUNT;
const EMPTY_COLOR_KEY: i32 = -1;
const HIGHLIGHT_LIGHTNESS_MIN: f64 = 82.0;
const HIGHLIGHT_CHROMA_MAX: f64 = 24.0;
const HIGHLIGHT_CONTRAST_MIN: f64 = 2500.0;
const HIGHLIGHT_LIGHTNESS_GAP_MIN: f64 = 10.0;
const HIGHLIGHT_NEAREST_DISTANCE_MIN: f64 = 140.0;
const PROTECTED_HIGHLIGHT_CANDIDATE_LIMIT: usize = 512;
const KMEANS_ITERATIONS: usize = 20;
const KMEANS_TRAINING_POINT_LIMIT: usize = 4096;
const KMEANS_TRAINING_RGB_BIN_SHIFT: u8 = 4;

#[derive(Debug, Clone)]
pub struct PaletteResult {
    pub image: RawImage,
    pub resolved_colors: Option<usize>,
    pub palette_colors: Vec<[u8; 3]>,
}

pub fn sample_cell_color(
    image: &RawImage,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    sample_grid: u32,
) -> [u8; 4] {
    sample_cell_color_with_min_opaque_coverage(image, x0, x1, y0, y1, sample_grid, 0.5)
}

pub(crate) fn sample_cell_color_with_min_opaque_coverage(
    image: &RawImage,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    sample_grid: u32,
    min_opaque_coverage: f32,
) -> [u8; 4] {
    if x1 <= x0 || y1 <= y0 {
        return [0, 0, 0, 0];
    }
    if sample_grid <= 1 {
        return sample_cell_center_color(image, x0, x1, y0, y1, sample_grid, min_opaque_coverage);
    }
    dominant_cell_color_by_binning(image, x0, x1, y0, y1, min_opaque_coverage)
}

fn sample_cell_center_color(
    image: &RawImage,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    sample_grid: u32,
    min_opaque_coverage: f32,
) -> [u8; 4] {
    let sample_xs = centered_sample_positions(x0, x1, sample_grid);
    let sample_ys = centered_sample_positions(y0, y1, sample_grid);
    let center_x = (x0 + x1 - 1) as f32 / 2.0;
    let center_y = (y0 + y1 - 1) as f32 / 2.0;
    let mut counts = HashMap::<u32, (u32, f32)>::new();
    let mut opaque_samples = 0u32;
    let mut total_samples = 0u32;
    let mut best_key = None;
    let mut best_count = 0u32;
    let mut best_distance = f32::INFINITY;

    for sample_y in sample_ys {
        for sample_x in &sample_xs {
            total_samples += 1;
            let pixel = image.pixel(*sample_x, sample_y);
            if pixel[3] < ALPHA_THRESHOLD {
                continue;
            }
            opaque_samples += 1;
            let key = pack_rgb([pixel[0], pixel[1], pixel[2]]);
            let distance = (*sample_x as f32 - center_x).abs() + (sample_y as f32 - center_y).abs();
            let entry = counts.entry(key).or_default();
            entry.0 += 1;
            entry.1 += distance;
            if entry.0 > best_count || (entry.0 == best_count && entry.1 < best_distance) {
                best_key = Some(key);
                best_count = entry.0;
                best_distance = entry.1;
            }
        }
    }

    if !has_min_opaque_coverage(opaque_samples, total_samples, min_opaque_coverage) {
        return [0, 0, 0, 0];
    }
    best_key
        .map(unpack_rgb)
        .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
        .unwrap_or([0, 0, 0, 0])
}

fn centered_sample_positions(start: u32, end: u32, sample_grid: u32) -> Vec<u32> {
    if end <= start {
        return Vec::new();
    }
    if sample_grid <= 1 {
        return vec![((start + end - 1) as f32 / 2.0).round() as u32];
    }
    let center = (start + end - 1) as f32 / 2.0;
    let radius = (end - start - 1) as f32 / 2.0;
    let sample_radius = radius.max(0.0) * CENTER_SAMPLE_SPREAD;
    (0..sample_grid)
        .map(|index| {
            let t = index as f32 / (sample_grid - 1).max(1) as f32;
            let offset = (t - 0.5) * 2.0 * sample_radius;
            (center + offset)
                .round()
                .clamp(start as f32, (end - 1) as f32) as u32
        })
        .collect()
}

fn dominant_cell_color_by_binning(
    image: &RawImage,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    min_opaque_coverage: f32,
) -> [u8; 4] {
    let mut primary_bin_counts = [0u32; DOMINANT_BIN_TOTAL];
    let mut shifted_bin_counts = [0u32; DOMINANT_BIN_TOTAL];
    let mut opaque_count = 0u32;
    let mut total_count = 0u32;
    let mut best_grid_index = 0u8;
    let mut best_bin_index = 0usize;
    let mut best_bin_count = 0u32;
    let mut single_opaque = [0, 0, 0, 255];

    for y in y0..y1 {
        for x in x0..x1 {
            total_count += 1;
            let pixel = image.pixel(x, y);
            if pixel[3] < ALPHA_THRESHOLD {
                continue;
            }
            if opaque_count == 0 {
                single_opaque = [pixel[0], pixel[1], pixel[2], 255];
            }
            let primary = color_bin_index(pixel[0], pixel[1], pixel[2], 0);
            primary_bin_counts[primary] += 1;
            if primary_bin_counts[primary] > best_bin_count {
                best_grid_index = 0;
                best_bin_index = primary;
                best_bin_count = primary_bin_counts[primary];
            }
            let shifted =
                color_bin_index(pixel[0], pixel[1], pixel[2], DOMINANT_BIN_SHIFTED_OFFSET);
            shifted_bin_counts[shifted] += 1;
            if shifted_bin_counts[shifted] > best_bin_count {
                best_grid_index = 1;
                best_bin_index = shifted;
                best_bin_count = shifted_bin_counts[shifted];
            }
            opaque_count += 1;
        }
    }

    if !has_min_opaque_coverage(opaque_count, total_count, min_opaque_coverage) {
        return [0, 0, 0, 0];
    }
    if opaque_count == 1 {
        return single_opaque;
    }

    let mut red_histogram = [0u32; 256];
    let mut green_histogram = [0u32; 256];
    let mut blue_histogram = [0u32; 256];
    let bin_offset = if best_grid_index == 0 {
        0
    } else {
        DOMINANT_BIN_SHIFTED_OFFSET
    };
    let mut selected_bin_count = 0u32;
    for y in y0..y1 {
        for x in x0..x1 {
            let pixel = image.pixel(x, y);
            if pixel[3] < ALPHA_THRESHOLD {
                continue;
            }
            if color_bin_index(pixel[0], pixel[1], pixel[2], bin_offset) != best_bin_index {
                continue;
            }
            red_histogram[pixel[0] as usize] += 1;
            green_histogram[pixel[1] as usize] += 1;
            blue_histogram[pixel[2] as usize] += 1;
            selected_bin_count += 1;
        }
    }

    [
        median_channel_from_histogram(&red_histogram, selected_bin_count),
        median_channel_from_histogram(&green_histogram, selected_bin_count),
        median_channel_from_histogram(&blue_histogram, selected_bin_count),
        255,
    ]
}

pub(crate) fn has_min_opaque_coverage(
    opaque_count: u32,
    total_count: u32,
    min_opaque_coverage: f32,
) -> bool {
    total_count > 0 && opaque_count as f32 / total_count as f32 > min_opaque_coverage
}

fn color_bin_index(r: u8, g: u8, b: u8, offset: u8) -> usize {
    let red = (((r as u16 + offset as u16).min(255) as u8) / DOMINANT_BIN_SIZE) as usize;
    let green = (((g as u16 + offset as u16).min(255) as u8) / DOMINANT_BIN_SIZE) as usize;
    let blue = (((b as u16 + offset as u16).min(255) as u8) / DOMINANT_BIN_SIZE) as usize;
    red * DOMINANT_BIN_COUNT * DOMINANT_BIN_COUNT + green * DOMINANT_BIN_COUNT + blue
}

fn median_channel_from_histogram(histogram: &[u32; 256], value_count: u32) -> u8 {
    let target = value_count / 2;
    let mut count = 0;
    for (value, bucket) in histogram.iter().enumerate() {
        count += bucket;
        if count > target {
            return value as u8;
        }
    }
    0
}

fn pack_rgb(rgb: [u8; 3]) -> u32 {
    ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32
}

fn unpack_rgb(key: u32) -> [u8; 3] {
    [
        ((key >> 16) & 0xff) as u8,
        ((key >> 8) & 0xff) as u8,
        (key & 0xff) as u8,
    ]
}

pub fn quantize_image(
    image: &RawImage,
    colors: ColorMode,
    merge_threshold: f32,
    strategy: PaletteStrategy,
) -> PaletteResult {
    let stats = collect_color_stats(image);
    match colors {
        ColorMode::Full => PaletteResult {
            image: normalize_transparent_rgb(image),
            resolved_colors: None,
            palette_colors: stats.iter().map(|stat| stat.rgb).collect(),
        },
        ColorMode::Auto => {
            if stats.len() <= AUTO_BYPASS_UNIQUE_COLOR_LIMIT {
                return PaletteResult {
                    image: normalize_transparent_rgb(image),
                    resolved_colors: Some(stats.len()),
                    palette_colors: stats.iter().map(|stat| stat.rgb).collect(),
                };
            }
            let target = auto_color_count(&stats, merge_threshold);
            if stats.len() <= target {
                PaletteResult {
                    image: normalize_transparent_rgb(image),
                    resolved_colors: Some(stats.len()),
                    palette_colors: stats.iter().map(|stat| stat.rgb).collect(),
                }
            } else {
                apply_quantized_palette(image, &stats, target, merge_threshold, strategy)
            }
        }
        ColorMode::Fixed(target) => {
            if stats.len() <= target {
                PaletteResult {
                    image: normalize_transparent_rgb(image),
                    resolved_colors: Some(stats.len()),
                    palette_colors: stats.iter().map(|stat| stat.rgb).collect(),
                }
            } else {
                apply_quantized_palette(image, &stats, target, merge_threshold, strategy)
            }
        }
    }
}

pub fn extract_palette_colors(image: &RawImage) -> Vec<[u8; 3]> {
    collect_color_stats(image)
        .into_iter()
        .map(|stat| stat.rgb)
        .collect()
}

pub fn create_palette_image(colors: &[[u8; 3]]) -> RawImage {
    if colors.is_empty() {
        return RawImage::transparent(1, 1);
    }

    let mut best_columns = 1usize;
    let mut best_rows = colors.len();
    let mut best_score = usize::MAX;
    for columns in 1..=colors.len() {
        let rows = colors.len().div_ceil(columns);
        let empty_cells = rows * columns - colors.len();
        let score = columns.abs_diff(rows) * 10 + empty_cells;
        if score < best_score {
            best_score = score;
            best_columns = columns;
            best_rows = rows;
        }
    }

    let mut image = RawImage::transparent(best_columns as u32, best_rows as u32);
    for (index, color) in colors.iter().enumerate() {
        let x = (index % best_columns) as u32;
        let y = (index / best_columns) as u32;
        image.set_pixel(x, y, [color[0], color[1], color[2], 255]);
    }
    image
}

fn auto_color_count(stats: &[ColorStat], merge_threshold: f32) -> usize {
    if stats.len() <= 1 || merge_threshold <= 0.0 {
        return stats.len();
    }
    let exact = weighted_points_from_stats(stats);
    let merged = merge_nearby_color_points(&exact, merge_threshold as f64);
    let analysis = training_color_points(&merged);
    if analysis.len() <= 8 {
        return analysis.len();
    }
    let candidates = palette_candidates(analysis.len());
    let errors = candidates
        .iter()
        .map(|candidate| {
            compute_weighted_kmeans_error(&analysis, *candidate, AUTO_COLOR_COUNT_KMEANS_ITERATIONS)
        })
        .collect::<Vec<_>>();
    pick_knee_candidate(&candidates, &errors)
}

fn apply_quantized_palette(
    image: &RawImage,
    stats: &[ColorStat],
    target: usize,
    merge_threshold: f32,
    strategy: PaletteStrategy,
) -> PaletteResult {
    let target = target.clamp(1, 256);
    let points = weighted_points_from_stats(stats);
    let protected = select_protected_highlight_points(stats, target);
    let quantized = if strategy == PaletteStrategy::Sampled {
        quantize_sampled_color_points(&points, &protected, target, merge_threshold as f64)
    } else {
        quantize_color_points_with_protected_highlights(&points, &protected, target)
    };
    let palette_colors = quantized.palette.clone();
    let color_lookup = palette_assignment_lookup(&points, &quantized);
    let mut out = image.clone();
    out.data.par_chunks_exact_mut(4).for_each(|pixel| {
        if pixel[3] < ALPHA_THRESHOLD {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            return;
        }
        if let Some(rgb) = color_lookup.get(&pack_rgb([pixel[0], pixel[1], pixel[2]])) {
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
        }
    });
    PaletteResult {
        image: out,
        resolved_colors: Some(quantized.palette.len()),
        palette_colors,
    }
}

fn palette_assignment_lookup(
    points: &[WeightedColorPoint],
    quantized: &QuantizedPalette,
) -> HashMap<u32, [u8; 3]> {
    let mut lookup = HashMap::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        if let Some(color) = quantized
            .assignments
            .get(index)
            .and_then(|palette_index| quantized.palette.get(*palette_index))
        {
            lookup.insert(pack_rgb(point.rgb), *color);
        }
    }
    lookup
}

fn normalize_transparent_rgb(image: &RawImage) -> RawImage {
    let mut out = image.clone();
    out.data.par_chunks_exact_mut(4).for_each(|pixel| {
        if pixel[3] < ALPHA_THRESHOLD {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        }
    });
    out
}

#[derive(Debug)]
struct ColorStat {
    rgb: [u8; 3],
    count: u32,
    contrast: u64,
    lab: [f64; 3],
}

fn collect_color_stats(image: &RawImage) -> Vec<ColorStat> {
    let mut counts = HashMap::<u32, u32>::new();
    let mut contrast = HashMap::<u32, u64>::new();
    let mut pixel_keys = vec![EMPTY_COLOR_KEY; (image.width * image.height) as usize];
    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.pixel(x, y);
            if pixel[3] < ALPHA_THRESHOLD {
                continue;
            }
            let key = pack_rgb([pixel[0], pixel[1], pixel[2]]);
            pixel_keys[(y * image.width + x) as usize] = key as i32;
            *counts.entry(key).or_default() += 1;
        }
    }

    for y in 0..image.height {
        for x in 0..image.width {
            let index = y * image.width + x;
            if x + 1 < image.width {
                accumulate_packed_contrast(&pixel_keys, &mut contrast, index, index + 1);
            }
            if y + 1 < image.height {
                accumulate_packed_contrast(&pixel_keys, &mut contrast, index, index + image.width);
            }
        }
    }

    let mut stats = counts
        .into_iter()
        .map(|(key, count)| ColorStat {
            rgb: unpack_rgb(key),
            count,
            contrast: *contrast.get(&key).unwrap_or(&0),
            lab: rgb_to_lab(unpack_rgb(key)),
        })
        .collect::<Vec<_>>();
    stats.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.rgb.cmp(&right.rgb))
    });
    stats
}

fn accumulate_packed_contrast(
    pixel_keys: &[i32],
    contrast: &mut HashMap<u32, u64>,
    left_index: u32,
    right_index: u32,
) {
    let left_key = pixel_keys[left_index as usize];
    let right_key = pixel_keys[right_index as usize];
    if left_key == EMPTY_COLOR_KEY || right_key == EMPTY_COLOR_KEY || left_key == right_key {
        return;
    }
    let distance = packed_rgb_distance_sq(left_key as u32, right_key as u32) as u64;
    *contrast.entry(left_key as u32).or_default() += distance;
    *contrast.entry(right_key as u32).or_default() += distance;
}

fn packed_rgb_distance_sq(left: u32, right: u32) -> u32 {
    let dr = ((left >> 16) & 0xff) as i32 - ((right >> 16) & 0xff) as i32;
    let dg = ((left >> 8) & 0xff) as i32 - ((right >> 8) & 0xff) as i32;
    let db = (left & 0xff) as i32 - (right & 0xff) as i32;
    (dr * dr + dg * dg + db * db) as u32
}

#[derive(Debug, Clone)]
struct WeightedColorPoint {
    rgb: [u8; 3],
    lab: [f64; 3],
    weight: f64,
}

fn weighted_points_from_stats(stats: &[ColorStat]) -> Vec<WeightedColorPoint> {
    stats
        .iter()
        .map(|stat| WeightedColorPoint {
            rgb: stat.rgb,
            lab: stat.lab,
            weight: stat.count as f64,
        })
        .collect()
}

fn select_protected_highlight_points(
    stats: &[ColorStat],
    target: usize,
) -> Vec<WeightedColorPoint> {
    if target <= 1 || stats.len() <= 1 {
        return Vec::new();
    }

    let max_protected = 2usize.min(target.saturating_sub(1));
    let mut candidates = stats
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let average_contrast = point.contrast as f64 / point.count.max(1) as f64;
            (point.lab[0] >= HIGHLIGHT_LIGHTNESS_MIN
                && color_chroma(point.lab) <= HIGHLIGHT_CHROMA_MAX
                && average_contrast >= HIGHLIGHT_CONTRAST_MIN)
                .then_some((point, index, average_contrast))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.2.total_cmp(&left.2));
    candidates.truncate(PROTECTED_HIGHLIGHT_CANDIDATE_LIMIT);

    let mut scored = candidates
        .into_iter()
        .filter_map(|(point, index, average_contrast)| {
            let mut nearest_heavier_distance = f64::INFINITY;
            let mut nearest_heavier_lightness_gap = 0.0;
            for heavier in stats.iter().take(index) {
                let distance = lab_distance_sq(point.lab, heavier.lab);
                if distance < nearest_heavier_distance {
                    nearest_heavier_distance = distance;
                    nearest_heavier_lightness_gap = (point.lab[0] - heavier.lab[0]).abs();
                    if nearest_heavier_distance < HIGHLIGHT_NEAREST_DISTANCE_MIN {
                        break;
                    }
                }
            }
            (nearest_heavier_distance >= HIGHLIGHT_NEAREST_DISTANCE_MIN
                && nearest_heavier_lightness_gap >= HIGHLIGHT_LIGHTNESS_GAP_MIN)
                .then_some((point, average_contrast))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));

    let mut protected = Vec::<&ColorStat>::new();
    for (candidate, _) in scored {
        if protected.iter().any(|existing| {
            lab_distance_sq(existing.lab, candidate.lab) < HIGHLIGHT_NEAREST_DISTANCE_MIN
        }) {
            continue;
        }
        protected.push(candidate);
        if protected.len() >= max_protected {
            break;
        }
    }

    protected
        .into_iter()
        .map(|point| WeightedColorPoint {
            rgb: point.rgb,
            lab: point.lab,
            weight: point.count as f64,
        })
        .collect()
}

fn color_chroma(lab: [f64; 3]) -> f64 {
    (lab[1] * lab[1] + lab[2] * lab[2]).sqrt()
}

fn merge_nearby_color_points(
    points: &[WeightedColorPoint],
    threshold: f64,
) -> Vec<WeightedColorPoint> {
    if threshold <= 0.0 {
        return points.to_vec();
    }
    let threshold_sq = threshold * threshold;
    let mut merged = Vec::<MergedColorCluster>::new();
    let mut grid = HashMap::<(i32, i32, i32), Vec<usize>>::new();
    for point in points {
        let cell = lab_grid_cell(point.lab, threshold);
        let mut best_index = None;
        let mut best_distance = f64::INFINITY;
        for dl in -1..=1 {
            for da in -1..=1 {
                for db in -1..=1 {
                    let Some(indices) = grid.get(&(cell.0 + dl, cell.1 + da, cell.2 + db)) else {
                        continue;
                    };
                    for index in indices {
                        let distance = lab_distance_sq(point.lab, merged[*index].lab);
                        if distance < threshold_sq
                            && (distance < best_distance
                                || (distance == best_distance
                                    && best_index.is_none_or(|best| *index < best)))
                        {
                            best_index = Some(*index);
                            best_distance = distance;
                        }
                    }
                }
            }
        }

        if let Some(index) = best_index {
            let cluster = &mut merged[index];
            cluster.rgb_sum[0] += point.rgb[0] as f64 * point.weight;
            cluster.rgb_sum[1] += point.rgb[1] as f64 * point.weight;
            cluster.rgb_sum[2] += point.rgb[2] as f64 * point.weight;
            cluster.lab_sum[0] += point.lab[0] * point.weight;
            cluster.lab_sum[1] += point.lab[1] * point.weight;
            cluster.lab_sum[2] += point.lab[2] * point.weight;
            cluster.weight += point.weight;
            cluster.lab = [
                cluster.lab_sum[0] / cluster.weight,
                cluster.lab_sum[1] / cluster.weight,
                cluster.lab_sum[2] / cluster.weight,
            ];

            let next_cell = lab_grid_cell(cluster.lab, threshold);
            if next_cell != cluster.bin_key {
                remove_grid_index(&mut grid, cluster.bin_key, index);
                cluster.bin_key = next_cell;
                grid.entry(next_cell).or_default().push(index);
            }
        } else {
            let bin_key = lab_grid_cell(point.lab, threshold);
            merged.push(MergedColorCluster {
                rgb_sum: [
                    point.rgb[0] as f64 * point.weight,
                    point.rgb[1] as f64 * point.weight,
                    point.rgb[2] as f64 * point.weight,
                ],
                lab_sum: [
                    point.lab[0] * point.weight,
                    point.lab[1] * point.weight,
                    point.lab[2] * point.weight,
                ],
                lab: point.lab,
                weight: point.weight,
                bin_key,
            });
            grid.entry(bin_key).or_default().push(merged.len() - 1);
        }
    }
    let mut points = merged
        .into_iter()
        .map(|cluster| WeightedColorPoint {
            rgb: [
                (cluster.rgb_sum[0] / cluster.weight).round() as u8,
                (cluster.rgb_sum[1] / cluster.weight).round() as u8,
                (cluster.rgb_sum[2] / cluster.weight).round() as u8,
            ],
            lab: cluster.lab,
            weight: cluster.weight,
        })
        .collect::<Vec<_>>();
    sort_weighted_points(&mut points);
    points
}

struct MergedColorCluster {
    rgb_sum: [f64; 3],
    lab_sum: [f64; 3],
    lab: [f64; 3],
    weight: f64,
    bin_key: (i32, i32, i32),
}

fn lab_grid_cell(lab: [f64; 3], threshold: f64) -> (i32, i32, i32) {
    (
        (lab[0] / threshold).floor() as i32,
        (lab[1] / threshold).floor() as i32,
        (lab[2] / threshold).floor() as i32,
    )
}

fn remove_grid_index(
    grid: &mut HashMap<(i32, i32, i32), Vec<usize>>,
    key: (i32, i32, i32),
    index: usize,
) {
    let Some(indices) = grid.get_mut(&key) else {
        return;
    };
    if let Some(position) = indices.iter().position(|candidate| *candidate == index) {
        indices.remove(position);
    }
    if indices.is_empty() {
        grid.remove(&key);
    }
}

fn training_color_points(points: &[WeightedColorPoint]) -> Vec<WeightedColorPoint> {
    if points.len() <= KMEANS_TRAINING_POINT_LIMIT {
        return points.to_vec();
    }
    let mut bin_indices = HashMap::<u32, usize>::new();
    let mut bins = Vec::<([f64; 3], [f64; 3], f64)>::new();
    for point in points {
        let key = ((point.rgb[0] as u32 >> KMEANS_TRAINING_RGB_BIN_SHIFT) << 8)
            | ((point.rgb[1] as u32 >> KMEANS_TRAINING_RGB_BIN_SHIFT) << 4)
            | (point.rgb[2] as u32 >> KMEANS_TRAINING_RGB_BIN_SHIFT);
        let index = *bin_indices.entry(key).or_insert_with(|| {
            bins.push(([0.0; 3], [0.0; 3], 0.0));
            bins.len() - 1
        });
        let entry = &mut bins[index];
        entry.0[0] += point.rgb[0] as f64 * point.weight;
        entry.0[1] += point.rgb[1] as f64 * point.weight;
        entry.0[2] += point.rgb[2] as f64 * point.weight;
        entry.1[0] += point.lab[0] * point.weight;
        entry.1[1] += point.lab[1] * point.weight;
        entry.1[2] += point.lab[2] * point.weight;
        entry.2 += point.weight;
    }
    let mut points = bins
        .into_iter()
        .map(|(rgb_sum, lab_sum, weight)| WeightedColorPoint {
            rgb: [
                (rgb_sum[0] / weight).round() as u8,
                (rgb_sum[1] / weight).round() as u8,
                (rgb_sum[2] / weight).round() as u8,
            ],
            lab: [
                lab_sum[0] / weight,
                lab_sum[1] / weight,
                lab_sum[2] / weight,
            ],
            weight,
        })
        .collect::<Vec<_>>();
    sort_weighted_points(&mut points);
    points
}

fn sort_weighted_points(points: &mut [WeightedColorPoint]) {
    points.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.rgb.cmp(&right.rgb))
    });
}

fn palette_candidates(unique_count: usize) -> Vec<usize> {
    let mut candidates = Vec::new();
    push_candidate_range(&mut candidates, unique_count, 2, 16, 1);
    push_candidate_range(&mut candidates, unique_count, 16, 32, 4);
    push_candidate_range(&mut candidates, unique_count, 32, 64, 16);
    push_candidate_range(&mut candidates, unique_count, 64, 256, 64);
    if candidates.last().copied() != Some(unique_count) {
        candidates.push(unique_count);
    }
    candidates
}

fn push_candidate_range(
    candidates: &mut Vec<usize>,
    unique_count: usize,
    start: usize,
    end: usize,
    step: usize,
) {
    let mut value = start;
    while value <= end && value <= unique_count {
        if candidates.last().copied() != Some(value) {
            candidates.push(value);
        }
        value += step;
    }
}

fn pick_knee_candidate(candidates: &[usize], errors: &[f64]) -> usize {
    if candidates.len() == 1 {
        return candidates[0];
    }
    let min_k = candidates[0] as f64;
    let max_k = *candidates.last().unwrap() as f64;
    let min_error = *errors.last().unwrap();
    let max_error = errors[0];
    if max_error <= min_error {
        return *candidates.last().unwrap();
    }
    candidates
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            let left_score = knee_score(
                **left,
                errors[*left_index],
                min_k,
                max_k,
                min_error,
                max_error,
            );
            let right_score = knee_score(
                **right,
                errors[*right_index],
                min_k,
                max_k,
                min_error,
                max_error,
            );
            left_score.total_cmp(&right_score)
        })
        .map(|(_, candidate)| *candidate)
        .unwrap_or(1)
}

fn knee_score(
    candidate: usize,
    error: f64,
    min_k: f64,
    max_k: f64,
    min_error: f64,
    max_error: f64,
) -> f64 {
    let x = (candidate as f64 - min_k) / (max_k - min_k).max(1.0);
    let normalized_error = (error - min_error) / (max_error - min_error);
    1.0 - normalized_error - x
}

fn compute_weighted_kmeans_error(
    points: &[WeightedColorPoint],
    count: usize,
    iterations: usize,
) -> f64 {
    let centers = cluster_centers(points, count, iterations);
    points
        .iter()
        .map(|point| {
            centers
                .iter()
                .map(|center| lab_distance_sq(point.lab, *center))
                .fold(f64::INFINITY, f64::min)
                * point.weight
        })
        .sum()
}

fn cluster_centers(
    points: &[WeightedColorPoint],
    count: usize,
    iterations: usize,
) -> Vec<[f64; 3]> {
    let count = count.max(1).min(points.len());
    if count == points.len() {
        return points.iter().map(|point| point.lab).collect();
    }
    let mut centers = select_initial_centers(points, count);
    for _ in 0..iterations {
        let mut sums = vec![[0.0; 3]; count];
        let mut weights = vec![0.0; count];
        for point in points {
            let index = nearest_center_index(point.lab, &centers);
            sums[index][0] += point.lab[0] * point.weight;
            sums[index][1] += point.lab[1] * point.weight;
            sums[index][2] += point.lab[2] * point.weight;
            weights[index] += point.weight;
        }
        for index in 0..count {
            if weights[index] > 0.0 {
                centers[index] = [
                    sums[index][0] / weights[index],
                    sums[index][1] / weights[index],
                    sums[index][2] / weights[index],
                ];
            }
        }
    }
    centers
}

fn select_initial_centers(points: &[WeightedColorPoint], count: usize) -> Vec<[f64; 3]> {
    let mut centers = vec![points[0].lab];
    let mut min_distances = vec![f64::INFINITY; points.len()];
    while centers.len() < count {
        let newest = *centers.last().unwrap();
        let mut best_index = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (index, point) in points.iter().enumerate() {
            let distance = lab_distance_sq(point.lab, newest);
            if distance < min_distances[index] {
                min_distances[index] = distance;
            }
            let score = min_distances[index] * point.weight;
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        centers.push(points[best_index].lab);
    }
    centers
}

#[derive(Debug)]
struct ClusteredColorPoints {
    centers: Vec<[f64; 3]>,
}

#[derive(Debug)]
struct QuantizedPalette {
    palette: Vec<[u8; 3]>,
    assignments: Vec<usize>,
}

fn quantize_color_points_with_protected_highlights(
    points: &[WeightedColorPoint],
    protected: &[WeightedColorPoint],
    count: usize,
) -> QuantizedPalette {
    if protected.is_empty() {
        let training = training_color_points(points);
        if training.len() == points.len() {
            return quantize_color_points(points, count, KMEANS_ITERATIONS);
        }
        let centers = cluster_color_points(&training, count, KMEANS_ITERATIONS).centers;
        return quantized_palette_from_centers(points, &centers);
    }

    quantize_with_fixed_palette_prefix(points, protected, count, |remaining, remaining_count| {
        let training = training_color_points(remaining);
        if training.len() == remaining.len() {
            quantize_color_points(remaining, remaining_count, KMEANS_ITERATIONS).palette
        } else {
            let centers =
                cluster_color_points(&training, remaining_count, KMEANS_ITERATIONS).centers;
            quantized_palette_from_centers(remaining, &centers).palette
        }
    })
}

fn quantize_sampled_color_points(
    points: &[WeightedColorPoint],
    protected: &[WeightedColorPoint],
    count: usize,
    merge_threshold: f64,
) -> QuantizedPalette {
    if protected.is_empty() {
        let coalesced = if merge_threshold > 0.0 {
            merge_nearby_color_points(points, merge_threshold)
        } else {
            points.to_vec()
        };
        let training = training_color_points(&coalesced);
        let centers = cluster_color_points(&training, count, KMEANS_ITERATIONS).centers;
        return quantized_palette_from_centers(points, &centers);
    }

    quantize_with_fixed_palette_prefix(points, protected, count, |remaining, remaining_count| {
        let coalesced = if merge_threshold > 0.0 {
            merge_nearby_color_points(remaining, merge_threshold)
        } else {
            remaining.to_vec()
        };
        let training = training_color_points(&coalesced);
        let centers = cluster_color_points(&training, remaining_count, KMEANS_ITERATIONS).centers;
        quantized_palette_from_centers(remaining, &centers).palette
    })
}

fn quantize_with_fixed_palette_prefix<F>(
    points: &[WeightedColorPoint],
    protected: &[WeightedColorPoint],
    count: usize,
    remaining_palette: F,
) -> QuantizedPalette
where
    F: FnOnce(&[WeightedColorPoint], usize) -> Vec<[u8; 3]>,
{
    let limited = protected
        .iter()
        .take(count.saturating_sub(1))
        .cloned()
        .collect::<Vec<_>>();
    let protected_keys = limited
        .iter()
        .map(|point| pack_rgb(point.rgb))
        .collect::<std::collections::HashSet<_>>();
    let remaining = points
        .iter()
        .filter(|point| !protected_keys.contains(&pack_rgb(point.rgb)))
        .cloned()
        .collect::<Vec<_>>();
    let remaining_count = count.saturating_sub(limited.len());
    let mut palette = limited.iter().map(|point| point.rgb).collect::<Vec<_>>();
    if remaining_count > 0 && !remaining.is_empty() {
        palette.extend(remaining_palette(&remaining, remaining_count));
    }
    let centers = palette
        .iter()
        .map(|rgb| rgb_to_lab(*rgb))
        .collect::<Vec<_>>();
    let assignments = points
        .iter()
        .map(|point| nearest_center_index(point.lab, &centers))
        .collect();
    QuantizedPalette {
        palette,
        assignments,
    }
}

fn quantize_color_points(
    points: &[WeightedColorPoint],
    count: usize,
    iterations: usize,
) -> QuantizedPalette {
    let clustered = cluster_color_points(points, count, iterations);
    quantized_palette_from_centers(points, &clustered.centers)
}

fn cluster_color_points(
    points: &[WeightedColorPoint],
    count: usize,
    iterations: usize,
) -> ClusteredColorPoints {
    let count = count.max(1).min(points.len());
    if count == points.len() {
        return ClusteredColorPoints {
            centers: points.iter().map(|point| point.lab).collect(),
        };
    }

    let mut centers = select_initial_centers(points, count);
    let (mut assignments, mut buckets) = assign_points_to_centers(points, &centers);
    for _ in 0..iterations {
        reseed_empty_clusters(points, &mut centers, &mut assignments, &mut buckets);
        centers = update_centers(points, &buckets, &centers);
        (assignments, buckets) = assign_points_to_centers(points, &centers);
    }
    reseed_empty_clusters(points, &mut centers, &mut assignments, &mut buckets);
    centers = update_centers(points, &buckets, &centers);
    (assignments, buckets) = assign_points_to_centers(points, &centers);
    reseed_empty_clusters(points, &mut centers, &mut assignments, &mut buckets);

    ClusteredColorPoints { centers }
}

fn assign_points_to_centers(
    points: &[WeightedColorPoint],
    centers: &[[f64; 3]],
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut assignments = vec![0; points.len()];
    let mut buckets = vec![Vec::new(); centers.len()];
    for (index, point) in points.iter().enumerate() {
        let center_index = nearest_center_index(point.lab, centers);
        assignments[index] = center_index;
        buckets[center_index].push(index);
    }
    (assignments, buckets)
}

fn reseed_empty_clusters(
    points: &[WeightedColorPoint],
    centers: &mut [[f64; 3]],
    assignments: &mut [usize],
    buckets: &mut [Vec<usize>],
) {
    for empty_index in 0..buckets.len() {
        if !buckets[empty_index].is_empty() {
            continue;
        }

        let mut best_point_index = None;
        let mut best_source_index = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (point_index, point) in points.iter().enumerate() {
            let source_index = assignments[point_index];
            if buckets[source_index].len() <= 1 {
                continue;
            }
            let score = lab_distance_sq(point.lab, centers[source_index]) * point.weight;
            if score > best_score {
                best_score = score;
                best_point_index = Some(point_index);
                best_source_index = source_index;
            }
        }

        let Some(point_index) = best_point_index else {
            continue;
        };
        if let Some(position) = buckets[best_source_index]
            .iter()
            .position(|candidate| *candidate == point_index)
        {
            buckets[best_source_index].remove(position);
        }
        buckets[empty_index].push(point_index);
        assignments[point_index] = empty_index;
        centers[empty_index] = points[point_index].lab;
    }
}

fn update_centers(
    points: &[WeightedColorPoint],
    buckets: &[Vec<usize>],
    centers: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    centers
        .iter()
        .enumerate()
        .map(|(index, center)| {
            let bucket = &buckets[index];
            if bucket.is_empty() {
                return *center;
            }
            let mut sum = [0.0; 3];
            let mut weight = 0.0;
            for point_index in bucket {
                let point = &points[*point_index];
                sum[0] += point.lab[0] * point.weight;
                sum[1] += point.lab[1] * point.weight;
                sum[2] += point.lab[2] * point.weight;
                weight += point.weight;
            }
            [sum[0] / weight, sum[1] / weight, sum[2] / weight]
        })
        .collect()
}

fn quantized_palette_from_centers(
    points: &[WeightedColorPoint],
    centers: &[[f64; 3]],
) -> QuantizedPalette {
    if centers.is_empty() {
        return QuantizedPalette {
            palette: Vec::new(),
            assignments: Vec::new(),
        };
    }

    let (mut assignments, mut buckets) = assign_points_to_centers(points, centers);
    let populated_centers = centers
        .iter()
        .enumerate()
        .filter_map(|(index, center)| (!buckets[index].is_empty()).then_some(*center))
        .collect::<Vec<_>>();
    let centers = if !populated_centers.is_empty() && populated_centers.len() != centers.len() {
        (assignments, buckets) = assign_points_to_centers(points, &populated_centers);
        &populated_centers
    } else {
        centers
    };

    let palette = buckets
        .iter()
        .enumerate()
        .map(|(center_index, bucket)| {
            bucket
                .iter()
                .map(|point_index| &points[*point_index])
                .min_by(|left, right| {
                    let left_distance = lab_distance_sq(left.lab, centers[center_index]);
                    let right_distance = lab_distance_sq(right.lab, centers[center_index]);
                    left_distance
                        .total_cmp(&right_distance)
                        .then_with(|| right.weight.total_cmp(&left.weight))
                })
                .unwrap()
                .rgb
        })
        .collect();

    QuantizedPalette {
        palette,
        assignments,
    }
}

fn nearest_center_index(lab: [f64; 3], centers: &[[f64; 3]]) -> usize {
    centers
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            lab_distance_sq(lab, **left).total_cmp(&lab_distance_sq(lab, **right))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn lab_distance_sq(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dl = left[0] - right[0];
    let da = left[1] - right[1];
    let db = left[2] - right[2];
    dl * dl + da * da + db * db
}

fn rgb_to_lab(rgb: [u8; 3]) -> [f64; 3] {
    let r = srgb_to_linear(rgb[0]);
    let g = srgb_to_linear(rgb[1]);
    let b = srgb_to_linear(rgb[2]);
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.072175;
    let z = r * 0.0193339 + g * 0.119192 + b * 0.9503041;
    let fx = lab_f(x / 0.95047);
    let fy = lab_f(y);
    let fz = lab_f(z / 1.08883);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn srgb_to_linear(channel: u8) -> f64 {
    let normalized = channel as f64 / 255.0;
    if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        ((normalized + 0.055) / 1.055).powf(2.4)
    }
}

fn lab_f(value: f64) -> f64 {
    if value > 216.0 / 24389.0 {
        value.cbrt()
    } else {
        ((24389.0 / 27.0) * value + 16.0) / 116.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_pixels_are_normalized() {
        let image = RawImage::new(1, 1, vec![200, 100, 50, 0]);
        let result = quantize_image(&image, ColorMode::Full, 1.0, PaletteStrategy::Global);
        assert_eq!(result.image.data, vec![0, 0, 0, 0]);
    }

    #[test]
    fn fixed_palette_reduces_colors() {
        let image = RawImage::new(3, 1, vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
        let result = quantize_image(&image, ColorMode::Fixed(2), 1.0, PaletteStrategy::Global);
        assert_eq!(result.resolved_colors, Some(2));
        assert!(extract_palette_colors(&result.image).len() <= 2);
    }

    #[test]
    fn samples_center_pixel_when_grid_size_is_one() {
        let image = RawImage::new(3, 1, vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
        assert_eq!(sample_cell_color(&image, 0, 3, 0, 1, 1), [0, 255, 0, 255]);
    }

    #[test]
    fn treats_half_transparent_cell_as_transparent_like_ts_sampler() {
        let image = RawImage::new(2, 1, vec![255, 0, 0, 0, 0, 0, 255, 255]);
        assert_eq!(sample_cell_color(&image, 0, 2, 0, 1, 5), [0, 0, 0, 0]);
    }

    #[test]
    fn samples_opaque_color_when_cell_is_majority_opaque() {
        let image = RawImage::new(3, 1, vec![255, 0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255]);
        assert_eq!(sample_cell_color(&image, 0, 3, 0, 1, 5), [0, 0, 255, 255]);
    }

    #[test]
    fn samples_sparse_opaque_color_with_lower_coverage_threshold() {
        let mut image = RawImage::transparent(5, 5);
        image.set_pixel(2, 2, [255, 0, 0, 255]);
        image.set_pixel(2, 3, [255, 0, 0, 255]);

        assert_eq!(sample_cell_color(&image, 0, 5, 0, 5, 5), [0, 0, 0, 0]);
        assert_eq!(
            sample_cell_color_with_min_opaque_coverage(&image, 0, 5, 0, 5, 5, 0.06),
            [255, 0, 0, 255]
        );
    }

    #[test]
    fn bypasses_quantization_for_full_color_mode() {
        let image = RawImage::new(2, 1, vec![255, 0, 0, 255, 0, 0, 255, 255]);
        let result = quantize_image(&image, ColorMode::Full, 1.0, PaletteStrategy::Sampled);
        assert_eq!(result.resolved_colors, None);
        assert_eq!(result.image.data, image.data);
    }

    #[test]
    fn auto_palette_preserves_already_low_color_pixel_art() {
        let mut data = Vec::new();
        for index in 0..12u8 {
            data.extend_from_slice(&[
                index.saturating_mul(17),
                255u8.saturating_sub(index.saturating_mul(11)),
                index.saturating_mul(7),
                255,
            ]);
        }
        let image = RawImage::new(12, 1, data);
        let result = quantize_image(&image, ColorMode::Auto, 1.0, PaletteStrategy::Global);
        assert_eq!(result.resolved_colors, Some(12));
        assert_eq!(result.image.data, image.data);
    }

    #[test]
    fn palette_image_is_rectangular_not_a_thin_line() {
        let colors = (0..12)
            .map(|value| [value * 10, 0, 255 - value * 10])
            .collect::<Vec<_>>();
        let image = create_palette_image(&colors);
        assert!(image.width > 1);
        assert!(image.height > 1);
        assert_eq!(image.width * image.height, 12);
    }

    #[test]
    fn auto_palette_matches_dragon_coffee_fixture_color_budget() {
        let image = fixture_transform_sample("dragon_coffee.png");
        let result = quantize_image(&image, ColorMode::Auto, 1.0, PaletteStrategy::Global);
        assert_eq!(result.resolved_colors, Some(48));
    }

    #[test]
    fn auto_palette_matches_dragon_coffee_2_fixture_color_budget() {
        let image = fixture_transform_sample("dragon_coffee_2.png");
        let result = quantize_image(&image, ColorMode::Auto, 1.0, PaletteStrategy::Global);
        assert_eq!(result.resolved_colors, Some(24));
    }

    fn fixture_transform_sample(name: &str) -> RawImage {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/sources")
            .join(name);
        let bytes = std::fs::read(path).unwrap();
        let image = crate::image::decode_image(&bytes).unwrap();
        let detection =
            crate::detection::detect_pixel_width(&image, crate::core::PixelWidthDetector::Hybrid);
        let mesh = crate::mesh::Mesh::regular_with_offset(
            image.width,
            image.height,
            detection.width,
            detection.offset_x,
            detection.offset_y,
        );
        crate::mesh::sample_cells(&image, &mesh, 5, false)
    }
}
