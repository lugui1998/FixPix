use rayon::prelude::*;

use crate::core::{PixelWidthDetector, PixelWidthSource};
use crate::image::{
    ALPHA_THRESHOLD, RawImage, choose_closest_integer_scale, limit_scale_for_max_dimension,
    scale_nearest,
};
use crate::palette::sample_cell_color;

const MIN_DEBUG_SEGMENT_LENGTH: u32 = 6;
const MAX_DEBUG_SEGMENTS_PER_FAMILY: usize = 250;
const DEBUG_PALETTE_MAX_SWATCH_SCALE: u32 = 64;
const DEBUG_PALETTE_MAX_WIDTH_RATIO: f32 = 0.36;
const DEBUG_PALETTE_MAX_HEIGHT_RATIO: f32 = 0.24;
const LOCAL_EDGE_REFINEMENT_RADIUS_RATIO: f32 = 0.4;
const LOCAL_EDGE_REFINEMENT_MAX_RADIUS: u32 = 14;
const LOCAL_EDGE_REFINEMENT_GAP_PENALTY: f32 = 1.15;
const LOCAL_EDGE_REFINEMENT_SHIFT_PENALTY: f32 = 0.06;
const LOCAL_EDGE_REFINEMENT_BAND_RADIUS: usize = 0;
const LOCAL_EDGE_REFINEMENT_SMOOTHING_PASSES: usize = 2;
const LOCAL_EDGE_STRONG_CONFIDENCE: f32 = 0.42;
const LOCAL_EDGE_WEAK_CONFIDENCE: f32 = 0.18;
const LOCAL_EDGE_CORNER_BONUS: f32 = 0.75;
const LOCAL_EDGE_WARP_MIN_ERROR_GAIN: f32 = 0.1;
const WARP_SUBDIVISION_MIN_EDGE_SCORE: f32 = 18.0;
const WARP_SUBDIVISION_MIN_EDGE_GAIN: f32 = 1.12;
const WARP_SUBDIVISION_MAX_RADIUS: u32 = 5;
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
    let width = mesh.lines_x.len().saturating_sub(1) as u32;
    let height = mesh.lines_y.len().saturating_sub(1) as u32;
    let mut out = vec![0; width as usize * height as usize * 4];
    out.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(index, pixel)| {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let mut color = if mesh.warp.is_some() {
                sample_warped_cell_color(image, mesh, x as usize, y as usize, sample_grid)
            } else {
                let (x0, x1, y0, y1) = mesh_cell_bounds(mesh, x as usize, y as usize, image);
                sample_cell_color(image, x0, x1, y0, y1, sample_grid)
            };
            if transparent_background && color[3] < 160 {
                color = [0, 0, 0, 0];
            }
            pixel.copy_from_slice(&color);
        });
    RawImage::new(width, height, out)
}

fn sample_warped_cell_color(
    image: &RawImage,
    mesh: &Mesh,
    cell_x: usize,
    cell_y: usize,
    sample_grid: u32,
) -> [u8; 4] {
    let Some(corners) = warped_cell_corners(mesh, cell_x, cell_y) else {
        let (x0, x1, y0, y1) = mesh_cell_bounds(mesh, cell_x, cell_y, image);
        return sample_cell_color(image, x0, x1, y0, y1, sample_grid);
    };

    let subdivision = subdivided_warped_cell_grid(image, corners);
    let grid = sample_grid.max(1);
    let center = subdivided_warp_point(&subdivision, 0.5, 0.5);
    let mut keys = [0u32; WARP_SAMPLE_COLOR_CAPACITY];
    let mut counts = [0u32; WARP_SAMPLE_COLOR_CAPACITY];
    let mut distances = [0.0f32; WARP_SAMPLE_COLOR_CAPACITY];
    let mut color_count = 0usize;
    let mut opaque_samples = 0u32;
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

    if total_samples == 0 || opaque_samples <= total_samples / 2 {
        return [0, 0, 0, 0];
    }
    best_key
        .map(unpack_rgb)
        .map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
        .unwrap_or([0, 0, 0, 0])
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

#[derive(Debug, Clone, Copy)]
struct WarpedCellSubgrid {
    points: [[(f32, f32); 3]; 3],
}

fn subdivided_warped_cell_grid(image: &RawImage, corners: WarpedCellCorners) -> WarpedCellSubgrid {
    let mut points = [[(0.0, 0.0); 3]; 3];
    points[0][0] = corners.top_left;
    points[0][2] = corners.top_right;
    points[2][0] = corners.bottom_left;
    points[2][2] = corners.bottom_right;
    points[0][1] = refine_horizontal_subdivision_point(
        image,
        midpoint(corners.top_left, corners.top_right),
        corners.top_left,
        corners.top_right,
    );
    points[2][1] = refine_horizontal_subdivision_point(
        image,
        midpoint(corners.bottom_left, corners.bottom_right),
        corners.bottom_left,
        corners.bottom_right,
    );
    points[1][0] = refine_vertical_subdivision_point(
        image,
        midpoint(corners.top_left, corners.bottom_left),
        corners.top_left,
        corners.bottom_left,
    );
    points[1][2] = refine_vertical_subdivision_point(
        image,
        midpoint(corners.top_right, corners.bottom_right),
        corners.top_right,
        corners.bottom_right,
    );
    let center = bilinear_point(corners, 0.5, 0.5);
    let edge_center = (
        (points[0][1].0 + points[2][1].0 + points[1][0].0 + points[1][2].0) / 4.0,
        (points[0][1].1 + points[2][1].1 + points[1][0].1 + points[1][2].1) / 4.0,
    );
    points[1][1] = (
        center.0 * 0.5 + edge_center.0 * 0.5,
        center.1 * 0.5 + edge_center.1 * 0.5,
    );
    WarpedCellSubgrid { points }
}

fn subdivided_warp_point(subdivision: &WarpedCellSubgrid, u: f32, v: f32) -> (f32, f32) {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let sub_x = if u < 0.5 { 0 } else { 1 };
    let sub_y = if v < 0.5 { 0 } else { 1 };
    let local_u = if sub_x == 0 { u * 2.0 } else { u * 2.0 - 1.0 };
    let local_v = if sub_y == 0 { v * 2.0 } else { v * 2.0 - 1.0 };
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

fn refine_horizontal_subdivision_point(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
) -> (f32, f32) {
    let radius = subdivision_search_radius(start, end);
    let original = point.1.round() as i32;
    let min_y = (original - radius as i32).max(1);
    let max_y = (original + radius as i32).min(image.height.saturating_sub(1) as i32);
    let x0 = start.0.min(end.0).floor().max(0.0) as u32;
    let x1 = start
        .0
        .max(end.0)
        .ceil()
        .min(image.width.saturating_sub(1) as f32) as u32;
    let base_score = horizontal_edge_segment_score(image, original.max(1) as u32, x0, x1);
    let mut best_y = original;
    let mut best_score = base_score;
    for y in min_y..=max_y {
        let score = horizontal_edge_segment_score(image, y as u32, x0, x1);
        if score > best_score {
            best_score = score;
            best_y = y;
        }
    }
    let Some(weight) = subdivision_snap_weight(base_score, best_score) else {
        return point;
    };
    (point.0, point.1 * (1.0 - weight) + best_y as f32 * weight)
}

fn refine_vertical_subdivision_point(
    image: &RawImage,
    point: (f32, f32),
    start: (f32, f32),
    end: (f32, f32),
) -> (f32, f32) {
    let radius = subdivision_search_radius(start, end);
    let original = point.0.round() as i32;
    let min_x = (original - radius as i32).max(1);
    let max_x = (original + radius as i32).min(image.width.saturating_sub(1) as i32);
    let y0 = start.1.min(end.1).floor().max(0.0) as u32;
    let y1 = start
        .1
        .max(end.1)
        .ceil()
        .min(image.height.saturating_sub(1) as f32) as u32;
    let base_score = vertical_edge_segment_score(image, original.max(1) as u32, y0, y1);
    let mut best_x = original;
    let mut best_score = base_score;
    for x in min_x..=max_x {
        let score = vertical_edge_segment_score(image, x as u32, y0, y1);
        if score > best_score {
            best_score = score;
            best_x = x;
        }
    }
    let Some(weight) = subdivision_snap_weight(base_score, best_score) else {
        return point;
    };
    (point.0 * (1.0 - weight) + best_x as f32 * weight, point.1)
}

fn subdivision_search_radius(start: (f32, f32), end: (f32, f32)) -> u32 {
    let span = (start.0 - end.0).abs().max((start.1 - end.1).abs());
    ((span * 0.25).round() as u32).clamp(2, WARP_SUBDIVISION_MAX_RADIUS)
}

fn subdivision_snap_weight(base_score: f32, best_score: f32) -> Option<f32> {
    if best_score < WARP_SUBDIVISION_MIN_EDGE_SCORE {
        return None;
    }
    if best_score < (base_score * WARP_SUBDIVISION_MIN_EDGE_GAIN).max(base_score + 2.0) {
        return None;
    }
    let gain = (best_score - base_score).max(0.0) / best_score.max(1.0);
    Some((0.35 + gain * 0.55).clamp(0.0, 0.9))
}

fn horizontal_edge_segment_score(image: &RawImage, y: u32, x0: u32, x1: u32) -> f32 {
    if y == 0 || y >= image.height || x1 < x0 {
        return 0.0;
    }
    let mut total = 0u32;
    let mut count = 0u32;
    for x in x0..=x1 {
        total += luma_delta(image.pixel(x, y - 1), image.pixel(x, y));
        count += 1;
    }
    total as f32 / count.max(1) as f32
}

fn vertical_edge_segment_score(image: &RawImage, x: u32, y0: u32, y1: u32) -> f32 {
    if x == 0 || x >= image.width || y1 < y0 {
        return 0.0;
    }
    let mut total = 0u32;
    let mut count = 0u32;
    for y in y0..=y1 {
        total += luma_delta(image.pixel(x - 1, y), image.pixel(x, y));
        count += 1;
    }
    total as f32 / count.max(1) as f32
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

    let computed_signals;
    let (horizontal, vertical) = match boundary_signals {
        Some(signals) => signals,
        None => {
            computed_signals = luma_boundary_signals(image);
            (computed_signals.0.as_slice(), computed_signals.1.as_slice())
        }
    };
    let luma = luma_buffer(image);
    let integral = build_integral_image(&luma, image.width, image.height);
    let integral_stride = image.width as usize + 1;
    let mut best_mesh = result.mesh.clone();
    let mut best_error = reconstruction_error(
        &luma,
        image.width,
        image.height,
        &integral,
        integral_stride,
        &best_mesh,
    );
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
    let refined_error = reconstruction_error(
        &luma,
        image.width,
        image.height,
        &integral,
        integral_stride,
        &refined,
    );
    if refined_error < best_error {
        best_mesh = refined;
        best_error = refined_error;
    }

    if let Some(warp) = create_warped_mesh_from_local_edges(image, &luma, &best_mesh, result) {
        let warped_mesh = Mesh {
            lines_x: best_mesh.lines_x.clone(),
            lines_y: best_mesh.lines_y.clone(),
            warp: Some(warp),
        };
        let warped_error = reconstruction_error(
            &luma,
            image.width,
            image.height,
            &integral,
            integral_stride,
            &warped_mesh,
        );
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
    luma: &[u8],
    mesh: &Mesh,
    result: &MeshResult,
) -> Option<MeshWarp> {
    let row_count = mesh.lines_y.len().saturating_sub(1);
    let column_count = mesh.lines_x.len().saturating_sub(1);
    if row_count < 2 || column_count < 2 {
        return None;
    }

    let energy = LocalEdgeEnergy::new(luma, image.width, image.height);
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
    fn new(luma: &[u8], width: u32, height: u32) -> Self {
        let len = width as usize * height as usize;
        let mut vertical = vec![0.0f32; len];
        let mut horizontal = vec![0.0f32; len];
        for y in 0..height {
            for x in 1..width {
                let index = (y * width + x) as usize;
                vertical[index] = luma[index].abs_diff(luma[index - 1]) as f32;
            }
        }
        for y in 1..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                horizontal[index] = luma[index].abs_diff(luma[index - width as usize]) as f32;
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
        return 0.92;
    }
    let t = (confidence - LOCAL_EDGE_WEAK_CONFIDENCE)
        / (LOCAL_EDGE_STRONG_CONFIDENCE - LOCAL_EDGE_WEAK_CONFIDENCE);
    0.25 + t * 0.67
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

fn luma_buffer(image: &RawImage) -> Vec<u8> {
    image
        .data
        .par_chunks_exact(4)
        .map(|pixel| luma([pixel[0], pixel[1], pixel[2], pixel[3]]) as u8)
        .collect()
}

fn reconstruction_error(
    luma: &[u8],
    width: u32,
    height: u32,
    integral: &[f64],
    stride: usize,
    mesh: &Mesh,
) -> f32 {
    let mut total_error = 0.0f64;
    let mut total_pixels = 0usize;

    for cell_y in 0..mesh.lines_y.len().saturating_sub(1) {
        for cell_x in 0..mesh.lines_x.len().saturating_sub(1) {
            let (x0, x1, y0, y1) = mesh_cell_bounds_for_size(mesh, cell_x, cell_y, width, height);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }

            let pixel_count = (x1 - x0) as usize * (y1 - y0) as usize;
            let mean_luma = rect_sum(integral, stride, x0, x1, y0, y1) / pixel_count as f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    total_error += (luma[(y * width + x) as usize] as f64 - mean_luma).abs();
                    total_pixels += 1;
                }
            }
        }
    }

    if total_pixels == 0 {
        f32::INFINITY
    } else {
        (total_error / total_pixels as f64) as f32
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

fn build_integral_image(luma: &[u8], width: u32, height: u32) -> Vec<f64> {
    let stride = width as usize + 1;
    let mut integral = vec![0.0; stride * (height as usize + 1)];
    for y in 0..height as usize {
        let mut row_sum = 0.0;
        let integral_row = (y + 1) * stride;
        let previous_integral_row = y * stride;
        for x in 0..width as usize {
            row_sum += luma[y * width as usize + x] as f64;
            integral[integral_row + x + 1] = integral[previous_integral_row + x + 1] + row_sum;
        }
    }
    integral
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
    let detection_image = if mesh.scale_used > 1 {
        scale_nearest(original, mesh.scale_used)
    } else {
        original.clone()
    };
    let preview_scale = limit_scale_for_max_dimension(&detection_image, debug_scale.max(1));
    let debug_display_multiplier = mesh.scale_used.max(1) * preview_scale;
    let edge_source = crop_debug_detection_image(&detection_image, mesh.debug_crop_offset);

    let original_preview = scale_nearest(original, debug_display_multiplier);
    let enlarged_detection_image = scale_nearest(&detection_image, preview_scale);
    let canny_mask = create_edge_preview(&edge_source);
    let canny_preview = scale_nearest(&canny_mask, preview_scale);
    let hough_preview = draw_detection_line_overlay(
        &enlarged_detection_image,
        &canny_mask,
        mesh.debug_crop_offset,
        preview_scale,
    );
    let hough_preview = draw_anchor_overlay(&hough_preview, mesh, preview_scale);
    let grid_preview = draw_grid_overlay(
        &enlarged_detection_image,
        &detection_image,
        mesh,
        preview_scale,
    );
    let unscaled_preview = unscaled.clone();
    let mut final_preview = scale_nearest(unscaled, debug_display_multiplier);

    let final_preview_scale =
        choose_closest_integer_scale(&final_preview, grid_preview.width, grid_preview.height);
    if final_preview_scale > 1 {
        final_preview = scale_nearest(&final_preview, final_preview_scale);
    }
    let palette_preview =
        create_debug_palette_preview(palette_colors, final_preview.width, final_preview.height);

    compose_debug_rows(&[
        &[original_preview, canny_preview, hough_preview, grid_preview],
        &[final_preview, unscaled_preview, palette_preview],
    ])
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

fn luma_boundary_signals(image: &RawImage) -> (Vec<u32>, Vec<u32>) {
    let horizontal = (0..image.width)
        .into_par_iter()
        .map(|x| {
            let mut sum = 0;
            for y in 0..image.height {
                if x > 0 {
                    sum += luma_delta(image.pixel(x - 1, y), image.pixel(x, y));
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
                    sum += luma_delta(image.pixel(x, y - 1), image.pixel(x, y));
                }
            }
            sum
        })
        .collect();
    (horizontal, vertical)
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
        .max(2.0)
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
            let improvement = (selected_score - base_score).max(0.0);
            (selected_score * 0.75 + improvement * 0.5).clamp(0.0, 1.0)
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

fn create_edge_preview(image: &RawImage) -> RawImage {
    let mut out = RawImage::transparent(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let current = image.pixel(x, y);
            let right = if x + 1 < image.width {
                image.pixel(x + 1, y)
            } else {
                current
            };
            let down = if y + 1 < image.height {
                image.pixel(x, y + 1)
            } else {
                current
            };
            let edge = color_delta(current, right).max(color_delta(current, down));
            let value = if edge > 80 { 255 } else { 0 };
            out.set_pixel(x, y, [value, value, value, 255]);
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

fn draw_grid_overlay(
    base: &RawImage,
    source: &RawImage,
    mesh: &MeshResult,
    preview_scale: u32,
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
                if let Some(corners) = warped_cell_corners(&mesh.mesh, cell_x, cell_y) {
                    draw_warped_subdivision_overlay(&mut out, source, corners, mesh, preview_scale);
                    draw_warped_grid_edge(
                        &mut out,
                        corners.top_left,
                        corners.top_right,
                        mesh,
                        preview_scale,
                        [255, 0, 0, 255],
                    );
                    draw_warped_grid_edge(
                        &mut out,
                        corners.bottom_left,
                        corners.bottom_right,
                        mesh,
                        preview_scale,
                        [255, 0, 0, 255],
                    );
                    draw_warped_grid_edge(
                        &mut out,
                        corners.top_left,
                        corners.bottom_left,
                        mesh,
                        preview_scale,
                        [255, 0, 0, 255],
                    );
                    draw_warped_grid_edge(
                        &mut out,
                        corners.top_right,
                        corners.bottom_right,
                        mesh,
                        preview_scale,
                        [255, 0, 0, 255],
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
                out.set_pixel(scaled_x, y, [255, 0, 0, 255]);
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
                out.set_pixel(x, scaled_y, [255, 0, 0, 255]);
            }
        }
    }
    out
}

fn draw_warped_subdivision_overlay(
    image: &mut RawImage,
    source: &RawImage,
    corners: WarpedCellCorners,
    mesh: &MeshResult,
    preview_scale: u32,
) {
    let subdivision = subdivided_warped_cell_grid(source, corners);
    let points = subdivision.points;
    let color = [0, 255, 80, 255];
    let base_points = base_subdivision_points(corners);
    draw_moved_warped_grid_edge(
        image,
        (points[0][1], base_points[0][1]),
        (points[1][1], base_points[1][1]),
        mesh,
        preview_scale,
        color,
    );
    draw_moved_warped_grid_edge(
        image,
        (points[1][1], base_points[1][1]),
        (points[2][1], base_points[2][1]),
        mesh,
        preview_scale,
        color,
    );
    draw_moved_warped_grid_edge(
        image,
        (points[1][0], base_points[1][0]),
        (points[1][1], base_points[1][1]),
        mesh,
        preview_scale,
        color,
    );
    draw_moved_warped_grid_edge(
        image,
        (points[1][1], base_points[1][1]),
        (points[1][2], base_points[1][2]),
        mesh,
        preview_scale,
        color,
    );
    for y in 0..3 {
        for x in 0..3 {
            if subdivision_point_moved(points[y][x], base_points[y][x]) {
                draw_warped_grid_point(image, points[y][x], mesh, preview_scale);
            }
        }
    }
}

fn base_subdivision_points(corners: WarpedCellCorners) -> [[(f32, f32); 3]; 3] {
    let mut points = [[(0.0, 0.0); 3]; 3];
    for (y, row) in points.iter_mut().enumerate() {
        for (x, point) in row.iter_mut().enumerate() {
            *point = bilinear_point(corners, x as f32 / 2.0, y as f32 / 2.0);
        }
    }
    points
}

fn draw_moved_warped_grid_edge(
    image: &mut RawImage,
    start: ((f32, f32), (f32, f32)),
    end: ((f32, f32), (f32, f32)),
    mesh: &MeshResult,
    preview_scale: u32,
    color: [u8; 4],
) {
    if subdivision_point_moved(start.0, start.1) || subdivision_point_moved(end.0, end.1) {
        draw_warped_grid_edge(image, start.0, end.0, mesh, preview_scale, color);
    }
}

fn subdivision_point_moved(point: (f32, f32), base: (f32, f32)) -> bool {
    (point.0 - base.0).abs().max((point.1 - base.1).abs()) >= 0.35
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

fn draw_warped_grid_point(
    image: &mut RawImage,
    point: (f32, f32),
    mesh: &MeshResult,
    preview_scale: u32,
) {
    let mesh_scale = mesh.scale_used.max(1);
    let x = debug_grid_coordinate(
        point.0.round().max(0.0) as u32,
        mesh.debug_crop_offset.0,
        mesh_scale,
        preview_scale,
        image.width,
    );
    let y = debug_grid_coordinate(
        point.1.round().max(0.0) as u32,
        mesh.debug_crop_offset.1,
        mesh_scale,
        preview_scale,
        image.height,
    );
    draw_anchor_point(image, x, y, [255, 255, 0, 255]);
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
            image.set_pixel(current_x as u32, current_y as u32, color);
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

fn color_delta(left: [u8; 4], right: [u8; 4]) -> u32 {
    left[0].abs_diff(right[0]) as u32
        + left[1].abs_diff(right[1]) as u32
        + left[2].abs_diff(right[2]) as u32
        + left[3].abs_diff(right[3]) as u32
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
    fn debug_sheet_includes_unscaled_output_at_natural_size() {
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
        assert_eq!(output_pixels, 5);
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
        let overlay = draw_grid_overlay(&base, &base, &mesh, 1);

        assert_eq!(corners.bottom_left, (1.0, 3.0));
        assert_eq!(overlay.pixel(1, 2), [255, 0, 0, 255]);
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
        let luma = luma_buffer(&image);

        let warp = create_warped_mesh_from_local_edges(&image, &luma, &mesh, &result).unwrap();

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
        let luma = luma_buffer(&image);
        let energy = LocalEdgeEnergy::new(&luma, image.width, image.height);

        let edge_only = energy.vertical_band_edge_score(3, 3, 6);
        let with_corner = energy.vertical_band_corner_score(3, 3, 6);

        assert!(with_corner > edge_only);
        assert!(energy.corner_score(3, 3) > 0.0);
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
