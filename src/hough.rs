use crate::image::RawImage;
use rayon::prelude::*;

const HOUGH_CLUSTER_TOLERANCE: u32 = 2;
const HOUGH_MIN_AXIS_CLUSTERS: usize = 3;
const HOUGH_GAP_TRIM_FRACTION: f64 = 0.2;
const HOUGH_SUPPORTED_CLUSTER_WEIGHT_RATIO: f64 = 0.28;
const HOUGH_PHASE_ALIGNED_CLUSTER_WEIGHT_RATIO: f64 = 0.16;
const MIN_CLUSTER_OCCUPANCY_RATIO: f32 = 0.55;
const MIN_SUBHARMONIC_OCCUPANCY_RATIO: f32 = 0.75;
const MIN_SUBHARMONIC_OCCUPANCY_GAIN: f32 = 0.08;

#[derive(Debug, Clone)]
struct AxisCluster {
    position: f64,
    weight: f64,
}

#[derive(Debug, Clone)]
pub struct HoughDetection {
    pub width: u32,
    pub width_x: u32,
    pub width_y: u32,
    pub lines_x: Vec<u32>,
    pub lines_y: Vec<u32>,
    pub confidence: f32,
}

pub fn detect_hough_width(image: &RawImage) -> Option<u32> {
    detect_hough(image).map(|detection| detection.width)
}

pub fn detect_hough(image: &RawImage) -> Option<HoughDetection> {
    if image.width < 4 || image.height < 4 {
        return None;
    }

    let x_signal = edge_projection_x(image);
    let y_signal = edge_projection_y(image);
    detect_hough_from_projections(&x_signal, &y_signal, image.width, image.height)
}

pub fn detect_hough_from_projections(
    x_signal: &[u32],
    y_signal: &[u32],
    width: u32,
    height: u32,
) -> Option<HoughDetection> {
    let clusters_x = cluster_axis_samples(x_signal);
    let clusters_y = cluster_axis_samples(y_signal);
    let raw_lines_x = merge_anchors_with_borders(cluster_positions(&clusters_x), width);
    let raw_lines_y = merge_anchors_with_borders(cluster_positions(&clusters_y), height);
    if raw_lines_x.len() < HOUGH_MIN_AXIS_CLUSTERS || raw_lines_y.len() < HOUGH_MIN_AXIS_CLUSTERS {
        return None;
    }

    let raw_width_x = spacing_from_clusters(&raw_lines_x)?;
    let raw_width_y = spacing_from_clusters(&raw_lines_y)?;
    let lines_x = merge_anchors_with_borders(
        cluster_positions(&filter_supported_clusters(&clusters_x, raw_width_x)),
        width,
    );
    let lines_y = merge_anchors_with_borders(
        cluster_positions(&filter_supported_clusters(&clusters_y, raw_width_y)),
        height,
    );
    if lines_x.len() < HOUGH_MIN_AXIS_CLUSTERS || lines_y.len() < HOUGH_MIN_AXIS_CLUSTERS {
        return None;
    }

    let width_x = spacing_from_clusters(&lines_x).unwrap_or(raw_width_x);
    let width_y = spacing_from_clusters(&lines_y).unwrap_or(raw_width_y);
    let width = ((width_x + width_y) as f32 / 2.0).round().max(1.0) as u32;
    let confidence =
        (phase_consensus(&lines_x, width_x) + phase_consensus(&lines_y, width_y)) / 2.0;

    Some(HoughDetection {
        width,
        width_x,
        width_y,
        lines_x,
        lines_y,
        confidence,
    })
}

fn edge_projection_x(image: &RawImage) -> Vec<u32> {
    (0..image.width)
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
        .collect()
}

fn edge_projection_y(image: &RawImage) -> Vec<u32> {
    (0..image.height)
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
        .collect()
}

fn cluster_axis_samples(signal: &[u32]) -> Vec<AxisCluster> {
    if signal.len() < 3 {
        return Vec::new();
    }

    let threshold = percentile(signal, 0.82).max(mean(signal) * 1.25);
    let mut clusters = Vec::<AxisCluster>::new();
    for index in 1..signal.len() - 1 {
        let value = signal[index] as f64;
        if value < threshold
            || signal[index] < signal[index - 1]
            || signal[index] < signal[index + 1]
        {
            continue;
        }

        if let Some(cluster) = clusters.last_mut()
            && index as f64 - cluster.position <= HOUGH_CLUSTER_TOLERANCE as f64
        {
            let combined = cluster.weight + value;
            cluster.position =
                (cluster.position * cluster.weight + index as f64 * value) / combined;
            cluster.weight = combined;
            continue;
        }
        clusters.push(AxisCluster {
            position: index as f64,
            weight: value,
        });
    }

    clusters
}

fn cluster_positions(clusters: &[AxisCluster]) -> Vec<u32> {
    clusters
        .iter()
        .map(|cluster| cluster.position.round() as u32)
        .collect()
}

fn filter_supported_clusters(clusters: &[AxisCluster], spacing: u32) -> Vec<AxisCluster> {
    if clusters.len() <= HOUGH_MIN_AXIS_CLUSTERS {
        return clusters.to_vec();
    }

    let max_weight = clusters
        .iter()
        .map(|cluster| cluster.weight)
        .fold(0.0f64, f64::max);
    if max_weight <= 0.0 {
        return clusters.to_vec();
    }

    let strong_threshold = max_weight * HOUGH_SUPPORTED_CLUSTER_WEIGHT_RATIO;
    let phase_threshold = max_weight * HOUGH_PHASE_ALIGNED_CLUSTER_WEIGHT_RATIO;
    let dominant_phase = dominant_cluster_phase(clusters, spacing);
    let filtered = clusters
        .iter()
        .filter(|cluster| {
            cluster.weight >= strong_threshold
                || (cluster.weight >= phase_threshold
                    && dominant_phase.is_some_and(|phase| {
                        phase_distance(cluster.position.round() as u32, phase, spacing)
                            <= HOUGH_CLUSTER_TOLERANCE + 1
                    }))
        })
        .cloned()
        .collect::<Vec<_>>();

    if filtered.len() >= HOUGH_MIN_AXIS_CLUSTERS {
        filtered
    } else {
        clusters.to_vec()
    }
}

fn dominant_cluster_phase(clusters: &[AxisCluster], spacing: u32) -> Option<u32> {
    if spacing <= 1 || clusters.is_empty() {
        return None;
    }

    let mut phases = vec![0.0f64; spacing as usize];
    for cluster in clusters {
        let phase = cluster.position.round() as u32 % spacing;
        phases[phase as usize] += cluster.weight;
    }
    phases
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(phase, _)| phase as u32)
}

fn phase_distance(position: u32, phase: u32, spacing: u32) -> u32 {
    if spacing <= 1 {
        return 0;
    }
    let residue = position % spacing;
    let delta = residue.abs_diff(phase);
    delta.min(spacing - delta)
}

fn merge_anchors_with_borders(mut anchors: Vec<u32>, size: u32) -> Vec<u32> {
    anchors.push(0);
    anchors.push(size.saturating_sub(1));
    anchors.sort_unstable();

    let mut merged = Vec::new();
    for anchor in anchors {
        if let Some(previous) = merged.last_mut()
            && anchor.abs_diff(*previous) <= HOUGH_CLUSTER_TOLERANCE
        {
            *previous = ((*previous + anchor) as f32 / 2.0).round() as u32;
            continue;
        }
        merged.push(anchor);
    }
    merged
}

fn spacing_from_clusters(lines: &[u32]) -> Option<u32> {
    if lines.len() < HOUGH_MIN_AXIS_CLUSTERS {
        return None;
    }

    let mut gaps = lines
        .windows(2)
        .filter_map(|pair| {
            let gap = pair[1].saturating_sub(pair[0]);
            (gap > 0).then_some(gap)
        })
        .collect::<Vec<_>>();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    let raw_spacing = median_u32(&gaps).round().max(1.0) as u32;
    let raw_occupancy = cluster_occupancy(lines, raw_spacing);

    let mut candidates = vec![raw_spacing];
    for gap in &gaps {
        for divisor in divisors(*gap) {
            if *gap / divisor <= 4 {
                candidates.push(divisor);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let mut best_spacing = raw_spacing;
    let mut best_occupancy = raw_occupancy;
    for candidate in candidates {
        let occupancy = cluster_occupancy(lines, candidate);
        if candidate < raw_spacing {
            if occupancy >= MIN_SUBHARMONIC_OCCUPANCY_RATIO
                && occupancy >= raw_occupancy + MIN_SUBHARMONIC_OCCUPANCY_GAIN
            {
                best_spacing = candidate;
                best_occupancy = occupancy;
            }
        } else if best_occupancy < MIN_CLUSTER_OCCUPANCY_RATIO
            && occupancy >= MIN_CLUSTER_OCCUPANCY_RATIO
            && occupancy > best_occupancy
        {
            best_spacing = candidate;
            best_occupancy = occupancy;
        }
    }
    if best_occupancy >= MIN_CLUSTER_OCCUPANCY_RATIO {
        return Some(best_spacing.max(1));
    }

    let low = percentile_u32(&gaps, HOUGH_GAP_TRIM_FRACTION);
    let high = percentile_u32(&gaps, 1.0 - HOUGH_GAP_TRIM_FRACTION);
    let middle = gaps
        .into_iter()
        .filter(|gap| *gap >= low && *gap <= high)
        .collect::<Vec<_>>();
    Some(median_u32(&middle).round().max(1.0) as u32)
}

fn cluster_occupancy(lines: &[u32], spacing: u32) -> f32 {
    if spacing <= 1 || lines.is_empty() {
        return 1.0;
    }

    let min_line = lines[0];
    let max_line = *lines.last().unwrap_or(&min_line);
    let mut offsets = lines.iter().map(|line| line % spacing).collect::<Vec<_>>();
    offsets.sort_unstable();
    offsets.dedup();

    let tolerance = HOUGH_CLUSTER_TOLERANCE;
    let mut best = 0.0f32;
    for offset in offsets {
        let mut start = offset;
        while start + spacing <= min_line {
            start += spacing;
        }
        while start > min_line {
            start = start.saturating_sub(spacing);
        }

        let mut expected = 0u32;
        let mut matched = 0u32;
        let mut line_index = 0usize;
        let mut position = start;
        while position <= max_line.saturating_add(tolerance) {
            if position >= min_line.saturating_sub(tolerance) {
                expected += 1;
                while line_index < lines.len()
                    && lines[line_index] < position.saturating_sub(tolerance)
                {
                    line_index += 1;
                }
                if line_index < lines.len() && lines[line_index].abs_diff(position) <= tolerance {
                    matched += 1;
                    line_index += 1;
                }
            }
            position = position.saturating_add(spacing);
            if spacing == 0 {
                break;
            }
        }
        if expected > 0 {
            best = best.max(matched as f32 / expected as f32);
        }
    }
    best
}

fn phase_consensus(lines: &[u32], spacing: u32) -> f32 {
    if lines.len() < 3 || spacing <= 1 {
        return 1.0;
    }

    let residues = lines[1..lines.len() - 1]
        .iter()
        .map(|line| line % spacing)
        .collect::<Vec<_>>();
    let mut best = 0u32;
    for candidate in &residues {
        let mut matched = 0u32;
        for residue in &residues {
            let delta = candidate.abs_diff(*residue);
            if delta.min(spacing - delta) <= HOUGH_CLUSTER_TOLERANCE + 1 {
                matched += 1;
            }
        }
        best = best.max(matched);
    }
    best as f32 / residues.len().max(1) as f32
}

fn divisors(value: u32) -> Vec<u32> {
    (2..value)
        .filter(|divisor| value.is_multiple_of(*divisor))
        .collect()
}

fn median_u32(values: &[u32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] as f64 + values[mid] as f64) / 2.0
    } else {
        values[mid] as f64
    }
}

fn percentile_u32(values: &[u32], percentile: f64) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

fn mean(signal: &[u32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    signal.iter().map(|value| *value as f64).sum::<f64>() / signal.len() as f64
}

fn percentile(signal: &[u32], percentile: f64) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let mut values = signal.to_vec();
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index] as f64
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing_does_not_prefer_sparse_subharmonic_noise() {
        let lines = vec![0, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80];

        assert_eq!(spacing_from_clusters(&lines), Some(8));
    }
}
