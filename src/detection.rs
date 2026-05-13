use rayon::prelude::*;

use crate::core::{PixelWidthDetector, PixelWidthSource};
use crate::image::RawImage;

const PROJECTION_MAX_LAG: u32 = 120;
const PROJECTION_AGREEMENT_TOLERANCE: f32 = 0.25;
const MIN_PEAK_OCCUPANCY_RATIO: f32 = 0.55;
const MIN_AUTOCORRELATION_SCORE_RATIO: f64 = 0.6;
const INTEGER_MULTIPLE_TOLERANCE: f32 = 0.12;
const MIN_HOUGH_CONFIDENCE: f32 = 0.72;
const MIN_ANISOTROPIC_HOUGH_CONFIDENCE: f32 = 0.9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelWidthDetection {
    pub width: u32,
    pub width_x: u32,
    pub width_y: u32,
    pub offset_x: u32,
    pub offset_y: u32,
    pub source: PixelWidthSource,
    pub anchor_lines_x: Option<Vec<u32>>,
    pub anchor_lines_y: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelWidthAnalysis {
    pub detection: PixelWidthDetection,
    pub edge_projection_x: Vec<u32>,
    pub edge_projection_y: Vec<u32>,
}

pub fn detect_pixel_width(image: &RawImage, detector: PixelWidthDetector) -> PixelWidthDetection {
    analyze_pixel_width(image, detector).detection
}

pub fn analyze_pixel_width(image: &RawImage, detector: PixelWidthDetector) -> PixelWidthAnalysis {
    let projections = EdgeProjections::new(image);
    let detection = detect_pixel_width_from_projections(image, detector, &projections);
    PixelWidthAnalysis {
        detection,
        edge_projection_x: projections.x,
        edge_projection_y: projections.y,
    }
}

fn detect_pixel_width_from_projections(
    image: &RawImage,
    detector: PixelWidthDetector,
    projections: &EdgeProjections,
) -> PixelWidthDetection {
    let projection = detect_projection_from_signals(image, projections);
    match detector {
        PixelWidthDetector::Projection => PixelWidthDetection {
            width: projection.width,
            width_x: projection.width_x,
            width_y: projection.width_y,
            offset_x: projection.offset_x,
            offset_y: projection.offset_y,
            source: PixelWidthSource::Projection,
            anchor_lines_x: None,
            anchor_lines_y: None,
        },
        PixelWidthDetector::Hough => {
            hough_detection(image, projections).unwrap_or(PixelWidthDetection {
                width: projection.width,
                width_x: projection.width_x,
                width_y: projection.width_y,
                offset_x: projection.offset_x,
                offset_y: projection.offset_y,
                source: PixelWidthSource::Projection,
                anchor_lines_x: None,
                anchor_lines_y: None,
            })
        }
        PixelWidthDetector::Hybrid => {
            let hough = hough_detection(image, projections);
            if let Some(hough) = hough
                && hough.width > 1
                && (projection.width.abs_diff(hough.width) <= projection.width.max(hough.width) / 3
                    || hough
                        .anchor_lines_x
                        .as_ref()
                        .is_some_and(|lines| lines.len() > 8)
                        && hough
                            .anchor_lines_y
                            .as_ref()
                            .is_some_and(|lines| lines.len() > 8))
            {
                return hough;
            }
            PixelWidthDetection {
                width: projection.width,
                width_x: projection.width_x,
                width_y: projection.width_y,
                offset_x: projection.offset_x,
                offset_y: projection.offset_y,
                source: PixelWidthSource::Hybrid,
                anchor_lines_x: None,
                anchor_lines_y: None,
            }
        }
    }
}

struct EdgeProjections {
    x: Vec<u32>,
    y: Vec<u32>,
}

impl EdgeProjections {
    fn new(image: &RawImage) -> Self {
        Self {
            x: edge_projection_x(image),
            y: edge_projection_y(image),
        }
    }
}

fn hough_detection(image: &RawImage, projections: &EdgeProjections) -> Option<PixelWidthDetection> {
    let hough = crate::hough::detect_hough_from_projections(
        &projections.x,
        &projections.y,
        image.width,
        image.height,
    )?;
    if hough.confidence < MIN_HOUGH_CONFIDENCE {
        return None;
    }
    if !hough_axis_widths_acceptable(hough.width_x, hough.width_y, hough.confidence) {
        return None;
    }
    Some(PixelWidthDetection {
        width: hough.width,
        width_x: hough.width_x,
        width_y: hough.width_y,
        offset_x: best_offset(&projections.x, hough.width_x),
        offset_y: best_offset(&projections.y, hough.width_y),
        source: PixelWidthSource::Hough,
        anchor_lines_x: Some(hough.lines_x),
        anchor_lines_y: Some(hough.lines_y),
    })
}

fn hough_axis_widths_acceptable(width_x: u32, width_y: u32, confidence: f32) -> bool {
    if widths_agree(width_x, width_y) {
        return true;
    }
    let low = width_x.min(width_y).max(1) as f32;
    let high = width_x.max(width_y).max(1) as f32;
    confidence >= MIN_ANISOTROPIC_HOUGH_CONFIDENCE && high / low <= 1.8
}

pub fn detect_projection_width(image: &RawImage) -> u32 {
    detect_projection(image).width
}

fn detect_projection(image: &RawImage) -> PixelWidthDetection {
    let projections = EdgeProjections::new(image);
    detect_projection_from_signals(image, &projections)
}

fn detect_projection_from_signals(
    image: &RawImage,
    projections: &EdgeProjections,
) -> PixelWidthDetection {
    if image.width < 4 || image.height < 4 {
        return PixelWidthDetection {
            width: 1,
            width_x: 1,
            width_y: 1,
            offset_x: 0,
            offset_y: 0,
            source: PixelWidthSource::Projection,
            anchor_lines_x: None,
            anchor_lines_y: None,
        };
    }
    let x = best_spacing(&projections.x, image.width);
    let y = best_spacing(&projections.y, image.height);
    let width = resolve_axis_widths(x, y);
    PixelWidthDetection {
        width,
        width_x: x,
        width_y: y,
        offset_x: best_offset(&projections.x, width),
        offset_y: best_offset(&projections.y, width),
        source: PixelWidthSource::Projection,
        anchor_lines_x: None,
        anchor_lines_y: None,
    }
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

fn best_spacing(signal: &[u32], length: u32) -> u32 {
    recover_fundamental_lag(signal, dominant_lag(signal, length))
}

fn dominant_lag(signal: &[u32], length: u32) -> u32 {
    let upper = PROJECTION_MAX_LAG.min(length).max(2);
    (2..upper)
        .into_par_iter()
        .map(|spacing| (spacing, autocorrelation_score(signal, spacing as usize)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(spacing, _)| spacing)
        .unwrap_or(1)
}

fn resolve_axis_widths(x: u32, y: u32) -> u32 {
    if x == 0 {
        return y.max(1);
    }
    if y == 0 {
        return x.max(1);
    }
    if widths_agree(x, y) {
        return ((x + y) as f32 / 2.0).round().max(1.0) as u32;
    }
    if detect_integer_multiple(x.max(y), x.min(y)).is_some() {
        return x.min(y).max(1);
    }
    x.max(y).max(1)
}

fn detect_integer_multiple(larger: u32, smaller: u32) -> Option<u32> {
    if larger <= smaller || smaller == 0 {
        return None;
    }
    for factor in 2..=4 {
        let scaled = smaller * factor;
        let error = larger.abs_diff(scaled) as f32 / smaller as f32;
        if error <= INTEGER_MULTIPLE_TOLERANCE {
            return Some(factor);
        }
    }
    None
}

fn widths_agree(left: u32, right: u32) -> bool {
    let low = left.min(right) as f32;
    let high = left.max(right) as f32;
    high <= low * (1.0 + PROJECTION_AGREEMENT_TOLERANCE)
}

fn recover_fundamental_lag(signal: &[u32], harmonic_lag: u32) -> u32 {
    if harmonic_lag <= 2 {
        return harmonic_lag;
    }

    let harmonic_score = autocorrelation_score(signal, harmonic_lag as usize);
    let harmonic_occupancy = periodic_peak_occupancy(signal, harmonic_lag);

    for candidate in divisors(harmonic_lag).into_iter().rev() {
        let candidate_score = autocorrelation_score(signal, candidate as usize);
        let candidate_occupancy = periodic_peak_occupancy(signal, candidate);
        if candidate_score >= harmonic_score * MIN_AUTOCORRELATION_SCORE_RATIO
            && candidate_occupancy >= harmonic_occupancy * MIN_PEAK_OCCUPANCY_RATIO
            && candidate_occupancy >= MIN_PEAK_OCCUPANCY_RATIO
        {
            return candidate;
        }
    }
    harmonic_lag
}

fn divisors(value: u32) -> Vec<u32> {
    (2..value)
        .filter(|divisor| value.is_multiple_of(*divisor))
        .collect()
}

fn autocorrelation_score(signal: &[u32], lag: usize) -> f64 {
    if lag <= 1 || lag >= signal.len() {
        return f64::NEG_INFINITY;
    }
    let mean = mean(signal);
    signal
        .iter()
        .zip(signal.iter().skip(lag))
        .map(|(left, right)| (*left as f64 - mean) * (*right as f64 - mean))
        .sum()
}

fn periodic_peak_occupancy(signal: &[u32], spacing: u32) -> f32 {
    if spacing <= 1 || signal.len() < 3 {
        return 1.0;
    }
    let offset = best_periodic_offset(signal, spacing);
    let peak_threshold = percentile(signal, 0.8).max(mean(signal));
    let mut total_samples = 0;
    let mut matched_samples = 0;
    let mut index = offset;
    while index < signal.len() {
        if index > 0 && index + 1 < signal.len() {
            total_samples += 1;
            let local_peak = signal[index - 1].max(signal[index]).max(signal[index + 1]);
            if local_peak as f64 >= peak_threshold {
                matched_samples += 1;
            }
        }
        index += spacing as usize;
    }
    if total_samples == 0 {
        0.0
    } else {
        matched_samples as f32 / total_samples as f32
    }
}

fn best_periodic_offset(signal: &[u32], spacing: u32) -> usize {
    let spacing = spacing.max(1) as usize;
    (0..spacing)
        .max_by_key(|offset| {
            let mut score = 0u64;
            let mut index = *offset;
            while index < signal.len() {
                score += signal[index] as u64;
                index += spacing;
            }
            score
        })
        .unwrap_or(0)
}

fn best_offset(signal: &[u32], spacing: u32) -> u32 {
    best_periodic_offset(signal, spacing.max(1)) as u32
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
    use crate::image::RawImage;

    fn fixture(name: &str) -> RawImage {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/sources")
            .join(name);
        let bytes = std::fs::read(path).unwrap();
        crate::image::decode_image(&bytes).unwrap()
    }

    #[test]
    fn detects_manual_like_striped_spacing() {
        let mut image = RawImage::transparent(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                let value = if x % 4 == 0 || y % 4 == 0 { 255 } else { 0 };
                image.set_pixel(x, y, [value, value, value, 255]);
            }
        }
        assert!(detect_projection_width(&image) >= 2);
    }

    #[test]
    fn recovers_pumpkin_fundamental_instead_of_larger_harmonic() {
        let image = fixture("pumpkin.png");
        let detection = detect_pixel_width(&image, PixelWidthDetector::Hybrid);
        assert!(detection.width <= 20, "width was {}", detection.width);
    }

    #[test]
    fn rejects_noisy_low_confidence_hough_for_jpeg_like_fixtures() {
        for name in ["tiles.png", "smw.jpg", "pumpkin.png"] {
            let image = fixture(name);
            let detection = detect_pixel_width(&image, PixelWidthDetector::Hybrid);
            assert_eq!(
                detection.source,
                PixelWidthSource::Hybrid,
                "{name} should not use noisy hough anchors"
            );
            assert!(detection.anchor_lines_x.is_none());
            assert!(detection.anchor_lines_y.is_none());
        }
    }

    #[test]
    fn explicit_projection_detector_keeps_projection_source() {
        let image = fixture("tiles.png");
        let detection = detect_pixel_width(&image, PixelWidthDetector::Projection);

        assert_eq!(detection.source, PixelWidthSource::Projection);
        assert!(detection.anchor_lines_x.is_none());
        assert!(detection.anchor_lines_y.is_none());
    }

    #[test]
    fn keeps_high_confidence_hough_for_text_fixture() {
        let image = fixture("dragon_tavern_3.png");
        let detection = detect_pixel_width(&image, PixelWidthDetector::Hybrid);

        assert_eq!(detection.source, PixelWidthSource::Hough);
        assert!(
            detection.width >= 7,
            "width collapsed below anchor spacing: {}",
            detection.width
        );
        assert!(
            detection
                .anchor_lines_x
                .as_ref()
                .is_some_and(|lines| lines.len() > 8)
        );
        assert!(
            detection
                .anchor_lines_y
                .as_ref()
                .is_some_and(|lines| lines.len() > 8)
        );
    }
}
