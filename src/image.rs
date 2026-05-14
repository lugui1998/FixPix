use std::io::Cursor;

use anyhow::{Result, bail};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::webp::WebPEncoder;
use image::{ColorType, ImageEncoder};
use rayon::prelude::*;

use crate::core::OutputFormat;

pub const ALPHA_THRESHOLD: u8 = 128;
const BOUNDARY_BACKGROUND_TRANSPARENCY_DISTANCE_LIMIT: i32 = 8000;
const BOUNDARY_BACKGROUND_FRINGE_DISTANCE_LIMIT: i32 = 36000;
const BOUNDARY_BACKGROUND_FRINGE_PASSES: usize = 2;
const VIVID_BACKGROUND_CHROMA_MIN: u8 = 48;
const VIVID_BACKGROUND_VALUE_MIN: u8 = 96;
const LOCAL_BACKGROUND_SUPPORT_RADIUS: u32 = 2;
const LOCAL_BACKGROUND_SUPPORT_REJECT_DISTANCE_LIMIT: i32 = 400;
const LOCAL_SIMILAR_COLOR_DISTANCE_LIMIT: i32 = 3500;
const LOCAL_SIMILAR_SUPPORT_MIN: u32 = 3;
const LOCAL_OPPOSING_SUPPORT_MARGIN: u32 = 3;
const LOCAL_STRONG_FOREGROUND_SUPPORT_MIN: u32 = 3;
const DOWNSCALE_MIN_OPAQUE_COVERAGE: f32 = 0.06;
const EDGE_MASK_COLOR_DELTA_THRESHOLD: u32 = 80;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitInsideBounds {
    pub width: u32,
    pub height: u32,
    pub offset_x: u32,
    pub offset_y: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundMask {
    pub width: u32,
    pub height: u32,
    data: Vec<u8>,
}

impl BackgroundMask {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; width as usize * height as usize],
        }
    }

    pub fn is_background(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.data[(y * self.width + x) as usize] != 0
    }

    fn set_background(&mut self, x: u32, y: u32) {
        self.data[(y * self.width + x) as usize] = 1;
    }
}

impl RawImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), width as usize * height as usize * 4);
        Self {
            data,
            width,
            height,
        }
    }

    pub fn transparent(width: u32, height: u32) -> Self {
        Self {
            data: vec![0; width as usize * height as usize * 4],
            width,
            height,
        }
    }

    #[inline]
    pub fn offset(&self, x: u32, y: u32) -> usize {
        ((y * self.width + x) * 4) as usize
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = self.offset(x, y);
        [
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
            self.data[offset + 3],
        ]
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        let offset = self.offset(x, y);
        self.data[offset..offset + 4].copy_from_slice(&rgba);
    }
}

pub fn decode_image(bytes: &[u8]) -> Result<RawImage> {
    let image = image::load_from_memory(bytes)?.to_rgba8();
    let (width, height) = image.dimensions();
    Ok(RawImage::new(width, height, image.into_raw()))
}

pub fn encode_image(
    image: &RawImage,
    format: OutputFormat,
    quality: Option<u8>,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    match format {
        OutputFormat::Png => {
            PngEncoder::new(&mut bytes).write_image(
                &image.data,
                image.width,
                image.height,
                ColorType::Rgba8.into(),
            )?;
        }
        OutputFormat::Jpeg => {
            let mut rgb = Vec::with_capacity(image.width as usize * image.height as usize * 3);
            for pixel in image.data.chunks_exact(4) {
                let alpha = pixel[3] as u16;
                let inv = 255 - alpha;
                rgb.push(((pixel[0] as u16 * alpha + 255 * inv) / 255) as u8);
                rgb.push(((pixel[1] as u16 * alpha + 255 * inv) / 255) as u8);
                rgb.push(((pixel[2] as u16 * alpha + 255 * inv) / 255) as u8);
            }
            JpegEncoder::new_with_quality(&mut bytes, quality.unwrap_or(90)).encode(
                &rgb,
                image.width,
                image.height,
                ColorType::Rgb8.into(),
            )?;
        }
        OutputFormat::Webp => {
            if quality.is_some() {
                bail!("webp quality is not supported by the pure Rust lossless encoder");
            }
            WebPEncoder::new_lossless(&mut bytes).write_image(
                &image.data,
                image.width,
                image.height,
                ColorType::Rgba8.into(),
            )?;
        }
    }
    Ok(bytes)
}

pub fn clear_fully_transparent_pixels(image: &RawImage) -> RawImage {
    let mut out = image.clone();
    out.data.par_chunks_exact_mut(4).for_each(|pixel| {
        if pixel[3] == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        }
    });
    out
}

pub fn crop_transparent_padding(image: &RawImage) -> RawImage {
    let Some((left, top, right, bottom)) = opaque_bounds(image) else {
        return image.clone();
    };
    copy_bounds(image, left, top, right - left + 1, bottom - top + 1)
}

pub fn center_transparent_content(image: &RawImage, width: u32, height: u32) -> Result<RawImage> {
    let Some((left, top, right, bottom)) = opaque_bounds(image) else {
        return Ok(RawImage::transparent(width, height));
    };
    let content_width = right - left + 1;
    let content_height = bottom - top + 1;
    if content_width > width || content_height > height {
        bail!("crop-size is smaller than the transparent content");
    }
    let cropped = copy_bounds(image, left, top, content_width, content_height);
    let mut out = RawImage::transparent(width, height);
    let offset_x = (width - content_width) / 2;
    let offset_y = (height - content_height) / 2;
    blit(&mut out, &cropped, offset_x, offset_y);
    Ok(out)
}

pub fn scale_nearest(image: &RawImage, scale: u32) -> RawImage {
    if scale <= 1 {
        return image.clone();
    }
    let width = image.width * scale;
    let height = image.height * scale;
    let mut out = vec![0; width as usize * height as usize * 4];
    let scale = scale as usize;
    let source_width = image.width as usize;
    let source_row_bytes = source_width * 4;
    out.par_chunks_exact_mut(width as usize * 4)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y = target_y / scale;
            let source_row_start = source_y * source_row_bytes;
            let source_row = &image.data[source_row_start..source_row_start + source_row_bytes];
            for source_x in 0..source_width {
                let source = source_x * 4;
                let target_start = source_x * scale * 4;
                for repeat_x in 0..scale {
                    let target = target_start + repeat_x * 4;
                    row[target..target + 4].copy_from_slice(&source_row[source..source + 4]);
                }
            }
        });
    RawImage::new(width, height, out)
}

pub fn fit_image_inside_dimensions(image: &RawImage, width: u32, height: u32) -> FitInsideBounds {
    let fit_scale =
        (width as f32 / image.width.max(1) as f32).min(height as f32 / image.height.max(1) as f32);
    let fit_width = width.min((image.width as f32 * fit_scale).round().max(1.0) as u32);
    let fit_height = height.min((image.height as f32 * fit_scale).round().max(1.0) as u32);
    FitInsideBounds {
        width: fit_width,
        height: fit_height,
        offset_x: (width - fit_width) / 2,
        offset_y: (height - fit_height) / 2,
    }
}

pub fn downscale_ignoring_transparent(image: &RawImage, width: u32, height: u32) -> RawImage {
    let fit = fit_image_inside_dimensions(image, width, height);
    let mut out = RawImage::transparent(width, height);
    let scale_x = image.width as f32 / fit.width.max(1) as f32;
    let scale_y = image.height as f32 / fit.height.max(1) as f32;

    out.data
        .par_chunks_exact_mut(width as usize * 4)
        .enumerate()
        .for_each(|(target_y, row)| {
            if target_y < fit.offset_y as usize || target_y >= (fit.offset_y + fit.height) as usize
            {
                return;
            }
            let local_y = target_y as u32 - fit.offset_y;
            let source_top = local_y as f32 * scale_y;
            let source_bottom = (local_y + 1) as f32 * scale_y;
            for target_x in fit.offset_x..fit.offset_x + fit.width {
                let local_x = target_x - fit.offset_x;
                let source_left = local_x as f32 * scale_x;
                let source_right = (local_x + 1) as f32 * scale_x;
                let color = average_opaque_area(
                    image,
                    source_left,
                    source_right,
                    source_top,
                    source_bottom,
                );
                let target = target_x as usize * 4;
                row[target..target + 4].copy_from_slice(&color);
            }
        });
    out
}

pub fn make_boundary_background_transparent(image: &RawImage) -> RawImage {
    let Some(background) = boundary_background_color(image) else {
        return image.clone();
    };
    make_boundary_background_transparent_with_color(image, &background)
}

pub fn make_boundary_background_transparent_with_edge_closing(
    image: &RawImage,
    edge_close_kernel_size: u32,
) -> RawImage {
    let Some(background) = boundary_background_color(image) else {
        return image.clone();
    };
    make_boundary_background_transparent_with_color_and_edge_closing(
        image,
        &background,
        edge_close_kernel_size,
    )
}

pub fn boundary_background_color(image: &RawImage) -> Option<[u8; 4]> {
    (!mostly_transparent_boundary(image)).then(|| most_common_boundary_color(image))
}

pub fn make_boundary_background_transparent_with_color(
    image: &RawImage,
    background: &[u8; 4],
) -> RawImage {
    make_boundary_background_transparent_with_color_and_edge_closing(image, background, 0)
}

pub fn make_boundary_background_transparent_with_color_and_edge_closing(
    image: &RawImage,
    background: &[u8; 4],
    edge_close_kernel_size: u32,
) -> RawImage {
    let mask = boundary_background_mask_with_color_and_edge_closing(
        image,
        background,
        edge_close_kernel_size,
    );
    let mut out = apply_background_mask(image, &mask);
    remove_boundary_background_fringe(&mut out, background);
    out
}

pub fn boundary_background_mask_with_color(
    image: &RawImage,
    background: &[u8; 4],
) -> BackgroundMask {
    boundary_background_mask_with_color_and_edge_closing(image, background, 0)
}

pub fn boundary_background_mask_with_color_and_edge_closing(
    image: &RawImage,
    background: &[u8; 4],
    edge_close_kernel_size: u32,
) -> BackgroundMask {
    let mut mask = BackgroundMask::new(image.width, image.height);
    let mut queue = std::collections::VecDeque::new();
    let mut visited = vec![false; image.width as usize * image.height as usize];
    let edge_barrier =
        (edge_close_kernel_size > 1).then(|| closed_color_edge_mask(image, edge_close_kernel_size));
    let background_is_vivid = has_vivid_chroma(background);

    for x in 0..image.width {
        queue.push_back((x, 0));
        queue.push_back((x, image.height - 1));
    }
    for y in 1..image.height.saturating_sub(1) {
        queue.push_back((0, y));
        queue.push_back((image.width - 1, y));
    }

    while let Some((x, y)) = queue.pop_front() {
        if x >= image.width || y >= image.height {
            continue;
        }
        let index = (y * image.width + x) as usize;
        if visited[index] {
            continue;
        }
        visited[index] = true;
        let pixel = image.pixel(x, y);
        let distance = color_distance_sq(&pixel, background);
        if pixel[3] < ALPHA_THRESHOLD || distance <= BOUNDARY_BACKGROUND_TRANSPARENCY_DISTANCE_LIMIT
        {
            let is_edge_barrier = edge_barrier
                .as_ref()
                .is_some_and(|barrier| barrier[index] != 0 && !is_boundary_pixel(image, x, y));
            if is_edge_barrier && pixel[3] >= ALPHA_THRESHOLD {
                let support = local_foreground_support(Some(&mask), image, background, pixel, x, y);
                if local_support_keeps_background_candidate(
                    pixel,
                    background,
                    background_is_vivid,
                    support,
                ) {
                    continue;
                }
            }
            mask.set_background(x, y);
            if is_edge_barrier {
                continue;
            }
            if x > 0 {
                queue.push_back((x - 1, y));
            }
            if y > 0 {
                queue.push_back((x, y - 1));
            }
            if x + 1 < image.width {
                queue.push_back((x + 1, y));
            }
            if y + 1 < image.height {
                queue.push_back((x, y + 1));
            }
        }
    }
    refine_boundary_background_mask(&mut mask, image, background, edge_barrier.as_deref());
    mask
}

pub(crate) fn color_edge_mask(image: &RawImage) -> Vec<u8> {
    let mut mask = vec![0; image.width as usize * image.height as usize];
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
            let edge = rgba_delta(current, right).max(rgba_delta(current, down));
            if edge > EDGE_MASK_COLOR_DELTA_THRESHOLD {
                mask[(y * image.width + x) as usize] = 1;
            }
        }
    }
    mask
}

pub(crate) fn closed_color_edge_mask(image: &RawImage, kernel_size: u32) -> Vec<u8> {
    close_binary_mask(
        image.width,
        image.height,
        &color_edge_mask(image),
        kernel_size,
    )
}

pub(crate) fn close_binary_mask(width: u32, height: u32, data: &[u8], kernel_size: u32) -> Vec<u8> {
    debug_assert_eq!(data.len(), width as usize * height as usize);
    if kernel_size <= 1 || width == 0 || height == 0 {
        return data.to_vec();
    }
    let radius = kernel_size / 2;
    let dilated = dilate_binary_mask(width, height, data, radius);
    erode_binary_mask(width, height, &dilated, radius)
}

fn dilate_binary_mask(width: u32, height: u32, data: &[u8], radius: u32) -> Vec<u8> {
    let mut out = vec![0; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let (x0, x1, y0, y1) = kernel_bounds(width, height, x, y, radius);
            let mut found = false;
            'scan: for yy in y0..=y1 {
                for xx in x0..=x1 {
                    if data[(yy * width + xx) as usize] != 0 {
                        found = true;
                        break 'scan;
                    }
                }
            }
            if found {
                out[(y * width + x) as usize] = 1;
            }
        }
    }
    out
}

fn erode_binary_mask(width: u32, height: u32, data: &[u8], radius: u32) -> Vec<u8> {
    let mut out = vec![0; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let (x0, x1, y0, y1) = kernel_bounds(width, height, x, y, radius);
            let mut all_set = true;
            'scan: for yy in y0..=y1 {
                for xx in x0..=x1 {
                    if data[(yy * width + xx) as usize] == 0 {
                        all_set = false;
                        break 'scan;
                    }
                }
            }
            if all_set {
                out[(y * width + x) as usize] = 1;
            }
        }
    }
    out
}

fn kernel_bounds(width: u32, height: u32, x: u32, y: u32, radius: u32) -> (u32, u32, u32, u32) {
    (
        x.saturating_sub(radius),
        x.saturating_add(radius).min(width.saturating_sub(1)),
        y.saturating_sub(radius),
        y.saturating_add(radius).min(height.saturating_sub(1)),
    )
}

fn is_boundary_pixel(image: &RawImage, x: u32, y: u32) -> bool {
    x == 0 || y == 0 || x + 1 == image.width || y + 1 == image.height
}

fn refine_boundary_background_mask(
    mask: &mut BackgroundMask,
    image: &RawImage,
    background: &[u8; 4],
    edge_barrier: Option<&[u8]>,
) {
    let background_is_vivid = has_vivid_chroma(background);
    for _ in 0..BOUNDARY_BACKGROUND_FRINGE_PASSES {
        let mut fringe = Vec::new();
        for y in 0..image.height {
            for x in 0..image.width {
                if !is_boundary_background_fringe_candidate(
                    mask,
                    image,
                    background,
                    edge_barrier,
                    background_is_vivid,
                    x,
                    y,
                ) {
                    continue;
                }
                fringe.push((x, y));
            }
        }

        if fringe.is_empty() {
            break;
        }

        for (x, y) in fringe {
            mask.set_background(x, y);
        }
    }
}

fn is_boundary_background_fringe_candidate(
    mask: &BackgroundMask,
    image: &RawImage,
    background: &[u8; 4],
    edge_barrier: Option<&[u8]>,
    background_is_vivid: bool,
    x: u32,
    y: u32,
) -> bool {
    if mask.is_background(x, y) || !touches_background_cardinal_neighbor(mask, x, y) {
        return false;
    }

    let pixel = image.pixel(x, y);
    if pixel[3] < ALPHA_THRESHOLD
        || color_distance_sq(&pixel, background) > BOUNDARY_BACKGROUND_FRINGE_DISTANCE_LIMIT
    {
        return false;
    }
    if color_distance_sq(&pixel, background) == 0 {
        return false;
    }

    let index = (y * image.width + x) as usize;
    let is_edge_barrier =
        edge_barrier.is_some_and(|barrier| barrier[index] != 0 && !is_boundary_pixel(image, x, y));
    let support = local_foreground_support(Some(mask), image, background, pixel, x, y);
    let keep_candidate =
        local_support_keeps_background_candidate(pixel, background, background_is_vivid, support);
    if is_edge_barrier {
        !keep_candidate
    } else if background_is_vivid && shares_background_dominant_channel(pixel, *background) {
        !keep_candidate
    } else {
        true
    }
}

fn touches_background_cardinal_neighbor(mask: &BackgroundMask, x: u32, y: u32) -> bool {
    (x > 0 && mask.is_background(x - 1, y))
        || (y > 0 && mask.is_background(x, y - 1))
        || (x + 1 < mask.width && mask.is_background(x + 1, y))
        || (y + 1 < mask.height && mask.is_background(x, y + 1))
}

#[derive(Debug, Clone, Copy, Default)]
struct LocalForegroundSupport {
    similar: u32,
    strong: u32,
    opposing: u32,
}

fn local_foreground_support(
    mask: Option<&BackgroundMask>,
    image: &RawImage,
    background: &[u8; 4],
    candidate: [u8; 4],
    x: u32,
    y: u32,
) -> LocalForegroundSupport {
    let x0 = x.saturating_sub(LOCAL_BACKGROUND_SUPPORT_RADIUS);
    let x1 = x
        .saturating_add(LOCAL_BACKGROUND_SUPPORT_RADIUS)
        .min(image.width.saturating_sub(1));
    let y0 = y.saturating_sub(LOCAL_BACKGROUND_SUPPORT_RADIUS);
    let y1 = y
        .saturating_add(LOCAL_BACKGROUND_SUPPORT_RADIUS)
        .min(image.height.saturating_sub(1));
    let mut support = LocalForegroundSupport::default();
    for yy in y0..=y1 {
        for xx in x0..=x1 {
            if xx == x && yy == y {
                continue;
            }
            let pixel = image.pixel(xx, yy);
            if pixel[3] < ALPHA_THRESHOLD || mask.is_some_and(|mask| mask.is_background(xx, yy)) {
                continue;
            }
            if color_distance_sq(&pixel, background)
                <= LOCAL_BACKGROUND_SUPPORT_REJECT_DISTANCE_LIMIT
            {
                continue;
            }
            let weight = local_support_weight(x.abs_diff(xx), y.abs_diff(yy));
            if color_distance_sq(&pixel, &candidate) <= LOCAL_SIMILAR_COLOR_DISTANCE_LIMIT {
                support.similar += weight;
            }
            if !shares_background_dominant_channel(pixel, *background)
                && color_distance_sq(&pixel, background)
                    > BOUNDARY_BACKGROUND_TRANSPARENCY_DISTANCE_LIMIT
            {
                support.opposing += weight;
            }
            if color_distance_sq(&pixel, background) > BOUNDARY_BACKGROUND_FRINGE_DISTANCE_LIMIT {
                support.strong += weight;
            }
        }
    }
    support
}

fn local_support_keeps_background_candidate(
    candidate: [u8; 4],
    background: &[u8; 4],
    background_is_vivid: bool,
    support: LocalForegroundSupport,
) -> bool {
    if color_distance_sq(&candidate, background) == 0 {
        return false;
    }
    let has_similar = support.similar >= LOCAL_SIMILAR_SUPPORT_MIN;
    let has_strong = support.strong >= LOCAL_STRONG_FOREGROUND_SUPPORT_MIN;
    if background_is_vivid && shares_background_dominant_channel(candidate, *background) {
        let opposing_blocks = support.opposing >= support.similar + LOCAL_OPPOSING_SUPPORT_MARGIN;
        (has_similar && !opposing_blocks) || (!has_strong && !opposing_blocks)
    } else {
        has_similar || has_strong
    }
}

fn local_support_weight(dx: u32, dy: u32) -> u32 {
    if dx <= 1 && dy <= 1 { 3 } else { 1 }
}

fn has_vivid_chroma(pixel: &[u8; 4]) -> bool {
    let max = pixel[0].max(pixel[1]).max(pixel[2]);
    let min = pixel[0].min(pixel[1]).min(pixel[2]);
    max >= VIVID_BACKGROUND_VALUE_MIN && max - min >= VIVID_BACKGROUND_CHROMA_MIN
}

fn shares_background_dominant_channel(pixel: [u8; 4], background: [u8; 4]) -> bool {
    let pixel_max = pixel[0].max(pixel[1]).max(pixel[2]);
    let background_max = background[0].max(background[1]).max(background[2]);
    (0..3).any(|channel| {
        pixel[channel].saturating_add(8) >= pixel_max
            && background[channel].saturating_add(8) >= background_max
    })
}

pub fn apply_background_mask(image: &RawImage, mask: &BackgroundMask) -> RawImage {
    debug_assert_eq!((image.width, image.height), (mask.width, mask.height));
    let mut out = image.clone();
    for y in 0..image.height {
        for x in 0..image.width {
            if mask.is_background(x, y) {
                out.set_pixel(x, y, [0, 0, 0, 0]);
            }
        }
    }
    out
}

pub fn compose_grid(images: &[RawImage], columns: u32) -> RawImage {
    if images.is_empty() {
        return RawImage::transparent(1, 1);
    }
    let gap = 12;
    let columns = columns.max(1);
    let rows = (images.len() as u32).div_ceil(columns);
    let cell_width = images.iter().map(|image| image.width).max().unwrap_or(1);
    let cell_height = images.iter().map(|image| image.height).max().unwrap_or(1);
    let mut out = RawImage::new(
        columns * cell_width + (columns + 1) * gap,
        rows * cell_height + (rows + 1) * gap,
        vec![
            245;
            ((columns * cell_width + (columns + 1) * gap)
                * (rows * cell_height + (rows + 1) * gap)
                * 4) as usize
        ],
    );
    for (index, image) in images.iter().enumerate() {
        let column = index as u32 % columns;
        let row = index as u32 / columns;
        let x = gap + column * (cell_width + gap) + (cell_width - image.width) / 2;
        let y = gap + row * (cell_height + gap) + (cell_height - image.height) / 2;
        blit(&mut out, image, x, y);
    }
    out
}

pub fn blit_image(base: &RawImage, overlay: &RawImage, offset_x: i32, offset_y: i32) -> RawImage {
    let mut out = base.clone();
    for y in 0..overlay.height {
        for x in 0..overlay.width {
            let target_x = offset_x + x as i32;
            let target_y = offset_y + y as i32;
            if target_x < 0
                || target_y < 0
                || target_x >= base.width as i32
                || target_y >= base.height as i32
            {
                continue;
            }

            let src = overlay.pixel(x, y);
            let dst = out.pixel(target_x as u32, target_y as u32);
            let alpha = src[3] as f32 / 255.0;
            let inv_alpha = 1.0 - alpha;
            out.set_pixel(
                target_x as u32,
                target_y as u32,
                [
                    (src[0] as f32 * alpha + dst[0] as f32 * inv_alpha).round() as u8,
                    (src[1] as f32 * alpha + dst[1] as f32 * inv_alpha).round() as u8,
                    (src[2] as f32 * alpha + dst[2] as f32 * inv_alpha).round() as u8,
                    255,
                ],
            );
        }
    }
    out
}

pub fn choose_closest_integer_scale(
    image: &RawImage,
    target_width: u32,
    target_height: u32,
) -> u32 {
    if image.width == 0 || image.height == 0 {
        return 1;
    }

    let ratio_x = target_width as f32 / image.width as f32;
    let ratio_y = target_height as f32 / image.height as f32;
    let max_candidate = ratio_x.max(ratio_y).ceil().max(1.0) as u32 + 1;

    let mut best_scale = 1;
    let mut best_error = u32::MAX;
    for scale in 1..=max_candidate {
        let error = image.width.saturating_mul(scale).abs_diff(target_width)
            + image.height.saturating_mul(scale).abs_diff(target_height);
        if error < best_error || (error == best_error && scale > best_scale) {
            best_error = error;
            best_scale = scale;
        }
    }
    best_scale
}

pub fn limit_scale_for_max_dimension(image: &RawImage, requested_scale: u32) -> u32 {
    let max_image_dimension = image.width.max(image.height);
    if max_image_dimension == 0 {
        return 1;
    }
    let max_allowed_scale = (2048 / max_image_dimension).max(1);
    requested_scale.max(1).min(max_allowed_scale)
}

fn opaque_bounds(image: &RawImage) -> Option<(u32, u32, u32, u32)> {
    let mut left = image.width;
    let mut top = image.height;
    let mut right = 0;
    let mut bottom = 0;
    let mut found = false;
    for y in 0..image.height {
        for x in 0..image.width {
            if image.pixel(x, y)[3] >= ALPHA_THRESHOLD {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x);
                bottom = bottom.max(y);
                found = true;
            }
        }
    }
    found.then_some((left, top, right, bottom))
}

fn copy_bounds(image: &RawImage, left: u32, top: u32, width: u32, height: u32) -> RawImage {
    let mut out = RawImage::transparent(width, height);
    for y in 0..height {
        let source = image.offset(left, top + y);
        let target = out.offset(0, y);
        let bytes = width as usize * 4;
        out.data[target..target + bytes].copy_from_slice(&image.data[source..source + bytes]);
    }
    out
}

fn blit(target: &mut RawImage, source: &RawImage, x: u32, y: u32) {
    for row in 0..source.height {
        if y + row >= target.height {
            break;
        }
        let source_offset = source.offset(0, row);
        let target_offset = target.offset(x, y + row);
        let width = source.width.min(target.width.saturating_sub(x)) as usize * 4;
        target.data[target_offset..target_offset + width]
            .copy_from_slice(&source.data[source_offset..source_offset + width]);
    }
}

fn average_opaque_area(
    image: &RawImage,
    source_left: f32,
    source_right: f32,
    source_top: f32,
    source_bottom: f32,
) -> [u8; 4] {
    let start_x = source_left.floor().max(0.0) as u32;
    let end_x = source_right.ceil().min(image.width as f32) as u32;
    let start_y = source_top.floor().max(0.0) as u32;
    let end_y = source_bottom.ceil().min(image.height as f32) as u32;
    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;
    let mut opaque_area = 0.0f32;
    let mut total_area = 0.0f32;

    for y in start_y..end_y {
        let y_overlap = ((y + 1) as f32).min(source_bottom) - (y as f32).max(source_top);
        if y_overlap <= 0.0 {
            continue;
        }
        for x in start_x..end_x {
            let x_overlap = ((x + 1) as f32).min(source_right) - (x as f32).max(source_left);
            if x_overlap <= 0.0 {
                continue;
            }
            let area = x_overlap * y_overlap;
            let pixel = image.pixel(x, y);
            total_area += area;
            if pixel[3] >= ALPHA_THRESHOLD {
                opaque_area += area;
                r += pixel[0] as f32 * area;
                g += pixel[1] as f32 * area;
                b += pixel[2] as f32 * area;
            }
        }
    }

    if opaque_area <= 0.0
        || opaque_area / total_area.max(f32::EPSILON) < DOWNSCALE_MIN_OPAQUE_COVERAGE
    {
        [0, 0, 0, 0]
    } else {
        [
            (r / opaque_area).round() as u8,
            (g / opaque_area).round() as u8,
            (b / opaque_area).round() as u8,
            255,
        ]
    }
}

fn mostly_transparent_boundary(image: &RawImage) -> bool {
    let mut transparent = 0u32;
    let mut total = 0u32;
    for x in 0..image.width {
        for y in [0, image.height - 1] {
            total += 1;
            transparent += u32::from(image.pixel(x, y)[3] < ALPHA_THRESHOLD);
        }
    }
    for y in 1..image.height.saturating_sub(1) {
        for x in [0, image.width - 1] {
            total += 1;
            transparent += u32::from(image.pixel(x, y)[3] < ALPHA_THRESHOLD);
        }
    }
    total > 0 && transparent as f32 / total as f32 >= 0.6
}

fn remove_boundary_background_fringe(image: &mut RawImage, background: &[u8; 4]) {
    for _ in 0..BOUNDARY_BACKGROUND_FRINGE_PASSES {
        let mut fringe = Vec::new();
        for y in 0..image.height {
            for x in 0..image.width {
                let pixel = image.pixel(x, y);
                if pixel[3] < ALPHA_THRESHOLD
                    || color_distance_sq(&pixel, background)
                        > BOUNDARY_BACKGROUND_FRINGE_DISTANCE_LIMIT
                    || !touches_transparent_cardinal_neighbor(image, x, y)
                {
                    continue;
                }
                fringe.push((x, y));
            }
        }

        if fringe.is_empty() {
            break;
        }

        for (x, y) in fringe {
            image.set_pixel(x, y, [0, 0, 0, 0]);
        }
    }
}

fn touches_transparent_cardinal_neighbor(image: &RawImage, x: u32, y: u32) -> bool {
    (x > 0 && image.pixel(x - 1, y)[3] < ALPHA_THRESHOLD)
        || (y > 0 && image.pixel(x, y - 1)[3] < ALPHA_THRESHOLD)
        || (x + 1 < image.width && image.pixel(x + 1, y)[3] < ALPHA_THRESHOLD)
        || (y + 1 < image.height && image.pixel(x, y + 1)[3] < ALPHA_THRESHOLD)
}

fn most_common_boundary_color(image: &RawImage) -> [u8; 4] {
    let mut colors = std::collections::HashMap::<u32, u32>::new();
    let mut push = |pixel: [u8; 4]| {
        if pixel[3] >= ALPHA_THRESHOLD {
            let key = ((pixel[0] as u32) << 16) | ((pixel[1] as u32) << 8) | pixel[2] as u32;
            *colors.entry(key).or_default() += 1;
        }
    };
    for x in 0..image.width {
        push(image.pixel(x, 0));
        push(image.pixel(x, image.height - 1));
    }
    for y in 1..image.height.saturating_sub(1) {
        push(image.pixel(0, y));
        push(image.pixel(image.width - 1, y));
    }
    let key = colors
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(key, _)| key)
        .unwrap_or(0);
    [
        ((key >> 16) & 0xff) as u8,
        ((key >> 8) & 0xff) as u8,
        (key & 0xff) as u8,
        255,
    ]
}

pub(crate) fn color_distance_sq(pixel: &[u8; 4], background: &[u8; 4]) -> i32 {
    let dr = pixel[0] as i32 - background[0] as i32;
    let dg = pixel[1] as i32 - background[1] as i32;
    let db = pixel[2] as i32 - background[2] as i32;
    dr * dr + dg * dg + db * db
}

fn rgba_delta(left: [u8; 4], right: [u8; 4]) -> u32 {
    left[0].abs_diff(right[0]) as u32
        + left[1].abs_diff(right[1]) as u32
        + left[2].abs_diff(right[2]) as u32
        + left[3].abs_diff(right[3]) as u32
}

#[allow(dead_code)]
fn _decode_cursor(bytes: Vec<u8>) -> Cursor<Vec<u8>> {
    Cursor::new(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_nearest_repeats_pixels() {
        let image = RawImage::new(1, 2, vec![255, 0, 0, 255, 0, 0, 255, 255]);
        let scaled = scale_nearest(&image, 2);
        assert_eq!(scaled.width, 2);
        assert_eq!(scaled.height, 4);
        assert_eq!(scaled.pixel(0, 0), [255, 0, 0, 255]);
        assert_eq!(scaled.pixel(1, 1), [255, 0, 0, 255]);
        assert_eq!(scaled.pixel(0, 2), [0, 0, 255, 255]);
    }

    #[test]
    fn crop_transparent_padding_removes_empty_edges() {
        let mut image = RawImage::transparent(3, 3);
        image.set_pixel(1, 2, [10, 20, 30, 255]);
        let cropped = crop_transparent_padding(&image);
        assert_eq!((cropped.width, cropped.height), (1, 1));
        assert_eq!(cropped.pixel(0, 0), [10, 20, 30, 255]);
    }

    #[test]
    fn crop_transparent_padding_leaves_all_transparent_image_unchanged() {
        let image = RawImage::transparent(3, 2);
        let cropped = crop_transparent_padding(&image);
        assert_eq!((cropped.width, cropped.height), (3, 2));
        assert_eq!(cropped.data, image.data);
    }

    #[test]
    fn center_transparent_content_places_opaque_pixels_on_canvas() {
        let mut image = RawImage::transparent(3, 3);
        image.set_pixel(2, 2, [10, 20, 30, 255]);
        let centered = center_transparent_content(&image, 5, 5).unwrap();
        assert_eq!((centered.width, centered.height), (5, 5));
        assert_eq!(centered.pixel(2, 2), [10, 20, 30, 255]);
    }

    #[test]
    fn center_transparent_content_rejects_too_small_canvas() {
        let mut image = RawImage::transparent(3, 3);
        image.set_pixel(0, 0, [10, 20, 30, 255]);
        image.set_pixel(2, 2, [10, 20, 30, 255]);
        assert!(center_transparent_content(&image, 2, 2).is_err());
    }

    #[test]
    fn downscale_ignoring_transparent_fits_inside_target_canvas() {
        let mut image = RawImage::transparent(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                image.set_pixel(x, y, [100, 120, 140, 255]);
            }
        }
        let downscaled = downscale_ignoring_transparent(&image, 2, 2);
        assert_eq!((downscaled.width, downscaled.height), (2, 2));
        assert_eq!(downscaled.pixel(0, 0), [100, 120, 140, 255]);
        assert_eq!(downscaled.pixel(1, 0), [100, 120, 140, 255]);
    }

    #[test]
    fn downscale_ignores_transparent_pixels_instead_of_blending_with_them() {
        let mut image = RawImage::transparent(2, 2);
        image.set_pixel(0, 0, [255, 0, 0, 255]);

        let downscaled = downscale_ignoring_transparent(&image, 1, 1);

        assert_eq!(downscaled.pixel(0, 0), [255, 0, 0, 255]);
    }

    #[test]
    fn boundary_background_cleanup_removes_antialiased_background_fringe() {
        let mut image = RawImage::new(4, 3, [255, 0, 255, 255].repeat(12));
        image.set_pixel(1, 1, [150, 60, 150, 255]);
        image.set_pixel(2, 1, [133, 56, 133, 255]);
        image.set_pixel(3, 1, [40, 40, 40, 255]);

        image = make_boundary_background_transparent(&image);

        assert_eq!(image.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(image.pixel(1, 1), [0, 0, 0, 0]);
        assert_eq!(image.pixel(2, 1), [0, 0, 0, 0]);
        assert_eq!(image.pixel(3, 1), [40, 40, 40, 255]);
    }

    #[test]
    fn boundary_background_cleanup_keeps_non_background_edge_colors() {
        let mut image = RawImage::new(3, 3, [255, 0, 255, 255].repeat(9));
        image.set_pixel(1, 1, [183, 175, 180, 255]);

        image = make_boundary_background_transparent(&image);

        assert_eq!(image.pixel(1, 1), [183, 175, 180, 255]);
    }

    #[test]
    fn binary_closing_bridges_single_pixel_edge_gaps() {
        let mut mask = vec![0; 7 * 5];
        mask[2 * 7 + 2] = 1;
        mask[2 * 7 + 4] = 1;

        let closed = close_binary_mask(7, 5, &mask, 3);

        let mut expected = vec![0; 7 * 5];
        expected[2 * 7 + 2] = 1;
        expected[2 * 7 + 3] = 1;
        expected[2 * 7 + 4] = 1;
        assert_eq!(closed, expected);
    }

    #[test]
    fn closed_edge_barrier_preserves_black_details_inside_subject() {
        let black = [0, 0, 0, 255];
        let red = [255, 0, 0, 255];
        let mut image = RawImage::new(7, 7, black.repeat(49));
        for y in 1..=5 {
            for x in 1..=5 {
                if x == 1 || x == 5 || y == 1 || y == 5 {
                    image.set_pixel(x, y, red);
                }
            }
        }
        image.set_pixel(3, 1, black);

        let leaking = boundary_background_mask_with_color(&image, &black);
        let protected = boundary_background_mask_with_color_and_edge_closing(&image, &black, 3);

        assert!(leaking.is_background(3, 3));
        assert!(protected.is_background(3, 1));
        assert!(!protected.is_background(3, 3));
    }

    #[test]
    fn closed_edge_barrier_keeps_background_like_subject_outline() {
        let background = [0, 1, 21, 255];
        let outline = [0, 0, 0, 255];
        let red = [220, 0, 0, 255];
        let mut image = RawImage::new(5, 5, background.repeat(25));
        image.set_pixel(2, 1, outline);
        image.set_pixel(2, 2, red);

        let mask = boundary_background_mask_with_color_and_edge_closing(&image, &background, 3);

        assert!(!mask.is_background(2, 1));
        assert!(!mask.is_background(2, 2));
    }

    #[test]
    fn local_mask_refinement_keeps_dark_subject_detail_with_neighbor_support() {
        let background = [0, 1, 21, 255];
        let dark_detail = [0, 0, 2, 255];
        let red = [220, 0, 0, 255];
        let mut image = RawImage::new(5, 5, background.repeat(25));
        image.set_pixel(2, 1, dark_detail);
        image.set_pixel(2, 2, red);

        let mask = boundary_background_mask_with_color_and_edge_closing(&image, &background, 3);

        assert!(!mask.is_background(2, 1));
        assert!(!mask.is_background(2, 2));
    }

    #[test]
    fn mask_refinement_removes_vivid_background_edge_fringe() {
        let background = [103, 214, 79, 255];
        let fringe = [105, 172, 76, 255];
        let orange = [230, 80, 0, 255];
        let mut image = RawImage::new(5, 5, background.repeat(25));
        image.set_pixel(2, 1, fringe);
        image.set_pixel(2, 2, orange);

        let mask = boundary_background_mask_with_color_and_edge_closing(&image, &background, 3);

        assert!(mask.is_background(2, 1));
        assert!(!mask.is_background(2, 2));
    }

    #[test]
    fn mask_refinement_removes_green_fringe_in_warm_foreground_context() {
        let background = [103, 214, 79, 255];
        let fringe = [105, 172, 76, 255];
        let similar_fringe = [92, 136, 57, 255];
        let orange = [230, 80, 0, 255];
        let mut image = RawImage::new(5, 5, background.repeat(25));
        image.set_pixel(2, 1, fringe);
        image.set_pixel(3, 1, similar_fringe);
        image.set_pixel(1, 2, orange);
        image.set_pixel(2, 2, orange);
        image.set_pixel(3, 2, orange);

        let mask = boundary_background_mask_with_color_and_edge_closing(&image, &background, 3);

        assert!(mask.is_background(2, 1));
        assert!(mask.is_background(3, 1));
        assert!(!mask.is_background(2, 2));
    }

    #[test]
    fn mask_refinement_keeps_vivid_background_colored_subject_patch() {
        let background = [103, 214, 79, 255];
        let leaf_edge = [86, 138, 45, 255];
        let leaf_fill = [92, 135, 54, 255];
        let mut image = RawImage::new(5, 5, background.repeat(25));
        image.set_pixel(2, 1, leaf_edge);
        image.set_pixel(2, 2, leaf_fill);

        let mask = boundary_background_mask_with_color_and_edge_closing(&image, &background, 3);

        assert!(!mask.is_background(2, 1));
        assert!(!mask.is_background(2, 2));
    }

    #[test]
    fn encodes_lossless_webp_without_quality() {
        let image = RawImage::new(1, 1, vec![25, 50, 75, 255]);
        let encoded = encode_image(&image, OutputFormat::Webp, None).unwrap();
        let decoded = image::load_from_memory_with_format(&encoded, image::ImageFormat::WebP)
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded.into_raw(), image.data);
    }

    #[test]
    fn rejects_webp_quality_until_lossy_encoder_is_chosen() {
        let image = RawImage::new(1, 1, vec![25, 50, 75, 255]);
        assert!(encode_image(&image, OutputFormat::Webp, Some(80)).is_err());
    }
}
