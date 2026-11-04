//! Point clouds as CE draws them (`Camera.display_point_cloud`): each point a square block of whole
//! pixels, a producer of fragments that cover their pixels exactly, with depth.
//!
//! - Place: the point through the record's placement and CE's display placement, then CE's subpixel
//!   coordinates, truncated toward zero (`astype(int64)`; a point at -0.5 lands on pixel 0).
//! - Size: `t = int(1 + (radius - 1) * factor)` pixels a side, the factor 1 (the camera's size is
//!   the frame's); offsets `range(-t // 2 + 1, t // 2 + 1)` on each axis (for an even t one more on
//!   the high side); no block when t < 1. Pixels off the frame are dropped one by one.
//! - Colour: `trunc(255 * c)` per channel, alpha (times the group opacity) included, as CE writes
//!   its bytes; premultiplied by that alpha for compositing (an opaque point keeps CE's bytes).
//! - Ties: CE writes every point at the first offset, then every point at the next, and a later
//!   write wins; a block pixel's sub-order is its offset's index times the points, plus the point's.
//!   Depth decides between points at different depths; CE's raw overwrite of translucent points is
//!   not reproduced.

use crate::composite::{
    Coverage, FULL, Fragment, OpaquePass, Plane, Producer, Rect, TransparentPass,
};
use crate::project::{Projection, placement, pre_display, subpixel_coords};
use crate::tiles::{TILE, tile_index, tiles_across};
use crate::xf::*;

/// A `points` record to draw: its positions and colour rows.
pub struct PointsDraw<'a> {
    pub record: &'a Record,
    pub positions: &'a [[f64; 3]],
    pub colours: &'a [[f64; 4]],
    pub order: u32,
}

/// A block's side in pixels for CE's stroke width (`adjusted_thickness`, then `int`), or 0.
pub fn side(radius: f64) -> i64 {
    let factor = 1.0;
    let t = (1.0 + (radius - 1.0) * factor).trunc();
    if t.is_nan() || t < 1.0 {
        0
    } else {
        t.min(i64::MAX as f64) as i64
    }
}

/// One point's block, clipped to the frame: pixels `x0..x1` by `y0..y1`.
struct Block {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    /// The pixel of offset (0, 0) and the low offset, to find a pixel's offset.
    at: [i64; 2],
    lo: i64,
    side: i64,
    point: u32,
    points: u32,
    colour: [u8; 4],
    key: f64,
    order: u32,
}

impl Block {
    /// The sub-order of the block's write at pixel `(x, y)`: CE's offsets are
    /// `itertools.product(range, range)` over (x, y), every point at one offset before the next.
    fn sub(&self, x: u32, y: u32) -> u32 {
        let a = x as i64 - self.at[0] - self.lo;
        let b = y as i64 - self.at[1] - self.lo;
        let offset = (a * self.side + b) as u64;
        (offset * self.points as u64 + self.point as u64) as u32
    }
}

/// Point blocks, binned by tile.
pub struct Blocks {
    blocks: Vec<Block>,
    bins: Vec<Vec<u32>>,
    width: u32,
}

/// `trunc(255 * c)` of a channel in 0..1.
fn byte(c: f64) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0) as u8
}

/// Lowers the points of a frame of `width x height` pixels drawn from `view` to blocks.
pub fn lower(view: &View, width: u32, height: u32, draws: &[PointsDraw<'_>]) -> Blocks {
    let projection = match &view.camera {
        Camera::ThreeD(c) => Some(Projection::new(c, view.frame_center)),
        _ => None,
    };
    let mut out = Blocks {
        blocks: Vec::new(),
        bins: vec![Vec::new(); (tiles_across(width) * height.div_ceil(TILE)) as usize],
        width,
    };
    for draw in draws {
        let rec = draw.record;
        let Body::Points(body) = &rec.body else {
            unreachable!("points are drawn from a points record")
        };
        let t = side(body.radius);
        if t == 0 {
            continue;
        }
        let lo = (-t).div_euclid(2) + 1;
        let (place_kind, key) = placement(
            projection.as_ref(),
            rec.depth.space,
            rec.depth.fixed_orientation_centre,
        );
        for (i, p) in draw.positions.iter().enumerate() {
            let world = place(&rec.geometry.place_a, *p);
            if world.iter().any(|v| !v.is_finite()) {
                continue; // CE draws a non-finite point nowhere on the frame.
            }
            let shown = pre_display(&[world], projection.as_ref(), place_kind)[0];
            let key = key.unwrap_or_else(|| -projection.as_ref().unwrap().factor(shown[2]));
            let [sx, sy] = subpixel_coords(
                &[shown],
                width,
                height,
                view.frame_width,
                view.frame_height,
                view.frame_center,
            )[0];
            if !(sx.is_finite() && sy.is_finite()) || key.is_nan() {
                continue;
            }
            // `astype(int64)`, beyond whose range no block reaches the frame.
            let limit = 4.0e18;
            if sx.abs() > limit || sy.abs() > limit {
                continue;
            }
            let at = [sx.trunc() as i64, sy.trunc() as i64];
            let span = |c: i64, size: u32| {
                let a = c.saturating_add(lo).max(0);
                let b = c.saturating_add(lo).saturating_add(t).min(size as i64);
                (a < b).then_some((a as u32, b as u32))
            };
            let (Some((x0, x1)), Some((y0, y1))) = (span(at[0], width), span(at[1], height)) else {
                continue;
            };
            let c = draw.colours[i];
            let raw = [byte(c[0]), byte(c[1]), byte(c[2]), byte(c[3] * rec.opacity)];
            if raw[3] == 0 {
                continue;
            }
            let colour = if raw[3] == 255 {
                raw
            } else {
                let a = raw[3] as u32;
                [
                    ((raw[0] as u32 * a + 127) / 255) as u8,
                    ((raw[1] as u32 * a + 127) / 255) as u8,
                    ((raw[2] as u32 * a + 127) / 255) as u8,
                    raw[3],
                ]
            };
            let index = out.blocks.len() as u32;
            for ty in y0 / TILE..=(y1 - 1) / TILE {
                for tx in x0 / TILE..=(x1 - 1) / TILE {
                    out.bins[tile_index(width, tx * TILE, ty * TILE)].push(index);
                }
            }
            out.blocks.push(Block {
                x0,
                y0,
                x1,
                y1,
                at,
                lo,
                side: t,
                point: i as u32,
                points: draw.positions.len() as u32,
                colour,
                key,
                order: draw.order,
            });
        }
    }
    out
}

impl Blocks {
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    fn raster(&self, rect: Rect, opaque: bool, mut add: impl FnMut(u32, u32, Fragment)) {
        for &i in &self.bins[tile_index(self.width, rect.x0, rect.y0)] {
            let b = &self.blocks[i as usize];
            if (b.colour[3] == 255) != opaque {
                continue;
            }
            for y in b.y0.max(rect.y0)..b.y1.min(rect.y1) {
                for x in b.x0.max(rect.x0)..b.x1.min(rect.x1) {
                    let f = Fragment::new(
                        Plane::constant(b.key),
                        b.colour,
                        Coverage::Mask(FULL),
                        b.order,
                        b.sub(x, y),
                    );
                    add(x, y, f);
                }
            }
        }
    }
}

impl Producer for Blocks {
    fn opaque(&self, pass: &mut OpaquePass<'_>) {
        self.raster(pass.rect(), true, |x, y, f| pass.add(x, y, f));
    }

    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        self.raster(pass.rect(), false, |x, y, f| pass.add(x, y, f));
    }
}
