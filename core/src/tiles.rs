//! A frame cut into square tiles, each drawn and resolved on its own, on threads scoped to the
//! draw call (no pool outlives it, so a process may fork between draws).
//!
//! A tile's pixels are a function of its own fragments only, and nothing is summed across tiles
//! but integer counts, so the frame's bytes are the same whichever thread drew which tile.

use crate::composite::{Frame, OpaquePass, Producer, Rect, Rendered, ResolveCounts};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A tile's side in pixels.
pub const TILE: u32 = 64;

/// Tiles across a frame `width` pixels wide.
pub fn tiles_across(width: u32) -> u32 {
    width.div_ceil(TILE)
}

/// The index of the tile holding the rectangle that starts at `(x0, y0)`.
pub fn tile_index(width: u32, x0: u32, y0: u32) -> usize {
    (y0 / TILE * tiles_across(width) + x0 / TILE) as usize
}

fn tile_rects(width: u32, height: u32) -> Vec<Rect> {
    let mut rects = Vec::new();
    for y0 in (0..height).step_by(TILE as usize) {
        for x0 in (0..width).step_by(TILE as usize) {
            rects.push(Rect {
                x0,
                y0,
                x1: (x0 + TILE).min(width),
                y1: (y0 + TILE).min(height),
            });
        }
    }
    rects
}

fn draw_tile(rect: Rect, frame: &Frame, producers: &[&dyn Producer]) -> Rendered {
    let mut opaque = OpaquePass::new(rect, producers);
    for (i, p) in producers.iter().enumerate() {
        opaque.source = i as u16;
        p.opaque(&mut opaque);
    }
    let mut pass = opaque.finish();
    for (i, p) in producers.iter().enumerate() {
        pass.source = i as u16;
        p.transparent(&mut pass);
    }
    let stride = frame.width() as usize * 4;
    let mut background = Vec::with_capacity(rect.pixels() * 4);
    for y in rect.y0..rect.y1 {
        let row = y as usize * stride;
        background.extend_from_slice(
            &frame.rgba()[row + rect.x0 as usize * 4..row + rect.x1 as usize * 4],
        );
    }
    pass.resolve(&background)
}

/// Draws the producers over a background frame, its tiles spread over `threads` threads (at
/// least one).
pub fn render(background: Frame, producers: &[&dyn Producer], threads: usize) -> Rendered {
    assert!(producers.len() <= u16::MAX as usize, "too many producers");
    let (width, height) = (background.width(), background.height());
    let rects = tile_rects(width, height);
    let next = AtomicUsize::new(0);
    let work = || {
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(rect) = rects.get(i) else {
                return done;
            };
            done.push((i, draw_tile(*rect, &background, producers)));
        }
    };
    let tiles = std::thread::scope(|s| {
        let helpers: Vec<_> = (1..threads.clamp(1, rects.len().max(1)))
            .map(|_| s.spawn(work))
            .collect();
        let mut tiles = work();
        for h in helpers {
            match h.join() {
                Ok(done) => tiles.extend(done),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        tiles
    });
    let mut out = Rendered {
        rgba: background.rgba().to_vec(),
        resolves: vec![0; width as usize * height as usize],
        counts: ResolveCounts::default(),
    };
    for (i, tile) in tiles {
        let rect = rects[i];
        let w = (rect.x1 - rect.x0) as usize;
        for (row, y) in (rect.y0..rect.y1).enumerate() {
            let at = y as usize * width as usize + rect.x0 as usize;
            out.rgba[at * 4..(at + w) * 4]
                .copy_from_slice(&tile.rgba[row * w * 4..(row + 1) * w * 4]);
            out.resolves[at..at + w].copy_from_slice(&tile.resolves[row * w..(row + 1) * w]);
        }
        out.counts.add(&tile.counts);
    }
    out
}
