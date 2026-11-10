//! A frame cut into square tiles, each drawn and resolved on its own, on threads scoped to the
//! draw call (no pool outlives it, so a process may fork between draws).
//!
//! A tile's pixels are a function of its own fragments only, and nothing is summed across tiles
//! but integer counts, so the frame's bytes are the same whichever thread drew which tile.

use crate::composite::{
    Frame, OpaquePass, Producer, Rect, Rendered, ResolveCounts, TransparentPass,
};
use crate::label_plan::LabelPlan;
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

pub(crate) fn tile_rects(width: u32, height: u32) -> Vec<Rect> {
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

/// A tile's opaque and transparent passes, every producer's fragments added, not yet resolved.
pub(crate) fn passes<'a>(rect: Rect, producers: &'a [&'a dyn Producer]) -> TransparentPass<'a> {
    passes_placed(rect, producers, None)
}

/// [`passes`], whose resolves place the frame's labels by `plan`, if any.
pub(crate) fn passes_placed<'a>(
    rect: Rect,
    producers: &'a [&'a dyn Producer],
    plan: Option<&'a LabelPlan>,
) -> TransparentPass<'a> {
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
    pass.plan = plan;
    pass
}

fn draw_tile(
    rect: Rect,
    frame: &Frame,
    producers: &[&dyn Producer],
    plan: Option<&LabelPlan>,
) -> Rendered {
    let pass = passes_placed(rect, producers, plan);
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

/// The most threads one draw call uses, whatever it asks for: more than any machine the core
/// draws on has cores, far fewer than a large frame has tiles.
pub const MAX_WORKERS: usize = 256;

/// How many threads draw a frame of `tiles` tiles when `threads` are asked for: at least one, and
/// at most one per tile and [`MAX_WORKERS`]. The frame's bytes do not depend on it.
pub fn workers(threads: usize, tiles: usize) -> usize {
    threads.clamp(1, tiles.clamp(1, MAX_WORKERS))
}

/// Draws the producers over a background frame, its tiles spread over `threads` threads (at
/// least one, [`workers`]). A thread the system will not start fails the draw with its error.
pub fn render(
    background: Frame,
    producers: &[&dyn Producer],
    threads: usize,
) -> std::io::Result<Rendered> {
    render_placed(background, producers, None, threads)
}

/// [`render`], the frame's labels placed by `plan`, if any ([`crate::label_plan`]).
pub fn render_placed(
    background: Frame,
    producers: &[&dyn Producer],
    plan: Option<&LabelPlan>,
    threads: usize,
) -> std::io::Result<Rendered> {
    render_spawning(
        background,
        producers,
        plan,
        threads,
        std::thread::Builder::new,
    )
}

fn render_spawning(
    background: Frame,
    producers: &[&dyn Producer],
    plan: Option<&LabelPlan>,
    threads: usize,
    builder: impl Fn() -> std::thread::Builder,
) -> std::io::Result<Rendered> {
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
            done.push((i, draw_tile(*rect, &background, producers, plan)));
        }
    };
    let tiles = std::thread::scope(|s| {
        let mut helpers = Vec::new();
        for _ in 1..workers(threads, rects.len()) {
            match builder().spawn_scoped(s, work) {
                Ok(helper) => helpers.push(helper),
                Err(err) => {
                    // No tile is handed out any more; the helpers already started finish the one
                    // they hold, and the scope waits for them.
                    next.store(rects.len(), Ordering::Relaxed);
                    return Err(err);
                }
            }
        }
        let mut tiles = work();
        for h in helpers {
            match h.join() {
                Ok(done) => tiles.extend(done),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        Ok(tiles)
    })?;
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
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draw_never_asks_for_more_threads_than_its_cap_or_its_tiles() {
        // 2,040 tiles at 3840 x 2160, 65,536 at the largest frame: one thread each at most.
        for (threads, tiles) in [(usize::MAX, 65_536), (10_000, 2_040), (1 << 20, 1 << 20)] {
            assert_eq!(
                workers(threads, tiles),
                MAX_WORKERS,
                "{threads} for {tiles}"
            );
        }
        assert_eq!(workers(8, 3), 3);
        assert_eq!(workers(4, 100), 4);
        assert_eq!(workers(1, 0), 1);
    }

    #[test]
    fn a_thread_that_cannot_start_fails_the_draw_and_does_not_panic() {
        // A stack larger than any address space: the system refuses the thread.
        let refused = || std::thread::Builder::new().stack_size(1 << 50);
        let frame = Frame::filled(4 * TILE, TILE, [0, 0, 0, 255]);
        assert!(render_spawning(frame.clone(), &[], None, 4, refused).is_err());
        // One thread starts none.
        let drawn = render_spawning(frame.clone(), &[], None, 1, refused).unwrap();
        assert_eq!(drawn.rgba, frame.rgba());
    }
}
