//! The coverage resolve against the sample resolve of the same pixel: exhaustively over the scope
//! of `COVERAGE_TO_SAMPLE_LSB`, and through the resolves themselves for every partial layer
//! `coverage_to_sample_bound` bounds, with random and adversarial stacks.

mod common;

use arachne_core::composite::*;
use arachne_core::scanline;
use arachne_core::tiles;
use common::resolve::EverySample;

/// A widest gap and its witness `(s, a, d, n, k)`.
type Widest = (u8, [u32; 5]);

/// The widest gap, in one channel, between the two resolves over the constant's scope, alone and
/// under one translucent full layer, each with its witness `(s, a, d, n, k)`: a partial layer of
/// channel `s` and alpha `a` (premultiplied, `s <= a`) covering `n` of the sub-samples, over `d`
/// (what the full layers beneath composite to, the same at every sub-sample on both resolves),
/// under nothing or under a translucent layer of alpha `255 - k` (`k` in `1..255`). That layer's
/// own channel adds the same whole byte to the coverage resolve and to every sub-sample, so it
/// leaves the gap as it is and is taken as 0.
fn widest_in_scope() -> (Widest, Widest) {
    let channel = |src: [u8; 2], coverage: u8, dst: u8| {
        blend([src[0], 0, 0, src[1]], coverage, [dst, 0, 0, 0])[0]
    };
    // The layer in front as a table: `front[x][k]` is what it leaves of `x` at alpha `255 - k`
    // (`k` = 0, an opaque layer, hides the rest and is never read).
    let front: Vec<[u8; 255]> = (0..=255u8)
        .map(|x| std::array::from_fn(|k| channel([0, 255 - k as u8], 255, x)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per_thread: Vec<_> = std::thread::scope(|scope| {
        let front = &front;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut alone = (0u8, [0u32; 5]);
                    let mut under = (0u8, [0u32; 5]);
                    for a in (0..=255u8).filter(|a| *a as usize % threads == t) {
                        for s in 0..=a {
                            for d in 0..=255u8 {
                                let covered = channel([s, a], 255, d);
                                let (fc, fd) = (&front[covered as usize], &front[d as usize]);
                                for n in 1..16u32 {
                                    let coverage = mask_coverage(((1u32 << n) - 1) as u16);
                                    let drawn = channel([s, a], coverage, d);
                                    let sampled =
                                        (n * covered as u32 + (16 - n) * d as u32 + 8) / 16;
                                    let gap = (drawn as u32).abs_diff(sampled) as u8;
                                    if gap > alone.0 {
                                        alone = (gap, [s as u32, a as u32, d as u32, n, 255]);
                                    }
                                    let fdrawn = &front[drawn as usize];
                                    let gap_at = |k: usize| {
                                        let sampled =
                                            (n * fc[k] as u32 + (16 - n) * fd[k] as u32 + 8) / 16;
                                        (fdrawn[k] as u32).abs_diff(sampled) as u8
                                    };
                                    let widest = (1..255).map(gap_at).max().unwrap();
                                    if widest > under.0 {
                                        let k = (1..255).find(|k| gap_at(*k) == widest).unwrap();
                                        under =
                                            (widest, [s as u32, a as u32, d as u32, n, k as u32]);
                                    }
                                }
                            }
                        }
                    }
                    (alone, under)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let widest = |pick: fn(&(Widest, Widest)) -> Widest| {
        per_thread.iter().map(pick).max_by_key(|w| w.0).unwrap()
    };
    (widest(|w| w.0), widest(|w| w.1))
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "about 10^10 blends: release only, as core/ci/test.sh runs the core's tests"
)]
fn over_the_constants_scope_the_widest_gap_is_exactly_the_constant() {
    let ((alone, at), (under, under_at)) = widest_in_scope();
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!(
            "one partial mask: widest {alone} at (s, a, d, n, k) {at:?}, under a layer {under} at {under_at:?}"
        );
    }
    assert_eq!(
        alone,
        coverage_to_sample_bound(Partial::Mask, 0),
        "alone, at {at:?}"
    );
    assert_eq!(
        under, COVERAGE_TO_SAMPLE_LSB,
        "under a layer, at {under_at:?}"
    );
}

/// Uniform in `0..1`, a fixed sequence.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.unit() * n as f64) as u32
    }

    fn colour(&mut self, alpha: f64) -> [u8; 4] {
        premultiply([self.unit(), self.unit(), self.unit(), alpha])
    }

    /// Opaque half the time, else translucent.
    fn any_colour(&mut self) -> [u8; 4] {
        let alpha = if self.unit() < 0.5 { 1.0 } else { self.unit() };
        self.colour(alpha)
    }

    /// Translucent, never opaque nor clear.
    fn translucent(&mut self) -> [u8; 4] {
        let alpha = 0.01 + 0.98 * self.unit();
        self.colour(alpha)
    }
}

/// The partial layer of a stack.
enum Part {
    /// One surface's masks, disjoint, each in its colour: one member is a plain mask.
    Masks(Vec<(u16, [u8; 4])>),
    /// A path's exact area in each sub-sample's cell, and its colour; its area is their mean.
    Path([f32; SAMPLES], [u8; 4]),
}

/// One pixel's layers, back to front: full layers, the partial layer, translucent full layers.
struct Stack {
    beneath: Vec<[u8; 4]>,
    partial: Part,
    fronts: Vec<[u8; 4]>,
}

impl Producer for Stack {
    fn transparent(&self, pass: &mut TransparentPass<'_>) {
        let mut order = 0u32;
        let mut next = |depth: &mut f64| {
            *depth -= 1.0;
            order += 1;
            (Plane::constant(*depth), order)
        };
        let mut depth = 100.0;
        for c in &self.beneath {
            let (plane, order) = next(&mut depth);
            pass.add(
                0,
                0,
                Fragment::new(plane, *c, Coverage::Mask(FULL), order, 0),
            );
        }
        let (plane, at) = next(&mut depth);
        match &self.partial {
            Part::Masks(members) => {
                for (sub, (mask, c)) in members.iter().enumerate() {
                    let f = Fragment::new(plane, *c, Coverage::Mask(*mask), at, sub as u32);
                    pass.add(0, 0, f.on_surface(1));
                }
            }
            Part::Path(cells, c) => {
                let area = (cells.iter().map(|a| *a as f64).sum::<f64>() / SAMPLES as f64) as f32;
                pass.add(
                    0,
                    0,
                    Fragment::new(plane, *c, Coverage::Area { area, id: 0 }, at, 0),
                );
            }
        }
        for c in &self.fronts {
            let (plane, order) = next(&mut depth);
            pass.add(
                0,
                0,
                Fragment::new(plane, *c, Coverage::Mask(FULL), order, 0),
            );
        }
    }

    fn cell_areas(&self, _id: u32, _x: u32, _y: u32) -> [f32; SAMPLES] {
        match &self.partial {
            Part::Path(cells, _) => *cells,
            Part::Masks(_) => unreachable!("masks have no cells"),
        }
    }
}

/// The widest channel gap between the stack's coverage resolve and its sample resolve, over a
/// background `bg`.
fn gap(stack: &Stack, bg: [u8; 4]) -> u8 {
    let drawn = tiles::render(Frame::filled(1, 1, bg), &[stack as &dyn Producer], 1).unwrap();
    let sampled = tiles::render(
        Frame::filled(1, 1, bg),
        &[stack as &dyn Producer, &EverySample],
        1,
    )
    .unwrap();
    assert_eq!(drawn.resolves[0], Resolve::Coverage as u8);
    assert_eq!(sampled.resolves[0], Resolve::Sample as u8);
    (0..4)
        .map(|k| drawn.rgba[k].abs_diff(sampled.rgba[k]))
        .max()
        .unwrap()
}

/// Holds random stacks with the partial layer `part` makes to `coverage_to_sample_bound`, for 0
/// to 3 layers in front; returns the widest gap per count of layers in front.
fn hold(
    kind: Partial,
    rounds: usize,
    seed: u64,
    mut part: impl FnMut(&mut Lcg) -> Part,
) -> [u8; 4] {
    let mut rng = Lcg(seed);
    let mut widest = [0u8; 4];
    for round in 0..rounds {
        let fronts = round % 4;
        let stack = Stack {
            beneath: (0..rng.below(3)).map(|_| rng.any_colour()).collect(),
            partial: part(&mut rng),
            fronts: (0..fronts).map(|_| rng.translucent()).collect(),
        };
        let bg = rng.colour(1.0);
        let g = gap(&stack, bg);
        let bound = coverage_to_sample_bound(kind, fronts as u32);
        assert!(
            g <= bound,
            "{kind:?}, {fronts} in front: a gap of {g}, past {bound}"
        );
        widest[fronts] = widest[fronts].max(g);
    }
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!("{kind:?}: widest gap by layers in front {widest:?}");
    }
    widest
}

/// A partial mask's bits: `n` of the sub-samples, 0 < n < 16, chosen at random.
fn partial_mask(rng: &mut Lcg) -> u16 {
    loop {
        let m = (rng.unit() * 65536.0) as u16;
        if m != 0 && m != FULL {
            return m;
        }
    }
}

#[test]
fn a_partial_mask_lands_within_its_derived_bound_under_any_number_of_layers() {
    hold(Partial::Mask, 40_000, 3, |rng| {
        Part::Masks(vec![(partial_mask(rng), rng.any_colour())])
    });
}

#[test]
fn a_surfaces_masks_in_several_colours_land_within_their_derived_bound() {
    hold(Partial::MergedMask, 40_000, 7, |rng| {
        // Two to four disjoint masks splitting a partial mask's bits, of one alpha (a surface's
        // shading changes its colour, not its opacity), or of several.
        let union = partial_mask(rng);
        let alpha = if rng.unit() < 0.5 { 1.0 } else { rng.unit() };
        let parts = 2 + rng.below(3) as usize;
        let mut masks = vec![0u16; parts];
        for bit in (0..16).filter(|b| union & (1 << b) != 0) {
            masks[rng.below(parts as u32) as usize] |= 1 << bit;
        }
        let same = rng.unit() < 0.5;
        Part::Masks(
            masks
                .into_iter()
                .filter(|m| *m != 0)
                .map(|m| {
                    (
                        m,
                        if same {
                            rng.colour(alpha)
                        } else {
                            rng.any_colour()
                        },
                    )
                })
                .collect(),
        )
    });
}

#[test]
fn a_paths_cells_land_within_their_derived_bound_random_and_adversarial() {
    // Random cells, each anywhere in 0..1.
    hold(Partial::Path, 20_000, 11, |rng| {
        Part::Path(std::array::from_fn(|_| rng.unit() as f32), rng.any_colour())
    });
    // An edge: cells full or empty, one between.
    hold(Partial::Path, 20_000, 13, |rng| {
        let full = rng.below(15) as usize;
        Part::Path(
            std::array::from_fn(|k| match k {
                k if k < full => 1.0,
                k if k == full => rng.unit() as f32,
                _ => 0.0,
            }),
            rng.any_colour(),
        )
    });
    // Adversarial: every cell's byte rounds one way and the area's the other. Cells sit just
    // below a half level, `t` of them a level above the rest, so their bytes average to a
    // sixteenth of `t` above the base while the area rounds up a whole level (or, at a half level
    // exactly, the cells round up and the area does not): the coverages differ by up to 15/16.
    hold(Partial::Path, 20_000, 17, |rng| {
        let base = rng.below(250) as f64;
        let t = 1 + rng.below(15) as usize;
        let shift = if rng.unit() < 0.5 { 0.5 - 1e-3 } else { 0.5 };
        Part::Path(
            std::array::from_fn(|k| ((base + (k < t) as u8 as f64 + shift) / 255.0) as f32),
            rng.any_colour(),
        )
    });
}

#[test]
fn a_rasterized_paths_area_lies_within_the_drift_of_its_cells_mean() {
    // Random triangles and quadrilaterals over a 12 x 12 grid, as the path rasterizer stores
    // them: each pixel's area and its cells' areas, as f32.
    let mut rng = Lcg(19);
    let (mut pixels, mut widest) = (0, 0f64);
    for round in 0..600 {
        let corners = 3 + round % 2;
        let ring: Vec<[f64; 2]> = (0..corners)
            .map(|_| [rng.unit() * 12.0, rng.unit() * 12.0])
            .collect();
        let edges = scanline::edges_of(&[ring]);
        let all: Vec<u32> = (0..edges.len() as u32).collect();
        let grid = Rect {
            x0: 0,
            y0: 0,
            x1: 12,
            y1: 12,
        };
        let mut runs = Vec::new();
        scanline::cover(
            &edges,
            &all,
            grid,
            &mut scanline::Scratch::default(),
            |y, x0, x1, a| {
                runs.extend((x0..x1).map(|x| (x, y, a)));
            },
        );
        for (x, y, area) in runs {
            if area >= 1.0 {
                continue;
            }
            let cells = scanline::cell_areas(&edges, &all, x, y);
            let mean = cells
                .iter()
                .flatten()
                .map(|c| *c as f32 as f64)
                .sum::<f64>()
                / SAMPLES as f64;
            widest = widest.max((area as f32 as f64 - mean).abs());
            pixels += 1;
        }
    }
    assert!(pixels > 10_000, "{pixels} pixels");
    assert!(
        widest <= AREA_DRIFT,
        "an area {widest:e} from its cells' mean"
    );
    if std::env::var_os("CORE_STATS").is_some() {
        eprintln!("area against its cells' mean: widest {widest:e} over {pixels} pixels");
    }
}
