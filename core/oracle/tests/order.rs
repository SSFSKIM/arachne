//! The order at a sample (contract §7): a pair's edges and label from its two reads, and which
//! orders of the present fragments are admissible, through `cycle_rank@1` where edges cycle.

use arachne_oracle::draw::Layer;
use arachne_oracle::keys::{Band, Model, Offered, Unable};
use arachne_oracle::order::{Edges, Frag, Label, Reads, Scene, admissible, rank};

fn frag(entry: u32, layer: Layer, group: u32) -> Frag {
    Frag {
        entry,
        sub: if layer == Layer::Stroke { 2 } else { 1 },
        layer,
        group,
        model: Model::NegInf,
        row: None,
        offered: None,
        lines: None,
    }
}

fn band(o: f64, half: f64) -> Reads {
    Reads {
        own: Ok(Band {
            o,
            r: 0.0,
            s_minus: half,
            s_plus: half,
        }),
        row: None,
        own_at: None,
        row_at: None,
    }
}

fn three() -> Scene {
    Scene::new(
        (0..3).map(|i| frag(i, Layer::Mesh, u32::MAX)).collect(),
        Vec::new(),
    )
}

const NONE: (f64, f64) = (f64::INFINITY, f64::INFINITY);

#[test]
fn the_nontransitive_triple_ties_its_neighbours_and_orders_its_ends() {
    // Keys 0, 1 and 2 with half-widths 0.6: the adjacent intervals meet, the ends do not.
    let s = three();
    let r = [band(0.0, 0.6), band(1.0, 0.6), band(2.0, 0.6)];
    for (a, b) in [(0, 1), (1, 2)] {
        let p = s.pair(a, b, &r[a], &r[b], NONE).unwrap();
        assert_eq!(p.label, Label::TieBand, "{a} {b}");
        assert_eq!(p.edges, Edges::ALL, "{a} {b}");
    }
    let ends = s.pair(0, 2, &r[0], &r[2], NONE).unwrap();
    assert_eq!(ends.label, Label::Depth(true));
    assert_eq!(
        ends.edges,
        Edges {
            weak: false,
            a_front: true,
            b_front: false
        }
    );
}

/// The edges of the triple, by position pair.
fn edges_of(e: [[Edges; 3]; 3]) -> impl Fn(usize, usize) -> Edges {
    move |i, j| e[i][j]
}

fn paint(f: usize) -> (u32, u32) {
    (f as u32, 0)
}

const WEAK: Edges = Edges::WEAK;
const A: Edges = Edges {
    weak: false,
    a_front: true,
    b_front: false,
};
const B: Edges = Edges {
    weak: false,
    a_front: false,
    b_front: true,
};
const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[test]
fn a_cycle_through_a_weak_order_keeps_every_strong_order() {
    // 0 before 1 and 1 before 2 by depth; 2 before 0 by paint (2 is painted later).
    let none = Edges::default();
    let e = edges_of([[none, A, WEAK], [none, none, A], [none, none, none]]);
    let present = [0, 1, 2];
    for order in PERMS {
        let want = order == [0, 1, 2];
        assert_eq!(
            admissible(&present, &paint, &e, &order),
            Some(want),
            "{order:?}"
        );
    }
}

#[test]
fn an_all_strong_cycle_is_ranked_by_wins_then_paint_order() {
    // 0 before 1, 1 before 2 and 2 before 0, each strong: every one has a strong edge into it and
    // each is in front of one other, so the later painted (2) comes first; then 0, free of 1.
    let none = Edges::default();
    let e = edges_of([[none, A, B], [none, none, A], [none, none, none]]);
    let present = [0, 1, 2];
    let front = vec![
        vec![false, true, false],
        vec![false, false, true],
        vec![true, false, false],
    ];
    assert_eq!(rank(3, &front, &front, &paint), vec![2, 0, 1]);
    for order in PERMS {
        let want = order == [2, 0, 1];
        assert_eq!(
            admissible(&present, &paint, &e, &order),
            Some(want),
            "{order:?}"
        );
    }
}

#[test]
fn a_fill_and_its_own_stroke_keep_paint_order_whatever_their_keys() {
    let s = Scene::new(
        vec![frag(0, Layer::Fill, 7), frag(0, Layer::Stroke, 7)],
        Vec::new(),
    );
    // The stroke is far behind its fill, and is still painted over it.
    let p = s
        .pair(0, 1, &band(-2.0, 0.0), &band(-1.0, 0.0), NONE)
        .unwrap();
    assert_eq!((p.label, p.edges), (Label::OwnLayers, Edges::WEAK));
    // Two strokes of one path keep depth.
    let t = Scene::new(
        vec![
            frag(0, Layer::BackgroundStroke, 7),
            frag(0, Layer::Stroke, 7),
        ],
        Vec::new(),
    );
    let p = t
        .pair(0, 1, &band(-2.0, 0.0), &band(-1.0, 0.0), NONE)
        .unwrap();
    assert_eq!(p.label, Label::Depth(true));
}

#[test]
fn keys_of_minus_infinity_tie_exactly_and_meet_any_finite_key_by_depth() {
    let s = three();
    let inf = band(f64::NEG_INFINITY, 0.0);
    let p = s.pair(0, 1, &inf, &inf, NONE).unwrap();
    assert_eq!((p.label, p.edges), (Label::ExactTie, Edges::WEAK));
    let p = s.pair(0, 1, &band(-1.0, 0.5), &inf, NONE).unwrap();
    assert_eq!((p.label, p.edges), (Label::Depth(false), B));
}

#[test]
fn intervals_that_only_touch_still_meet() {
    // I_a = [-1.5, -0.5], I_b = [-0.5, 0.5]: they meet at one point, so both orders stand.
    let s = three();
    let p = s
        .pair(0, 1, &band(-1.0, 0.5), &band(0.0, 0.5), NONE)
        .unwrap();
    assert_eq!(p.label, Label::TieBand);
    assert!(p.edges.weak && p.edges.a_front);
    assert!(!p.edges.b_front, "{:?}", p.edges);
}

#[test]
fn lines_straight_cubic_step_charges_at_least_the_core_s_step() {
    // Contract §12's `lines_straight_cubic_step` (`lines@1`): a straight cubic of four keys −1,
    // controls 6 px apart, unplaced. Its L = 18 sits on a count step, so it is exposed; the core
    // charges 320u (1 + 16u) there, and B gives step⁺ >= 328u.
    let u = 1.0 / 9_007_199_254_740_992.0;
    let x = [[0.0, 0.0], [6.0, 0.0], [12.0, 0.0], [18.0, 0.0]];
    let step = arachne_oracle::lines::step_plus(&x, &[-1.0; 4], 18.0, 0.0, 0.0, 18.0);
    assert!(step >= 328.0 * u, "{}", step / u);
    assert!(step >= 320.0 * u * (1.0 + 16.0 * u));
    assert!(step <= 329.0 * u, "{}", step / u);
    // Off a count step (L = 19, unexposed) nothing is charged.
    let x = [[0.0, 0.0], [6.0, 0.0], [12.0, 0.0], [19.0, 0.0]];
    assert_eq!(
        arachne_oracle::lines::step_plus(&x, &[-1.0; 4], 19.0, 0.0, 0.0, 19.0),
        0.0
    );
}

#[test]
fn the_nontransitive_triple_admits_its_reversal_through_an_all_strong_cycle() {
    // Contract §7 read literally: a completion picks any edge of each pair's E. The adjacent pairs
    // of the triple admit both strong edges, so the completion 0 → 2, 2 → 1, 1 → 0 is an all-strong
    // cycle; its rank (equal wins, the later painted first) is [2, 1, 0]. That order breaks the
    // ends' strong edge, and the literal reading admits it. Whether the contract means a tighter
    // reading is an open question; until it says so, this pins the literal one.
    let s = three();
    let r = [band(0.0, 0.6), band(1.0, 0.6), band(2.0, 0.6)];
    let pair = |a: usize, b: usize| s.pair(a, b, &r[a], &r[b], NONE).unwrap().edges;
    let e = edges_of([
        [Edges::default(), pair(0, 1), pair(0, 2)],
        [Edges::default(), Edges::default(), pair(1, 2)],
        [Edges::default(); 3],
    ]);
    let present = [0, 1, 2];
    // The orders one key per fragment can give (0 is always in front of 2, the ends being strong):
    // every one is admitted, so the literal reading makes no false failure.
    let one_key = [[0, 1, 2], [1, 0, 2], [0, 2, 1]];
    for order in one_key {
        assert_eq!(
            admissible(&present, &paint, &e, &order),
            Some(true),
            "{order:?}"
        );
    }
    // What the literal reading admits beyond them, through all-strong cycles.
    let added: Vec<[usize; 3]> = PERMS
        .into_iter()
        .filter(|o| !one_key.contains(o))
        .filter(|o| admissible(&present, &paint, &e, o) == Some(true))
        .collect();
    assert_eq!(added, vec![[2, 1, 0]], "{added:?}");
}

/// Two strokes of records with no plane of their own on the axis plane z = 0 alone, `a` read as
/// `own`: the pair's verdict.
fn axis_only(own: Result<Band, Unable>) -> Result<(Edges, Label), Unable> {
    let line = |entry: u32| Frag {
        offered: Some(entry as usize),
        ..frag(entry, Layer::Stroke, entry)
    };
    let offered = |y: f64| Offered {
        flat: None,
        retains: true,
        points: vec![[-1.0, y, 0.0], [1.0, y, 0.0]],
        tau_plus: 0.0,
        tau_own: 0.0,
        axes: vec![(2, 0.0)],
        snap: 0.0,
        path: true,
    };
    let s = Scene::new(vec![line(0), line(1)], vec![offered(0.0), offered(0.5)]);
    let ra = Reads {
        own,
        ..band(-1.0, 0.0)
    };
    s.pair(0, 1, &ra, &band(-1.0, 0.0), NONE)
        .map(|p| (p.edges, p.label))
}

#[test]
fn an_axis_only_pair_needs_no_own_read_but_a_fragment_with_no_bound_stays_unable() {
    // §6: `curve_recut` is per read, and the axis-only pair reads neither own key (§7);
    // `band_unbounded` is per fragment, its samples `truth_unable` whatever the pair reads.
    assert_eq!(
        axis_only(band(-1.0, 0.0).own),
        Ok((Edges::WEAK, Label::ExactTie))
    );
    assert_eq!(
        axis_only(Err(Unable::CurveRecut)),
        Ok((Edges::WEAK, Label::ExactTie))
    );
    assert_eq!(
        axis_only(Err(Unable::BandUnbounded)),
        Err(Unable::BandUnbounded)
    );
}

#[test]
fn a_fragment_with_no_key_stated_is_unable_against_its_own_record_s_other_layer() {
    // §6 `band_unbounded` and §2 `pivot_unknown` are per fragment: its samples are
    // `truth_unable` whatever the pair reads, its own record's other layer included. One record's
    // fill and stroke (entry 0, group 0), the fill unable.
    let s = Scene::new(
        vec![frag(0, Layer::Fill, 0), frag(0, Layer::Stroke, 0)],
        Vec::new(),
    );
    for u in [Unable::BandUnbounded, Unable::PivotUnknown] {
        let ra = Reads {
            own: Err(u),
            ..band(-1.0, 0.0)
        };
        assert_eq!(
            s.pair(0, 1, &ra, &band(-1.0, 0.0), NONE).map(|p| p.label),
            Err(u),
            "{u:?}"
        );
    }
    // A `curve_recut` read is unable only where a scenario reads it, and the record's own layers
    // read none.
    let ra = Reads {
        own: Err(Unable::CurveRecut),
        ..band(-1.0, 0.0)
    };
    assert_eq!(
        s.pair(0, 1, &ra, &band(-1.0, 0.0), NONE).map(|p| p.label),
        Ok(Label::OwnLayers)
    );
}
