//! The order at a sample (contract §7): a pair's edges and label from its two reads, and which
//! orders of the present fragments are admissible, through `cycle_rank@1` where edges cycle.

use arachne_oracle::draw::Layer;
use arachne_oracle::keys::{Band, Model};
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
