//! The order at a sample (contract §7): each present pair's admitted edges `E` and label, from its
//! keys, bands and tie policies; and whether an order of the present fragments is admissible, by
//! the rank of some completion (`cycle_rank@1`).

use std::cell::RefCell;
use std::collections::HashMap;

use crate::band::width_lower_bound;
use crate::draw::Layer;
use crate::keys::{Band, Computation, Model, Offered, Unable};
use crate::shift::U;

/// The edges a pair admits: weak (paint order), strong `a` in front, strong `b` in front.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Edges {
    pub weak: bool,
    pub a_front: bool,
    pub b_front: bool,
}

impl Edges {
    pub const WEAK: Edges = Edges {
        weak: true,
        a_front: false,
        b_front: false,
    };
    pub const ALL: Edges = Edges {
        weak: true,
        a_front: true,
        b_front: true,
    };
    pub fn union(self, o: Edges) -> Edges {
        Edges {
            weak: self.weak || o.weak,
            a_front: self.a_front || o.a_front,
            b_front: self.b_front || o.b_front,
        }
    }
    pub fn bits(self) -> u8 {
        self.weak as u8 | (self.a_front as u8) << 1 | (self.b_front as u8) << 2
    }
    fn count(self) -> usize {
        self.weak as usize + self.a_front as usize + self.b_front as usize
    }
}

/// A pair's label (contract §7): codes are `ORACLE_FORMAT`'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Label {
    /// The single strong edge: `true` when `a` is in front.
    Depth(bool),
    TieBand,
    ExactTie,
    OwnLayers,
    OnePlane,
    Lines,
    NearParallel,
}

impl Label {
    pub fn code(self) -> u8 {
        match self {
            Label::Depth(true) => 1,
            Label::Depth(false) => 2,
            Label::TieBand => 3,
            Label::ExactTie => 4,
            Label::OwnLayers => 5,
            Label::OnePlane => 6,
            Label::Lines => 7,
            Label::NearParallel => 8,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Label::Depth(_) => "depth",
            Label::TieBand => "tie_band",
            Label::ExactTie => "exact_tie",
            Label::OwnLayers => "own_layers@1",
            Label::OnePlane => "one_plane@1",
            Label::Lines => "lines@1",
            Label::NearParallel => "near_parallel@1",
        }
    }
}

/// What the order reads of one fragment.
#[derive(Clone, Debug)]
pub struct Frag {
    pub entry: u32,
    pub sub: u32,
    pub layer: Layer,
    /// The drawn path it belongs to (its three layers share it), else `u32::MAX`.
    pub group: u32,
    pub model: Model,
    /// Its row on its own plane (a flat path's layer, read with no fringe).
    pub row: Option<Model>,
    /// Its record's registry facts, an index into [`Scene::offered`].
    pub offered: Option<usize>,
    /// A stroke's `lines@1` share terms, where its model reads a centreline.
    pub lines: Option<crate::lines::LinesBound>,
}

impl Frag {
    /// Later painted: later in the order, then the higher `sub`.
    pub fn paint(&self) -> (u32, u32) {
        (self.entry, self.sub)
    }
    fn stroke(&self) -> bool {
        matches!(self.layer, Layer::Stroke | Layer::BackgroundStroke)
    }
}

/// `own_layers@1`: a path's fill and one of its own strokes or background strokes.
pub fn own_layers(a: &Frag, b: &Frag) -> bool {
    a.group != u32::MAX
        && a.group == b.group
        && a.entry == b.entry
        && a.sub / 3 == b.sub / 3
        && a.layer != b.layer
        && (a.layer == Layer::Fill || b.layer == Layer::Fill)
}

/// A frame's fragments and its registry, with what the tie policies read.
pub struct Scene {
    pub frags: Vec<Frag>,
    pub offered: Vec<Offered>,
    width_cache: RefCell<HashMap<(usize, usize), bool>>,
}

/// One sample's reads of a present fragment: its own band (adjusted on its own fill), its row's.
#[derive(Clone, Copy, Debug)]
pub struct Reads {
    pub own: Result<Band, Unable>,
    pub row: Option<Result<Band, Unable>>,
    /// What its own read and its row's compute, where each is one computation (§7, `ExactTie`).
    pub own_at: Option<Computation>,
    pub row_at: Option<Computation>,
}

/// How a strong edge is read (contract §7, the completions rule): from the two fragments' own
/// keys in every scenario that admits it, or by one of the two reads that are not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Via {
    /// Every scenario admitting it reads both fragments at their own keys, or no edge.
    #[default]
    Own,
    /// Some scenario admits it through a §5 row read (`shared_plane_read@1`).
    Row,
    /// The pair may share a flat identity with a record of no plane of its own (`one_plane@1`).
    Flat,
}

impl Via {
    pub fn code(self) -> u8 {
        match self {
            Via::Own => 0,
            Via::Row => 1,
            Via::Flat => 2,
        }
    }
}

/// The plate model's relation of a pair at a sample (`PAIR`'s `plate`, contract §10's
/// `label_plate_model`): what the keys the core orders by, read without the input snap, establish
/// in every scenario the pair's sets allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plate {
    /// The keys are equal: `tied`, a weak order, which is paint order (§7).
    Tie,
    /// The keys are separated beyond their bounds: `true` where `a` is in front.
    Front(bool),
    /// Neither: no order is the plate model's by the facts.
    Undetermined,
}

impl Plate {
    /// `PAIR`'s `plate` byte: a tie as `edges`' weak bit, a separation as its strong edge's bit,
    /// undetermined as 0.
    pub fn code(self) -> u8 {
        match self {
            Plate::Tie => Edges::WEAK.bits(),
            Plate::Front(true) => 2,
            Plate::Front(false) => 4,
            Plate::Undetermined => 0,
        }
    }

    /// The relation of keys `a`, `b` in one scenario; `policy` where a tie policy's term held
    /// there, which establishes no separation.
    fn of(a: &Band, b: &Band, policy: bool) -> Plate {
        let e = by_depth(a, b);
        if a.o == b.o {
            Plate::Tie
        } else if !policy && !e.weak && e.count() == 1 {
            Plate::Front(e.a_front)
        } else {
            Plate::Undetermined
        }
    }

    /// The relation over two scenarios: established only where both establish it.
    fn and(self, o: Plate) -> Plate {
        if self == o { self } else { Plate::Undetermined }
    }
}

/// Folds one scenario's relation into the pair's.
fn fold(acc: &mut Option<Plate>, p: Plate) {
    *acc = Some(acc.map_or(p, |q| q.and(p)));
}

/// A pair's verdict at a sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairAt {
    pub edges: Edges,
    pub label: Label,
    /// How its strong edges are read: `a_front`'s, then `b_front`'s.
    pub via: [Via; 2],
}

impl PairAt {
    /// `PAIR`'s `provenance`: `a_front`'s code in bits 0–1, `b_front`'s in bits 2–3.
    pub fn provenance(&self) -> u8 {
        self.via[0].code() | self.via[1].code() << 2
    }
}

/// `|o_a − o_b| <= β_a + β_b + (|o_a| + 2R_a) s_a + (|o_b| + 2R_b) s_b`: a policy may hold.
fn policy_term(a: &Band, sa: f64, b: &Band, sb: f64) -> bool {
    if !(sa.is_finite() && sb.is_finite()) {
        return true;
    }
    (a.o - b.o).abs()
        <= a.beta() + b.beta() + (a.o.abs() + 2.0 * a.r) * sa + (b.o.abs() + 2.0 * b.r) * sb
}

/// The edges two bands admit by depth: strong `a` in front where `lo_a < hi_b`, likewise `b`;
/// weak where they meet.
fn by_depth(a: &Band, b: &Band) -> Edges {
    if a.o == f64::NEG_INFINITY || b.o == f64::NEG_INFINITY {
        if a.o == b.o {
            return Edges::WEAK;
        }
        return Edges {
            weak: false,
            a_front: a.o == f64::NEG_INFINITY,
            b_front: b.o == f64::NEG_INFINITY,
        };
    }
    Edges {
        weak: a.lo() <= b.hi() && b.lo() <= a.hi(),
        a_front: a.lo() < b.hi(),
        b_front: b.lo() < a.hi(),
    }
}

fn meet(a: &Band, b: &Band) -> bool {
    a.lo() <= b.hi() && b.lo() <= a.hi()
}

impl Scene {
    pub fn new(frags: Vec<Frag>, offered: Vec<Offered>) -> Scene {
        Scene {
            frags,
            offered,
            width_cache: RefCell::new(HashMap::new()),
        }
    }

    /// `lines@1`'s eligibility: two strokes (or background strokes) of registry-offered paths
    /// with no plane of their own.
    pub fn lines_eligible(&self, a: usize, b: usize) -> bool {
        let (fa, fb) = (&self.frags[a], &self.frags[b]);
        let planeless = |f: &Frag| {
            f.offered
                .is_some_and(|i| self.offered[i].path && self.offered[i].flat.is_none())
        };
        fa.stroke() && fb.stroke() && planeless(fa) && planeless(fb)
    }

    /// The may-share width test of two offered records, at least one flat (contract §7, 1).
    fn width_passes(&self, i: usize, j: usize) -> bool {
        let key = (i.min(j), i.max(j));
        if let Some(v) = self.width_cache.borrow().get(&key) {
            return *v;
        }
        let (a, b) = (&self.offered[i], &self.offered[j]);
        let v = (a.flat.is_some() || b.flat.is_some()) && {
            let w = width_lower_bound(&a.points, &b.points);
            w <= 2.0 * a.tau_plus.max(b.tau_plus) * (1.0 + 1.0 / 1_099_511_627_776.0)
        };
        self.width_cache.borrow_mut().insert(key, v);
        v
    }

    /// Whether two offered records may share an identity through a flat record: (1) for a pair
    /// with a flat one, (2) a flat record passing (1) with each of them. Only flat records found
    /// identities, so a record with no plane of its own shares one with itself only through a
    /// flat record too; `M` need not retain one (contract §5: it still took part in the
    /// partition).
    fn may_share_flat(&self, i: usize, j: usize) -> bool {
        if i == j && self.offered[i].flat.is_some() {
            return self.offered[i].retains;
        }
        let (a, b) = (&self.offered[i], &self.offered[j]);
        if !(a.retains && b.retains) {
            return false;
        }
        if a.flat.is_some() || b.flat.is_some() {
            if self.width_passes(i, j) {
                return true;
            }
            if a.flat.is_some() && b.flat.is_some() {
                return false;
            }
        }
        (0..self.offered.len()).any(|m| {
            self.offered[m].flat.is_some()
                && (m == i || self.width_passes(m, i))
                && (m == j || self.width_passes(m, j))
        })
    }

    /// Whether a record with no plane of its own passes (1) with some flat record, retaining an
    /// identity or not.
    fn attaches_somewhere(&self, i: usize) -> bool {
        (0..self.offered.len())
            .any(|m| m != i && self.offered[m].flat.is_some() && self.width_passes(m, i))
    }

    /// The axis rule: both with no plane of their own on one axis plane, first points equal.
    fn axis_shared(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.offered[i], &self.offered[j]);
        a.flat.is_none()
            && b.flat.is_none()
            && a.axes
                .iter()
                .any(|(k, v)| b.axes.iter().any(|(k2, v2)| k == k2 && v == v2))
    }

    /// `near_parallel@1`'s test, admitting both outcomes within `2^-40 (high + 1)`.
    fn near_parallel_may(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.offered[i], &self.offered[j]);
        let (Some(pa), Some(pb)) = (a.flat, b.flat) else {
            return false;
        };
        let (na, nb) = (pa.unrounded_n, pb.unrounded_n);
        let c = crate::depth::cross(na, nb);
        let s = crate::depth::norm(c);
        let dotab = ((na[0] * nb[0] + na[1] * nb[1]) + na[2] * nb[2]).abs();
        let theta = libm::atan2(s, dotab);
        let high = theta * (1.0 + 16.0 * U) + 16.0 * U;
        let off = if s >= 32.0 * U {
            40.0 * U / s + 16.0 * U
        } else {
            f64::INFINITY
        };
        let axis = if s > 0.0 { c.map(|v| v / s) } else { [0.0; 3] };
        let theta_minus = |r: &Offered, n: [f64; 3]| {
            let along = crate::depth::cross(n, axis);
            let first = r.points[0];
            let ts: Vec<f64> = r
                .points
                .iter()
                .map(|p| {
                    let d = [p[0] - first[0], p[1] - first[1], p[2] - first[2]];
                    (d[0] * along[0] + d[1] * along[1]) + d[2] * along[2]
                })
                .collect();
            let lo = ts.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = ts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let far = r
                .points
                .iter()
                .map(|p| crate::depth::norm([p[0] - first[0], p[1] - first[1], p[2] - first[2]]))
                .fold(0.0, f64::max);
            let extent = if s > 0.0 { hi - lo } else { 2.0 * far };
            let read = if off <= 1.0 {
                extent + 2.0 * far * (off + 16.0 * U)
            } else {
                f64::INFINITY
            };
            let span = read.min(2.0 * far) * (1.0 + 8.0 * U);
            (2.0 * r.tau_own / span) * (1.0 - 2.0 * U)
        };
        let rhs = (theta_minus(a, na) + theta_minus(b, nb)) * (1.0 - 2.0 * U);
        high <= rhs + (high + 1.0) / 1_099_511_627_776.0
    }

    /// The edges and label of fragments `a`, `b`, both present at a sample, from their reads
    /// there (contract §7, "Oracle reading: the edges a pair admits"). `Err` where a read is not
    /// bounded.
    pub fn pair(
        &self,
        a: usize,
        b: usize,
        ra: &Reads,
        rb: &Reads,
        lines_share: (f64, f64),
    ) -> Result<PairAt, Unable> {
        self.pair_with(a, b, ra, rb, lines_share, true)
            .map(|(p, _)| p)
    }

    /// The pair's relation as the plate model orders it (contract §10, `facing_plane@1`), read in
    /// the same scenarios from the keys without the input snap, which the core's own keys are, so
    /// that the snap's shares in the policies' terms are 0 as well.
    pub fn plate_pair(
        &self,
        a: usize,
        b: usize,
        ra: &Reads,
        rb: &Reads,
        lines_share: (f64, f64),
    ) -> Result<Plate, Unable> {
        let unsnapped = |r: &Reads| {
            let keep = |b: Band| Band {
                s_minus: 0.0,
                s_plus: 0.0,
                ..b
            };
            Reads {
                own: r.own.map(keep),
                row: r.row.map(|w| w.map(keep)),
                ..*r
            }
        };
        self.pair_with(a, b, &unsnapped(ra), &unsnapped(rb), lines_share, false)
            .map(|(_, plate)| plate)
    }

    /// The pair's verdict, and the relation its keys establish (`Plate`) in the scenarios read.
    fn pair_with(
        &self,
        a: usize,
        b: usize,
        ra: &Reads,
        rb: &Reads,
        lines_share: (f64, f64),
        snap: bool,
    ) -> Result<(PairAt, Plate), Unable> {
        let snap_of = |o: &Offered| if snap { o.snap } else { 0.0 };
        let (fa, fb) = (&self.frags[a], &self.frags[b]);
        // A fragment with no key bounded at all is `truth_unable` whatever the pair reads,
        // its own record's other layer included (`band_unbounded`, §6, by fragment; a pivot with no
        // move, §2). A `curve_recut` read is unable only where a scenario reads it (§6, per read).
        for r in [ra, rb] {
            if let Err(u @ (Unable::BandUnbounded | Unable::PivotUnknown)) = r.own {
                return Err(u);
            }
        }
        if own_layers(fa, fb) {
            return Ok((
                PairAt {
                    edges: Edges::WEAK,
                    label: Label::OwnLayers,
                    via: [Via::Own; 2],
                },
                Plate::Tie,
            ));
        }
        if let (Ok(x), Ok(y)) = (ra.own, rb.own)
            && x.o == f64::NEG_INFINITY
            && y.o == f64::NEG_INFINITY
        {
            return Ok((
                PairAt {
                    edges: Edges::WEAK,
                    label: Label::ExactTie,
                    via: [Via::Own; 2],
                },
                Plate::Tie,
            ));
        }
        let (ia, ib) = (fa.offered, fb.offered);
        let both = ia.zip(ib);
        // A record with no plane of its own that may share a flat identity: every edge.
        if let Some((i, j)) = both {
            let planeless = self.offered[i].flat.is_none() || self.offered[j].flat.is_none();
            if planeless && self.may_share_flat(i, j) {
                // Its rows hang on what the registry settles, which the facts do not hold.
                return Ok((
                    PairAt {
                        edges: Edges::ALL,
                        label: Label::OnePlane,
                        via: [Via::Flat; 2],
                    },
                    Plate::Undetermined,
                ));
            }
        }
        // The scenarios §7's sets allow, each giving its edges. Where every scenario reads the two
        // keys as one computation, the pair is an `ExactTie` (§7): weak alone.
        let mut edges = Edges::default();
        let mut tie_meet = false;
        let mut one_computation = true;
        let mut policy: Option<Label> = None;
        let mut not_shared = true;
        let mut via = [Via::Own; 2];
        let mut plate: Option<Plate> = None;
        if let Some((i, j)) = both {
            let (ra_, rb_) = (&self.offered[i], &self.offered[j]);
            if ra_.flat.is_some() && rb_.flat.is_some() && self.may_share_flat(i, j) {
                // Shared: each reads its row, or its own key where it has none (a mesh triangle,
                // a point block: §5); `within_term` with the snap shares.
                let read = |r: &Reads| match r.row {
                    Some(w) => w.map(|w| (w, r.row_at)),
                    None => r.own.map(|o| (o, r.own_at)),
                };
                let ((wa, ca), (wb, cb)) = (read(ra)?, read(rb)?);
                let one = ca.is_some() && ca == cb;
                one_computation &= one;
                let mut e = by_depth(&wa, &wb);
                let mut held = false;
                if one {
                    e = Edges::WEAK;
                } else if policy_term(&wa, snap_of(ra_), &wb, snap_of(rb_)) {
                    e.weak = true;
                    held = true;
                    policy.get_or_insert(Label::OnePlane);
                }
                fold(
                    &mut plate,
                    if one {
                        Plate::Tie
                    } else {
                        Plate::of(&wa, &wb, held)
                    },
                );
                tie_meet |= meet(&wa, &wb);
                if ra.row.is_some() || rb.row.is_some() {
                    for (v, strong) in via.iter_mut().zip([e.a_front, e.b_front]) {
                        if strong {
                            *v = Via::Row;
                        }
                    }
                }
                edges = edges.union(e);
            }
            if self.axis_shared(i, j) {
                // Two axis-plane rows read at one sample are one computation.
                edges = edges.union(Edges::WEAK);
                fold(&mut plate, Plate::Tie);
                if !(self.attaches_somewhere(i) || self.attaches_somewhere(j)) {
                    not_shared = false;
                }
            }
        }
        if not_shared {
            let (oa, ob) = (ra.own?, rb.own?);
            let one = ra.own_at.is_some() && ra.own_at == rb.own_at;
            one_computation &= one;
            let mut e = if one { Edges::WEAK } else { by_depth(&oa, &ob) };
            let met = e.weak;
            tie_meet |= meet(&oa, &ob);
            if let Some((i, j)) = both {
                let (ra_, rb_) = (&self.offered[i], &self.offered[j]);
                if ra_.flat.is_some()
                    && rb_.flat.is_some()
                    && !e.weak
                    && self.near_parallel_may(i, j)
                    && policy_term(&oa, snap_of(ra_), &ob, snap_of(rb_))
                {
                    e.weak = true;
                    policy.get_or_insert(Label::NearParallel);
                }
                if self.lines_eligible(a, b)
                    && !e.weak
                    && policy_term(&oa, lines_share.0, &ob, lines_share.1)
                {
                    e.weak = true;
                    policy.get_or_insert(Label::Lines);
                }
            }
            fold(
                &mut plate,
                if one {
                    Plate::Tie
                } else {
                    Plate::of(&oa, &ob, e.weak && !met)
                },
            );
            edges = edges.union(e);
        }
        let label = if one_computation {
            Label::ExactTie
        } else if edges.count() == 1 && !edges.weak {
            Label::Depth(edges.a_front)
        } else if tie_meet {
            Label::TieBand
        } else {
            policy.unwrap_or(Label::TieBand)
        };
        Ok((
            PairAt { edges, label, via },
            plate.unwrap_or(Plate::Undetermined),
        ))
    }
}

/// Whether `order` (front first, indices into `present`) is admissible at a sample, given each
/// pair's edges (`edges(i, j)` for positions `i < j` of `present`) and paint order. `None` where
/// the completion search would exceed its bound.
pub fn admissible(
    present: &[usize],
    paint: &dyn Fn(usize) -> (u32, u32),
    edges: &dyn Fn(usize, usize) -> Edges,
    order: &[usize],
) -> Option<bool> {
    let n = present.len();
    if order.len() != n {
        return Some(false);
    }
    let pos: HashMap<usize, usize> = order.iter().enumerate().map(|(i, f)| (*f, i)).collect();
    if present.iter().any(|f| !pos.contains_key(f)) {
        return Some(false);
    }
    // The sufficient test: every pair agrees with some edge it admits.
    let mut all_agree = true;
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (present[i], present[j]);
            let e = edges(i, j);
            let a_first = pos[&a] < pos[&b];
            let paint_a_front = paint(a) > paint(b);
            let ok = (a_first && e.a_front)
                || (!a_first && e.b_front)
                || (e.weak && a_first == paint_a_front);
            if !ok {
                all_agree = false;
            }
        }
    }
    if all_agree {
        return Some(true);
    }
    // Every completion's rank, within a bound.
    let pairs: Vec<(usize, usize, Vec<u8>)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            let e = edges(i, j);
            let mut opts = Vec::new();
            if e.weak {
                opts.push(0);
            }
            if e.a_front {
                opts.push(1);
            }
            if e.b_front {
                opts.push(2);
            }
            (i, j, opts)
        })
        .collect();
    if pairs.iter().any(|(_, _, o)| o.is_empty()) {
        return Some(false);
    }
    let total: f64 = pairs.iter().map(|(_, _, o)| o.len() as f64).product();
    if total > 262_144.0 {
        return None;
    }
    let target: Vec<usize> = order
        .iter()
        .map(|f| present.iter().position(|p| p == f).unwrap())
        .collect();
    let mut choice = vec![0usize; pairs.len()];
    loop {
        // front[i][j]: whether i is in front of j, and strong[i][j] whether strongly.
        let mut front = vec![vec![false; n]; n];
        let mut strong = vec![vec![false; n]; n];
        for (p, (i, j, opts)) in pairs.iter().enumerate() {
            let (i, j) = (*i, *j);
            match opts[choice[p]] {
                0 => {
                    let i_front = paint(present[i]) > paint(present[j]);
                    front[i][j] = i_front;
                    front[j][i] = !i_front;
                }
                1 => {
                    front[i][j] = true;
                    strong[i][j] = true;
                }
                _ => {
                    front[j][i] = true;
                    strong[j][i] = true;
                }
            }
        }
        if rank(n, &front, &strong, &|i| paint(present[i])) == target {
            return Some(true);
        }
        let mut p = 0;
        loop {
            if p == pairs.len() {
                return Some(false);
            }
            choice[p] += 1;
            if choice[p] < pairs[p].2.len() {
                break;
            }
            choice[p] = 0;
            p += 1;
        }
    }
}

/// `cycle_rank@1`'s rank of a completion, front first: where one order agrees with every pair,
/// that order; otherwise, among the remaining fragments no remaining one is strongly in front
/// of, the later painted; where every remaining one has a strong edge into it, the one in front
/// of the most remaining, the later painted on ties.
pub fn rank(
    n: usize,
    front: &[Vec<bool>],
    strong: &[Vec<bool>],
    paint: &dyn Fn(usize) -> (u32, u32),
) -> Vec<usize> {
    // One order: insert each before the first one it is in front of.
    let mut one: Vec<usize> = Vec::new();
    for (i, row) in front.iter().enumerate().take(n) {
        let at = one.iter().position(|&j| row[j]).unwrap_or(one.len());
        one.insert(at, i);
    }
    let agrees = (0..n).all(|x| {
        (0..n).all(|y| {
            x == y || {
                let px = one.iter().position(|&v| v == x).unwrap();
                let py = one.iter().position(|&v| v == y).unwrap();
                front[x][y] == (px < py)
            }
        })
    });
    if agrees {
        return one;
    }
    let mut left: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    while !left.is_empty() {
        let free: Vec<usize> = left
            .iter()
            .copied()
            .filter(|&x| !left.iter().any(|&y| y != x && strong[y][x]))
            .collect();
        let pick = if !free.is_empty() {
            *free.iter().max_by_key(|&&x| paint(x)).unwrap()
        } else {
            *left
                .iter()
                .max_by_key(|&&x| {
                    let wins = left.iter().filter(|&&y| y != x && front[x][y]).count();
                    (wins, paint(x))
                })
                .unwrap()
        };
        out.push(pick);
        left.retain(|&x| x != pick);
    }
    out
}
