//! A world's bytes do not depend on how its steps are grouped into calls, on reads between
//! them, or on another world having run before it in the same process.

#[path = "../examples/probe/three.rs"]
mod three;
#[path = "../examples/probe/trace.rs"]
mod trace;
#[path = "../examples/probe/two.rs"]
mod two;

const TICKS: usize = 600;
const PIECES: [usize; 6] = [1, 7, 60, 2, 230, 300];

#[test]
fn a_2d_world_stepped_in_pieces_or_after_another_gives_the_same_bytes() {
    let mut whole = two::World::new();
    whole.advance(TICKS);
    let mut pieces = two::World::new();
    for ticks in PIECES {
        pieces.advance(ticks);
        assert!(pieces.contacts() < 1000);
    }
    let mut second = two::World::new();
    second.advance(TICKS);
    assert_eq!(PIECES.iter().sum::<usize>(), TICKS);
    assert!(
        whole.trace == pieces.trace,
        "stepping in pieces changed the bytes"
    );
    assert!(
        whole.trace == second.trace,
        "a second world gave other bytes"
    );
}

#[test]
fn a_3d_world_stepped_in_pieces_or_after_another_gives_the_same_bytes() {
    let mut whole = three::World::new();
    whole.advance(TICKS);
    let mut pieces = three::World::new();
    for ticks in PIECES {
        pieces.advance(ticks);
        assert!(pieces.contacts() < 1000);
    }
    let mut second = three::World::new();
    second.advance(TICKS);
    assert!(
        whole.trace == pieces.trace,
        "stepping in pieces changed the bytes"
    );
    assert!(
        whole.trace == second.trace,
        "a second world gave other bytes"
    );
}
