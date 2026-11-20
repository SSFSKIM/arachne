//! Many compositors in one process: opened and dropped one after another, kept open together, and
//! drawing on several threads at once. A process that opens a compositor per size, per session or
//! per test opens many over its life, and a driver can run out of what each device costs long
//! before that.

mod common;

use std::sync::{Arc, Barrier};

use arachne_gpu::{Compositor, Footprint, Frame, Layer, NEAREST};
use common::{compositor, painted_picture, rectangle, upload};

/// How many compositors the first two probes open: past where a device for each ran out.
const MANY: usize = 100;
/// The side of their frames: small, so what is counted is compositors, not memory.
const SIDE: u32 = 48;

/// Draws one painted picture through `c` and asserts that it comes back with its bytes.
fn passes_a_picture_through(c: &mut Compositor, seed: u64) {
    let picture = painted_picture(SIDE, SIDE, seed);
    let key = upload(c, SIDE, SIDE, &picture);
    let drawn = c
        .draw(&Frame::new([0; 4], vec![Layer::picture(key)]))
        .unwrap();
    assert!(
        drawn == picture,
        "compositor {seed}: the picture came back changed"
    );
}

#[test]
fn a_process_opens_and_drops_a_hundred_compositors() {
    for seed in 0..MANY as u64 {
        let mut c = compositor(SIDE, SIDE);
        passes_a_picture_through(&mut c, seed);
    }
}

#[test]
fn a_process_keeps_a_hundred_compositors_open_and_each_draws() {
    let mut open: Vec<Compositor> = (0..MANY).map(|_| compositor(SIDE, SIDE)).collect();
    for (seed, c) in open.iter_mut().enumerate() {
        passes_a_picture_through(c, seed as u64);
    }
    // The first still draws once the last has drawn.
    passes_a_picture_through(&mut open[0], 0);
}

const THREADS: usize = 8;
const ROUNDS: usize = 10;
const FRAME: (u32, u32) = (96, 96);

/// A frame through every path a draw takes: a picture, a blurred quad under a feathered mask, and
/// a window; each named, for the object map.
fn busy_frame(c: &mut Compositor) -> Frame {
    let (w, h) = FRAME;
    let below = upload(c, w, h, &painted_picture(w, h, 1));
    let small = upload(c, 8, 8, &common::opaque(painted_picture(8, 8, 2)));
    let mask: Vec<u8> = (0..w * h)
        .flat_map(|i| if i % w < w / 2 { [255; 4] } else { [0; 4] })
        .collect();
    let mask = upload(c, w, h, &mask);
    let inner = Frame::new(
        [10, 20, 30, 255],
        vec![Layer::quad(small, rectangle(4.0, 4.0, 40.0, 40.0), NEAREST)],
    );
    Frame::new(
        [0; 4],
        vec![
            Layer::picture(below).named("below"),
            Layer::quad(small, rectangle(10.0, 12.0, 70.0, 60.0), NEAREST)
                .with_blur(2.0)
                .with_mask(mask, 6.0, false)
                .named("blurred"),
            Layer::window(inner, rectangle(50.0, 50.0, 92.0, 90.0), NEAREST)
                .with_opacity(0.5)
                .named("window"),
        ],
    )
}

fn draw_and_map(c: &mut Compositor) -> (Vec<u8>, Vec<Footprint>) {
    let frame = busy_frame(c);
    (c.draw(&frame).unwrap(), c.object_map(&frame).unwrap())
}

#[test]
fn compositors_on_several_threads_draw_at_once_and_each_draws_what_one_alone_draws() {
    let (w, h) = FRAME;
    let expected = Arc::new(draw_and_map(&mut compositor(w, h)));
    // Every thread opens its own compositor at the same time as the others, then all draw at once.
    let start = Arc::new(Barrier::new(THREADS));
    let threads: Vec<_> = (0..THREADS)
        .map(|thread| {
            let (start, expected) = (Arc::clone(&start), Arc::clone(&expected));
            std::thread::spawn(move || {
                let mut c = compositor(w, h);
                start.wait();
                for round in 0..ROUNDS {
                    let drawn = draw_and_map(&mut c);
                    assert!(
                        drawn.0 == expected.0,
                        "thread {thread}, round {round}: another frame"
                    );
                    assert_eq!(drawn.1, expected.1, "thread {thread}, round {round}");
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("a drawing thread failed");
    }
}
