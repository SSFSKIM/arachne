//! The determinism probe. Builds a 2D and a 3D world from literal numbers, steps each for 600
//! ticks and prints the SHA-256 of its trace (every body's pose and velocity, and the pairs in
//! contact, at every tick); then each audited path's case, as the bits of its result. Its output
//! is compared between builds, processes and systems (README.md, "Determinism").

mod cases;
mod three;
mod trace;
mod two;

use sha2::{Digest, Sha256};

const TICKS: usize = 600;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn main() {
    let mut world = two::World::new();
    world.advance(TICKS);
    println!("world 2d {}", digest(&world.trace.0));
    println!("contacts 2d {} at the end", world.contacts());
    let mut world = three::World::new();
    world.advance(TICKS);
    println!("world 3d {}", digest(&world.trace.0));
    println!("contacts 3d {} at the end", world.contacts());
    for (name, bits) in cases::all() {
        println!("case {name} {bits}");
    }
}
