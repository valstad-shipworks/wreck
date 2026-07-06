//! Fuzz the SIMD `cuboid_vs_cuboids_broad` batch (via `Collider::collides` for a cuboid query
//! against a cuboid-only collider) against the scalar 15-axis SAT reference.

use glam::{Quat, Vec3};
use std::f32::consts::PI;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use wreck::{Collider, Collides, Cuboid};

fn rand_cuboid(rng: &mut SmallRng, axis_aligned_chance: f64) -> Cuboid {
    let center = Vec3::new(
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
    );
    let he = [
        rng.random_range(0.1..1.2),
        rng.random_range(0.1..1.2),
        rng.random_range(0.1..1.2),
    ];
    if rng.random_bool(axis_aligned_chance) {
        Cuboid::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    } else {
        let q = Quat::from_euler(
            glam::EulerRot::XYZ,
            rng.random_range(-PI..PI),
            rng.random_range(-PI..PI),
            rng.random_range(-PI..PI),
        );
        let axes = [q * Vec3::X, q * Vec3::Y, q * Vec3::Z];
        Cuboid::new(center, axes, he)
    }
}

#[test]
fn cuboid_vs_cuboid_collection_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0x5A7);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let cubs: Vec<Cuboid> = (0..count).map(|_| rand_cuboid(&mut rng, 0.3)).collect();
        let mut col: Collider = Collider::new();
        for c in &cubs {
            col.add(*c);
        }
        let q = rand_cuboid(&mut rng, 0.3);
        let want = cubs.iter().any(|c| q.collides(c));
        let got = col.collides(&q);
        assert_eq!(got, want, "iter {iter}: mismatch over {count} cuboids");
        if want {
            hits += 1;
        }
    }
    assert!(hits > 1000, "too few positive cases ({hits})");
}
