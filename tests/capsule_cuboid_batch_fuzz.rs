//! Fuzz the SIMD capsule↔cuboid cross-type batches (both query directions, via
//! `Collider::collides`) against the scalar single-pair narrowphase reference.

use glam::{Quat, Vec3};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::PI;
use wreck::{Capsule, Collider, Collides, Cuboid};

fn rand_capsule(rng: &mut SmallRng) -> Capsule {
    let p1 = Vec3::new(
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
    );
    let dir = if rng.random_bool(0.1) {
        Vec3::ZERO
    } else {
        Vec3::new(
            rng.random_range(-2.0..2.0),
            rng.random_range(-2.0..2.0),
            rng.random_range(-2.0..2.0),
        )
    };
    Capsule::new(p1, p1 + dir, rng.random_range(0.05..0.8))
}

fn rand_cuboid(rng: &mut SmallRng) -> Cuboid {
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
    if rng.random_bool(0.3) {
        Cuboid::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    } else {
        let q = Quat::from_euler(
            glam::EulerRot::XYZ,
            rng.random_range(-PI..PI),
            rng.random_range(-PI..PI),
            rng.random_range(-PI..PI),
        );
        Cuboid::new(center, [q * Vec3::X, q * Vec3::Y, q * Vec3::Z], he)
    }
}

#[test]
fn capsule_query_vs_cuboids_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xCAB0);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let cubs: Vec<Cuboid> = (0..count).map(|_| rand_cuboid(&mut rng)).collect();
        let mut col: Collider = Collider::new();
        for c in &cubs {
            col.add(*c);
        }
        let q = rand_capsule(&mut rng);
        let want = cubs.iter().any(|c| q.collides(c));
        let got = col.collides(&q);
        assert_eq!(got, want, "iter {iter}: capsule q vs {count} cuboids");
        if want {
            hits += 1;
        }
    }
    assert!(hits > 500, "too few positive cases ({hits})");
}

#[test]
fn cuboid_query_vs_capsules_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xCAB1);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let caps: Vec<Capsule> = (0..count).map(|_| rand_capsule(&mut rng)).collect();
        let mut col: Collider = Collider::new();
        for c in &caps {
            col.add(*c);
        }
        let q = rand_cuboid(&mut rng);
        let want = caps.iter().any(|c| q.collides(c));
        let got = col.collides(&q);
        assert_eq!(got, want, "iter {iter}: cuboid q vs {count} capsules");
        if want {
            hits += 1;
        }
    }
    assert!(hits > 500, "too few positive cases ({hits})");
}
