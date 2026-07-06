//! Fuzz the SIMD cuboid↔cylinder cross-type batches (both query directions, via
//! `Collider::collides`) against the scalar single-pair narrowphase reference.

use glam::{Quat, Vec3};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::PI;
use wreck::{Collider, Collides, Cuboid, Cylinder};

fn rand_cylinder(rng: &mut SmallRng) -> Cylinder {
    let p1 = Vec3::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0));
    let dir = if rng.random_bool(0.08) { Vec3::ZERO } else {
        Vec3::new(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0))
    };
    Cylinder::new(p1, p1 + dir, rng.random_range(0.05..0.8))
}

fn rand_cuboid(rng: &mut SmallRng) -> Cuboid {
    let center = Vec3::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0));
    let he = [rng.random_range(0.1..1.2), rng.random_range(0.1..1.2), rng.random_range(0.1..1.2)];
    if rng.random_bool(0.3) {
        Cuboid::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    } else {
        let q = Quat::from_euler(glam::EulerRot::XYZ, rng.random_range(-PI..PI), rng.random_range(-PI..PI), rng.random_range(-PI..PI));
        Cuboid::new(center, [q * Vec3::X, q * Vec3::Y, q * Vec3::Z], he)
    }
}

#[test]
fn cylinder_query_vs_cuboids_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xC0C0);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let cubs: Vec<Cuboid> = (0..count).map(|_| rand_cuboid(&mut rng)).collect();
        let mut col: Collider = Collider::new();
        for c in &cubs { col.add(*c); }
        let q = rand_cylinder(&mut rng);
        let want = cubs.iter().any(|c| q.collides(c));
        assert_eq!(col.collides(&q), want, "iter {iter}: cylinder q vs {count} cuboids");
        if want { hits += 1; }
    }
    assert!(hits > 500, "too few positive cases ({hits})");
}

#[test]
fn cuboid_query_vs_cylinders_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xC0C1);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let cyls: Vec<Cylinder> = (0..count).map(|_| rand_cylinder(&mut rng)).collect();
        let mut col: Collider = Collider::new();
        for c in &cyls { col.add(*c); }
        let q = rand_cuboid(&mut rng);
        let want = cyls.iter().any(|c| q.collides(c));
        assert_eq!(col.collides(&q), want, "iter {iter}: cuboid q vs {count} cylinders");
        if want { hits += 1; }
    }
    assert!(hits > 500, "too few positive cases ({hits})");
}
