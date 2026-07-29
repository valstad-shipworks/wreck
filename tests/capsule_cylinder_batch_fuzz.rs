//! Fuzz the SIMD capsule↔cylinder cross-type batches (both query directions, via
//! `Collider::collides`) against the scalar single-pair narrowphase reference.

use glam::Vec3;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use wreck::{Capsule, Collider, Collides, Cylinder};

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

fn rand_cylinder(rng: &mut SmallRng) -> Cylinder {
    let p1 = Vec3::new(
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
    );
    let dir = if rng.random_bool(0.08) {
        Vec3::ZERO
    } else {
        Vec3::new(
            rng.random_range(-2.0..2.0),
            rng.random_range(-2.0..2.0),
            rng.random_range(-2.0..2.0),
        )
    };
    Cylinder::new(p1, p1 + dir, rng.random_range(0.05..0.8))
}

#[test]
fn capsule_query_vs_cylinders_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xCAC0);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let cyls: Vec<Cylinder> = (0..count).map(|_| rand_cylinder(&mut rng)).collect();
        let mut col: Collider = Collider::new();
        for c in &cyls {
            col.add(*c);
        }
        let q = rand_capsule(&mut rng);
        let want = cyls.iter().any(|c| q.collides(c));
        assert_eq!(
            col.collides(&q),
            want,
            "iter {iter}: capsule q vs {count} cylinders"
        );
        if want {
            hits += 1;
        }
    }
    assert!(hits > 500, "too few positive cases ({hits})");
}

#[test]
fn cylinder_query_vs_capsules_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xCAC1);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let caps: Vec<Capsule> = (0..count).map(|_| rand_capsule(&mut rng)).collect();
        let mut col: Collider = Collider::new();
        for c in &caps {
            col.add(*c);
        }
        let q = rand_cylinder(&mut rng);
        let want = caps.iter().any(|c| q.collides(c));
        assert_eq!(
            col.collides(&q),
            want,
            "iter {iter}: cylinder q vs {count} capsules"
        );
        if want {
            hits += 1;
        }
    }
    assert!(hits > 500, "too few positive cases ({hits})");
}
