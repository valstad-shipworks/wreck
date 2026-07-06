//! Fuzz the SIMD `cylinder_vs_cylinders_broad` batch (via `Collider::collides` for a cylinder
//! query against a cylinder-only collider) against the scalar single-pair reference.

use glam::Vec3;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use wreck::{Collider, Collides, Cylinder};

fn rand_cylinder(rng: &mut SmallRng, degen: f64) -> Cylinder {
    let p1 = Vec3::new(
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
    );
    let dir = if rng.random_bool(degen) {
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
fn cylinder_vs_cylinder_collection_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xBEEF);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let cyls: Vec<Cylinder> = (0..count).map(|_| rand_cylinder(&mut rng, 0.08)).collect();
        let mut col: Collider = Collider::new();
        for c in &cyls {
            col.add(*c);
        }
        let q = rand_cylinder(&mut rng, 0.08);
        let want = cyls.iter().any(|c| q.collides(c));
        let got = col.collides(&q);
        assert_eq!(got, want, "iter {iter}: mismatch over {count} cylinders, q={q:?}");
        if want {
            hits += 1;
        }
    }
    assert!(hits > 1000, "too few positive cases ({hits})");
}
