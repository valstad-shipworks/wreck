//! Fuzz the SIMD `capsule_vs_capsules_broad` batch (reached via `Collider::collides` for a
//! capsule query against a capsule-only collider) against the scalar single-pair reference.

use glam::Vec3;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use wreck::{Capsule, Collider, Collides};

fn rand_capsule(rng: &mut SmallRng, degenerate_chance: f64) -> Capsule {
    let p1 = Vec3::new(
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
        rng.random_range(-3.0..3.0),
    );
    let dir = if rng.random_bool(degenerate_chance) {
        Vec3::ZERO // zero-length capsule == sphere; exercises the a<=eps / e<=eps paths
    } else {
        Vec3::new(
            rng.random_range(-2.0..2.0),
            rng.random_range(-2.0..2.0),
            rng.random_range(-2.0..2.0),
        )
    };
    Capsule::new(p1, p1 + dir, rng.random_range(0.05..0.8))
}

#[test]
fn capsule_vs_capsule_collection_matches_scalar() {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    let mut hits = 0u32;
    for iter in 0..20_000 {
        let count = rng.random_range(0..24);
        let caps: Vec<Capsule> = (0..count).map(|_| rand_capsule(&mut rng, 0.1)).collect();

        let mut col: Collider = Collider::new();
        for c in &caps {
            col.add(*c);
        }

        let q = rand_capsule(&mut rng, 0.1);

        // reference: any single-pair capsule-capsule collision
        let want = caps.iter().any(|c| q.collides(c));
        // batch SIMD path
        let got = col.collides(&q);

        assert_eq!(
            got, want,
            "iter {iter}: q={q:?} mismatch (got {got}, want {want}) over {count} capsules"
        );
        if want {
            hits += 1;
        }
    }
    // sanity: the fuzz space actually exercises both outcomes
    assert!(hits > 1000, "too few positive cases ({hits}) — fuzz not covering hits");
}
