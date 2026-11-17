//! Cylinder narrowphase checked against double-precision references.
//!
//! A miss is proven by a witness: alternating projections in f64 onto both shapes, each shrunk
//! by `BAND`, converge to a point inside both. A false hit is proven by the same iteration
//! settling at a positive distance between the shapes grown by `BAND`.

use glam::{DQuat, DVec3, EulerRot, Vec3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use wreck::{Capsule, Collides, Cuboid, Cylinder};

const BAND: f64 = 1e-4;

#[derive(Clone, Copy)]
struct Tube {
    a: DVec3,
    d: DVec3,
    r: f64,
    round: bool,
}

#[derive(Clone, Copy)]
struct Brick {
    c: DVec3,
    ax: [DVec3; 3],
    he: DVec3,
}

#[derive(Clone, Copy)]
enum Solid {
    Tube(Tube),
    Brick(Brick),
}

impl Solid {
    fn project(&self, p: DVec3) -> DVec3 {
        match self {
            Solid::Brick(b) => {
                let w = p - b.c;
                (0..3).fold(b.c, |o, i| {
                    o + b.ax[i] * w.dot(b.ax[i]).clamp(-b.he[i], b.he[i])
                })
            }
            Solid::Tube(t) => {
                let l = t.d.length();
                let u = t.d / l;
                let w = p - t.a;
                let s = w.dot(u);
                if t.round {
                    let on_axis = t.a + u * s.clamp(0.0, l);
                    let off = p - on_axis;
                    let n = off.length();
                    if n > t.r {
                        on_axis + off * (t.r / n)
                    } else {
                        p
                    }
                } else {
                    let perp = w - u * s;
                    let n = perp.length();
                    let perp = if n > t.r { perp * (t.r / n) } else { perp };
                    t.a + u * s.clamp(0.0, l) + perp
                }
            }
        }
    }

    fn start(&self) -> DVec3 {
        match self {
            Solid::Brick(b) => b.c,
            Solid::Tube(t) => t.a + t.d * 0.5,
        }
    }

    fn resized(&self, by: f64) -> Solid {
        match *self {
            Solid::Brick(b) => Solid::Brick(Brick {
                he: (b.he + DVec3::splat(by)).max(DVec3::ZERO),
                ..b
            }),
            Solid::Tube(t) => Solid::Tube(Tube {
                r: (t.r + by).max(0.0),
                ..t
            }),
        }
    }
}

fn distance(a: &Solid, b: &Solid) -> f64 {
    let mut x = b.project(a.start());
    let mut last = f64::INFINITY;
    for _ in 0..100_000 {
        let y = a.project(x);
        let nx = b.project(y);
        let d = (nx - y).length();
        if d < 1e-10 || (last - d).abs() < 1e-14 {
            return d;
        }
        last = d;
        x = nx;
    }
    last
}

fn tube(c: &Cylinder) -> Tube {
    Tube {
        a: c.p1.as_dvec3(),
        d: c.dir.as_dvec3(),
        r: c.radius as f64,
        round: false,
    }
}

fn brick(c: &Cuboid) -> Brick {
    Brick {
        c: c.center.as_dvec3(),
        ax: c.axes.map(|a| a.as_dvec3()),
        he: Vec3::from(c.half_extents).as_dvec3(),
    }
}

fn unit(rng: &mut StdRng) -> DVec3 {
    loop {
        let v = DVec3::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        if v.length_squared() > 1e-3 {
            return v.normalize();
        }
    }
}

fn random_plate(rng: &mut StdRng) -> Cuboid {
    let he = Vec3::new(
        rng.random_range(0.05..1.5),
        rng.random_range(0.05..0.8),
        rng.random_range(0.003..0.05),
    );
    let q = if rng.random_bool(0.5) {
        DQuat::IDENTITY
    } else {
        DQuat::from_euler(
            EulerRot::XYZ,
            rng.random_range(-3.0..3.0),
            rng.random_range(-3.0..3.0),
            rng.random_range(-3.0..3.0),
        )
    };
    let c = Vec3::new(
        rng.random_range(-3.0..3.0),
        rng.random_range(-1.0..1.0),
        rng.random_range(0.0..1.0),
    );
    Cuboid::new(
        c,
        [DVec3::X, DVec3::Y, DVec3::Z].map(|a| (q * a).as_vec3()),
        he.to_array(),
    )
}

/// A cylinder whose middle sits within about a radius of `near`.
fn random_cylinder_near(rng: &mut StdRng, near: DVec3) -> Cylinder {
    let r = rng.random_range(0.005..0.06);
    let len = rng.random_range(0.01..0.3);
    let u = unit(rng);
    let mid = near + unit(rng) * rng.random_range(0.0..1.8) * (r + 0.004);
    Cylinder::new(
        (mid - u * len * 0.5).as_vec3(),
        (mid + u * len * 0.5).as_vec3(),
        r as f32,
    )
}

fn point_on_plate(rng: &mut StdRng, plate: &Cuboid) -> DVec3 {
    let b = brick(plate);
    let mut s = [0.0; 3];
    for (i, edge_bias) in [0.5, 0.5, 0.7].into_iter().enumerate() {
        s[i] = if rng.random_bool(edge_bias) {
            if rng.random_bool(0.5) { 1.0 } else { -1.0 }
        } else {
            rng.random_range(-1.0..1.0)
        };
    }
    b.c + b.ax[0] * b.he.x * s[0] + b.ax[1] * b.he.y * s[1] + b.ax[2] * b.he.z * s[2]
}

fn check(name: &str, hit: bool, a: Solid, b: Solid, describe: impl Fn() -> String) {
    if hit {
        let gap = distance(&a.resized(BAND), &b);
        assert!(
            gap < 1e-5,
            "{name}: reported contact across a gap of {gap:e} plus {BAND:e}: {}",
            describe()
        );
    } else {
        let gap = distance(&a.resized(-BAND), &b);
        assert!(
            gap > 1e-9,
            "{name}: missed an overlap deeper than {BAND:e}: {}",
            describe()
        );
    }
}

#[test]
fn a_barrel_cutting_a_plate_is_found() {
    let p1 = Vec3::new(2.1547298, -0.23941937, 0.43518144);
    let dir = Vec3::new(0.017233849, 0.02432762, 0.037789553);
    let barrel = Cylinder::new(p1, p1 + dir, 0.035804093);
    let plate = Cuboid::new(
        Vec3::new(1.10408, -0.763669, 0.4024439),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [1.0744462, 0.4978862, 0.039178587],
    );
    assert!(barrel.collides(&plate));
    assert!(plate.collides(&barrel));
}

#[test]
fn cylinder_cuboid_matches_the_reference() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..60_000 {
        let plate = random_plate(&mut rng);
        let near = point_on_plate(&mut rng, &plate);
        let cyl = random_cylinder_near(&mut rng, near);
        check(
            "cylinder-cuboid",
            cyl.collides(&plate),
            Solid::Tube(tube(&cyl)),
            Solid::Brick(brick(&plate)),
            || format!("{cyl:?} {plate:?}"),
        );
    }
}

#[test]
fn a_hit_survives_growing_the_cuboid() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..200_000 {
        let plate = random_plate(&mut rng);
        let near = point_on_plate(&mut rng, &plate);
        let cyl = random_cylinder_near(&mut rng, near);
        if !cyl.collides(&plate) {
            continue;
        }
        let grow = rng.random_range(1e-4..6e-3f32);
        let grown = Cuboid::new(
            plate.center,
            plate.axes,
            plate.half_extents.map(|h| h + grow),
        );
        assert!(
            cyl.collides(&grown),
            "lost {cyl:?} against {plate:?} grown by {grow}"
        );
    }
}

#[test]
fn cylinder_cylinder_matches_the_reference() {
    let mut rng = StdRng::seed_from_u64(13);
    for _ in 0..40_000 {
        let c = DVec3::new(
            rng.random_range(-3.0..3.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(0.0..1.0),
        );
        let a = random_cylinder_near(&mut rng, c);
        let b = random_cylinder_near(&mut rng, c);
        check(
            "cylinder-cylinder",
            a.collides(&b),
            Solid::Tube(tube(&a)),
            Solid::Tube(tube(&b)),
            || format!("{a:?} {b:?}"),
        );
    }
}

#[test]
fn cylinder_capsule_matches_the_reference() {
    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..40_000 {
        let c = DVec3::new(
            rng.random_range(-3.0..3.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(0.0..1.0),
        );
        let a = random_cylinder_near(&mut rng, c);
        let spine = random_cylinder_near(&mut rng, c);
        let cap = Capsule::new(spine.p1, spine.p2(), spine.radius);
        check(
            "cylinder-capsule",
            a.collides(&cap),
            Solid::Tube(tube(&a)),
            Solid::Tube(Tube {
                round: true,
                ..tube(&spine)
            }),
            || format!("{a:?} {cap:?}"),
        );
    }
}

#[test]
fn capsule_capsule_matches_the_reference() {
    let mut rng = StdRng::seed_from_u64(19);
    for _ in 0..40_000 {
        let c = DVec3::new(
            rng.random_range(-3.0..3.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(0.0..1.0),
        );
        let a = random_cylinder_near(&mut rng, c);
        let b = random_cylinder_near(&mut rng, c);
        let round = |c: &Cylinder| {
            Solid::Tube(Tube {
                round: true,
                ..tube(c)
            })
        };
        let hit =
            Capsule::new(a.p1, a.p2(), a.radius).collides(&Capsule::new(b.p1, b.p2(), b.radius));
        check("capsule-capsule", hit, round(&a), round(&b), || {
            format!("{a:?} {b:?}")
        });
    }
}
