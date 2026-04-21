#![cfg(feature = "sdf")]

use criterion::{Criterion, criterion_group, criterion_main};
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use std::hint::black_box;

use wreck::sdf::gjk_epa_signed_distance;
use wreck::{Capsule, Cuboid, Line, LineSegment, Plane, Point, Ray, SignedDistance, Sphere};

fn rand_vec3(rng: &mut SmallRng, range: f32) -> Vec3 {
    Vec3::new(
        rng.random_range(-range..range),
        rng.random_range(-range..range),
        rng.random_range(-range..range),
    )
}

fn rand_sphere(rng: &mut SmallRng) -> Sphere {
    Sphere::new(rand_vec3(rng, 5.0), rng.random_range(0.1..1.0))
}

fn rand_capsule(rng: &mut SmallRng) -> Capsule {
    let p1 = rand_vec3(rng, 5.0);
    let p2 = p1 + rand_vec3(rng, 2.0);
    Capsule::new(p1, p2, rng.random_range(0.1..0.5))
}

fn rand_cuboid(rng: &mut SmallRng) -> Cuboid {
    let center = rand_vec3(rng, 5.0);
    let he = [
        rng.random_range(0.2..1.5),
        rng.random_range(0.2..1.5),
        rng.random_range(0.2..1.5),
    ];
    let quat = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
    );
    Cuboid::new(
        center,
        [quat * Vec3::X, quat * Vec3::Y, quat * Vec3::Z],
        he,
    )
}

fn rand_aabb(rng: &mut SmallRng) -> Cuboid {
    let center = rand_vec3(rng, 5.0);
    Cuboid::new(
        center,
        [Vec3::X, Vec3::Y, Vec3::Z],
        [
            rng.random_range(0.2..1.5),
            rng.random_range(0.2..1.5),
            rng.random_range(0.2..1.5),
        ],
    )
}

const N: usize = 128;

fn generate<T>(seed: u64, mut f: impl FnMut(&mut SmallRng) -> T) -> Vec<T> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..N).map(|_| f(&mut rng)).collect()
}

fn bench_sphere_sphere(c: &mut Criterion) {
    let a = generate(1, rand_sphere);
    let b = generate(2, rand_sphere);
    c.bench_function("sdf/sphere-sphere/closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/sphere-sphere/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
}

fn bench_sphere_cuboid(c: &mut Criterion) {
    let a = generate(3, rand_sphere);
    let b = generate(4, rand_cuboid);
    c.bench_function("sdf/sphere-cuboid/closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/sphere-cuboid/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
}

fn bench_capsule_capsule(c: &mut Criterion) {
    let a = generate(5, rand_capsule);
    let b = generate(6, rand_capsule);
    c.bench_function("sdf/capsule-capsule/closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/capsule-capsule/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
}

fn bench_cuboid_cuboid(c: &mut Criterion) {
    let a = generate(7, rand_cuboid);
    let b = generate(8, rand_cuboid);
    c.bench_function("sdf/cuboid-cuboid/sat", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/cuboid-cuboid/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
    let aa = generate(9, rand_aabb);
    let bb = generate(10, rand_aabb);
    c.bench_function("sdf/aabb-aabb/sat", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in aa.iter().zip(bb.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
}

fn bench_parametric(c: &mut Criterion) {
    let lines: Vec<Line> = {
        let mut rng = SmallRng::seed_from_u64(11);
        (0..N)
            .map(|_| Line::new(rand_vec3(&mut rng, 5.0), rand_vec3(&mut rng, 1.0)))
            .collect()
    };
    let spheres = generate(12, rand_sphere);
    let cubes = generate(13, rand_cuboid);
    let points: Vec<Point> = {
        let mut rng = SmallRng::seed_from_u64(14);
        (0..N).map(|_| Point(rand_vec3(&mut rng, 5.0))).collect()
    };
    let planes: Vec<Plane> = {
        let mut rng = SmallRng::seed_from_u64(15);
        (0..N)
            .map(|_| {
                let n = rand_vec3(&mut rng, 1.0).normalize_or_zero();
                let n = if n.length_squared() < 0.5 { Vec3::Y } else { n };
                Plane::new(n, rng.random_range(-2.0..2.0))
            })
            .collect()
    };

    c.bench_function("sdf/line-sphere", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, s) in lines.iter().zip(spheres.iter()) {
                acc += black_box(l).signed_distance(black_box(s));
            }
            acc
        })
    });
    c.bench_function("sdf/line-cuboid", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, cube) in lines.iter().zip(cubes.iter()) {
                acc += black_box(l).signed_distance(black_box(cube));
            }
            acc
        })
    });
    c.bench_function("sdf/line-point", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, p) in lines.iter().zip(points.iter()) {
                acc += black_box(l).signed_distance(black_box(p));
            }
            acc
        })
    });
    c.bench_function("sdf/line-plane", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, p) in lines.iter().zip(planes.iter()) {
                acc += black_box(l).signed_distance(black_box(p));
            }
            acc
        })
    });

    let rays: Vec<Ray> = {
        let mut rng = SmallRng::seed_from_u64(16);
        (0..N)
            .map(|_| Ray::new(rand_vec3(&mut rng, 5.0), rand_vec3(&mut rng, 1.0)))
            .collect()
    };
    c.bench_function("sdf/ray-sphere", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (r, s) in rays.iter().zip(spheres.iter()) {
                acc += black_box(r).signed_distance(black_box(s));
            }
            acc
        })
    });

    let segs: Vec<LineSegment> = {
        let mut rng = SmallRng::seed_from_u64(17);
        (0..N)
            .map(|_| {
                let p1 = rand_vec3(&mut rng, 5.0);
                LineSegment::new(p1, p1 + rand_vec3(&mut rng, 2.0))
            })
            .collect()
    };
    c.bench_function("sdf/segment-segment", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (s1, s2) in segs.iter().zip(segs.iter().rev()) {
                acc += black_box(s1).signed_distance(black_box(s2));
            }
            acc
        })
    });
}

criterion_group!(
    sdf_benches,
    bench_sphere_sphere,
    bench_sphere_cuboid,
    bench_capsule_capsule,
    bench_cuboid_cuboid,
    bench_parametric,
);
criterion_main!(sdf_benches);
