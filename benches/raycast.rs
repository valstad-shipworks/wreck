use criterion::{Criterion, criterion_group, criterion_main};
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use std::hint::black_box;

use wreck::{ArrayConvexPolytope, ConvexPolytope, Line, LineSegment, Ray, Raycast};

const N_PAIRS: usize = 256;

fn rand_vec3(rng: &mut SmallRng, range: f32) -> Vec3 {
    Vec3::new(
        rng.random_range(-range..range),
        rng.random_range(-range..range),
        rng.random_range(-range..range),
    )
}

fn rand_unit(rng: &mut SmallRng) -> Vec3 {
    loop {
        let v = rand_vec3(rng, 1.0);
        let len_sq = v.length_squared();
        if len_sq > 1e-4 {
            return v / len_sq.sqrt();
        }
    }
}

/// A randomly rotated beveled cube: 6 face planes + 12 edge bevels + 8 corner bevels.
fn beveled_cube(rng: &mut SmallRng, center: Vec3) -> (Vec<(Vec3, f32)>, Vec<Vec3>) {
    let quat = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
    );
    let half = rng.random_range(0.5..1.5);
    let bevel = half * 0.3;
    let axes = [quat * Vec3::X, quat * Vec3::Y, quat * Vec3::Z];

    let mut planes = Vec::with_capacity(26);
    for axis in axes {
        planes.push((axis, axis.dot(center) + half));
        planes.push((-axis, (-axis).dot(center) + half));
    }
    for &(a, b) in &[(0, 1), (0, 2), (1, 2)] {
        for sa in [-1.0f32, 1.0] {
            for sb in [-1.0f32, 1.0] {
                let n = (axes[a] * sa + axes[b] * sb).normalize();
                planes.push((n, n.dot(center) + half - bevel));
            }
        }
    }
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let n = (axes[0] * sx + axes[1] * sy + axes[2] * sz).normalize();
                planes.push((n, n.dot(center) + half - bevel * 1.5));
            }
        }
    }

    let mut vertices = Vec::with_capacity(24);
    for &(a, b, c) in &[(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        for sa in [-1.0f32, 1.0] {
            for sb in [-1.0f32, 1.0] {
                for sc in [-1.0f32, 1.0] {
                    vertices.push(
                        center
                            + axes[a] * (half * sa)
                            + axes[b] * ((half - bevel) * sb)
                            + axes[c] * ((half - bevel) * sc),
                    );
                }
            }
        }
    }
    (planes, vertices)
}

fn rand_polytope(rng: &mut SmallRng) -> ConvexPolytope {
    let center = rand_vec3(rng, 3.0);
    let (planes, vertices) = beveled_cube(rng, center);
    ConvexPolytope::new(planes, vertices)
}

fn rand_array_polytope(rng: &mut SmallRng) -> ArrayConvexPolytope<26, 24> {
    let p = rand_polytope(rng);
    ArrayConvexPolytope::new(
        p.planes.clone().try_into().unwrap(),
        p.vertices.clone().try_into().unwrap(),
        p.obb,
    )
}

/// A ray aimed straight at the polytope's centre from a random direction — always a hit.
fn aimed_ray(rng: &mut SmallRng, target: Vec3) -> Ray {
    let dir = rand_unit(rng);
    Ray::new(target - dir * rng.random_range(4.0..10.0), dir)
}

/// A ray aimed just past the polytope — survives the bounding volumes, then misses.
fn grazing_ray(rng: &mut SmallRng, target: Vec3) -> Ray {
    let dir = rand_unit(rng);
    let side = rand_unit(rng).cross(dir).normalize_or_zero();
    Ray::new(
        target - dir * rng.random_range(4.0..10.0) + side * rng.random_range(1.6..2.2),
        dir,
    )
}

fn polytopes(seed: u64) -> Vec<ConvexPolytope> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..N_PAIRS).map(|_| rand_polytope(&mut rng)).collect()
}

fn bench_ray_polytope_hit(c: &mut Criterion) {
    let shapes = polytopes(42);
    let mut rng = SmallRng::seed_from_u64(7);
    let rays: Vec<_> = shapes
        .iter()
        .map(|p| aimed_ray(&mut rng, p.obb.center))
        .collect();

    c.bench_function("ray_polytope_26p_hit", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for (r, p) in rays.iter().zip(shapes.iter()) {
                if let Some(h) = black_box(r).raycast(black_box(p)) {
                    acc += h.t + h.point.x;
                }
            }
            acc
        })
    });
}

fn bench_ray_polytope_graze(c: &mut Criterion) {
    let shapes = polytopes(42);
    let mut rng = SmallRng::seed_from_u64(7);
    let rays: Vec<_> = shapes
        .iter()
        .map(|p| grazing_ray(&mut rng, p.obb.center))
        .collect();

    c.bench_function("ray_polytope_26p_graze", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for (r, p) in rays.iter().zip(shapes.iter()) {
                if let Some(h) = black_box(r).raycast(black_box(p)) {
                    acc += h.t + h.point.x;
                }
            }
            acc
        })
    });
}

fn bench_ray_polytope_random(c: &mut Criterion) {
    let shapes = polytopes(42);
    let mut rng = SmallRng::seed_from_u64(7);
    let rays: Vec<_> = (0..N_PAIRS)
        .map(|_| Ray::new(rand_vec3(&mut rng, 5.0), rand_unit(&mut rng)))
        .collect();

    c.bench_function("ray_polytope_26p_random", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for (r, p) in rays.iter().zip(shapes.iter()) {
                if let Some(h) = black_box(r).raycast(black_box(p)) {
                    acc += h.t + h.point.x;
                }
            }
            acc
        })
    });
}

fn bench_line_polytope_random(c: &mut Criterion) {
    let shapes = polytopes(42);
    let mut rng = SmallRng::seed_from_u64(7);
    let lines: Vec<_> = (0..N_PAIRS)
        .map(|_| Line::new(rand_vec3(&mut rng, 5.0), rand_unit(&mut rng)))
        .collect();

    c.bench_function("line_polytope_26p_random", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for (l, p) in lines.iter().zip(shapes.iter()) {
                if let Some(h) = black_box(l).raycast(black_box(p)) {
                    acc += h.t + h.point.x;
                }
            }
            acc
        })
    });
}

fn bench_segment_polytope_random(c: &mut Criterion) {
    let shapes = polytopes(42);
    let mut rng = SmallRng::seed_from_u64(7);
    let segments: Vec<_> = shapes
        .iter()
        .map(|p| {
            let r = aimed_ray(&mut rng, p.obb.center);
            LineSegment::new(r.origin, r.origin + r.dir * 12.0)
        })
        .collect();

    c.bench_function("segment_polytope_26p_hit", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for (s, p) in segments.iter().zip(shapes.iter()) {
                if let Some(h) = black_box(s).raycast(black_box(p)) {
                    acc += h.t + h.point.x;
                }
            }
            acc
        })
    });
}

fn bench_ray_array_polytope(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let shapes: Vec<_> = (0..N_PAIRS)
        .map(|_| rand_array_polytope(&mut rng))
        .collect();
    let mut rng = SmallRng::seed_from_u64(7);
    let rays: Vec<_> = shapes
        .iter()
        .map(|p| aimed_ray(&mut rng, p.obb.center))
        .collect();

    c.bench_function("ray_array_polytope_26p_hit", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for (r, p) in rays.iter().zip(shapes.iter()) {
                if let Some(h) = black_box(r).raycast(black_box(p)) {
                    acc += h.t + h.point.x;
                }
            }
            acc
        })
    });
}

criterion_group!(
    raycast_benches,
    bench_ray_polytope_hit,
    bench_ray_polytope_graze,
    bench_ray_polytope_random,
    bench_line_polytope_random,
    bench_segment_polytope_random,
    bench_ray_array_polytope,
);
criterion_main!(raycast_benches);
