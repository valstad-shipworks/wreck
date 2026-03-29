use criterion::{Criterion, criterion_group, criterion_main};
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::hint::black_box;

use wreck::{Sphere, Transformable};

fn rand_sphere(rng: &mut SmallRng) -> Sphere {
    Sphere::new(
        Vec3::new(
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ),
        rng.random_range(0.1..1.0),
    )
}

fn make_soa(rng: &mut SmallRng, n: usize) -> wreck::soa::SpheresSoA {
    let spheres: Vec<Sphere> = (0..n).map(|_| rand_sphere(rng)).collect();
    wreck::soa::SpheresSoA::from_slice(&spheres)
}

fn bench_soa_translate(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(99);
    let offset = glam::Vec3A::new(1.0, 2.0, 3.0);

    for &n in &[4, 32, 256] {
        let soa = make_soa(&mut rng, n);
        c.bench_function(&format!("soa_translate_{n}"), |b| {
            b.iter_batched(
                || soa.clone(),
                |mut s| {
                    s.translate(black_box(offset));
                    s
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}

fn bench_soa_rotate(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(99);
    let mat = glam::Mat3A::from_rotation_y(0.5);

    for &n in &[4, 32, 256] {
        let soa = make_soa(&mut rng, n);
        c.bench_function(&format!("soa_rotate_{n}"), |b| {
            b.iter_batched(
                || soa.clone(),
                |mut s| {
                    s.rotate_mat(black_box(mat));
                    s
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}

fn bench_soa_transform(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(99);
    let mat = glam::Affine3A::from_rotation_translation(
        glam::Quat::from_rotation_y(0.5),
        glam::Vec3::new(1.0, 2.0, 3.0),
    );

    for &n in &[4, 32, 256] {
        let soa = make_soa(&mut rng, n);
        c.bench_function(&format!("soa_transform_{n}"), |b| {
            b.iter_batched(
                || soa.clone(),
                |mut s| {
                    s.transform(black_box(mat));
                    s
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(
    soa_benches,
    bench_soa_translate,
    bench_soa_rotate,
    bench_soa_transform,
);

criterion_main!(soa_benches);
