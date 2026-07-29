//! Drift-free, same-binary comparison of the sphere-vs-SoA broadphase "does the query
//! overlap any sphere?" kernel: the original `wide` f32x8 forms vs the hydroplane forms,
//! over identical data across collection sizes. Both implementations run adjacently in one
//! process so machine-state drift cancels.
//!
//!   cargo bench --bench wide_vs_hydroplane
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench wide_vs_hydroplane
//!   RUSTFLAGS="--cfg static_dispatch" cargo bench --bench wide_vs_hydroplane
//!
//! The query is positioned to always MISS, so every variant scans the whole collection —
//! isolating raw loop throughput + per-call overhead, with no early-exit asymmetry.
//!
//! The hydroplane kernels themselves live in `wreck::bench_kernels`, so `wreck`'s build-time
//! hydroplane-auto MIR analysis tunes them; this bench only measures them against the baselines.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::hint::black_box;
use wide::f32x8;
use wreck::bench_kernels::{
    hydro_anyn, hydro_len, hydro_opt, hydro_padded, hydro_zipn, opt_2col, zipany_2col,
};

const Q: [f32; 4] = [1000.0, 0.0, 0.0, 0.1]; // far away -> always miss -> full scan

struct Cols {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    r: Vec<f32>,
}

fn make_cols(n: usize, pad_to: usize) -> Cols {
    let mut rng = SmallRng::seed_from_u64(42);
    let padded = n.next_multiple_of(pad_to.max(1));
    let mut x = vec![0.0f32; padded];
    let mut y = vec![0.0f32; padded];
    let mut z = vec![0.0f32; padded];
    let mut r = vec![f32::NAN; padded];
    for i in 0..n {
        x[i] = rng.random_range(-5.0..5.0);
        y[i] = rng.random_range(-5.0..5.0);
        z[i] = rng.random_range(-5.0..5.0);
        r[i] = rng.random_range(0.1..1.0);
    }
    x.truncate(if pad_to == 1 { n } else { padded });
    y.truncate(if pad_to == 1 { n } else { padded });
    z.truncate(if pad_to == 1 { n } else { padded });
    r.truncate(if pad_to == 1 { n } else { padded });
    Cols { x, y, z, r }
}

// wide: full f32x8 loads over padded columns (original wreck form)
fn wide_padded(xs: &[f32], ys: &[f32], zs: &[f32], rs: &[f32], q: [f32; 4]) -> bool {
    let cx = f32x8::splat(q[0]);
    let cy = f32x8::splat(q[1]);
    let cz = f32x8::splat(q[2]);
    let sr = f32x8::splat(q[3]);
    let chunks = xs.len() / 8;
    for i in 0..chunks {
        let b = i * 8;
        let ox = f32x8::new(xs[b..b + 8].try_into().unwrap());
        let oy = f32x8::new(ys[b..b + 8].try_into().unwrap());
        let oz = f32x8::new(zs[b..b + 8].try_into().unwrap());
        let or = f32x8::new(rs[b..b + 8].try_into().unwrap());
        let dx = cx - ox;
        let dy = cy - oy;
        let dz = cz - oz;
        let rsum = sr + or;
        if (dx * dx + dy * dy + dz * dz).simd_le(rsum * rsum).any() {
            return true;
        }
    }
    false
}

// wide: f32x8 full chunks + scalar remainder over exact-length columns
fn wide_remainder(xs: &[f32], ys: &[f32], zs: &[f32], rs: &[f32], q: [f32; 4]) -> bool {
    let cx = f32x8::splat(q[0]);
    let cy = f32x8::splat(q[1]);
    let cz = f32x8::splat(q[2]);
    let sr = f32x8::splat(q[3]);
    let n = xs.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let b = i * 8;
        let ox = f32x8::new(xs[b..b + 8].try_into().unwrap());
        let oy = f32x8::new(ys[b..b + 8].try_into().unwrap());
        let oz = f32x8::new(zs[b..b + 8].try_into().unwrap());
        let or = f32x8::new(rs[b..b + 8].try_into().unwrap());
        let dx = cx - ox;
        let dy = cy - oy;
        let dz = cz - oz;
        let rsum = sr + or;
        if (dx * dx + dy * dy + dz * dz).simd_le(rsum * rsum).any() {
            return true;
        }
    }
    for i in chunks * 8..n {
        let dx = q[0] - xs[i];
        let dy = q[1] - ys[i];
        let dz = q[2] - zs[i];
        let rsum = q[3] + rs[i];
        if dx * dx + dy * dy + dz * dz <= rsum * rsum {
            return true;
        }
    }
    false
}

fn scalar_any(xs: &[f32], ys: &[f32], zs: &[f32], rs: &[f32], q: [f32; 4]) -> bool {
    for i in 0..xs.len() {
        let dx = q[0] - xs[i];
        let dy = q[1] - ys[i];
        let dz = q[2] - zs[i];
        let rsum = q[3] + rs[i];
        if dx * dx + dy * dy + dz * dz <= rsum * rsum {
            return true;
        }
    }
    false
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("broadphase_any");
    for &n in &[3usize, 4, 8, 12, 16, 17, 32, 64, 256, 1024] {
        let exact = make_cols(n, 1);
        let padded = make_cols(n, 16);

        g.bench_with_input(BenchmarkId::new("wide_padded", n), &n, |b, _| {
            b.iter(|| {
                wide_padded(
                    black_box(&padded.x),
                    &padded.y,
                    &padded.z,
                    &padded.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("wide_remainder", n), &n, |b, _| {
            b.iter(|| {
                wide_remainder(
                    black_box(&exact.x),
                    &exact.y,
                    &exact.z,
                    &exact.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("hydro_len", n), &n, |b, _| {
            b.iter(|| {
                hydro_len(
                    black_box(&exact.x),
                    &exact.y,
                    &exact.z,
                    &exact.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("hydro_opt", n), &n, |b, _| {
            b.iter(|| {
                hydro_opt(
                    black_box(&exact.x),
                    &exact.y,
                    &exact.z,
                    &exact.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("hydro_zipn", n), &n, |b, _| {
            b.iter(|| {
                hydro_zipn(
                    black_box(&exact.x),
                    &exact.y,
                    &exact.z,
                    &exact.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("hydro_anyn", n), &n, |b, _| {
            b.iter(|| {
                hydro_anyn(
                    black_box(&exact.x),
                    &exact.y,
                    &exact.z,
                    &exact.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("hydro_padded", n), &n, |b, _| {
            b.iter(|| {
                hydro_padded(
                    black_box(&padded.x),
                    &padded.y,
                    &padded.z,
                    &padded.r,
                    black_box(Q),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                scalar_any(
                    black_box(&exact.x),
                    &exact.y,
                    &exact.z,
                    &exact.r,
                    black_box(Q),
                )
            })
        });
        let q2 = [1000.0f32, 0.0, 0.01];
        g.bench_with_input(BenchmarkId::new("zipany_2col", n), &n, |b, _| {
            b.iter(|| zipany_2col(black_box(&exact.x), &exact.y, black_box(q2)))
        });
        g.bench_with_input(BenchmarkId::new("opt_2col", n), &n, |b, _| {
            b.iter(|| opt_2col(black_box(&exact.x), &exact.y, black_box(q2)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
