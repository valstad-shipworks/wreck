//! Candidate hydroplane forms of the sphere-vs-SoA broadphase "does the query overlap any sphere?"
//! kernel, exercised by the `wreck-benches` crate. They live here so `build.rs`'s hydroplane-auto
//! MIR analysis measures them and bakes per-kernel `k_cap`/`noalias` decisions into their codegen.

use hydroplane::{Backend, Gang, Mask, Varying, kernel};

// hydroplane: real-length, load_partial (current wreck form)
#[kernel]
pub fn hydro_len<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
) -> bool {
    let n = ctx.lanes::<f32>();
    let len = xs.len();
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let cz = ctx.splat(q[2]);
    let sr = ctx.splat(q[3]);
    let mut off = 0;
    while off < len {
        let cnt = (len - off).min(n);
        let dx = cx - ctx.load_partial(&xs[off..off + cnt], 0.0);
        let dy = cy - ctx.load_partial(&ys[off..off + cnt], 0.0);
        let dz = cz - ctx.load_partial(&zs[off..off + cnt], 0.0);
        let rsum = sr + ctx.load_partial(&rs[off..off + cnt], f32::NAN);
        if (dx * dx + dy * dy + dz * dz).le(rsum * rsum).any() {
            return true;
        }
        off += cnt;
    }
    false
}

// hydroplane: full loads over padded columns (no tail staging)
#[kernel]
pub fn hydro_padded<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
) -> bool {
    let n = ctx.lanes::<f32>();
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let cz = ctx.splat(q[2]);
    let sr = ctx.splat(q[3]);
    let mut k = 0;
    while k < xs.len() {
        let dx = cx - ctx.load(&xs[k..k + n]);
        let dy = cy - ctx.load(&ys[k..k + n]);
        let dz = cz - ctx.load(&zs[k..k + n]);
        let rsum = sr + ctx.load(&rs[k..k + n]);
        if (dx * dx + dy * dy + dz * dz).le(rsum * rsum).any() {
            return true;
        }
        k += n;
    }
    false
}

// hydroplane: full-stride loop + single masked tail (no padding, no per-iter branch)
#[kernel]
pub fn hydro_opt<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
) -> bool {
    let n = ctx.lanes::<f32>();
    let len = xs.len();
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let cz = ctx.splat(q[2]);
    let sr = ctx.splat(q[3]);
    let mut i = 0;
    while i + n <= len {
        let dx = cx - ctx.load(&xs[i..i + n]);
        let dy = cy - ctx.load(&ys[i..i + n]);
        let dz = cz - ctx.load(&zs[i..i + n]);
        let rsum = sr + ctx.load(&rs[i..i + n]);
        if (dx * dx + dy * dy + dz * dz).le(rsum * rsum).any() {
            return true;
        }
        i += n;
    }
    if i < len {
        let dx = cx - ctx.load_partial(&xs[i..len], 0.0);
        let dy = cy - ctx.load_partial(&ys[i..len], 0.0);
        let dz = cz - ctx.load_partial(&zs[i..len], 0.0);
        let rsum = sr + ctx.load_partial(&rs[i..len], f32::NAN);
        if (dx * dx + dy * dy + dz * dz).le(rsum * rsum).any() {
            return true;
        }
    }
    false
}

// const-generic N-column `any` helper (candidate hydroplane `zip_any_n`).
// `core::array::from_fn` unrolls for const N, every op is `#[inline(always)]`, so this
// should monomorphize to the same flat code as a hand-unrolled zip_any.
#[inline]
fn zip_any_n<const N: usize, S: Backend<f32>>(
    ctx: Gang<S>,
    cols: [&[f32]; N],
    fills: [f32; N],
    pred: impl Fn([Varying<f32, S>; N]) -> Mask<f32, S>,
) -> bool {
    let n = ctx.lanes::<f32>();
    let len = cols[0].len();
    let mut off = 0;
    while off + n <= len {
        let vs = core::array::from_fn(|j| ctx.load(&cols[j][off..off + n]));
        if pred(vs).any() {
            return true;
        }
        off += n;
    }
    off < len
        && pred(core::array::from_fn(|j| {
            ctx.load_partial(&cols[j][off..len], fills[j])
        }))
        .any()
}

// 4-column broadphase expressed via the const-generic helper.
#[kernel]
pub fn hydro_zipn<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
) -> bool {
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let cz = ctx.splat(q[2]);
    let sr = ctx.splat(q[3]);
    zip_any_n(
        ctx,
        [xs, ys, zs, rs],
        [0.0, 0.0, 0.0, f32::NAN],
        |[x, y, z, r]| {
            let dx = cx - x;
            let dy = cy - y;
            let dz = cz - z;
            let rsum = sr + r;
            (dx * dx + dy * dy + dz * dz).le(rsum * rsum)
        },
    )
}

// 4-column broadphase via the shipped `ctx.any_n` (active-masked tail, no fills).
#[kernel]
pub fn hydro_anyn<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
) -> bool {
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let cz = ctx.splat(q[2]);
    let sr = ctx.splat(q[3]);
    ctx.any_n([xs, ys, zs, rs], |[x, y, z, r]| {
        let dx = cx - x;
        let dy = cy - y;
        let dz = cz - z;
        let rsum = sr + r;
        (dx * dx + dy * dy + dz * dz).le(rsum * rsum)
    })
}

// 2-column: zip_any helper vs hand-written opt (same computation).
// "any (cx-x)² + (cy-y)² <= R²" — 2 column streams, short-circuit.
#[kernel]
pub fn zipany_2col<'a>(ctx: Gang<f32>, xs: &'a [f32], ys: &'a [f32], q: [f32; 3]) -> bool {
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let r2 = ctx.splat(q[2]);
    ctx.zip_any(xs, ys, 1e30, 1e30, |xv, yv| {
        let dx = cx - xv;
        let dy = cy - yv;
        (dx * dx + dy * dy).le(r2)
    })
}

#[kernel]
pub fn opt_2col<'a>(ctx: Gang<f32>, xs: &'a [f32], ys: &'a [f32], q: [f32; 3]) -> bool {
    let n = ctx.lanes::<f32>();
    let len = xs.len();
    let cx = ctx.splat(q[0]);
    let cy = ctx.splat(q[1]);
    let r2 = ctx.splat(q[2]);
    let mut i = 0;
    while i + n <= len {
        let dx = cx - ctx.load(&xs[i..i + n]);
        let dy = cy - ctx.load(&ys[i..i + n]);
        if (dx * dx + dy * dy).le(r2).any() {
            return true;
        }
        i += n;
    }
    if i < len {
        let dx = cx - ctx.load_partial(&xs[i..len], 1e30);
        let dy = cy - ctx.load_partial(&ys[i..len], 1e30);
        if (dx * dx + dy * dy).le(r2).any() {
            return true;
        }
    }
    false
}
