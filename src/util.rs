/// Assertion macro gated by the `wreck-assert` and `debug-wreck-assert` features.
///
/// - `wreck-assert`: assertions fire in all builds (debug and release).
/// - `debug-wreck-assert`: assertions fire only in debug builds.
/// - Neither feature: assertions are compiled out entirely.
///
/// This macro is const-compatible.
macro_rules! wreck_assert {
    ($cond:expr, $msg:literal) => {
        if cfg!(feature = "wreck-assert")
            || (cfg!(feature = "debug-wreck-assert") && cfg!(debug_assertions))
        {
            assert!($cond, $msg);
        }
    };
    ($cond:expr) => {
        if cfg!(feature = "wreck-assert")
            || (cfg!(feature = "debug-wreck-assert") && cfg!(debug_assertions))
        {
            assert!($cond);
        }
    };
}
pub(crate) use wreck_assert;

/// Generalized Ericson closest-point between parametric line (p1 + s*d1, s ∈ [s_min, s_max])
/// and segment (p2 + t*d2, t ∈ [0, 1]). Returns squared distance.
#[inline]
pub(crate) fn clamped_line_segment_dist_sq(
    p1: glam::Vec3, d1: glam::Vec3, s_min: f32, s_max: f32,
    p2: glam::Vec3, d2: glam::Vec3,
) -> f32 {
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let eps = f32::EPSILON;

    let (s, t);

    if a <= eps && e <= eps {
        s = 0.0f32.clamp(s_min, s_max);
        t = 0.0;
    } else if a <= eps {
        s = 0.0f32.clamp(s_min, s_max);
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= eps {
            t = 0.0;
            s = (-c / a).clamp(s_min, s_max);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            let mut s_n = if denom.abs() > eps {
                ((b * f - c * e) / denom).clamp(s_min, s_max)
            } else {
                0.0f32.clamp(s_min, s_max)
            };

            let mut t_n = (b * s_n + f) / e;

            if t_n < 0.0 {
                t_n = 0.0;
                s_n = (-c / a).clamp(s_min, s_max);
            } else if t_n > 1.0 {
                t_n = 1.0;
                s_n = ((b - c) / a).clamp(s_min, s_max);
            }

            s = s_n;
            t = t_n;
        }
    }

    let closest1 = p1 + d1 * s;
    let closest2 = p2 + d2 * t;
    let diff = closest1 - closest2;
    diff.dot(diff)
}

pub(crate) const fn dot(a: glam::Vec3, b: glam::Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// SIMD bounding-sphere broadphase filter for `collides_many`.
///
/// Tests self's bounding sphere against each target's bounding sphere 8-at-a-time,
/// then only runs `narrowphase` on candidates that pass. Remainder is checked scalar.
#[inline]
pub(crate) fn broadphase_collides_many<T>(
    self_center: glam::Vec3,
    self_radius: f32,
    others: &[T],
    bounding_sphere: impl Fn(&T) -> (glam::Vec3, f32),
    narrowphase: impl Fn(&T) -> bool,
) -> bool {
    use wide::{CmpLe, f32x8};

    let scx = f32x8::splat(self_center.x);
    let scy = f32x8::splat(self_center.y);
    let scz = f32x8::splat(self_center.z);

    let chunks = others.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut ox = [0.0f32; 8];
        let mut oy = [0.0f32; 8];
        let mut oz = [0.0f32; 8];
        let mut orads = [0.0f32; 8];
        for (i, other) in chunk.iter().enumerate() {
            let (oc, orad) = bounding_sphere(other);
            ox[i] = oc.x;
            oy[i] = oc.y;
            oz[i] = oc.z;
            orads[i] = orad;
        }
        let dx = f32x8::new(ox) - scx;
        let dy = f32x8::new(oy) - scy;
        let dz = f32x8::new(oz) - scz;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let sum_r = f32x8::splat(self_radius) + f32x8::new(orads);
        let pass = dist_sq.simd_le(sum_r * sum_r).to_bitmask();
        if pass != 0 {
            for i in 0..8 {
                if (pass >> i) & 1 != 0 && narrowphase(&chunk[i]) {
                    return true;
                }
            }
        }
    }

    for other in remainder {
        if narrowphase(other) {
            return true;
        }
    }
    false
}