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

