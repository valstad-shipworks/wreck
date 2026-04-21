#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use glam::Vec3;

use super::support::SupportFn;

/// Hard cap on GJK iterations. In practice GJK converges in under ~20 on
/// well-conditioned shapes; the cap guards against pathological inputs.
const GJK_MAX_ITERATIONS: usize = 64;

/// Convergence tolerance on the squared distance improvement per iteration.
const GJK_EPS: f32 = 1e-10;

/// A Minkowski-difference vertex together with the support pair that produced
/// it. The pair is needed if GJK's result is later fed into EPA (which needs
/// to evaluate support functions on the original shapes, not their difference).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SupportVertex {
    pub w: Vec3,
}

impl SupportVertex {
    #[inline]
    pub(crate) fn new<A: SupportFn + ?Sized, B: SupportFn + ?Sized>(
        a: &A,
        b: &B,
        direction: Vec3,
    ) -> Self {
        let pa = a.support(direction);
        let pb = b.support(-direction);
        Self { w: pa - pb }
    }
}

/// Outcome of a GJK run.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GjkResult {
    /// Shapes are disjoint; field is the Euclidean distance between them.
    Separated(f32),
    /// Shapes overlap; the final simplex is a tetrahedron containing the
    /// origin, suitable as an EPA seed.
    Penetrating([SupportVertex; 4]),
}

/// Computes the Euclidean distance between two convex shapes. When the
/// shapes overlap, returns a tetrahedral simplex enclosing the origin of
/// the Minkowski difference — seed data for EPA.
pub(crate) fn gjk_distance<A, B>(a: &A, b: &B) -> GjkResult
where
    A: SupportFn + ?Sized,
    B: SupportFn + ?Sized,
{
    let initial_dir = Vec3::X;
    let mut simplex: [SupportVertex; 4] = [SupportVertex { w: Vec3::ZERO }; 4];
    let mut n: usize = 1;
    simplex[0] = SupportVertex::new(a, b, initial_dir);

    let mut direction = -simplex[0].w;

    for _ in 0..GJK_MAX_ITERATIONS {
        if direction.dot(direction) < GJK_EPS {
            return produce_enclosure(a, b, &mut simplex, &mut n);
        }

        let w = SupportVertex::new(a, b, direction);

        let dir_len_sq = direction.dot(direction);
        if direction.dot(w.w) < direction.dot(simplex[0].w) + GJK_EPS * dir_len_sq.sqrt() {
            break;
        }

        simplex[n] = w;
        n += 1;

        let mut new_points = [SupportVertex { w: Vec3::ZERO }; 4];
        let mut new_n = 0;
        let closest = do_simplex(&simplex[..n], &mut new_points, &mut new_n);

        simplex[..new_n].copy_from_slice(&new_points[..new_n]);
        n = new_n;

        if closest.dot(closest) < GJK_EPS {
            return produce_enclosure(a, b, &mut simplex, &mut n);
        }
        direction = -closest;
    }

    let p = closest_point_on_simplex(&simplex[..n]);
    GjkResult::Separated(p.dot(p).sqrt())
}

fn produce_enclosure<A, B>(
    a: &A,
    b: &B,
    simplex: &mut [SupportVertex; 4],
    n: &mut usize,
) -> GjkResult
where
    A: SupportFn + ?Sized,
    B: SupportFn + ?Sized,
{
    while *n < 4 {
        let d = pick_fallback_direction(&simplex[..*n]);
        let w = SupportVertex::new(a, b, d);
        simplex[*n] = w;
        *n += 1;
    }
    let v0 = simplex[0].w;
    let v1 = simplex[1].w;
    let v2 = simplex[2].w;
    let v3 = simplex[3].w;
    let volume = (v1 - v0).dot((v2 - v0).cross(v3 - v0)).abs();
    if volume < GJK_EPS {
        return GjkResult::Separated(0.0);
    }
    GjkResult::Penetrating(*simplex)
}

fn pick_fallback_direction(simplex: &[SupportVertex]) -> Vec3 {
    match simplex.len() {
        0 => Vec3::X,
        1 => {
            if simplex[0].w.dot(Vec3::X).abs() > 0.9 {
                Vec3::Y
            } else {
                Vec3::X
            }
        }
        2 => {
            let e = simplex[1].w - simplex[0].w;
            let base = if e.dot(Vec3::X).abs() < 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            e.cross(base).cross(e)
        }
        _ => {
            let e1 = simplex[1].w - simplex[0].w;
            let e2 = simplex[2].w - simplex[0].w;
            e1.cross(e2)
        }
    }
}

#[inline]
fn closest_point_on_simplex(simplex: &[SupportVertex]) -> Vec3 {
    let mut tmp = [SupportVertex { w: Vec3::ZERO }; 4];
    let mut n = 0;
    do_simplex(simplex, &mut tmp, &mut n)
}

/// Given a simplex (1–4 vertices) in the Minkowski difference, computes the
/// point on the simplex closest to the origin and reduces the simplex to the
/// sub-simplex containing that point.
fn do_simplex(
    simplex: &[SupportVertex],
    out: &mut [SupportVertex; 4],
    out_n: &mut usize,
) -> Vec3 {
    match simplex.len() {
        1 => {
            out[0] = simplex[0];
            *out_n = 1;
            simplex[0].w
        }
        2 => closest_line(simplex[0], simplex[1], out, out_n),
        3 => closest_triangle(simplex[0], simplex[1], simplex[2], out, out_n),
        4 => closest_tetrahedron(simplex[0], simplex[1], simplex[2], simplex[3], out, out_n),
        _ => {
            *out_n = 0;
            Vec3::ZERO
        }
    }
}

fn closest_line(
    a: SupportVertex,
    b: SupportVertex,
    out: &mut [SupportVertex; 4],
    out_n: &mut usize,
) -> Vec3 {
    let ab = b.w - a.w;
    let t = (-a.w).dot(ab) / ab.dot(ab);
    if t <= 0.0 {
        out[0] = a;
        *out_n = 1;
        a.w
    } else if t >= 1.0 {
        out[0] = b;
        *out_n = 1;
        b.w
    } else {
        out[0] = a;
        out[1] = b;
        *out_n = 2;
        a.w + ab * t
    }
}

fn closest_triangle(
    a: SupportVertex,
    b: SupportVertex,
    c: SupportVertex,
    out: &mut [SupportVertex; 4],
    out_n: &mut usize,
) -> Vec3 {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let ao = -a.w;

    let d1 = ab.dot(ao);
    let d2 = ac.dot(ao);
    if d1 <= 0.0 && d2 <= 0.0 {
        out[0] = a;
        *out_n = 1;
        return a.w;
    }

    let bo = -b.w;
    let d3 = ab.dot(bo);
    let d4 = ac.dot(bo);
    if d3 >= 0.0 && d4 <= d3 {
        out[0] = b;
        *out_n = 1;
        return b.w;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        out[0] = a;
        out[1] = b;
        *out_n = 2;
        return a.w + ab * t;
    }

    let co = -c.w;
    let d5 = ab.dot(co);
    let d6 = ac.dot(co);
    if d6 >= 0.0 && d5 <= d6 {
        out[0] = c;
        *out_n = 1;
        return c.w;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        out[0] = a;
        out[1] = c;
        *out_n = 2;
        return a.w + ac * t;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        out[0] = b;
        out[1] = c;
        *out_n = 2;
        return b.w + (c.w - b.w) * t;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    out[0] = a;
    out[1] = b;
    out[2] = c;
    *out_n = 3;
    a.w + ab * v + ac * w
}

fn closest_tetrahedron(
    a: SupportVertex,
    b: SupportVertex,
    c: SupportVertex,
    d: SupportVertex,
    out: &mut [SupportVertex; 4],
    out_n: &mut usize,
) -> Vec3 {
    let mut best_point = Vec3::ZERO;
    let mut best_sq = f32::INFINITY;
    let mut best_buf = [SupportVertex { w: Vec3::ZERO }; 4];
    let mut best_n = 0usize;

    for (p0, p1, p2, oppo) in [
        (a, b, c, d),
        (a, c, d, b),
        (a, d, b, c),
        (b, d, c, a),
    ] {
        let n = (p1.w - p0.w).cross(p2.w - p0.w);
        let outward = if n.dot(oppo.w - p0.w) > 0.0 { -n } else { n };
        if outward.dot(-p0.w) <= 0.0 {
            continue;
        }
        let mut buf = [SupportVertex { w: Vec3::ZERO }; 4];
        let mut bn = 0;
        let cp = closest_triangle(p0, p1, p2, &mut buf, &mut bn);
        let sq = cp.dot(cp);
        if sq < best_sq {
            best_sq = sq;
            best_point = cp;
            best_buf = buf;
            best_n = bn;
        }
    }

    if best_sq.is_finite() {
        out[..best_n].copy_from_slice(&best_buf[..best_n]);
        *out_n = best_n;
        best_point
    } else {
        out[0] = a;
        out[1] = b;
        out[2] = c;
        out[3] = d;
        *out_n = 4;
        Vec3::ZERO
    }
}
