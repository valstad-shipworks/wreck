//! Exact convex-convex narrowphase used to confirm collisions that the fast
//! SIMD kernels cannot decide. Those kernels test incomplete separating-axis
//! sets (or approximate cylinders by capsules), so their "separated" verdicts
//! are trustworthy but their "colliding" verdicts are not; callers run this
//! only on pairs the cheap tests failed to separate.
//!
//! Distance-form GJK over support functions: two shapes collide iff the distance
//! between their cores is at most the sum of their margins (a sphere is a point
//! with margin r, a capsule a segment with margin r). A brute-force
//! closest-feature search keeps the simplex update robust against the
//! degeneracies that make textbook single-precision boolean GJK cycle.
//!
//! Runs in f32 to match the rest of the crate (and to stay portable to GPU
//! targets that lack f64). The boolean is decided within a small contact
//! tolerance of the true surface: inside that band the accumulated single-
//! precision error is comparable to the f32 quantization of the inputs
//! themselves, so the answer there is inherently ambiguous. The band is biased
//! toward reporting contact, so a genuine touch is never missed.

use glam::Vec3;

use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::cylinder::Cylinder;
use crate::sphere::Sphere;

/// Half-width of the ambiguous surface band, relative to the working scale. A
/// separation below this (after the margins) is reported as contact rather than
/// risk missing a real collision to single-precision noise.
const CONTACT_TOL: f32 = 1e-5;

pub(crate) enum Core<'a> {
    Point(Vec3),
    Segment(Vec3, Vec3),
    Cuboid {
        center: Vec3,
        axes: [Vec3; 3],
        he: [f32; 3],
    },
    Cylinder {
        p1: Vec3,
        p2: Vec3,
        radius: f32,
    },
    Hull(&'a [Vec3]),
}

pub(crate) struct ConvexBody<'a> {
    core: Core<'a>,
    margin: f32,
}

impl<'a> ConvexBody<'a> {
    #[inline]
    pub fn sphere(s: &Sphere) -> Self {
        ConvexBody {
            core: Core::Point(s.center),
            margin: s.radius,
        }
    }

    #[inline]
    pub fn capsule(c: &Capsule) -> Self {
        ConvexBody {
            core: Core::Segment(c.p1, c.p2()),
            margin: c.radius,
        }
    }

    #[inline]
    pub fn cuboid(c: &Cuboid) -> Self {
        ConvexBody {
            core: Core::Cuboid {
                center: c.center,
                axes: c.axes,
                he: c.half_extents,
            },
            margin: 0.0,
        }
    }

    #[inline]
    pub fn cylinder(c: &Cylinder) -> Self {
        ConvexBody {
            core: Core::Cylinder {
                p1: c.p1,
                p2: c.p2(),
                radius: c.radius,
            },
            margin: 0.0,
        }
    }

    #[inline]
    pub fn hull(vertices: &'a [Vec3]) -> Self {
        ConvexBody {
            core: Core::Hull(vertices),
            margin: 0.0,
        }
    }

    #[inline]
    pub fn point_with_margin(p: Vec3, margin: f32) -> Self {
        ConvexBody {
            core: Core::Point(p),
            margin,
        }
    }

    fn support(&self, d: Vec3) -> Vec3 {
        match &self.core {
            Core::Point(p) => *p,
            Core::Segment(a, b) => {
                if d.dot(*b - *a) > 0.0 {
                    *b
                } else {
                    *a
                }
            }
            Core::Cuboid { center, axes, he } => {
                let mut p = *center;
                for i in 0..3 {
                    p += axes[i] * he[i].copysign(d.dot(axes[i]));
                }
                p
            }
            Core::Cylinder { p1, p2, radius } => {
                let axis = *p2 - *p1;
                let e = if d.dot(axis) > 0.0 { *p2 } else { *p1 };
                let len_sq = axis.length_squared();
                let radial = if len_sq > 0.0 {
                    d - axis * (d.dot(axis) / len_sq)
                } else {
                    d
                };
                // For d nearly parallel to the axis the subtraction cancels and
                // normalizing the noise leaks an axial component, placing the
                // "support" outside the cylinder and breaking GJK's upper bound.
                // The threshold is relative to |d|; the cap-center fallback is
                // always inside.
                let r_sq = radial.length_squared();
                if r_sq > d.length_squared() * 1e-6 {
                    e + radial * (*radius / r_sq.sqrt())
                } else {
                    e
                }
            }
            Core::Hull(verts) => {
                let mut best = verts[0];
                let mut best_p = d.dot(best);
                for v in &verts[1..] {
                    let p = d.dot(*v);
                    if p > best_p {
                        best_p = p;
                        best = *v;
                    }
                }
                best
            }
        }
    }

    fn centroid(&self) -> Vec3 {
        match &self.core {
            Core::Point(p) => *p,
            Core::Segment(a, b) => (*a + *b) * 0.5,
            Core::Cuboid { center, .. } => *center,
            Core::Cylinder { p1, p2, .. } => (*p1 + *p2) * 0.5,
            Core::Hull(verts) => {
                let mut sum = Vec3::ZERO;
                for v in *verts {
                    sum += *v;
                }
                sum / verts.len() as f32
            }
        }
    }
}

/// Closest point to the origin on the convex hull of `pts[..len]`, by checking
/// every vertex, edge, and face feature. Compacts the supporting feature's
/// points to the front of `pts` and returns (closest point, feature size).
fn closest_on_simplex(pts: &mut [Vec3; 4], len: usize) -> (Vec3, usize) {
    let mut best_d2 = f32::INFINITY;
    let mut best_p = pts[0];
    let mut best_idx = [0usize; 3];
    let mut best_len = 1usize;

    for (i, p) in pts.iter().enumerate().take(len) {
        let d2 = p.length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_p = *p;
            best_idx = [i, 0, 0];
            best_len = 1;
        }
    }
    for i in 0..len {
        for j in (i + 1)..len {
            let (a, b) = (pts[i], pts[j]);
            let ab = b - a;
            let l2 = ab.length_squared();
            if l2 < 1e-16 {
                continue;
            }
            let t = -a.dot(ab) / l2;
            if t > 0.0 && t < 1.0 {
                let p = a + ab * t;
                let d2 = p.length_squared();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_p = p;
                    best_idx = [i, j, 0];
                    best_len = 2;
                }
            }
        }
    }
    for i in 0..len {
        for j in (i + 1)..len {
            for k in (j + 1)..len {
                let (a, b, c) = (pts[i], pts[j], pts[k]);
                let e1 = b - a;
                let e2 = c - a;
                let g11 = e1.dot(e1);
                let g12 = e1.dot(e2);
                let g22 = e2.dot(e2);
                let det = g11 * g22 - g12 * g12;
                if det.abs() < 1e-16 {
                    continue;
                }
                let r1 = -a.dot(e1);
                let r2 = -a.dot(e2);
                let u = (r1 * g22 - r2 * g12) / det;
                let v = (r2 * g11 - r1 * g12) / det;
                if u > 0.0 && v > 0.0 && u + v < 1.0 {
                    let p = a + e1 * u + e2 * v;
                    let d2 = p.length_squared();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_p = p;
                        best_idx = [i, j, k];
                        best_len = 3;
                    }
                }
            }
        }
    }
    if len == 4 {
        // Origin-in-tetrahedron by the signed-volume (same-side) test: for each
        // face, the origin must lie on the same side as the opposite vertex.
        // Division-free, so the sign stays correct in f32 on the flat-ish
        // simplices GJK builds, where the barycentric form loses it.
        let (a, b, c, d) = (pts[0], pts[1], pts[2], pts[3]);
        let same_side = |v0: Vec3, v1: Vec3, v2: Vec3, opp: Vec3| {
            let n = (v1 - v0).cross(v2 - v0);
            // sign(n·(opp - v0)) vs sign(n·(origin - v0)) = sign(-n·v0)
            (n.dot(opp - v0)) * (n.dot(-v0)) >= 0.0
        };
        if same_side(a, b, c, d)
            && same_side(b, c, d, a)
            && same_side(c, d, a, b)
            && same_side(d, a, b, c)
        {
            return (Vec3::ZERO, 4);
        }
    }

    let feature = [
        pts[best_idx[0]],
        pts[best_idx[1.min(best_len - 1)]],
        pts[best_idx[2.min(best_len - 1)]],
    ];
    pts[..best_len].copy_from_slice(&feature[..best_len]);
    (best_p, best_len)
}

/// Do two convex bodies intersect (touching, to within the contact tolerance,
/// counts as intersecting)?
pub(crate) fn bodies_collide(a: &ConvexBody<'_>, b: &ConvexBody<'_>) -> bool {
    let support = |d: Vec3| a.support(d) - b.support(-d);

    let mut d0 = b.centroid() - a.centroid();
    if d0.length_squared() < 1e-12 {
        d0 = Vec3::X;
    }
    // Widen the collision margin by the tolerance, scaled to the working
    // geometry, so a separation lost to single-precision noise reads as contact
    // instead of a missed collision.
    let scale = d0.length().max(a.margin + b.margin).max(1.0);
    let margin = a.margin + b.margin + CONTACT_TOL * scale;
    let margin_sq = margin * margin;

    let mut pts = [Vec3::ZERO; 4];
    pts[0] = support(d0);
    let mut len = 1usize;

    for _ in 0..96 {
        let (v, new_len) = closest_on_simplex(&mut pts, len);
        len = new_len;
        let dist_sq = v.length_squared();
        // v is the closest point on a subset of the Minkowski difference, so |v|
        // bounds the true distance from above.
        if dist_sq <= margin_sq {
            return true;
        }
        let dist = dist_sq.sqrt();
        // Robust search direction: when the closest feature is a triangle, the
        // origin sits near its plane and `v` is tiny; normalizing it in f32 loses
        // the perpendicular direction and stalls the search on a duplicate
        // support. The face normal, built from full-size edges, stays accurate.
        let dir = if len == 3 {
            let n = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
            let n = if n.dot(-pts[0]) >= 0.0 { n } else { -n };
            let nl = n.length();
            if nl > 0.0 { n / nl } else { -v / dist }
        } else {
            -v / dist
        };
        let w = support(dir);
        let h = w.dot(dir);
        // The difference lies inside {x . x·dir <= h}, so the true distance is
        // at least -h; past the margin means provably separated.
        if -h > margin {
            return false;
        }
        // dist - (-h) bounds the achievable improvement; once it is exhausted the
        // current distance is the true one, and it already exceeds the margin.
        if dist + h.min(0.0) < 1e-6 * dist.max(1.0) {
            return false;
        }
        let mut duplicate = false;
        for p in &pts[..len] {
            if (*p - w).length_squared() < 1e-12 {
                duplicate = true;
                break;
            }
        }
        if duplicate || len == 4 {
            return false;
        }
        pts[len] = w;
        len += 1;
    }
    false
}
