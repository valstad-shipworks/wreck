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

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;
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

/// Closest point to the origin on segment `a`-`b`, with the supporting feature: the segment
/// itself (2) or the end the origin lies beyond (1, that end first).
fn closest_on_segment(a: Vec3, b: Vec3) -> (Vec3, [Vec3; 2], usize) {
    let ab = b - a;
    let l2 = ab.length_squared();
    let t = if l2 > 0.0 { -a.dot(ab) / l2 } else { 0.0 };
    if t <= 0.0 {
        (a, [a, b], 1)
    } else if t >= 1.0 {
        (b, [b, a], 1)
    } else {
        (a + ab * t, [a, b], 2)
    }
}

/// Closest point to the origin on triangle `a`-`b`-`c`, with the supporting feature.
///
/// The origin projects inside the triangle iff the three sub-triangle areas it forms with the
/// edges share the normal's sign. Those areas come from cross products of the vertices
/// themselves, which stay accurate on the long thin triangles GJK builds between a far support
/// point and a cluster of near ones, where solving the edge Gram system in f32 does not.
fn closest_on_triangle(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [Vec3; 3], usize) {
    let e1 = b - a;
    let e2 = c - a;
    let n = e1.cross(e2);
    let n2 = n.length_squared();
    if n2 > 1e-12 * e1.length_squared() * e2.length_squared() {
        let u = n.dot(b.cross(c));
        let v = n.dot(c.cross(a));
        let w = n.dot(a.cross(b));
        if u >= 0.0 && v >= 0.0 && w >= 0.0 {
            let nearest = if a.length_squared() <= b.length_squared() {
                if a.length_squared() <= c.length_squared() {
                    a
                } else {
                    c
                }
            } else if b.length_squared() <= c.length_squared() {
                b
            } else {
                c
            };
            return (n * (n.dot(nearest) / n2), [a, b, c], 3);
        }
    }
    let mut best = (Vec3::ZERO, [a, b, c], 0usize);
    let mut best_d2 = f32::INFINITY;
    for (p, q) in [(a, b), (b, c), (c, a)] {
        let (x, f, n) = closest_on_segment(p, q);
        let d2 = x.length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best = (x, [f[0], f[1], Vec3::ZERO], n);
        }
    }
    best
}

/// Is the origin inside tetrahedron `pts`? Each face must see the origin on the side of the
/// opposite vertex. A tetrahedron too flat to have a reliable interior is reported as not
/// containing it, so the caller falls back to its faces.
fn tetrahedron_contains_origin(pts: &[Vec3; 4]) -> bool {
    let side = |v0: Vec3, v1: Vec3, v2: Vec3, opp: Vec3| {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let n = e1.cross(e2);
        let height = n.dot(opp - v0);
        let flat = 1e-6 * n.length() * (opp - v0).length();
        if height.abs() <= flat {
            return false;
        }
        height * n.dot(-v0) >= 0.0
    };
    let [a, b, c, d] = *pts;
    side(a, b, c, d) && side(b, c, d, a) && side(c, d, a, b) && side(d, a, b, c)
}

/// Closest point to the origin on the convex hull of `pts[..len]`. Compacts the supporting
/// feature's points to the front of `pts` and returns (closest point, feature size); a feature
/// size of 4 means the origin is inside the tetrahedron.
fn closest_on_simplex(pts: &mut [Vec3; 4], len: usize) -> (Vec3, usize) {
    match len {
        1 => (pts[0], 1),
        2 => {
            let (x, f, n) = closest_on_segment(pts[0], pts[1]);
            pts[..2].copy_from_slice(&f);
            (x, n)
        }
        3 => {
            let (x, f, n) = closest_on_triangle(pts[0], pts[1], pts[2]);
            pts[..3].copy_from_slice(&f);
            (x, n)
        }
        _ => {
            if tetrahedron_contains_origin(pts) {
                return (Vec3::ZERO, 4);
            }
            let [a, b, c, d] = *pts;
            let mut best = (Vec3::ZERO, [a, b, c], 0usize);
            let mut best_d2 = f32::INFINITY;
            for (p, q, r) in [(a, b, c), (a, b, d), (a, c, d), (b, c, d)] {
                let (x, f, n) = closest_on_triangle(p, q, r);
                let d2 = x.length_squared();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = (x, f, n);
                }
            }
            pts[..3].copy_from_slice(&best.1);
            (best.0, best.2)
        }
    }
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
    let mut prev_dist_sq = f32::INFINITY;

    for _ in 0..96 {
        let (v, new_len) = closest_on_simplex(&mut pts, len);
        len = new_len;
        let dist_sq = v.length_squared();
        // v is the closest point on a subset of the Minkowski difference, so |v|
        // bounds the true distance from above.
        if dist_sq <= margin_sq {
            return true;
        }
        // Every support point added so far promised a shorter distance. A step that fails to
        // deliver one has run out of single precision without proving separation, and the
        // bodies are then within rounding of touching.
        if dist_sq >= prev_dist_sq {
            return true;
        }
        prev_dist_sq = dist_sq;
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
        if pts[..len].iter().any(|p| (*p - w).length_squared() < 1e-12) {
            return false;
        }
        pts[len] = w;
        len += 1;
    }
    true
}
