pub(crate) mod line;
pub(crate) mod ray;
pub(crate) mod segment;

pub use line::Line;
pub use line::LineStretch;
pub use ray::Ray;
pub use ray::RayStretch;
pub use segment::LineSegment;
pub use segment::LineSegmentStretch;

use glam::Vec3;

use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::plane::Plane;
use crate::sphere::Sphere;

// ---------------------------------------------------------------------------
// Shared helpers for parametric line collision (t ∈ [t_min, t_max])
// ---------------------------------------------------------------------------

/// Closest t on parametric line to a point, clamped to [t_min, t_max].
#[inline]
pub(crate) fn closest_t_to_point(
    origin: Vec3,
    dir: Vec3,
    rdv: f32,
    point: Vec3,
    t_min: f32,
    t_max: f32,
) -> f32 {
    let diff = point - origin;
    let t = diff.dot(dir) * rdv;
    t.clamp(t_min, t_max)
}

/// Does the parametric line collide with a sphere?
#[inline]
pub(crate) fn line_sphere_collides(
    origin: Vec3,
    dir: Vec3,
    rdv: f32,
    sphere: &Sphere,
    t_min: f32,
    t_max: f32,
) -> bool {
    let t = closest_t_to_point(origin, dir, rdv, sphere.center, t_min, t_max);
    let closest = origin + dir * t;
    let d = closest - sphere.center;
    d.dot(d) <= sphere.radius * sphere.radius
}

/// Does the parametric line collide with a capsule?
#[inline]
pub(crate) fn line_capsule_collides(
    origin: Vec3,
    dir: Vec3,
    capsule: &Capsule,
    t_min: f32,
    t_max: f32,
) -> bool {
    let dist_sq =
        crate::clamped_line_segment_dist_sq(origin, dir, t_min, t_max, capsule.p1, capsule.dir);
    dist_sq <= capsule.radius * capsule.radius
}

/// Slab test: does parametric line intersect cuboid?
pub(crate) fn line_cuboid_collides(
    origin: Vec3,
    dir: Vec3,
    cuboid: &Cuboid,
    t_min: f32,
    t_max: f32,
) -> bool {
    let d = origin - cuboid.center;
    let mut t_near = t_min;
    let mut t_far = t_max;

    for i in 0..3 {
        let origin_proj = d.dot(cuboid.axes[i]);
        let dir_proj = dir.dot(cuboid.axes[i]);
        let he = cuboid.half_extents[i];

        if dir_proj.abs() > f32::EPSILON {
            let inv = 1.0 / dir_proj;
            let t1 = (-he - origin_proj) * inv;
            let t2 = (he - origin_proj) * inv;
            let (tn, tf) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            t_near = t_near.max(tn);
            t_far = t_far.min(tf);
            if t_near > t_far {
                return false;
            }
        } else if origin_proj.abs() > he {
            return false;
        }
    }

    t_near <= t_far
}

/// Cyrus-Beck clip: does parametric line intersect convex polytope?
pub(crate) fn line_polytope_collides(
    origin: Vec3,
    dir: Vec3,
    planes: &[(Vec3, f32)],
    obb: &Cuboid,
    t_min: f32,
    t_max: f32,
) -> bool {
    // Broadphase: slab test against OBB
    if !line_cuboid_collides(origin, dir, obb, t_min, t_max) {
        return false;
    }

    let mut t_near = t_min;
    let mut t_far = t_max;

    for &(normal, d) in planes {
        let denom = normal.dot(dir);
        let numer = d - normal.dot(origin);

        if denom.abs() > f32::EPSILON {
            let t = numer / denom;
            if denom < 0.0 {
                t_near = t_near.max(t);
            } else {
                t_far = t_far.min(t);
            }
            if t_near > t_far {
                return false;
            }
        } else if numer < 0.0 {
            return false;
        }
    }

    t_near <= t_far
}

/// Does parametric line collide with infinite half-space n·x <= d?
#[inline]
pub(crate) fn line_infinite_plane_collides(
    origin: Vec3,
    dir: Vec3,
    plane: &Plane,
    t_min: f32,
    t_max: f32,
) -> bool {
    let n_dot_o = plane.normal.dot(origin);
    let n_dot_d = plane.normal.dot(dir);
    let rhs = plane.d - n_dot_o;

    if n_dot_d.abs() <= f32::EPSILON {
        return rhs >= 0.0;
    }

    let t_cross = rhs / n_dot_d;

    if n_dot_d > 0.0 {
        // Valid when t <= t_cross
        t_min <= t_cross
    } else {
        // Valid when t >= t_cross
        t_cross <= t_max
    }
}

// ---------------------------------------------------------------------------
// Parametric SDF helpers (shared by Line, Ray, LineSegment)
// ---------------------------------------------------------------------------

#[cfg(feature = "sdf")]
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

/// Squared perpendicular distance from a point to the parametric ray
/// `origin + t·direction`, with `t` clamped to `[t_min, t_max]`.
#[cfg(feature = "sdf")]
#[inline]
pub(crate) fn parametric_point_dist_sq(
    origin: Vec3,
    direction: Vec3,
    rdv: f32,
    t_min: f32,
    t_max: f32,
    point: Vec3,
) -> f32 {
    let t = closest_t_to_point(origin, direction, rdv, point, t_min, t_max);
    let foot = origin + direction * t;
    let v = point - foot;
    v.dot(v)
}

/// Unsigned distance between two parametric rays, each with its own t-range.
/// Standard Lumsdaine/Ericson solver; handles parallel lines and half-infinite
/// ranges uniformly.
#[cfg(feature = "sdf")]
pub(crate) fn parametric_pair_dist_sq(
    o1: Vec3,
    d1: Vec3,
    s_min: f32,
    s_max: f32,
    o2: Vec3,
    d2: Vec3,
    t_min: f32,
    t_max: f32,
) -> f32 {
    let r = o1 - o2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let b = d1.dot(d2);
    let c = d1.dot(r);
    let f = d2.dot(r);
    let eps = f32::EPSILON;

    let denom = a * e - b * b;

    let (s, t) = if denom.abs() > eps && a > eps && e > eps {
        let s0 = ((b * f - c * e) / denom).clamp(s_min, s_max);
        let mut t0 = (b * s0 + f) / e;
        if t0 < t_min {
            let t1 = t_min;
            let s1 = ((b * t1 - c) / a).clamp(s_min, s_max);
            (s1, t1)
        } else if t0 > t_max {
            let t1 = t_max;
            let s1 = ((b * t1 - c) / a).clamp(s_min, s_max);
            (s1, t1)
        } else {
            t0 = t0.clamp(t_min, t_max);
            (s0, t0)
        }
    } else if a > eps {
        let t1 = (0.0f32).clamp(t_min, t_max);
        let s1 = ((b * t1 - c) / a).clamp(s_min, s_max);
        (s1, t1)
    } else if e > eps {
        let s1 = (0.0f32).clamp(s_min, s_max);
        let t1 = ((b * s1 + f) / e).clamp(t_min, t_max);
        (s1, t1)
    } else {
        (
            (0.0f32).clamp(s_min, s_max),
            (0.0f32).clamp(t_min, t_max),
        )
    };

    let diff = (o1 + d1 * s) - (o2 + d2 * t);
    diff.dot(diff)
}

// ---------------------------------------------------------------------------
// Zero-thickness pairs — always false in 3D
// ---------------------------------------------------------------------------

use crate::Collides;

impl Collides<Line> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Line) -> bool {
        false
    }
}

impl Collides<Ray> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Ray) -> bool {
        false
    }
}

impl Collides<LineSegment> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &LineSegment) -> bool {
        false
    }
}

impl Collides<Ray> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Ray) -> bool {
        false
    }
}

impl Collides<Line> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Line) -> bool {
        false
    }
}

impl Collides<LineSegment> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &LineSegment) -> bool {
        false
    }
}

impl Collides<Line> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Line) -> bool {
        false
    }
}

impl Collides<LineSegment> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &LineSegment) -> bool {
        false
    }
}

impl Collides<Ray> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Ray) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Parametric-curve SDF impls
// ---------------------------------------------------------------------------

#[cfg(feature = "sdf")]
use crate::point::Point;
#[cfg(feature = "sdf")]
use crate::SignedDistance;

#[cfg(feature = "sdf")]
const LINE_T_MIN: f32 = f32::NEG_INFINITY;
#[cfg(feature = "sdf")]
const LINE_T_MAX: f32 = f32::INFINITY;
#[cfg(feature = "sdf")]
const RAY_T_MIN: f32 = 0.0;
#[cfg(feature = "sdf")]
const RAY_T_MAX: f32 = f32::INFINITY;
#[cfg(feature = "sdf")]
const SEG_T_MIN: f32 = 0.0;
#[cfg(feature = "sdf")]
const SEG_T_MAX: f32 = 1.0;

// Line SDF impls --------------------------------------------------------

#[cfg(feature = "sdf")]
impl SignedDistance<Sphere> for Line {
    #[inline]
    fn signed_distance(&self, other: &Sphere) -> f32 {
        parametric_point_dist_sq(
            self.origin, self.dir, self.rdv, LINE_T_MIN, LINE_T_MAX, other.center,
        )
        .sqrt()
            - other.radius
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Sphere {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Point> for Line {
    #[inline]
    fn signed_distance(&self, other: &Point) -> f32 {
        parametric_point_dist_sq(self.origin, self.dir, self.rdv, LINE_T_MIN, LINE_T_MAX, other.0)
            .sqrt()
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Point {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Capsule> for Line {
    #[inline]
    fn signed_distance(&self, other: &Capsule) -> f32 {
        parametric_pair_dist_sq(
            self.origin, self.dir, LINE_T_MIN, LINE_T_MAX, other.p1, other.dir, 0.0, 1.0,
        )
        .sqrt()
            - other.radius
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Capsule {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Plane> for Line {
    #[inline]
    fn signed_distance(&self, plane: &Plane) -> f32 {
        let n_dot_d = plane.normal.dot(self.dir);
        if n_dot_d.abs() > f32::EPSILON {
            return f32::NEG_INFINITY;
        }
        plane.normal.dot(self.origin) - plane.d
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Plane {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Cuboid> for Line {
    #[inline]
    fn signed_distance(&self, cuboid: &Cuboid) -> f32 {
        crate::cuboid::parametric_cuboid_signed_distance(
            self.origin, self.dir, LINE_T_MIN, LINE_T_MAX, cuboid,
        )
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Line {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        parametric_pair_dist_sq(
            self.origin, self.dir, LINE_T_MIN, LINE_T_MAX, other.origin, other.dir, LINE_T_MIN,
            LINE_T_MAX,
        )
        .sqrt()
    }
}

// Ray SDF impls ---------------------------------------------------------

#[cfg(feature = "sdf")]
impl SignedDistance<Sphere> for Ray {
    #[inline]
    fn signed_distance(&self, other: &Sphere) -> f32 {
        parametric_point_dist_sq(
            self.origin, self.dir, self.rdv, RAY_T_MIN, RAY_T_MAX, other.center,
        )
        .sqrt()
            - other.radius
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Sphere {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Point> for Ray {
    #[inline]
    fn signed_distance(&self, other: &Point) -> f32 {
        parametric_point_dist_sq(self.origin, self.dir, self.rdv, RAY_T_MIN, RAY_T_MAX, other.0)
            .sqrt()
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Point {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Capsule> for Ray {
    #[inline]
    fn signed_distance(&self, other: &Capsule) -> f32 {
        parametric_pair_dist_sq(
            self.origin, self.dir, RAY_T_MIN, RAY_T_MAX, other.p1, other.dir, 0.0, 1.0,
        )
        .sqrt()
            - other.radius
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Capsule {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Plane> for Ray {
    #[inline]
    fn signed_distance(&self, plane: &Plane) -> f32 {
        let n_dot_o = plane.normal.dot(self.origin);
        let n_dot_d = plane.normal.dot(self.dir);
        let origin_sd = n_dot_o - plane.d;
        if n_dot_d < 0.0 {
            f32::NEG_INFINITY
        } else {
            origin_sd
        }
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Plane {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Cuboid> for Ray {
    #[inline]
    fn signed_distance(&self, cuboid: &Cuboid) -> f32 {
        crate::cuboid::parametric_cuboid_signed_distance(
            self.origin, self.dir, RAY_T_MIN, RAY_T_MAX, cuboid,
        )
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Ray {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        parametric_pair_dist_sq(
            self.origin, self.dir, RAY_T_MIN, RAY_T_MAX, other.origin, other.dir, RAY_T_MIN,
            RAY_T_MAX,
        )
        .sqrt()
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Line> for Ray {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        parametric_pair_dist_sq(
            self.origin, self.dir, RAY_T_MIN, RAY_T_MAX, other.origin, other.dir, LINE_T_MIN,
            LINE_T_MAX,
        )
        .sqrt()
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for Line {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        other.signed_distance(self)
    }
}

// LineSegment SDF impls -------------------------------------------------

#[cfg(feature = "sdf")]
impl SignedDistance<Plane> for LineSegment {
    #[inline]
    fn signed_distance(&self, plane: &Plane) -> f32 {
        let p2 = self.p2();
        plane
            .normal
            .dot(self.p1)
            .min(plane.normal.dot(p2))
            - plane.d
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<LineSegment> for Plane {
    #[inline]
    fn signed_distance(&self, other: &LineSegment) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<LineSegment> for LineSegment {
    #[inline]
    fn signed_distance(&self, other: &LineSegment) -> f32 {
        parametric_pair_dist_sq(
            self.p1, self.dir, SEG_T_MIN, SEG_T_MAX, other.p1, other.dir, SEG_T_MIN, SEG_T_MAX,
        )
        .sqrt()
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Line> for LineSegment {
    #[inline]
    fn signed_distance(&self, other: &Line) -> f32 {
        parametric_pair_dist_sq(
            self.p1, self.dir, SEG_T_MIN, SEG_T_MAX, other.origin, other.dir, LINE_T_MIN,
            LINE_T_MAX,
        )
        .sqrt()
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<LineSegment> for Line {
    #[inline]
    fn signed_distance(&self, other: &LineSegment) -> f32 {
        other.signed_distance(self)
    }
}

#[cfg(feature = "sdf")]
impl SignedDistance<Ray> for LineSegment {
    #[inline]
    fn signed_distance(&self, other: &Ray) -> f32 {
        parametric_pair_dist_sq(
            self.p1, self.dir, SEG_T_MIN, SEG_T_MAX, other.origin, other.dir, RAY_T_MIN, RAY_T_MAX,
        )
        .sqrt()
    }
}
#[cfg(feature = "sdf")]
impl SignedDistance<LineSegment> for Ray {
    #[inline]
    fn signed_distance(&self, other: &LineSegment) -> f32 {
        other.signed_distance(self)
    }
}
