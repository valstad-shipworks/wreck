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
use wide::{f32x8, CmpLe};

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
    origin: Vec3, dir: Vec3, rdv: f32,
    point: Vec3, t_min: f32, t_max: f32,
) -> f32 {
    let diff = point - origin;
    let t = diff.dot(dir) * rdv;
    t.clamp(t_min, t_max)
}

/// Does the parametric line collide with a sphere?
#[inline]
pub(crate) fn line_sphere_collides(
    origin: Vec3, dir: Vec3, rdv: f32,
    sphere: &Sphere, t_min: f32, t_max: f32,
) -> bool {
    let t = closest_t_to_point(origin, dir, rdv, sphere.center, t_min, t_max);
    let closest = origin + dir * t;
    let d = closest - sphere.center;
    d.dot(d) <= sphere.radius * sphere.radius
}

/// SIMD collides_many for parametric line vs spheres.
pub(crate) fn simd_collides_many_spheres(
    origin: Vec3, dir: Vec3, rdv: f32,
    t_min: f32, t_max: f32,
    others: &[Sphere],
    scalar_check: impl Fn(&Sphere) -> bool,
) -> bool {
    let ox = f32x8::splat(origin.x);
    let oy = f32x8::splat(origin.y);
    let oz = f32x8::splat(origin.z);
    let dx = f32x8::splat(dir.x);
    let dy = f32x8::splat(dir.y);
    let dz = f32x8::splat(dir.z);
    let rdv8 = f32x8::splat(rdv);
    let lo = f32x8::splat(t_min);
    let hi = f32x8::splat(t_max);

    let chunks = others.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut cx = [0.0f32; 8];
        let mut cy = [0.0f32; 8];
        let mut cz = [0.0f32; 8];
        let mut r = [0.0f32; 8];
        for (i, s) in chunk.iter().enumerate() {
            cx[i] = s.center.x;
            cy[i] = s.center.y;
            cz[i] = s.center.z;
            r[i] = s.radius;
        }
        let cx = f32x8::new(cx);
        let cy = f32x8::new(cy);
        let cz = f32x8::new(cz);
        let r = f32x8::new(r);

        let dfx = cx - ox;
        let dfy = cy - oy;
        let dfz = cz - oz;
        let t = ((dfx * dx + dfy * dy + dfz * dz) * rdv8).max(lo).min(hi);

        let px = ox + dx * t;
        let py = oy + dy * t;
        let pz = oz + dz * t;

        let ex = cx - px;
        let ey = cy - py;
        let ez = cz - pz;
        let dist_sq = ex * ex + ey * ey + ez * ez;

        if dist_sq.simd_le(r * r).any() {
            return true;
        }
    }

    remainder.iter().any(|s| scalar_check(s))
}

/// Does the parametric line collide with a capsule?
#[inline]
pub(crate) fn line_capsule_collides(
    origin: Vec3, dir: Vec3,
    capsule: &Capsule, t_min: f32, t_max: f32,
) -> bool {
    let dist_sq = crate::clamped_line_segment_dist_sq(
        origin, dir, t_min, t_max,
        capsule.p1, capsule.dir,
    );
    dist_sq <= capsule.radius * capsule.radius
}

/// Slab test: does parametric line intersect cuboid?
pub(crate) fn line_cuboid_collides(
    origin: Vec3, dir: Vec3,
    cuboid: &Cuboid, t_min: f32, t_max: f32,
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
    origin: Vec3, dir: Vec3,
    planes: &[(Vec3, f32)], obb: &Cuboid,
    t_min: f32, t_max: f32,
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
    origin: Vec3, dir: Vec3,
    plane: &Plane, t_min: f32, t_max: f32,
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

// Zero-thickness pairs — always false in 3D

use crate::Collides;

impl Collides<Line> for Line {
    #[inline]
    fn collides(&self, _other: &Line) -> bool { false }
}

impl Collides<Ray> for Ray {
    #[inline]
    fn collides(&self, _other: &Ray) -> bool { false }
}

impl Collides<LineSegment> for LineSegment {
    #[inline]
    fn collides(&self, _other: &LineSegment) -> bool { false }
}

impl Collides<Ray> for Line {
    #[inline]
    fn collides(&self, _other: &Ray) -> bool { false }
}

impl Collides<Line> for Ray {
    #[inline]
    fn collides(&self, _other: &Line) -> bool { false }
}

impl Collides<LineSegment> for Line {
    #[inline]
    fn collides(&self, _other: &LineSegment) -> bool { false }
}

impl Collides<Line> for LineSegment {
    #[inline]
    fn collides(&self, _other: &Line) -> bool { false }
}

impl Collides<LineSegment> for Ray {
    #[inline]
    fn collides(&self, _other: &LineSegment) -> bool { false }
}

impl Collides<Ray> for LineSegment {
    #[inline]
    fn collides(&self, _other: &Ray) -> bool { false }
}
