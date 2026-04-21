use alloc::vec::Vec;
use core::fmt;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use glam::{DMat3, DVec3, Vec3};
use wide::{CmpGt, CmpLe, f32x8};

use inherent::inherent;

use crate::capsule::Capsule;
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Bounded, Collides, Scalable, Transformable};
use crate::{ConvexPolytope, Stretchable};

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cuboid {
    pub center: Vec3,
    pub axes: [Vec3; 3],
    pub half_extents: [f32; 3],
    pub axis_aligned: bool,
}

#[inline]
const fn is_axis_aligned(axes: &[Vec3; 3]) -> bool {
    // Check off-diagonal components are zero (allows negative/permuted axes)
    axes[0].y == 0.0
        && axes[0].z == 0.0
        && axes[1].x == 0.0
        && axes[1].z == 0.0
        && axes[2].x == 0.0
        && axes[2].y == 0.0
}

impl Cuboid {
    pub const fn new(center: Vec3, axes: [Vec3; 3], half_extents: [f32; 3]) -> Self {
        wreck_assert!(
            half_extents[0] >= 0.0 && half_extents[1] >= 0.0 && half_extents[2] >= 0.0,
            "Cuboid half_extents must be non-negative"
        );
        let axis_aligned = is_axis_aligned(&axes);
        Self {
            center,
            axes,
            half_extents,
            axis_aligned,
        }
    }

    pub const fn from_aabb(min: Vec3, max: Vec3) -> Self {
        wreck_assert!(
            min.x <= max.x && min.y <= max.y && min.z <= max.z,
            "AABB min must be <= max"
        );
        let center = Vec3::new(
            (min.x + max.x) * 0.5,
            (min.y + max.y) * 0.5,
            (min.z + max.z) * 0.5,
        );
        let half = Vec3::new(
            (max.x - min.x) * 0.5,
            (max.y - min.y) * 0.5,
            (max.z - min.z) * 0.5,
        );
        Self {
            center,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: [half.x, half.y, half.z],
            axis_aligned: true,
        }
    }

    /// Create a `Cuboid` from center (DVec3), full-extents tuple, and a rotation matrix.
    pub fn from_center_size_orientation(
        center: DVec3,
        size: (f64, f64, f64),
        orientation: DMat3,
    ) -> Self {
        let axes = [
            Vec3::from(orientation.col(0).as_vec3()),
            Vec3::from(orientation.col(1).as_vec3()),
            Vec3::from(orientation.col(2).as_vec3()),
        ];
        Self::new(
            center.as_vec3(),
            axes,
            [
                size.0 as f32 * 0.5,
                size.1 as f32 * 0.5,
                size.2 as f32 * 0.5,
            ],
        )
    }

    /// Extract the orientation as a DMat3 from this cuboid's axes.
    pub fn orientation_as_dmat3(&self) -> DMat3 {
        DMat3::from_cols(
            self.axes[0].as_dvec3(),
            self.axes[1].as_dvec3(),
            self.axes[2].as_dvec3(),
        )
    }

    /// Extract full-extents as an (f64, f64, f64) tuple.
    pub fn full_extents(&self) -> (f64, f64, f64) {
        (
            self.half_extents[0] as f64 * 2.0,
            self.half_extents[1] as f64 * 2.0,
            self.half_extents[2] as f64 * 2.0,
        )
    }

    /// Compute the 8 corners of this cuboid.
    pub fn corners(&self) -> [DVec3; 8] {
        let center = self.center.as_dvec3();
        let he = self.half_extents;
        let axes = [
            self.axes[0].as_dvec3(),
            self.axes[1].as_dvec3(),
            self.axes[2].as_dvec3(),
        ];
        let signs: [(f64, f64, f64); 8] = [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ];
        signs.map(|(sx, sy, sz)| {
            center
                + axes[0] * (sx * he[0] as f64)
                + axes[1] * (sy * he[1] as f64)
                + axes[2] * (sz * he[2] as f64)
        })
    }

    /// Returns the minimum corner of this cuboid's AABB.
    pub fn min(&self) -> Vec3 {
        if self.axis_aligned {
            return self.center - Vec3::from(self.half_extents);
        }
        let aabb = self.aabb();
        aabb.center - Vec3::from(aabb.half_extents)
    }

    /// Returns the maximum corner of this cuboid's AABB.
    pub fn max(&self) -> Vec3 {
        if self.axis_aligned {
            return self.center + Vec3::from(self.half_extents);
        }
        let aabb = self.aabb();
        aabb.center + Vec3::from(aabb.half_extents)
    }

    #[inline]
    pub fn bounding_sphere_radius(&self) -> f32 {
        let e = self.half_extents;
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }

    /// Squared distance from a point to the closest point on this cuboid's surface/interior.
    #[inline]
    pub fn point_dist_sq(&self, point: Vec3) -> f32 {
        if self.axis_aligned {
            return self.point_dist_sq_aa(point);
        }
        let d = point - self.center;
        let mut dist_sq = 0.0;
        for i in 0..3 {
            let proj = d.dot(self.axes[i]).abs() - self.half_extents[i];
            if proj > 0.0 {
                dist_sq += proj * proj;
            }
        }
        dist_sq
    }

    /// Axis-aligned fast path: no dot products needed.
    #[inline]
    fn point_dist_sq_aa(&self, point: Vec3) -> f32 {
        let d = point - self.center;
        let da = [d.x, d.y, d.z];
        let mut dist_sq = 0.0;
        for i in 0..3 {
            let excess = da[i].abs() - self.half_extents[i];
            if excess > 0.0 {
                dist_sq += excess * excess;
            }
        }
        dist_sq
    }

    /// Signed distance from a point to the cuboid surface.
    ///
    /// Negative inside the cuboid (magnitude = distance to the nearest face),
    /// zero on the surface, positive outside (magnitude = Euclidean distance
    /// to the nearest surface point).
    #[cfg(feature = "sdf")]
    #[inline]
    pub fn point_signed_distance(&self, point: Vec3) -> f32 {
        #[cfg(not(feature = "std"))]
        #[allow(unused_imports)]
        use crate::F32Ext;

        let d = point - self.center;
        let (d0, d1, d2) = if self.axis_aligned {
            (d.x.abs(), d.y.abs(), d.z.abs())
        } else {
            (
                d.dot(self.axes[0]).abs(),
                d.dot(self.axes[1]).abs(),
                d.dot(self.axes[2]).abs(),
            )
        };
        let q = [
            d0 - self.half_extents[0],
            d1 - self.half_extents[1],
            d2 - self.half_extents[2],
        ];
        let outside_sq = q[0].max(0.0) * q[0].max(0.0)
            + q[1].max(0.0) * q[1].max(0.0)
            + q[2].max(0.0) * q[2].max(0.0);
        let inside = q[0].max(q[1]).max(q[2]).min(0.0);
        outside_sq.sqrt() + inside
    }

    /// SIMD batched `point_signed_distance` — evaluates 8 points in parallel
    /// against this cuboid. Returns an `f32x8` of signed distances lane-wise.
    ///
    /// Used internally by the parametric SDF sampler (which checks up to 23
    /// candidate t-values per call) and by the Pointcloud × Cuboid fast path.
    #[cfg(feature = "sdf")]
    #[inline]
    pub(crate) fn point_signed_distance_x8(
        &self,
        px: f32x8,
        py: f32x8,
        pz: f32x8,
    ) -> f32x8 {
        let cx = f32x8::splat(self.center.x);
        let cy = f32x8::splat(self.center.y);
        let cz = f32x8::splat(self.center.z);
        let dx = px - cx;
        let dy = py - cy;
        let dz = pz - cz;

        let (d0, d1, d2) = if self.axis_aligned {
            (dx.abs(), dy.abs(), dz.abs())
        } else {
            let a0 = self.axes[0];
            let a1 = self.axes[1];
            let a2 = self.axes[2];
            let d0 = (dx * f32x8::splat(a0.x)
                + dy * f32x8::splat(a0.y)
                + dz * f32x8::splat(a0.z))
            .abs();
            let d1 = (dx * f32x8::splat(a1.x)
                + dy * f32x8::splat(a1.y)
                + dz * f32x8::splat(a1.z))
            .abs();
            let d2 = (dx * f32x8::splat(a2.x)
                + dy * f32x8::splat(a2.y)
                + dz * f32x8::splat(a2.z))
            .abs();
            (d0, d1, d2)
        };

        let he0 = f32x8::splat(self.half_extents[0]);
        let he1 = f32x8::splat(self.half_extents[1]);
        let he2 = f32x8::splat(self.half_extents[2]);
        let q0 = d0 - he0;
        let q1 = d1 - he1;
        let q2 = d2 - he2;

        let zero = f32x8::splat(0.0);
        let q0p = q0.max(zero);
        let q1p = q1.max(zero);
        let q2p = q2.max(zero);
        let outside_sq = q0p * q0p + q1p * q1p + q2p * q2p;
        let inside = q0.max(q1).max(q2).min(zero);
        outside_sq.sqrt() + inside
    }
}

impl Cuboid {
    pub const fn contains_point(&self, point: Vec3) -> bool {
        let d = Vec3::new(
            (point.x - self.center.x).abs(),
            (point.y - self.center.y).abs(),
            (point.z - self.center.z).abs(),
        );
        let mut i = 0;
        while i < 3 {
            let proj = crate::dot(d, self.axes[i]).abs();
            if proj > self.half_extents[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

#[inherent]
impl Bounded for Cuboid {
    pub fn broadphase(&self) -> Sphere {
        Sphere::new(self.center, self.bounding_sphere_radius())
    }

    pub fn obb(&self) -> Cuboid {
        *self
    }

    pub fn aabb(&self) -> Cuboid {
        if self.axis_aligned {
            return *self;
        }
        let mut he = [0.0f32; 3];
        let world = [Vec3::X, Vec3::Y, Vec3::Z];
        for i in 0..3 {
            he[i] = self.half_extents[0] * self.axes[0].dot(world[i]).abs()
                + self.half_extents[1] * self.axes[1].dot(world[i]).abs()
                + self.half_extents[2] * self.axes[2].dot(world[i]).abs();
        }
        Cuboid::new(self.center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }
}

#[inherent]
impl Scalable for Cuboid {
    pub fn scale(&mut self, factor: f32) {
        for e in &mut self.half_extents {
            *e *= factor;
        }
    }
}

#[inherent]
impl Transformable for Cuboid {
    pub fn translate(&mut self, offset: glam::Vec3A) {
        self.center = Vec3::from(glam::Vec3A::from(self.center) + offset);
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3A) {
        self.center = Vec3::from(mat * glam::Vec3A::from(self.center));
        for ax in &mut self.axes {
            *ax = Vec3::from(mat * glam::Vec3A::from(*ax));
        }
        self.axis_aligned = is_axis_aligned(&self.axes);
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.center = quat * self.center;
        for ax in &mut self.axes {
            *ax = quat * *ax;
        }
        self.axis_aligned = is_axis_aligned(&self.axes);
    }

    pub fn transform(&mut self, mat: glam::Affine3A) {
        self.center = Vec3::from(mat.transform_point3a(glam::Vec3A::from(self.center)));
        let rot = mat.matrix3;
        for ax in &mut self.axes {
            *ax = Vec3::from(rot * glam::Vec3A::from(*ax));
        }
        self.axis_aligned = is_axis_aligned(&self.axes);
    }
}

// AABB-AABB: simple overlap when both cuboids are axis-aligned
#[inline]
fn aabb_aabb_collides(a: &Cuboid, b: &Cuboid) -> bool {
    let d = b.center - a.center;
    d.x.abs() <= a.half_extents[0] + b.half_extents[0]
        && d.y.abs() <= a.half_extents[1] + b.half_extents[1]
        && d.z.abs() <= a.half_extents[2] + b.half_extents[2]
}

// Sphere-Cuboid collision
#[inline]
fn sphere_cuboid_collides(sphere: &Sphere, cuboid: &Cuboid) -> bool {
    cuboid.point_dist_sq(sphere.center) <= sphere.radius * sphere.radius
}

impl Collides<Cuboid> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Cuboid) -> bool {
        sphere_cuboid_collides(self, other)
    }
}

impl Collides<Sphere> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Sphere) -> bool {
        sphere_cuboid_collides(other, self)
    }
}

#[cfg(feature = "sdf")]
#[inline]
fn sphere_cuboid_signed_distance(sphere: &Sphere, cuboid: &Cuboid) -> f32 {
    cuboid.point_signed_distance(sphere.center) - sphere.radius
}

#[cfg(feature = "sdf")]
impl crate::SignedDistance<Cuboid> for Sphere {
    #[inline]
    fn signed_distance(&self, other: &Cuboid) -> f32 {
        sphere_cuboid_signed_distance(self, other)
    }
}

#[cfg(feature = "sdf")]
impl crate::SignedDistance<Sphere> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &Sphere) -> f32 {
        sphere_cuboid_signed_distance(other, self)
    }
}

/// Signed distance from a parametric ray `origin + t·direction` restricted to
/// `t ∈ (t_min, t_max)` to this cuboid's surface.
///
/// The cuboid SDF is globally convex, so its restriction to the line is a
/// convex 1D function whose minimum lies at an endpoint (if finite) or at a
/// piecewise breakpoint. Samples `point_signed_distance` at:
///   - finite endpoints,
///   - face-plane crossings (`p_i(t) = ±he_i`, 6 candidates),
///   - axis-plane crossings (`p_i(t) = 0`, 3 candidates — kinks in `|p_i|`),
///   - equal-distance crossings (`|p_i|-he_i = |p_j|-he_j`, up to 12
///     candidates per pair/sign combination — kinks where the interior
///     argmax switches axes).
///
/// Pass `f32::NEG_INFINITY` / `f32::INFINITY` for unbounded t.
#[cfg(feature = "sdf")]
pub(crate) fn parametric_cuboid_signed_distance(
    origin: Vec3,
    direction: Vec3,
    t_min: f32,
    t_max: f32,
    cuboid: &Cuboid,
) -> f32 {
    let c = cuboid.center;
    let axes: [Vec3; 3] = if cuboid.axis_aligned {
        [Vec3::X, Vec3::Y, Vec3::Z]
    } else {
        cuboid.axes
    };
    let he = cuboid.half_extents;

    let p1_proj = [
        (origin - c).dot(axes[0]),
        (origin - c).dot(axes[1]),
        (origin - c).dot(axes[2]),
    ];
    let d_proj = [
        direction.dot(axes[0]),
        direction.dot(axes[1]),
        direction.dot(axes[2]),
    ];

    // Gather up to 23 candidate t-values into a fixed buffer. Populating the
    // whole buffer unconditionally (with NaN for filtered-out candidates) lets
    // us evaluate them in 3 f32x8 SIMD batches without any per-lane branches.
    let mut ts = [f32::NAN; 24];
    let mut n = 0usize;
    let mut push = |t: f32, n: &mut usize, ts: &mut [f32; 24]| {
        if t > t_min && t < t_max {
            ts[*n] = t;
            *n += 1;
        }
    };

    if t_min.is_finite() {
        ts[n] = t_min;
        n += 1;
    }
    if t_max.is_finite() {
        ts[n] = t_max;
        n += 1;
    }

    for i in 0..3 {
        if d_proj[i].abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / d_proj[i];
        push((he[i] - p1_proj[i]) * inv, &mut n, &mut ts);
        push((-he[i] - p1_proj[i]) * inv, &mut n, &mut ts);
        push(-p1_proj[i] * inv, &mut n, &mut ts);
    }

    for i in 0..3 {
        for j in (i + 1)..3 {
            for si in [-1.0f32, 1.0f32] {
                for sj in [-1.0f32, 1.0f32] {
                    let denom = si * d_proj[i] - sj * d_proj[j];
                    if denom.abs() < 1e-12 {
                        continue;
                    }
                    let numer = he[i] - he[j] - si * p1_proj[i] + sj * p1_proj[j];
                    push(numer / denom, &mut n, &mut ts);
                }
            }
        }
    }

    if n == 0 {
        return f32::INFINITY;
    }

    // Evaluate the up-to-24 candidate t's in 3 SIMD batches. Unused/NaN lanes
    // produce NaN SDFs which never win the min reduction (NaN compares false).
    let ox = f32x8::splat(origin.x);
    let oy = f32x8::splat(origin.y);
    let oz = f32x8::splat(origin.z);
    let dx = f32x8::splat(direction.x);
    let dy = f32x8::splat(direction.y);
    let dz = f32x8::splat(direction.z);

    let mut min_sd = f32::INFINITY;
    for chunk_start in (0..n).step_by(8) {
        let mut lane_ts = [f32::NAN; 8];
        let end = (chunk_start + 8).min(n);
        lane_ts[..end - chunk_start].copy_from_slice(&ts[chunk_start..end]);
        let tv = f32x8::new(lane_ts);
        let px = ox + dx * tv;
        let py = oy + dy * tv;
        let pz = oz + dz * tv;
        let sds = cuboid.point_signed_distance_x8(px, py, pz);
        let arr = sds.to_array();
        for (i, &sd) in arr.iter().enumerate() {
            if (chunk_start + i) < end && sd < min_sd {
                min_sd = sd;
            }
        }
    }

    min_sd
}

#[cfg(feature = "sdf")]
#[inline]
fn segment_cuboid_signed_distance(p1: Vec3, p2: Vec3, cuboid: &Cuboid) -> f32 {
    let d = p2 - p1;
    let min_sd = parametric_cuboid_signed_distance(p1, d, 0.0, 1.0, cuboid);
    cuboid
        .point_signed_distance(p1)
        .min(cuboid.point_signed_distance(p2))
        .min(min_sd)
}

#[cfg(feature = "sdf")]
#[inline]
fn capsule_cuboid_signed_distance(capsule: &Capsule, cuboid: &Cuboid) -> f32 {
    segment_cuboid_signed_distance(capsule.p1, capsule.p2(), cuboid) - capsule.radius
}

#[cfg(feature = "sdf")]
impl crate::SignedDistance<Cuboid> for Capsule {
    #[inline]
    fn signed_distance(&self, other: &Cuboid) -> f32 {
        capsule_cuboid_signed_distance(self, other)
    }
}

#[cfg(feature = "sdf")]
impl crate::SignedDistance<Capsule> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &Capsule) -> f32 {
        capsule_cuboid_signed_distance(other, self)
    }
}

/// Signed distance between two oriented cuboids via the Separating Axis Theorem.
///
/// For each of the 15 SAT axes (3 face normals per cuboid + 9 edge-edge cross
/// products), the signed gap is `|Δ·a| − (rA + rB)` where `rA`, `rB` are the
/// projected half-extents. The SDF is the maximum gap across axes.
///
/// - Overlapping: all gaps are negative; the maximum (least-negative) is the
///   minimum penetration depth — exact among the SAT axes (SAT-optimal).
/// - Separated: at least one gap is positive; the maximum is the separation
///   along the best separating axis. This is a tight lower bound on the
///   Euclidean distance between the cuboids (exact when axis-aligned and
///   separated along a single axis; a conservative underestimate when the
///   closest features are a pair of vertices or edges not aligned with any
///   SAT axis). Exact Euclidean distance for arbitrary OBBs requires GJK.
#[cfg(feature = "sdf")]
#[inline]
fn cuboid_cuboid_signed_distance(a: &Cuboid, b: &Cuboid) -> f32 {
    let delta = b.center - a.center;

    // Pack the 15 SAT axes into two f32x8 batches (padding the last slot).
    // Lane layout of batch 1: a.x, a.y, a.z, b.x, b.y, b.z, a.x×b.x, a.x×b.y
    // Lane layout of batch 2: a.x×b.z, a.y×b.x, a.y×b.y, a.y×b.z, a.z×b.x,
    //                          a.z×b.y, a.z×b.z, <pad>
    let cross_axes = [
        a.axes[0].cross(b.axes[0]),
        a.axes[0].cross(b.axes[1]),
        a.axes[0].cross(b.axes[2]),
        a.axes[1].cross(b.axes[0]),
        a.axes[1].cross(b.axes[1]),
        a.axes[1].cross(b.axes[2]),
        a.axes[2].cross(b.axes[0]),
        a.axes[2].cross(b.axes[1]),
        a.axes[2].cross(b.axes[2]),
    ];

    let pad = Vec3::new(1.0, 0.0, 0.0);

    let all_axes: [Vec3; 16] = [
        a.axes[0],
        a.axes[1],
        a.axes[2],
        b.axes[0],
        b.axes[1],
        b.axes[2],
        cross_axes[0],
        cross_axes[1],
        cross_axes[2],
        cross_axes[3],
        cross_axes[4],
        cross_axes[5],
        cross_axes[6],
        cross_axes[7],
        cross_axes[8],
        pad,
    ];

    fn compute_batch(axes: &[Vec3; 8], a: &Cuboid, b: &Cuboid, delta: Vec3) -> f32x8 {
        let ax = f32x8::new([
            axes[0].x, axes[1].x, axes[2].x, axes[3].x, axes[4].x, axes[5].x, axes[6].x, axes[7].x,
        ]);
        let ay = f32x8::new([
            axes[0].y, axes[1].y, axes[2].y, axes[3].y, axes[4].y, axes[5].y, axes[6].y, axes[7].y,
        ]);
        let az = f32x8::new([
            axes[0].z, axes[1].z, axes[2].z, axes[3].z, axes[4].z, axes[5].z, axes[6].z, axes[7].z,
        ]);

        let len_sq = ax * ax + ay * ay + az * az;
        let inv_len = f32x8::splat(1.0) / len_sq.sqrt();

        let nx = ax * inv_len;
        let ny = ay * inv_len;
        let nz = az * inv_len;

        #[inline]
        fn proj_he_abs(
            nx: f32x8,
            ny: f32x8,
            nz: f32x8,
            axes: [Vec3; 3],
            he: [f32; 3],
        ) -> f32x8 {
            let mut r = f32x8::splat(0.0);
            for i in 0..3 {
                let dot = nx * f32x8::splat(axes[i].x)
                    + ny * f32x8::splat(axes[i].y)
                    + nz * f32x8::splat(axes[i].z);
                r += f32x8::splat(he[i]) * dot.abs();
            }
            r
        }

        let ra = proj_he_abs(nx, ny, nz, a.axes, a.half_extents);
        let rb = proj_he_abs(nx, ny, nz, b.axes, b.half_extents);

        let delta_dot = nx * f32x8::splat(delta.x)
            + ny * f32x8::splat(delta.y)
            + nz * f32x8::splat(delta.z);

        let gap = delta_dot.abs() - (ra + rb);

        // For degenerate axes (len_sq near 0, inv_len = +inf), the math
        // produces NaN. Zero those lanes out so they don't dominate max.
        let mask = len_sq.simd_gt(f32x8::splat(1e-12));
        mask.blend(gap, f32x8::splat(f32::NEG_INFINITY))
    }

    let batch1_arr: [Vec3; 8] = [
        all_axes[0],
        all_axes[1],
        all_axes[2],
        all_axes[3],
        all_axes[4],
        all_axes[5],
        all_axes[6],
        all_axes[7],
    ];
    let batch2_arr: [Vec3; 8] = [
        all_axes[8],
        all_axes[9],
        all_axes[10],
        all_axes[11],
        all_axes[12],
        all_axes[13],
        all_axes[14],
        all_axes[15],
    ];

    // The pad axis at batch2 lane 7 is (1,0,0), a valid SAT direction (it
    // coincides with a.axes[0] in this layout — redundant but not spurious).
    // Adding it as a 16th axis cannot change the SAT result, so no masking
    // is needed.
    let gaps1 = compute_batch(&batch1_arr, a, b, delta);
    let gaps2 = compute_batch(&batch2_arr, a, b, delta);
    let combined = gaps1.max(gaps2);
    let arr = combined.to_array();
    let mut max_gap = arr[0];
    for &v in &arr[1..] {
        if v > max_gap {
            max_gap = v;
        }
    }
    max_gap
}

#[cfg(feature = "sdf")]
impl crate::SignedDistance<Cuboid> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &Cuboid) -> f32 {
        cuboid_cuboid_signed_distance(self, other)
    }
}

// Capsule-Cuboid: Z-aligned capsule + axis-aligned cuboid.
// X/Y distances are constant along the capsule axis, only Z varies.
#[inline]
fn capsule_cuboid_za_aa(capsule: &Capsule, cuboid: &Cuboid) -> bool {
    let rs_sq = capsule.radius * capsule.radius;
    let d = capsule.p1 - cuboid.center;
    let he = cuboid.half_extents;

    // X and Y excess are constant (capsule only extends along Z)
    let ex = d.x.abs() - he[0];
    let ey = d.y.abs() - he[1];
    let xy_dist_sq = ex.max(0.0) * ex.max(0.0) + ey.max(0.0) * ey.max(0.0);

    // Early out: if XY distance alone exceeds radius, no collision possible
    if xy_dist_sq > rs_sq {
        return false;
    }

    // Only need to check Z overlap: capsule spans [p1.z, p1.z + dir.z]
    let z0 = d.z;
    let z1 = d.z + capsule.dir.z;
    let z_min = z0.min(z1);
    let z_max = z0.max(z1);

    // Closest Z distance from capsule segment to cuboid Z-extent [-he[2], he[2]]
    let ez = if z_max < -he[2] {
        -he[2] - z_max
    } else if z_min > he[2] {
        z_min - he[2]
    } else {
        0.0
    };

    xy_dist_sq + ez * ez <= rs_sq
}

// Capsule-Cuboid collision
// Evaluate all 8 candidate t-values (2 endpoints + 6 face-plane intersections)
// in a single SIMD pass using f32x8.
#[inline]
fn capsule_cuboid_collides<const BROADPHASE: bool>(capsule: &Capsule, cuboid: &Cuboid) -> bool {
    // Bounding sphere early-out
    if BROADPHASE {
        let (bc, br) = capsule.bounding_sphere();
        let d = bc - cuboid.center;
        let max_r = br + cuboid.bounding_sphere_radius();
        if d.dot(d) > max_r * max_r {
            return false;
        }
    }

    // Fastest path: Z-aligned capsule + axis-aligned cuboid
    if capsule.z_aligned && cuboid.axis_aligned {
        return capsule_cuboid_za_aa(capsule, cuboid);
    }

    let rs_sq = capsule.radius * capsule.radius;
    let p0_world = capsule.p1 - cuboid.center;

    // Axis-aligned cuboid fast path: skip 6 dot products for local frame projection
    let (p0, dir) = if cuboid.axis_aligned {
        (
            [p0_world.x, p0_world.y, p0_world.z],
            [capsule.dir.x, capsule.dir.y, capsule.dir.z],
        )
    } else {
        (
            [
                p0_world.dot(cuboid.axes[0]),
                p0_world.dot(cuboid.axes[1]),
                p0_world.dot(cuboid.axes[2]),
            ],
            [
                capsule.dir.dot(cuboid.axes[0]),
                capsule.dir.dot(cuboid.axes[1]),
                capsule.dir.dot(cuboid.axes[2]),
            ],
        )
    };
    let he = cuboid.half_extents;

    // Compute 6 critical t-values where capsule axis meets cuboid face planes,
    // plus 2 endpoints. Pack all 8 into f32x8.
    // t_i = (±he[i] - p0[i]) / dir[i], clamped to [0,1]
    // For near-zero dir components, the division gives ±inf which clamps to 0 or 1.
    let inv_dir = [
        if dir[0].abs() > f32::EPSILON {
            1.0 / dir[0]
        } else {
            f32::MAX
        },
        if dir[1].abs() > f32::EPSILON {
            1.0 / dir[1]
        } else {
            f32::MAX
        },
        if dir[2].abs() > f32::EPSILON {
            1.0 / dir[2]
        } else {
            f32::MAX
        },
    ];

    let ts = f32x8::new([
        0.0,
        1.0,
        ((-he[0] - p0[0]) * inv_dir[0]).clamp(0.0, 1.0),
        ((he[0] - p0[0]) * inv_dir[0]).clamp(0.0, 1.0),
        ((-he[1] - p0[1]) * inv_dir[1]).clamp(0.0, 1.0),
        ((he[1] - p0[1]) * inv_dir[1]).clamp(0.0, 1.0),
        ((-he[2] - p0[2]) * inv_dir[2]).clamp(0.0, 1.0),
        ((he[2] - p0[2]) * inv_dir[2]).clamp(0.0, 1.0),
    ]);

    // Evaluate squared distance from capsule-axis point at each t to cuboid, branchless.
    // For each axis: excess = max(0, |p0[i] + dir[i]*t| - he[i])
    let zero = f32x8::splat(0.0);
    let mut dist_sq = zero;

    for i in 0..3 {
        let pos = f32x8::splat(p0[i]) + f32x8::splat(dir[i]) * ts;
        let abs_pos = pos.max(-pos); // branchless abs
        let excess = (abs_pos - f32x8::splat(he[i])).max(zero);
        dist_sq = dist_sq + excess * excess;
    }

    // Check if any of the 8 evaluations is within capsule radius
    dist_sq.simd_le(f32x8::splat(rs_sq)).any()
}

impl Collides<Cuboid> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Cuboid) -> bool {
        capsule_cuboid_collides::<BROADPHASE>(self, other)
    }
}

impl Collides<Capsule> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Capsule) -> bool {
        capsule_cuboid_collides::<BROADPHASE>(other, self)
    }
}

// Cuboid-Cuboid collision via Separating Axis Theorem (15 axes)
impl Collides<Cuboid> for Cuboid {
    fn test<const BROADPHASE: bool>(&self, other: &Cuboid) -> bool {
        // Both axis-aligned: simple AABB overlap test
        if self.axis_aligned && other.axis_aligned {
            return aabb_aabb_collides(self, other);
        }

        let t_vec = other.center - self.center;

        // Bounding sphere early-out
        if BROADPHASE {
            let max_dist = self.bounding_sphere_radius() + other.bounding_sphere_radius();
            if t_vec.dot(t_vec) > max_dist * max_dist {
                return false;
            }
        }

        let eps = 1e-6f32;

        // Rotation matrix: R[i][j] = self.axes[i].dot(other.axes[j])
        let mut r = [[0.0f32; 3]; 3];
        let mut abs_r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = self.axes[i].dot(other.axes[j]);
                abs_r[i][j] = r[i][j].abs() + eps;
            }
        }

        // Translation in A's frame
        let t = [
            t_vec.dot(self.axes[0]),
            t_vec.dot(self.axes[1]),
            t_vec.dot(self.axes[2]),
        ];

        let ea = self.half_extents;
        let eb = other.half_extents;

        // Test axes L = A0, A1, A2
        for i in 0..3 {
            let ra = ea[i];
            let rb = eb[0] * abs_r[i][0] + eb[1] * abs_r[i][1] + eb[2] * abs_r[i][2];
            if t[i].abs() > ra + rb {
                return false;
            }
        }

        // Test axes L = B0, B1, B2
        for j in 0..3 {
            let ra = ea[0] * abs_r[0][j] + ea[1] * abs_r[1][j] + ea[2] * abs_r[2][j];
            let rb = eb[j];
            let sep = (t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j]).abs();
            if sep > ra + rb {
                return false;
            }
        }

        // Test 9 cross-product axes
        // L = A0 x B0
        {
            let ra = ea[1] * abs_r[2][0] + ea[2] * abs_r[1][0];
            let rb = eb[1] * abs_r[0][2] + eb[2] * abs_r[0][1];
            let sep = (t[2] * r[1][0] - t[1] * r[2][0]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A0 x B1
        {
            let ra = ea[1] * abs_r[2][1] + ea[2] * abs_r[1][1];
            let rb = eb[0] * abs_r[0][2] + eb[2] * abs_r[0][0];
            let sep = (t[2] * r[1][1] - t[1] * r[2][1]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A0 x B2
        {
            let ra = ea[1] * abs_r[2][2] + ea[2] * abs_r[1][2];
            let rb = eb[0] * abs_r[0][1] + eb[1] * abs_r[0][0];
            let sep = (t[2] * r[1][2] - t[1] * r[2][2]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A1 x B0
        {
            let ra = ea[0] * abs_r[2][0] + ea[2] * abs_r[0][0];
            let rb = eb[1] * abs_r[1][2] + eb[2] * abs_r[1][1];
            let sep = (t[0] * r[2][0] - t[2] * r[0][0]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A1 x B1
        {
            let ra = ea[0] * abs_r[2][1] + ea[2] * abs_r[0][1];
            let rb = eb[0] * abs_r[1][2] + eb[2] * abs_r[1][0];
            let sep = (t[0] * r[2][1] - t[2] * r[0][1]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A1 x B2
        {
            let ra = ea[0] * abs_r[2][2] + ea[2] * abs_r[0][2];
            let rb = eb[0] * abs_r[1][1] + eb[1] * abs_r[1][0];
            let sep = (t[0] * r[2][2] - t[2] * r[0][2]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A2 x B0
        {
            let ra = ea[0] * abs_r[1][0] + ea[1] * abs_r[0][0];
            let rb = eb[1] * abs_r[2][2] + eb[2] * abs_r[2][1];
            let sep = (t[1] * r[0][0] - t[0] * r[1][0]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A2 x B1
        {
            let ra = ea[0] * abs_r[1][1] + ea[1] * abs_r[0][1];
            let rb = eb[0] * abs_r[2][2] + eb[2] * abs_r[2][0];
            let sep = (t[1] * r[0][1] - t[0] * r[1][1]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A2 x B2
        {
            let ra = ea[0] * abs_r[1][2] + ea[1] * abs_r[0][2];
            let rb = eb[0] * abs_r[2][1] + eb[1] * abs_r[2][0];
            let sep = (t[1] * r[0][2] - t[0] * r[1][2]).abs();
            if sep > ra + rb {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CuboidStretch {
    Aligned(Cuboid),
    Unaligned(ConvexPolytope),
}

impl Stretchable for Cuboid {
    type Output = CuboidStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        // Check if translation is along one of the cuboid's axes
        for i in 0..3 {
            let cross = self.axes[i].cross(translation);
            if cross.length_squared() < 1e-10 {
                let proj = translation.dot(self.axes[i]);
                let mut result = *self;
                result.center += translation * 0.5;
                result.half_extents[i] += proj.abs() * 0.5;
                return CuboidStretch::Aligned(result);
            }
        }

        // Unaligned: Minkowski sum of cuboid and line segment = convex polytope
        let he = self.half_extents;
        let ax = self.axes;
        let c = self.center;

        // 16 vertices: 8 original cuboid corners + 8 translated
        let mut vertices = Vec::with_capacity(16);
        for &sx in &[-1.0f32, 1.0] {
            for &sy in &[-1.0f32, 1.0] {
                for &sz in &[-1.0f32, 1.0] {
                    let v = c + ax[0] * (he[0] * sx) + ax[1] * (he[1] * sy) + ax[2] * (he[2] * sz);
                    vertices.push(v);
                    vertices.push(v + translation);
                }
            }
        }

        // Plane normals: 6 original face normals + up to 6 side normals from edge×translation
        let mut normals: Vec<Vec3> = Vec::with_capacity(12);
        for i in 0..3 {
            normals.push(ax[i]);
            normals.push(-ax[i]);
            let side = ax[i].cross(translation);
            if side.length_squared() > 1e-10 {
                let side_n = side.normalize();
                normals.push(side_n);
                normals.push(-side_n);
            }
        }

        let planes: Vec<(Vec3, f32)> = normals
            .into_iter()
            .map(|n| {
                let d = crate::convex_polytope::max_projection(&vertices, n);
                (n, d)
            })
            .collect();

        // Derive OBB analytically: same axes, extended extents
        let mut obb = *self;
        obb.center += translation * 0.5;
        for i in 0..3 {
            obb.half_extents[i] += translation.dot(ax[i]).abs() * 0.5;
        }

        CuboidStretch::Unaligned(ConvexPolytope::with_obb(planes, vertices, obb))
    }
}

impl fmt::Display for Cuboid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = self.center;
        let he = self.half_extents;
        write!(
            f,
            "Cuboid(center: [{}, {}, {}], half_extents: [{}, {}, {}])",
            c.x, c.y, c.z, he[0], he[1], he[2]
        )
    }
}
