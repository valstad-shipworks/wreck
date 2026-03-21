mod capt;
mod no_pcl;

use std::fmt::Debug;

use glam::Vec3;
use inherent::inherent;
use wide::{CmpLe, f32x8};

use crate::Bounded;
use crate::Collides;
use crate::ConvexPolytope;
use crate::Scalable;
use crate::Transformable;
use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::convex_polytope::refer::RefConvexPolytope;
use crate::cuboid::Cuboid;
use crate::line::{Line, LineSegment, Ray};
use crate::plane::ConvexPolygon;
use crate::plane::Plane;
use crate::sphere::Sphere;

pub use no_pcl::NoPcl;

#[derive(Debug, Clone)]
pub struct Pointcloud {
    tree: capt::Capt<3, f32, u32>,
    points: Vec<[f32; 3]>,
    point_radius: f32,
    r_range: (f32, f32),
    inverse_transform: Option<glam::Affine3>,
}

impl Pointcloud {
    pub fn new(points: &[[f32; 3]], r_range: (f32, f32), point_radius: f32) -> Self {
        let tree = capt::Capt::<3, f32, u32>::with_point_radius(points, r_range, point_radius, 8);
        Self {
            tree,
            points: points.to_vec(),
            point_radius,
            r_range,
            inverse_transform: None,
        }
    }
}

#[inherent]
impl Transformable for Pointcloud {
    pub fn translate(&mut self, offset: Vec3) {
        self.transform(glam::Affine3::from_translation(offset));
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.transform(glam::Affine3::from_mat3(mat));
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.transform(glam::Affine3::from_quat(quat));
    }

    #[inline]
    pub fn transform(&mut self, mat: glam::Affine3) {
        let new_inv = mat.inverse();
        self.inverse_transform = Some(match self.inverse_transform {
            Some(existing) => existing * new_inv,
            None => new_inv,
        });
    }
}

#[inherent]
impl Scalable for Pointcloud {
    pub fn scale(&mut self, factor: f32) {
        for pt in &mut self.points {
            pt[0] *= factor;
            pt[1] *= factor;
            pt[2] *= factor;
        }
        self.point_radius *= factor;
        self.r_range.0 *= factor;
        self.r_range.1 *= factor;
        // Flush the lazy transform: apply it to points, then clear it
        if let Some(inv) = self.inverse_transform.take() {
            let fwd = inv.inverse();
            for pt in &mut self.points {
                let v = fwd.transform_point3(Vec3::from(*pt));
                *pt = v.to_array();
            }
        }
        self.tree = capt::Capt::<3, f32, u32>::with_point_radius(
            &self.points,
            self.r_range,
            self.point_radius,
            8,
        );
    }
}

#[inherent]
impl Bounded for Pointcloud {
    pub fn broadphase(&self) -> Sphere {
        if self.points.is_empty() {
            return Sphere::new(Vec3::ZERO, 0.0);
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in &self.points {
            let v = Vec3::from(*p);
            min = min.min(v);
            max = max.max(v);
        }
        let center = (min + max) * 0.5;
        let half_diag = (max - min).length() * 0.5;
        Sphere::new(center, half_diag + self.point_radius)
    }

    pub fn obb(&self) -> Cuboid {
        self.aabb()
    }

    pub fn aabb(&self) -> Cuboid {
        if self.points.is_empty() {
            return Cuboid::from_aabb(Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in &self.points {
            let v = Vec3::from(*p);
            min = min.min(v);
            max = max.max(v);
        }
        let r = Vec3::splat(self.point_radius);
        Cuboid::from_aabb(min - r, max + r)
    }
}

// Sphere-CAPT: delegate to capt crate, SIMD batch for collides_many
impl Collides<Sphere> for Pointcloud {
    #[inline]
    fn collides(&self, sphere: &Sphere) -> bool {
        let center = match &self.inverse_transform {
            Some(inv) => inv.transform_point3(sphere.center),
            None => sphere.center,
        };
        self.tree.collides(&center.to_array(), sphere.radius)
    }

    fn collides_many(&self, spheres: &[Sphere]) -> bool {
        let chunks = spheres.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let centers: [f32x8; 3] = match &self.inverse_transform {
                Some(inv) => {
                    let transformed: [Vec3; 8] =
                        std::array::from_fn(|i| inv.transform_point3(chunk[i].center));
                    [
                        f32x8::new(std::array::from_fn(|i| transformed[i].x)),
                        f32x8::new(std::array::from_fn(|i| transformed[i].y)),
                        f32x8::new(std::array::from_fn(|i| transformed[i].z)),
                    ]
                }
                None => [
                    f32x8::new(std::array::from_fn(|i| chunk[i].center.x)),
                    f32x8::new(std::array::from_fn(|i| chunk[i].center.y)),
                    f32x8::new(std::array::from_fn(|i| chunk[i].center.z)),
                ],
            };
            let radii = f32x8::new(std::array::from_fn(|i| chunk[i].radius));
            if self.tree.collides_simd(&centers, radii) {
                return true;
            }
        }

        for sphere in remainder {
            let center = match &self.inverse_transform {
                Some(inv) => inv.transform_point3(sphere.center),
                None => sphere.center,
            };
            if self.tree.collides(&center.to_array(), sphere.radius) {
                return true;
            }
        }
        false
    }
}

impl Collides<Pointcloud> for Sphere {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// Point-Pointcloud: a point is a zero-radius sphere
impl Collides<crate::Point> for Pointcloud {
    #[inline]
    fn collides(&self, point: &crate::Point) -> bool {
        self.collides(&Sphere::new(point.0, 0.0))
    }
}

impl Collides<Pointcloud> for crate::Point {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// Capsule-CAPT: bounding sphere broadphase + SIMD raw point narrow-phase
impl Collides<Capsule> for Pointcloud {
    fn collides(&self, capsule: &Capsule) -> bool {
        let transformed;
        let capsule = if let Some(inv) = &self.inverse_transform {
            transformed = {
                let mut c = *capsule;
                c.transform(*inv);
                c
            };
            &transformed
        } else {
            capsule
        };
        let (bc, br) = capsule.bounding_sphere();
        if !self.tree.collides(&bc.to_array(), br) {
            return false;
        }

        let r_total = capsule.radius + self.point_radius;
        let r_total_sq = f32x8::splat(r_total * r_total);

        let p1x = f32x8::splat(capsule.p1.x);
        let p1y = f32x8::splat(capsule.p1.y);
        let p1z = f32x8::splat(capsule.p1.z);
        let dx = f32x8::splat(capsule.dir.x);
        let dy = f32x8::splat(capsule.dir.y);
        let dz = f32x8::splat(capsule.dir.z);
        let rdv = f32x8::splat(capsule.rdv);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = self.points.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut px = [0.0f32; 8];
            let mut py = [0.0f32; 8];
            let mut pz = [0.0f32; 8];
            for (i, pt) in chunk.iter().enumerate() {
                px[i] = pt[0];
                py[i] = pt[1];
                pz[i] = pt[2];
            }
            let px = f32x8::new(px);
            let py = f32x8::new(py);
            let pz = f32x8::new(pz);

            // Project point onto capsule segment, clamp t
            let dfx = px - p1x;
            let dfy = py - p1y;
            let dfz = pz - p1z;
            let t = (dfx * dx + dfy * dy + dfz * dz) * rdv;
            let t = t.max(zero).min(one);

            // Closest point on segment
            let cx = p1x + dx * t;
            let cy = p1y + dy * t;
            let cz = p1z + dz * t;

            let ex = px - cx;
            let ey = py - cy;
            let ez = pz - cz;
            let dist_sq = ex * ex + ey * ey + ez * ez;

            if dist_sq.simd_le(r_total_sq).any() {
                return true;
            }
        }

        // Scalar remainder
        let r_total_sq_s = r_total * r_total;
        for &pt in remainder {
            let p = Vec3::from(pt);
            let closest = capsule.closest_point_to(p);
            let d = p - closest;
            if d.dot(d) <= r_total_sq_s {
                return true;
            }
        }
        false
    }
}

impl Collides<Pointcloud> for Capsule {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// Cuboid-CAPT: bounding sphere broadphase + SIMD raw point narrow-phase
impl Collides<Cuboid> for Pointcloud {
    fn collides(&self, cuboid: &Cuboid) -> bool {
        let transformed;
        let cuboid = if let Some(inv) = &self.inverse_transform {
            transformed = {
                let mut c = *cuboid;
                c.transform(*inv);
                c
            };
            &transformed
        } else {
            cuboid
        };
        let br = cuboid.bounding_sphere_radius();
        if !self.tree.collides(&cuboid.center.to_array(), br) {
            return false;
        }

        let r_sq = f32x8::splat(self.point_radius * self.point_radius);
        let ccx = f32x8::splat(cuboid.center.x);
        let ccy = f32x8::splat(cuboid.center.y);
        let ccz = f32x8::splat(cuboid.center.z);

        // Precompute axis components and half-extents as SIMD
        let ax = [
            f32x8::splat(cuboid.axes[0].x),
            f32x8::splat(cuboid.axes[0].y),
            f32x8::splat(cuboid.axes[0].z),
        ];
        let ay = [
            f32x8::splat(cuboid.axes[1].x),
            f32x8::splat(cuboid.axes[1].y),
            f32x8::splat(cuboid.axes[1].z),
        ];
        let az = [
            f32x8::splat(cuboid.axes[2].x),
            f32x8::splat(cuboid.axes[2].y),
            f32x8::splat(cuboid.axes[2].z),
        ];
        let he = [
            f32x8::splat(cuboid.half_extents[0]),
            f32x8::splat(cuboid.half_extents[1]),
            f32x8::splat(cuboid.half_extents[2]),
        ];
        let zero = f32x8::splat(0.0);
        let axes_comp = [ax, ay, az];

        let chunks = self.points.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut px = [0.0f32; 8];
            let mut py = [0.0f32; 8];
            let mut pz = [0.0f32; 8];
            for (i, pt) in chunk.iter().enumerate() {
                px[i] = pt[0];
                py[i] = pt[1];
                pz[i] = pt[2];
            }
            let dfx = f32x8::new(px) - ccx;
            let dfy = f32x8::new(py) - ccy;
            let dfz = f32x8::new(pz) - ccz;

            let mut dist_sq = zero;
            for i in 0..3 {
                let proj = dfx * axes_comp[i][0] + dfy * axes_comp[i][1] + dfz * axes_comp[i][2];
                // abs via flip_signs trick: proj.abs() not available, use max(-proj, proj)
                let abs_proj = proj.max(-proj);
                let excess = (abs_proj - he[i]).max(zero);
                dist_sq = dist_sq + excess * excess;
            }

            if dist_sq.simd_le(r_sq).any() {
                return true;
            }
        }

        let r_sq_s = self.point_radius * self.point_radius;
        for &pt in remainder {
            let p = Vec3::from(pt);
            if cuboid.point_dist_sq(p) <= r_sq_s {
                return true;
            }
        }
        false
    }
}

impl Collides<Pointcloud> for Cuboid {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// ConvexPolytope-Pointcloud: CAPT broadphase + SIMD half-plane containment narrowphase
impl Pointcloud {
    /// SIMD narrowphase: test 8 cloud points at a time against all half-planes.
    /// A point is inside the polytope if `n·p - d - point_radius <= 0` for ALL planes.
    /// We track `max_sep` per point across planes; if any point's max_sep <= 0, it's inside.
    fn collides_polytope_ref(&self, polytope: &RefConvexPolytope<'_>) -> bool {
        // Broadphase: polytope OBB bounding sphere vs CAPT
        let br = polytope.obb.bounding_sphere_radius();
        if !self.tree.collides(&polytope.obb.center.to_array(), br) {
            return false;
        }

        let r = f32x8::splat(self.point_radius);
        let zero = f32x8::ZERO;

        let chunks = self.points.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut px = [0.0f32; 8];
            let mut py = [0.0f32; 8];
            let mut pz = [0.0f32; 8];
            for (i, pt) in chunk.iter().enumerate() {
                px[i] = pt[0];
                py[i] = pt[1];
                pz[i] = pt[2];
            }
            let px = f32x8::new(px);
            let py = f32x8::new(py);
            let pz = f32x8::new(pz);

            // For each point, track max separation across all planes.
            // A point is inside iff max_sep <= 0.
            let mut max_sep = f32x8::splat(f32::NEG_INFINITY);
            for &(normal, d) in polytope.planes {
                let sep = f32x8::splat(normal.x) * px
                    + f32x8::splat(normal.y) * py
                    + f32x8::splat(normal.z) * pz
                    - f32x8::splat(d)
                    - r;
                max_sep = max_sep.max(sep);
            }

            if max_sep.simd_le(zero).any() {
                return true;
            }
        }

        // Scalar remainder
        let r_s = self.point_radius;
        for &pt in remainder {
            let p = Vec3::from(pt);
            let mut inside = true;
            for &(normal, d) in polytope.planes {
                if normal.dot(p) - d - r_s > 0.0 {
                    inside = false;
                    break;
                }
            }
            if inside {
                return true;
            }
        }
        false
    }
}

impl Collides<ConvexPolytope> for Pointcloud {
    #[inline]
    fn collides(&self, polytope: &ConvexPolytope) -> bool {
        if let Some(inv) = &self.inverse_transform {
            let mut p = polytope.clone();
            p.transform(*inv);
            return self.collides_polytope_ref(&RefConvexPolytope::from_heap(&p));
        }
        self.collides_polytope_ref(&RefConvexPolytope::from_heap(polytope))
    }
}

impl Collides<Pointcloud> for ConvexPolytope {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Pointcloud {
    #[inline]
    fn collides(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        if let Some(inv) = &self.inverse_transform {
            let mut p = *polytope;
            p.transform(*inv);
            return self.collides_polytope_ref(&RefConvexPolytope::from_array(&p));
        }
        self.collides_polytope_ref(&RefConvexPolytope::from_array(polytope))
    }
}

impl<const P: usize, const V: usize> Collides<Pointcloud> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// Plane-Pointcloud: each point (with point_radius) acts as a sphere against the plane.
// SIMD: test 8 points at a time against the half-space n·p <= d + point_radius.
impl Collides<Plane> for Pointcloud {
    fn collides(&self, plane: &Plane) -> bool {
        let (normal, d) = match &self.inverse_transform {
            Some(inv) => {
                let n = inv.matrix3 * plane.normal;
                let d = plane.d + plane.normal.dot(inv.translation);
                (n, d)
            }
            None => (plane.normal, plane.d),
        };

        let nx = f32x8::splat(normal.x);
        let ny = f32x8::splat(normal.y);
        let nz = f32x8::splat(normal.z);
        let threshold = f32x8::splat(d + self.point_radius);

        let chunks = self.points.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut px = [0.0f32; 8];
            let mut py = [0.0f32; 8];
            let mut pz = [0.0f32; 8];
            for (i, pt) in chunk.iter().enumerate() {
                px[i] = pt[0];
                py[i] = pt[1];
                pz[i] = pt[2];
            }
            let proj = nx * f32x8::new(px) + ny * f32x8::new(py) + nz * f32x8::new(pz);
            if proj.simd_le(threshold).any() {
                return true;
            }
        }

        let threshold_s = d + self.point_radius;
        for &pt in remainder {
            let p = Vec3::from(pt);
            if normal.dot(p) <= threshold_s {
                return true;
            }
        }
        false
    }
}

impl Collides<Pointcloud> for Plane {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// ConvexPolygon-Pointcloud: point-polygon distance for each point (with point_radius).
impl Collides<ConvexPolygon> for Pointcloud {
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        let polygon = if let Some(inv) = &self.inverse_transform {
            let mut p = polygon.clone();
            p.transform(*inv);
            std::borrow::Cow::Owned(p)
        } else {
            std::borrow::Cow::Borrowed(polygon)
        };

        let r_sq = self.point_radius * self.point_radius;
        for &pt in &self.points {
            let p = Vec3::from(pt);
            if polygon.point_dist_sq(p) <= r_sq {
                return true;
            }
        }
        false
    }
}

impl Collides<Pointcloud> for ConvexPolygon {
    #[inline]
    fn collides(&self, other: &Pointcloud) -> bool {
        other.collides(self)
    }
}

// Line/Ray/LineSegment-Pointcloud: each point (with point_radius) is a sphere.
// Use the shared SIMD sphere-test helper.
macro_rules! impl_line_pcl {
    ($LineType:ty, $t_min:expr, $t_max:expr) => {
        impl Collides<$LineType> for Pointcloud {
            fn collides(&self, line: &$LineType) -> bool {
                let (origin, dir, rdv) = match &self.inverse_transform {
                    Some(inv) => {
                        let o = inv.transform_point3(line.origin_());
                        let d = inv.matrix3 * line.dir_();
                        let len_sq = d.dot(d);
                        let rdv = if len_sq > f32::EPSILON {
                            1.0 / len_sq
                        } else {
                            0.0
                        };
                        (o, d, rdv)
                    }
                    None => (line.origin_(), line.dir_(), line.rdv_()),
                };

                let r = self.point_radius;
                let r_sq = r * r;
                for &pt in &self.points {
                    let p = Vec3::from(pt);
                    let t = crate::line::closest_t_to_point(origin, dir, rdv, p, $t_min, $t_max);
                    let closest = origin + dir * t;
                    let d = p - closest;
                    if d.dot(d) <= r_sq {
                        return true;
                    }
                }
                false
            }
        }

        impl Collides<Pointcloud> for $LineType {
            #[inline]
            fn collides(&self, other: &Pointcloud) -> bool {
                other.collides(self)
            }
        }
    };
}

// Helper trait to access fields uniformly across Line/Ray/LineSegment
trait LineAccess {
    fn origin_(&self) -> Vec3;
    fn dir_(&self) -> Vec3;
    fn rdv_(&self) -> f32;
}

impl LineAccess for Line {
    fn origin_(&self) -> Vec3 {
        self.origin
    }
    fn dir_(&self) -> Vec3 {
        self.dir
    }
    fn rdv_(&self) -> f32 {
        self.rdv
    }
}

impl LineAccess for Ray {
    fn origin_(&self) -> Vec3 {
        self.origin
    }
    fn dir_(&self) -> Vec3 {
        self.dir
    }
    fn rdv_(&self) -> f32 {
        self.rdv
    }
}

impl LineAccess for LineSegment {
    fn origin_(&self) -> Vec3 {
        self.p1
    }
    fn dir_(&self) -> Vec3 {
        self.dir
    }
    fn rdv_(&self) -> f32 {
        self.rdv
    }
}

impl_line_pcl!(Line, f32::NEG_INFINITY, f32::INFINITY);
impl_line_pcl!(Ray, 0.0, f32::INFINITY);
impl_line_pcl!(LineSegment, 0.0, 1.0);

pub trait PointCloudMarker: __private::Sealed + Sized + Clone + Debug + Transformable + Scalable + Bounded {}

impl __private::Sealed for Pointcloud {}
impl PointCloudMarker for Pointcloud {}
impl __private::Sealed for NoPcl {}
impl PointCloudMarker for NoPcl {}

#[doc(hidden)]
mod __private {
    pub trait Sealed {}
}
