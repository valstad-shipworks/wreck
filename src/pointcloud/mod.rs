mod capt;
mod no_pcl;

use alloc::vec::Vec;
use core::fmt::Debug;

use glam::Vec3;
use hydroplane::{Gang, GangGlamExt, Vec3Wide, kernel};
use inherent::inherent;


use crate::Bounded;
use crate::Collides;
use crate::ConvexPolytope;
use crate::Scalable;
use crate::Transformable;
use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::convex_polytope::refer::RefConvexPolytope;
use crate::cuboid::Cuboid;
use crate::cylinder::Cylinder;
use crate::line::{Line, LineSegment, Ray};
use crate::plane::ConvexPolygon;
use crate::plane::Plane;
use crate::plane::ref_convex::RefConvexPolygon;
use crate::soa::SpheresSoA;
use crate::sphere::Sphere;

pub use no_pcl::NoPcl;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pointcloud {
    pub(crate) tree: capt::Capt<3, f32, u32>,
    pub(crate) spheres: SpheresSoA,
    pub(crate) point_radius: f32,
    pub(crate) r_range: (f32, f32),
    pub(crate) inverse_transform: Option<glam::Affine3A>,
}

impl Pointcloud {
    pub fn inverse_transform(&self) -> Option<&glam::Affine3A> {
        self.inverse_transform.as_ref()
    }

    pub fn tree(&self) -> &capt::Capt<3, f32, u32> {
        &self.tree
    }

    pub fn new(points: &[[f32; 3]], r_range: (f32, f32), point_radius: f32) -> Self {
        let tree = capt::Capt::<3, f32, u32>::with_point_radius(points, r_range, point_radius, 8);
        let mut spheres = SpheresSoA::with_capacity(points.len());
        for &pt in points {
            spheres.push(Sphere::new(Vec3::from(pt), point_radius));
        }
        Self {
            tree,
            spheres,
            point_radius,
            r_range,
            inverse_transform: None,
        }
    }

    #[inline]
    fn point_count(&self) -> usize {
        self.spheres.len()
    }

    /// The CAPT descends to a single leaf, whose affordance buffer only covers points within the
    /// construction radius `r_range.1` of that cell. A `query_simd` of a larger ball can miss points
    /// that fall in sibling leaves and wrongly report no collision, so a broadphase reject is only
    /// trustworthy when the queried ball (plus the baked-in `point_radius`) stays within that bound.
    /// When it doesn't, the caller skips the broadphase and runs the always-correct narrowphase scan.
    #[inline]
    fn broadphase_reject_sound(&self, radius: f32) -> bool {
        radius + self.point_radius <= self.r_range.1
    }

    /// Axis-aligned min/max of all point centres (without `point_radius`), via a SIMD reduction
    /// over the columnar SoA. Returns `(ZERO, ZERO)` for an empty cloud.
    fn point_bounds(&self) -> (Vec3, Vec3) {
        if self.point_count() == 0 {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let b = point_bounds_k(self.spheres.x(), self.spheres.y(), self.spheres.z());
        (Vec3::new(b[0], b[1], b[2]), Vec3::new(b[3], b[4], b[5]))
    }
}

#[inherent]
impl Transformable for Pointcloud {
    pub fn translate(&mut self, offset: glam::Vec3A) {
        self.transform(glam::Affine3A::from_translation(Vec3::from(offset)));
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3A) {
        self.transform(glam::Affine3A::from_mat3(mat.into()));
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.transform(glam::Affine3A::from_quat(quat));
    }

    #[inline]
    pub fn transform(&mut self, mat: glam::Affine3A) {
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
        let n = self.point_count();
        {
            let (xs, ys, zs, rs) = self.spheres.slices_mut();
            for i in 0..n {
                xs[i] *= factor;
                ys[i] *= factor;
                zs[i] *= factor;
                rs[i] *= factor;
            }
        }
        self.point_radius *= factor;
        self.r_range.0 *= factor;
        self.r_range.1 *= factor;
        if let Some(inv) = self.inverse_transform.take() {
            let fwd = inv.inverse();
            let (xs, ys, zs, _) = self.spheres.slices_mut();
            for i in 0..n {
                let v = fwd.transform_point3a(glam::Vec3A::new(xs[i], ys[i], zs[i]));
                xs[i] = v.x;
                ys[i] = v.y;
                zs[i] = v.z;
            }
        }
        let xs = self.spheres.x();
        let ys = self.spheres.y();
        let zs = self.spheres.z();
        let points: Vec<[f32; 3]> = (0..n)
            .map(|i| [xs[i], ys[i], zs[i]])
            .collect();
        self.tree = capt::Capt::<3, f32, u32>::with_point_radius(
            &points,
            self.r_range,
            self.point_radius,
            8,
        );
    }
}

#[inherent]
impl Bounded for Pointcloud {
    pub fn broadphase(&self) -> Sphere {
        let n = self.point_count();
        if n == 0 {
            return Sphere::new(Vec3::ZERO, 0.0);
        }
        let (min, max) = self.point_bounds();
        let center = (min + max) * 0.5;
        let half_diag = (max - min).length() * 0.5;
        Sphere::new(center, half_diag + self.point_radius)
    }

    pub fn obb(&self) -> Cuboid {
        self.aabb()
    }

    pub fn aabb(&self) -> Cuboid {
        if self.point_count() == 0 {
            return Cuboid::from_aabb(Vec3::ZERO, Vec3::ZERO);
        }
        let (min, max) = self.point_bounds();
        let r = Vec3::splat(self.point_radius);
        Cuboid::from_aabb(min - r, max + r)
    }
}

impl core::fmt::Display for Pointcloud {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Pointcloud(points: {}, radius: {})",
            self.point_count(),
            self.point_radius
        )
    }
}

// Sphere-CAPT: delegate to capt crate, SIMD batch for collides_many
impl Collides<Sphere> for Pointcloud {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        let center = match &self.inverse_transform {
            Some(inv) => Vec3::from(inv.transform_point3a(glam::Vec3A::from(sphere.center))),
            None => sphere.center,
        };
        self.tree.query_simd(&center.to_array(), sphere.radius)
    }
}

impl Collides<Pointcloud> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// Point-Pointcloud: a point is a zero-radius sphere
impl Collides<crate::Point> for Pointcloud {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, point: &crate::Point) -> bool {
        self.test::<BROADPHASE>(&Sphere::new(point.0, 0.0))
    }
}

impl Collides<Pointcloud> for crate::Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// Capsule-CAPT: bounding sphere broadphase + SIMD raw point narrow-phase
impl Collides<Capsule> for Pointcloud {
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
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
        if BROADPHASE && self.broadphase_reject_sound(br) && !self.tree.query_simd(&bc.to_array(), br) {
            return false;
        }

        let r_total = capsule.radius + self.point_radius;
        capsule_pcl_k(self.spheres.x(), self.spheres.y(), self.spheres.z(), capsule.p1, capsule.dir, capsule.rdv, r_total)
    }
}

impl Collides<Pointcloud> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// Cuboid-CAPT: bounding sphere broadphase + SIMD raw point narrow-phase
impl Collides<Cuboid> for Pointcloud {
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
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
        if BROADPHASE && self.broadphase_reject_sound(br) && !self.tree.query_simd(&cuboid.center.to_array(), br) {
            return false;
        }

        cuboid_pcl_k(
            self.spheres.x(),
            self.spheres.y(),
            self.spheres.z(),
            cuboid.center,
            cuboid.axes,
            cuboid.half_extents,
            self.point_radius * self.point_radius,
        )
    }
}

impl Collides<Pointcloud> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

impl Collides<Cylinder> for Pointcloud {
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        let transformed;
        let cyl = if let Some(inv) = &self.inverse_transform {
            transformed = {
                let mut c = *cyl;
                c.transform(*inv);
                c
            };
            &transformed
        } else {
            cyl
        };
        let (bc, br) = cyl.bounding_sphere();
        if BROADPHASE && self.broadphase_reject_sound(br) && !self.tree.query_simd(&bc.to_array(), br) {
            return false;
        }

        cylinder_pcl_k(
            self.spheres.x(),
            self.spheres.y(),
            self.spheres.z(),
            cyl.p1,
            cyl.dir,
            cyl.rdv,
            cyl.radius,
            self.point_radius,
            cyl.dir.dot(cyl.dir),
        )
    }
}

impl Collides<Pointcloud> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// ConvexPolytope-Pointcloud: CAPT broadphase + SIMD half-plane containment narrowphase
impl Pointcloud {
    /// SIMD narrowphase: test 8 cloud points at a time against all half-planes.
    /// A point is inside the polytope if `n·p - d - point_radius <= 0` for ALL planes.
    /// We track `max_sep` per point across planes; if any point's max_sep <= 0, it's inside.
    fn collides_polytope_ref<const BROADPHASE: bool>(
        &self,
        polytope: &RefConvexPolytope<'_>,
    ) -> bool {
        // Broadphase: polytope OBB bounding sphere vs CAPT
        let br = polytope.obb.bounding_sphere_radius();
        if BROADPHASE && self.broadphase_reject_sound(br) && !self.tree.query_simd(&polytope.obb.center.to_array(), br) {
            return false;
        }

        polytope_pcl_k(
            self.spheres.x(),
            self.spheres.y(),
            self.spheres.z(),
            polytope.planes,
            self.point_radius,
        )
    }
}

impl Collides<ConvexPolytope> for Pointcloud {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        if let Some(inv) = &self.inverse_transform {
            let mut p = polytope.clone();
            p.transform(*inv);
            return self.collides_polytope_ref::<BROADPHASE>(&RefConvexPolytope::from_heap(&p));
        }
        self.collides_polytope_ref::<BROADPHASE>(&RefConvexPolytope::from_heap(polytope))
    }
}

impl Collides<Pointcloud> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Pointcloud {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        if let Some(inv) = &self.inverse_transform {
            let mut p = *polytope;
            p.transform(*inv);
            return self.collides_polytope_ref::<BROADPHASE>(&RefConvexPolytope::from_array(&p));
        }
        self.collides_polytope_ref::<BROADPHASE>(&RefConvexPolytope::from_array(polytope))
    }
}

impl<const P: usize, const V: usize> Collides<Pointcloud> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// Plane-Pointcloud: each point (with point_radius) acts as a sphere against the plane.
// SIMD: test 8 points at a time against the half-space n·p <= d + point_radius.
impl Collides<Plane> for Pointcloud {
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        let (normal, d) = match &self.inverse_transform {
            Some(inv) => {
                let n = Vec3::from(inv.matrix3 * glam::Vec3A::from(plane.normal));
                let d = plane.d + glam::Vec3A::from(plane.normal).dot(inv.translation);
                (n, d)
            }
            None => (plane.normal, plane.d),
        };

        plane_pcl_k(self.spheres.x(), self.spheres.y(), self.spheres.z(), normal, d + self.point_radius)
    }
}

impl Collides<Pointcloud> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// ConvexPolygon-Pointcloud: point-polygon distance for each point (with point_radius).
impl Collides<ConvexPolygon> for Pointcloud {
    fn test<const BROADPHASE: bool>(&self, polygon: &ConvexPolygon) -> bool {
        let polygon = if let Some(inv) = &self.inverse_transform {
            let mut p = polygon.clone();
            p.transform(*inv);
            alloc::borrow::Cow::Owned(p)
        } else {
            alloc::borrow::Cow::Borrowed(polygon)
        };

        let r_sq = self.point_radius * self.point_radius;
        let poly = RefConvexPolygon::from_heap(polygon.as_ref());
        polygon_pcl_k(self.spheres.x(), self.spheres.y(), self.spheres.z(), poly, r_sq)
    }
}

impl Collides<Pointcloud> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        other.test::<BROADPHASE>(self)
    }
}

// Line/Ray/LineSegment-Pointcloud: each point (with point_radius) is a sphere.
// Use the shared SIMD sphere-test helper.
macro_rules! impl_line_pcl {
    ($LineType:ty, $t_min:expr, $t_max:expr) => {
        impl Collides<$LineType> for Pointcloud {
            fn test<const BROADPHASE: bool>(&self, line: &$LineType) -> bool {
                let (origin, dir, rdv) = match &self.inverse_transform {
                    Some(inv) => {
                        let o = Vec3::from(inv.transform_point3a(glam::Vec3A::from(line.origin_())));
                        let d = Vec3::from(inv.matrix3 * glam::Vec3A::from(line.dir_()));
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

                line_pcl_k(
                    self.spheres.x(),
                    self.spheres.y(),
                    self.spheres.z(),
                    origin,
                    dir,
                    rdv,
                    self.point_radius,
                    $t_min,
                    $t_max,
                )
            }
        }

        impl Collides<Pointcloud> for $LineType {
            #[inline]
            fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
                other.test::<BROADPHASE>(self)
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
        crate::line::rdv(self.dir)
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
        crate::line::rdv(self.dir)
    }
}

impl LineAccess for LineSegment {
    fn origin_(&self) -> Vec3 {
        self.start
    }
    fn dir_(&self) -> Vec3 {
        self.dir()
    }
    fn rdv_(&self) -> f32 {
        crate::line::rdv(self.dir())
    }
}

impl_line_pcl!(Line, f32::NEG_INFINITY, f32::INFINITY);
impl_line_pcl!(Ray, 0.0, f32::INFINITY);
impl_line_pcl!(LineSegment, 0.0, 1.0);

impl Collides<Pointcloud> for Pointcloud {
    fn test<const BROADPHASE: bool>(&self, other: &Pointcloud) -> bool {
        if self.point_count() == 0 || other.point_count() == 0 {
            return false;
        }

        let (iter_cloud, tree_cloud) = if self.point_count() <= other.point_count() {
            (self, other)
        } else {
            (other, self)
        };

        let combined_radius = iter_cloud.point_radius + tree_cloud.point_radius;

        let transform = match (&iter_cloud.inverse_transform, &tree_cloud.inverse_transform) {
            (None, None) => None,
            (None, Some(inv)) => Some(*inv),
            (Some(fwd_inv), None) => Some(fwd_inv.inverse()),
            (Some(fwd_inv), Some(tree_inv)) => Some(*tree_inv * fwd_inv.inverse()),
        };

        let sxs = iter_cloud.spheres.x();
        let sys = iter_cloud.spheres.y();
        let szs = iter_cloud.spheres.z();
        let n = iter_cloud.point_count();

        if let Some(mat) = &transform {
            for i in 0..n {
                let tp = mat.transform_point3a(glam::Vec3A::new(sxs[i], sys[i], szs[i]));
                if tree_cloud
                    .tree
                    .query_simd(&[tp.x, tp.y, tp.z], combined_radius)
                {
                    return true;
                }
            }
        } else {
            for i in 0..n {
                if tree_cloud
                    .tree
                    .query_simd(&[sxs[i], sys[i], szs[i]], combined_radius)
                {
                    return true;
                }
            }
        }

        false
    }
}

pub trait PointCloudMarker:
    __private::Sealed + Sized + Clone + Debug + Transformable + Scalable + Bounded
{
}

impl __private::Sealed for Pointcloud {}
impl PointCloudMarker for Pointcloud {}
impl __private::Sealed for NoPcl {}
impl PointCloudMarker for NoPcl {}

#[doc(hidden)]
mod __private {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------------
// Narrowphase kernels: one shape against every cloud point (with point_radius),
// walking the point SoA columns. A `lane < cnt` mask drops the inactive tail
// lanes a short final chunk leaves.
// ---------------------------------------------------------------------------

#[kernel]
#[allow(clippy::too_many_arguments)]
fn capsule_pcl_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    p1: Vec3,
    dir: Vec3,
    rdv: f32,
    r_total: f32,
) -> bool {
    let r_total_sq = ctx.splat(r_total * r_total);
    let p1v = ctx.splat_vec3(p1);
    let dv = ctx.splat_vec3(dir);
    let zero = ctx.splat(0.0);
    let one = ctx.splat(1.0);

    ctx.any_n([xs, ys, zs], |[x, y, z]| {
        let p = Vec3Wide::from([x, y, z]);
        let t = ((p - p1v).dot(dv) * rdv).max(zero).min(one);
        (p - p1v.add_scaled(dv, t)).length_squared().le(r_total_sq)
    })
}

#[kernel]
#[allow(clippy::too_many_arguments)]
fn cuboid_pcl_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    center: Vec3,
    axes: [Vec3; 3],
    he: [f32; 3],
    r_sq: f32,
) -> bool {
    let c = ctx.splat_vec3(center);
    let axes = axes.map(|a| ctx.splat_vec3(a));
    let rs = ctx.splat(r_sq);
    let zero = ctx.splat(0.0);

    ctx.any_n([xs, ys, zs], |[x, y, z]| {
        let df = Vec3Wide::from([x, y, z]) - c;
        let mut dist_sq = zero;
        for a in 0..3 {
            let proj = df.dot(axes[a]);
            let excess = (proj.abs() - he[a]).max(zero);
            dist_sq = dist_sq + excess * excess;
        }
        dist_sq.le(rs)
    })
}

#[kernel]
#[allow(clippy::too_many_arguments)]
fn cylinder_pcl_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    p1: Vec3,
    dir: Vec3,
    rdv: f32,
    cyl_radius: f32,
    pt_radius: f32,
    dir_sq_s: f32,
) -> bool {
    let p1v = ctx.splat_vec3(p1);
    let dv = ctx.splat_vec3(dir);
    let zero = ctx.splat(0.0);
    let one = ctx.splat(1.0);
    let r_total_sq = ctx.splat((cyl_radius + pt_radius) * (cyl_radius + pt_radius));
    let cyl_r_sq = ctx.splat(cyl_radius * cyl_radius);
    let pt_r_sq = ctx.splat(pt_radius * pt_radius);
    let dir_sq = ctx.splat(dir_sq_s);

    ctx.any_n([xs, ys, zs], |[x, y, z]| {
        let w = Vec3Wide::from([x, y, z]) - p1v;

        let t = w.dot(dv) * rdv;
        let t_c = t.max(zero).min(one);

        let perp = w - dv * t;
        let r_sq = perp.length_squared();

        let in_barrel = zero.le(t) & t.le(one);
        let barrel_hit = in_barrel & r_sq.le(r_total_sq);

        let t_excess = t - t_c;
        let d_axial_sq = t_excess * t_excess * dir_sq;

        let inside_r = r_sq.le(cyl_r_sq);
        let endcap_inside = inside_r & d_axial_sq.le(pt_r_sq);

        let l = r_sq + cyl_r_sq + d_axial_sq - pt_r_sq;
        let endcap_outside = l.le(zero) | (l * l).le(cyl_r_sq * r_sq * 4.0);

        let not_barrel = !in_barrel;
        barrel_hit | (not_barrel & (endcap_inside | endcap_outside))
    })
}

#[kernel]
fn polytope_pcl_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    planes: &'a [(Vec3, f32)],
    r: f32,
) -> bool {
    let neg_inf = ctx.splat(f32::NEG_INFINITY);
    let zero = ctx.splat(0.0);

    ctx.any_n([xs, ys, zs], |[px, py, pz]| {
        let mut max_sep = neg_inf;
        for &(normal, d) in planes {
            let sep = px * normal.x + py * normal.y + pz * normal.z - d - r;
            max_sep = max_sep.max(sep);
        }
        max_sep.le(zero)
    })
}

#[kernel]
fn plane_pcl_k<'a>(ctx: Gang, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32], normal: Vec3, threshold: f32) -> bool {
    let n = ctx.splat_vec3(normal);
    let thr = ctx.splat(threshold);

    ctx.any_n([xs, ys, zs], |[x, y, z]| n.dot(Vec3Wide::from([x, y, z])).le(thr))
}

/// Line/Ray/Segment narrowphase: is any cloud point within `point_radius` of the line?
#[kernel]
#[allow(clippy::too_many_arguments)]
fn line_pcl_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    origin: Vec3,
    dir: Vec3,
    rdv: f32,
    r: f32,
    t_min: f32,
    t_max: f32,
) -> bool {
    let o = ctx.splat_vec3(origin);
    let d = ctx.splat_vec3(dir);
    let lo = ctx.splat(t_min);
    let hi = ctx.splat(t_max);
    let r_sq = ctx.splat(r * r);

    ctx.any_n([xs, ys, zs], |[x, y, z]| {
        let p = Vec3Wide::from([x, y, z]);
        let t = ((p - o).dot(d) * rdv).max(lo).min(hi);
        (p - o.add_scaled(d, t)).length_squared().le(r_sq)
    })
}

/// Convex-polygon narrowphase: is any cloud point within `r` (squared `r_sq`) of the flat polygon?
/// Projects each point into the polygon's tangent frame, tests half-plane containment against every
/// edge, and for outside points adds the squared distance to the nearest edge segment — all in the
/// `(u, v)` plane. Edge count is tiny (3–8) and loop-invariant, so the per-edge inner loops stay in
/// registers and unroll.
#[kernel]
fn polygon_pcl_k<'a>(ctx: Gang, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32], poly: RefConvexPolygon<'a>, r_sq: f32) -> bool {
    let center = ctx.splat_vec3(poly.center);
    let normal = ctx.splat_vec3(poly.normal);
    let u_axis = ctx.splat_vec3(poly.u_axis);
    let v_axis = ctx.splat_vec3(poly.v_axis);
    let rs = ctx.splat(r_sq);
    let zero = ctx.splat(0.0);
    let one = ctx.splat(1.0);
    let big = ctx.splat(f32::MAX);

    let m = poly.vertices_2d.len();
    let vs = &poly.vertices_2d[..m];
    let ens = &poly.edge_normals_2d[..m];
    let eos = &poly.edge_offsets_2d[..m];

    ctx.any_n([xs, ys, zs], |[x, y, z]| {
        let d = Vec3Wide::from([x, y, z]) - center;
        let perp = d.dot(normal);
        let perp_sq = perp * perp;
        // dist_sq >= perp_sq, so a register with no lane within `r` of the plane cannot collide —
        // skip the per-edge distance work (the common case for a thin polygon in a wide cloud).
        let near = perp_sq.le(rs);
        if !near.any() {
            return near;
        }
        let u = d.dot(u_axis);
        let v = d.dot(v_axis);

        let mut inside = zero.le(zero);
        let mut min_dsq = big;
        for i in 0..m {
            let j = if i + 1 == m { 0 } else { i + 1 };
            let ax = vs[i][0];
            let ay = vs[i][1];
            inside = inside & (u * ens[i][0] + v * ens[i][1] - (eos[i] + 1e-6)).le(zero);

            let dx = vs[j][0] - ax;
            let dy = vs[j][1] - ay;
            let len_sq = dx * dx + dy * dy;
            let inv = if len_sq > f32::EPSILON { 1.0 / len_sq } else { 0.0 };
            let t = (((u - ax) * dx + (v - ay) * dy) * inv).max(zero).min(one);
            let ddx = u - (t * dx + ax);
            let ddy = v - (t * dy + ay);
            min_dsq = min_dsq.min(ddx * ddx + ddy * ddy);
        }

        (perp_sq + zero.select(inside, min_dsq)).le(rs)
    })
}

/// Axis-aligned min/max of three point columns as `[min_x, min_y, min_z, max_x, max_y, max_z]`,
/// in one SIMD pass (full-stride loop + a single masked tail; inactive tail lanes are forced to
/// the identities so they don't move the result).
#[kernel]
fn point_bounds_k<'a>(ctx: Gang, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32]) -> [f32; 6] {
    let n = ctx.lanes::<f32>();
    let len = xs.len();
    let pinf = ctx.splat(f32::INFINITY);
    let ninf = ctx.splat(f32::NEG_INFINITY);
    let mut mnx = pinf;
    let mut mny = pinf;
    let mut mnz = pinf;
    let mut mxx = ninf;
    let mut mxy = ninf;
    let mut mxz = ninf;

    let mut i = 0;
    while i + n <= len {
        let [x, y, z] = ctx.load_vec3([&xs[i..i + n], &ys[i..i + n], &zs[i..i + n]]).0;
        mnx = mnx.min(x);
        mny = mny.min(y);
        mnz = mnz.min(z);
        mxx = mxx.max(x);
        mxy = mxy.max(y);
        mxz = mxz.max(z);
        i += n;
    }
    if i < len {
        let active = ctx.active_mask(len - i);
        let [x, y, z] = ctx.load_partial_vec3([&xs[i..len], &ys[i..len], &zs[i..len]], 0.0).0;
        mnx = mnx.min(x.select(active, pinf));
        mny = mny.min(y.select(active, pinf));
        mnz = mnz.min(z.select(active, pinf));
        mxx = mxx.max(x.select(active, ninf));
        mxy = mxy.max(y.select(active, ninf));
        mxz = mxz.max(z.select(active, ninf));
    }
    [
        mnx.reduce_min(),
        mny.reduce_min(),
        mnz.reduce_min(),
        mxx.reduce_max(),
        mxy.reduce_max(),
        mxz.reduce_max(),
    ]
}

#[cfg(test)]
mod polygon_pcl_fuzz {
    use super::*;
    use crate::plane::ConvexPolygon;
    use glam::Vec3;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    fn rand_poly(rng: &mut SmallRng) -> ConvexPolygon {
        let center = Vec3::new(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0));
        let normal = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
            .normalize_or(Vec3::Y);
        let m = rng.random_range(3..=8usize);
        let radius = rng.random_range(0.3..2.0);
        let verts: Vec<[f32; 2]> = (0..m)
            .map(|i| {
                let a = core::f32::consts::TAU * i as f32 / m as f32;
                let r = radius * rng.random_range(0.7..1.0);
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        ConvexPolygon::new(center, normal, verts)
    }

    #[test]
    fn simd_matches_scalar_reference() {
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..400 {
            let poly = rand_poly(&mut rng);
            let refp = poly.as_ref();
            let n = rng.random_range(1..40usize);
            let pts: Vec<[f32; 3]> = (0..n)
                .map(|_| {
                    let p = poly.center
                        + Vec3::new(rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5));
                    [p.x, p.y, p.z]
                })
                .collect();
            let pr = rng.random_range(0.05..0.6);
            let pcl = Pointcloud::new(&pts, (pr, pr), pr);

            let r_sq = pcl.point_radius * pcl.point_radius;
            let expected = pts.iter().any(|&p| refp.point_dist_sq(Vec3::from(p)) <= r_sq);
            assert_eq!(pcl.collides(&poly), expected, "center {:?} pr {}", poly.center, pcl.point_radius);
        }
    }

    // The tree broadphase must never flip a collision answer: for any shape and cloud, the
    // broadphase-on path must agree with the always-correct narrowphase-only scan. A small `r_range`
    // relative to the shape size forces the CAPT out of its valid query radius, which used to yield
    // broadphase false-negatives (points in sibling leaves missed by the single-leaf descent).
    #[test]
    fn broadphase_never_flips_answer() {
        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..600 {
            let n = rng.random_range(4..80usize);
            let pts: Vec<[f32; 3]> = (0..n)
                .map(|_| [rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])
                .collect();
            let pr = rng.random_range(0.05..0.4);
            let pcl = Pointcloud::new(&pts, (pr, pr), pr);

            let c = Vec3::new(rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0));
            let seg = Vec3::new(rng_ext(&mut rng), rng_ext(&mut rng), rng_ext(&mut rng));

            let cap = Capsule::new(c, c + seg, rng.random_range(0.2..2.5));
            assert_eq!(
                Collides::<Capsule>::test::<true>(&pcl, &cap),
                Collides::<Capsule>::test::<false>(&pcl, &cap),
                "capsule broadphase flipped"
            );

            let he = [rng.random_range(0.2..2.5), rng.random_range(0.2..2.5), rng.random_range(0.2..2.5)];
            let cub = Cuboid::new(c, [Vec3::X, Vec3::Y, Vec3::Z], he);
            assert_eq!(
                Collides::<Cuboid>::test::<true>(&pcl, &cub),
                Collides::<Cuboid>::test::<false>(&pcl, &cub),
                "cuboid broadphase flipped"
            );

            let cyl = Cylinder::new(c, c + seg, rng.random_range(0.2..2.5));
            assert_eq!(
                Collides::<Cylinder>::test::<true>(&pcl, &cyl),
                Collides::<Cylinder>::test::<false>(&pcl, &cyl),
                "cylinder broadphase flipped"
            );
        }
    }

    fn rng_ext(rng: &mut SmallRng) -> f32 {
        let v = rng.random_range(0.5..4.0);
        if rng.random_bool(0.5) { v } else { -v }
    }
}
