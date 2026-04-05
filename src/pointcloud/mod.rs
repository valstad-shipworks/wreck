mod capt;
mod no_pcl;

use alloc::vec::Vec;
use core::fmt::Debug;

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
use crate::cylinder::Cylinder;
use crate::line::{Line, LineSegment, Ray};
use crate::plane::ConvexPolygon;
use crate::plane::Plane;
use crate::soa::SpheresSoA;
use crate::sphere::Sphere;

pub use no_pcl::NoPcl;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pointcloud {
    tree: capt::Capt<3, f32, u32>,
    spheres: SpheresSoA,
    point_radius: f32,
    r_range: (f32, f32),
    inverse_transform: Option<glam::Affine3A>,
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

    #[inline]
    fn full_chunks(&self) -> usize {
        self.spheres.len() / 8
    }

    #[inline]
    fn remainder_start(&self) -> usize {
        self.full_chunks() * 8
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
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let xs = self.spheres.x();
        let ys = self.spheres.y();
        let zs = self.spheres.z();
        for i in 0..n {
            let v = Vec3::new(xs[i], ys[i], zs[i]);
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
        let n = self.point_count();
        if n == 0 {
            return Cuboid::from_aabb(Vec3::ZERO, Vec3::ZERO);
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let xs = self.spheres.x();
        let ys = self.spheres.y();
        let zs = self.spheres.z();
        for i in 0..n {
            let v = Vec3::new(xs[i], ys[i], zs[i]);
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
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        let center = match &self.inverse_transform {
            Some(inv) => Vec3::from(inv.transform_point3a(glam::Vec3A::from(sphere.center))),
            None => sphere.center,
        };
        self.tree.collides(&center.to_array(), sphere.radius)
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
        if BROADPHASE {
            if !self.tree.collides(&bc.to_array(), br) {
                return false;
            }
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

        let full_chunks = self.full_chunks();
        let sxs = self.spheres.x();
        let sys = self.spheres.y();
        let szs = self.spheres.z();
        for i in 0..full_chunks {
            let base = i * 8;
            let px = f32x8::new(sxs[base..base + 8].try_into().unwrap());
            let py = f32x8::new(sys[base..base + 8].try_into().unwrap());
            let pz = f32x8::new(szs[base..base + 8].try_into().unwrap());

            let dfx = px - p1x;
            let dfy = py - p1y;
            let dfz = pz - p1z;
            let t = (dfx * dx + dfy * dy + dfz * dz) * rdv;
            let t = t.max(zero).min(one);

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

        let r_total_sq_s = r_total * r_total;
        for i in self.remainder_start()..self.point_count() {
            let p = Vec3::new(sxs[i], sys[i], szs[i]);
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
        if BROADPHASE {
            if !self.tree.collides(&cuboid.center.to_array(), br) {
                return false;
            }
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

        let full_chunks = self.full_chunks();
        let sxs = self.spheres.x();
        let sys = self.spheres.y();
        let szs = self.spheres.z();
        for i in 0..full_chunks {
            let base = i * 8;
            let dfx = f32x8::new(sxs[base..base + 8].try_into().unwrap()) - ccx;
            let dfy = f32x8::new(sys[base..base + 8].try_into().unwrap()) - ccy;
            let dfz = f32x8::new(szs[base..base + 8].try_into().unwrap()) - ccz;

            let mut dist_sq = zero;
            for i in 0..3 {
                let proj = dfx * axes_comp[i][0] + dfy * axes_comp[i][1] + dfz * axes_comp[i][2];
                let abs_proj = proj.max(-proj);
                let excess = (abs_proj - he[i]).max(zero);
                dist_sq = dist_sq + excess * excess;
            }

            if dist_sq.simd_le(r_sq).any() {
                return true;
            }
        }

        let r_sq_s = self.point_radius * self.point_radius;
        for i in self.remainder_start()..self.point_count() {
            let p = Vec3::new(sxs[i], sys[i], szs[i]);
            if cuboid.point_dist_sq(p) <= r_sq_s {
                return true;
            }
        }
        false
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
        if BROADPHASE {
            if !self.tree.collides(&bc.to_array(), br) {
                return false;
            }
        }

        let r_total = cyl.radius + self.point_radius;
        let r_total_sq = r_total * r_total;

        let p1x8 = f32x8::splat(cyl.p1.x);
        let p1y8 = f32x8::splat(cyl.p1.y);
        let p1z8 = f32x8::splat(cyl.p1.z);
        let dx8 = f32x8::splat(cyl.dir.x);
        let dy8 = f32x8::splat(cyl.dir.y);
        let dz8 = f32x8::splat(cyl.dir.z);
        let rdv8 = f32x8::splat(cyl.rdv);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let r_total_sq8 = f32x8::splat(r_total_sq);
        let cyl_r_sq8 = f32x8::splat(cyl.radius * cyl.radius);
        let pt_r_sq8 = f32x8::splat(self.point_radius * self.point_radius);
        let dir_sq8 = f32x8::splat(cyl.dir.dot(cyl.dir));
        let four_cyl_r_sq = f32x8::splat(4.0 * cyl.radius * cyl.radius);

        let full_chunks = self.full_chunks();
        let sxs = self.spheres.x();
        let sys = self.spheres.y();
        let szs = self.spheres.z();
        for i in 0..full_chunks {
            let base = i * 8;
            let px = f32x8::new(sxs[base..base + 8].try_into().unwrap());
            let py = f32x8::new(sys[base..base + 8].try_into().unwrap());
            let pz = f32x8::new(szs[base..base + 8].try_into().unwrap());

            let wx = px - p1x8;
            let wy = py - p1y8;
            let wz = pz - p1z8;

            let t = (wx * dx8 + wy * dy8 + wz * dz8) * rdv8;
            let t_c = t.max(zero).min(one);

            let perpx = wx - dx8 * t;
            let perpy = wy - dy8 * t;
            let perpz = wz - dz8 * t;
            let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;

            let in_barrel = zero.simd_le(t) & t.simd_le(one);
            let barrel_hit = in_barrel & r_sq.simd_le(r_total_sq8);

            let t_excess = t - t_c;
            let d_axial_sq = t_excess * t_excess * dir_sq8;

            let inside_r = r_sq.simd_le(cyl_r_sq8);
            let endcap_inside = inside_r & d_axial_sq.simd_le(pt_r_sq8);

            let l = r_sq + cyl_r_sq8 + d_axial_sq - pt_r_sq8;
            let endcap_outside = l.simd_le(zero) | (l * l).simd_le(four_cyl_r_sq * r_sq);

            let not_barrel = !in_barrel;
            let hit = barrel_hit | (not_barrel & (endcap_inside | endcap_outside));
            if hit.any() {
                return true;
            }
        }

        for i in self.remainder_start()..self.point_count() {
            let p = Vec3::new(sxs[i], sys[i], szs[i]);
            if cyl.point_dist_sq(p) <= self.point_radius * self.point_radius {
                return true;
            }
        }
        false
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
        if BROADPHASE {
            let br = polytope.obb.bounding_sphere_radius();
            if !self.tree.collides(&polytope.obb.center.to_array(), br) {
                return false;
            }
        }

        let r = f32x8::splat(self.point_radius);
        let zero = f32x8::ZERO;

        let full_chunks = self.full_chunks();
        let sxs = self.spheres.x();
        let sys = self.spheres.y();
        let szs = self.spheres.z();
        for i in 0..full_chunks {
            let base = i * 8;
            let px = f32x8::new(sxs[base..base + 8].try_into().unwrap());
            let py = f32x8::new(sys[base..base + 8].try_into().unwrap());
            let pz = f32x8::new(szs[base..base + 8].try_into().unwrap());

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

        let r_s = self.point_radius;
        for i in self.remainder_start()..self.point_count() {
            let p = Vec3::new(sxs[i], sys[i], szs[i]);
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

        let nx = f32x8::splat(normal.x);
        let ny = f32x8::splat(normal.y);
        let nz = f32x8::splat(normal.z);
        let threshold = f32x8::splat(d + self.point_radius);

        let full_chunks = self.full_chunks();
        let sxs = self.spheres.x();
        let sys = self.spheres.y();
        let szs = self.spheres.z();
        for i in 0..full_chunks {
            let base = i * 8;
            let proj = nx * f32x8::new(sxs[base..base + 8].try_into().unwrap())
                + ny * f32x8::new(sys[base..base + 8].try_into().unwrap())
                + nz * f32x8::new(szs[base..base + 8].try_into().unwrap());
            if proj.simd_le(threshold).any() {
                return true;
            }
        }

        let threshold_s = d + self.point_radius;
        for i in self.remainder_start()..self.point_count() {
            let p = Vec3::new(sxs[i], sys[i], szs[i]);
            if normal.dot(p) <= threshold_s {
                return true;
            }
        }
        false
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
        let sxs = self.spheres.x();
        let sys = self.spheres.y();
        let szs = self.spheres.z();
        for i in 0..self.point_count() {
            let p = Vec3::new(sxs[i], sys[i], szs[i]);
            if polygon.point_dist_sq(p) <= r_sq {
                return true;
            }
        }
        false
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

                let r = self.point_radius;
                let r_sq = r * r;
                let sxs = self.spheres.x();
                let sys = self.spheres.y();
                let szs = self.spheres.z();
                for i in 0..self.point_count() {
                    let p = Vec3::new(sxs[i], sys[i], szs[i]);
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
