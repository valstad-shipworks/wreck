use alloc::vec::Vec;
use core::fmt;

use glam::Vec3;

use inherent::inherent;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::plane::{ConvexPolygon, Plane};
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

const T_MIN: f32 = 0.0;
const T_MAX: f32 = f32::INFINITY;

/// A ray: `origin + t * dir` for `t ∈ [0, ∞)`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    pub(crate) rdv: f32,
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        wreck_assert!(
            dir.dot(dir) > f32::EPSILON,
            "Ray direction must be non-zero"
        );
        let len_sq = dir.dot(dir);
        Self {
            origin,
            dir,
            rdv: if len_sq > f32::EPSILON {
                1.0 / len_sq
            } else {
                0.0
            },
        }
    }
}

#[inherent]
impl Scalable for Ray {
    pub fn scale(&mut self, factor: f32) {
        self.dir *= factor;
        let len_sq = self.dir.dot(self.dir);
        self.rdv = if len_sq > f32::EPSILON {
            1.0 / len_sq
        } else {
            0.0
        };
    }
}

#[inherent]
impl Transformable for Ray {
    pub fn translate(&mut self, offset: glam::Vec3A) {
        self.origin = Vec3::from(glam::Vec3A::from(self.origin) + offset);
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3A) {
        self.origin = Vec3::from(mat * glam::Vec3A::from(self.origin));
        self.dir = Vec3::from(mat * glam::Vec3A::from(self.dir));
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.origin = quat * self.origin;
        self.dir = quat * self.dir;
    }

    pub fn transform(&mut self, mat: glam::Affine3A) {
        self.origin = Vec3::from(mat.transform_point3a(glam::Vec3A::from(self.origin)));
        self.dir = Vec3::from(mat.matrix3 * glam::Vec3A::from(self.dir));
    }
}

// --- Ray–Stretch ---

const INF: f32 = 1e12;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RayStretch {
    Parallel(Ray),
    Polygon(ConvexPolygon),
}

impl Stretchable for Ray {
    type Output = RayStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let cross = self.dir.cross(translation);
        if cross.length_squared() < 1e-10 {
            // Parallel: shift the ray origin if needed
            let proj = translation.dot(self.dir);
            let new_origin = if proj >= 0.0 {
                self.origin
            } else {
                self.origin + translation
            };
            return RayStretch::Parallel(Ray::new(new_origin, self.dir));
        }

        // Non-parallel: semi-infinite strip approximated with 1e12
        let dir_norm = self.dir.normalize();
        let far = dir_norm * INF;

        let normal = cross.normalize();
        let up = if normal.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u_axis = normal.cross(up).normalize();
        let v_axis = u_axis.cross(normal);

        let corners = [
            self.origin,
            self.origin + far,
            self.origin + far + translation,
            self.origin + translation,
        ];

        let center = self.origin + far * 0.5 + translation * 0.5;
        let verts_2d: Vec<[f32; 2]> = corners
            .iter()
            .map(|&c| {
                let d = c - center;
                [d.dot(u_axis), d.dot(v_axis)]
            })
            .collect();

        RayStretch::Polygon(ConvexPolygon::with_axes(
            center, normal, u_axis, v_axis, verts_2d,
        ))
    }
}

// --- Ray–Sphere ---

impl Collides<Sphere> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        super::line_sphere_collides(self.origin, self.dir, self.rdv, sphere, T_MIN, T_MAX)
    }
}

impl Collides<Ray> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

// --- Ray–Capsule ---

impl Collides<Capsule> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        super::line_capsule_collides(self.origin, self.dir, capsule, T_MIN, T_MAX)
    }
}

impl Collides<Ray> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

// --- Ray–Cuboid ---

impl Collides<Cuboid> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        super::line_cuboid_collides(self.origin, self.dir, cuboid, T_MIN, T_MAX)
    }
}

impl Collides<Ray> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

// --- Ray–ConvexPolytope ---

impl Collides<ConvexPolytope> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        super::line_polytope_collides(
            self.origin,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            T_MIN,
            T_MAX,
        )
    }
}

impl Collides<Ray> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        super::line_polytope_collides(
            self.origin,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            T_MIN,
            T_MAX,
        )
    }
}

impl<const P: usize, const V: usize> Collides<Ray> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

// --- Ray–InfinitePlane ---

impl Collides<Plane> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        super::line_infinite_plane_collides(self.origin, self.dir, plane, T_MIN, T_MAX)
    }
}

impl Collides<Ray> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

// --- Ray–ConvexPolygon ---

impl Collides<ConvexPolygon> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ConvexPolygon) -> bool {
        polygon.parametric_line_dist_sq(self.origin, self.dir, T_MIN, T_MAX) <= 0.0
    }
}

impl Collides<Ray> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        ray.test::<BROADPHASE>(self)
    }
}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let o = self.origin;
        let d = self.dir;
        write!(
            f,
            "Ray(origin: [{}, {}, {}], dir: [{}, {}, {}])",
            o.x, o.y, o.z, d.x, d.y, d.z
        )
    }
}
