use glam::Vec3;

use inherent::inherent;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

/// An infinite plane defined as a half-space: `normal · x <= d`.
/// The plane surface is at `normal · x = d`; everything on the side
/// opposite to the normal is considered solid.
///
/// `normal` must be unit-length.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    pub const fn new(normal: Vec3, d: f32) -> Self {
        wreck_assert!(
            crate::dot(normal, normal) > f32::EPSILON,
            "Plane normal must be non-zero"
        );
        Self { normal, d }
    }

    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        wreck_assert!(
            crate::dot(normal, normal) > f32::EPSILON,
            "Plane normal must be non-zero"
        );
        Self {
            normal,
            d: normal.dot(point),
        }
    }
}

#[inherent]
impl Scalable for Plane {
    pub fn scale(&mut self, factor: f32) {
        self.d *= factor;
    }
}

#[inherent]
impl Transformable for Plane {
    pub fn translate(&mut self, offset: Vec3) {
        self.d += self.normal.dot(offset);
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.normal = mat * self.normal;
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.normal = quat * self.normal;
    }

    pub fn transform(&mut self, mat: glam::Affine3) {
        self.normal = mat.matrix3 * self.normal;
        self.d += self.normal.dot(mat.translation);
    }
}

impl Stretchable for Plane {
    type Output = Plane;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        // Minkowski sum of half-space {x : n·x <= d} with segment [0, t]:
        // result is {x : n·x <= d + max(0, n·t)}.
        let proj = self.normal.dot(translation);
        Plane::new(self.normal, self.d + proj.max(0.0))
    }
}

// ---------------------------------------------------------------------------
// Plane – Sphere
// ---------------------------------------------------------------------------

#[inline]
fn plane_sphere_collides(plane: &Plane, sphere: &Sphere) -> bool {
    plane.normal.dot(sphere.center) - plane.d <= sphere.radius
}

impl Collides<Sphere> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        plane_sphere_collides(self, sphere)
    }
}

impl Collides<Plane> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        plane_sphere_collides(plane, self)
    }
}

// ---------------------------------------------------------------------------
// Plane – Capsule
// ---------------------------------------------------------------------------

#[inline]
fn plane_capsule_collides(plane: &Plane, capsule: &Capsule) -> bool {
    let p2 = capsule.p2();
    let proj1 = plane.normal.dot(capsule.p1);
    let proj2 = plane.normal.dot(p2);
    proj1.min(proj2) - plane.d <= capsule.radius
}

impl Collides<Capsule> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        plane_capsule_collides(self, capsule)
    }
}

impl Collides<Plane> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        plane_capsule_collides(plane, self)
    }
}

// ---------------------------------------------------------------------------
// Plane – Cuboid
// ---------------------------------------------------------------------------

#[inline]
fn plane_cuboid_collides(plane: &Plane, cuboid: &Cuboid) -> bool {
    let center_proj = plane.normal.dot(cuboid.center);
    let extent_proj = plane.normal.dot(cuboid.axes[0]).abs() * cuboid.half_extents[0]
        + plane.normal.dot(cuboid.axes[1]).abs() * cuboid.half_extents[1]
        + plane.normal.dot(cuboid.axes[2]).abs() * cuboid.half_extents[2];
    center_proj - extent_proj <= plane.d
}

impl Collides<Cuboid> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        plane_cuboid_collides(self, cuboid)
    }
}

impl Collides<Plane> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        plane_cuboid_collides(plane, self)
    }
}

// ---------------------------------------------------------------------------
// Plane – ConvexPolytope (Heap & Array)
// ---------------------------------------------------------------------------

#[inline]
fn plane_polytope_collides_ref<const BROADPHASE: bool>(
    plane: &Plane,
    vertices: &[Vec3],
    obb: &Cuboid,
) -> bool {
    if BROADPHASE && !plane_cuboid_collides(plane, obb) {
        return false;
    }
    crate::convex_polytope::min_projection(vertices, plane.normal) <= plane.d
}

impl Collides<ConvexPolytope> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        plane_polytope_collides_ref::<BROADPHASE>(self, &polytope.vertices, &polytope.obb)
    }
}

impl Collides<Plane> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        plane_polytope_collides_ref::<BROADPHASE>(plane, &self.vertices, &self.obb)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        plane_polytope_collides_ref::<BROADPHASE>(self, &polytope.vertices, &polytope.obb)
    }
}

impl<const P: usize, const V: usize> Collides<Plane> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        plane_polytope_collides_ref::<BROADPHASE>(plane, &self.vertices, &self.obb)
    }
}

// ---------------------------------------------------------------------------
// Plane – Plane
// ---------------------------------------------------------------------------

impl Collides<Plane> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Plane) -> bool {
        // Two half-spaces always overlap unless their normals are antiparallel
        // and the planes are separated (d1 + d2 < 0).
        let dot = self.normal.dot(other.normal);
        if dot > -1.0 + 1e-6 {
            return true;
        }
        // Antiparallel: half-spaces overlap iff d1 + d2 >= 0
        self.d + other.d >= 0.0
    }
}
