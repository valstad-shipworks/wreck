use glam::Vec3;

use inherent::inherent;

use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::line::LineSegment;
use crate::sphere::Sphere;
use crate::{Bounded, Collides, Scalable, Stretchable, Transformable};

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point(pub Vec3);

impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
}

#[inherent]
impl Scalable for Point {
    pub fn scale(&mut self, _factor: f32) {
        // 0 * factor is still 0
    }
}

#[inherent]
impl Transformable for Point {
    pub fn translate(&mut self, offset: glam::Vec3A) {
        self.0 = Vec3::from(glam::Vec3A::from(self.0) + offset);
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3A) {
        self.0 = Vec3::from(mat * glam::Vec3A::from(self.0));
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.0 = quat * self.0;
    }

    pub fn transform(&mut self, mat: glam::Affine3A) {
        self.0 = Vec3::from(mat.transform_point3a(glam::Vec3A::from(self.0)));
    }
}

#[inherent]
impl Bounded for Point {
    pub fn broadphase(&self) -> crate::Sphere {
        crate::Sphere::new(self.0, 0.0)
    }

    pub fn obb(&self) -> crate::Cuboid {
        crate::Cuboid::new(self.0, [Vec3::X, Vec3::Y, Vec3::Z], [0.0, 0.0, 0.0])
    }

    pub fn aabb(&self) -> crate::Cuboid {
        crate::Cuboid::new(self.0, [Vec3::X, Vec3::Y, Vec3::Z], [0.0, 0.0, 0.0])
    }
}

impl Stretchable for Point {
    type Output = LineSegment;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        LineSegment::new(self.0, self.0 + translation)
    }
}

// Point-Sphere
impl Collides<Sphere> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        let d = self.0 - sphere.center;
        d.dot(d) <= sphere.radius * sphere.radius
    }
}

impl Collides<Point> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        point.test::<BROADPHASE>(self)
    }
}

// Point-Capsule
impl Collides<Capsule> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        let closest = capsule.closest_point_to(self.0);
        let d = self.0 - closest;
        d.dot(d) <= capsule.radius * capsule.radius
    }
}

impl Collides<Point> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        point.test::<BROADPHASE>(self)
    }
}

// Point-Cuboid
impl Collides<Cuboid> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        cuboid.point_dist_sq(self.0) <= 0.0
    }
}

impl Collides<Point> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        point.test::<BROADPHASE>(self)
    }
}

// Point-ConvexPolytope

impl Collides<crate::ConvexPolytope> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &crate::ConvexPolytope) -> bool {
        crate::convex_polytope::refer::RefConvexPolytope::from_heap(polytope)
            .collides_point::<BROADPHASE>(self)
    }
}

impl Collides<Point> for crate::ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        point.test::<BROADPHASE>(self)
    }
}

// Point-Point
impl Collides<Point> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Point) -> bool {
        self.0 == other.0
    }
}

// Zero-thickness pairs — always false in 3D

impl Collides<crate::Plane> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &crate::Plane) -> bool {
        false
    }
}

impl Collides<Point> for crate::Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Point) -> bool {
        false
    }
}

impl Collides<crate::ConvexPolygon> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &crate::ConvexPolygon) -> bool {
        false
    }
}

impl Collides<Point> for crate::ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Point) -> bool {
        false
    }
}

impl Collides<crate::Line> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &crate::Line) -> bool {
        false
    }
}

impl Collides<Point> for crate::Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Point) -> bool {
        false
    }
}

impl Collides<crate::Ray> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &crate::Ray) -> bool {
        false
    }
}

impl Collides<Point> for crate::Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Point) -> bool {
        false
    }
}

impl Collides<LineSegment> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &LineSegment) -> bool {
        false
    }
}

impl Collides<Point> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &Point) -> bool {
        false
    }
}
