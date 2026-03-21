use glam::{DVec3, Vec3};

use inherent::inherent;


use crate::Stretchable;
use crate::{Bounded, Capsule, Collides, Cuboid, Scalable, Transformable};
use crate::wreck_assert;

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    pub const fn new(center: Vec3, radius: f32) -> Self {
        wreck_assert!(radius >= 0.0, "Sphere radius must be non-negative");
        Self { center, radius }
    }

    pub const fn new_d(center: DVec3, radius: f64) -> Self {
        wreck_assert!(radius >= 0.0, "Sphere radius must be non-negative");
        Self { center: Vec3 { x: center.x as f32, y: center.y as f32, z: center.z as f32 }, radius: radius as f32 }
    }
}

#[inherent]
impl Scalable for Sphere {
    pub fn scale(&mut self, factor: f32) {
        self.radius *= factor;
    }
}

#[inherent]
impl Transformable for Sphere {
    pub fn translate(&mut self, offset: Vec3) {
        self.center += offset;
    }

    pub fn rotate_mat(&mut self, _mat: glam::Mat3) {}

    pub fn rotate_quat(&mut self, _quat: glam::Quat) {}

    pub fn transform(&mut self, mat: glam::Affine3) {
        self.center = mat.transform_point3(self.center);
    }
}

#[inherent]
impl Bounded for Sphere {
    pub fn broadphase(&self) -> Sphere {
        *self
    }

    pub fn obb(&self) -> Cuboid {
        Cuboid::new(
            self.center,
            [Vec3::X, Vec3::Y, Vec3::Z],
            [self.radius, self.radius, self.radius],
        )
    }

    pub fn aabb(&self) -> Cuboid {
        Cuboid::new(
            self.center,
            [Vec3::X, Vec3::Y, Vec3::Z],
            [self.radius, self.radius, self.radius],
        )
    }
}

impl Collides<Sphere> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Sphere) -> bool {
        let d = self.center - other.center;
        let dist_sq = d.dot(d);
        let rs = self.radius + other.radius;
        dist_sq <= rs * rs
    }
}


#[derive(Debug, Clone, Copy)]
pub enum SphereStretch {
    NoStretch(Sphere),
    Stretch(Capsule)
}

impl Stretchable for Sphere {
    type Output = SphereStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let dir = translation.normalize_or_zero();
        if dir.length_squared() < f32::EPSILON {
            return SphereStretch::NoStretch(*self);
        }
        SphereStretch::Stretch(Capsule::new(self.center, self.center + translation, self.radius))
    }
}
