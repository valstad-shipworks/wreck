use glam::{DVec3, Vec3};
use wide::{f32x8, CmpLe};

use inherent::inherent;


use crate::Stretchable;
use crate::{Capsule, Collides, Scalable, Transformable};
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

impl Collides<Sphere> for Sphere {
    #[inline]
    fn collides(&self, other: &Sphere) -> bool {
        let d = self.center - other.center;
        let dist_sq = d.dot(d);
        let rs = self.radius + other.radius;
        dist_sq <= rs * rs
    }

    fn collides_many(&self, others: &[Sphere]) -> bool {
        let cx = f32x8::splat(self.center.x);
        let cy = f32x8::splat(self.center.y);
        let cz = f32x8::splat(self.center.z);
        let sr = f32x8::splat(self.radius);

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut ox = [0.0f32; 8];
            let mut oy = [0.0f32; 8];
            let mut oz = [0.0f32; 8];
            let mut or = [0.0f32; 8];
            for (i, s) in chunk.iter().enumerate() {
                ox[i] = s.center.x;
                oy[i] = s.center.y;
                oz[i] = s.center.z;
                or[i] = s.radius;
            }
            let dx = cx - f32x8::new(ox);
            let dy = cy - f32x8::new(oy);
            let dz = cz - f32x8::new(oz);
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let rs = sr + f32x8::new(or);
            let rs_sq = rs * rs;
            if dist_sq.simd_le(rs_sq).any() {
                return true;
            }
        }

        remainder.iter().any(|s| self.collides(s))
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
