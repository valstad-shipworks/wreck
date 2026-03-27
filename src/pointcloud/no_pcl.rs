use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::{Bounded, Collides, Cuboid, Scalable, Sphere, Stretchable, Transformable};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoPcl;

macro_rules! impl_collides_nopcl {
    ($($ty:ty),* $(,)?) => {$(
        impl Collides<NoPcl> for $ty {
            #[inline]
            fn test<const BROADPHASE: bool>(&self, _other: &NoPcl) -> bool { false }
        }
        impl Collides<$ty> for NoPcl {
            #[inline]
            fn test<const BROADPHASE: bool>(&self, _other: &$ty) -> bool { false }
        }
    )*};
}

impl_collides_nopcl!(
    crate::Sphere,
    crate::Cuboid,
    crate::Capsule,
    crate::Cylinder,
    crate::Point,
    crate::Plane,
    crate::ConvexPolygon,
    crate::ConvexPolytope,
    crate::Line,
    crate::Ray,
    crate::LineSegment,
    crate::Pointcloud
);

impl<const P: usize, const V: usize> Collides<NoPcl> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, _other: &NoPcl) -> bool {
        false
    }
}

impl Bounded for NoPcl {
    fn broadphase(&self) -> Sphere {
        Sphere::new(glam::Vec3::ZERO, 0.0)
    }
    fn obb(&self) -> Cuboid {
        Cuboid::from_aabb(glam::Vec3::ZERO, glam::Vec3::ZERO)
    }
    fn aabb(&self) -> Cuboid {
        Cuboid::from_aabb(glam::Vec3::ZERO, glam::Vec3::ZERO)
    }
}

impl Transformable for NoPcl {
    #[inline]
    fn translate(&mut self, _offset: glam::Vec3A) {}
    #[inline]
    fn rotate_mat(&mut self, _mat: glam::Mat3A) {}
    #[inline]
    fn rotate_quat(&mut self, _quat: glam::Quat) {}
    #[inline]
    fn transform(&mut self, _mat: glam::Affine3A) {}
}

impl Scalable for NoPcl {
    #[inline]
    fn scale(&mut self, _factor: f32) {}
}

impl Stretchable for NoPcl {
    type Output = NoPcl;

    #[inline]
    fn stretch(&self, _translation: glam::Vec3) -> Self::Output {
        NoPcl
    }
}
