use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::{Collides, Scalable, Stretchable, Transformable};

#[derive(Debug, Clone, Copy)]
pub struct NoPcl;


macro_rules! impl_collides_nopcl {
    ($($ty:ty),* $(,)?) => {$(
        impl Collides<NoPcl> for $ty {
            #[inline]
            fn collides(&self, _other: &NoPcl) -> bool { false }
        }
        impl Collides<$ty> for NoPcl {
            #[inline]
            fn collides(&self, _other: &$ty) -> bool { false }
        }
    )*};
}

impl_collides_nopcl!(
    crate::Sphere,
    crate::Cuboid,
    crate::Capsule,
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
    fn collides(&self, _other: &NoPcl) -> bool {
        false
    }
}

impl Transformable for NoPcl {
    #[inline]
    fn translate(&mut self, _offset: glam::Vec3) {
        // No points to translate
    }
    #[inline]
    fn rotate_mat(&mut self, _mat: glam::Mat3) {
        // No points to rotate
    }
    #[inline]
    fn rotate_quat(&mut self, _quat: glam::Quat) {
        // No points to rotate
    }
    #[inline]
    fn transform(&mut self, _mat: glam::Affine3) {
        // No points to transform
    }
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