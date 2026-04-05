use core::fmt;

use glam::Vec3;

use inherent::inherent;

use super::refer::RefConvexPolytope;
use crate::ConvexPolytope;
use crate::cuboid::Cuboid;
use crate::sphere::Sphere;
use crate::{Collides, Scalable, Stretchable, Transformable};

/// A convex polytope defined by half-planes and vertices, using const generics
/// so it can be constructed and stored in `const` / `static` contexts.
///
/// `P` is the number of half-planes, `V` is the number of vertices.
///
/// This is mostly meant for usage with codegen where you statically define obstacles based on
/// some sort of file at build time, and want to be able to use them in `const` contexts.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArrayConvexPolytope<const P: usize, const V: usize> {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array_as_seq"))]
    pub planes: [(Vec3, f32); P],
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array_as_seq"))]
    pub vertices: [Vec3; V],
    pub obb: Cuboid,
}

impl<const P: usize, const V: usize> ArrayConvexPolytope<P, V> {
    /// Create a new const convex polytope. This is a `const fn` so it can be
    /// used in `const` and `static` initializers.
    pub const fn new(planes: [(Vec3, f32); P], vertices: [Vec3; V], obb: Cuboid) -> Self {
        let mut i = 0;
        while i < P {
            let (n, d) = planes[i];
            debug_assert!(
                crate::dot(n, n) >= f32::EPSILON,
                "Plane normal cannot be zero"
            );
            debug_assert!(d >= 0.0, "Plane distance cannot be negative");
            i += 1;
        }
        i = 0;
        while i < V {
            let v = vertices[i];
            debug_assert!(
                !v.x.is_nan() && !v.y.is_nan() && !v.z.is_nan(),
                "Vertex cannot contain NaN"
            );
            debug_assert!(obb.contains_point(v), "Vertex cannot be outside the OBB");
            i += 1;
        }
        Self {
            planes,
            vertices,
            obb,
        }
    }

    #[inline]
    fn as_ref(&self) -> RefConvexPolytope<'_> {
        RefConvexPolytope::from_array(self)
    }
}

#[inherent]
impl<const P: usize, const V: usize> Scalable for ArrayConvexPolytope<P, V> {
    pub fn scale(&mut self, factor: f32) {
        for (_, d) in &mut self.planes {
            *d *= factor;
        }
        for v in &mut self.vertices {
            *v *= factor;
        }
        self.obb.scale(factor);
    }
}

#[inherent]
impl<const P: usize, const V: usize> Transformable for ArrayConvexPolytope<P, V> {
    pub fn translate(&mut self, offset: glam::Vec3A) {
        let off = Vec3::from(offset);
        for (n, d) in &mut self.planes {
            *d += n.dot(off);
        }
        for v in &mut self.vertices {
            *v += off;
        }
        self.obb.translate(offset);
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3A) {
        for (n, _) in &mut self.planes {
            *n = Vec3::from(mat * glam::Vec3A::from(*n));
        }
        for v in &mut self.vertices {
            *v = Vec3::from(mat * glam::Vec3A::from(*v));
        }
        self.obb.rotate_mat(mat);
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        for (n, _) in &mut self.planes {
            *n = quat * *n;
        }
        for v in &mut self.vertices {
            *v = quat * *v;
        }
        self.obb.rotate_quat(quat);
    }

    pub fn transform(&mut self, mat: glam::Affine3A) {
        let rot = mat.matrix3;
        let trans = Vec3::from(mat.translation);
        for (n, d) in &mut self.planes {
            *n = Vec3::from(rot * glam::Vec3A::from(*n));
            *d += n.dot(trans);
        }
        for v in &mut self.vertices {
            *v = Vec3::from(mat.transform_point3a(glam::Vec3A::from(*v)));
        }
        self.obb.transform(mat);
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolytope<P, V>) -> bool {
        other.as_ref().collides_sphere::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<Sphere> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Sphere) -> bool {
        self.as_ref().collides_sphere::<BROADPHASE>(other)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolytope<P, V>) -> bool {
        other.as_ref().collides_cuboid::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<Cuboid> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Cuboid) -> bool {
        self.as_ref().collides_cuboid::<BROADPHASE>(other)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>>
    for crate::capsule::Capsule
{
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolytope<P, V>) -> bool {
        other.as_ref().collides_capsule::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<crate::capsule::Capsule>
    for ArrayConvexPolytope<P, V>
{
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &crate::capsule::Capsule) -> bool {
        self.as_ref().collides_capsule::<BROADPHASE>(other)
    }
}

// ---------------------------------------------------------------------------
// Collision: ArrayConvexPolytope vs ArrayConvexPolytope
// ---------------------------------------------------------------------------

impl<const P: usize, const V: usize, const P2: usize, const V2: usize>
    Collides<ArrayConvexPolytope<P2, V2>> for ArrayConvexPolytope<P, V>
{
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolytope<P2, V2>) -> bool {
        self.as_ref()
            .collides_polytope::<BROADPHASE>(&other.as_ref())
    }
}

impl<const P: usize, const V: usize> Stretchable for ArrayConvexPolytope<P, V> {
    type Output = ConvexPolytope;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let heap = ConvexPolytope::with_obb(self.planes.to_vec(), self.vertices.to_vec(), self.obb);
        heap.stretch(translation)
    }
}

impl<const P: usize, const V: usize> fmt::Display for ArrayConvexPolytope<P, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArrayConvexPolytope(planes: {}, vertices: {})",
            P, V
        )
    }
}
