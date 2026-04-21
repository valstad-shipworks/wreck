//! Signed distance fields between convex shapes.
//!
//! The [`SignedDistance`] trait returns a signed scalar between two shapes:
//!
//! - **positive** — shapes are disjoint; the value is the Euclidean distance
//!   between the closest points,
//! - **zero** — the surfaces touch,
//! - **negative** — shapes interpenetrate; the magnitude is the penetration
//!   depth, i.e. the minimum translation along the separating normal required
//!   to make the shapes disjoint.
//!
//! The convention matches the Minkowski-difference definition of signed
//! distance, so two concentric shapes produce `-(sum of "radii")` (for
//! spheres) or `-(he_outer + he_inner)` (for nested cuboids) — not "slack to
//! nearest face".
//!
//! # Shape coverage
//!
//! Every shape in the crate participates. Pairs are grouped by implementation
//! strategy:
//!
//! - **Closed-form** (cheapest, O(1)):
//!   [`Sphere`]/[`Capsule`]/[`Cuboid`]/[`Plane`]/[`Point`] amongst each other,
//!   and [`Line`]/[`Ray`]/[`LineSegment`] against those same solids. The
//!   cuboid SDF is globally convex so its restriction to a line is sampled at
//!   a constant number of piecewise-linear breakpoints.
//! - **SAT** — [`Cuboid`] × [`Cuboid`] uses all 15 separating axes. This
//!   returns exact penetration depth when overlapping and a tight lower bound
//!   on Euclidean distance when separated. For exact separation use GJK.
//! - **GJK + EPA** — the generic fallback. Any pair of [`SupportFn`]-enabled
//!   shapes can be measured through [`gjk_epa_signed_distance`]. This powers
//!   all pairs involving [`Cylinder`], [`ConvexPolytope`], [`ConvexPolygon`],
//!   and their [`ArrayConvexPolytope`]/[`ArrayConvexPolygon`] variants.
//! - **Pointcloud** — [`Pointcloud`] is the minimum-reduction over all
//!   point-spheres; the [`Sphere`] SDF against the target is evaluated once
//!   per point. If the cloud carries an inverse transform, the target is
//!   mapped into the cloud's local frame first.
//!
//! # Curve-solid semantics
//!
//! [`Line`] and [`Ray`] are unbounded and zero-thickness. A line passing
//! through a half-space returns [`f32::NEG_INFINITY`] for the signed distance
//! to a [`Plane`], because no finite perpendicular translation separates
//! them. Against finite solids the sign behaves as expected (negative when
//! the curve enters the solid, magnitude = Minkowski-difference scalar).
//!
//! # Extending
//!
//! To add a custom convex shape to the SDF system:
//!
//! ```ignore
//! use wreck::sdf::SupportFn;
//! use glam::Vec3;
//!
//! struct MyShape { /* ... */ }
//!
//! impl SupportFn for MyShape {
//!     fn support(&self, direction: Vec3) -> Vec3 {
//!         // Return the point of self farthest along `direction`.
//!         todo!()
//!     }
//! }
//!
//! // Then wire up SDF pairs with the built-in shapes:
//! wreck::impl_gjk_sdf_pair!(MyShape, wreck::Sphere);
//! wreck::impl_gjk_sdf_pair!(MyShape);  // self-pair
//! ```
//!
//! [`SignedDistance`]: crate::SignedDistance
//! [`Sphere`]: crate::Sphere
//! [`Capsule`]: crate::Capsule
//! [`Cuboid`]: crate::Cuboid
//! [`Plane`]: crate::Plane
//! [`Point`]: crate::Point
//! [`Line`]: crate::Line
//! [`Ray`]: crate::Ray
//! [`LineSegment`]: crate::LineSegment
//! [`Cylinder`]: crate::Cylinder
//! [`ConvexPolytope`]: crate::ConvexPolytope
//! [`ConvexPolygon`]: crate::ConvexPolygon
//! [`ArrayConvexPolytope`]: crate::ArrayConvexPolytope
//! [`ArrayConvexPolygon`]: crate::ArrayConvexPolygon
//! [`Pointcloud`]: crate::Pointcloud

pub(crate) mod epa;
pub(crate) mod gjk;
pub(crate) mod support;

pub use support::SupportFn;

/// Generates symmetric `SignedDistance` impls for a pair of shape types by
/// delegating to `gjk_epa_signed_distance`. Use when no closed-form is
/// available and both types already implement `SupportFn`.
#[macro_export]
macro_rules! impl_gjk_sdf_pair {
    ($a:ty, $b:ty) => {
        impl $crate::SignedDistance<$b> for $a {
            #[inline]
            fn signed_distance(&self, other: &$b) -> f32 {
                $crate::sdf::gjk_epa_signed_distance(self, other)
            }
        }
        impl $crate::SignedDistance<$a> for $b {
            #[inline]
            fn signed_distance(&self, other: &$a) -> f32 {
                $crate::sdf::gjk_epa_signed_distance(self, other)
            }
        }
    };
    ($a:ty) => {
        impl $crate::SignedDistance<$a> for $a {
            #[inline]
            fn signed_distance(&self, other: &$a) -> f32 {
                $crate::sdf::gjk_epa_signed_distance(self, other)
            }
        }
    };
}

/// Generic signed distance between two convex shapes via GJK (for disjoint
/// cases) and EPA (for penetrating cases). Shapes need only expose a support
/// function — no interior representation is required.
///
/// This is the fallback used by `SignedDistance` impls for pairs without a
/// closed-form. It is also exposed publicly so callers with their own convex
/// shapes can participate in the same SDF system by implementing `SupportFn`.
#[inline]
pub fn gjk_epa_signed_distance<A, B>(a: &A, b: &B) -> f32
where
    A: SupportFn + ?Sized,
    B: SupportFn + ?Sized,
{
    match gjk::gjk_distance(a, b) {
        gjk::GjkResult::Separated(d) => d,
        gjk::GjkResult::Penetrating(seed) => -epa::epa_penetration_depth(a, b, seed),
    }
}

use crate::{
    ArrayConvexPolygon, ArrayConvexPolytope, Capsule, ConvexPolygon, ConvexPolytope, Cuboid,
    Cylinder, LineSegment, Point, Sphere,
};

// ---------------------------------------------------------------------------
// GJK+EPA-backed SDF pairs for convex shapes that do not have a closed-form
// implementation elsewhere. Every type involved implements `SupportFn`.
// ---------------------------------------------------------------------------

impl_gjk_sdf_pair!(Cylinder);
impl_gjk_sdf_pair!(Cylinder, Sphere);
impl_gjk_sdf_pair!(Cylinder, Cuboid);
impl_gjk_sdf_pair!(Cylinder, Capsule);
impl_gjk_sdf_pair!(Cylinder, Point);
impl_gjk_sdf_pair!(Cylinder, LineSegment);

impl_gjk_sdf_pair!(ConvexPolytope);
impl_gjk_sdf_pair!(ConvexPolytope, Sphere);
impl_gjk_sdf_pair!(ConvexPolytope, Cuboid);
impl_gjk_sdf_pair!(ConvexPolytope, Capsule);
impl_gjk_sdf_pair!(ConvexPolytope, Cylinder);
impl_gjk_sdf_pair!(ConvexPolytope, Point);
impl_gjk_sdf_pair!(ConvexPolytope, LineSegment);

impl_gjk_sdf_pair!(ConvexPolygon);
impl_gjk_sdf_pair!(ConvexPolygon, Sphere);
impl_gjk_sdf_pair!(ConvexPolygon, Cuboid);
impl_gjk_sdf_pair!(ConvexPolygon, Capsule);
impl_gjk_sdf_pair!(ConvexPolygon, Cylinder);
impl_gjk_sdf_pair!(ConvexPolygon, Point);
impl_gjk_sdf_pair!(ConvexPolygon, LineSegment);
impl_gjk_sdf_pair!(ConvexPolygon, ConvexPolytope);

// LineSegment × LineSegment has a closed-form Ericson impl in src/line/mod.rs.
impl_gjk_sdf_pair!(LineSegment, Sphere);
impl_gjk_sdf_pair!(LineSegment, Cuboid);
impl_gjk_sdf_pair!(LineSegment, Capsule);
impl_gjk_sdf_pair!(LineSegment, Point);

// Generic impls for the Array variants (const generics preclude macro pairs).

impl<const P: usize, const V: usize> SignedDistance<Sphere> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn signed_distance(&self, other: &Sphere) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ArrayConvexPolytope<P, V>> for Sphere {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolytope<P, V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<Cuboid> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn signed_distance(&self, other: &Cuboid) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ArrayConvexPolytope<P, V>> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolytope<P, V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<Capsule> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn signed_distance(&self, other: &Capsule) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ArrayConvexPolytope<P, V>> for Capsule {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolytope<P, V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<Cylinder> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn signed_distance(&self, other: &Cylinder) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ArrayConvexPolytope<P, V>> for Cylinder {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolytope<P, V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ArrayConvexPolytope<P, V>>
    for ArrayConvexPolytope<P, V>
{
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolytope<P, V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ConvexPolytope> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn signed_distance(&self, other: &ConvexPolytope) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const P: usize, const V: usize> SignedDistance<ArrayConvexPolytope<P, V>> for ConvexPolytope {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolytope<P, V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}

impl<const V: usize> SignedDistance<Sphere> for ArrayConvexPolygon<V> {
    #[inline]
    fn signed_distance(&self, other: &Sphere) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const V: usize> SignedDistance<ArrayConvexPolygon<V>> for Sphere {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolygon<V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const V: usize> SignedDistance<Cuboid> for ArrayConvexPolygon<V> {
    #[inline]
    fn signed_distance(&self, other: &Cuboid) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const V: usize> SignedDistance<ArrayConvexPolygon<V>> for Cuboid {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolygon<V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const V: usize> SignedDistance<Capsule> for ArrayConvexPolygon<V> {
    #[inline]
    fn signed_distance(&self, other: &Capsule) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const V: usize> SignedDistance<ArrayConvexPolygon<V>> for Capsule {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolygon<V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}
impl<const V: usize> SignedDistance<ArrayConvexPolygon<V>> for ArrayConvexPolygon<V> {
    #[inline]
    fn signed_distance(&self, other: &ArrayConvexPolygon<V>) -> f32 {
        gjk_epa_signed_distance(self, other)
    }
}

/// Trait for computing the signed distance between two shapes.
///
/// The returned scalar is:
/// - **Positive** when the shapes are separated; the magnitude is the Euclidean
///   distance between the closest points on each shape.
/// - **Zero** when the surfaces are exactly touching.
/// - **Negative** when the shapes interpenetrate; the magnitude is the penetration
///   depth — the minimum translation along the separating normal required to
///   bring the shapes apart.
///
/// Semantics by shape category:
/// - Solid × Solid: signed in both directions.
/// - 1D curve (Line / Ray / LineSegment) × Solid: signed; negative when the
///   curve passes through the solid.
/// - 1D curve × 1D curve: always non-negative (1D curves have no interior, so
///   the signed variant is degenerate — the result is the unsigned distance).
/// - Pointcloud × X: each point is treated as a sphere of radius `point_radius`
///   and the result is the minimum over the cloud. Currently a linear scan;
///   a spatial-index prune is planned.
///
/// Implementations are symmetric: `a.signed_distance(&b) == b.signed_distance(&a)`.
pub trait SignedDistance<T> {
    /// Returns the signed distance between `self` and `other`.
    ///
    /// See the trait documentation for sign conventions.
    fn signed_distance(&self, other: &T) -> f32;
}
