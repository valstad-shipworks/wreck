use std::fmt::Debug;

use crate::capsule::Capsule;
use crate::convex_polytope::heap::ConvexPolytope;
use crate::cuboid::Cuboid;
use crate::line::{Line, LineSegment, Ray};
use crate::plane::{ConvexPolygon, Plane};
use crate::point::Point;
use crate::pointcloud::{NoPcl, PointCloudMarker};
use crate::soa::{BroadCollection, SpheresSoA};
use crate::sphere::Sphere;
use crate::{Bounded, Collider, Scalable, Transformable};
use approx::{AbsDiffEq, RelativeEq};

impl AbsDiffEq for Sphere {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.center.abs_diff_eq(other.center, epsilon)
            && f32::abs_diff_eq(&self.radius, &other.radius, epsilon)
    }
}

impl RelativeEq for Sphere {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.center
            .relative_eq(&other.center, epsilon, max_relative)
            && f32::relative_eq(&self.radius, &other.radius, epsilon, max_relative)
    }
}

impl AbsDiffEq for Point {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.0.abs_diff_eq(other.0, epsilon)
    }
}

impl RelativeEq for Point {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.0.relative_eq(&other.0, epsilon, max_relative)
    }
}

impl AbsDiffEq for Capsule {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.p1.abs_diff_eq(other.p1, epsilon)
            && self.dir.abs_diff_eq(other.dir, epsilon)
            && f32::abs_diff_eq(&self.radius, &other.radius, epsilon)
    }
}

impl RelativeEq for Capsule {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.p1.relative_eq(&other.p1, epsilon, max_relative)
            && self.dir.relative_eq(&other.dir, epsilon, max_relative)
            && f32::relative_eq(&self.radius, &other.radius, epsilon, max_relative)
    }
}

// --- Cuboid ---
// Only compare geometric fields; axis_aligned is derived.

impl AbsDiffEq for Cuboid {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.center.abs_diff_eq(other.center, epsilon)
            && self.axes[0].abs_diff_eq(other.axes[0], epsilon)
            && self.axes[1].abs_diff_eq(other.axes[1], epsilon)
            && self.axes[2].abs_diff_eq(other.axes[2], epsilon)
            && f32::abs_diff_eq(&self.half_extents[0], &other.half_extents[0], epsilon)
            && f32::abs_diff_eq(&self.half_extents[1], &other.half_extents[1], epsilon)
            && f32::abs_diff_eq(&self.half_extents[2], &other.half_extents[2], epsilon)
    }
}

impl RelativeEq for Cuboid {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.center
            .relative_eq(&other.center, epsilon, max_relative)
            && self.axes[0].relative_eq(&other.axes[0], epsilon, max_relative)
            && self.axes[1].relative_eq(&other.axes[1], epsilon, max_relative)
            && self.axes[2].relative_eq(&other.axes[2], epsilon, max_relative)
            && f32::relative_eq(
                &self.half_extents[0],
                &other.half_extents[0],
                epsilon,
                max_relative,
            )
            && f32::relative_eq(
                &self.half_extents[1],
                &other.half_extents[1],
                epsilon,
                max_relative,
            )
            && f32::relative_eq(
                &self.half_extents[2],
                &other.half_extents[2],
                epsilon,
                max_relative,
            )
    }
}

// --- Plane ---

impl AbsDiffEq for Plane {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.normal.abs_diff_eq(other.normal, epsilon)
            && f32::abs_diff_eq(&self.d, &other.d, epsilon)
    }
}

impl RelativeEq for Plane {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.normal
            .relative_eq(&other.normal, epsilon, max_relative)
            && f32::relative_eq(&self.d, &other.d, epsilon, max_relative)
    }
}

// --- Line ---
// Only compare geometric fields; rdv is derived.

impl AbsDiffEq for Line {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.origin.abs_diff_eq(other.origin, epsilon) && self.dir.abs_diff_eq(other.dir, epsilon)
    }
}

impl RelativeEq for Line {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.origin
            .relative_eq(&other.origin, epsilon, max_relative)
            && self.dir.relative_eq(&other.dir, epsilon, max_relative)
    }
}

// --- Ray ---
// Only compare geometric fields; rdv is derived.

impl AbsDiffEq for Ray {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.origin.abs_diff_eq(other.origin, epsilon) && self.dir.abs_diff_eq(other.dir, epsilon)
    }
}

impl RelativeEq for Ray {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.origin
            .relative_eq(&other.origin, epsilon, max_relative)
            && self.dir.relative_eq(&other.dir, epsilon, max_relative)
    }
}

// --- LineSegment ---
// Only compare geometric fields; rdv is derived.

impl AbsDiffEq for LineSegment {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.p1.abs_diff_eq(other.p1, epsilon) && self.dir.abs_diff_eq(other.dir, epsilon)
    }
}

impl RelativeEq for LineSegment {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.p1.relative_eq(&other.p1, epsilon, max_relative)
            && self.dir.relative_eq(&other.dir, epsilon, max_relative)
    }
}

// --- ConvexPolygon ---
// Only compare defining geometric fields; precomputed caches are derived.

impl AbsDiffEq for ConvexPolygon {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.center.abs_diff_eq(other.center, epsilon)
            && self.normal.abs_diff_eq(other.normal, epsilon)
            && self.u_axis.abs_diff_eq(other.u_axis, epsilon)
            && self.v_axis.abs_diff_eq(other.v_axis, epsilon)
            && self.vertices_2d.len() == other.vertices_2d.len()
            && self
                .vertices_2d
                .iter()
                .zip(other.vertices_2d.iter())
                .all(|(a, b)| {
                    f32::abs_diff_eq(&a[0], &b[0], epsilon)
                        && f32::abs_diff_eq(&a[1], &b[1], epsilon)
                })
    }
}

impl RelativeEq for ConvexPolygon {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.center
            .relative_eq(&other.center, epsilon, max_relative)
            && self
                .normal
                .relative_eq(&other.normal, epsilon, max_relative)
            && self
                .u_axis
                .relative_eq(&other.u_axis, epsilon, max_relative)
            && self
                .v_axis
                .relative_eq(&other.v_axis, epsilon, max_relative)
            && self.vertices_2d.len() == other.vertices_2d.len()
            && self
                .vertices_2d
                .iter()
                .zip(other.vertices_2d.iter())
                .all(|(a, b)| {
                    f32::relative_eq(&a[0], &b[0], epsilon, max_relative)
                        && f32::relative_eq(&a[1], &b[1], epsilon, max_relative)
                })
    }
}

// --- ConvexPolytope ---

impl AbsDiffEq for ConvexPolytope {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.obb.abs_diff_eq(&other.obb, epsilon)
            && self.planes.len() == other.planes.len()
            && self.planes.iter().zip(other.planes.iter()).all(|(a, b)| {
                a.0.abs_diff_eq(b.0, epsilon) && f32::abs_diff_eq(&a.1, &b.1, epsilon)
            })
            && self.vertices.len() == other.vertices.len()
            && self
                .vertices
                .iter()
                .zip(other.vertices.iter())
                .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl RelativeEq for ConvexPolytope {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.obb.relative_eq(&other.obb, epsilon, max_relative)
            && self.planes.len() == other.planes.len()
            && self.planes.iter().zip(other.planes.iter()).all(|(a, b)| {
                a.0.relative_eq(&b.0, epsilon, max_relative)
                    && f32::relative_eq(&a.1, &b.1, epsilon, max_relative)
            })
            && self.vertices.len() == other.vertices.len()
            && self
                .vertices
                .iter()
                .zip(other.vertices.iter())
                .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }
}

// --- NoPcl ---

impl AbsDiffEq for NoPcl {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, _other: &Self, _epsilon: Self::Epsilon) -> bool {
        true
    }
}

impl RelativeEq for NoPcl {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        _other: &Self,
        _epsilon: Self::Epsilon,
        _max_relative: Self::Epsilon,
    ) -> bool {
        true
    }
}

// --- BroadCollection ---
// Compare only the items; the broadphase SoA is derived.

impl<T> PartialEq for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.items() == other.items()
    }
}

impl AbsDiffEq for SpheresSoA {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| a.abs_diff_eq(&b, epsilon))
    }
}

impl approx::RelativeEq for SpheresSoA {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| a.relative_eq(&b, epsilon, max_relative))
    }
}

impl<T> AbsDiffEq for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + AbsDiffEq<Epsilon = f32>,
{
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.items().len() == other.items().len()
            && self
                .items()
                .iter()
                .zip(other.items().iter())
                .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl<T> RelativeEq for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + RelativeEq<Epsilon = f32>,
{
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.items().len() == other.items().len()
            && self
                .items()
                .iter()
                .zip(other.items().iter())
                .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }
}

// --- Collider ---
// Compare all shape collections and the bounding sphere.

impl<PCL> PartialEq for Collider<PCL>
where
    PCL: PointCloudMarker + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bounding == other.bounding
            && self.planes == other.planes
            && self.lines == other.lines
            && self.rays == other.rays
            && self.spheres == other.spheres
            && self.capsules == other.capsules
            && self.cuboids == other.cuboids
            && self.polygons == other.polygons
            && self.polytopes == other.polytopes
            && self.points == other.points
            && self.segments == other.segments
            && self.pointclouds == other.pointclouds
    }
}

impl<PCL> AbsDiffEq for Collider<PCL>
where
    PCL: PointCloudMarker + AbsDiffEq<Epsilon = f32>,
{
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.bounding.abs_diff_eq(&other.bounding, epsilon)
            && slice_abs_diff_eq(&self.planes, &other.planes, epsilon)
            && slice_abs_diff_eq(&self.lines, &other.lines, epsilon)
            && slice_abs_diff_eq(&self.rays, &other.rays, epsilon)
            && self.spheres.abs_diff_eq(&other.spheres, epsilon)
            && self.capsules.abs_diff_eq(&other.capsules, epsilon)
            && self.cuboids.abs_diff_eq(&other.cuboids, epsilon)
            && self.polygons.abs_diff_eq(&other.polygons, epsilon)
            && self.polytopes.abs_diff_eq(&other.polytopes, epsilon)
            && self.points.abs_diff_eq(&other.points, epsilon)
            && self.segments.abs_diff_eq(&other.segments, epsilon)
            && self.pointclouds.abs_diff_eq(&other.pointclouds, epsilon)
    }
}

impl<PCL> RelativeEq for Collider<PCL>
where
    PCL: PointCloudMarker + RelativeEq<Epsilon = f32>,
{
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.bounding
            .relative_eq(&other.bounding, epsilon, max_relative)
            && slice_relative_eq(&self.planes, &other.planes, epsilon, max_relative)
            && slice_relative_eq(&self.lines, &other.lines, epsilon, max_relative)
            && slice_relative_eq(&self.rays, &other.rays, epsilon, max_relative)
            && self
                .spheres
                .relative_eq(&other.spheres, epsilon, max_relative)
            && self
                .capsules
                .relative_eq(&other.capsules, epsilon, max_relative)
            && self
                .cuboids
                .relative_eq(&other.cuboids, epsilon, max_relative)
            && self
                .polygons
                .relative_eq(&other.polygons, epsilon, max_relative)
            && self
                .polytopes
                .relative_eq(&other.polytopes, epsilon, max_relative)
            && self
                .points
                .relative_eq(&other.points, epsilon, max_relative)
            && self
                .segments
                .relative_eq(&other.segments, epsilon, max_relative)
            && self
                .pointclouds
                .relative_eq(&other.pointclouds, epsilon, max_relative)
    }
}

fn slice_abs_diff_eq<T: AbsDiffEq<Epsilon = f32>>(a: &[T], b: &[T], epsilon: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.abs_diff_eq(y, epsilon))
}

fn slice_relative_eq<T: RelativeEq<Epsilon = f32>>(
    a: &[T],
    b: &[T],
    epsilon: f32,
    max_relative: f32,
) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
}
