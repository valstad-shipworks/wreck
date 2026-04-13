#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("at least one of the `std` or `libm` features must be enabled");
#[macro_use]
extern crate alloc;
use alloc::vec::Vec;

pub(crate) mod capsule;
pub(crate) mod cuboid;
pub(crate) mod cylinder;
pub(crate) mod point;
pub(crate) mod sphere;

pub(crate) mod convex_polytope;
pub(crate) mod line;
pub(crate) mod plane;
pub(crate) mod pointcloud;

mod util;
pub(crate) use util::*;

#[cfg(feature = "approx")]
mod approx;

#[cfg(feature = "serde")]
pub(crate) mod serde_arrays;

#[cfg(feature = "quote")]
mod quote;

pub mod soa;

pub use capsule::Capsule;
pub use convex_polytope::array::ArrayConvexPolytope;
pub use convex_polytope::heap::ConvexPolytope;
pub use cuboid::Cuboid;
pub use cylinder::Cylinder;
pub use line::Line;
pub use line::LineSegment;
pub use line::Ray;
pub use plane::ArrayConvexPolygon;
pub use plane::ConvexPolygon;
pub use plane::Plane;
pub use point::Point;
pub use pointcloud::Pointcloud;
pub use sphere::Sphere;

pub use crate::pointcloud::NoPcl;
use crate::pointcloud::PointCloudMarker;

pub mod stretched {
    pub use crate::sphere::SphereStretch;
    pub use crate::capsule::CapsuleStretch;
    pub use crate::cuboid::CuboidStretch;
    pub use crate::cylinder::CylinderStretch;
    pub use crate::line::LineSegmentStretch;
    pub use crate::line::LineStretch;
    pub use crate::line::RayStretch;
    pub use crate::plane::ConvexPolygonStretch;
}

pub trait Scalable: Sized + Clone {
    /// Scales the shape by the given factor, in-place.
    ///
    /// For shapes with zero volume (e.g. Point, Line, InfinitePlane), this can be a no-op,
    /// since scaling a point by any factor still results in a point.
    fn scale(&mut self, factor: f32);
    /// Returns a scaled copy of the shape.
    ///
    /// For shapes with zero volume (e.g. Point, Line, InfinitePlane), this can be a no-op,
    /// since scaling a point by any factor still results in a point.
    fn scaled(&self, factor: f32) -> Self {
        let mut cloned = self.clone();
        cloned.scale(factor);
        cloned
    }
    #[inline]
    fn scale_d(&mut self, factor: f64) {
        self.scale(factor as f32);
    }
    #[inline]
    fn scaled_d(&self, factor: f64) -> Self {
        self.scaled(factor as f32)
    }
}

pub trait Transformable: Sized + Clone {
    fn translate(&mut self, offset: glam::Vec3A);
    #[inline]
    fn translated(&self, offset: glam::Vec3A) -> Self {
        let mut cloned = self.clone();
        cloned.translate(offset);
        cloned
    }
    #[inline]
    fn translate_d(&mut self, offset: glam::DVec3) {
        self.translate(glam::Vec3A::from(offset.as_vec3()));
    }
    #[inline]
    fn translated_d(&self, offset: glam::DVec3) -> Self {
        self.translated(glam::Vec3A::from(offset.as_vec3()))
    }
    fn rotate_mat(&mut self, mat: glam::Mat3A);
    #[inline]
    fn rotated_mat(&self, mat: glam::Mat3A) -> Self {
        let mut cloned = self.clone();
        cloned.rotate_mat(mat);
        cloned
    }
    #[inline]
    fn rotate_mat_d(&mut self, mat: glam::DMat3) {
        self.rotate_mat(glam::Mat3A::from(mat.as_mat3()));
    }
    #[inline]
    fn rotated_mat_d(&self, mat: glam::DMat3) -> Self {
        self.rotated_mat(glam::Mat3A::from(mat.as_mat3()))
    }
    fn rotate_quat(&mut self, quat: glam::Quat);
    #[inline]
    fn rotated_quat(&self, quat: glam::Quat) -> Self {
        let mut cloned = self.clone();
        cloned.rotate_quat(quat);
        cloned
    }
    #[inline]
    fn rotate_quat_d(&mut self, quat: glam::DQuat) {
        self.rotate_quat(quat.as_quat());
    }
    #[inline]
    fn rotated_quat_d(&self, quat: glam::DQuat) -> Self {
        self.rotated_quat(quat.as_quat())
    }
    fn transform(&mut self, mat: glam::Affine3A);
    #[inline]
    fn transformed(&self, mat: glam::Affine3A) -> Self {
        let mut cloned = self.clone();
        cloned.transform(mat);
        cloned
    }
    #[inline]
    fn transform_d(&mut self, mat: glam::DAffine3) {
        self.transform(glam::Affine3A::from(mat.as_affine3()));
    }
    #[inline]
    fn transformed_d(&self, mat: glam::DAffine3) -> Self {
        self.transformed(glam::Affine3A::from(mat.as_affine3()))
    }
}

pub trait Stretchable: Sized + Clone {
    type Output;

    #[must_use]
    fn stretch(&self, translation: glam::Vec3) -> Self::Output;
    #[inline]
    #[must_use]
    fn stretch_d(&self, translation: glam::DVec3) -> Self::Output {
        self.stretch(translation.as_vec3())
    }
}

pub trait Collides<T>: Sized + Clone
where
    T: Sized + Clone,
{
    /// Collision test with compile-time control over broadphase.
    /// When `BROADPHASE` is true, broadphase checks run before narrowphase.
    /// When false, only the narrowphase runs.
    #[must_use]
    #[doc(hidden)]
    fn test<const BROADPHASE: bool>(&self, other: &T) -> bool;

    /// Collision test (broadphase + narrowphase).
    #[must_use]
    fn collides(&self, other: &T) -> bool {
        self.test::<true>(other)
    }
}

pub trait Bounded: Sized + Clone {
    #[must_use]
    fn broadphase(&self) -> Sphere;
    #[must_use]
    fn obb(&self) -> Cuboid;
    #[must_use]
    fn aabb(&self) -> Cuboid;
}

pub trait CollidesWithEverything<T>:
    Sized
    + Clone
    + Bounded
    + Collides<T>
    + Collides<Capsule>
    + Collides<Cuboid>
    + Collides<Cylinder>
    + Collides<Plane>
    + Collides<ConvexPolygon>
    + Collides<ConvexPolytope>
    + Collides<Point>
    + Collides<Sphere>
    + Collides<Line>
    + Collides<Ray>
    + Collides<LineSegment>
where
    T: PointCloudMarker,
{
}

impl<T, U> CollidesWithEverything<U> for T
where
    T: Sized
        + Clone
        + Bounded
        + Collides<U>
        + Collides<Capsule>
        + Collides<Cuboid>
        + Collides<Cylinder>
        + Collides<Plane>
        + Collides<ConvexPolygon>
        + Collides<ConvexPolytope>
        + Collides<Point>
        + Collides<Sphere>
        + Collides<Line>
        + Collides<Ray>
        + Collides<LineSegment>,
    U: PointCloudMarker,
{
}

pub trait ColliderComponent<PCL: PointCloudMarker = Pointcloud>: Sized + Clone {
    fn add_to_shapes(self, shapes: &mut Collider<PCL>);
}

macro_rules! impl_shape_for_bounded {
    ($ty:ty, $field:ident, $mask:ident) => {
        impl<PCL: PointCloudMarker> ColliderComponent<PCL> for $ty {
            fn add_to_shapes(self, c: &mut Collider<PCL>) {
                c.expand_bounding(&self.broadphase());
                c.$field.push(self);
                c.mask |= Collider::<PCL>::$mask;
            }
        }
    };
}

macro_rules! impl_shape_for_unbounded {
    ($ty:ty, $field:ident, $mask:ident) => {
        impl<PCL: PointCloudMarker> ColliderComponent<PCL> for $ty {
            fn add_to_shapes(self, c: &mut Collider<PCL>) {
                c.bounding.radius = f32::INFINITY;
                c.$field.push(self);
                c.mask |= Collider::<PCL>::$mask;
            }
        }
    };
}

impl<PCL: PointCloudMarker> ColliderComponent<PCL> for Sphere {
    fn add_to_shapes(self, c: &mut Collider<PCL>) {
        c.expand_bounding(&self.broadphase());
        c.spheres.push(self);
        c.mask |= Collider::<PCL>::MASK_SPHERES;
    }
}
impl_shape_for_bounded!(Capsule, capsules, MASK_CAPSULES);
impl_shape_for_bounded!(Cuboid, cuboids, MASK_CUBOIDS);
impl_shape_for_bounded!(Cylinder, cylinders, MASK_CYLINDERS);
impl_shape_for_bounded!(ConvexPolytope, polytopes, MASK_POLYTOPES);
impl_shape_for_bounded!(ConvexPolygon, polygons, MASK_POLYGONS);
impl_shape_for_bounded!(Point, points, MASK_POINTS);
impl_shape_for_bounded!(LineSegment, segments, MASK_SEGMENTS);
impl_shape_for_unbounded!(Plane, planes, MASK_PLANES);
impl_shape_for_unbounded!(Line, lines, MASK_LINES);
impl_shape_for_unbounded!(Ray, rays, MASK_RAYS);

impl ColliderComponent<Pointcloud> for Pointcloud {
    fn add_to_shapes(self, c: &mut Collider<Pointcloud>) {
        c.expand_bounding(&self.broadphase());
        c.pointclouds.push(self);
        c.mask |= Collider::<Pointcloud>::MASK_POINTCLOUDS;
    }
}

impl<const P: usize, const V: usize, PCL: PointCloudMarker> ColliderComponent<PCL>
    for ArrayConvexPolytope<P, V>
{
    fn add_to_shapes(self, c: &mut Collider<PCL>) {
        let poly = ConvexPolytope::from(self);
        c.expand_bounding(&poly.broadphase());
        c.polytopes.push(poly);
        c.mask |= Collider::<PCL>::MASK_POLYTOPES;
    }
}

#[derive(Debug, Clone)]
pub struct Collider<PCL: PointCloudMarker = Pointcloud> {
    capsules: soa::BroadCollection<Capsule>,
    cuboids: soa::BroadCollection<Cuboid>,
    cylinders: soa::BroadCollection<Cylinder>,
    planes: Vec<Plane>,
    polygons: soa::BroadCollection<ConvexPolygon>,
    polytopes: soa::BroadCollection<ConvexPolytope>,
    points: soa::BroadCollection<Point>,
    spheres: soa::SpheresSoA,
    lines: Vec<Line>,
    rays: Vec<Ray>,
    segments: soa::BroadCollection<LineSegment>,
    pointclouds: soa::BroadCollection<PCL>,
    bounding: Sphere,
    mask: u16,
}

impl<PCL: PointCloudMarker> Collider<PCL> {
    pub const MASK_CAPSULES: u16 = 1 << 0;
    pub const MASK_CUBOIDS: u16 = 1 << 1;
    pub const MASK_CYLINDERS: u16 = 1 << 2;
    pub const MASK_PLANES: u16 = 1 << 3;
    pub const MASK_POLYGONS: u16 = 1 << 4;
    pub const MASK_POLYTOPES: u16 = 1 << 5;
    pub const MASK_POINTS: u16 = 1 << 6;
    pub const MASK_SPHERES: u16 = 1 << 7;
    pub const MASK_LINES: u16 = 1 << 8;
    pub const MASK_RAYS: u16 = 1 << 9;
    pub const MASK_SEGMENTS: u16 = 1 << 10;
    pub const MASK_POINTCLOUDS: u16 = 1 << 11;

    pub fn mask(&self) -> u16 {
        self.mask
    }

    fn recompute_mask(&mut self) {
        let mut m = 0u16;
        if !self.capsules.is_empty() { m |= Self::MASK_CAPSULES; }
        if !self.cuboids.is_empty() { m |= Self::MASK_CUBOIDS; }
        if !self.cylinders.is_empty() { m |= Self::MASK_CYLINDERS; }
        if !self.planes.is_empty() { m |= Self::MASK_PLANES; }
        if !self.polygons.is_empty() { m |= Self::MASK_POLYGONS; }
        if !self.polytopes.is_empty() { m |= Self::MASK_POLYTOPES; }
        if !self.points.is_empty() { m |= Self::MASK_POINTS; }
        if !self.spheres.is_empty() { m |= Self::MASK_SPHERES; }
        if !self.lines.is_empty() { m |= Self::MASK_LINES; }
        if !self.rays.is_empty() { m |= Self::MASK_RAYS; }
        if !self.segments.is_empty() { m |= Self::MASK_SEGMENTS; }
        if !self.pointclouds.is_empty() { m |= Self::MASK_POINTCLOUDS; }
        self.mask = m;
    }
}

impl<PCL: PointCloudMarker> Default for Collider<PCL> {
    fn default() -> Self {
        Self {
            capsules: Default::default(),
            cuboids: Default::default(),
            cylinders: Default::default(),
            planes: Default::default(),
            polygons: Default::default(),
            polytopes: Default::default(),
            points: Default::default(),
            spheres: Default::default(),
            lines: Default::default(),
            rays: Default::default(),
            segments: Default::default(),
            pointclouds: Default::default(),
            bounding: Sphere::new(glam::Vec3::ZERO, 0.0),
            mask: 0,
        }
    }
}

impl<PCL: PointCloudMarker> core::fmt::Display for Collider<PCL> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Collider(spheres: {}, capsules: {}, cuboids: {}, cylinders: {}, \
             planes: {}, polygons: {}, polytopes: {}, points: {}, \
             lines: {}, rays: {}, segments: {}, pointclouds: {})",
            self.spheres.len(),
            self.capsules.len(),
            self.cuboids.len(),
            self.cylinders.len(),
            self.planes.len(),
            self.polygons.len(),
            self.polytopes.len(),
            self.points.len(),
            self.lines.len(),
            self.rays.len(),
            self.segments.len(),
            self.pointclouds.len(),
        )
    }
}

impl<PCL: PointCloudMarker> Transformable for Collider<PCL> {
    fn translate(&mut self, offset: glam::Vec3A) {
        let m = self.mask;
        if m & Self::MASK_CAPSULES != 0 { self.capsules.translate(offset); }
        if m & Self::MASK_CUBOIDS != 0 { self.cuboids.translate(offset); }
        if m & Self::MASK_CYLINDERS != 0 { self.cylinders.translate(offset); }
        if m & Self::MASK_PLANES != 0 {
            for plane in &mut self.planes { plane.translate(offset); }
        }
        if m & Self::MASK_POLYGONS != 0 { self.polygons.translate(offset); }
        if m & Self::MASK_POLYTOPES != 0 { self.polytopes.translate(offset); }
        if m & Self::MASK_POINTS != 0 { self.points.translate(offset); }
        if m & Self::MASK_SPHERES != 0 { self.spheres.translate(offset); }
        if m & Self::MASK_LINES != 0 {
            for line in &mut self.lines { line.translate(offset); }
        }
        if m & Self::MASK_RAYS != 0 {
            for ray in &mut self.rays { ray.translate(offset); }
        }
        if m & Self::MASK_SEGMENTS != 0 { self.segments.translate(offset); }
        if m & Self::MASK_POINTCLOUDS != 0 { self.pointclouds.translate(offset); }
        self.bounding.translate(offset);
    }

    fn rotate_mat(&mut self, mat: glam::Mat3A) {
        let m = self.mask;
        if m & Self::MASK_CAPSULES != 0 { self.capsules.rotate_mat(mat); }
        if m & Self::MASK_CUBOIDS != 0 { self.cuboids.rotate_mat(mat); }
        if m & Self::MASK_CYLINDERS != 0 { self.cylinders.rotate_mat(mat); }
        if m & Self::MASK_PLANES != 0 {
            for plane in &mut self.planes { plane.rotate_mat(mat); }
        }
        if m & Self::MASK_POLYGONS != 0 { self.polygons.rotate_mat(mat); }
        if m & Self::MASK_POLYTOPES != 0 { self.polytopes.rotate_mat(mat); }
        if m & Self::MASK_POINTS != 0 { self.points.rotate_mat(mat); }
        if m & Self::MASK_SPHERES != 0 { self.spheres.rotate_mat(mat); }
        if m & Self::MASK_LINES != 0 {
            for line in &mut self.lines { line.rotate_mat(mat); }
        }
        if m & Self::MASK_RAYS != 0 {
            for ray in &mut self.rays { ray.rotate_mat(mat); }
        }
        if m & Self::MASK_SEGMENTS != 0 { self.segments.rotate_mat(mat); }
        if m & Self::MASK_POINTCLOUDS != 0 { self.pointclouds.rotate_mat(mat); }
        self.bounding.center = glam::Vec3::from(mat * glam::Vec3A::from(self.bounding.center));
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        let m = self.mask;
        if m & Self::MASK_CAPSULES != 0 { self.capsules.rotate_quat(quat); }
        if m & Self::MASK_CUBOIDS != 0 { self.cuboids.rotate_quat(quat); }
        if m & Self::MASK_CYLINDERS != 0 { self.cylinders.rotate_quat(quat); }
        if m & Self::MASK_PLANES != 0 {
            for plane in &mut self.planes { plane.rotate_quat(quat); }
        }
        if m & Self::MASK_POLYGONS != 0 { self.polygons.rotate_quat(quat); }
        if m & Self::MASK_POLYTOPES != 0 { self.polytopes.rotate_quat(quat); }
        if m & Self::MASK_POINTS != 0 { self.points.rotate_quat(quat); }
        if m & Self::MASK_SPHERES != 0 { self.spheres.rotate_quat(quat); }
        if m & Self::MASK_LINES != 0 {
            for line in &mut self.lines { line.rotate_quat(quat); }
        }
        if m & Self::MASK_RAYS != 0 {
            for ray in &mut self.rays { ray.rotate_quat(quat); }
        }
        if m & Self::MASK_SEGMENTS != 0 { self.segments.rotate_quat(quat); }
        if m & Self::MASK_POINTCLOUDS != 0 { self.pointclouds.rotate_quat(quat); }
        self.bounding.center = quat * self.bounding.center;
    }

    fn transform(&mut self, mat: glam::Affine3A) {
        let m = self.mask;
        if m & Self::MASK_CAPSULES != 0 { self.capsules.transform(mat); }
        if m & Self::MASK_CUBOIDS != 0 { self.cuboids.transform(mat); }
        if m & Self::MASK_CYLINDERS != 0 { self.cylinders.transform(mat); }
        if m & Self::MASK_PLANES != 0 {
            for plane in &mut self.planes { plane.transform(mat); }
        }
        if m & Self::MASK_POLYGONS != 0 { self.polygons.transform(mat); }
        if m & Self::MASK_POLYTOPES != 0 { self.polytopes.transform(mat); }
        if m & Self::MASK_POINTS != 0 { self.points.transform(mat); }
        if m & Self::MASK_SPHERES != 0 { self.spheres.transform(mat); }
        if m & Self::MASK_LINES != 0 {
            for line in &mut self.lines { line.transform(mat); }
        }
        if m & Self::MASK_RAYS != 0 {
            for ray in &mut self.rays { ray.transform(mat); }
        }
        if m & Self::MASK_SEGMENTS != 0 { self.segments.transform(mat); }
        if m & Self::MASK_POINTCLOUDS != 0 { self.pointclouds.transform(mat); }
        self.bounding.center = glam::Vec3::from(mat.transform_point3a(glam::Vec3A::from(self.bounding.center)));
    }
}

impl<PCL: PointCloudMarker> Scalable for Collider<PCL> {
    fn scale(&mut self, factor: f32) {
        self.capsules.scale(factor);
        self.cuboids.scale(factor);
        self.cylinders.scale(factor);
        for plane in &mut self.planes {
            plane.scale(factor);
        }
        self.polygons.scale(factor);
        self.polytopes.scale(factor);
        self.points.scale(factor);
        self.spheres.scale(factor);
        for line in &mut self.lines {
            line.scale(factor);
        }
        for ray in &mut self.rays {
            ray.scale(factor);
        }
        self.segments.scale(factor);
        self.bounding.radius *= factor;
    }
}

impl<PCL: PointCloudMarker> Bounded for Collider<PCL> {
    #[inline]
    fn broadphase(&self) -> Sphere {
        self.bounding
    }

    #[inline]
    fn obb(&self) -> Cuboid {
        self.aabb()
    }

    #[inline]
    fn aabb(&self) -> Cuboid {
        let r = self.bounding.radius;
        Cuboid::from_aabb(
            self.bounding.center - glam::Vec3::splat(r),
            self.bounding.center + glam::Vec3::splat(r),
        )
    }
}

impl Stretchable for Collider<NoPcl> {
    type Output = Self;

    fn stretch(&self, translation: glam::Vec3) -> Self::Output {
        let mut out = Collider::default();

        for sphere in self.spheres.iter() {
            match sphere.stretch(translation) {
                sphere::SphereStretch::NoStretch(s) => out.spheres.push(s),
                sphere::SphereStretch::Stretch(c) => out.capsules.push(c),
            }
        }

        for capsule in &self.capsules {
            match capsule.stretch(translation) {
                capsule::CapsuleStretch::Aligned(c) => out.capsules.push(c),
                capsule::CapsuleStretch::Unaligned(edges, poly) => {
                    out.capsules.extend(edges);
                    out.polytopes.push(poly);
                }
            }
        }

        for cuboid in &self.cuboids {
            match cuboid.stretch(translation) {
                cuboid::CuboidStretch::Aligned(c) => out.cuboids.push(c),
                cuboid::CuboidStretch::Unaligned(p) => out.polytopes.push(p),
            }
        }

        for cylinder in &self.cylinders {
            match cylinder.stretch(translation) {
                cylinder::CylinderStretch::Aligned(c) => out.cylinders.push(c),
                cylinder::CylinderStretch::Unaligned(edges, poly) => {
                    out.capsules.extend(edges);
                    out.polytopes.push(poly);
                }
            }
        }

        for polytope in &self.polytopes {
            out.polytopes.push(polytope.stretch(translation));
        }

        for plane in &self.planes {
            out.planes.push(plane.stretch(translation));
        }

        for polygon in &self.polygons {
            match polygon.stretch(translation) {
                plane::ConvexPolygonStretch::InPlane(p) => out.polygons.push(p),
                plane::ConvexPolygonStretch::OutOfPlane(p) => out.polytopes.push(p),
            }
        }

        for point in &self.points {
            out.segments.push(point.stretch(translation));
        }

        for line in &self.lines {
            match line.stretch(translation) {
                line::LineStretch::Parallel(l) => out.lines.push(l),
                line::LineStretch::Polygon(p) => out.polygons.push(p),
            }
        }

        for ray in &self.rays {
            match ray.stretch(translation) {
                line::RayStretch::Parallel(r) => out.rays.push(r),
                line::RayStretch::Polygon(p) => out.polygons.push(p),
            }
        }

        for segment in &self.segments {
            match segment.stretch(translation) {
                line::LineSegmentStretch::Parallel(s) => out.segments.push(s),
                line::LineSegmentStretch::Polygon(p) => out.polygons.push(p),
            }
        }

        out.recompute_bounding();
        out.recompute_mask();
        out
    }
}

impl Collider<Pointcloud> {
    /// Attempts to stretch the collider by the given translation.
    /// Returns `None` if the collider contains pointclouds, since
    /// pointclouds cannot be stretched.
    #[must_use]
    pub fn try_stretch(&self, translation: glam::Vec3) -> Option<Collider<NoPcl>> {
        if !self.pointclouds.is_empty() {
            return None;
        }
        let no_pcl: Collider<NoPcl> = self.clone().into();
        Some(no_pcl.stretch(translation))
    }

    /// Attempts to stretch the collider by the given translation (double precision).
    /// Returns `None` if the collider contains pointclouds, since
    /// pointclouds cannot be stretched.
    #[inline]
    #[must_use]
    pub fn try_stretch_d(&self, translation: glam::DVec3) -> Option<Collider<NoPcl>> {
        self.try_stretch(translation.as_vec3())
    }
}

impl<PCL: PointCloudMarker> Collider<PCL> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clone_from(colliders: &[&Collider<PCL>]) -> Self {
        let mut capsule_len = 0;
        let mut cuboid_len = 0;
        let mut cylinder_len = 0;
        let mut plane_len = 0;
        let mut polygon_len = 0;
        let mut polytope_len = 0;
        let mut point_len = 0;
        let mut sphere_len = 0;
        let mut line_len = 0;
        let mut ray_len = 0;
        let mut segment_len = 0;
        let mut pointcloud_len = 0;

        for c in colliders {
            capsule_len += c.capsules.len();
            cuboid_len += c.cuboids.len();
            cylinder_len += c.cylinders.len();
            plane_len += c.planes.len();
            polygon_len += c.polygons.len();
            polytope_len += c.polytopes.len();
            point_len += c.points.len();
            sphere_len += c.spheres.len();
            line_len += c.lines.len();
            ray_len += c.rays.len();
            segment_len += c.segments.len();
            pointcloud_len += c.pointclouds.len();
        }

        let mut capsules = soa::BroadCollection::with_capacity(capsule_len);
        let mut cuboids = soa::BroadCollection::with_capacity(cuboid_len);
        let mut cylinders = soa::BroadCollection::with_capacity(cylinder_len);
        let mut planes = Vec::with_capacity(plane_len);
        let mut polygons = soa::BroadCollection::with_capacity(polygon_len);
        let mut polytopes = soa::BroadCollection::with_capacity(polytope_len);
        let mut points = soa::BroadCollection::with_capacity(point_len);
        let mut spheres = soa::SpheresSoA::with_capacity(sphere_len);
        let mut lines = Vec::with_capacity(line_len);
        let mut rays = Vec::with_capacity(ray_len);
        let mut segments = soa::BroadCollection::with_capacity(segment_len);
        let mut pointclouds = soa::BroadCollection::with_capacity(pointcloud_len);

        for c in colliders {
            capsules.extend_from_slice(c.capsules.items());
            cuboids.extend_from_slice(c.cuboids.items());
            cylinders.extend_from_slice(c.cylinders.items());
            planes.extend_from_slice(&c.planes);
            polygons.extend_from_slice(c.polygons.items());
            polytopes.extend_from_slice(c.polytopes.items());
            points.extend_from_slice(c.points.items());
            spheres.extend_from(&c.spheres);
            lines.extend_from_slice(&c.lines);
            rays.extend_from_slice(&c.rays);
            segments.extend_from_slice(c.segments.items());
            pointclouds.extend_from_slice(c.pointclouds.items());
        }

        let mut collider = Self {
            capsules,
            cuboids,
            cylinders,
            planes,
            polygons,
            polytopes,
            points,
            spheres,
            lines,
            rays,
            segments,
            pointclouds,
            bounding: Sphere::new(glam::Vec3::ZERO, 0.0),
            mask: 0,
        };
        collider.recompute_bounding();
        collider.recompute_mask();
        collider
    }

    /// Expand the cached bounding sphere to enclose `other`.
    fn expand_bounding(&mut self, other: &Sphere) {
        if self.bounding.radius == 0.0 && self.bounding.center == glam::Vec3::ZERO {
            // First shape — just adopt its bounding sphere.
            self.bounding = *other;
            return;
        }
        let d = other.center - self.bounding.center;
        let dist = d.length();
        let needed = dist + other.radius;
        if needed > self.bounding.radius {
            // New sphere that encloses both.
            let new_radius = (self.bounding.radius + needed) * 0.5;
            let shift = new_radius - self.bounding.radius;
            if dist > f32::EPSILON {
                self.bounding.center += d * (shift / dist);
            }
            self.bounding.radius = new_radius;
        }
    }

    /// Recompute bounding sphere from scratch (e.g. after stretch).
    fn recompute_bounding(&mut self) {
        if !self.planes.is_empty() || !self.lines.is_empty() || !self.rays.is_empty() {
            self.bounding = Sphere::new(glam::Vec3::ZERO, f32::INFINITY);
            return;
        }

        let mut center = glam::Vec3::ZERO;
        let mut radius = 0.0f32;
        let mut first = true;

        macro_rules! merge {
            ($col:expr) => {
                for item in $col.iter() {
                    let bp = item.broadphase();
                    if first {
                        center = bp.center;
                        radius = bp.radius;
                        first = false;
                        continue;
                    }
                    let d = bp.center - center;
                    let dist = d.length();
                    let needed = dist + bp.radius;
                    if needed > radius {
                        let new_radius = (radius + needed) * 0.5;
                        let shift = new_radius - radius;
                        if dist > f32::EPSILON {
                            center += d * (shift / dist);
                        }
                        radius = new_radius;
                    }
                }
            };
        }
        merge!(self.spheres);
        merge!(self.capsules);
        merge!(self.cuboids);
        merge!(self.cylinders);
        merge!(self.polytopes);
        merge!(self.polygons);
        merge!(self.points);
        merge!(self.segments);
        merge!(self.pointclouds);

        if first {
            self.bounding = Sphere::new(glam::Vec3::ZERO, 0.0);
        } else {
            self.bounding = Sphere::new(center, radius);
        }
    }

    /// Recompute the bounding sphere using iterative refinement for a tighter fit.
    ///
    /// More expensive than the default incremental bounding (`O(8n)` vs `O(n)`)
    /// but typically produces a sphere within ~1% of the minimum enclosing ball.
    pub fn refine_bounding(&mut self) {
        if !self.planes.is_empty() || !self.lines.is_empty() || !self.rays.is_empty() {
            self.bounding = Sphere::new(glam::Vec3::ZERO, f32::INFINITY);
            return;
        }

        macro_rules! for_each_broad {
            ($self:expr, |$s:ident| $body:expr) => {
                for $s in $self.spheres.iter() { $body }
                for $s in $self.capsules.broad.iter() { $body }
                for $s in $self.cuboids.broad.iter() { $body }
                for $s in $self.cylinders.broad.iter() { $body }
                for $s in $self.polytopes.broad.iter() { $body }
                for $s in $self.polygons.broad.iter() { $body }
                for $s in $self.points.broad.iter() { $body }
                for $s in $self.segments.broad.iter() { $body }
                for $s in $self.pointclouds.broad.iter() { $body }
            };
        }

        let mut centroid = glam::Vec3::ZERO;
        let mut count = 0u32;
        for_each_broad!(self, |s| {
            centroid += s.center;
            count += 1;
        });

        if count == 0 {
            self.bounding = Sphere::new(glam::Vec3::ZERO, 0.0);
            return;
        }

        centroid /= count as f32;
        let mut center = centroid;
        let mut radius = 0.0f32;
        for_each_broad!(self, |s| {
            let extent = (s.center - center).length() + s.radius;
            radius = radius.max(extent);
        });

        for i in 0..8u32 {
            let mut farthest_point = center;
            let mut farthest_dist = 0.0f32;
            for_each_broad!(self, |s| {
                let d = s.center - center;
                let dist = d.length();
                let extent = dist + s.radius;
                if extent > farthest_dist {
                    farthest_dist = extent;
                    farthest_point = if dist > f32::EPSILON {
                        s.center + d * (s.radius / dist)
                    } else {
                        center + glam::Vec3::X * s.radius
                    };
                }
            });

            let step = 1.0 / (i as f32 + 2.0);
            center += (farthest_point - center) * step;

            radius = 0.0;
            for_each_broad!(self, |s| {
                let extent = (s.center - center).length() + s.radius;
                radius = radius.max(extent);
            });
        }

        self.bounding = Sphere::new(center, radius);
    }

    /// Check if `query` sphere overlaps any per-shape broadphase sphere
    /// in this collider (bounded shapes only).
    fn broad_overlaps_any(&self, query: &Sphere) -> bool {
        if self.mask == 0 {
            return false;
        }
        if self.mask & (Self::MASK_PLANES | Self::MASK_LINES | Self::MASK_RAYS) != 0 {
            return true;
        }
        (self.mask & Self::MASK_SPHERES != 0 && self.spheres.any_collides_sphere(query))
            || (self.mask & Self::MASK_CAPSULES != 0 && self.capsules.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_CUBOIDS != 0 && self.cuboids.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_CYLINDERS != 0 && self.cylinders.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_POLYGONS != 0 && self.polygons.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_POLYTOPES != 0 && self.polytopes.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_POINTS != 0 && self.points.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_SEGMENTS != 0 && self.segments.broad.any_collides_sphere(query))
            || (self.mask & Self::MASK_POINTCLOUDS != 0 && self.pointclouds.broad.any_collides_sphere(query))
    }

    pub fn capsules(&self) -> &[Capsule] {
        self.capsules.items()
    }
    pub fn cuboids(&self) -> &[Cuboid] {
        self.cuboids.items()
    }
    pub fn cylinders(&self) -> &[Cylinder] {
        self.cylinders.items()
    }
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }
    pub fn polygons(&self) -> &[ConvexPolygon] {
        self.polygons.items()
    }
    pub fn polytopes(&self) -> &[ConvexPolytope] {
        self.polytopes.items()
    }
    pub fn points(&self) -> &[Point] {
        self.points.items()
    }
    pub fn spheres(&self) -> &soa::SpheresSoA {
        &self.spheres
    }
    pub fn lines(&self) -> &[Line] {
        &self.lines
    }
    pub fn rays(&self) -> &[Ray] {
        &self.rays
    }
    pub fn segments(&self) -> &[LineSegment] {
        self.segments.items()
    }
    pub fn pointclouds(&self) -> &[PCL] {
        self.pointclouds.items()
    }

    /// Collision test — dispatches to SIMD-accelerated batch paths when
    /// available for the concrete query type, otherwise broadphase + scalar
    /// narrowphase via [`BroadCollection`].
    #[inline]
    #[must_use]
    pub fn collides<T: ColliderQuery<PCL>>(&self, shape: &T) -> bool {
        shape.query_collider(self)
    }

    #[inline]
    pub fn add<T: ColliderComponent<PCL>>(&mut self, shape: T) {
        shape.add_to_shapes(self);
    }

    #[inline]
    pub fn add_slice<T: ColliderComponent<PCL>>(&mut self, shapes: &[T]) {
        for shape in shapes {
            self.add(shape.clone());
        }
    }

    #[inline]
    pub fn add_vec<T: ColliderComponent<PCL>>(&mut self, shapes: Vec<T>) {
        for shape in shapes {
            self.add(shape);
        }
    }

    /// Consumes `other` and moves all of its contents into `self`.
    ///
    /// Uses bulk Vec/SoA appends instead of per-item inserts, avoiding
    /// repeated SIMD re-padding. The bounding sphere is merged once at
    /// the end.
    pub fn include(&mut self, mut other: Collider<PCL>) {
        self.capsules.append(&mut other.capsules);
        self.cuboids.append(&mut other.cuboids);
        self.cylinders.append(&mut other.cylinders);
        self.polygons.append(&mut other.polygons);
        self.polytopes.append(&mut other.polytopes);
        self.points.append(&mut other.points);
        self.spheres.append(&mut other.spheres);
        self.segments.append(&mut other.segments);
        self.pointclouds.append(&mut other.pointclouds);
        self.planes.append(&mut other.planes);
        self.lines.append(&mut other.lines);
        self.rays.append(&mut other.rays);

        self.mask |= other.mask;

        if other.bounding.radius == f32::INFINITY {
            self.bounding.radius = f32::INFINITY;
        } else {
            self.expand_bounding(&other.bounding);
        }
    }
}

impl From<Collider<Pointcloud>> for Collider<NoPcl> {
    fn from(collider: Collider<Pointcloud>) -> Self {
        let mut mask = collider.mask;
        mask &= !Collider::<Pointcloud>::MASK_POINTCLOUDS;
        Self {
            capsules: collider.capsules,
            cuboids: collider.cuboids,
            cylinders: collider.cylinders,
            planes: collider.planes,
            polygons: collider.polygons,
            polytopes: collider.polytopes,
            points: collider.points,
            spheres: collider.spheres,
            lines: collider.lines,
            rays: collider.rays,
            segments: collider.segments,
            pointclouds: Default::default(),
            bounding: collider.bounding,
            mask,
        }
    }
}

impl From<Collider<NoPcl>> for Collider<Pointcloud> {
    fn from(collider: Collider<NoPcl>) -> Self {
        Self {
            capsules: collider.capsules,
            cuboids: collider.cuboids,
            cylinders: collider.cylinders,
            planes: collider.planes,
            polygons: collider.polygons,
            polytopes: collider.polytopes,
            points: collider.points,
            spheres: collider.spheres,
            lines: collider.lines,
            rays: collider.rays,
            segments: collider.segments,
            pointclouds: Default::default(),
            bounding: collider.bounding,
            mask: collider.mask,
        }
    }
}

#[cfg(feature = "serde")]
impl<PCL> serde::Serialize for Collider<PCL>
where
    PCL: PointCloudMarker + serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Collider", 12)?;
        s.serialize_field("capsules", &self.capsules)?;
        s.serialize_field("cuboids", &self.cuboids)?;
        s.serialize_field("cylinders", &self.cylinders)?;
        s.serialize_field("planes", &self.planes)?;
        s.serialize_field("polygons", &self.polygons)?;
        s.serialize_field("polytopes", &self.polytopes)?;
        s.serialize_field("points", &self.points)?;
        s.serialize_field("spheres", &self.spheres)?;
        s.serialize_field("lines", &self.lines)?;
        s.serialize_field("rays", &self.rays)?;
        s.serialize_field("segments", &self.segments)?;
        s.serialize_field("pointclouds", &self.pointclouds)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, PCL> serde::Deserialize<'de> for Collider<PCL>
where
    PCL: PointCloudMarker + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(bound = "P: PointCloudMarker + serde::Deserialize<'de>")]
        struct ColliderHelper<P: PointCloudMarker> {
            capsules: soa::BroadCollection<Capsule>,
            cuboids: soa::BroadCollection<Cuboid>,
            cylinders: soa::BroadCollection<Cylinder>,
            planes: Vec<Plane>,
            polygons: soa::BroadCollection<ConvexPolygon>,
            polytopes: soa::BroadCollection<ConvexPolytope>,
            points: soa::BroadCollection<Point>,
            spheres: soa::SpheresSoA,
            lines: Vec<Line>,
            rays: Vec<Ray>,
            segments: soa::BroadCollection<LineSegment>,
            pointclouds: soa::BroadCollection<P>,
        }
        let h = ColliderHelper::<PCL>::deserialize(deserializer)?;
        let mut collider = Self {
            capsules: h.capsules,
            cuboids: h.cuboids,
            cylinders: h.cylinders,
            planes: h.planes,
            polygons: h.polygons,
            polytopes: h.polytopes,
            points: h.points,
            spheres: h.spheres,
            lines: h.lines,
            rays: h.rays,
            segments: h.segments,
            pointclouds: h.pointclouds,
            bounding: Sphere::new(glam::Vec3::ZERO, 0.0),
            mask: 0,
        };
        collider.recompute_bounding();
        collider.recompute_mask();
        Ok(collider)
    }
}

/// Trait for types that can query a [`Collider`] for collisions.
/// Each shape type implements this with its own optimized dispatch path,
/// so `Collider::collides` and `collides_other` just call through here.
pub trait ColliderQuery<PCL: PointCloudMarker>: Sized + Clone {
    fn query_collider(&self, collider: &Collider<PCL>) -> bool;
}

/// Generic broadphase + scalar narrowphase path for bounded shape types.
macro_rules! impl_collider_query_generic {
    ($ty:ty) => {
        impl ColliderQuery<NoPcl> for $ty {
            fn query_collider(&self, c: &Collider<NoPcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<NoPcl>::MASK_SPHERES != 0 && c.spheres.any_collides_sphere(&self.broadphase()))
                    || (c.mask & Collider::<NoPcl>::MASK_POINTS != 0 && c.points.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_CAPSULES != 0 && c.capsules.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_CUBOIDS != 0 && c.cuboids.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_CYLINDERS != 0 && c.cylinders.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_SEGMENTS != 0 && c.segments.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_POLYGONS != 0 && c.polygons.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_POLYTOPES != 0 && c.polytopes.collides(self))
                    || (c.mask & Collider::<NoPcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<NoPcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<NoPcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<NoPcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.collides(self))
            }
        }
        impl ColliderQuery<Pointcloud> for $ty {
            fn query_collider(&self, c: &Collider<Pointcloud>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<Pointcloud>::MASK_SPHERES != 0 && c.spheres.any_collides_sphere(&self.broadphase()))
                    || (c.mask & Collider::<Pointcloud>::MASK_POINTS != 0 && c.points.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_CAPSULES != 0 && c.capsules.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_CUBOIDS != 0 && c.cuboids.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_CYLINDERS != 0 && c.cylinders.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_SEGMENTS != 0 && c.segments.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_POLYGONS != 0 && c.polygons.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_POLYTOPES != 0 && c.polytopes.collides(self))
                    || (c.mask & Collider::<Pointcloud>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<Pointcloud>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<Pointcloud>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<Pointcloud>::MASK_POINTCLOUDS != 0 && c.pointclouds.collides(self))
            }
        }
    };
}

impl_collider_query_generic!(Capsule);
impl_collider_query_generic!(Cuboid);
impl_collider_query_generic!(Cylinder);
impl_collider_query_generic!(ConvexPolygon);
impl_collider_query_generic!(ConvexPolytope);
/// Point: broadphase-only for spheres and points (zero-radius sphere check).
macro_rules! impl_collider_query_point {
    ($pcl:ty) => {
        impl ColliderQuery<$pcl> for Point {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<$pcl>::MASK_SPHERES != 0 && c.spheres.any_collides_sphere(&self.broadphase()))
                    || (c.mask & Collider::<$pcl>::MASK_POINTS != 0 && c.points.collides_only_broadphase(self))
                    || (c.mask & Collider::<$pcl>::MASK_CAPSULES != 0 && c.capsules.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CUBOIDS != 0 && c.cuboids.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CYLINDERS != 0 && c.cylinders.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_SEGMENTS != 0 && c.segments.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYGONS != 0 && c.polygons.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYTOPES != 0 && c.polytopes.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.collides(self))
            }
        }
    };
}
impl_collider_query_point!(NoPcl);
impl_collider_query_point!(Pointcloud);

/// Sphere: broadphase-only for spheres/points, SIMD batch for capsules/cuboids.
macro_rules! impl_collider_query_sphere {
    ($pcl:ty) => {
        impl ColliderQuery<$pcl> for Sphere {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<$pcl>::MASK_SPHERES != 0 && c.spheres.any_collides_sphere(self))
                    || (c.mask & Collider::<$pcl>::MASK_POINTS != 0 && c.points.collides_only_broadphase(self))
                    || (c.mask & Collider::<$pcl>::MASK_CAPSULES != 0
                        && soa::batch::sphere_vs_capsules_broad(self, &c.capsules))
                    || (c.mask & Collider::<$pcl>::MASK_CUBOIDS != 0
                        && soa::batch::sphere_vs_cuboids_broad(self, &c.cuboids))
                    || (c.mask & Collider::<$pcl>::MASK_CYLINDERS != 0
                        && soa::batch::sphere_vs_cylinders_broad(self, &c.cylinders))
                    || (c.mask & Collider::<$pcl>::MASK_SEGMENTS != 0 && c.segments.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYGONS != 0 && c.polygons.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYTOPES != 0 && c.polytopes.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.collides(self))
            }
        }
    };
}
impl_collider_query_sphere!(NoPcl);
impl_collider_query_sphere!(Pointcloud);

/// Plane: SIMD batch for spheres/capsules/cuboids, scalar for rest.
macro_rules! impl_collider_query_plane {
    ($pcl:ty) => {
        impl ColliderQuery<$pcl> for Plane {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<$pcl>::MASK_SPHERES != 0 && soa::batch::plane_vs_spheres_soa(self, &c.spheres))
                    || (c.mask & Collider::<$pcl>::MASK_CAPSULES != 0 && soa::batch::plane_vs_capsules_broad(self, &c.capsules))
                    || (c.mask & Collider::<$pcl>::MASK_CUBOIDS != 0 && soa::batch::plane_vs_cuboids_broad(self, &c.cuboids))
                    || (c.mask & Collider::<$pcl>::MASK_CYLINDERS != 0 && soa::batch::plane_vs_cylinders_broad(self, &c.cylinders))
                    || (c.mask & Collider::<$pcl>::MASK_POINTS != 0 && c.points.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_SEGMENTS != 0 && c.segments.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POLYGONS != 0 && c.polygons.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POLYTOPES != 0 && c.polytopes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.iter().any(|x| self.collides(x)))
            }
        }
    };
}
impl_collider_query_plane!(NoPcl);
impl_collider_query_plane!(Pointcloud);

/// Line/Ray: SIMD batch for spheres, scalar for rest.
macro_rules! impl_collider_query_line_like {
    ($ty:ty, $batch_fn:path, $pcl:ty) => {
        impl ColliderQuery<$pcl> for $ty {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<$pcl>::MASK_SPHERES != 0 && $batch_fn(self, &c.spheres))
                    || (c.mask & Collider::<$pcl>::MASK_CAPSULES != 0 && c.capsules.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_CUBOIDS != 0 && c.cuboids.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_CYLINDERS != 0 && c.cylinders.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_SEGMENTS != 0 && c.segments.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POLYGONS != 0 && c.polygons.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POLYTOPES != 0 && c.polytopes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTS != 0 && c.points.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.iter().any(|x| self.collides(x)))
            }
        }
    };
}
impl_collider_query_line_like!(Line, soa::batch::line_vs_spheres_soa, NoPcl);
impl_collider_query_line_like!(Line, soa::batch::line_vs_spheres_soa, Pointcloud);
impl_collider_query_line_like!(Ray, soa::batch::ray_vs_spheres_soa, NoPcl);
impl_collider_query_line_like!(Ray, soa::batch::ray_vs_spheres_soa, Pointcloud);

/// LineSegment: SIMD batch for spheres, broadphase for bounded types.
macro_rules! impl_collider_query_segment {
    ($pcl:ty) => {
        impl ColliderQuery<$pcl> for LineSegment {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<$pcl>::MASK_SPHERES != 0
                    && c.spheres.any_collides_sphere(&self.broadphase())
                    && soa::batch::segment_vs_spheres_soa(self, &c.spheres))
                    || (c.mask & Collider::<$pcl>::MASK_CAPSULES != 0 && c.capsules.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CUBOIDS != 0 && c.cuboids.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CYLINDERS != 0 && c.cylinders.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_SEGMENTS != 0 && c.segments.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYGONS != 0 && c.polygons.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYTOPES != 0 && c.polytopes.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTS != 0 && c.points.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.collides(self))
            }
        }
    };
}
impl_collider_query_segment!(NoPcl);
impl_collider_query_segment!(Pointcloud);

macro_rules! impl_collider_query_pointcloud {
    ($pcl:ty) => {
        impl ColliderQuery<$pcl> for Pointcloud {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                if c.mask == 0 { return false; }
                (c.mask & Collider::<$pcl>::MASK_SPHERES != 0 && c.spheres.any_collides_sphere(&self.broadphase()))
                    || (c.mask & Collider::<$pcl>::MASK_POINTS != 0 && c.points.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CAPSULES != 0 && c.capsules.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CUBOIDS != 0 && c.cuboids.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_CYLINDERS != 0 && c.cylinders.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_SEGMENTS != 0 && c.segments.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYGONS != 0 && c.polygons.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_POLYTOPES != 0 && c.polytopes.collides(self))
                    || (c.mask & Collider::<$pcl>::MASK_PLANES != 0 && c.planes.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_LINES != 0 && c.lines.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_RAYS != 0 && c.rays.iter().any(|x| self.collides(x)))
                    || (c.mask & Collider::<$pcl>::MASK_POINTCLOUDS != 0 && c.pointclouds.collides(self))
            }
        }
    };
}
impl_collider_query_pointcloud!(NoPcl);
impl_collider_query_pointcloud!(Pointcloud);

macro_rules! impl_collider_collides_other {
    ($pcl:ty) => {
        impl Collider<$pcl> {
            #[must_use]
            pub fn collides_other(&self, other: &Collider<$pcl>) -> bool {
                if self.mask == 0 || other.mask == 0 {
                    return false;
                }
                // Top-level broadphase: bounding spheres of both colliders
                if !self.bounding.collides(&other.bounding) {
                    return false;
                }
                // Check each collider's bounding sphere against the other's
                // per-shape broadphase SoA — if neither can see the other,
                // no individual shape pair can collide.
                // Skip when both colliders hold a single shape type: the
                // top-level bounding-sphere test already covers that case.
                let single_type = |m: u16| m & (m - 1) == 0;
                if !(single_type(self.mask) && single_type(other.mask))
                    && !other.broad_overlaps_any(&self.bounding)
                    && !self.broad_overlaps_any(&other.bounding)
                {
                    return false;
                }

                (other.mask & Self::MASK_SPHERES != 0 && self.spheres.any_collides_soa(&other.spheres))
                    || (other.mask & Self::MASK_POINTS != 0 && other.points.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_CAPSULES != 0 && other.capsules.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_CUBOIDS != 0 && other.cuboids.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_CYLINDERS != 0 && other.cylinders.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_SEGMENTS != 0 && other.segments.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_POLYGONS != 0 && other.polygons.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_POLYTOPES != 0 && other.polytopes.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_PLANES != 0 && other.planes.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_LINES != 0 && other.lines.iter().any(|x| self.collides(x)))
                    || (other.mask & Self::MASK_RAYS != 0 && other.rays.iter().any(|x| self.collides(x)))
            }
        }
    };
}

impl_collider_collides_other!(NoPcl);

impl Collider<Pointcloud> {
    #[must_use]
    pub fn collides_other(&self, other: &Collider<Pointcloud>) -> bool {
        if self.mask == 0 || other.mask == 0 {
            return false;
        }
        if !self.bounding.collides(&other.bounding) {
            return false;
        }
        let single_type = |m: u16| m & (m - 1) == 0;
        if !(single_type(self.mask) && single_type(other.mask))
            && !other.broad_overlaps_any(&self.bounding)
            && !self.broad_overlaps_any(&other.bounding)
        {
            return false;
        }

        (other.mask & Self::MASK_SPHERES != 0 && self.spheres.any_collides_soa(&other.spheres))
            || (other.mask & Self::MASK_POINTS != 0 && other.points.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_CAPSULES != 0 && other.capsules.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_CUBOIDS != 0 && other.cuboids.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_CYLINDERS != 0 && other.cylinders.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_SEGMENTS != 0 && other.segments.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_POLYGONS != 0 && other.polygons.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_POLYTOPES != 0 && other.polytopes.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_PLANES != 0 && other.planes.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_LINES != 0 && other.lines.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_RAYS != 0 && other.rays.iter().any(|x| self.collides(x)))
            || (other.mask & Self::MASK_POINTCLOUDS != 0 && other.pointclouds.iter().any(|x| self.collides(x)))
    }
}
