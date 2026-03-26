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

pub mod soa;

pub use capsule::Capsule;
pub use convex_polytope::array::ArrayConvexPolytope;
pub use convex_polytope::heap::ConvexPolytope;
pub use cuboid::Cuboid;
pub use cylinder::Cylinder;
pub use line::Line;
pub use line::LineSegment;
pub use line::LineSegmentStretch;
pub use line::LineStretch;
pub use line::Ray;
pub use line::RayStretch;
pub use plane::ArrayConvexPolygon;
pub use plane::ConvexPolygon;
pub use plane::ConvexPolygonStretch;
pub use plane::Plane;
pub use point::Point;
pub use pointcloud::Pointcloud;
pub use sphere::Sphere;

pub use crate::pointcloud::NoPcl;
use crate::pointcloud::PointCloudMarker;

pub mod stretched {
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
    fn scaled(self, factor: f32) -> Self {
        let mut cloned = self.clone();
        cloned.scale(factor);
        cloned
    }
    #[inline]
    fn scale_d(&mut self, factor: f64) {
        self.scale(factor as f32);
    }
    #[inline]
    fn scaled_d(self, factor: f64) -> Self {
        self.scaled(factor as f32)
    }
}

pub trait Transformable: Sized + Clone {
    fn translate(&mut self, offset: glam::Vec3);
    #[inline]
    fn translated(self, offset: glam::Vec3) -> Self {
        let mut cloned = self.clone();
        cloned.translate(offset);
        cloned
    }
    #[inline]
    fn translate_d(&mut self, offset: glam::DVec3) {
        self.translate(offset.as_vec3());
    }
    #[inline]
    fn translated_d(self, offset: glam::DVec3) -> Self {
        self.translated(offset.as_vec3())
    }
    fn rotate_mat(&mut self, mat: glam::Mat3);
    #[inline]
    fn rotated_mat(self, mat: glam::Mat3) -> Self {
        let mut cloned = self.clone();
        cloned.rotate_mat(mat);
        cloned
    }
    #[inline]
    fn rotate_mat_d(&mut self, mat: glam::DMat3) {
        self.rotate_mat(mat.as_mat3());
    }
    #[inline]
    fn rotated_mat_d(self, mat: glam::DMat3) -> Self {
        self.rotated_mat(mat.as_mat3())
    }
    fn rotate_quat(&mut self, quat: glam::Quat);
    #[inline]
    fn rotated_quat(self, quat: glam::Quat) -> Self {
        let mut cloned = self.clone();
        cloned.rotate_quat(quat);
        cloned
    }
    #[inline]
    fn rotate_quat_d(&mut self, quat: glam::DQuat) {
        self.rotate_quat(quat.as_quat());
    }
    #[inline]
    fn rotated_quat_d(self, quat: glam::DQuat) -> Self {
        self.rotated_quat(quat.as_quat())
    }
    fn transform(&mut self, mat: glam::Affine3);
    #[inline]
    fn transformed(self, mat: glam::Affine3) -> Self {
        let mut cloned = self.clone();
        cloned.transform(mat);
        cloned
    }
    #[inline]
    fn transform_d(&mut self, mat: glam::DAffine3) {
        self.transform(mat.as_affine3());
    }
    #[inline]
    fn transformed_d(self, mat: glam::DAffine3) -> Self {
        self.transformed(mat.as_affine3())
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
    ($ty:ty, $field:ident) => {
        impl<PCL: PointCloudMarker> ColliderComponent<PCL> for $ty {
            fn add_to_shapes(self, c: &mut Collider<PCL>) {
                c.expand_bounding(&self.broadphase());
                c.$field.push(self);
            }
        }
    };
}

macro_rules! impl_shape_for_unbounded {
    ($ty:ty, $field:ident) => {
        impl<PCL: PointCloudMarker> ColliderComponent<PCL> for $ty {
            fn add_to_shapes(self, c: &mut Collider<PCL>) {
                c.bounding.radius = f32::INFINITY;
                c.$field.push(self);
            }
        }
    };
}

impl_shape_for_bounded!(Sphere, spheres);
impl_shape_for_bounded!(Capsule, capsules);
impl_shape_for_bounded!(Cuboid, cuboids);
impl_shape_for_bounded!(Cylinder, cylinders);
impl_shape_for_bounded!(ConvexPolytope, polytopes);
impl_shape_for_bounded!(ConvexPolygon, polygons);
impl_shape_for_bounded!(Point, points);
impl_shape_for_bounded!(LineSegment, segments);
impl_shape_for_unbounded!(Plane, planes);
impl_shape_for_unbounded!(Line, lines);
impl_shape_for_unbounded!(Ray, rays);

impl ColliderComponent<Pointcloud> for Pointcloud {
    fn add_to_shapes(self, c: &mut Collider<Pointcloud>) {
        c.expand_bounding(&self.broadphase());
        c.pointclouds.push(self);
    }
}

impl<const P: usize, const V: usize, PCL: PointCloudMarker> ColliderComponent<PCL>
    for ArrayConvexPolytope<P, V>
{
    fn add_to_shapes(self, c: &mut Collider<PCL>) {
        let poly = ConvexPolytope::from(self);
        c.expand_bounding(&poly.broadphase());
        c.polytopes.push(poly);
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
    spheres: soa::BroadCollection<Sphere>,
    lines: Vec<Line>,
    rays: Vec<Ray>,
    segments: soa::BroadCollection<LineSegment>,
    pointclouds: soa::BroadCollection<PCL>,
    bounding: Sphere,
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
        }
    }
}

impl<PCL: PointCloudMarker> Transformable for Collider<PCL> {
    fn translate(&mut self, offset: glam::Vec3) {
        self.capsules.translate(offset);
        self.cuboids.translate(offset);
        self.cylinders.translate(offset);
        for plane in &mut self.planes {
            plane.translate(offset);
        }
        self.polygons.translate(offset);
        self.polytopes.translate(offset);
        self.points.translate(offset);
        self.spheres.translate(offset);
        for line in &mut self.lines {
            line.translate(offset);
        }
        for ray in &mut self.rays {
            ray.translate(offset);
        }
        self.segments.translate(offset);
        self.bounding.translate(offset);
    }

    fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.capsules.rotate_mat(mat);
        self.cuboids.rotate_mat(mat);
        self.cylinders.rotate_mat(mat);
        for plane in &mut self.planes {
            plane.rotate_mat(mat);
        }
        self.polygons.rotate_mat(mat);
        self.polytopes.rotate_mat(mat);
        self.points.rotate_mat(mat);
        self.spheres.rotate_mat(mat);
        for line in &mut self.lines {
            line.rotate_mat(mat);
        }
        for ray in &mut self.rays {
            ray.rotate_mat(mat);
        }
        self.segments.rotate_mat(mat);
        self.bounding.center = mat * self.bounding.center;
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        self.capsules.rotate_quat(quat);
        self.cuboids.rotate_quat(quat);
        self.cylinders.rotate_quat(quat);
        for plane in &mut self.planes {
            plane.rotate_quat(quat);
        }
        self.polygons.rotate_quat(quat);
        self.polytopes.rotate_quat(quat);
        self.points.rotate_quat(quat);
        self.spheres.rotate_quat(quat);
        for line in &mut self.lines {
            line.rotate_quat(quat);
        }
        for ray in &mut self.rays {
            ray.rotate_quat(quat);
        }
        self.segments.rotate_quat(quat);
        self.bounding.center = quat * self.bounding.center;
    }

    fn transform(&mut self, mat: glam::Affine3) {
        self.capsules.transform(mat);
        self.cuboids.transform(mat);
        self.cylinders.transform(mat);
        for plane in &mut self.planes {
            plane.transform(mat);
        }
        self.polygons.transform(mat);
        self.polytopes.transform(mat);
        self.points.transform(mat);
        self.spheres.transform(mat);
        for line in &mut self.lines {
            line.transform(mat);
        }
        for ray in &mut self.rays {
            ray.transform(mat);
        }
        self.segments.transform(mat);
        self.bounding.center = mat.transform_point3(self.bounding.center);
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

        for sphere in &self.spheres {
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
        out
    }
}

impl<PCL: PointCloudMarker> Collider<PCL> {
    pub fn new() -> Self {
        Self::default()
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

    /// Check if `query` sphere overlaps any per-shape broadphase sphere
    /// in this collider (bounded shapes only).
    fn broad_overlaps_any(&self, query: &Sphere) -> bool {
        self.spheres.broad.any_collides_sphere(query)
            || self.capsules.broad.any_collides_sphere(query)
            || self.cuboids.broad.any_collides_sphere(query)
            || self.cylinders.broad.any_collides_sphere(query)
            || self.polygons.broad.any_collides_sphere(query)
            || self.polytopes.broad.any_collides_sphere(query)
            || self.points.broad.any_collides_sphere(query)
            || self.segments.broad.any_collides_sphere(query)
            || self.pointclouds.broad.any_collides_sphere(query)
            || !self.planes.is_empty()
            || !self.lines.is_empty()
            || !self.rays.is_empty()
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
    pub fn spheres(&self) -> &[Sphere] {
        self.spheres.items()
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

        if other.bounding.radius == f32::INFINITY {
            self.bounding.radius = f32::INFINITY;
        } else {
            self.expand_bounding(&other.bounding);
        }
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
                c.spheres.collides_only_broadphase(self)
                    || c.points.collides(self)
                    || c.capsules.collides(self)
                    || c.cuboids.collides(self)
                    || c.cylinders.collides(self)
                    || c.segments.collides(self)
                    || c.polygons.collides(self)
                    || c.polytopes.collides(self)
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.collides(self)
            }
        }
        impl ColliderQuery<Pointcloud> for $ty {
            fn query_collider(&self, c: &Collider<Pointcloud>) -> bool {
                c.spheres.collides_only_broadphase(self)
                    || c.points.collides(self)
                    || c.capsules.collides(self)
                    || c.cuboids.collides(self)
                    || c.cylinders.collides(self)
                    || c.segments.collides(self)
                    || c.polygons.collides(self)
                    || c.polytopes.collides(self)
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.collides(self)
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
                c.spheres.collides_only_broadphase(self)
                    || c.points.collides_only_broadphase(self)
                    || c.capsules.collides(self)
                    || c.cuboids.collides(self)
                    || c.cylinders.collides(self)
                    || c.segments.collides(self)
                    || c.polygons.collides(self)
                    || c.polytopes.collides(self)
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.collides(self)
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
                c.spheres.collides_only_broadphase(self)
                    || c.points.collides_only_broadphase(self)
                    || (c.capsules.collides_only_broadphase(self)
                        && soa::batch::sphere_vs_capsules(self, c.capsules.items()))
                    || (c.cuboids.collides_only_broadphase(self)
                        && soa::batch::sphere_vs_cuboids(self, c.cuboids.items()))
                    || (c.cylinders.collides_only_broadphase(self)
                        && soa::batch::sphere_vs_cylinders(self, c.cylinders.items()))
                    || c.segments.collides(self)
                    || c.polygons.collides(self)
                    || c.polytopes.collides(self)
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.collides(self)
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
                soa::batch::plane_vs_spheres(self, c.spheres.items())
                    || soa::batch::plane_vs_capsules(self, c.capsules.items())
                    || soa::batch::plane_vs_cuboids(self, c.cuboids.items())
                    || soa::batch::plane_vs_cylinders(self, c.cylinders.items())
                    || c.points.iter().any(|x| self.collides(x))
                    || c.segments.iter().any(|x| self.collides(x))
                    || c.polygons.iter().any(|x| self.collides(x))
                    || c.polytopes.iter().any(|x| self.collides(x))
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.iter().any(|x| self.collides(x))
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
                $batch_fn(self, c.spheres.items())
                    || c.capsules.iter().any(|x| self.collides(x))
                    || c.cuboids.iter().any(|x| self.collides(x))
                    || c.cylinders.iter().any(|x| self.collides(x))
                    || c.segments.iter().any(|x| self.collides(x))
                    || c.polygons.iter().any(|x| self.collides(x))
                    || c.polytopes.iter().any(|x| self.collides(x))
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.points.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.iter().any(|x| self.collides(x))
            }
        }
    };
}
impl_collider_query_line_like!(Line, soa::batch::line_vs_spheres, NoPcl);
impl_collider_query_line_like!(Line, soa::batch::line_vs_spheres, Pointcloud);
impl_collider_query_line_like!(Ray, soa::batch::ray_vs_spheres, NoPcl);
impl_collider_query_line_like!(Ray, soa::batch::ray_vs_spheres, Pointcloud);

/// LineSegment: SIMD batch for spheres, broadphase for bounded types.
macro_rules! impl_collider_query_segment {
    ($pcl:ty) => {
        impl ColliderQuery<$pcl> for LineSegment {
            fn query_collider(&self, c: &Collider<$pcl>) -> bool {
                (c.spheres.collides_only_broadphase(self)
                    && soa::batch::segment_vs_spheres(self, c.spheres.items()))
                    || c.capsules.collides(self)
                    || c.cuboids.collides(self)
                    || c.cylinders.collides(self)
                    || c.segments.collides(self)
                    || c.polygons.collides(self)
                    || c.polytopes.collides(self)
                    || c.planes.iter().any(|x| self.collides(x))
                    || c.points.iter().any(|x| self.collides(x))
                    || c.lines.iter().any(|x| self.collides(x))
                    || c.rays.iter().any(|x| self.collides(x))
                    || c.pointclouds.collides(self)
            }
        }
    };
}
impl_collider_query_segment!(NoPcl);
impl_collider_query_segment!(Pointcloud);

/// Pointcloud: can only query NoPcl colliders.
impl ColliderQuery<NoPcl> for Pointcloud {
    fn query_collider(&self, c: &Collider<NoPcl>) -> bool {
        c.spheres.collides_only_broadphase(self)
            || c.points.collides(self)
            || c.capsules.collides(self)
            || c.cuboids.collides(self)
            || c.cylinders.collides(self)
            || c.segments.collides(self)
            || c.polygons.collides(self)
            || c.polytopes.collides(self)
            || c.planes.iter().any(|x| self.collides(x))
            || c.lines.iter().any(|x| self.collides(x))
            || c.rays.iter().any(|x| self.collides(x))
            || c.pointclouds.collides(self)
    }
}

macro_rules! impl_collider_collides_other {
    ($pcl:ty) => {
        impl Collider<$pcl> {
            #[must_use]
            pub fn collides_other(&self, other: &Collider<$pcl>) -> bool {
                // Top-level broadphase: bounding spheres of both colliders
                if !self.bounding.collides(&other.bounding) {
                    return false;
                }
                // Check each collider's bounding sphere against the other's
                // per-shape broadphase SoA — if neither can see the other,
                // no individual shape pair can collide.
                if !other.broad_overlaps_any(&self.bounding)
                    && !self.broad_overlaps_any(&other.bounding)
                {
                    return false;
                }

                other.spheres.iter().any(|x| self.collides(x))
                    || other.points.iter().any(|x| self.collides(x))
                    || other.capsules.iter().any(|x| self.collides(x))
                    || other.cuboids.iter().any(|x| self.collides(x))
                    || other.cylinders.iter().any(|x| self.collides(x))
                    || other.segments.iter().any(|x| self.collides(x))
                    || other.polygons.iter().any(|x| self.collides(x))
                    || other.polytopes.iter().any(|x| self.collides(x))
                    || other.planes.iter().any(|x| self.collides(x))
                    || other.lines.iter().any(|x| self.collides(x))
                    || other.rays.iter().any(|x| self.collides(x))
            }
        }
    };
}

impl_collider_collides_other!(NoPcl);
impl_collider_collides_other!(Pointcloud);
