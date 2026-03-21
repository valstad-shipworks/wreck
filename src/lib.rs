pub(crate) mod capsule;
pub(crate) mod cuboid;
pub(crate) mod point;
pub(crate) mod sphere;

pub(crate) mod convex_polytope;
pub(crate) mod line;
pub(crate) mod plane;
pub(crate) mod pointcloud;

mod util;
pub(crate) use util::*;

pub use capsule::Capsule;
pub use convex_polytope::array::ArrayConvexPolytope;
pub use convex_polytope::heap::ConvexPolytope;
pub use cuboid::Cuboid;
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

    fn stretch(&self, translation: glam::Vec3) -> Self::Output;
    #[inline]
    fn stretch_d(&self, translation: glam::DVec3) -> Self::Output {
        self.stretch(translation.as_vec3())
    }
}

pub trait Collides<T>: Sized + Clone
where
    T: Sized + Clone,
{
    fn collides(&self, other: &T) -> bool;
    fn collides_many(&self, others: &[T]) -> bool {
        others.iter().any(|other| self.collides(other))
    }
}

pub trait Bounded: Sized + Clone {
    fn broadphase(&self) -> Sphere;
    fn obb(&self) -> Cuboid;
    fn aabb(&self) -> Cuboid;
}

pub trait CollidesWithEverything<T>:
    Sized
    + Clone
    + Collides<T>
    + Collides<Capsule>
    + Collides<Cuboid>
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
        + Collides<U>
        + Collides<Capsule>
        + Collides<Cuboid>
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

macro_rules! impl_shape_for {
    ($ty:ty, $field:ident) => {
        impl<PCL: PointCloudMarker> ColliderComponent<PCL> for $ty {
            fn add_to_shapes(self, shapes: &mut Collider<PCL>) {
                shapes.$field.push(self);
            }
        }
    };
}

impl_shape_for!(Sphere, spheres);
impl_shape_for!(Capsule, capsules);
impl_shape_for!(Cuboid, cuboids);
impl_shape_for!(ConvexPolytope, polytopes);
impl_shape_for!(Plane, planes);
impl_shape_for!(ConvexPolygon, polygons);
impl_shape_for!(Point, points);
impl_shape_for!(Line, lines);
impl_shape_for!(Ray, rays);
impl_shape_for!(LineSegment, segments);

impl ColliderComponent<Pointcloud> for Pointcloud {
    fn add_to_shapes(self, shapes: &mut Collider<Pointcloud>) {
        shapes.pointclouds.push(self);
    }
}

impl<const P: usize, const V: usize, PCL: PointCloudMarker> ColliderComponent<PCL> for ArrayConvexPolytope<P, V> {
    fn add_to_shapes(self, shapes: &mut Collider<PCL>) {
        shapes.polytopes.push(ConvexPolytope::from(self));
    }
}

#[derive(Debug, Clone, Default)]
pub struct Collider<PCL: PointCloudMarker = Pointcloud> {
    pub capsules: Vec<Capsule>,
    pub cuboids: Vec<Cuboid>,
    pub planes: Vec<Plane>,
    pub polygons: Vec<ConvexPolygon>,
    pub polytopes: Vec<ConvexPolytope>,
    pub points: Vec<Point>,
    pub spheres: Vec<Sphere>,
    pub lines: Vec<Line>,
    pub rays: Vec<Ray>,
    pub segments: Vec<LineSegment>,
    pub pointclouds: Vec<PCL>,
}

impl<PCL: PointCloudMarker> Transformable for Collider<PCL> {
    fn translate(&mut self, offset: glam::Vec3) {
        for capsule in &mut self.capsules {
            capsule.translate(offset);
        }
        for cuboid in &mut self.cuboids {
            cuboid.translate(offset);
        }
        for plane in &mut self.planes {
            plane.translate(offset);
        }
        for polygon in &mut self.polygons {
            polygon.translate(offset);
        }
        for polytope in &mut self.polytopes {
            polytope.translate(offset);
        }
        for point in &mut self.points {
            point.translate(offset);
        }
        for sphere in &mut self.spheres {
            sphere.translate(offset);
        }
        for line in &mut self.lines {
            line.translate(offset);
        }
        for ray in &mut self.rays {
            ray.translate(offset);
        }
        for segment in &mut self.segments {
            segment.translate(offset);
        }
    }

    fn rotate_mat(&mut self, mat: glam::Mat3) {
        for capsule in &mut self.capsules {
            capsule.rotate_mat(mat);
        }
        for cuboid in &mut self.cuboids {
            cuboid.rotate_mat(mat);
        }
        for plane in &mut self.planes {
            plane.rotate_mat(mat);
        }
        for polygon in &mut self.polygons {
            polygon.rotate_mat(mat);
        }
        for polytope in &mut self.polytopes {
            polytope.rotate_mat(mat);
        }
        for point in &mut self.points {
            point.rotate_mat(mat);
        }
        for sphere in &mut self.spheres {
            sphere.rotate_mat(mat);
        }
        for line in &mut self.lines {
            line.rotate_mat(mat);
        }
        for ray in &mut self.rays {
            ray.rotate_mat(mat);
        }
        for segment in &mut self.segments {
            segment.rotate_mat(mat);
        }
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        for capsule in &mut self.capsules {
            capsule.rotate_quat(quat);
        }
        for cuboid in &mut self.cuboids {
            cuboid.rotate_quat(quat);
        }
        for plane in &mut self.planes {
            plane.rotate_quat(quat);
        }
        for polygon in &mut self.polygons {
            polygon.rotate_quat(quat);
        }
        for polytope in &mut self.polytopes {
            polytope.rotate_quat(quat);
        }
        for point in &mut self.points {
            point.rotate_quat(quat);
        }
        for sphere in &mut self.spheres {
            sphere.rotate_quat(quat);
        }
        for line in &mut self.lines {
            line.rotate_quat(quat);
        }
        for ray in &mut self.rays {
            ray.rotate_quat(quat);
        }
        for segment in &mut self.segments {
            segment.rotate_quat(quat);
        }
    }

    fn transform(&mut self, mat: glam::Affine3) {
        for capsule in &mut self.capsules {
            capsule.transform(mat);
        }
        for cuboid in &mut self.cuboids {
            cuboid.transform(mat);
        }
        for plane in &mut self.planes {
            plane.transform(mat);
        }
        for polygon in &mut self.polygons {
            polygon.transform(mat);
        }
        for polytope in &mut self.polytopes {
            polytope.transform(mat);
        }
        for point in &mut self.points {
            point.transform(mat);
        }
        for sphere in &mut self.spheres {
            sphere.transform(mat);
        }
        for line in &mut self.lines {
            line.transform(mat);
        }
        for ray in &mut self.rays {
            ray.transform(mat);
        }
        for segment in &mut self.segments {
            segment.transform(mat);
        }
    }
}

impl<PCL: PointCloudMarker> Scalable for Collider<PCL> {
    fn scale(&mut self, factor: f32) {
        for capsule in &mut self.capsules {
            capsule.scale(factor);
        }
        for cuboid in &mut self.cuboids {
            cuboid.scale(factor);
        }
        for plane in &mut self.planes {
            plane.scale(factor);
        }
        for polygon in &mut self.polygons {
            polygon.scale(factor);
        }
        for polytope in &mut self.polytopes {
            polytope.scale(factor);
        }
        for point in &mut self.points {
            point.scale(factor);
        }
        for sphere in &mut self.spheres {
            sphere.scale(factor);
        }
        for line in &mut self.lines {
            line.scale(factor);
        }
        for ray in &mut self.rays {
            ray.scale(factor);
        }
        for segment in &mut self.segments {
            segment.scale(factor);
        }
    }
}

impl<PCL: PointCloudMarker> Bounded for Collider<PCL> {
    fn broadphase(&self) -> Sphere {
        if !self.planes.is_empty() || !self.lines.is_empty() || !self.rays.is_empty() {
            return Sphere::new(glam::Vec3::ZERO, f32::INFINITY);
        }
        let aabb = self.aabb();
        Sphere::new(aabb.center, aabb.bounding_sphere_radius())
    }

    fn obb(&self) -> Cuboid {
        self.aabb()
    }

    fn aabb(&self) -> Cuboid {
        if !self.planes.is_empty() || !self.lines.is_empty() || !self.rays.is_empty() {
            return Cuboid::new(
                glam::Vec3::ZERO,
                [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z],
                [f32::INFINITY, f32::INFINITY, f32::INFINITY],
            );
        }

        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        let mut has_any = false;

        macro_rules! merge_aabb {
            ($items:expr) => {
                for item in &$items {
                    let bb = item.aabb();
                    let lo = bb.center - glam::Vec3::new(bb.half_extents[0], bb.half_extents[1], bb.half_extents[2]);
                    let hi = bb.center + glam::Vec3::new(bb.half_extents[0], bb.half_extents[1], bb.half_extents[2]);
                    min = min.min(lo);
                    max = max.max(hi);
                    has_any = true;
                }
            };
        }

        merge_aabb!(self.spheres);
        merge_aabb!(self.capsules);
        merge_aabb!(self.cuboids);
        merge_aabb!(self.polytopes);
        merge_aabb!(self.polygons);
        merge_aabb!(self.segments);
        merge_aabb!(self.pointclouds);

        for pt in &self.points {
            min = min.min(pt.0);
            max = max.max(pt.0);
            has_any = true;
        }

        if !has_any {
            return Cuboid::from_aabb(glam::Vec3::ZERO, glam::Vec3::ZERO);
        }

        Cuboid::from_aabb(min, max)
    }
}

impl Stretchable for Collider<NoPcl> {
    type Output = Self;

    fn stretch(&self, translation: glam::Vec3) -> Self::Output {
        let mut out = Collider {
            spheres: Vec::with_capacity(self.spheres.len()),
            capsules: Vec::with_capacity(self.spheres.len() + 4 * self.capsules.len()),
            cuboids: Vec::with_capacity(self.cuboids.len()),
            polytopes: Vec::with_capacity(
                self.capsules.len()
                    + self.cuboids.len()
                    + self.polytopes.len()
                    + self.polygons.len(),
            ),
            planes: Vec::with_capacity(self.planes.len()),
            polygons: Vec::with_capacity(
                self.polygons.len()
                    + self.lines.len()
                    + self.rays.len()
                    + self.segments.len(),
            ),
            points: Vec::new(),
            lines: Vec::with_capacity(self.lines.len()),
            rays: Vec::with_capacity(self.rays.len()),
            segments: Vec::with_capacity(self.points.len() + self.segments.len()),
            pointclouds: Vec::new(),
        };

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

        out
    }
}

impl<PCL: PointCloudMarker> Collider<PCL> {
    pub fn new() -> Self {
        Collider {
            capsules: Vec::new(),
            cuboids: Vec::new(),
            planes: Vec::new(),
            polygons: Vec::new(),
            polytopes: Vec::new(),
            points: Vec::new(),
            spheres: Vec::new(),
            lines: Vec::new(),
            rays: Vec::new(),
            segments: Vec::new(),
            pointclouds: Vec::new(),
        }
    }

    pub fn collides<T: CollidesWithEverything<PCL>>(&self, shape: &T) -> bool {
        false
            || if !&self.capsules.is_empty() { shape.collides_many(&self.capsules) } else { false }
            || if !&self.cuboids.is_empty() { shape.collides_many(&self.cuboids) } else { false }
            || if !&self.planes.is_empty() { shape.collides_many(&self.planes) } else { false }
            || if !&self.polygons.is_empty() { shape.collides_many(&self.polygons) } else { false }
            || if !&self.polytopes.is_empty() { shape.collides_many(&self.polytopes) } else { false }
            || if !&self.points.is_empty() { shape.collides_many(&self.points) } else { false }
            || if !&self.spheres.is_empty() { shape.collides_many(&self.spheres) } else { false }
            || if !&self.lines.is_empty() { shape.collides_many(&self.lines) } else { false }
            || if !&self.rays.is_empty() { shape.collides_many(&self.rays) } else { false }
            || if !&self.segments.is_empty() { shape.collides_many(&self.segments) } else { false }
            || if !&self.pointclouds.is_empty() { shape.collides_many(&self.pointclouds) } else { false }
    }

    #[inline]
    pub fn collides_many<T: CollidesWithEverything<PCL>>(&self, shapes: &[T]) -> bool {
        shapes.iter().any(|shape| self.collides(shape))
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
}

impl Collider<NoPcl> {
    pub fn collides_other(&self, other: &Collider<NoPcl>) -> bool {
        self.collides_many(&other.capsules)
            || self.collides_many(&other.cuboids)
            || self.collides_many(&other.planes)
            || self.collides_many(&other.polygons)
            || self.collides_many(&other.polytopes)
            || self.collides_many(&other.points)
            || self.collides_many(&other.spheres)
            || self.collides_many(&other.lines)
            || self.collides_many(&other.rays)
            || self.collides_many(&other.segments)
    }
}


impl Collider<Pointcloud> {
    pub fn collides_other(&self, other: &Collider<Pointcloud>) -> bool {
        self.collides_many(&other.capsules)
            || self.collides_many(&other.cuboids)
            || self.collides_many(&other.planes)
            || self.collides_many(&other.polygons)
            || self.collides_many(&other.polytopes)
            || self.collides_many(&other.points)
            || self.collides_many(&other.spheres)
            || self.collides_many(&other.lines)
            || self.collides_many(&other.rays)
            || self.collides_many(&other.segments)
    }
}