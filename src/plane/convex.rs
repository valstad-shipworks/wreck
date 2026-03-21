use glam::Vec3;

use inherent::inherent;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use super::Plane;
use super::ref_convex::{
    RefConvexPolygon,
    ref_polygon_sphere_collides, ref_polygon_capsule_collides,
    ref_polygon_cuboid_collides, ref_polygon_polytope_collides,
    ref_polygon_infinite_plane_collides, ref_polygon_polygon_collides,
};
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

/// A bounded convex polygon in 3D: a flat convex shape defined by a center,
/// a normal direction, and a list of 2D vertices (in a local tangent frame)
/// forming a convex boundary.
///
/// The polygon has zero thickness. `normal` must be unit-length.
/// 2D vertices must be in counter-clockwise order when viewed from the
/// normal direction. If provided clockwise, the constructor reverses them.
#[derive(Debug, Clone)]
pub struct ConvexPolygon {
    pub center: Vec3,
    pub normal: Vec3,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub vertices_2d: Vec<[f32; 2]>,
    // Precomputed
    pub(crate) vertices_3d: Vec<Vec3>,
    pub(crate) edge_normals_2d: Vec<[f32; 2]>,
    pub(crate) edge_offsets_2d: Vec<f32>,
    pub(crate) bounding_radius: f32,
}

fn make_tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = normal.cross(up).normalize();
    let v = u.cross(normal);
    (u, v)
}

impl ConvexPolygon {
    pub fn new(center: Vec3, normal: Vec3, vertices_2d: Vec<[f32; 2]>) -> Self {
        let (u, v) = make_tangent_frame(normal);
        Self::with_axes(center, normal, u, v, vertices_2d)
    }

    pub fn with_axes(
        center: Vec3,
        normal: Vec3,
        u_axis: Vec3,
        v_axis: Vec3,
        mut vertices_2d: Vec<[f32; 2]>,
    ) -> Self {
        wreck_assert!(vertices_2d.len() >= 3, "Polygon must have at least 3 vertices");
        wreck_assert!(normal.dot(normal) > f32::EPSILON, "Polygon normal must be non-zero");
        // Ensure CCW winding (positive signed area)
        let mut area2 = 0.0f32;
        let n = vertices_2d.len();
        for i in 0..n {
            let j = (i + 1) % n;
            area2 += vertices_2d[i][0] * vertices_2d[j][1]
                - vertices_2d[j][0] * vertices_2d[i][1];
        }
        if area2 < 0.0 {
            vertices_2d.reverse();
        }

        let mut poly = Self {
            center,
            normal,
            u_axis,
            v_axis,
            vertices_2d,
            vertices_3d: Vec::new(),
            edge_normals_2d: Vec::new(),
            edge_offsets_2d: Vec::new(),
            bounding_radius: 0.0,
        };
        poly.recompute_edges();
        poly.recompute_3d();
        poly
    }

    fn recompute_3d(&mut self) {
        self.vertices_3d = self
            .vertices_2d
            .iter()
            .map(|v| self.center + self.u_axis * v[0] + self.v_axis * v[1])
            .collect();
        self.bounding_radius = self
            .vertices_2d
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0f32, f32::max);
    }

    fn recompute_edges(&mut self) {
        let n = self.vertices_2d.len();
        self.edge_normals_2d = Vec::with_capacity(n);
        self.edge_offsets_2d = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.vertices_2d[j][0] - self.vertices_2d[i][0];
            let dy = self.vertices_2d[j][1] - self.vertices_2d[i][1];
            // Outward normal for CCW polygon: (dy, -dx)
            let len = (dx * dx + dy * dy).sqrt();
            let nx = dy / len;
            let ny = -dx / len;
            let offset = nx * self.vertices_2d[i][0] + ny * self.vertices_2d[i][1];
            self.edge_normals_2d.push([nx, ny]);
            self.edge_offsets_2d.push(offset);
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> RefConvexPolygon<'_> {
        RefConvexPolygon::from_heap(self)
    }

    #[inline]
    pub(crate) fn point_dist_sq(&self, point: Vec3) -> f32 {
        self.as_ref().point_dist_sq(point)
    }

    pub(crate) fn parametric_line_dist_sq(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> f32 {
        self.as_ref().parametric_line_dist_sq(origin, dir, t_min, t_max)
    }

    #[inline]
    pub(crate) fn bounding_sphere(&self) -> (Vec3, f32) {
        (self.center, self.bounding_radius)
    }
}

#[inherent]
impl Scalable for ConvexPolygon {
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.vertices_2d {
            v[0] *= factor;
            v[1] *= factor;
        }
        self.recompute_edges();
        self.recompute_3d();
    }
}

#[inherent]
impl Transformable for ConvexPolygon {
    pub fn translate(&mut self, offset: Vec3) {
        self.center += offset;
        for v in &mut self.vertices_3d {
            *v += offset;
        }
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.center = mat * self.center;
        self.normal = mat * self.normal;
        self.u_axis = mat * self.u_axis;
        self.v_axis = mat * self.v_axis;
        for v in &mut self.vertices_3d {
            *v = mat * *v;
        }
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.center = quat * self.center;
        self.normal = quat * self.normal;
        self.u_axis = quat * self.u_axis;
        self.v_axis = quat * self.v_axis;
        for v in &mut self.vertices_3d {
            *v = quat * *v;
        }
    }

    pub fn transform(&mut self, mat: glam::Affine3) {
        self.center = mat.transform_point3(self.center);
        self.normal = mat.matrix3 * self.normal;
        self.u_axis = mat.matrix3 * self.u_axis;
        self.v_axis = mat.matrix3 * self.v_axis;
        for v in &mut self.vertices_3d {
            *v = mat.transform_point3(*v);
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConvexPolygonStretch {
    InPlane(ConvexPolygon),
    OutOfPlane(ConvexPolytope),
}

impl Stretchable for ConvexPolygon {
    type Output = ConvexPolygonStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let normal_comp = translation.dot(self.normal);

        if normal_comp.abs() < 1e-6 {
            // Translation is in the polygon's plane.
            // Minkowski sum of 2D convex polygon with a 2D segment.
            let tu = translation.dot(self.u_axis);
            let tv = translation.dot(self.v_axis);

            // For each vertex, produce v and v+[tu,tv], then compute 2D convex hull.
            let n = self.vertices_2d.len();
            let mut points = Vec::with_capacity(n * 2);
            for v in &self.vertices_2d {
                points.push([v[0], v[1]]);
                points.push([v[0] + tu, v[1] + tv]);
            }

            let hull = convex_hull_2d(&mut points);
            return ConvexPolygonStretch::InPlane(ConvexPolygon::with_axes(
                self.center,
                self.normal,
                self.u_axis,
                self.v_axis,
                hull,
            ));
        }

        // Out-of-plane: build a prism (ConvexPolytope).
        // Vertices: original polygon vertices + translated copies.
        let n = self.vertices_3d.len();
        let mut vertices = Vec::with_capacity(n * 2);
        vertices.extend_from_slice(&self.vertices_3d);
        for &v in &self.vertices_3d {
            vertices.push(v + translation);
        }

        // Planes:
        // 1. Top face: normal · x <= normal · center (original polygon plane)
        // 2. Bottom face: (-normal) · x <= (-normal) · (center + translation)
        // 3. Side faces: one per edge, extruded along translation
        let mut planes = Vec::with_capacity(n + 2);

        let d_top = self.normal.dot(self.center);
        // Orient top/bottom so the half-space contains the prism
        if normal_comp > 0.0 {
            // Translation goes in normal direction: top face is the translated one
            planes.push((self.normal, d_top + normal_comp));
            planes.push((-self.normal, -d_top));
        } else {
            planes.push((self.normal, d_top));
            planes.push((-self.normal, -d_top - normal_comp));
        }

        // Side faces
        for i in 0..n {
            let j = (i + 1) % n;
            let edge = self.vertices_3d[j] - self.vertices_3d[i];
            let side_normal = edge.cross(translation);
            let len = side_normal.length();
            if len < 1e-10 {
                continue;
            }
            let side_normal = side_normal / len;
            let d = side_normal.dot(self.vertices_3d[i]);
            // Ensure outward: check that center is on the inside
            let center_proj = side_normal.dot(self.center);
            if center_proj > d {
                planes.push((-side_normal, -d));
            } else {
                planes.push((side_normal, d));
            }
        }

        ConvexPolygonStretch::OutOfPlane(ConvexPolytope::new(planes, vertices))
    }
}

/// 2D convex hull (Andrew's monotone chain). Returns CCW-ordered vertices.
fn convex_hull_2d(points: &mut Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    points.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
    points.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-7 && (a[1] - b[1]).abs() < 1e-7);
    let n = points.len();
    if n <= 2 {
        return points.clone();
    }

    let mut hull = Vec::with_capacity(2 * n);

    // Lower hull
    for i in 0..n {
        while hull.len() >= 2 {
            let a: [f32; 2] = hull[hull.len() - 2];
            let b: [f32; 2] = hull[hull.len() - 1];
            let cross = (b[0] - a[0]) * (points[i][1] - a[1])
                - (b[1] - a[1]) * (points[i][0] - a[0]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(points[i]);
    }

    // Upper hull
    let lower_len = hull.len() + 1;
    for i in (0..n).rev() {
        while hull.len() >= lower_len {
            let a: [f32; 2] = hull[hull.len() - 2];
            let b: [f32; 2] = hull[hull.len() - 1];
            let cross = (b[0] - a[0]) * (points[i][1] - a[1])
                - (b[1] - a[1]) * (points[i][0] - a[0]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(points[i]);
    }

    hull.pop(); // Remove last (duplicate of first)
    hull
}

// ---------------------------------------------------------------------------
// ConvexPolygon – Sphere
// ---------------------------------------------------------------------------

impl Collides<Sphere> for ConvexPolygon {
    #[inline]
    fn collides(&self, sphere: &Sphere) -> bool {
        ref_polygon_sphere_collides(&self.as_ref(), sphere)
    }

    fn collides_many(&self, others: &[Sphere]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_radius,
            others,
            |s| (s.center, s.radius),
            |s| ref_polygon_sphere_collides(&self.as_ref(), s),
        )
    }
}

impl Collides<ConvexPolygon> for Sphere {
    #[inline]
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        ref_polygon_sphere_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ConvexPolygon – Capsule
// ---------------------------------------------------------------------------

impl Collides<Capsule> for ConvexPolygon {
    #[inline]
    fn collides(&self, capsule: &Capsule) -> bool {
        ref_polygon_capsule_collides(&self.as_ref(), capsule)
    }

    fn collides_many(&self, others: &[Capsule]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_radius,
            others,
            |c| c.bounding_sphere(),
            |c| ref_polygon_capsule_collides(&self.as_ref(), c),
        )
    }
}

impl Collides<ConvexPolygon> for Capsule {
    #[inline]
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        ref_polygon_capsule_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ConvexPolygon – Cuboid (SAT)
// ---------------------------------------------------------------------------

impl Collides<Cuboid> for ConvexPolygon {
    #[inline]
    fn collides(&self, cuboid: &Cuboid) -> bool {
        ref_polygon_cuboid_collides(&self.as_ref(), cuboid)
    }

    fn collides_many(&self, others: &[Cuboid]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_radius,
            others,
            |c| (c.center, c.bounding_sphere_radius()),
            |c| ref_polygon_cuboid_collides(&self.as_ref(), c),
        )
    }
}

impl Collides<ConvexPolygon> for Cuboid {
    #[inline]
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        ref_polygon_cuboid_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ConvexPolygon – ConvexPolytope (SAT with face normals)
// ---------------------------------------------------------------------------

impl Collides<ConvexPolytope> for ConvexPolygon {
    #[inline]
    fn collides(&self, polytope: &ConvexPolytope) -> bool {
        ref_polygon_polytope_collides(&self.as_ref(), &polytope.planes, &polytope.vertices, &polytope.obb)
    }

    fn collides_many(&self, others: &[ConvexPolytope]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_radius,
            others,
            |p| (p.obb.center, p.obb.bounding_sphere_radius()),
            |p| self.collides(p),
        )
    }
}

impl Collides<ConvexPolygon> for ConvexPolytope {
    #[inline]
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        ref_polygon_polytope_collides(&polygon.as_ref(), &self.planes, &self.vertices, &self.obb)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for ConvexPolygon {
    #[inline]
    fn collides(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        ref_polygon_polytope_collides(&self.as_ref(), &polytope.planes, &polytope.vertices, &polytope.obb)
    }
}

impl<const P: usize, const V: usize> Collides<ConvexPolygon> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        ref_polygon_polytope_collides(&polygon.as_ref(), &self.planes, &self.vertices, &self.obb)
    }
}

// ---------------------------------------------------------------------------
// ConvexPolygon – InfinitePlane
// ---------------------------------------------------------------------------

impl Collides<Plane> for ConvexPolygon {
    #[inline]
    fn collides(&self, plane: &Plane) -> bool {
        ref_polygon_infinite_plane_collides(&self.as_ref(), plane)
    }
}

impl Collides<ConvexPolygon> for Plane {
    #[inline]
    fn collides(&self, polygon: &ConvexPolygon) -> bool {
        polygon.collides(self)
    }
}

// ---------------------------------------------------------------------------
// ConvexPolygon – ConvexPolygon (SAT)
// ---------------------------------------------------------------------------

impl Collides<ConvexPolygon> for ConvexPolygon {
    fn collides(&self, other: &ConvexPolygon) -> bool {
        ref_polygon_polygon_collides(&self.as_ref(), &other.as_ref())
    }

    fn collides_many(&self, others: &[ConvexPolygon]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_radius,
            others,
            |p| p.bounding_sphere(),
            |p| self.collides(p),
        )
    }
}

