use glam::Vec3;

use inherent::inherent;

use super::ConvexPolygon;
use super::Plane;
use super::convex::ConvexPolygonStretch;
use super::ref_convex::{
    RefConvexPolygon, ref_polygon_capsule_collides, ref_polygon_cuboid_collides,
    ref_polygon_infinite_plane_collides, ref_polygon_polygon_collides,
    ref_polygon_polytope_collides, ref_polygon_sphere_collides,
};
use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::sphere::Sphere;
use crate::{Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

/// A convex polygon backed by fixed-size arrays, using const generics
/// so it can be constructed and stored in `const` / `static` contexts.
///
/// `V` is the number of vertices (which equals the number of edges).
///
/// The polygon has zero thickness. `normal` must be unit-length.
/// 2D vertices must be in counter-clockwise order when viewed from the
/// normal direction. The const constructor panics if winding is clockwise.
#[derive(Debug, Clone, Copy)]
pub struct ArrayConvexPolygon<const V: usize> {
    pub center: Vec3,
    pub normal: Vec3,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub vertices_2d: [[f32; 2]; V],
    pub(crate) vertices_3d: [Vec3; V],
    pub(crate) edge_normals_2d: [[f32; 2]; V],
    pub(crate) edge_offsets_2d: [f32; V],
    pub(crate) bounding_radius: f32,
}

const fn const_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x;
    let mut i = 0;
    while i < 30 {
        guess = (guess + x / guess) * 0.5;
        i += 1;
    }
    guess
}

impl<const V: usize> ArrayConvexPolygon<V> {
    /// Create a new const convex polygon. This is a `const fn` so it can be
    /// used in `const` and `static` initializers.
    ///
    /// `vertices_2d` must be in CCW winding order. Panics if CW or degenerate.
    pub const fn new(
        center: Vec3,
        normal: Vec3,
        u_axis: Vec3,
        v_axis: Vec3,
        vertices_2d: [[f32; 2]; V],
    ) -> Self {
        debug_assert!(V >= 3, "Polygon must have at least 3 vertices");
        debug_assert!(crate::dot(normal, normal) >= f32::EPSILON, "Normal cannot be zero");

        // Check CCW winding (positive signed area)
        let mut area2 = 0.0f32;
        let mut i = 0;
        while i < V {
            let j = (i + 1) % V;
            area2 += vertices_2d[i][0] * vertices_2d[j][1] - vertices_2d[j][0] * vertices_2d[i][1];
            i += 1;
        }
        debug_assert!(area2 >= 0.0, "Vertices must be in CCW winding order");

        // Compute vertices_3d
        let mut vertices_3d = [Vec3::ZERO; V];
        i = 0;
        while i < V {
            vertices_3d[i] = Vec3::new(
                center.x + u_axis.x * vertices_2d[i][0] + v_axis.x * vertices_2d[i][1],
                center.y + u_axis.y * vertices_2d[i][0] + v_axis.y * vertices_2d[i][1],
                center.z + u_axis.z * vertices_2d[i][0] + v_axis.z * vertices_2d[i][1],
            );
            i += 1;
        }

        // Compute edge normals and offsets
        let mut edge_normals_2d = [[0.0f32; 2]; V];
        let mut edge_offsets_2d = [0.0f32; V];
        i = 0;
        while i < V {
            let j = (i + 1) % V;
            let dx = vertices_2d[j][0] - vertices_2d[i][0];
            let dy = vertices_2d[j][1] - vertices_2d[i][1];
            let len = const_sqrt(dx * dx + dy * dy);
            let nx = dy / len;
            let ny = -dx / len;
            edge_normals_2d[i] = [nx, ny];
            edge_offsets_2d[i] = nx * vertices_2d[i][0] + ny * vertices_2d[i][1];
            i += 1;
        }

        // Compute bounding radius
        let mut bounding_radius = 0.0f32;
        i = 0;
        while i < V {
            let r = const_sqrt(
                vertices_2d[i][0] * vertices_2d[i][0] + vertices_2d[i][1] * vertices_2d[i][1],
            );
            if r > bounding_radius {
                bounding_radius = r;
            }
            i += 1;
        }

        Self {
            center,
            normal,
            u_axis,
            v_axis,
            vertices_2d,
            vertices_3d,
            edge_normals_2d,
            edge_offsets_2d,
            bounding_radius,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> RefConvexPolygon<'_> {
        RefConvexPolygon::from_array(self)
    }

    /// Convert to a heap-allocated `ConvexPolygon`.
    pub fn to_heap(&self) -> ConvexPolygon {
        ConvexPolygon {
            center: self.center,
            normal: self.normal,
            u_axis: self.u_axis,
            v_axis: self.v_axis,
            vertices_2d: self.vertices_2d.to_vec(),
            vertices_3d: self.vertices_3d.to_vec(),
            edge_normals_2d: self.edge_normals_2d.to_vec(),
            edge_offsets_2d: self.edge_offsets_2d.to_vec(),
            bounding_radius: self.bounding_radius,
        }
    }
}

#[inherent]
impl<const V: usize> Scalable for ArrayConvexPolygon<V> {
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.vertices_2d {
            v[0] *= factor;
            v[1] *= factor;
        }
        // Recompute edges
        for i in 0..V {
            let j = (i + 1) % V;
            let dx = self.vertices_2d[j][0] - self.vertices_2d[i][0];
            let dy = self.vertices_2d[j][1] - self.vertices_2d[i][1];
            let len = (dx * dx + dy * dy).sqrt();
            let nx = dy / len;
            let ny = -dx / len;
            self.edge_normals_2d[i] = [nx, ny];
            self.edge_offsets_2d[i] = nx * self.vertices_2d[i][0] + ny * self.vertices_2d[i][1];
        }
        // Recompute 3d
        for i in 0..V {
            self.vertices_3d[i] = self.center
                + self.u_axis * self.vertices_2d[i][0]
                + self.v_axis * self.vertices_2d[i][1];
        }
        self.bounding_radius = self
            .vertices_2d
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0f32, f32::max);
    }
}

#[inherent]
impl<const V: usize> Transformable for ArrayConvexPolygon<V> {
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

impl<const V: usize> Stretchable for ArrayConvexPolygon<V> {
    type Output = ConvexPolygonStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        self.to_heap().stretch(translation)
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – Sphere
// ---------------------------------------------------------------------------

impl<const V: usize> Collides<Sphere> for ArrayConvexPolygon<V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        ref_polygon_sphere_collides(&self.as_ref(), sphere)
    }
}

impl<const V: usize> Collides<ArrayConvexPolygon<V>> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_sphere_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – Capsule
// ---------------------------------------------------------------------------

impl<const V: usize> Collides<Capsule> for ArrayConvexPolygon<V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        ref_polygon_capsule_collides(&self.as_ref(), capsule)
    }
}

impl<const V: usize> Collides<ArrayConvexPolygon<V>> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_capsule_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – Cuboid
// ---------------------------------------------------------------------------

impl<const V: usize> Collides<Cuboid> for ArrayConvexPolygon<V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        ref_polygon_cuboid_collides(&self.as_ref(), cuboid)
    }
}

impl<const V: usize> Collides<ArrayConvexPolygon<V>> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_cuboid_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – ConvexPolytope
// ---------------------------------------------------------------------------

impl<const V: usize> Collides<ConvexPolytope> for ArrayConvexPolygon<V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        ref_polygon_polytope_collides(
            &self.as_ref(),
            &polytope.planes,
            &polytope.vertices,
            &polytope.obb,
        )
    }
}

impl<const V: usize> Collides<ArrayConvexPolygon<V>> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_polytope_collides(&polygon.as_ref(), &self.planes, &self.vertices, &self.obb)
    }
}

impl<const V: usize, const P: usize, const PV: usize> Collides<ArrayConvexPolytope<P, PV>>
    for ArrayConvexPolygon<V>
{
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ArrayConvexPolytope<P, PV>) -> bool {
        ref_polygon_polytope_collides(
            &self.as_ref(),
            &polytope.planes,
            &polytope.vertices,
            &polytope.obb,
        )
    }
}

impl<const V: usize, const P: usize, const PV: usize> Collides<ArrayConvexPolygon<V>>
    for ArrayConvexPolytope<P, PV>
{
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_polytope_collides(&polygon.as_ref(), &self.planes, &self.vertices, &self.obb)
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – InfinitePlane
// ---------------------------------------------------------------------------

impl<const V: usize> Collides<Plane> for ArrayConvexPolygon<V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        ref_polygon_infinite_plane_collides(&self.as_ref(), plane)
    }
}

impl<const V: usize> Collides<ArrayConvexPolygon<V>> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_infinite_plane_collides(&polygon.as_ref(), self)
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – ConvexPolygon (cross-type)
// ---------------------------------------------------------------------------

impl<const V: usize> Collides<ConvexPolygon> for ArrayConvexPolygon<V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ConvexPolygon) -> bool {
        ref_polygon_polygon_collides(&self.as_ref(), &other.as_ref())
    }
}

impl<const V: usize> Collides<ArrayConvexPolygon<V>> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolygon<V>) -> bool {
        ref_polygon_polygon_collides(&self.as_ref(), &other.as_ref())
    }
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon – ArrayConvexPolygon
// ---------------------------------------------------------------------------

impl<const V1: usize, const V2: usize> Collides<ArrayConvexPolygon<V2>> for ArrayConvexPolygon<V1> {
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolygon<V2>) -> bool {
        ref_polygon_polygon_collides(&self.as_ref(), &other.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_SQUARE: ArrayConvexPolygon<4> = ArrayConvexPolygon::new(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );

    #[test]
    fn const_construction() {
        // Verify const construction works (this is the main point)
        static _STATIC_SQUARE: ArrayConvexPolygon<4> = UNIT_SQUARE;
        assert_eq!(UNIT_SQUARE.center, Vec3::ZERO);
        assert_eq!(UNIT_SQUARE.normal, Vec3::Y);
    }

    #[test]
    fn const_sphere_hit() {
        let s = Sphere::new(Vec3::new(0.0, 0.3, 0.0), 0.5);
        assert!(UNIT_SQUARE.collides(&s));
        assert!(s.collides(&UNIT_SQUARE));
    }

    #[test]
    fn const_sphere_miss() {
        let s = Sphere::new(Vec3::new(0.0, 2.0, 0.0), 0.5);
        assert!(!UNIT_SQUARE.collides(&s));
    }

    #[test]
    fn const_sphere_miss_lateral() {
        let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5);
        assert!(!UNIT_SQUARE.collides(&s));
    }

    #[test]
    fn const_capsule_through() {
        let c = Capsule::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.1);
        assert!(UNIT_SQUARE.collides(&c));
        assert!(c.collides(&UNIT_SQUARE));
    }

    #[test]
    fn const_capsule_miss() {
        let c = Capsule::new(Vec3::new(-1.0, 2.0, 0.0), Vec3::new(1.0, 2.0, 0.0), 0.1);
        assert!(!UNIT_SQUARE.collides(&c));
    }

    #[test]
    fn const_cuboid_overlapping() {
        let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [0.5, 0.5, 0.5]);
        assert!(UNIT_SQUARE.collides(&c));
        assert!(c.collides(&UNIT_SQUARE));
    }

    #[test]
    fn const_cuboid_miss() {
        let c = Cuboid::new(
            Vec3::new(0.0, 3.0, 0.0),
            [Vec3::X, Vec3::Y, Vec3::Z],
            [1.0, 1.0, 1.0],
        );
        assert!(!UNIT_SQUARE.collides(&c));
    }

    #[test]
    fn const_infinite_plane() {
        let ip = Plane::new(Vec3::Y, -0.5);
        assert!(!UNIT_SQUARE.collides(&ip));
        assert!(!ip.collides(&UNIT_SQUARE));

        let ip2 = Plane::new(Vec3::Y, 0.5);
        assert!(UNIT_SQUARE.collides(&ip2));
        assert!(ip2.collides(&UNIT_SQUARE));
    }

    #[test]
    fn const_polytope() {
        let cube = ConvexPolytope::new(
            vec![
                (Vec3::X, 0.5),
                (Vec3::NEG_X, 0.5),
                (Vec3::Y, 0.5),
                (Vec3::NEG_Y, 0.5),
                (Vec3::Z, 0.5),
                (Vec3::NEG_Z, 0.5),
            ],
            vec![
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(-0.5, -0.5, 0.5),
                Vec3::new(0.5, 0.5, -0.5),
                Vec3::new(-0.5, 0.5, -0.5),
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(-0.5, 0.5, 0.5),
                Vec3::new(0.5, -0.5, -0.5),
            ],
        );
        assert!(UNIT_SQUARE.collides(&cube));
        assert!(cube.collides(&UNIT_SQUARE));
    }

    #[test]
    fn const_vs_heap_polygon() {
        let heap = ConvexPolygon::with_axes(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::X,
            Vec3::NEG_Z,
            vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        );
        assert!(UNIT_SQUARE.collides(&heap));
        assert!(heap.collides(&UNIT_SQUARE));
    }

    #[test]
    fn const_vs_heap_polygon_separated() {
        let heap = ConvexPolygon::with_axes(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::Y,
            Vec3::X,
            Vec3::NEG_Z,
            vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        );
        assert!(!UNIT_SQUARE.collides(&heap));
    }

    #[test]
    fn const_vs_const_polygon() {
        const OTHER: ArrayConvexPolygon<4> = ArrayConvexPolygon::new(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::X,
            Vec3::NEG_Z,
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        );
        assert!(UNIT_SQUARE.collides(&OTHER));
    }

    #[test]
    fn const_triangle() {
        const TRI: ArrayConvexPolygon<3> = ArrayConvexPolygon::new(
            Vec3::ZERO,
            Vec3::Y,
            Vec3::X,
            Vec3::NEG_Z,
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        );
        let s = Sphere::new(Vec3::new(0.0, 0.3, 0.0), 0.5);
        assert!(TRI.collides(&s));
    }

    #[test]
    fn const_translate() {
        let mut poly = UNIT_SQUARE;
        poly.translate(Vec3::new(0.0, 5.0, 0.0));
        let s = Sphere::new(Vec3::ZERO, 0.5);
        assert!(!poly.collides(&s));
    }

    #[test]
    fn const_scale() {
        let mut poly = UNIT_SQUARE;
        poly.scale(2.0);
        let s = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 0.1);
        assert!(poly.collides(&s));
    }

    #[test]
    fn const_matches_heap() {
        // Verify array and heap produce identical collision results
        let heap = UNIT_SQUARE.to_heap();
        let test_spheres = [
            Sphere::new(Vec3::new(0.0, 0.3, 0.0), 0.5),
            Sphere::new(Vec3::new(0.0, 2.0, 0.0), 0.5),
            Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5),
            Sphere::new(Vec3::new(1.2, 0.0, 0.0), 0.5),
            Sphere::new(Vec3::new(1.1, 0.1, -1.1), 0.3),
        ];
        for s in &test_spheres {
            assert_eq!(UNIT_SQUARE.collides(s), heap.collides(s));
        }
    }
}
