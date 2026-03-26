use glam::Vec3;

use inherent::inherent;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::plane::{ConvexPolygon, Plane};
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

const T_MIN: f32 = f32::NEG_INFINITY;
const T_MAX: f32 = f32::INFINITY;

/// An infinite line: `origin + t * dir` for all `t ∈ (-∞, ∞)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub origin: Vec3,
    pub dir: Vec3,
    pub(crate) rdv: f32,
}

impl Line {
    pub fn new(origin: Vec3, dir: Vec3) -> Self {
        wreck_assert!(
            dir.dot(dir) > f32::EPSILON,
            "Line direction must be non-zero"
        );
        let len_sq = dir.dot(dir);
        Self {
            origin,
            dir,
            rdv: if len_sq > f32::EPSILON {
                1.0 / len_sq
            } else {
                0.0
            },
        }
    }

    pub fn from_points(a: Vec3, b: Vec3) -> Self {
        Self::new(a, b - a)
    }
}

#[inherent]
impl Scalable for Line {
    pub fn scale(&mut self, factor: f32) {
        self.dir *= factor;
        let len_sq = self.dir.dot(self.dir);
        self.rdv = if len_sq > f32::EPSILON {
            1.0 / len_sq
        } else {
            0.0
        };
    }
}

#[inherent]
impl Transformable for Line {
    pub fn translate(&mut self, offset: Vec3) {
        self.origin += offset;
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.origin = mat * self.origin;
        self.dir = mat * self.dir;
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.origin = quat * self.origin;
        self.dir = quat * self.dir;
    }

    pub fn transform(&mut self, mat: glam::Affine3) {
        self.origin = mat.transform_point3(self.origin);
        self.dir = mat.matrix3 * self.dir;
    }
}

// --- Line–Stretch ---

const INF: f32 = 1e12;

#[derive(Debug, Clone)]
pub enum LineStretch {
    Parallel(Line),
    Polygon(ConvexPolygon),
}

impl Stretchable for Line {
    type Output = LineStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let cross = self.dir.cross(translation);
        if cross.length_squared() < 1e-10 {
            // Parallel: line is unchanged (it already spans its entire axis)
            return LineStretch::Parallel(*self);
        }

        // Non-parallel: infinite strip approximated with 1e12
        let dir_norm = self.dir.normalize();
        let far_pos = dir_norm * INF;
        let far_neg = dir_norm * -INF;

        let normal = cross.normalize();
        let up = if normal.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u_axis = normal.cross(up).normalize();
        let v_axis = u_axis.cross(normal);

        let corners = [
            self.origin + far_neg,
            self.origin + far_pos,
            self.origin + far_pos + translation,
            self.origin + far_neg + translation,
        ];

        let center = self.origin + translation * 0.5;
        let verts_2d: Vec<[f32; 2]> = corners
            .iter()
            .map(|&c| {
                let d = c - center;
                [d.dot(u_axis), d.dot(v_axis)]
            })
            .collect();

        LineStretch::Polygon(ConvexPolygon::with_axes(
            center, normal, u_axis, v_axis, verts_2d,
        ))
    }
}

// --- Line–Sphere ---

impl Collides<Sphere> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        super::line_sphere_collides(self.origin, self.dir, self.rdv, sphere, T_MIN, T_MAX)
    }
}

impl Collides<Line> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}

// --- Line–Capsule ---

impl Collides<Capsule> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        super::line_capsule_collides(self.origin, self.dir, capsule, T_MIN, T_MAX)
    }
}

impl Collides<Line> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}

// --- Line–Cuboid ---

impl Collides<Cuboid> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        super::line_cuboid_collides(self.origin, self.dir, cuboid, T_MIN, T_MAX)
    }
}

impl Collides<Line> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}

// --- Line–ConvexPolytope ---

impl Collides<ConvexPolytope> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        super::line_polytope_collides(
            self.origin,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            T_MIN,
            T_MAX,
        )
    }
}

impl Collides<Line> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        super::line_polytope_collides(
            self.origin,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            T_MIN,
            T_MAX,
        )
    }
}

impl<const P: usize, const V: usize> Collides<Line> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}

// --- Line–InfinitePlane ---

impl Collides<Plane> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        super::line_infinite_plane_collides(self.origin, self.dir, plane, T_MIN, T_MAX)
    }
}

impl Collides<Line> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}

// --- Line–ConvexPolygon ---

impl Collides<ConvexPolygon> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ConvexPolygon) -> bool {
        polygon.parametric_line_dist_sq(self.origin, self.dir, T_MIN, T_MAX) <= 0.0
    }
}

impl Collides<Line> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line.test::<BROADPHASE>(self)
    }
}
