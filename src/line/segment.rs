use glam::Vec3;

use inherent::inherent;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::plane::{ConvexPolygon, Plane};
use crate::sphere::Sphere;
use crate::{Bounded, Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

const T_MIN: f32 = 0.0;
const T_MAX: f32 = 1.0;

/// A line segment from `p1` to `p1 + dir`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSegment {
    pub p1: Vec3,
    pub dir: Vec3,
    pub(crate) rdv: f32,
}

impl LineSegment {
    pub fn new(p1: Vec3, p2: Vec3) -> Self {
        let dir = p2 - p1;
        let len_sq = dir.dot(dir);
        Self {
            p1,
            dir,
            rdv: if len_sq > f32::EPSILON {
                1.0 / len_sq
            } else {
                0.0
            },
        }
    }

    #[inline]
    pub fn p2(&self) -> Vec3 {
        self.p1 + self.dir
    }

    #[inline]
    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        let center = self.p1 + 0.5 * self.dir;
        let half_len = self.dir.length() * 0.5;
        (center, half_len)
    }
}

#[inherent]
impl Bounded for LineSegment {
    pub fn broadphase(&self) -> Sphere {
        let (center, radius) = self.bounding_sphere();
        Sphere::new(center, radius)
    }

    pub fn obb(&self) -> Cuboid {
        let center = self.p1 + 0.5 * self.dir;
        let len = self.dir.length();
        if len < f32::EPSILON {
            return Cuboid::from_aabb(self.p1, self.p1);
        }
        let ax0 = self.dir / len;
        let ref_vec = if ax0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let ax1 = ax0.cross(ref_vec).normalize();
        let ax2 = ax0.cross(ax1);
        Cuboid::new(center, [ax0, ax1, ax2], [len * 0.5, 0.0, 0.0])
    }

    pub fn aabb(&self) -> Cuboid {
        let p2 = self.p1 + self.dir;
        Cuboid::from_aabb(self.p1.min(p2), self.p1.max(p2))
    }
}

#[inherent]
impl Scalable for LineSegment {
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
impl Transformable for LineSegment {
    pub fn translate(&mut self, offset: Vec3) {
        self.p1 += offset;
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.p1 = mat * self.p1;
        self.dir = mat * self.dir;
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.p1 = quat * self.p1;
        self.dir = quat * self.dir;
    }

    pub fn transform(&mut self, mat: glam::Affine3) {
        let p2 = mat.transform_point3(self.p1 + self.dir);
        self.p1 = mat.transform_point3(self.p1);
        self.dir = p2 - self.p1;
        let len_sq = self.dir.dot(self.dir);
        self.rdv = if len_sq > f32::EPSILON {
            1.0 / len_sq
        } else {
            0.0
        };
    }
}

// --- LineSegment–Stretch ---

#[derive(Debug, Clone)]
pub enum LineSegmentStretch {
    Parallel(LineSegment),
    Polygon(ConvexPolygon),
}

impl Stretchable for LineSegment {
    type Output = LineSegmentStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let cross = self.dir.cross(translation);
        if cross.length_squared() < 1e-10 {
            // Parallel: extend the segment
            let proj = translation.dot(self.dir);
            let (new_p1, new_p2) = if proj >= 0.0 {
                (self.p1, self.p1 + self.dir + translation)
            } else {
                (self.p1 + translation, self.p1 + self.dir)
            };
            return LineSegmentStretch::Parallel(LineSegment::new(new_p1, new_p2));
        }

        // Non-parallel: parallelogram
        let normal = cross.normalize();
        let up = if normal.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u_axis = normal.cross(up).normalize();
        let v_axis = u_axis.cross(normal);

        let center = self.p1 + (self.dir + translation) * 0.5;
        let corners = [
            self.p1,
            self.p1 + self.dir,
            self.p1 + self.dir + translation,
            self.p1 + translation,
        ];

        let verts_2d: Vec<[f32; 2]> = corners
            .iter()
            .map(|&c| {
                let d = c - center;
                [d.dot(u_axis), d.dot(v_axis)]
            })
            .collect();

        LineSegmentStretch::Polygon(ConvexPolygon::with_axes(
            center, normal, u_axis, v_axis, verts_2d,
        ))
    }
}

// --- LineSegment–Sphere ---

#[inline]
fn segment_sphere_collides(seg: &LineSegment, sphere: &Sphere) -> bool {
    super::line_sphere_collides(seg.p1, seg.dir, seg.rdv, sphere, T_MIN, T_MAX)
}

impl Collides<Sphere> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        segment_sphere_collides(self, sphere)
    }
}

impl Collides<LineSegment> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}

// --- LineSegment–Capsule ---

impl Collides<Capsule> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        super::line_capsule_collides(self.p1, self.dir, capsule, T_MIN, T_MAX)
    }
}

impl Collides<LineSegment> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}

// --- LineSegment–Cuboid ---

impl Collides<Cuboid> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        super::line_cuboid_collides(self.p1, self.dir, cuboid, T_MIN, T_MAX)
    }
}

impl Collides<LineSegment> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}

// --- LineSegment–ConvexPolytope ---

impl Collides<ConvexPolytope> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        super::line_polytope_collides(
            self.p1,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            T_MIN,
            T_MAX,
        )
    }
}

impl Collides<LineSegment> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polytope: &ArrayConvexPolytope<P, V>) -> bool {
        super::line_polytope_collides(
            self.p1,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            T_MIN,
            T_MAX,
        )
    }
}

impl<const P: usize, const V: usize> Collides<LineSegment> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}

// --- LineSegment–InfinitePlane ---

impl Collides<Plane> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        super::line_infinite_plane_collides(self.p1, self.dir, plane, T_MIN, T_MAX)
    }
}

impl Collides<LineSegment> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}

// --- LineSegment–ConvexPolygon ---

impl Collides<ConvexPolygon> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, polygon: &ConvexPolygon) -> bool {
        polygon.parametric_line_dist_sq(self.p1, self.dir, T_MIN, T_MAX) <= 0.0
    }
}

impl Collides<LineSegment> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}
