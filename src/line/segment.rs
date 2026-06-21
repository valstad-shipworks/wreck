use alloc::vec::Vec;

use glam::Vec3;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::line::{LineSegment, rdv};
use crate::plane::{ConvexPolygon, Plane};
use crate::sphere::Sphere;
use crate::{Bounded, Collides, ConvexPolytope, Scalable, Stretchable, Transformable};

const T_MIN: f32 = 0.0;
const T_MAX: f32 = 1.0;

impl Bounded for LineSegment {
    fn broadphase(&self) -> Sphere {
        let center = self.midpoint();
        let radius = (self.end - self.start).length() * 0.5;
        Sphere::new(center, radius)
    }

    fn obb(&self) -> Cuboid {
        let dir = self.dir();
        let center = self.midpoint();
        let len = dir.length();
        if len < f32::EPSILON {
            return Cuboid::from_aabb(self.start, self.start);
        }
        let ax0 = dir / len;
        let ref_vec = if ax0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let ax1 = ax0.cross(ref_vec).normalize();
        let ax2 = ax0.cross(ax1);
        Cuboid::new(center, [ax0, ax1, ax2], [len * 0.5, 0.0, 0.0])
    }

    fn aabb(&self) -> Cuboid {
        Cuboid::from_aabb(self.start.min(self.end), self.start.max(self.end))
    }
}

impl Scalable for LineSegment {
    fn scale(&mut self, factor: f32) {
        self.end = self.start + self.dir() * factor;
    }
}

impl Transformable for LineSegment {
    fn translate(&mut self, offset: glam::Vec3A) {
        self.start = Vec3::from(glam::Vec3A::from(self.start) + offset);
        self.end = Vec3::from(glam::Vec3A::from(self.end) + offset);
    }

    fn rotate_mat(&mut self, mat: glam::Mat3A) {
        self.start = Vec3::from(mat * glam::Vec3A::from(self.start));
        self.end = Vec3::from(mat * glam::Vec3A::from(self.end));
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        self.start = quat * self.start;
        self.end = quat * self.end;
    }

    fn transform(&mut self, mat: glam::Affine3A) {
        self.start = Vec3::from(mat.transform_point3a(glam::Vec3A::from(self.start)));
        self.end = Vec3::from(mat.transform_point3a(glam::Vec3A::from(self.end)));
    }
}

// --- LineSegment–Stretch ---

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LineSegmentStretch {
    Parallel(LineSegment),
    Polygon(ConvexPolygon),
}

impl Stretchable for LineSegment {
    type Output = LineSegmentStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let p1 = self.start;
        let dir = self.dir();
        let cross = dir.cross(translation);
        if cross.length_squared() < 1e-10 {
            // Parallel: extend the segment
            let proj = translation.dot(dir);
            let (new_p1, new_p2) = if proj >= 0.0 {
                (p1, p1 + dir + translation)
            } else {
                (p1 + translation, p1 + dir)
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

        let center = p1 + (dir + translation) * 0.5;
        let corners = [
            p1,
            p1 + dir,
            p1 + dir + translation,
            p1 + translation,
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
    let dir = seg.dir();
    super::line_sphere_collides(seg.start, dir, rdv(dir), sphere, T_MIN, T_MAX)
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
        super::line_capsule_collides(self.start, self.dir(), capsule, T_MIN, T_MAX)
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
        super::line_cuboid_collides(self.start, self.dir(), cuboid, T_MIN, T_MAX)
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
            self.start,
            self.dir(),
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
            self.start,
            self.dir(),
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
        super::line_infinite_plane_collides(self.start, self.dir(), plane, T_MIN, T_MAX)
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
        polygon.parametric_line_dist_sq(self.start, self.dir(), T_MIN, T_MAX) <= 0.0
    }
}

impl Collides<LineSegment> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        seg.test::<BROADPHASE>(self)
    }
}
