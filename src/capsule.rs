use glam::Vec3;

use inherent::inherent;

use crate::ConvexPolytope;

use crate::Stretchable;
use crate::wreck_assert;

use crate::Cuboid;
use crate::sphere::Sphere;
use crate::{Bounded, Collides, Scalable, Transformable};

#[derive(Debug, Clone, Copy)]
pub struct Capsule {
    pub p1: Vec3,
    pub dir: Vec3,
    pub radius: f32,
    pub rdv: f32,
    pub z_aligned: bool,
}

impl Capsule {
    pub const fn new(p1: Vec3, p2: Vec3, radius: f32) -> Self {
        wreck_assert!(radius >= 0.0, "Capsule radius must be non-negative");
        let dir = Vec3::new(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
        let len_sq = crate::dot(dir, dir);
        let rdv = if len_sq > f32::EPSILON {
            1.0 / len_sq
        } else {
            0.0
        };
        let z_aligned = dir.x == 0.0 && dir.y == 0.0;
        Self {
            p1,
            dir,
            radius,
            rdv,
            z_aligned,
        }
    }

    #[inline]
    pub fn p2(&self) -> Vec3 {
        self.p1 + self.dir
    }

    /// Closest point on capsule segment to a given point.
    #[inline]
    pub fn closest_point_to(&self, point: Vec3) -> Vec3 {
        if self.z_aligned {
            return self.closest_point_to_z(point);
        }
        let diff = point - self.p1;
        let t = (diff.dot(self.dir) * self.rdv).clamp(0.0, 1.0);
        self.p1 + self.dir * t
    }

    /// Z-aligned fast path: only project onto Z component.
    #[inline]
    fn closest_point_to_z(&self, point: Vec3) -> Vec3 {
        let dz = point.z - self.p1.z;
        let t = (dz * self.dir.z * self.rdv).clamp(0.0, 1.0);
        Vec3::new(self.p1.x, self.p1.y, self.p1.z + self.dir.z * t)
    }

    #[inline]
    pub fn bounding_sphere(&self) -> (Vec3, f32) {
        let center = self.p1 + 0.5 * self.dir;
        let half_len = if self.z_aligned {
            self.dir.z.abs() * 0.5
        } else {
            self.dir.length() * 0.5
        };
        (center, half_len + self.radius)
    }
}

#[inherent]
impl Bounded for Capsule {
    pub fn broadphase(&self) -> Sphere {
        let (center, radius) = self.bounding_sphere();
        Sphere::new(center, radius)
    }

    pub fn obb(&self) -> Cuboid {
        let center = self.p1 + 0.5 * self.dir;
        let dir_len = self.dir.length();
        if dir_len < f32::EPSILON {
            // Degenerate capsule: just a sphere
            return Cuboid::new(
                center,
                [Vec3::X, Vec3::Y, Vec3::Z],
                [self.radius, self.radius, self.radius],
            );
        }
        let ax0 = self.dir / dir_len;
        // Build perpendicular axes
        let ref_vec = if ax0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let ax1 = ax0.cross(ref_vec).normalize();
        let ax2 = ax0.cross(ax1);
        Cuboid::new(
            center,
            [ax0, ax1, ax2],
            [dir_len * 0.5 + self.radius, self.radius, self.radius],
        )
    }

    pub fn aabb(&self) -> Cuboid {
        let p2 = self.p1 + self.dir;
        let min = self.p1.min(p2) - Vec3::splat(self.radius);
        let max = self.p1.max(p2) + Vec3::splat(self.radius);
        Cuboid::from_aabb(min, max)
    }
}

#[inherent]
impl Scalable for Capsule {
    pub fn scale(&mut self, factor: f32) {
        self.dir *= factor;
        self.radius *= factor;
        let len_sq = self.dir.dot(self.dir);
        self.rdv = if len_sq > f32::EPSILON {
            1.0 / len_sq
        } else {
            0.0
        };
    }
}

#[inherent]
impl Transformable for Capsule {
    pub fn translate(&mut self, offset: Vec3) {
        self.p1 += offset;
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.p1 = mat * self.p1;
        self.dir = mat * self.dir;
        self.z_aligned = self.dir.x == 0.0 && self.dir.y == 0.0;
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.p1 = quat * self.p1;
        self.dir = quat * self.dir;
        self.z_aligned = self.dir.x == 0.0 && self.dir.y == 0.0;
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
        self.z_aligned = self.dir.x == 0.0 && self.dir.y == 0.0;
    }
}

// Sphere-Capsule collision
#[inline]
fn sphere_capsule_collides(sphere: &Sphere, capsule: &Capsule) -> bool {
    let closest = capsule.closest_point_to(sphere.center);
    let d = sphere.center - closest;
    let dist_sq = d.dot(d);
    let rs = sphere.radius + capsule.radius;
    dist_sq <= rs * rs
}

impl Collides<Capsule> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Capsule) -> bool {
        sphere_capsule_collides(self, other)
    }
}

impl Collides<Sphere> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Sphere) -> bool {
        sphere_capsule_collides(other, self)
    }
}

// Capsule-Capsule: closest distance between two line segments (Ericson RTCD)
impl Collides<Capsule> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Capsule) -> bool {
        // Bounding sphere early-out
        if BROADPHASE {
            let (c1, r1) = self.bounding_sphere();
            let (c2, r2) = other.bounding_sphere();
            let d = c1 - c2;
            let max_r = r1 + r2;
            if d.dot(d) > max_r * max_r {
                return false;
            }
        }

        let dist_sq = segment_segment_dist_sq(self.p1, self.dir, other.p1, other.dir);
        let rs = self.radius + other.radius;
        dist_sq <= rs * rs
    }
}

#[inline]
pub(crate) fn segment_segment_dist_sq(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> f32 {
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let eps = f32::EPSILON;

    let (s, t);

    if a <= eps && e <= eps {
        // Both degenerate to points
        s = 0.0;
        t = 0.0;
    } else if a <= eps {
        // First segment degenerates to a point
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= eps {
            // Second segment degenerates to a point
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General case
            let b = d1.dot(d2);
            let denom = a * e - b * b;

            // If segments not parallel, compute closest point on L1 to L2 and clamp
            let mut s_n = if denom.abs() > eps {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Compute point on L2 closest to S1(s)
            let mut t_n = (b * s_n + f) / e;

            // If t_n outside [0,1], clamp and recompute s
            if t_n < 0.0 {
                t_n = 0.0;
                s_n = (-c / a).clamp(0.0, 1.0);
            } else if t_n > 1.0 {
                t_n = 1.0;
                s_n = ((b - c) / a).clamp(0.0, 1.0);
            }

            s = s_n;
            t = t_n;
        }
    }

    let closest1 = p1 + d1 * s;
    let closest2 = p2 + d2 * t;
    let diff = closest1 - closest2;
    diff.dot(diff)
}

#[derive(Debug, Clone)]
pub enum CapsuleStretch {
    Aligned(Capsule),
    Unaligned([Capsule; 4], ConvexPolytope),
}

impl Stretchable for Capsule {
    type Output = CapsuleStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let p2 = self.p2();

        // Degenerate capsule (point) or zero translation: always aligned
        let cross = self.dir.cross(translation);
        if cross.length_squared() < 1e-10 {
            // Translation is parallel to capsule axis (or one/both are zero)
            let proj = translation.dot(self.dir);
            let (new_p1, new_p2) = if proj >= 0.0 {
                (self.p1, p2 + translation)
            } else {
                (self.p1 + translation, p2)
            };
            return CapsuleStretch::Aligned(Capsule::new(new_p1, new_p2, self.radius));
        }

        // Unaligned: 4 edge capsules + convex polytope interior
        let p1t = self.p1 + translation;
        let p2t = p2 + translation;

        let edges = [
            Capsule::new(self.p1, p2, self.radius),
            Capsule::new(p1t, p2t, self.radius),
            Capsule::new(self.p1, p1t, self.radius),
            Capsule::new(p2, p2t, self.radius),
        ];

        // Convex polytope: parallelogram extruded by ±radius in the normal direction
        let n = cross.normalize();
        let rn = self.radius * n;
        let corners = [self.p1, p2, p1t, p2t];
        let vertices: Vec<Vec3> = corners.iter().flat_map(|&c| [c + rn, c - rn]).collect();

        // 6 planes: ±n caps, ±s1 (perp to dir), ±s2 (perp to translation)
        let s1 = n.cross(self.dir).normalize();
        let s2 = n.cross(translation).normalize();
        let normals = [n, -n, s1, -s1, s2, -s2];
        let planes: Vec<(Vec3, f32)> = normals
            .iter()
            .map(|&norm| {
                let d = crate::convex_polytope::max_projection(&vertices, norm);
                (norm, d)
            })
            .collect();

        // Derive OBB: axes are dir_normalized, s1, n; extents computed from vertices
        let dir_n = self.dir.normalize_or_zero();
        let obb_axes = [dir_n, s1, n];
        let mut obb_he = [0.0f32; 3];
        let obb_center = self.p1 + self.dir * 0.5 + translation * 0.5;
        for i in 0..3 {
            let max_p = crate::convex_polytope::max_projection(&vertices, obb_axes[i]);
            let min_p = crate::convex_polytope::min_projection(&vertices, obb_axes[i]);
            obb_he[i] = (max_p - min_p) * 0.5;
        }
        let obb = Cuboid::new(obb_center, obb_axes, obb_he);

        CapsuleStretch::Unaligned(edges, ConvexPolytope::with_obb(planes, vertices, obb))
    }
}
