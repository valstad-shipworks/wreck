use glam::Vec3;
use wide::{CmpLe, f32x8};

use inherent::inherent;

use crate::ConvexPolytope;
use crate::Stretchable;
use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::line::{Line, LineSegment, Ray};
use crate::plane::{ConvexPolygon, Plane};
use crate::point::Point;
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Bounded, Collides, Scalable, Transformable};

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cylinder {
    pub p1: Vec3,
    pub dir: Vec3,
    pub radius: f32,
    pub(crate) rdv: f32,
    pub(crate) z_aligned: bool,
}

impl Cylinder {
    pub const fn new(p1: Vec3, p2: Vec3, radius: f32) -> Self {
        wreck_assert!(radius >= 0.0, "Cylinder radius must be non-negative");
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

    /// Squared distance from a point to the nearest point on the cylinder surface/interior.
    #[inline]
    pub fn point_dist_sq(&self, p: Vec3) -> f32 {
        let w = p - self.p1;
        let t = w.dot(self.dir) * self.rdv;
        let along = self.dir * t;
        let perp = w - along;
        let r_sq = perp.dot(perp);
        let cyl_r_sq = self.radius * self.radius;

        let t_c = t.clamp(0.0, 1.0);
        let t_excess = t - t_c;
        let d_axial_sq = t_excess * t_excess * self.dir.dot(self.dir);

        if r_sq <= cyl_r_sq {
            d_axial_sq
        } else {
            let r = r_sq.sqrt();
            let d_radial = r - self.radius;
            d_radial * d_radial + d_axial_sq
        }
    }

    /// Returns true if the point lies inside or on the cylinder surface.
    #[inline]
    pub(crate) fn contains_point(&self, p: Vec3) -> bool {
        let w = p - self.p1;
        let t = w.dot(self.dir) * self.rdv;
        if t < 0.0 || t > 1.0 {
            return false;
        }
        let along = self.dir * t;
        let perp = w - along;
        perp.dot(perp) <= self.radius * self.radius
    }
}

// ─── Trait impls ─────────────────────────────────────────────────────────────

#[inherent]
impl Bounded for Cylinder {
    pub fn broadphase(&self) -> Sphere {
        let (center, radius) = self.bounding_sphere();
        Sphere::new(center, radius)
    }

    pub fn obb(&self) -> Cuboid {
        let center = self.p1 + 0.5 * self.dir;
        let dir_len = self.dir.length();
        if dir_len < f32::EPSILON {
            return Cuboid::new(
                center,
                [Vec3::X, Vec3::Y, Vec3::Z],
                [self.radius, self.radius, self.radius],
            );
        }
        let ax0 = self.dir / dir_len;
        let ref_vec = if ax0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let ax1 = ax0.cross(ref_vec).normalize();
        let ax2 = ax0.cross(ax1);
        Cuboid::new(
            center,
            [ax0, ax1, ax2],
            [dir_len * 0.5, self.radius, self.radius],
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
impl Scalable for Cylinder {
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
impl Transformable for Cylinder {
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

// ─── Cylinder-Sphere ─────────────────────────────────────────────────────────

#[inline]
fn cylinder_sphere_collides(cyl: &Cylinder, sphere: &Sphere) -> bool {
    let w = sphere.center - cyl.p1;
    let t = w.dot(cyl.dir) * cyl.rdv;
    let along = cyl.dir * t;
    let perp = w - along;
    let r_sq = perp.dot(perp);

    if t >= 0.0 && t <= 1.0 {
        let combined = cyl.radius + sphere.radius;
        r_sq <= combined * combined
    } else {
        let t_c = if t < 0.0 { 0.0 } else { 1.0 };
        let t_excess = t - t_c;
        let dir_sq = cyl.dir.dot(cyl.dir);
        let d_axial_sq = t_excess * t_excess * dir_sq;
        let sr_sq = sphere.radius * sphere.radius;
        let cyl_r_sq = cyl.radius * cyl.radius;

        if r_sq <= cyl_r_sq {
            d_axial_sq <= sr_sq
        } else {
            // sqrt-free: (r - R)^2 + d_axial_sq <= sr^2
            // ⇔ r_sq + R^2 + d_axial_sq - sr^2 <= 2*r*R
            // Let L = r_sq + R^2 + d_axial_sq - sr^2
            // True when L <= 0 OR L^2 <= 4*R^2*r_sq
            let l = r_sq + cyl_r_sq + d_axial_sq - sr_sq;
            l <= 0.0 || l * l <= 4.0 * cyl_r_sq * r_sq
        }
    }
}

impl Collides<Sphere> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Sphere) -> bool {
        cylinder_sphere_collides(self, other)
    }
}

impl Collides<Cylinder> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Cylinder) -> bool {
        cylinder_sphere_collides(other, self)
    }
}

// ─── Cylinder-Point ──────────────────────────────────────────────────────────

impl Collides<Point> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        self.contains_point(point.0)
    }
}

impl Collides<Cylinder> for Point {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        cyl.contains_point(self.0)
    }
}

// ─── Cylinder-Capsule ────────────────────────────────────────────────────────

/// Evaluate cylinder distance at 8 sample points along the capsule axis.
#[inline]
fn cylinder_capsule_collides<const BROADPHASE: bool>(cyl: &Cylinder, capsule: &Capsule) -> bool {
    if BROADPHASE {
        let (bc, br) = cyl.bounding_sphere();
        let (cc, cr) = capsule.bounding_sphere();
        let d = bc - cc;
        let max_r = br + cr;
        if d.dot(d) > max_r * max_r {
            return false;
        }
    }

    let cr_sq = capsule.radius * capsule.radius;

    // Compute key s-values on the capsule axis
    // s_closest: closest approach between the two axis segments
    let r = capsule.p1 - cyl.p1;
    let a = capsule.dir.dot(capsule.dir);
    let e = cyl.dir.dot(cyl.dir);
    let f = cyl.dir.dot(r);
    let eps = f32::EPSILON;

    let s_closest = if a > eps {
        let c_val = capsule.dir.dot(r);
        if e > eps {
            let b = capsule.dir.dot(cyl.dir);
            let denom = a * e - b * b;
            if denom.abs() > eps {
                ((b * f - c_val * e) / denom).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            (-c_val / a).clamp(0.0, 1.0)
        }
    } else {
        0.0
    };

    // s where cylinder axis parameter t = 0 and t = 1
    // t(s) = ((capsule.p1 + s*capsule.dir - cyl.p1) · cyl.dir) * cyl.rdv
    // t(s) = (r·cyl.dir + s * capsule.dir·cyl.dir) * cyl.rdv
    // t = 0 → s = -r·cyl.dir / (capsule.dir·cyl.dir)
    // t = 1 → s = (|cyl.dir|² - r·cyl.dir) / (capsule.dir·cyl.dir)
    let cap_dot_cyl = capsule.dir.dot(cyl.dir);
    let r_dot_cyl = r.dot(cyl.dir);
    let (s_t0, s_t1) = if cap_dot_cyl.abs() > eps {
        let inv = 1.0 / cap_dot_cyl;
        (
            (-r_dot_cyl * inv).clamp(0.0, 1.0),
            ((e - r_dot_cyl) * inv).clamp(0.0, 1.0),
        )
    } else {
        (0.0, 1.0)
    };

    let samples = f32x8::new([0.0, 1.0, s_closest, s_t0, s_t1, 0.25, 0.5, 0.75]);

    // Evaluate point_dist_sq at each sample on capsule axis
    // p(s) = capsule.p1 + s * capsule.dir
    let px = f32x8::splat(capsule.p1.x) + f32x8::splat(capsule.dir.x) * samples;
    let py = f32x8::splat(capsule.p1.y) + f32x8::splat(capsule.dir.y) * samples;
    let pz = f32x8::splat(capsule.p1.z) + f32x8::splat(capsule.dir.z) * samples;

    // w = p - cyl.p1
    let wx = px - f32x8::splat(cyl.p1.x);
    let wy = py - f32x8::splat(cyl.p1.y);
    let wz = pz - f32x8::splat(cyl.p1.z);

    let cdx = f32x8::splat(cyl.dir.x);
    let cdy = f32x8::splat(cyl.dir.y);
    let cdz = f32x8::splat(cyl.dir.z);
    let crdv = f32x8::splat(cyl.rdv);

    // t = w · cyl.dir * rdv
    let t = (wx * cdx + wy * cdy + wz * cdz) * crdv;
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let t_c = t.max(zero).min(one);

    // perp = w - t * cyl.dir (using unclamped t for perpendicular distance)
    let perpx = wx - cdx * t;
    let perpy = wy - cdy * t;
    let perpz = wz - cdz * t;
    let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;

    // d_axial_sq = (t - t_c)^2 * |cyl.dir|^2
    let t_excess = t - t_c;
    let dir_sq = f32x8::splat(e);
    let d_axial_sq = t_excess * t_excess * dir_sq;

    let cyl_r = f32x8::splat(cyl.radius);
    let cyl_r_sq = cyl_r * cyl_r;
    let cap_r_sq = f32x8::splat(cr_sq);

    // Barrel check: t in [0,1] and r_sq <= (cyl.radius + capsule.radius)^2
    let in_barrel = t.simd_le(one) & t.simd_le(one.max(t)) & zero.simd_le(t);
    let combined = cyl_r + f32x8::splat(capsule.radius);
    let barrel_hit = in_barrel & r_sq.simd_le(combined * combined);

    // End cap, radially inside: d_axial_sq <= capsule.radius^2
    let inside_r = r_sq.simd_le(cyl_r_sq);
    let endcap_inside = inside_r & d_axial_sq.simd_le(cap_r_sq);

    // End cap, radially outside: sqrt-free algebraic check
    // (r - R)^2 + d_axial_sq <= cap_r_sq
    // r_sq + R^2 + d_axial_sq - cap_r_sq <= 2*r*R
    // Let L = r_sq + R^2 + d_axial_sq - cap_r_sq
    // L <= 0 OR L^2 <= 4*R^2*r_sq
    let l = r_sq + cyl_r_sq + d_axial_sq - cap_r_sq;
    let four_r_sq = f32x8::splat(4.0) * cyl_r_sq;
    let endcap_outside = l.simd_le(zero) | (l * l).simd_le(four_r_sq * r_sq);

    let hit = barrel_hit | endcap_inside | endcap_outside;
    hit.any()
}

impl Collides<Capsule> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        cylinder_capsule_collides::<BROADPHASE>(self, capsule)
    }
}

impl Collides<Cylinder> for Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        cylinder_capsule_collides::<BROADPHASE>(cyl, self)
    }
}

// ─── Cylinder-Cuboid ─────────────────────────────────────────────────────────

#[inline]
fn cylinder_cuboid_collides<const BROADPHASE: bool>(cyl: &Cylinder, cuboid: &Cuboid) -> bool {
    if BROADPHASE {
        let (bc, br) = cyl.bounding_sphere();
        let d = bc - cuboid.center;
        let max_r = br + cuboid.bounding_sphere_radius();
        if d.dot(d) > max_r * max_r {
            return false;
        }
    }

    // Test 1: Sample cylinder axis at 8 t-values against cuboid (barrel test)
    // Same approach as capsule-cuboid but the hit means barrel collision.
    let p0_world = cyl.p1 - cuboid.center;
    let (p0, dir) = if cuboid.axis_aligned {
        (
            [p0_world.x, p0_world.y, p0_world.z],
            [cyl.dir.x, cyl.dir.y, cyl.dir.z],
        )
    } else {
        (
            [
                p0_world.dot(cuboid.axes[0]),
                p0_world.dot(cuboid.axes[1]),
                p0_world.dot(cuboid.axes[2]),
            ],
            [
                cyl.dir.dot(cuboid.axes[0]),
                cyl.dir.dot(cuboid.axes[1]),
                cyl.dir.dot(cuboid.axes[2]),
            ],
        )
    };
    let he = cuboid.half_extents;
    let rs_sq = cyl.radius * cyl.radius;

    let inv_dir = [
        if dir[0].abs() > f32::EPSILON {
            1.0 / dir[0]
        } else {
            f32::MAX
        },
        if dir[1].abs() > f32::EPSILON {
            1.0 / dir[1]
        } else {
            f32::MAX
        },
        if dir[2].abs() > f32::EPSILON {
            1.0 / dir[2]
        } else {
            f32::MAX
        },
    ];

    let ts = f32x8::new([
        0.0,
        1.0,
        ((-he[0] - p0[0]) * inv_dir[0]).clamp(0.0, 1.0),
        ((he[0] - p0[0]) * inv_dir[0]).clamp(0.0, 1.0),
        ((-he[1] - p0[1]) * inv_dir[1]).clamp(0.0, 1.0),
        ((he[1] - p0[1]) * inv_dir[1]).clamp(0.0, 1.0),
        ((-he[2] - p0[2]) * inv_dir[2]).clamp(0.0, 1.0),
        ((he[2] - p0[2]) * inv_dir[2]).clamp(0.0, 1.0),
    ]);

    let zero = f32x8::splat(0.0);
    let mut dist_sq = zero;
    for i in 0..3 {
        let pos = f32x8::splat(p0[i]) + f32x8::splat(dir[i]) * ts;
        let abs_pos = pos.max(-pos);
        let excess = (abs_pos - f32x8::splat(he[i])).max(zero);
        dist_sq = dist_sq + excess * excess;
    }

    if dist_sq.simd_le(f32x8::splat(rs_sq)).any() {
        return true;
    }

    // Test 2: Check 8 cuboid corners against cylinder
    let mut corners = [[0.0f32; 3]; 8];
    let mut idx = 0;
    for &sx in &[-1.0f32, 1.0] {
        for &sy in &[-1.0f32, 1.0] {
            for &sz in &[-1.0f32, 1.0] {
                let v = cuboid.center
                    + cuboid.axes[0] * (he[0] * sx)
                    + cuboid.axes[1] * (he[1] * sy)
                    + cuboid.axes[2] * (he[2] * sz);
                corners[idx] = [v.x, v.y, v.z];
                idx += 1;
            }
        }
    }

    // SIMD check: all 8 corners against cylinder
    let cx = f32x8::new(corners.map(|c| c[0]));
    let cy = f32x8::new(corners.map(|c| c[1]));
    let cz = f32x8::new(corners.map(|c| c[2]));

    let wx = cx - f32x8::splat(cyl.p1.x);
    let wy = cy - f32x8::splat(cyl.p1.y);
    let wz = cz - f32x8::splat(cyl.p1.z);

    let cdx = f32x8::splat(cyl.dir.x);
    let cdy = f32x8::splat(cyl.dir.y);
    let cdz = f32x8::splat(cyl.dir.z);
    let crdv = f32x8::splat(cyl.rdv);

    let t = (wx * cdx + wy * cdy + wz * cdz) * crdv;
    let one = f32x8::splat(1.0);
    let in_slab = zero.simd_le(t) & t.simd_le(one);

    let perpx = wx - cdx * t;
    let perpy = wy - cdy * t;
    let perpz = wz - cdz * t;
    let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;

    let corner_in_barrel = in_slab & r_sq.simd_le(f32x8::splat(rs_sq));
    if corner_in_barrel.any() {
        return true;
    }

    false
}

impl Collides<Cuboid> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        cylinder_cuboid_collides::<BROADPHASE>(self, cuboid)
    }
}

impl Collides<Cylinder> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        cylinder_cuboid_collides::<BROADPHASE>(cyl, self)
    }
}

// ─── Cylinder-Cylinder ───────────────────────────────────────────────────────

impl Collides<Cylinder> for Cylinder {
    fn test<const BROADPHASE: bool>(&self, other: &Cylinder) -> bool {
        if BROADPHASE {
            let (c1, r1) = self.bounding_sphere();
            let (c2, r2) = other.bounding_sphere();
            let d = c1 - c2;
            let max_r = r1 + r2;
            if d.dot(d) > max_r * max_r {
                return false;
            }
        }

        // Check axis-axis closest approach (barrel-barrel)
        let dist_sq =
            crate::capsule::segment_segment_dist_sq(self.p1, self.dir, other.p1, other.dir);
        let combined = self.radius + other.radius;
        if dist_sq <= combined * combined {
            // Need to verify closest approach is in barrel region of both cylinders
            // The segment-segment distance already clamps to [0,1] on both,
            // so if the distance is within combined radii, the barrels overlap.
            return true;
        }

        // Check end caps: sample 5 points on each cylinder against the other
        for cyl_a in [self, other] {
            let cyl_b = if std::ptr::eq(cyl_a, self) {
                other
            } else {
                self
            };
            for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let p = cyl_a.p1 + cyl_a.dir * t;
                if cyl_b.point_dist_sq(p) <= cyl_a.radius * cyl_a.radius {
                    return true;
                }
            }
        }

        false
    }
}

// ─── Cylinder-Plane ──────────────────────────────────────────────────────────

#[inline]
fn plane_cylinder_collides(plane: &Plane, cyl: &Cylinder) -> bool {
    let proj1 = plane.normal.dot(cyl.p1);
    let proj2 = plane.normal.dot(cyl.p1 + cyl.dir);
    let min_proj = proj1.min(proj2);

    let dir_sq = cyl.dir.dot(cyl.dir);
    let n_dot_dir = plane.normal.dot(cyl.dir);
    let n_perp_sq = if dir_sq > f32::EPSILON {
        (1.0 - n_dot_dir * n_dot_dir / dir_sq).max(0.0)
    } else {
        1.0
    };
    let disc_extent = cyl.radius * n_perp_sq.sqrt();

    min_proj - disc_extent <= plane.d
}

impl Collides<Plane> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, plane: &Plane) -> bool {
        plane_cylinder_collides(plane, self)
    }
}

impl Collides<Cylinder> for Plane {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        plane_cylinder_collides(self, cyl)
    }
}

// ─── Cylinder-Line/Ray/Segment ───────────────────────────────────────────────

/// Line-cylinder intersection using quadratic for barrel + disc checks for end caps.
pub(crate) fn line_cylinder_collides(
    origin: Vec3,
    line_dir: Vec3,
    cyl: &Cylinder,
    t_min: f32,
    t_max: f32,
) -> bool {
    let dir_sq = cyl.dir.dot(cyl.dir);
    if dir_sq < f32::EPSILON {
        // Degenerate cylinder (disc at p1): treat as sphere
        let sphere = Sphere::new(cyl.p1, cyl.radius);
        return crate::line::line_sphere_collides(
            origin,
            line_dir,
            if line_dir.dot(line_dir) > f32::EPSILON {
                1.0 / line_dir.dot(line_dir)
            } else {
                0.0
            },
            &sphere,
            t_min,
            t_max,
        );
    }

    let cyl_len = dir_sq.sqrt();
    let cyl_axis = cyl.dir / cyl_len;

    let w = origin - cyl.p1;
    let d_par = line_dir.dot(cyl_axis);
    let w_par = w.dot(cyl_axis);

    let d_perp = line_dir - cyl_axis * d_par;
    let w_perp = w - cyl_axis * w_par;

    let a = d_perp.dot(d_perp);
    let b = 2.0 * w_perp.dot(d_perp);
    let c = w_perp.dot(w_perp) - cyl.radius * cyl.radius;

    // Check barrel intersection
    if a > f32::EPSILON {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            let inv_2a = 0.5 / a;
            let t_cyl_enter = (-b - sqrt_disc) * inv_2a;
            let t_cyl_exit = (-b + sqrt_disc) * inv_2a;

            // Clip to slab
            let (t_slab_enter, t_slab_exit) = if d_par.abs() > f32::EPSILON {
                let t0 = -w_par / d_par;
                let t1 = (cyl_len - w_par) / d_par;
                if t0 < t1 { (t0, t1) } else { (t1, t0) }
            } else if w_par >= 0.0 && w_par <= cyl_len {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else {
                (1.0, -1.0) // empty
            };

            let t_enter = t_cyl_enter.max(t_slab_enter).max(t_min);
            let t_exit = t_cyl_exit.min(t_slab_exit).min(t_max);
            if t_enter <= t_exit {
                return true;
            }
        }
    } else {
        // Line parallel to axis: check if inside infinite cylinder
        if c <= 0.0 {
            // Inside cylinder radius — check slab overlap
            let (t_slab_enter, t_slab_exit) = if d_par.abs() > f32::EPSILON {
                let t0 = -w_par / d_par;
                let t1 = (cyl_len - w_par) / d_par;
                if t0 < t1 { (t0, t1) } else { (t1, t0) }
            } else if w_par >= 0.0 && w_par <= cyl_len {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else {
                (1.0, -1.0)
            };
            if t_slab_enter.max(t_min) <= t_slab_exit.min(t_max) {
                return true;
            }
        }
    }

    // Check end cap discs
    let r_sq = cyl.radius * cyl.radius;
    if d_par.abs() > f32::EPSILON {
        // Disc at p1 (axial pos = 0)
        let t0 = -w_par / d_par;
        if t0 >= t_min && t0 <= t_max {
            let hit = origin + line_dir * t0;
            let diff = hit - cyl.p1;
            let perp = diff - cyl_axis * diff.dot(cyl_axis);
            if perp.dot(perp) <= r_sq {
                return true;
            }
        }
        // Disc at p2 (axial pos = cyl_len)
        let t1 = (cyl_len - w_par) / d_par;
        if t1 >= t_min && t1 <= t_max {
            let hit = origin + line_dir * t1;
            let p2 = cyl.p1 + cyl.dir;
            let diff = hit - p2;
            let perp = diff - cyl_axis * diff.dot(cyl_axis);
            if perp.dot(perp) <= r_sq {
                return true;
            }
        }
    }

    false
}

impl Collides<Line> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, line: &Line) -> bool {
        line_cylinder_collides(
            line.origin,
            line.dir,
            self,
            f32::NEG_INFINITY,
            f32::INFINITY,
        )
    }
}

impl Collides<Cylinder> for Line {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        line_cylinder_collides(self.origin, self.dir, cyl, f32::NEG_INFINITY, f32::INFINITY)
    }
}

impl Collides<Ray> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, ray: &Ray) -> bool {
        line_cylinder_collides(ray.origin, ray.dir, self, 0.0, f32::INFINITY)
    }
}

impl Collides<Cylinder> for Ray {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        line_cylinder_collides(self.origin, self.dir, cyl, 0.0, f32::INFINITY)
    }
}

impl Collides<LineSegment> for Cylinder {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, seg: &LineSegment) -> bool {
        line_cylinder_collides(seg.p1, seg.dir, self, 0.0, 1.0)
    }
}

impl Collides<Cylinder> for LineSegment {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        line_cylinder_collides(self.p1, self.dir, cyl, 0.0, 1.0)
    }
}

// ─── Cylinder-ConvexPolygon ──────────────────────────────────────────────────

impl Collides<ConvexPolygon> for Cylinder {
    fn test<const BROADPHASE: bool>(&self, polygon: &ConvexPolygon) -> bool {
        if BROADPHASE {
            let (bc, br) = self.bounding_sphere();
            let bp = polygon.broadphase();
            let d = bc - bp.center;
            let max_r = br + bp.radius;
            if d.dot(d) > max_r * max_r {
                return false;
            }
        }

        // Check polygon vertices against cylinder
        for &v in &polygon.vertices_3d {
            if self.contains_point(v) {
                return true;
            }
        }

        // Check cylinder axis vs polygon (capsule-like distance)
        let dist_sq = polygon.parametric_line_dist_sq(self.p1, self.dir, 0.0, 1.0);
        if dist_sq <= self.radius * self.radius {
            return true;
        }

        false
    }
}

impl Collides<Cylinder> for ConvexPolygon {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        cyl.test::<BROADPHASE>(self)
    }
}

// ─── Cylinder-ConvexPolytope ─────────────────────────────────────────────────

impl Collides<ConvexPolytope> for Cylinder {
    fn test<const BROADPHASE: bool>(&self, polytope: &ConvexPolytope) -> bool {
        if BROADPHASE {
            let (bc, br) = self.bounding_sphere();
            let bp = polytope.broadphase();
            let d = bc - bp.center;
            let max_r = br + bp.radius;
            if d.dot(d) > max_r * max_r {
                return false;
            }
        }

        // Check if cylinder axis passes through (OBB-expanded) polytope
        if crate::line::line_polytope_collides(
            self.p1,
            self.dir,
            &polytope.planes,
            &polytope.obb,
            0.0,
            1.0,
        ) {
            return true;
        }

        // Check polytope vertices against cylinder
        for &v in &polytope.vertices {
            if self.contains_point(v) {
                return true;
            }
        }

        // SAT: cylinder support function against polytope planes
        let dir_sq = self.dir.dot(self.dir);
        if dir_sq > f32::EPSILON {
            let cyl_len = dir_sq.sqrt();
            let cyl_axis = self.dir / cyl_len;
            for &(normal, d) in &polytope.planes {
                let proj1 = normal.dot(self.p1);
                let proj2 = normal.dot(self.p1 + self.dir);
                let min_proj = proj1.min(proj2);

                let n_dot_axis = normal.dot(cyl_axis);
                let n_perp_sq = (1.0 - n_dot_axis * n_dot_axis).max(0.0);
                let disc_extent = self.radius * n_perp_sq.sqrt();

                if min_proj - disc_extent > d {
                    return false;
                }
            }
        }

        true
    }
}

impl Collides<Cylinder> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, cyl: &Cylinder) -> bool {
        cyl.test::<BROADPHASE>(self)
    }
}

// ─── Stretchable ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CylinderStretch {
    Aligned(Cylinder),
    Unaligned([Capsule; 4], ConvexPolytope),
}

impl Stretchable for Cylinder {
    type Output = CylinderStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        let p2 = self.p2();
        let cross = self.dir.cross(translation);
        if cross.length_squared() < 1e-10 {
            let proj = translation.dot(self.dir);
            let (new_p1, new_p2) = if proj >= 0.0 {
                (self.p1, p2 + translation)
            } else {
                (self.p1 + translation, p2)
            };
            return CylinderStretch::Aligned(Cylinder::new(new_p1, new_p2, self.radius));
        }

        let p1t = self.p1 + translation;
        let p2t = p2 + translation;

        let edges = [
            Capsule::new(self.p1, p2, self.radius),
            Capsule::new(p1t, p2t, self.radius),
            Capsule::new(self.p1, p1t, self.radius),
            Capsule::new(p2, p2t, self.radius),
        ];

        let n = cross.normalize();
        let rn = self.radius * n;
        let corners = [self.p1, p2, p1t, p2t];
        let vertices: Vec<Vec3> = corners.iter().flat_map(|&c| [c + rn, c - rn]).collect();

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

        CylinderStretch::Unaligned(edges, ConvexPolytope::with_obb(planes, vertices, obb))
    }
}
