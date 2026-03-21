use glam::Vec3;
use wide::{CmpLe, f32x8};

use inherent::inherent;

use crate::capsule::Capsule;
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Bounded, Collides, Scalable, Transformable};
use crate::{ConvexPolytope, Stretchable};

#[derive(Debug, Clone, Copy)]
pub struct Cuboid {
    pub center: Vec3,
    pub axes: [Vec3; 3],
    pub half_extents: [f32; 3],
    pub axis_aligned: bool,
}

#[inline]
const fn is_axis_aligned(axes: &[Vec3; 3]) -> bool {
    // Check off-diagonal components are zero (allows negative/permuted axes)
    axes[0].y == 0.0
        && axes[0].z == 0.0
        && axes[1].x == 0.0
        && axes[1].z == 0.0
        && axes[2].x == 0.0
        && axes[2].y == 0.0
}

impl Cuboid {
    pub const fn new(center: Vec3, axes: [Vec3; 3], half_extents: [f32; 3]) -> Self {
        wreck_assert!(
            half_extents[0] >= 0.0 && half_extents[1] >= 0.0 && half_extents[2] >= 0.0,
            "Cuboid half_extents must be non-negative"
        );
        let axis_aligned = is_axis_aligned(&axes);
        Self {
            center,
            axes,
            half_extents,
            axis_aligned,
        }
    }

    pub const fn from_aabb(min: Vec3, max: Vec3) -> Self {
        wreck_assert!(
            min.x <= max.x && min.y <= max.y && min.z <= max.z,
            "AABB min must be <= max"
        );
        let center = Vec3::new(
            (min.x + max.x) * 0.5,
            (min.y + max.y) * 0.5,
            (min.z + max.z) * 0.5,
        );
        let half = Vec3::new(
            (max.x - min.x) * 0.5,
            (max.y - min.y) * 0.5,
            (max.z - min.z) * 0.5,
        );
        Self {
            center,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents: [half.x, half.y, half.z],
            axis_aligned: true,
        }
    }

    #[inline]
    pub fn bounding_sphere_radius(&self) -> f32 {
        let e = self.half_extents;
        (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
    }

    /// Squared distance from a point to the closest point on this cuboid's surface/interior.
    #[inline]
    pub(crate) fn point_dist_sq(&self, point: Vec3) -> f32 {
        if self.axis_aligned {
            return self.point_dist_sq_aa(point);
        }
        let d = point - self.center;
        let mut dist_sq = 0.0;
        for i in 0..3 {
            let proj = d.dot(self.axes[i]).abs() - self.half_extents[i];
            if proj > 0.0 {
                dist_sq += proj * proj;
            }
        }
        dist_sq
    }

    /// Axis-aligned fast path: no dot products needed.
    #[inline]
    fn point_dist_sq_aa(&self, point: Vec3) -> f32 {
        let d = point - self.center;
        let da = [d.x, d.y, d.z];
        let mut dist_sq = 0.0;
        for i in 0..3 {
            let excess = da[i].abs() - self.half_extents[i];
            if excess > 0.0 {
                dist_sq += excess * excess;
            }
        }
        dist_sq
    }
}

impl Cuboid {
    pub(crate) const fn contains_point(&self, point: Vec3) -> bool {
        let d = Vec3::new(
            (point.x - self.center.x).abs(),
            (point.y - self.center.y).abs(),
            (point.z - self.center.z).abs(),
        );
        let mut i = 0;
        while i < 3 {
            let proj = crate::dot(d, self.axes[i]).abs();
            if proj > self.half_extents[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

#[inherent]
impl Bounded for Cuboid {
    pub fn broadphase(&self) -> Sphere {
        Sphere::new(self.center, self.bounding_sphere_radius())
    }

    pub fn obb(&self) -> Cuboid {
        *self
    }

    pub fn aabb(&self) -> Cuboid {
        if self.axis_aligned {
            return *self;
        }
        let mut he = [0.0f32; 3];
        let world = [Vec3::X, Vec3::Y, Vec3::Z];
        for i in 0..3 {
            he[i] = self.half_extents[0] * self.axes[0].dot(world[i]).abs()
                + self.half_extents[1] * self.axes[1].dot(world[i]).abs()
                + self.half_extents[2] * self.axes[2].dot(world[i]).abs();
        }
        Cuboid::new(self.center, [Vec3::X, Vec3::Y, Vec3::Z], he)
    }
}

#[inherent]
impl Scalable for Cuboid {
    pub fn scale(&mut self, factor: f32) {
        for e in &mut self.half_extents {
            *e *= factor;
        }
    }
}

#[inherent]
impl Transformable for Cuboid {
    pub fn translate(&mut self, offset: Vec3) {
        self.center += offset;
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3) {
        self.center = mat * self.center;
        for ax in &mut self.axes {
            *ax = mat * *ax;
        }
        self.axis_aligned = is_axis_aligned(&self.axes);
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        self.center = quat * self.center;
        for ax in &mut self.axes {
            *ax = quat * *ax;
        }
        self.axis_aligned = is_axis_aligned(&self.axes);
    }

    pub fn transform(&mut self, mat: glam::Affine3) {
        self.center = mat.transform_point3(self.center);
        let rot = mat.matrix3;
        for ax in &mut self.axes {
            *ax = rot * *ax;
        }
        self.axis_aligned = is_axis_aligned(&self.axes);
    }
}

// AABB-AABB: simple overlap when both cuboids are axis-aligned
#[inline]
fn aabb_aabb_collides(a: &Cuboid, b: &Cuboid) -> bool {
    let d = b.center - a.center;
    d.x.abs() <= a.half_extents[0] + b.half_extents[0]
        && d.y.abs() <= a.half_extents[1] + b.half_extents[1]
        && d.z.abs() <= a.half_extents[2] + b.half_extents[2]
}

// Sphere-Cuboid collision
#[inline]
fn sphere_cuboid_collides(sphere: &Sphere, cuboid: &Cuboid) -> bool {
    cuboid.point_dist_sq(sphere.center) <= sphere.radius * sphere.radius
}

impl Collides<Cuboid> for Sphere {
    #[inline]
    fn collides(&self, other: &Cuboid) -> bool {
        sphere_cuboid_collides(self, other)
    }

    fn collides_many(&self, others: &[Cuboid]) -> bool {
        let cx = f32x8::splat(self.center.x);
        let cy = f32x8::splat(self.center.y);
        let cz = f32x8::splat(self.center.z);
        let r_sq = f32x8::splat(self.radius * self.radius);
        let zero = f32x8::splat(0.0);

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // For each cuboid we need center, 3 axes, 3 half-extents = 13 values
            // Process them in SoA layout
            let mut ocx = [0.0f32; 8];
            let mut ocy = [0.0f32; 8];
            let mut ocz = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                ocx[i] = c.center.x;
                ocy[i] = c.center.y;
                ocz[i] = c.center.z;
            }
            let dfx = cx - f32x8::new(ocx);
            let dfy = cy - f32x8::new(ocy);
            let dfz = cz - f32x8::new(ocz);

            let mut dist_sq = zero;
            for axis_idx in 0..3 {
                let mut ax = [0.0f32; 8];
                let mut ay = [0.0f32; 8];
                let mut az = [0.0f32; 8];
                let mut he = [0.0f32; 8];
                for (i, c) in chunk.iter().enumerate() {
                    ax[i] = c.axes[axis_idx].x;
                    ay[i] = c.axes[axis_idx].y;
                    az[i] = c.axes[axis_idx].z;
                    he[i] = c.half_extents[axis_idx];
                }
                let proj = dfx * f32x8::new(ax) + dfy * f32x8::new(ay) + dfz * f32x8::new(az);
                let abs_proj = proj.max(-proj);
                let excess = (abs_proj - f32x8::new(he)).max(zero);
                dist_sq = dist_sq + excess * excess;
            }

            if dist_sq.simd_le(r_sq).any() {
                return true;
            }
        }

        remainder.iter().any(|c| sphere_cuboid_collides(self, c))
    }
}

impl Collides<Sphere> for Cuboid {
    #[inline]
    fn collides(&self, other: &Sphere) -> bool {
        sphere_cuboid_collides(other, self)
    }
}

// Capsule-Cuboid: Z-aligned capsule + axis-aligned cuboid.
// X/Y distances are constant along the capsule axis, only Z varies.
#[inline]
fn capsule_cuboid_za_aa(capsule: &Capsule, cuboid: &Cuboid) -> bool {
    let rs_sq = capsule.radius * capsule.radius;
    let d = capsule.p1 - cuboid.center;
    let he = cuboid.half_extents;

    // X and Y excess are constant (capsule only extends along Z)
    let ex = d.x.abs() - he[0];
    let ey = d.y.abs() - he[1];
    let xy_dist_sq = ex.max(0.0) * ex.max(0.0) + ey.max(0.0) * ey.max(0.0);

    // Early out: if XY distance alone exceeds radius, no collision possible
    if xy_dist_sq > rs_sq {
        return false;
    }

    // Only need to check Z overlap: capsule spans [p1.z, p1.z + dir.z]
    let z0 = d.z;
    let z1 = d.z + capsule.dir.z;
    let z_min = z0.min(z1);
    let z_max = z0.max(z1);

    // Closest Z distance from capsule segment to cuboid Z-extent [-he[2], he[2]]
    let ez = if z_max < -he[2] {
        -he[2] - z_max
    } else if z_min > he[2] {
        z_min - he[2]
    } else {
        0.0
    };

    xy_dist_sq + ez * ez <= rs_sq
}

// Capsule-Cuboid collision
// Evaluate all 8 candidate t-values (2 endpoints + 6 face-plane intersections)
// in a single SIMD pass using f32x8.
#[inline]
fn capsule_cuboid_collides(capsule: &Capsule, cuboid: &Cuboid) -> bool {
    // Bounding sphere early-out
    let (bc, br) = capsule.bounding_sphere();
    let d = bc - cuboid.center;
    let max_r = br + cuboid.bounding_sphere_radius();
    if d.dot(d) > max_r * max_r {
        return false;
    }

    // Fastest path: Z-aligned capsule + axis-aligned cuboid
    if capsule.z_aligned && cuboid.axis_aligned {
        return capsule_cuboid_za_aa(capsule, cuboid);
    }

    let rs_sq = capsule.radius * capsule.radius;
    let p0_world = capsule.p1 - cuboid.center;

    // Axis-aligned cuboid fast path: skip 6 dot products for local frame projection
    let (p0, dir) = if cuboid.axis_aligned {
        (
            [p0_world.x, p0_world.y, p0_world.z],
            [capsule.dir.x, capsule.dir.y, capsule.dir.z],
        )
    } else {
        (
            [
                p0_world.dot(cuboid.axes[0]),
                p0_world.dot(cuboid.axes[1]),
                p0_world.dot(cuboid.axes[2]),
            ],
            [
                capsule.dir.dot(cuboid.axes[0]),
                capsule.dir.dot(cuboid.axes[1]),
                capsule.dir.dot(cuboid.axes[2]),
            ],
        )
    };
    let he = cuboid.half_extents;

    // Compute 6 critical t-values where capsule axis meets cuboid face planes,
    // plus 2 endpoints. Pack all 8 into f32x8.
    // t_i = (±he[i] - p0[i]) / dir[i], clamped to [0,1]
    // For near-zero dir components, the division gives ±inf which clamps to 0 or 1.
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

    // Evaluate squared distance from capsule-axis point at each t to cuboid, branchless.
    // For each axis: excess = max(0, |p0[i] + dir[i]*t| - he[i])
    let zero = f32x8::splat(0.0);
    let mut dist_sq = zero;

    for i in 0..3 {
        let pos = f32x8::splat(p0[i]) + f32x8::splat(dir[i]) * ts;
        let abs_pos = pos.max(-pos); // branchless abs
        let excess = (abs_pos - f32x8::splat(he[i])).max(zero);
        dist_sq = dist_sq + excess * excess;
    }

    // Check if any of the 8 evaluations is within capsule radius
    dist_sq.simd_le(f32x8::splat(rs_sq)).any()
}

impl Collides<Cuboid> for Capsule {
    #[inline]
    fn collides(&self, other: &Cuboid) -> bool {
        capsule_cuboid_collides(self, other)
    }

    fn collides_many(&self, others: &[Cuboid]) -> bool {
        let (sc, sr) = self.bounding_sphere();
        crate::broadphase_collides_many(
            sc,
            sr,
            others,
            |other| (other.center, other.bounding_sphere_radius()),
            |other| self.collides(other),
        )
    }
}

impl Collides<Capsule> for Cuboid {
    #[inline]
    fn collides(&self, other: &Capsule) -> bool {
        capsule_cuboid_collides(other, self)
    }

    fn collides_many(&self, others: &[Capsule]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_sphere_radius(),
            others,
            |other| other.bounding_sphere(),
            |other| self.collides(other),
        )
    }
}

// Cuboid-Cuboid collision via Separating Axis Theorem (15 axes)
impl Collides<Cuboid> for Cuboid {
    fn collides_many(&self, others: &[Cuboid]) -> bool {
        crate::broadphase_collides_many(
            self.center,
            self.bounding_sphere_radius(),
            others,
            |other| (other.center, other.bounding_sphere_radius()),
            |other| self.collides(other),
        )
    }

    fn collides(&self, other: &Cuboid) -> bool {
        // Both axis-aligned: simple AABB overlap test
        if self.axis_aligned && other.axis_aligned {
            return aabb_aabb_collides(self, other);
        }

        // Bounding sphere early-out
        let t_vec = other.center - self.center;
        let max_dist = self.bounding_sphere_radius() + other.bounding_sphere_radius();
        if t_vec.dot(t_vec) > max_dist * max_dist {
            return false;
        }

        let eps = 1e-6f32;

        // Rotation matrix: R[i][j] = self.axes[i].dot(other.axes[j])
        let mut r = [[0.0f32; 3]; 3];
        let mut abs_r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = self.axes[i].dot(other.axes[j]);
                abs_r[i][j] = r[i][j].abs() + eps;
            }
        }

        // Translation in A's frame
        let t = [
            t_vec.dot(self.axes[0]),
            t_vec.dot(self.axes[1]),
            t_vec.dot(self.axes[2]),
        ];

        let ea = self.half_extents;
        let eb = other.half_extents;

        // Test axes L = A0, A1, A2
        for i in 0..3 {
            let ra = ea[i];
            let rb = eb[0] * abs_r[i][0] + eb[1] * abs_r[i][1] + eb[2] * abs_r[i][2];
            if t[i].abs() > ra + rb {
                return false;
            }
        }

        // Test axes L = B0, B1, B2
        for j in 0..3 {
            let ra = ea[0] * abs_r[0][j] + ea[1] * abs_r[1][j] + ea[2] * abs_r[2][j];
            let rb = eb[j];
            let sep = (t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j]).abs();
            if sep > ra + rb {
                return false;
            }
        }

        // Test 9 cross-product axes
        // L = A0 x B0
        {
            let ra = ea[1] * abs_r[2][0] + ea[2] * abs_r[1][0];
            let rb = eb[1] * abs_r[0][2] + eb[2] * abs_r[0][1];
            let sep = (t[2] * r[1][0] - t[1] * r[2][0]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A0 x B1
        {
            let ra = ea[1] * abs_r[2][1] + ea[2] * abs_r[1][1];
            let rb = eb[0] * abs_r[0][2] + eb[2] * abs_r[0][0];
            let sep = (t[2] * r[1][1] - t[1] * r[2][1]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A0 x B2
        {
            let ra = ea[1] * abs_r[2][2] + ea[2] * abs_r[1][2];
            let rb = eb[0] * abs_r[0][1] + eb[1] * abs_r[0][0];
            let sep = (t[2] * r[1][2] - t[1] * r[2][2]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A1 x B0
        {
            let ra = ea[0] * abs_r[2][0] + ea[2] * abs_r[0][0];
            let rb = eb[1] * abs_r[1][2] + eb[2] * abs_r[1][1];
            let sep = (t[0] * r[2][0] - t[2] * r[0][0]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A1 x B1
        {
            let ra = ea[0] * abs_r[2][1] + ea[2] * abs_r[0][1];
            let rb = eb[0] * abs_r[1][2] + eb[2] * abs_r[1][0];
            let sep = (t[0] * r[2][1] - t[2] * r[0][1]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A1 x B2
        {
            let ra = ea[0] * abs_r[2][2] + ea[2] * abs_r[0][2];
            let rb = eb[0] * abs_r[1][1] + eb[1] * abs_r[1][0];
            let sep = (t[0] * r[2][2] - t[2] * r[0][2]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A2 x B0
        {
            let ra = ea[0] * abs_r[1][0] + ea[1] * abs_r[0][0];
            let rb = eb[1] * abs_r[2][2] + eb[2] * abs_r[2][1];
            let sep = (t[1] * r[0][0] - t[0] * r[1][0]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A2 x B1
        {
            let ra = ea[0] * abs_r[1][1] + ea[1] * abs_r[0][1];
            let rb = eb[0] * abs_r[2][2] + eb[2] * abs_r[2][0];
            let sep = (t[1] * r[0][1] - t[0] * r[1][1]).abs();
            if sep > ra + rb {
                return false;
            }
        }
        // L = A2 x B2
        {
            let ra = ea[0] * abs_r[1][2] + ea[1] * abs_r[0][2];
            let rb = eb[0] * abs_r[2][1] + eb[1] * abs_r[2][0];
            let sep = (t[1] * r[0][2] - t[0] * r[1][2]).abs();
            if sep > ra + rb {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone)]
pub enum CuboidStretch {
    Aligned(Cuboid),
    Unaligned(ConvexPolytope),
}

impl Stretchable for Cuboid {
    type Output = CuboidStretch;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        // Check if translation is along one of the cuboid's axes
        for i in 0..3 {
            let cross = self.axes[i].cross(translation);
            if cross.length_squared() < 1e-10 {
                let proj = translation.dot(self.axes[i]);
                let mut result = *self;
                result.center += translation * 0.5;
                result.half_extents[i] += proj.abs() * 0.5;
                return CuboidStretch::Aligned(result);
            }
        }

        // Unaligned: Minkowski sum of cuboid and line segment = convex polytope
        let he = self.half_extents;
        let ax = self.axes;
        let c = self.center;

        // 16 vertices: 8 original cuboid corners + 8 translated
        let mut vertices = Vec::with_capacity(16);
        for &sx in &[-1.0f32, 1.0] {
            for &sy in &[-1.0f32, 1.0] {
                for &sz in &[-1.0f32, 1.0] {
                    let v = c + ax[0] * (he[0] * sx) + ax[1] * (he[1] * sy) + ax[2] * (he[2] * sz);
                    vertices.push(v);
                    vertices.push(v + translation);
                }
            }
        }

        // Plane normals: 6 original face normals + up to 6 side normals from edge×translation
        let mut normals: Vec<Vec3> = Vec::with_capacity(12);
        for i in 0..3 {
            normals.push(ax[i]);
            normals.push(-ax[i]);
            let side = ax[i].cross(translation);
            if side.length_squared() > 1e-10 {
                let side_n = side.normalize();
                normals.push(side_n);
                normals.push(-side_n);
            }
        }

        let planes: Vec<(Vec3, f32)> = normals
            .into_iter()
            .map(|n| {
                let d = crate::convex_polytope::max_projection(&vertices, n);
                (n, d)
            })
            .collect();

        // Derive OBB analytically: same axes, extended extents
        let mut obb = *self;
        obb.center += translation * 0.5;
        for i in 0..3 {
            obb.half_extents[i] += translation.dot(ax[i]).abs() * 0.5;
        }

        CuboidStretch::Unaligned(ConvexPolytope::with_obb(planes, vertices, obb))
    }
}
