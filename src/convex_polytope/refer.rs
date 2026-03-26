use glam::Vec3;
use wide::{CmpGt, f32x8};

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::point::Point;
use crate::sphere::Sphere;
use crate::{Collides, ConvexPolytope};

#[derive(Debug, Clone)]
pub struct RefConvexPolytope<'a> {
    pub planes: &'a [(Vec3, f32)],
    pub vertices: &'a [Vec3],
    pub obb: &'a Cuboid,
}

impl<'a> RefConvexPolytope<'a> {
    #[inline]
    pub fn from_heap(heap: &'a ConvexPolytope) -> Self {
        RefConvexPolytope {
            planes: &heap.planes,
            vertices: &heap.vertices,
            obb: &heap.obb,
        }
    }

    #[inline]
    pub fn from_array<const P: usize, const V: usize>(
        array: &'a ArrayConvexPolytope<P, V>,
    ) -> Self {
        RefConvexPolytope {
            planes: &array.planes,
            vertices: &array.vertices,
            obb: &array.obb,
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD helpers: gather 8 planes into SoA layout
// ---------------------------------------------------------------------------

#[inline]
fn gather_planes(chunk: &[(Vec3, f32)]) -> (f32x8, f32x8, f32x8, f32x8) {
    let mut nx = [0.0f32; 8];
    let mut ny = [0.0f32; 8];
    let mut nz = [0.0f32; 8];
    let mut d = [0.0f32; 8];
    for (i, &(normal, dist)) in chunk.iter().enumerate() {
        nx[i] = normal.x;
        ny[i] = normal.y;
        nz[i] = normal.z;
        d[i] = dist;
    }
    (
        f32x8::new(nx),
        f32x8::new(ny),
        f32x8::new(nz),
        f32x8::new(d),
    )
}

/// Dot product of 8 plane normals (SoA) with a single point, minus d.
#[inline]
fn dot8_minus_d(
    nx: f32x8,
    ny: f32x8,
    nz: f32x8,
    d: f32x8,
    px: f32x8,
    py: f32x8,
    pz: f32x8,
) -> f32x8 {
    nx * px + ny * py + nz * pz - d
}

// ---------------------------------------------------------------------------
// Collision implementations on RefConvexPolytope
// ---------------------------------------------------------------------------

impl RefConvexPolytope<'_> {
    #[inline]
    pub(crate) fn collides_sphere<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        if BROADPHASE && !sphere.collides(self.obb) {
            return false;
        }

        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let r = f32x8::splat(sphere.radius);
        let zero = f32x8::ZERO;

        let chunks = self.planes.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let (nx, ny, nz, d) = gather_planes(chunk);
            let sep = dot8_minus_d(nx, ny, nz, d, cx, cy, cz) - r;
            if sep.simd_gt(zero).any() {
                return false;
            }
        }
        for &(normal, d) in remainder {
            if normal.dot(sphere.center) - d - sphere.radius > 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub(crate) fn collides_cuboid<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        if BROADPHASE && !cuboid.collides(self.obb) {
            return false;
        }

        let cx = f32x8::splat(cuboid.center.x);
        let cy = f32x8::splat(cuboid.center.y);
        let cz = f32x8::splat(cuboid.center.z);
        let ax0 = f32x8::splat(cuboid.axes[0].x);
        let ay0 = f32x8::splat(cuboid.axes[0].y);
        let az0 = f32x8::splat(cuboid.axes[0].z);
        let ax1 = f32x8::splat(cuboid.axes[1].x);
        let ay1 = f32x8::splat(cuboid.axes[1].y);
        let az1 = f32x8::splat(cuboid.axes[1].z);
        let ax2 = f32x8::splat(cuboid.axes[2].x);
        let ay2 = f32x8::splat(cuboid.axes[2].y);
        let az2 = f32x8::splat(cuboid.axes[2].z);
        let h0 = f32x8::splat(cuboid.half_extents[0]);
        let h1 = f32x8::splat(cuboid.half_extents[1]);
        let h2 = f32x8::splat(cuboid.half_extents[2]);
        let zero = f32x8::ZERO;

        let chunks = self.planes.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let (nx, ny, nz, d) = gather_planes(chunk);
            let center_proj = nx * cx + ny * cy + nz * cz;
            let extent_proj = (nx * ax0 + ny * ay0 + nz * az0).abs() * h0
                + (nx * ax1 + ny * ay1 + nz * az1).abs() * h1
                + (nx * ax2 + ny * ay2 + nz * az2).abs() * h2;
            let sep = center_proj - extent_proj - d;
            if sep.simd_gt(zero).any() {
                return false;
            }
        }
        for &(normal, d) in remainder {
            let center_proj = normal.dot(cuboid.center);
            let extent_proj = (normal.dot(cuboid.axes[0]).abs() * cuboid.half_extents[0])
                + (normal.dot(cuboid.axes[1]).abs() * cuboid.half_extents[1])
                + (normal.dot(cuboid.axes[2]).abs() * cuboid.half_extents[2]);
            if center_proj - extent_proj - d > 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub(crate) fn collides_capsule<const BROADPHASE: bool>(&self, capsule: &Capsule) -> bool {
        if BROADPHASE {
            let (bc, br) = capsule.bounding_sphere();
            let bounding = Sphere::new(bc, br);
            if !bounding.collides(self.obb) {
                return false;
            }
        }

        let p1x = f32x8::splat(capsule.p1.x);
        let p1y = f32x8::splat(capsule.p1.y);
        let p1z = f32x8::splat(capsule.p1.z);
        let p2 = capsule.p2();
        let p2x = f32x8::splat(p2.x);
        let p2y = f32x8::splat(p2.y);
        let p2z = f32x8::splat(p2.z);
        let r = f32x8::splat(capsule.radius);
        let zero = f32x8::ZERO;

        let chunks = self.planes.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let (nx, ny, nz, d) = gather_planes(chunk);
            let proj1 = nx * p1x + ny * p1y + nz * p1z;
            let proj2 = nx * p2x + ny * p2y + nz * p2z;
            let min_proj = proj1.min(proj2);
            let sep = min_proj - r - d;
            if sep.simd_gt(zero).any() {
                return false;
            }
        }
        for &(normal, d) in remainder {
            let proj_p1 = normal.dot(capsule.p1);
            let proj_p2 = normal.dot(p2);
            let min_proj = proj_p1.min(proj_p2);
            if min_proj - capsule.radius - d > 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub(crate) fn collides_point<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        if BROADPHASE && self.obb.point_dist_sq(point.0) > 0.0 {
            return false;
        }

        let px = f32x8::splat(point.0.x);
        let py = f32x8::splat(point.0.y);
        let pz = f32x8::splat(point.0.z);
        let zero = f32x8::ZERO;

        let chunks = self.planes.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let (nx, ny, nz, d) = gather_planes(chunk);
            let sep = dot8_minus_d(nx, ny, nz, d, px, py, pz);
            if sep.simd_gt(zero).any() {
                return false;
            }
        }
        for &(normal, d) in remainder {
            if normal.dot(point.0) > d {
                return false;
            }
        }
        true
    }

    #[inline]
    pub(crate) fn collides_polytope<const BROADPHASE: bool>(
        &self,
        other: &RefConvexPolytope<'_>,
    ) -> bool {
        // Broadphase: OBB vs OBB
        if BROADPHASE && !self.obb.collides(other.obb) {
            return false;
        }

        // Test self's planes as separating axes against other's vertices
        for &(normal, d) in self.planes {
            let min_proj = super::min_projection(other.vertices, normal);
            if min_proj > d {
                return false;
            }
        }

        // Test other's planes as separating axes against self's vertices
        for &(normal, d) in other.planes {
            let min_proj = super::min_projection(self.vertices, normal);
            if min_proj > d {
                return false;
            }
        }

        true
    }
}
