use glam::Vec3;
use hydroplane::{Gang, GangGlamExt, kernel};

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::gjk;
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
// Collision implementations on RefConvexPolytope
// ---------------------------------------------------------------------------
//
// A convex polytope is the intersection of its half-spaces, so a shape is
// separated from it whenever some plane is a separating axis. Each kernel walks
// the plane list and returns `true` as soon as one active lane separates; a
// `lane < cnt` mask removes the tail lanes a short final chunk leaves inactive.
//
// The plane sweep alone over-reports: it never tests the other shape's face
// normals or edge-cross axes, and it inflates the polytope's corners for shapes
// with a radius. Its "separated" verdict is trustworthy, so it stays as the
// fast reject; survivors are confirmed with the exact GJK narrowphase.

/// Is `p` inside every half-space? A cheap collision certificate for any point
/// known to lie inside the other shape.
#[inline]
pub(crate) fn point_inside(planes: &[(Vec3, f32)], p: Vec3) -> bool {
    planes.iter().all(|&(n, d)| n.dot(p) <= d)
}

impl RefConvexPolytope<'_> {
    #[inline]
    pub(crate) fn collides_sphere<const BROADPHASE: bool>(&self, sphere: &Sphere) -> bool {
        if BROADPHASE && !sphere.collides(self.obb) {
            return false;
        }
        if sphere_separated_k(self.planes, sphere.center, sphere.radius) {
            return false;
        }
        if point_inside(self.planes, sphere.center) {
            return true;
        }
        gjk::bodies_collide(
            &gjk::ConvexBody::hull(self.vertices),
            &gjk::ConvexBody::sphere(sphere),
        )
    }

    #[inline]
    pub(crate) fn collides_cuboid<const BROADPHASE: bool>(&self, cuboid: &Cuboid) -> bool {
        if BROADPHASE && !cuboid.collides(self.obb) {
            return false;
        }
        if cuboid_separated_k(self.planes, cuboid.center, cuboid.axes, cuboid.half_extents) {
            return false;
        }
        if point_inside(self.planes, cuboid.center) {
            return true;
        }
        gjk::bodies_collide(
            &gjk::ConvexBody::hull(self.vertices),
            &gjk::ConvexBody::cuboid(cuboid),
        )
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
        if capsule_separated_k(self.planes, capsule.p1, capsule.p2(), capsule.radius) {
            return false;
        }
        if point_inside(self.planes, capsule.p1) || point_inside(self.planes, capsule.p2()) {
            return true;
        }
        gjk::bodies_collide(
            &gjk::ConvexBody::hull(self.vertices),
            &gjk::ConvexBody::capsule(capsule),
        )
    }

    #[inline]
    pub(crate) fn collides_point<const BROADPHASE: bool>(&self, point: &Point) -> bool {
        if BROADPHASE && self.obb.point_dist_sq(point.0) > 0.0 {
            return false;
        }
        !point_separated_k(self.planes, point.0)
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

        if polytopes_separated_k(self.planes, self.vertices, other.planes, other.vertices) {
            return false;
        }
        let centroid = |verts: &[Vec3]| verts.iter().copied().sum::<Vec3>() / verts.len() as f32;
        if point_inside(self.planes, centroid(other.vertices))
            || point_inside(other.planes, centroid(self.vertices))
        {
            return true;
        }
        gjk::bodies_collide(
            &gjk::ConvexBody::hull(self.vertices),
            &gjk::ConvexBody::hull(other.vertices),
        )
    }
}

/// Bilateral SAT over both polytopes' planes, one dispatch for the whole test. Each vertex set
/// is staged column-wise so the per-plane sweeps use column loads instead of per-lane gathers;
/// `a_verts` is staged lazily since a separating plane of `a` skips it entirely.
#[kernel]
fn polytopes_separated_k<'a>(
    ctx: Gang,
    a_planes: &'a [(Vec3, f32)],
    a_verts: &'a [Vec3],
    b_planes: &'a [(Vec3, f32)],
    b_verts: &'a [Vec3],
) -> bool {
    let b_cols = super::stage_cols(b_verts);
    let (bx, by, bz) = super::cols3(&b_cols);
    for &(normal, d) in a_planes {
        if super::min_projection_cols_k_on(ctx, bx, by, bz, normal) > d {
            return true;
        }
    }

    let a_cols = super::stage_cols(a_verts);
    let (ax, ay, az) = super::cols3(&a_cols);
    for &(normal, d) in b_planes {
        if super::min_projection_cols_k_on(ctx, ax, ay, az, normal) > d {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Separating-axis kernels: `true` if some plane separates the shape from the
// polytope (so the shape is outside).
// ---------------------------------------------------------------------------

#[kernel]
fn sphere_separated_k<'a>(ctx: Gang, planes: &'a [(Vec3, f32)], center: Vec3, r: f32) -> bool {
    let zero = ctx.splat(0.0);
    let c = ctx.splat_vec3(center);

    for (off, cnt, active) in ctx.masked_chunks::<f32>(planes.len()) {
        let (n, d) = ctx.gather_plane(&planes[off..off + cnt], 0.0);
        let sep = n.dot(c) - d - r;
        if (sep.gt(zero) & active).any() {
            return true;
        }
    }
    false
}

#[kernel]
fn point_separated_k<'a>(ctx: Gang, planes: &'a [(Vec3, f32)], p: Vec3) -> bool {
    let zero = ctx.splat(0.0);
    let pv = ctx.splat_vec3(p);

    for (off, cnt, active) in ctx.masked_chunks::<f32>(planes.len()) {
        let (n, d) = ctx.gather_plane(&planes[off..off + cnt], 0.0);
        let sep = n.dot(pv) - d;
        if (sep.gt(zero) & active).any() {
            return true;
        }
    }
    false
}

#[kernel]
fn capsule_separated_k<'a>(
    ctx: Gang,
    planes: &'a [(Vec3, f32)],
    p1: Vec3,
    p2: Vec3,
    r: f32,
) -> bool {
    let zero = ctx.splat(0.0);
    let p1v = ctx.splat_vec3(p1);
    let p2v = ctx.splat_vec3(p2);

    for (off, cnt, active) in ctx.masked_chunks::<f32>(planes.len()) {
        let (n, d) = ctx.gather_plane(&planes[off..off + cnt], 0.0);
        let sep = n.dot(p1v).min(n.dot(p2v)) - d - r;
        if (sep.gt(zero) & active).any() {
            return true;
        }
    }
    false
}

#[kernel]
fn cuboid_separated_k<'a>(
    ctx: Gang,
    planes: &'a [(Vec3, f32)],
    center: Vec3,
    axes: [Vec3; 3],
    he: [f32; 3],
) -> bool {
    let zero = ctx.splat(0.0);
    let c = ctx.splat_vec3(center);
    let axes = axes.map(|a| ctx.splat_vec3(a));

    for (off, cnt, active) in ctx.masked_chunks::<f32>(planes.len()) {
        let (n, d) = ctx.gather_plane(&planes[off..off + cnt], 0.0);
        let center_proj = n.dot(c);
        let mut extent_proj = zero;
        for a in 0..3 {
            extent_proj = extent_proj + n.dot(axes[a]).abs() * he[a];
        }
        let sep = center_proj - extent_proj - d;
        if (sep.gt(zero) & active).any() {
            return true;
        }
    }
    false
}
