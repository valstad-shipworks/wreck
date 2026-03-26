use glam::Vec3;

use crate::capsule::segment_segment_dist_sq;

/// Borrowed view of a convex polygon's data, used to share collision logic
/// between `ConvexPolygon` (heap) and `ArrayConvexPolygon` (stack).
#[derive(Debug, Clone, Copy)]
pub struct RefConvexPolygon<'a> {
    pub center: Vec3,
    pub normal: Vec3,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub vertices_2d: &'a [[f32; 2]],
    pub vertices_3d: &'a [Vec3],
    pub edge_normals_2d: &'a [[f32; 2]],
    pub edge_offsets_2d: &'a [f32],
    pub bounding_radius: f32,
}

impl<'a> RefConvexPolygon<'a> {
    #[inline]
    pub fn from_heap(poly: &'a super::ConvexPolygon) -> Self {
        Self {
            center: poly.center,
            normal: poly.normal,
            u_axis: poly.u_axis,
            v_axis: poly.v_axis,
            vertices_2d: &poly.vertices_2d,
            vertices_3d: &poly.vertices_3d,
            edge_normals_2d: &poly.edge_normals_2d,
            edge_offsets_2d: &poly.edge_offsets_2d,
            bounding_radius: poly.bounding_radius,
        }
    }

    #[inline]
    pub fn from_array<const V: usize>(poly: &'a super::ArrayConvexPolygon<V>) -> Self {
        Self {
            center: poly.center,
            normal: poly.normal,
            u_axis: poly.u_axis,
            v_axis: poly.v_axis,
            vertices_2d: &poly.vertices_2d,
            vertices_3d: &poly.vertices_3d,
            edge_normals_2d: &poly.edge_normals_2d,
            edge_offsets_2d: &poly.edge_offsets_2d,
            bounding_radius: poly.bounding_radius,
        }
    }

    #[inline]
    pub fn contains_2d(&self, u: f32, v: f32) -> bool {
        for i in 0..self.edge_normals_2d.len() {
            let n = self.edge_normals_2d[i];
            if n[0] * u + n[1] * v > self.edge_offsets_2d[i] + 1e-6 {
                return false;
            }
        }
        true
    }

    /// Squared distance from a 3D point to the nearest point on the polygon.
    #[inline]
    pub fn point_dist_sq(&self, point: Vec3) -> f32 {
        let d = point - self.center;
        let perp = d.dot(self.normal);
        let u = d.dot(self.u_axis);
        let v = d.dot(self.v_axis);

        if self.contains_2d(u, v) {
            return perp * perp;
        }

        let boundary_dist_sq = self.closest_edge_dist_sq_2d(u, v);
        perp * perp + boundary_dist_sq
    }

    /// Squared 2D distance from (u,v) to the closest point on the polygon boundary.
    fn closest_edge_dist_sq_2d(&self, u: f32, v: f32) -> f32 {
        let n = self.vertices_2d.len();
        let mut min_dist_sq = f32::MAX;
        for i in 0..n {
            let j = (i + 1) % n;
            let ax = self.vertices_2d[i][0];
            let ay = self.vertices_2d[i][1];
            let dx = self.vertices_2d[j][0] - ax;
            let dy = self.vertices_2d[j][1] - ay;
            let len_sq = dx * dx + dy * dy;
            let t = if len_sq > f32::EPSILON {
                ((u - ax) * dx + (v - ay) * dy) / len_sq
            } else {
                0.0
            };
            let t = t.clamp(0.0, 1.0);
            let cx = ax + dx * t;
            let cy = ay + dy * t;
            let dist_sq = (u - cx) * (u - cx) + (v - cy) * (v - cy);
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
            }
        }
        min_dist_sq
    }

    /// Minimum distance squared from a line segment to this polygon.
    pub fn segment_dist_sq(&self, seg_p1: Vec3, seg_dir: Vec3) -> f32 {
        let mut min_dist_sq = f32::MAX;

        min_dist_sq = min_dist_sq.min(self.point_dist_sq(seg_p1));
        min_dist_sq = min_dist_sq.min(self.point_dist_sq(seg_p1 + seg_dir));

        let d1 = (seg_p1 - self.center).dot(self.normal);
        let d2 = d1 + seg_dir.dot(self.normal);
        if d1 * d2 <= 0.0 {
            let denom = d1 - d2;
            if denom.abs() > f32::EPSILON {
                let t = d1 / denom;
                let crossing = seg_p1 + seg_dir * t;
                let cd = crossing - self.center;
                let cu = cd.dot(self.u_axis);
                let cv = cd.dot(self.v_axis);
                if self.contains_2d(cu, cv) {
                    return 0.0;
                }
            }
        }

        let n = self.vertices_3d.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let edge_dir = self.vertices_3d[j] - self.vertices_3d[i];
            let d = segment_segment_dist_sq(seg_p1, seg_dir, self.vertices_3d[i], edge_dir);
            if d < min_dist_sq {
                min_dist_sq = d;
            }
        }

        min_dist_sq
    }

    /// Minimum squared distance from a parametric line to this polygon.
    pub fn parametric_line_dist_sq(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> f32 {
        let mut min_dist_sq = f32::MAX;

        if t_min.is_finite() {
            min_dist_sq = min_dist_sq.min(self.point_dist_sq(origin + dir * t_min));
        }
        if t_max.is_finite() {
            min_dist_sq = min_dist_sq.min(self.point_dist_sq(origin + dir * t_max));
        }

        let d_origin = (origin - self.center).dot(self.normal);
        let d_dir = dir.dot(self.normal);
        if d_dir.abs() > f32::EPSILON {
            let t = -d_origin / d_dir;
            if t >= t_min && t <= t_max {
                let crossing = origin + dir * t;
                let cd = crossing - self.center;
                let cu = cd.dot(self.u_axis);
                let cv = cd.dot(self.v_axis);
                if self.contains_2d(cu, cv) {
                    return 0.0;
                }
            }
        }

        let n = self.vertices_3d.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let edge_dir = self.vertices_3d[j] - self.vertices_3d[i];
            let d = crate::clamped_line_segment_dist_sq(
                origin,
                dir,
                t_min,
                t_max,
                self.vertices_3d[i],
                edge_dir,
            );
            min_dist_sq = min_dist_sq.min(d);
        }

        min_dist_sq
    }

    /// Project all polygon 3D vertices onto an axis, return (min, max).
    #[inline]
    pub fn project_onto(&self, axis: Vec3) -> (f32, f32) {
        let mut min_p = f32::MAX;
        let mut max_p = f32::MIN;
        for &v in self.vertices_3d {
            let p = axis.dot(v);
            if p < min_p {
                min_p = p;
            }
            if p > max_p {
                max_p = p;
            }
        }
        (min_p, max_p)
    }
}

// ---------------------------------------------------------------------------
// Collision helpers that work on RefConvexPolygon
// ---------------------------------------------------------------------------

use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::sphere::Sphere;

#[inline]
pub(crate) fn ref_polygon_sphere_collides(polygon: &RefConvexPolygon, sphere: &Sphere) -> bool {
    let d = sphere.center - polygon.center;
    let max_r = sphere.radius + polygon.bounding_radius;
    if d.dot(d) > max_r * max_r {
        return false;
    }
    polygon.point_dist_sq(sphere.center) <= sphere.radius * sphere.radius
}

#[inline]
pub(crate) fn ref_polygon_capsule_collides(polygon: &RefConvexPolygon, capsule: &Capsule) -> bool {
    let (bc, br) = capsule.bounding_sphere();
    let d = bc - polygon.center;
    let max_r = br + polygon.bounding_radius;
    if d.dot(d) > max_r * max_r {
        return false;
    }
    let dist_sq = polygon.segment_dist_sq(capsule.p1, capsule.dir);
    dist_sq <= capsule.radius * capsule.radius
}

pub(crate) fn ref_polygon_cuboid_collides(polygon: &RefConvexPolygon, cuboid: &Cuboid) -> bool {
    let d = cuboid.center - polygon.center;
    let max_r = polygon.bounding_radius + cuboid.bounding_sphere_radius();
    if d.dot(d) > max_r * max_r {
        return false;
    }

    let eps = 1e-6f32;

    let poly_d = polygon.normal.dot(polygon.center);
    let cuboid_center_proj = polygon.normal.dot(cuboid.center);
    let cuboid_extent = polygon.normal.dot(cuboid.axes[0]).abs() * cuboid.half_extents[0]
        + polygon.normal.dot(cuboid.axes[1]).abs() * cuboid.half_extents[1]
        + polygon.normal.dot(cuboid.axes[2]).abs() * cuboid.half_extents[2];
    if (cuboid_center_proj - poly_d).abs() > cuboid_extent + eps {
        return false;
    }

    for i in 0..3 {
        let axis = cuboid.axes[i];
        let (pmin, pmax) = polygon.project_onto(axis);
        let cproj = axis.dot(cuboid.center);
        let cext = cuboid.half_extents[i];
        if pmin > cproj + cext + eps || pmax < cproj - cext - eps {
            return false;
        }
    }

    let n = polygon.vertices_3d.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let edge = polygon.vertices_3d[j] - polygon.vertices_3d[i];
        for k in 0..3 {
            let axis = edge.cross(cuboid.axes[k]);
            if axis.length_squared() < 1e-10 {
                continue;
            }
            let axis = axis.normalize();
            let (pmin, pmax) = polygon.project_onto(axis);
            let cproj = axis.dot(cuboid.center);
            let cext = axis.dot(cuboid.axes[0]).abs() * cuboid.half_extents[0]
                + axis.dot(cuboid.axes[1]).abs() * cuboid.half_extents[1]
                + axis.dot(cuboid.axes[2]).abs() * cuboid.half_extents[2];
            if pmin > cproj + cext + eps || pmax < cproj - cext - eps {
                return false;
            }
        }
    }

    true
}

pub(crate) fn ref_polygon_polytope_collides(
    polygon: &RefConvexPolygon,
    planes: &[(Vec3, f32)],
    vertices: &[Vec3],
    obb: &Cuboid,
) -> bool {
    let d = obb.center - polygon.center;
    let max_r = polygon.bounding_radius + obb.bounding_sphere_radius();
    if d.dot(d) > max_r * max_r {
        return false;
    }

    let poly_d = polygon.normal.dot(polygon.center);
    let min_poly_proj = crate::convex_polytope::min_projection(vertices, polygon.normal);
    let max_poly_proj = crate::convex_polytope::max_projection(vertices, polygon.normal);
    if min_poly_proj > poly_d || max_poly_proj < poly_d {
        return false;
    }

    let n = polygon.vertices_2d.len();
    for i in 0..n {
        let en = polygon.edge_normals_2d[i];
        let axis = polygon.u_axis * en[0] + polygon.v_axis * en[1];
        let d3d = polygon.edge_offsets_2d[i] + axis.dot(polygon.center);
        let min_proj = crate::convex_polytope::min_projection(vertices, axis);
        if min_proj > d3d {
            return false;
        }
    }

    for &(normal, d) in planes {
        let min_proj = crate::convex_polytope::min_projection(polygon.vertices_3d, normal);
        if min_proj > d {
            return false;
        }
    }

    true
}

pub(crate) fn ref_polygon_infinite_plane_collides(
    polygon: &RefConvexPolygon,
    plane: &super::Plane,
) -> bool {
    crate::convex_polytope::min_projection(polygon.vertices_3d, plane.normal) <= plane.d
}

pub(crate) fn ref_polygon_polygon_collides(a: &RefConvexPolygon, b: &RefConvexPolygon) -> bool {
    let d = b.center - a.center;
    let max_r = a.bounding_radius + b.bounding_radius;
    if d.dot(d) > max_r * max_r {
        return false;
    }

    let eps = 1e-6f32;

    // Both polygon normals (bilateral zero-thickness check)
    for (poly, other_verts) in [(a, b.vertices_3d), (b, a.vertices_3d)] {
        let poly_d = poly.normal.dot(poly.center);
        let min_proj = crate::convex_polytope::min_projection(other_verts, poly.normal);
        let max_proj = crate::convex_polytope::max_projection(other_verts, poly.normal);
        if min_proj > poly_d + eps || max_proj < poly_d - eps {
            return false;
        }
    }

    // Edge normals from both polygons
    for (poly, other_verts) in [(a, b.vertices_3d), (b, a.vertices_3d)] {
        let n = poly.vertices_2d.len();
        for i in 0..n {
            let en = poly.edge_normals_2d[i];
            let axis = poly.u_axis * en[0] + poly.v_axis * en[1];
            let d3d = poly.edge_offsets_2d[i] + axis.dot(poly.center);
            let min_proj = crate::convex_polytope::min_projection(other_verts, axis);
            if min_proj > d3d + eps {
                return false;
            }
        }
    }

    // Cross products of edge directions
    let n_a = a.vertices_3d.len();
    let n_b = b.vertices_3d.len();
    for i in 0..n_a {
        let ei = a.vertices_3d[(i + 1) % n_a] - a.vertices_3d[i];
        for j in 0..n_b {
            let ej = b.vertices_3d[(j + 1) % n_b] - b.vertices_3d[j];
            let axis = ei.cross(ej);
            if axis.length_squared() < 1e-10 {
                continue;
            }
            let axis = axis.normalize();
            let (smin, smax) = a.project_onto(axis);
            let (omin, omax) = b.project_onto(axis);
            if smin > omax + eps || omin > smax + eps {
                return false;
            }
        }
    }

    true
}
