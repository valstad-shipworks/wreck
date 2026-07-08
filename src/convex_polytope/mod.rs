pub(crate) mod array;
pub(crate) mod heap;
pub(crate) mod refer;

use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use glam::Vec3;
use hydroplane::{Gang, GangGlamExt, kernel};

use crate::{Capsule, ConvexPolytope, Cuboid, Sphere, convex_polytope::array::ArrayConvexPolytope};

/// SIMD min projection of vertices onto a normal direction.
#[inline]
pub(crate) fn min_projection(vertices: &[Vec3], normal: Vec3) -> f32 {
    if vertices.is_empty() {
        return f32::INFINITY;
    }
    min_projection_k(vertices, normal)
}

#[kernel]
pub(crate) fn min_projection_k<'a>(ctx: Gang, verts: &'a [Vec3], normal: Vec3) -> f32 {
    let n = ctx.splat_vec3(normal);
    let pos_inf = ctx.splat(f32::INFINITY);
    let mut acc = pos_inf;
    for (off, cnt, active) in ctx.masked_chunks::<f32>(verts.len()) {
        let proj = ctx.gather_vec3(&verts[off..off + cnt], 0.0).dot(n);
        acc = acc.min(proj.select(active, pos_inf));
    }
    acc.reduce_min()
}

/// Min and max projection of `verts` onto `normal` in one sweep — half the passes of separate
/// min/max calls when both bounds are needed.
#[kernel]
pub(crate) fn minmax_projection_k<'a>(ctx: Gang, verts: &'a [Vec3], normal: Vec3) -> (f32, f32) {
    let n = ctx.splat_vec3(normal);
    let pos_inf = ctx.splat(f32::INFINITY);
    let neg_inf = ctx.splat(f32::NEG_INFINITY);
    let mut min_acc = pos_inf;
    let mut max_acc = neg_inf;
    for (off, cnt, active) in ctx.masked_chunks::<f32>(verts.len()) {
        let proj = ctx.gather_vec3(&verts[off..off + cnt], 0.0).dot(n);
        min_acc = min_acc.min(proj.select(active, pos_inf));
        max_acc = max_acc.max(proj.select(active, neg_inf));
    }
    (min_acc.reduce_min(), max_acc.reduce_max())
}

/// Transpose AoS vertices into one contiguous x/y/z column buffer so repeated direction sweeps
/// use column loads instead of per-lane gathers. Split with [`cols3`]. Worth it from a handful
/// of sweeps up; a single sweep should keep gathering.
pub(crate) fn stage_cols(verts: &[Vec3]) -> Vec<f32> {
    let mut cols = Vec::with_capacity(verts.len() * 3);
    cols.extend(verts.iter().map(|v| v.x));
    cols.extend(verts.iter().map(|v| v.y));
    cols.extend(verts.iter().map(|v| v.z));
    cols
}

/// The three column slices of a [`stage_cols`] buffer.
pub(crate) fn cols3(cols: &[f32]) -> (&[f32], &[f32], &[f32]) {
    let len = cols.len() / 3;
    let (xs, rest) = cols.split_at(len);
    let (ys, zs) = rest.split_at(len);
    (xs, ys, zs)
}

#[kernel]
pub(crate) fn min_projection_cols_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    normal: Vec3,
) -> f32 {
    let n = ctx.splat_vec3(normal);
    let pos_inf = ctx.splat(f32::INFINITY);
    let mut acc = pos_inf;
    for (off, cnt, active) in ctx.masked_chunks::<f32>(xs.len()) {
        let r = off..off + cnt;
        let v = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r]], 0.0);
        acc = acc.min(v.dot(n).select(active, pos_inf));
    }
    acc.reduce_min()
}

#[kernel]
pub(crate) fn max_projection_cols_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    normal: Vec3,
) -> f32 {
    let n = ctx.splat_vec3(normal);
    let neg_inf = ctx.splat(f32::NEG_INFINITY);
    let mut acc = neg_inf;
    for (off, cnt, active) in ctx.masked_chunks::<f32>(xs.len()) {
        let r = off..off + cnt;
        let v = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r]], 0.0);
        acc = acc.max(v.dot(n).select(active, neg_inf));
    }
    acc.reduce_max()
}

/// Fused min/max projection over staged columns; one sweep serves a ± plane-orientation pair
/// (`max((-n)·v) == -min(n·v)` exactly) or an OBB extent.
#[kernel]
pub(crate) fn minmax_projection_cols_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    normal: Vec3,
) -> (f32, f32) {
    let n = ctx.splat_vec3(normal);
    let pos_inf = ctx.splat(f32::INFINITY);
    let neg_inf = ctx.splat(f32::NEG_INFINITY);
    let mut min_acc = pos_inf;
    let mut max_acc = neg_inf;
    for (off, cnt, active) in ctx.masked_chunks::<f32>(xs.len()) {
        let r = off..off + cnt;
        let proj = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r]], 0.0).dot(n);
        min_acc = min_acc.min(proj.select(active, pos_inf));
        max_acc = max_acc.max(proj.select(active, neg_inf));
    }
    (min_acc.reduce_min(), max_acc.reduce_max())
}

/// `ds[i] = max_projection(verts, normals[i])` for every normal behind one dispatch, with the
/// vertices staged column-wise once instead of re-gathered per normal.
#[kernel]
pub(crate) fn max_projections_k<'a>(ctx: Gang, verts: &'a [Vec3], normals: &'a [Vec3], ds: &'a mut [f32]) {
    let cols = stage_cols(verts);
    let (xs, ys, zs) = cols3(&cols);
    for (n, d) in normals.iter().zip(ds.iter_mut()) {
        *d = max_projection_cols_k_on(ctx, xs, ys, zs, *n);
    }
}

/// `minmax[i] = (min, max)` projection of `verts` onto `normals[i]`, staged column-wise once —
/// each fused sweep serves a ± plane pair or an OBB extent in the stretch constructors.
#[kernel]
pub(crate) fn minmax_projections_k<'a>(
    ctx: Gang,
    verts: &'a [Vec3],
    normals: &'a [Vec3],
    minmax: &'a mut [(f32, f32)],
) {
    let cols = stage_cols(verts);
    let (xs, ys, zs) = cols3(&cols);
    for (n, mm) in normals.iter().zip(minmax.iter_mut()) {
        *mm = minmax_projection_cols_k_on(ctx, xs, ys, zs, *n);
    }
}

/// Min and max projections of `verts` onto all three `axes` in one pass over the vertices,
/// so an OBB fit costs one dispatch and one memory sweep instead of six.
#[kernel]
fn minmax_projections3_k<'a>(ctx: Gang, verts: &'a [Vec3], axes: [Vec3; 3]) -> ([f32; 3], [f32; 3]) {
    let n = axes.map(|a| ctx.splat_vec3(a));
    let neg_inf = ctx.splat(f32::NEG_INFINITY);
    let pos_inf = ctx.splat(f32::INFINITY);
    let mut mins = [pos_inf; 3];
    let mut maxs = [neg_inf; 3];
    for (off, cnt, active) in ctx.masked_chunks::<f32>(verts.len()) {
        let v = ctx.gather_vec3(&verts[off..off + cnt], 0.0);
        for i in 0..3 {
            let proj = v.dot(n[i]);
            mins[i] = mins[i].min(proj.select(active, pos_inf));
            maxs[i] = maxs[i].max(proj.select(active, neg_inf));
        }
    }
    (
        [mins[0].reduce_min(), mins[1].reduce_min(), mins[2].reduce_min()],
        [maxs[0].reduce_max(), maxs[1].reduce_max(), maxs[2].reduce_max()],
    )
}

fn compute_obb(vertices: &[Vec3]) -> Cuboid {
    if vertices.is_empty() {
        return Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [0.0; 3]);
    }

    let n = vertices.len() as f32;
    let mean = vertices.iter().copied().sum::<Vec3>() / n;

    // Compute covariance matrix
    let mut cov = [[0.0f32; 3]; 3];
    for v in vertices {
        let d = *v - mean;
        let da = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] += da[i] * da[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] /= n;
            if j != i {
                cov[j][i] = cov[i][j];
            }
        }
    }

    // Jacobi eigenvalue iteration for 3x3 symmetric matrix
    let axes = jacobi_eigenvectors_3x3(cov);

    // Project vertices onto axes to find half-extents (SIMD)
    let (min_proj, max_proj) = minmax_projections3_k(vertices, axes);

    let center_proj: Vec3 = Vec3::new(
        (min_proj[0] + max_proj[0]) * 0.5,
        (min_proj[1] + max_proj[1]) * 0.5,
        (min_proj[2] + max_proj[2]) * 0.5,
    );
    let center = axes[0] * center_proj.x + axes[1] * center_proj.y + axes[2] * center_proj.z;
    let half_extents = [
        (max_proj[0] - min_proj[0]) * 0.5,
        (max_proj[1] - min_proj[1]) * 0.5,
        (max_proj[2] - min_proj[2]) * 0.5,
    ];

    Cuboid::new(center, axes, half_extents)
}

fn jacobi_eigenvectors_3x3(mut a: [[f32; 3]; 3]) -> [Vec3; 3] {
    let mut v = [[0.0f32; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    for _ in 0..50 {
        // Find largest off-diagonal element
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        let theta = 0.5 * (a[q][q] - a[p][p]).atan2(a[p][q]);
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A, columns then rows
        let mut a_col = a;
        for i in 0..3 {
            a_col[i][p] = c * a[i][p] + s * a[i][q];
            a_col[i][q] = -s * a[i][p] + c * a[i][q];
        }
        a = a_col;
        let mut a_row = a;
        for j in 0..3 {
            a_row[p][j] = c * a[p][j] + s * a[q][j];
            a_row[q][j] = -s * a[p][j] + c * a[q][j];
        }
        a = a_row;

        // Apply rotation to V
        let mut new_v = v;
        for i in 0..3 {
            new_v[i][p] = c * v[i][p] + s * v[i][q];
            new_v[i][q] = -s * v[i][p] + c * v[i][q];
        }
        v = new_v;
    }

    [
        Vec3::new(v[0][0], v[1][0], v[2][0]).normalize_or_zero(),
        Vec3::new(v[0][1], v[1][1], v[2][1]).normalize_or_zero(),
        Vec3::new(v[0][2], v[1][2], v[2][2]).normalize_or_zero(),
    ]
}

use crate::Collides;
use refer::RefConvexPolytope;

impl<const P: usize, const V: usize> Collides<ConvexPolytope> for ArrayConvexPolytope<P, V> {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ConvexPolytope) -> bool {
        RefConvexPolytope::from_array(self)
            .collides_polytope::<BROADPHASE>(&RefConvexPolytope::from_heap(other))
    }
}

impl<const P: usize, const V: usize> Collides<ArrayConvexPolytope<P, V>> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ArrayConvexPolytope<P, V>) -> bool {
        RefConvexPolytope::from_heap(self)
            .collides_polytope::<BROADPHASE>(&RefConvexPolytope::from_array(other))
    }
}

/// Approximate a sphere as a convex polytope using an icosphere with 42 vertices.
impl From<Sphere> for ConvexPolytope {
    fn from(sphere: Sphere) -> Self {
        // Generate icosphere vertices (12 base + 30 edge midpoints = 42 vertices)
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let len = (1.0 + phi * phi).sqrt();
        let a = 1.0 / len;
        let b = phi / len;

        // 12 icosahedron vertices (normalized to unit sphere)
        let ico = [
            Vec3::new(-a, b, 0.0),
            Vec3::new(a, b, 0.0),
            Vec3::new(-a, -b, 0.0),
            Vec3::new(a, -b, 0.0),
            Vec3::new(0.0, -a, b),
            Vec3::new(0.0, a, b),
            Vec3::new(0.0, -a, -b),
            Vec3::new(0.0, a, -b),
            Vec3::new(b, 0.0, -a),
            Vec3::new(b, 0.0, a),
            Vec3::new(-b, 0.0, -a),
            Vec3::new(-b, 0.0, a),
        ];

        // 20 icosahedron faces (indices)
        let faces: [[usize; 3]; 20] = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        // Subdivide once: each face -> 4 faces, project midpoints onto unit sphere
        let mut vertices = Vec::new();
        let mut normals_set: Vec<Vec3> = Vec::new();

        #[cfg(feature = "std")]
        let mut get_or_insert = {
            let mut vert_map = std::collections::HashMap::new();
            move |v: Vec3, verts: &mut Vec<Vec3>| -> usize {
                let key = ((v.x * 1e5) as i32, (v.y * 1e5) as i32, (v.z * 1e5) as i32);
                *vert_map.entry(key).or_insert_with(|| {
                    let idx = verts.len();
                    verts.push(v);
                    idx
                })
            }
        };

        #[cfg(not(feature = "std"))]
        let get_or_insert = |v: Vec3, verts: &mut Vec<Vec3>| -> usize {
            let key = ((v.x * 1e5) as i32, (v.y * 1e5) as i32, (v.z * 1e5) as i32);
            for (i, existing) in verts.iter().enumerate() {
                let ek = (
                    (existing.x * 1e5) as i32,
                    (existing.y * 1e5) as i32,
                    (existing.z * 1e5) as i32,
                );
                if ek == key {
                    return i;
                }
            }
            let idx = verts.len();
            verts.push(v);
            idx
        };

        let mut sub_faces: Vec<[usize; 3]> = Vec::new();
        for face in &faces {
            let v0 = ico[face[0]];
            let v1 = ico[face[1]];
            let v2 = ico[face[2]];
            let m01 = ((v0 + v1) * 0.5).normalize();
            let m12 = ((v1 + v2) * 0.5).normalize();
            let m20 = ((v2 + v0) * 0.5).normalize();

            let i0 = get_or_insert(v0, &mut vertices);
            let i1 = get_or_insert(v1, &mut vertices);
            let i2 = get_or_insert(v2, &mut vertices);
            let i01 = get_or_insert(m01, &mut vertices);
            let i12 = get_or_insert(m12, &mut vertices);
            let i20 = get_or_insert(m20, &mut vertices);

            sub_faces.push([i0, i01, i20]);
            sub_faces.push([i01, i1, i12]);
            sub_faces.push([i20, i12, i2]);
            sub_faces.push([i01, i12, i20]);
        }

        // Compute face normals as plane normals
        for face in &sub_faces {
            let v0 = vertices[face[0]];
            let v1 = vertices[face[1]];
            let v2 = vertices[face[2]];
            let n = (v1 - v0).cross(v2 - v0);
            if n.length_squared() > 1e-10 {
                let n = n.normalize();
                // Ensure outward-facing
                let n = if n.dot(v0) > 0.0 { n } else { -n };
                if !normals_set.iter().any(|existing| existing.dot(n) > 0.9999) {
                    normals_set.push(n);
                }
            }
        }

        // Scale and translate vertices
        let scaled_verts: Vec<Vec3> = vertices
            .iter()
            .map(|v| sphere.center + *v * sphere.radius)
            .collect();

        // Build planes: for each normal, d = n·center + radius (since vertices are on the sphere surface)
        let mut ds = vec![0.0f32; normals_set.len()];
        max_projections_k(&scaled_verts, &normals_set, &mut ds);
        let planes: Vec<(Vec3, f32)> = normals_set.iter().copied().zip(ds).collect();

        let obb = Cuboid::new(
            sphere.center,
            [Vec3::X, Vec3::Y, Vec3::Z],
            [sphere.radius; 3],
        );

        ConvexPolytope::with_obb(planes, scaled_verts, obb)
    }
}

impl From<Cuboid> for ConvexPolytope {
    fn from(cuboid: Cuboid) -> Self {
        // 6 face normals (positive and negative for each axis)
        let planes = vec![
            (
                cuboid.axes[0],
                cuboid.axes[0].dot(cuboid.center) + cuboid.half_extents[0],
            ),
            (
                -cuboid.axes[0],
                (-cuboid.axes[0]).dot(cuboid.center) + cuboid.half_extents[0],
            ),
            (
                cuboid.axes[1],
                cuboid.axes[1].dot(cuboid.center) + cuboid.half_extents[1],
            ),
            (
                -cuboid.axes[1],
                (-cuboid.axes[1]).dot(cuboid.center) + cuboid.half_extents[1],
            ),
            (
                cuboid.axes[2],
                cuboid.axes[2].dot(cuboid.center) + cuboid.half_extents[2],
            ),
            (
                -cuboid.axes[2],
                (-cuboid.axes[2]).dot(cuboid.center) + cuboid.half_extents[2],
            ),
        ];

        // 8 corner vertices
        let mut vertices = Vec::with_capacity(8);
        for &sx in &[-1.0_f32, 1.0] {
            for &sy in &[-1.0_f32, 1.0] {
                for &sz in &[-1.0_f32, 1.0] {
                    vertices.push(
                        cuboid.center
                            + cuboid.axes[0] * (sx * cuboid.half_extents[0])
                            + cuboid.axes[1] * (sy * cuboid.half_extents[1])
                            + cuboid.axes[2] * (sz * cuboid.half_extents[2]),
                    );
                }
            }
        }

        ConvexPolytope::with_obb(planes, vertices, cuboid)
    }
}

impl From<Capsule> for ConvexPolytope {
    fn from(capsule: Capsule) -> Self {
        // Approximate capsule as a convex hull of two hemispheres.
        // Use a ring of vertices around each endpoint plus the endpoints themselves.
        let p1 = capsule.p1;
        let p2 = capsule.p2();
        let dir = capsule.dir;
        let dir_len = dir.length();

        // Build a local frame
        let (ax_fwd, ax_u, ax_v) = if dir_len > 1e-6 {
            let fwd = dir / dir_len;
            let up = if fwd.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let u = fwd.cross(up).normalize();
            let v = u.cross(fwd);
            (fwd, u, v)
        } else {
            // Degenerate capsule (point-like) → use sphere conversion
            return ConvexPolytope::from(Sphere::new(p1, capsule.radius));
        };

        let r = capsule.radius;
        let n_ring = 8;
        let mut vertices = Vec::new();

        // Hemisphere vertices at p1 (backward hemisphere)
        vertices.push(p1 - ax_fwd * r); // pole
        for i in 0..n_ring {
            let angle = core::f32::consts::TAU * i as f32 / n_ring as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            // Equator
            vertices.push(p1 + (ax_u * cos_a + ax_v * sin_a) * r);
            // 45-degree ring toward back pole
            let lat = core::f32::consts::FRAC_PI_4;
            vertices.push(
                p1 - ax_fwd * (r * lat.sin()) + (ax_u * cos_a + ax_v * sin_a) * (r * lat.cos()),
            );
        }

        // Hemisphere vertices at p2 (forward hemisphere)
        vertices.push(p2 + ax_fwd * r); // pole
        for i in 0..n_ring {
            let angle = core::f32::consts::TAU * i as f32 / n_ring as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            // Equator
            vertices.push(p2 + (ax_u * cos_a + ax_v * sin_a) * r);
            // 45-degree ring toward front pole
            let lat = core::f32::consts::FRAC_PI_4;
            vertices.push(
                p2 + ax_fwd * (r * lat.sin()) + (ax_u * cos_a + ax_v * sin_a) * (r * lat.cos()),
            );
        }

        // Build planes from unique outward normals
        // End caps
        let mut planes: Vec<(Vec3, f32)> = vec![
            (ax_fwd, ax_fwd.dot(p2) + r),
            (-ax_fwd, (-ax_fwd).dot(p1) + r),
        ];

        // Side planes from ring directions, plus diagonal normals between radial and
        // forward/backward
        let mut side_normals: Vec<Vec3> = Vec::with_capacity(n_ring * 3);
        for i in 0..n_ring {
            let angle = core::f32::consts::TAU * i as f32 / n_ring as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            let radial = (ax_u * cos_a + ax_v * sin_a).normalize();
            side_normals.push(radial);
            for &blend_fwd in &[0.5_f32, -0.5] {
                side_normals.push((radial + ax_fwd * blend_fwd).normalize());
            }
        }
        let mut ds = vec![0.0f32; side_normals.len()];
        max_projections_k(&vertices, &side_normals, &mut ds);
        planes.extend(side_normals.into_iter().zip(ds));

        let obb = compute_obb(&vertices);
        ConvexPolytope::with_obb(planes, vertices, obb)
    }
}

impl<const P: usize, const V: usize> From<ArrayConvexPolytope<P, V>> for ConvexPolytope {
    fn from(polytope: ArrayConvexPolytope<P, V>) -> Self {
        ConvexPolytope::with_obb(
            polytope.planes.to_vec(),
            polytope.vertices.to_vec(),
            polytope.obb,
        )
    }
}
