pub(crate) mod array;
pub(crate) mod heap;
pub(crate) mod refer;

use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use glam::Vec3;
use wide::f32x8;

use crate::{Capsule, ConvexPolytope, Cuboid, Sphere, convex_polytope::array::ArrayConvexPolytope};

/// SIMD max projection of vertices onto a normal direction.
#[inline]
pub(crate) fn max_projection(vertices: &[Vec3], normal: Vec3) -> f32 {
    let nx = f32x8::splat(normal.x);
    let ny = f32x8::splat(normal.y);
    let nz = f32x8::splat(normal.z);
    let mut max_val = f32x8::splat(f32::NEG_INFINITY);

    let chunks = vertices.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut vx = [0.0f32; 8];
        let mut vy = [0.0f32; 8];
        let mut vz = [0.0f32; 8];
        for (i, v) in chunk.iter().enumerate() {
            vx[i] = v.x;
            vy[i] = v.y;
            vz[i] = v.z;
        }
        let proj = nx * f32x8::new(vx) + ny * f32x8::new(vy) + nz * f32x8::new(vz);
        max_val = max_val.max(proj);
    }

    let arr = max_val.to_array();
    let mut result = arr[0];
    for &v in &arr[1..] {
        if v > result {
            result = v;
        }
    }
    for v in remainder {
        let p = normal.dot(*v);
        if p > result {
            result = p;
        }
    }
    result
}

/// SIMD min projection of vertices onto a normal direction.
#[inline]
pub(crate) fn min_projection(vertices: &[Vec3], normal: Vec3) -> f32 {
    let nx = f32x8::splat(normal.x);
    let ny = f32x8::splat(normal.y);
    let nz = f32x8::splat(normal.z);
    let mut min_val = f32x8::splat(f32::INFINITY);

    let chunks = vertices.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut vx = [0.0f32; 8];
        let mut vy = [0.0f32; 8];
        let mut vz = [0.0f32; 8];
        for (i, v) in chunk.iter().enumerate() {
            vx[i] = v.x;
            vy[i] = v.y;
            vz[i] = v.z;
        }
        let proj = nx * f32x8::new(vx) + ny * f32x8::new(vy) + nz * f32x8::new(vz);
        min_val = min_val.min(proj);
    }

    let arr = min_val.to_array();
    let mut result = arr[0];
    for &v in &arr[1..] {
        if v < result {
            result = v;
        }
    }
    for v in remainder {
        let p = normal.dot(*v);
        if p < result {
            result = p;
        }
    }
    result
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
    let mut min_proj = [0.0f32; 3];
    let mut max_proj = [0.0f32; 3];
    for i in 0..3 {
        max_proj[i] = max_projection(vertices, axes[i]);
        min_proj[i] = min_projection(vertices, axes[i]);
    }

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

        // Apply rotation to A
        let mut new_a = a;
        for i in 0..3 {
            new_a[i][p] = c * a[i][p] + s * a[i][q];
            new_a[i][q] = -s * a[i][p] + c * a[i][q];
        }
        a = new_a;
        let mut new_a = a;
        for j in 0..3 {
            new_a[p][j] = c * a[p][j] + s * a[q][j];
            new_a[q][j] = -s * a[p][j] + c * a[q][j];
        }
        a = new_a;

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
        let planes: Vec<(Vec3, f32)> = normals_set
            .iter()
            .map(|&n| {
                let d = max_projection(&scaled_verts, n);
                (n, d)
            })
            .collect();

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

        // Side planes from ring directions
        for i in 0..n_ring {
            let angle = core::f32::consts::TAU * i as f32 / n_ring as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            let radial = (ax_u * cos_a + ax_v * sin_a).normalize();
            let d = max_projection(&vertices, radial);
            planes.push((radial, d));

            // Diagonal normals (between radial and forward/backward)
            for &blend_fwd in &[0.5_f32, -0.5] {
                let diag = (radial + ax_fwd * blend_fwd).normalize();
                let d = max_projection(&vertices, diag);
                planes.push((diag, d));
            }
        }

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
