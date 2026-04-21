#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use alloc::vec::Vec;

use glam::Vec3;

use super::gjk::SupportVertex;
use super::support::SupportFn;

const EPA_MAX_ITERATIONS: usize = 64;
const EPA_EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    distance: f32,
}

/// Given a GJK tetrahedral enclosure of the origin in the Minkowski
/// difference, expands the polytope outward until it contacts the surface.
/// Returns the penetration depth — the distance from the origin to the
/// closest face of the Minkowski difference's convex hull.
pub(crate) fn epa_penetration_depth<A, B>(a: &A, b: &B, seed: [SupportVertex; 4]) -> f32
where
    A: SupportFn + ?Sized,
    B: SupportFn + ?Sized,
{
    let mut vertices: Vec<SupportVertex> = seed.to_vec();

    if !ensure_origin_enclosed(&mut vertices) {
        return 0.0;
    }

    let mut faces: Vec<Face> = Vec::with_capacity(16);
    for tri in [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]] {
        faces.push(build_face(&vertices, tri));
    }

    for _ in 0..EPA_MAX_ITERATIONS {
        let mut min_idx = 0usize;
        let mut min_dist = faces[0].distance;
        for (i, f) in faces.iter().enumerate().skip(1) {
            if f.distance < min_dist {
                min_dist = f.distance;
                min_idx = i;
            }
        }

        let closest = faces[min_idx];
        let w = SupportVertex::new(a, b, closest.normal);
        let d = closest.normal.dot(w.w);

        if d - closest.distance < EPA_EPS {
            return closest.distance;
        }

        let new_idx = vertices.len();
        vertices.push(w);

        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut i = 0usize;
        while i < faces.len() {
            if faces[i].normal.dot(w.w - vertices[faces[i].indices[0]].w) > 0.0 {
                for &edge in &[
                    [faces[i].indices[0], faces[i].indices[1]],
                    [faces[i].indices[1], faces[i].indices[2]],
                    [faces[i].indices[2], faces[i].indices[0]],
                ] {
                    add_unique_edge(&mut edges, edge);
                }
                faces.swap_remove(i);
            } else {
                i += 1;
            }
        }

        if edges.is_empty() {
            return closest.distance;
        }

        for edge in &edges {
            faces.push(build_face(&vertices, [edge[0], edge[1], new_idx]));
        }
    }

    let mut min_dist = faces[0].distance;
    for f in faces.iter().skip(1) {
        if f.distance < min_dist {
            min_dist = f.distance;
        }
    }
    min_dist
}

fn build_face(vertices: &[SupportVertex], indices: [usize; 3]) -> Face {
    let a = vertices[indices[0]].w;
    let b = vertices[indices[1]].w;
    let c = vertices[indices[2]].w;
    let mut n = (b - a).cross(c - a);
    let ls = n.dot(n);
    if ls <= 0.0 {
        return Face {
            indices,
            normal: Vec3::ZERO,
            distance: f32::INFINITY,
        };
    }
    n *= 1.0 / ls.sqrt();
    let mut dist = n.dot(a);
    let (normal, distance, fixed_indices) = if dist < 0.0 {
        dist = -dist;
        (-n, dist, [indices[0], indices[2], indices[1]])
    } else {
        (n, dist, indices)
    };
    Face {
        indices: fixed_indices,
        normal,
        distance,
    }
}

fn add_unique_edge(edges: &mut Vec<[usize; 2]>, edge: [usize; 2]) {
    for i in 0..edges.len() {
        if edges[i] == [edge[1], edge[0]] {
            edges.swap_remove(i);
            return;
        }
    }
    edges.push(edge);
}

/// Reorders (and, if needed, flips) the 4 seed vertices so they form a
/// tetrahedron with consistent outward winding that encloses the origin.
/// Returns `false` if the seed is degenerate.
fn ensure_origin_enclosed(vertices: &mut Vec<SupportVertex>) -> bool {
    if vertices.len() < 4 {
        return false;
    }
    let a = vertices[0].w;
    let b = vertices[1].w;
    let c = vertices[2].w;
    let d = vertices[3].w;

    let n = (b - a).cross(c - a);
    if n.dot(d - a) > 0.0 {
        vertices.swap(2, 3);
    }
    true
}
