use glam::Vec3;

use inherent::inherent;

use super::refer::RefConvexPolytope;
use super::*;
use crate::cuboid::Cuboid;
use crate::sphere::Sphere;
use crate::wreck_assert;
use crate::{Bounded, Collides, Scalable, Stretchable, Transformable};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvexPolytope {
    pub planes: Vec<(Vec3, f32)>,
    pub vertices: Vec<Vec3>,
    pub obb: Cuboid,
}

impl ConvexPolytope {
    pub fn new(planes: Vec<(Vec3, f32)>, vertices: Vec<Vec3>) -> Self {
        wreck_assert!(
            !planes.is_empty(),
            "ConvexPolytope must have at least one plane"
        );
        wreck_assert!(
            !vertices.is_empty(),
            "ConvexPolytope must have at least one vertex"
        );
        let obb = compute_obb(&vertices);
        Self {
            planes,
            vertices,
            obb,
        }
    }

    /// Construct with a precomputed OBB, skipping the expensive Jacobi eigenvalue iteration.
    pub fn with_obb(planes: Vec<(Vec3, f32)>, vertices: Vec<Vec3>, obb: Cuboid) -> Self {
        wreck_assert!(
            !planes.is_empty(),
            "ConvexPolytope must have at least one plane"
        );
        wreck_assert!(
            !vertices.is_empty(),
            "ConvexPolytope must have at least one vertex"
        );
        Self {
            planes,
            vertices,
            obb,
        }
    }

    #[inline]
    fn as_ref(&self) -> RefConvexPolytope<'_> {
        RefConvexPolytope::from_heap(self)
    }
}

#[inherent]
impl Bounded for ConvexPolytope {
    pub fn broadphase(&self) -> Sphere {
        Sphere::new(self.obb.center, self.obb.bounding_sphere_radius())
    }

    pub fn obb(&self) -> Cuboid {
        self.obb
    }

    pub fn aabb(&self) -> Cuboid {
        self.obb.aabb()
    }
}

#[inherent]
impl Scalable for ConvexPolytope {
    pub fn scale(&mut self, factor: f32) {
        for (_, d) in &mut self.planes {
            *d *= factor;
        }
        for v in &mut self.vertices {
            *v *= factor;
        }
        self.obb.scale(factor);
    }
}

#[inherent]
impl Transformable for ConvexPolytope {
    pub fn translate(&mut self, offset: glam::Vec3A) {
        let off = Vec3::from(offset);
        for (n, d) in &mut self.planes {
            *d += n.dot(off);
        }
        for v in &mut self.vertices {
            *v += off;
        }
        self.obb.translate(offset);
    }

    pub fn rotate_mat(&mut self, mat: glam::Mat3A) {
        for (n, _) in &mut self.planes {
            *n = Vec3::from(mat * glam::Vec3A::from(*n));
        }
        for v in &mut self.vertices {
            *v = Vec3::from(mat * glam::Vec3A::from(*v));
        }
        self.obb.rotate_mat(mat);
    }

    pub fn rotate_quat(&mut self, quat: glam::Quat) {
        for (n, _) in &mut self.planes {
            *n = quat * *n;
        }
        for v in &mut self.vertices {
            *v = quat * *v;
        }
        self.obb.rotate_quat(quat);
    }

    pub fn transform(&mut self, mat: glam::Affine3A) {
        let rot = mat.matrix3;
        let trans = Vec3::from(mat.translation);
        for (n, d) in &mut self.planes {
            *n = Vec3::from(rot * glam::Vec3A::from(*n));
            *d += n.dot(trans);
        }
        for v in &mut self.vertices {
            *v = Vec3::from(mat.transform_point3a(glam::Vec3A::from(*v)));
        }
        self.obb.transform(mat);
    }
}

// ---------------------------------------------------------------------------
// Collision impls: delegate to RefConvexPolytope
// ---------------------------------------------------------------------------

impl Collides<ConvexPolytope> for Sphere {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ConvexPolytope) -> bool {
        other.as_ref().collides_sphere::<BROADPHASE>(self)
    }
}

impl Collides<Sphere> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Sphere) -> bool {
        self.as_ref().collides_sphere::<BROADPHASE>(other)
    }
}

impl Collides<ConvexPolytope> for Cuboid {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ConvexPolytope) -> bool {
        other.as_ref().collides_cuboid::<BROADPHASE>(self)
    }
}

impl Collides<Cuboid> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &Cuboid) -> bool {
        self.as_ref().collides_cuboid::<BROADPHASE>(other)
    }
}

impl Collides<ConvexPolytope> for crate::capsule::Capsule {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ConvexPolytope) -> bool {
        other.as_ref().collides_capsule::<BROADPHASE>(self)
    }
}

impl Collides<crate::capsule::Capsule> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &crate::capsule::Capsule) -> bool {
        self.as_ref().collides_capsule::<BROADPHASE>(other)
    }
}

impl Collides<ConvexPolytope> for ConvexPolytope {
    #[inline]
    fn test<const BROADPHASE: bool>(&self, other: &ConvexPolytope) -> bool {
        self.as_ref()
            .collides_polytope::<BROADPHASE>(&other.as_ref())
    }
}

impl Stretchable for ConvexPolytope {
    type Output = Self;

    fn stretch(&self, translation: Vec3) -> Self::Output {
        // Minkowski sum of convex polytope with line segment [0, translation]
        // Vertices: original + translated
        let mut vertices = Vec::with_capacity(self.vertices.len() * 2);
        for &v in &self.vertices {
            vertices.push(v);
            vertices.push(v + translation);
        }

        // For each original plane (n, d): new d = d + max(0, n·translation)
        let mut planes: Vec<(Vec3, f32)> = self
            .planes
            .iter()
            .map(|&(n, d)| (n, d + n.dot(translation).max(0.0)))
            .collect();

        // Add side planes from silhouette edges: pairs of adjacent faces
        // that straddle the translation direction.
        let mut side_normals: Vec<Vec3> = Vec::new();
        for i in 0..self.planes.len() {
            for j in (i + 1)..self.planes.len() {
                let (n1, _) = self.planes[i];
                let (n2, _) = self.planes[j];
                let dot1 = n1.dot(translation);
                let dot2 = n2.dot(translation);
                if dot1 * dot2 >= 0.0 {
                    continue;
                }
                let edge_dir = n1.cross(n2);
                if edge_dir.length_squared() < 1e-10 {
                    continue;
                }
                let side_n = edge_dir.cross(translation);
                if side_n.length_squared() < 1e-10 {
                    continue;
                }
                let side_n = side_n.normalize();
                side_normals.push(side_n);
                side_normals.push(-side_n);
            }
        }

        for sn in &side_normals {
            let d = max_projection(&vertices, *sn);
            let min_d = min_projection(&vertices, *sn);
            if (d - min_d).abs() > 1e-6 {
                let dominated = planes.iter().any(|&(n, _)| n.dot(*sn) > 0.9999);
                if !dominated {
                    planes.push((*sn, d));
                }
            }
        }

        // Derive OBB analytically from original OBB: same axes, extended extents
        let obb = stretch_obb(&self.obb, translation);

        ConvexPolytope::with_obb(planes, vertices, obb)
    }
}

/// Compute stretched OBB: same axes as original, half-extents grow by |t·axis|/2,
/// center shifts by t/2.
fn stretch_obb(obb: &Cuboid, translation: Vec3) -> Cuboid {
    let mut result = *obb;
    result.center += translation * 0.5;
    for i in 0..3 {
        result.half_extents[i] += translation.dot(obb.axes[i]).abs() * 0.5;
    }
    result
}
