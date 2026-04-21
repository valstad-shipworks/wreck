#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use glam::Vec3;
use wide::f32x8;

use crate::{
    ArrayConvexPolygon, ArrayConvexPolytope, Capsule, ConvexPolygon, ConvexPolytope, Cuboid,
    Cylinder, LineSegment, Point, Sphere,
};

/// A convex shape's support function: returns the point of the shape that is
/// farthest along `direction`. The direction does not need to be normalized.
///
/// GJK and EPA operate purely through support functions — they never see the
/// shape's interior representation. Every convex shape that participates in
/// generic SDF pair computation must implement this trait.
pub trait SupportFn {
    fn support(&self, direction: Vec3) -> Vec3;
}

#[inline]
fn normalize_or_zero(v: Vec3) -> Vec3 {
    let ls = v.dot(v);
    if ls > f32::EPSILON {
        v * (1.0 / ls.sqrt())
    } else {
        Vec3::ZERO
    }
}

impl SupportFn for Sphere {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        self.center + normalize_or_zero(direction) * self.radius
    }
}

impl SupportFn for Point {
    #[inline]
    fn support(&self, _direction: Vec3) -> Vec3 {
        self.0
    }
}

impl SupportFn for Cuboid {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        let mut p = self.center;
        for i in 0..3 {
            let sign = if direction.dot(self.axes[i]) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            p += self.axes[i] * (sign * self.half_extents[i]);
        }
        p
    }
}

impl SupportFn for Capsule {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        let p2 = self.p2();
        let endpoint = if direction.dot(self.p1) >= direction.dot(p2) {
            self.p1
        } else {
            p2
        };
        endpoint + normalize_or_zero(direction) * self.radius
    }
}

impl SupportFn for Cylinder {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        let p2 = self.p2();
        let endpoint = if direction.dot(self.p1) >= direction.dot(p2) {
            self.p1
        } else {
            p2
        };
        let axis = normalize_or_zero(self.dir);
        let radial = direction - axis * direction.dot(axis);
        endpoint + normalize_or_zero(radial) * self.radius
    }
}

impl SupportFn for LineSegment {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        let p2 = self.p2();
        if direction.dot(self.p1) >= direction.dot(p2) {
            self.p1
        } else {
            p2
        }
    }
}

/// SIMD-accelerated `argmax_v v·direction` over a vertex slice.
///
/// Processes 8 vertices per f32x8 chunk: computes projections with three
/// fused-multiply-adds, reduces the chunk to a scalar best via a short
/// in-register scan, then combines with the running best. The scalar tail
/// handles a partial last chunk. For polytopes with 8+ vertices this is a
/// substantial speedup over the naive scalar scan — each GJK iteration calls
/// support twice, so this is on the hottest path of generic SDF.
#[inline]
fn support_over_vertices(vertices: &[Vec3], direction: Vec3) -> Vec3 {
    if vertices.len() < 8 {
        let mut best = vertices[0];
        let mut best_proj = direction.dot(best);
        for v in &vertices[1..] {
            let p = direction.dot(*v);
            if p > best_proj {
                best_proj = p;
                best = *v;
            }
        }
        return best;
    }

    let dx = f32x8::splat(direction.x);
    let dy = f32x8::splat(direction.y);
    let dz = f32x8::splat(direction.z);

    let chunks = vertices.chunks_exact(8);
    let rem = chunks.remainder();

    let mut best = vertices[0];
    let mut best_proj = direction.dot(best);

    let mut vx = [0.0f32; 8];
    let mut vy = [0.0f32; 8];
    let mut vz = [0.0f32; 8];

    for chunk in chunks {
        for (i, v) in chunk.iter().enumerate() {
            vx[i] = v.x;
            vy[i] = v.y;
            vz[i] = v.z;
        }
        let projs = dx * f32x8::new(vx) + dy * f32x8::new(vy) + dz * f32x8::new(vz);
        let arr = projs.to_array();
        for (i, &p) in arr.iter().enumerate() {
            if p > best_proj {
                best_proj = p;
                best = chunk[i];
            }
        }
    }

    for v in rem {
        let p = direction.dot(*v);
        if p > best_proj {
            best_proj = p;
            best = *v;
        }
    }

    best
}

impl SupportFn for ConvexPolytope {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        support_over_vertices(&self.vertices, direction)
    }
}

impl<const P: usize, const V: usize> SupportFn for ArrayConvexPolytope<P, V> {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        support_over_vertices(&self.vertices, direction)
    }
}

impl SupportFn for ConvexPolygon {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        support_over_vertices(&self.vertices_3d, direction)
    }
}

impl<const V: usize> SupportFn for ArrayConvexPolygon<V> {
    #[inline]
    fn support(&self, direction: Vec3) -> Vec3 {
        support_over_vertices(&self.vertices_3d, direction)
    }
}
