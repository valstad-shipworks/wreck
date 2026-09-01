#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::F32Ext;

use glam::Vec3;
#[cfg(has_quad_load)]
use glam::Vec4;

use crate::capsule::Capsule;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::cuboid::Cuboid;
use crate::cylinder::Cylinder;
use crate::line::{Line, LineSegment, Ray};
use crate::plane::ref_convex::RefConvexPolygon;
use crate::plane::{ArrayConvexPolygon, ConvexPolygon, Plane};
use crate::pointcloud::{PointCloudMarker, Pointcloud};
use crate::sphere::Sphere;
use crate::{Collider, ConvexPolytope};

/// Where a line first meets a shape.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Hit {
    /// Parameter of the hit along the line: `origin + dir * t` for [`Line`] and [`Ray`],
    /// `start + (end - start) * t` for [`LineSegment`]. Direction vectors are not normalized,
    /// so `t` is a distance only when the direction happens to be unit length.
    pub t: f32,
    /// The intersection point.
    pub point: Vec3,
}

/// Where a [`Line`], [`Ray`] or [`LineSegment`] first meets a shape.
///
/// The hit is the smallest parameter within the line's own domain — `(-∞, ∞)` for [`Line`],
/// `[0, ∞)` for [`Ray`], `[0, 1]` for [`LineSegment`] — at which the line lies inside or on
/// the shape. `raycast` therefore returns `Some` exactly when
/// [`collides`](crate::Collides::collides) returns `true`, and a line that starts inside a
/// shape reports its own start point (`t = 0`) rather than the far surface.
///
/// Shapes unbounded in the negative direction have no finite first point — an infinite [`Line`]
/// lying inside a [`Plane`]'s half-space, say. Those report the point of the line closest to
/// `t = 0` that is still inside the shape, keeping the reported position finite.
///
/// Every pair is implemented in both directions, so `ray.raycast(&sphere)` and
/// `sphere.raycast(&ray)` are the same query and both report `t` in the *line's* parameter space.
///
/// Only shapes a line can meet in more than a vanishing set of configurations implement this:
/// the volumes, [`Plane`], [`ConvexPolygon`], [`Pointcloud`] and [`Collider`]. Line-against-line
/// and line-against-[`Point`](crate::Point) pairs are left out for the same reason
/// [`Collides`](crate::Collides) always answers `false` for them.
pub trait Raycast<T> {
    #[must_use]
    fn raycast(&self, other: &T) -> Option<Hit>;
}

/// The span of a constraint that rules nothing out.
const FULL: (f32, f32) = (f32::NEG_INFINITY, f32::INFINITY);

/// How far off a polygon's plane a parallel line may sit and still count as lying in it.
const COPLANAR_TOL: f32 = 1e-6;

#[inline]
fn intersect(a: (f32, f32), b: (f32, f32)) -> Option<(f32, f32)> {
    let lo = a.0.max(b.0);
    let hi = a.1.min(b.1);
    (lo <= hi).then_some((lo, hi))
}

/// Span of `t` satisfying `num + den * t <= 0`.
#[inline]
fn linear_span(num: f32, den: f32) -> Option<(f32, f32)> {
    if den.abs() <= f32::EPSILON {
        return (num <= 0.0).then_some(FULL);
    }
    let t = -num / den;
    Some(if den > 0.0 {
        (f32::NEG_INFINITY, t)
    } else {
        (t, f32::INFINITY)
    })
}

/// Span of `t` satisfying `a t² + b t + c <= 0`, for `a >= 0`.
#[inline]
fn quadratic_span(a: f32, b: f32, c: f32) -> Option<(f32, f32)> {
    if a <= f32::EPSILON {
        return (c <= 0.0).then_some(FULL);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let inv_2a = 0.5 / a;
    Some(((-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a))
}

/// Span of `t` keeping `origin + dir * t` between the two planes `axis · x = 0` and
/// `axis · x = len`, measured from `p1`. `axis` must be unit length.
#[inline]
fn slab_span(w_par: f32, d_par: f32, len: f32) -> Option<(f32, f32)> {
    intersect(
        linear_span(-w_par, -d_par)?,
        linear_span(w_par - len, d_par)?,
    )
}

fn sphere_span(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<(f32, f32)> {
    let a = dir.dot(dir);
    let m = origin - center;
    let r_sq = radius * radius;
    if a <= f32::EPSILON {
        return (m.dot(m) <= r_sq).then_some(FULL);
    }
    // Solved around the closest approach rather than by the raw quadratic formula: `gap` is
    // then a difference of two same-magnitude radii instead of a cancelling `b² - 4ac`.
    let t_mid = -m.dot(dir) / a;
    let closest = m + dir * t_mid;
    let gap = r_sq - closest.dot(closest);
    if gap < 0.0 {
        return None;
    }
    let half = (gap / a).sqrt();
    Some((t_mid - half, t_mid + half))
}

/// Span inside the finite cylinder of radius `radius` around the segment `p1 .. p1 + axis * len`,
/// i.e. the infinite cylinder clipped by the two end planes. `axis` must be unit length.
fn tube_span(
    origin: Vec3,
    dir: Vec3,
    p1: Vec3,
    axis: Vec3,
    len: f32,
    radius: f32,
) -> Option<(f32, f32)> {
    let w = origin - p1;
    let d_par = dir.dot(axis);
    let w_par = w.dot(axis);
    let d_perp = dir - axis * d_par;
    let w_perp = w - axis * w_par;

    let barrel = quadratic_span(
        d_perp.dot(d_perp),
        2.0 * w_perp.dot(d_perp),
        w_perp.dot(w_perp) - radius * radius,
    )?;
    intersect(barrel, slab_span(w_par, d_par, len)?)
}

fn capsule_span(origin: Vec3, dir: Vec3, capsule: &Capsule) -> Option<(f32, f32)> {
    let len_sq = capsule.dir.dot(capsule.dir);
    if len_sq <= f32::EPSILON {
        return sphere_span(origin, dir, capsule.p1, capsule.radius);
    }
    let len = len_sq.sqrt();
    let p2 = capsule.p1 + capsule.dir;

    // A capsule is convex, so the union of its barrel and its two end caps is a single span.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for part in [
        tube_span(
            origin,
            dir,
            capsule.p1,
            capsule.dir / len,
            len,
            capsule.radius,
        ),
        sphere_span(origin, dir, capsule.p1, capsule.radius),
        sphere_span(origin, dir, p2, capsule.radius),
    ]
    .into_iter()
    .flatten()
    {
        lo = lo.min(part.0);
        hi = hi.max(part.1);
    }
    (lo <= hi).then_some((lo, hi))
}

fn cylinder_span(origin: Vec3, dir: Vec3, cylinder: &Cylinder) -> Option<(f32, f32)> {
    let len_sq = cylinder.dir.dot(cylinder.dir);
    if len_sq <= f32::EPSILON {
        return sphere_span(origin, dir, cylinder.p1, cylinder.radius);
    }
    let len = len_sq.sqrt();
    tube_span(
        origin,
        dir,
        cylinder.p1,
        cylinder.dir / len,
        len,
        cylinder.radius,
    )
}

/// Narrows the running window `[lo, hi]` to the parameters satisfying `num + den * t <= 0`,
/// reporting whether any survive.
///
/// A line within `f32::EPSILON` of parallel constrains nothing unless it lies outside, and then
/// nothing survives at all. Dividing by such a denominator would amplify the rounding already in
/// `num` into a boundary anywhere on the line, so the near-parallel case is answered by `num`'s
/// sign alone — the same cutoff [`linear_span`] uses.
#[inline]
fn clip(lo: &mut f32, hi: &mut f32, num: f32, den: f32) -> bool {
    if den < -f32::EPSILON {
        *lo = lo.max(-num / den);
    } else if den > f32::EPSILON {
        *hi = hi.min(-num / den);
    } else if num > 0.0 {
        return false;
    }
    *lo <= *hi
}

/// Narrows `[lo, hi]` to the parameters inside the cuboid, two opposed faces at a time.
#[inline]
fn clip_cuboid(lo: &mut f32, hi: &mut f32, origin: Vec3, dir: Vec3, cuboid: &Cuboid) -> bool {
    let w = origin - cuboid.center;
    for i in 0..3 {
        let w_par = w.dot(cuboid.axes[i]);
        let d_par = dir.dot(cuboid.axes[i]);
        let e = cuboid.half_extents[i];
        if !clip(lo, hi, w_par - e, d_par) || !clip(lo, hi, -w_par - e, -d_par) {
            return false;
        }
    }
    true
}

fn clip_planes_scalar(
    lo: &mut f32,
    hi: &mut f32,
    planes: &[(Vec3, f32)],
    origin: Vec3,
    dir: Vec3,
) -> bool {
    for &(n, d) in planes {
        if !clip(lo, hi, n.dot(origin) - d, n.dot(dir)) {
            return false;
        }
    }
    true
}

/// Whether a plane row is sixteen contiguous bytes holding `n.x, n.y, n.z, d` — the layout
/// [`plane_cols`] reads four rows at a time. Rust makes no promise about tuple field order, so
/// the wide walk is gated on it and falls back to the scalar one if a future layout moves them.
#[cfg(has_quad_load)]
const PLANE_ROW_IS_XYZW: bool = size_of::<(Vec3, f32)>() == 16
    && core::mem::offset_of!((Vec3, f32), 0) == 0
    && core::mem::offset_of!((Vec3, f32), 1) == 12;

/// The four planes at `planes[i..i + 4]`, deinterleaved into `n.x`, `n.y`, `n.z` and `d` columns.
#[cfg(has_quad_load)]
#[inline(always)]
fn plane_cols(planes: &[(Vec3, f32)], i: usize) -> [Vec4; 4] {
    debug_assert!(i + 4 <= planes.len());
    // SAFETY: `PLANE_ROW_IS_XYZW` (checked by the caller) pins the sixteen floats of
    // `planes[i..i + 4]` to the sixteen `f32` slots at `i * 4`, and the four rows are in bounds.
    // Both loads are element-aligned only, which is all either instruction needs.
    let base = unsafe { planes.as_ptr().cast::<f32>().add(i * 4) };
    #[cfg(target_arch = "aarch64")]
    {
        let rows = unsafe { core::arch::aarch64::vld4q_f32(base) };
        [rows.0.into(), rows.1.into(), rows.2.into(), rows.3.into()]
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        unsafe {
            let (r0, r1) = (_mm_loadu_ps(base), _mm_loadu_ps(base.add(4)));
            let (r2, r3) = (_mm_loadu_ps(base.add(8)), _mm_loadu_ps(base.add(12)));
            let (a, b) = (_mm_unpacklo_ps(r0, r1), _mm_unpackhi_ps(r0, r1));
            let (c, d) = (_mm_unpacklo_ps(r2, r3), _mm_unpackhi_ps(r2, r3));
            [
                _mm_movelh_ps(a, c).into(),
                _mm_movehl_ps(c, a).into(),
                _mm_movelh_ps(b, d).into(),
                _mm_movehl_ps(d, b).into(),
            ]
        }
    }
}

/// Narrows `[lo, hi]` to the parameters inside every half-space, four planes per step.
///
/// The window is carried in all four lanes and reduced after each step: the reduction pays for
/// itself by letting a line that has already left the polytope stop within a step of the plane
/// that ruled it out, which is most of the work for a line that misses.
#[cfg(has_quad_load)]
fn clip_planes(
    lo: &mut f32,
    hi: &mut f32,
    planes: &[(Vec3, f32)],
    origin: Vec3,
    dir: Vec3,
) -> bool {
    if !PLANE_ROW_IS_XYZW {
        return clip_planes_scalar(lo, hi, planes, origin, dir);
    }
    let zero = Vec4::ZERO;
    let eps = Vec4::splat(f32::EPSILON);
    let pos_inf = Vec4::splat(f32::INFINITY);
    let neg_inf = Vec4::splat(f32::NEG_INFINITY);
    let (ox, oy, oz) = (
        Vec4::splat(origin.x),
        Vec4::splat(origin.y),
        Vec4::splat(origin.z),
    );
    let (dx, dy, dz) = (Vec4::splat(dir.x), Vec4::splat(dir.y), Vec4::splat(dir.z));

    let (mut near, mut far) = (Vec4::splat(*lo), Vec4::splat(*hi));
    let (mut l, mut h) = (*lo, *hi);
    let quads = planes.len() / 4;
    for q in 0..quads {
        let [nx, ny, nz, d] = plane_cols(planes, q * 4);
        let num = nx * ox + ny * oy + nz * oz - d;
        let den = nx * dx + ny * dy + nz * dz;
        let t = -num / den;
        // Neither bound holds within `f32::EPSILON` of parallel, which leaves `t` unused there;
        // see [`clip`], whose cutoff these two compares reproduce.
        let entering = den.cmplt(-eps);
        let exiting = den.cmpgt(eps);
        // Parallel and outside: no `t` to clip against, the whole window dies instead.
        let outside = num.cmpgt(zero) & !(entering | exiting);
        near = near.max(Vec4::select(entering, t, neg_inf));
        far = far.min(Vec4::select(
            outside,
            neg_inf,
            Vec4::select(exiting, t, pos_inf),
        ));
        l = near.max_element();
        h = far.min_element();
        if l > h {
            return false;
        }
    }
    (*lo, *hi) = (l, h);
    clip_planes_scalar(lo, hi, &planes[quads * 4..], origin, dir)
}

#[cfg(not(has_quad_load))]
use clip_planes_scalar as clip_planes;

fn polytope_hit(
    origin: Vec3,
    dir: Vec3,
    planes: &[(Vec3, f32)],
    obb: &Cuboid,
    t_min: f32,
    t_max: f32,
) -> Option<Hit> {
    let (mut lo, mut hi) = (t_min, t_max);
    (clip_cuboid(&mut lo, &mut hi, origin, dir, obb)
        && clip_planes(&mut lo, &mut hi, planes, origin, dir))
    .then(|| hit_at(origin, dir, lo, hi))
}

fn cuboid_span(origin: Vec3, dir: Vec3, cuboid: &Cuboid) -> Option<(f32, f32)> {
    let (mut lo, mut hi) = FULL;
    clip_cuboid(&mut lo, &mut hi, origin, dir, cuboid).then_some((lo, hi))
}

#[inline]
fn plane_span(origin: Vec3, dir: Vec3, plane: &Plane) -> Option<(f32, f32)> {
    linear_span(plane.normal.dot(origin) - plane.d, plane.normal.dot(dir))
}

fn polygon_span(origin: Vec3, dir: Vec3, poly: RefConvexPolygon<'_>) -> Option<(f32, f32)> {
    let w = origin - poly.center;
    let (ou, ov) = (w.dot(poly.u_axis), w.dot(poly.v_axis));
    let (du, dv) = (dir.dot(poly.u_axis), dir.dot(poly.v_axis));
    let perp = w.dot(poly.normal);
    let den = dir.dot(poly.normal);

    if den.abs() > f32::EPSILON {
        let t = -perp / den;
        return poly.contains_2d(ou + du * t, ov + dv * t).then_some((t, t));
    }

    // Parallel to the polygon's plane: only a line lying in that plane can touch it, and then
    // the overlap is a whole span, clipped edge by edge in the polygon's own 2D frame.
    if perp.abs() > COPLANAR_TOL {
        return None;
    }
    let mut span = FULL;
    for (&normal, &offset) in poly.edge_normals_2d.iter().zip(poly.edge_offsets_2d.iter()) {
        let num = normal[0] * ou + normal[1] * ov - offset;
        let den = normal[0] * du + normal[1] * dv;
        span = intersect(span, linear_span(num, den)?)?;
    }
    Some(span)
}

/// Turns a span of parameters inside a shape into the hit the line reports, or `None` when the
/// span misses the line's own domain.
#[inline]
fn hit(origin: Vec3, dir: Vec3, t_min: f32, t_max: f32, span: Option<(f32, f32)>) -> Option<Hit> {
    let (lo, hi) = intersect(span?, (t_min, t_max))?;
    Some(hit_at(origin, dir, lo, hi))
}

/// The hit a surviving window reports: its start, or the point closest to `t = 0` when the
/// window reaches back to infinity and has no first point.
#[inline]
fn hit_at(origin: Vec3, dir: Vec3, lo: f32, hi: f32) -> Hit {
    let t = if lo.is_finite() {
        lo
    } else {
        0.0f32.clamp(lo, hi)
    };
    Hit {
        t,
        point: origin + dir * t,
    }
}

#[inline]
fn nearer(a: Option<Hit>, b: Option<Hit>) -> Option<Hit> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if y.t < x.t { y } else { x }),
        (Some(x), None) => Some(x),
        (None, y) => y,
    }
}

/// Generates the six `Raycast` impls pairing a shape with the three line types, given a
/// `fn(origin, dir, t_min, t_max, &shape) -> Option<Hit>`.
macro_rules! impl_raycast_with {
    ([$($generics:tt)*] $shape:ty, $probe:expr) => {
        impl<$($generics)*> Raycast<$shape> for Line {
            #[inline]
            fn raycast(&self, shape: &$shape) -> Option<Hit> {
                ($probe)(self.origin, self.dir, f32::NEG_INFINITY, f32::INFINITY, shape)
            }
        }

        impl<$($generics)*> Raycast<$shape> for Ray {
            #[inline]
            fn raycast(&self, shape: &$shape) -> Option<Hit> {
                ($probe)(self.origin, self.dir, 0.0, f32::INFINITY, shape)
            }
        }

        impl<$($generics)*> Raycast<$shape> for LineSegment {
            #[inline]
            fn raycast(&self, shape: &$shape) -> Option<Hit> {
                ($probe)(self.start, self.dir(), 0.0, 1.0, shape)
            }
        }

        impl<$($generics)*> Raycast<Line> for $shape {
            #[inline]
            fn raycast(&self, line: &Line) -> Option<Hit> {
                line.raycast(self)
            }
        }

        impl<$($generics)*> Raycast<Ray> for $shape {
            #[inline]
            fn raycast(&self, ray: &Ray) -> Option<Hit> {
                ray.raycast(self)
            }
        }

        impl<$($generics)*> Raycast<LineSegment> for $shape {
            #[inline]
            fn raycast(&self, segment: &LineSegment) -> Option<Hit> {
                segment.raycast(self)
            }
        }
    };
}

/// The common case: the shape yields a span of parameters it contains.
macro_rules! impl_raycast {
    ([$($generics:tt)*] $shape:ty, $span:expr) => {
        impl_raycast_with!([$($generics)*] $shape,
            |origin, dir, t_min, t_max, shape: &$shape| {
                hit(origin, dir, t_min, t_max, ($span)(origin, dir, shape))
            });
    };
}

impl_raycast!([] Sphere, |o, d, s: &Sphere| sphere_span(o, d, s.center, s.radius));
impl_raycast!([] Capsule, capsule_span);
impl_raycast!([] Cylinder, cylinder_span);
impl_raycast!([] Cuboid, cuboid_span);
impl_raycast!([] Plane, plane_span);
// The polytopes take the line's own domain into the clip rather than intersecting afterwards,
// so a plane that puts the whole line out of range ends the walk there.
impl_raycast_with!([] ConvexPolytope,
|origin, dir, t_min, t_max, p: &ConvexPolytope| polytope_hit(
    origin, dir, &p.planes, &p.obb, t_min, t_max
));
impl_raycast_with!([const P: usize, const V: usize] ArrayConvexPolytope<P, V>,
|origin, dir, t_min, t_max, p: &ArrayConvexPolytope<P, V>| polytope_hit(
    origin, dir, &p.planes, &p.obb, t_min, t_max
));
impl_raycast!([] ConvexPolygon, |o, d, p: &ConvexPolygon| polygon_span(
    o,
    d,
    RefConvexPolygon::from_heap(p)
));
impl_raycast!([const V: usize] ArrayConvexPolygon<V>,
    |o, d, p: &ArrayConvexPolygon<V>| polygon_span(o, d, RefConvexPolygon::from_array(p)));

// A cloud is a union of disjoint balls, not a convex span, so it reports its nearest entry
// parameter directly instead of a span.
impl_raycast_with!([] Pointcloud, |origin, dir, t_min, t_max, pcl: &Pointcloud| {
    pcl.nearest_entry(origin, dir, t_min, t_max)
        .map(|t| Hit { t, point: origin + dir * t })
});

impl<PCL: PointCloudMarker> Collider<PCL> {
    /// Nearest hit on any shape in the collider, in the parameter space of the querying line.
    ///
    /// Points, lines, rays and segments held by the collider are skipped — a line meets those
    /// in no more configurations than [`Collides`](crate::Collides) reports, which is never.
    fn nearest_hit(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> Option<Hit> {
        if self.mask == 0 {
            return None;
        }
        // Bounding-sphere reject. Colliders holding a plane, line or ray carry an infinite
        // radius, which passes everything through as it should.
        hit(
            origin,
            dir,
            t_min,
            t_max,
            sphere_span(origin, dir, self.bounding.center, self.bounding.radius),
        )?;

        let mut best = None;
        let m = self.mask;
        if m & Self::MASK_SPHERES != 0 {
            for s in self.spheres.iter() {
                let span = sphere_span(origin, dir, s.center, s.radius);
                best = nearer(best, hit(origin, dir, t_min, t_max, span));
            }
        }
        if m & Self::MASK_CAPSULES != 0 {
            for c in self.capsules.iter() {
                let span = capsule_span(origin, dir, &c);
                best = nearer(best, hit(origin, dir, t_min, t_max, span));
            }
        }
        if m & Self::MASK_CUBOIDS != 0 {
            for c in self.cuboids.iter() {
                let span = cuboid_span(origin, dir, &c);
                best = nearer(best, hit(origin, dir, t_min, t_max, span));
            }
        }
        if m & Self::MASK_CYLINDERS != 0 {
            for c in self.cylinders.iter() {
                let span = cylinder_span(origin, dir, &c);
                best = nearer(best, hit(origin, dir, t_min, t_max, span));
            }
        }
        if m & Self::MASK_POLYTOPES != 0 {
            for p in self.polytopes.iter() {
                best = nearer(
                    best,
                    polytope_hit(origin, dir, &p.planes, &p.obb, t_min, t_max),
                );
            }
        }
        if m & Self::MASK_POLYGONS != 0 {
            for p in self.polygons.iter() {
                let span = polygon_span(origin, dir, RefConvexPolygon::from_heap(p));
                best = nearer(best, hit(origin, dir, t_min, t_max, span));
            }
        }
        if m & Self::MASK_PLANES != 0 {
            for p in self.planes.iter() {
                let span = plane_span(origin, dir, p);
                best = nearer(best, hit(origin, dir, t_min, t_max, span));
            }
        }
        if m & Self::MASK_POINTCLOUDS != 0 {
            for p in self.pointclouds.iter() {
                let cloud = p.nearest_entry(origin, dir, t_min, t_max).map(|t| Hit {
                    t,
                    point: origin + dir * t,
                });
                best = nearer(best, cloud);
            }
        }
        best
    }
}

impl<PCL: PointCloudMarker> Raycast<Collider<PCL>> for Line {
    #[inline]
    fn raycast(&self, collider: &Collider<PCL>) -> Option<Hit> {
        collider.nearest_hit(self.origin, self.dir, f32::NEG_INFINITY, f32::INFINITY)
    }
}

impl<PCL: PointCloudMarker> Raycast<Collider<PCL>> for Ray {
    #[inline]
    fn raycast(&self, collider: &Collider<PCL>) -> Option<Hit> {
        collider.nearest_hit(self.origin, self.dir, 0.0, f32::INFINITY)
    }
}

impl<PCL: PointCloudMarker> Raycast<Collider<PCL>> for LineSegment {
    #[inline]
    fn raycast(&self, collider: &Collider<PCL>) -> Option<Hit> {
        collider.nearest_hit(self.start, self.dir(), 0.0, 1.0)
    }
}

impl<PCL: PointCloudMarker> Raycast<Line> for Collider<PCL> {
    #[inline]
    fn raycast(&self, line: &Line) -> Option<Hit> {
        line.raycast(self)
    }
}

impl<PCL: PointCloudMarker> Raycast<Ray> for Collider<PCL> {
    #[inline]
    fn raycast(&self, ray: &Ray) -> Option<Hit> {
        ray.raycast(self)
    }
}

impl<PCL: PointCloudMarker> Raycast<LineSegment> for Collider<PCL> {
    #[inline]
    fn raycast(&self, segment: &LineSegment) -> Option<Hit> {
        segment.raycast(self)
    }
}

#[cfg(all(test, has_quad_load))]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    /// The four-at-a-time walk clips to exactly what the plane-at-a-time walk does, at every
    /// plane count from one to twenty — covering an empty wide pass, a bare tail, and both
    /// together.
    #[test]
    fn quad_walk_matches_the_scalar_walk() {
        fn vec(rng: &mut SmallRng, range: f32) -> Vec3 {
            Vec3::new(
                rng.random_range(-range..range),
                rng.random_range(-range..range),
                rng.random_range(-range..range),
            )
        }
        let rng = &mut SmallRng::seed_from_u64(0x5EED);
        let mut survived = 0;
        for count in 1..=20 {
            for _ in 0..2_000 {
                // Half unit normals, half arbitrary, so near-parallel denominators show up too.
                let planes: Vec<(Vec3, f32)> = (0..count)
                    .map(|i| {
                        let n = vec(rng, 1.0);
                        let n = if i % 2 == 0 {
                            n.normalize_or(Vec3::X)
                        } else {
                            n
                        };
                        (n, rng.random_range(-2.0..2.0))
                    })
                    .collect();
                let (origin, dir) = (vec(rng, 4.0), vec(rng, 2.0));
                for (t_min, t_max) in [
                    (f32::NEG_INFINITY, f32::INFINITY),
                    (0.0, f32::INFINITY),
                    (0.0, 1.0),
                ] {
                    let (mut lo, mut hi) = (t_min, t_max);
                    let wide = clip_planes(&mut lo, &mut hi, &planes, origin, dir);
                    let (mut slo, mut shi) = (t_min, t_max);
                    let thin = clip_planes_scalar(&mut slo, &mut shi, &planes, origin, dir);
                    assert_eq!(wide, thin, "{count} planes, {origin} -> {dir}");
                    if wide {
                        survived += 1;
                        assert_eq!((lo, hi), (slo, shi), "{count} planes, {origin} -> {dir}");
                    }
                }
            }
        }
        assert!(survived > 1_000, "only {survived} windows survived");
    }
}
