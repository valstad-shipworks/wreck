use alloc::vec::Vec;
use core::fmt::{self, Debug, Display};

use glam::{Mat3, Vec3};
use hydroplane::{Gang, GangGlamExt, MAX_LANES, Vec3Wide, kernel};

use crate::shape_soa::{ShapeSoa, SoaShape};
use crate::{Bounded, Collides, Scalable, Sphere, Transformable};

/// Structure-of-Arrays storage for spheres.
///
/// Backed by a single contiguous allocation laid out as `[x; padded][y; padded][z; padded][r; padded]`
/// so that all four channels share one cache-line neighbourhood.
/// Each channel is padded to a multiple of `MAX_LANES` so SIMD loops never need a
/// scalar remainder path, whatever lane width the dispatched backend uses.
/// Padding slots use `r = NaN` so that SIMD lane comparisons
/// (`dist_sq <= (r_a + r_b)²`) always return false, preventing false positives.
#[derive(Clone, PartialEq)]
pub struct SpheresSoA {
    buf: Vec<f32>,
    padded: usize,
    len: usize,
}

impl Debug for SpheresSoA {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpheresSoA")
            .field("x", &self.x())
            .field("y", &self.y())
            .field("z", &self.z())
            .field("r", &self.r())
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SpheresSoA {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("SpheresSoA", 5)?;
        s.serialize_field("x", self.x())?;
        s.serialize_field("y", self.y())?;
        s.serialize_field("z", self.z())?;
        s.serialize_field("r", self.r())?;
        s.serialize_field("len", &self.len)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SpheresSoA {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Helper {
            x: Vec<f32>,
            y: Vec<f32>,
            z: Vec<f32>,
            r: Vec<f32>,
            len: usize,
        }
        let h = Helper::deserialize(deserializer)?;
        let padded = h.x.len();
        let mut buf = Vec::with_capacity(4 * padded);
        buf.extend_from_slice(&h.x);
        buf.extend_from_slice(&h.y);
        buf.extend_from_slice(&h.z);
        buf.extend_from_slice(&h.r);
        Ok(Self {
            buf,
            padded,
            len: h.len,
        })
    }
}

impl Display for SpheresSoA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpheresSoA(len: {})", self.len)
    }
}

const PAD: usize = MAX_LANES;
const PAD_MASK: usize = !(PAD - 1);

#[inline]
fn pad(n: usize) -> usize {
    (n + PAD - 1) & PAD_MASK
}

impl SpheresSoA {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            padded: 0,
            len: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        let padded = pad(cap);
        Self {
            buf: Vec::with_capacity(4 * padded),
            padded: 0,
            len: 0,
        }
    }

    pub fn from_slice(spheres: &[Sphere]) -> Self {
        let len = spheres.len();
        let padded = pad(len);
        let mut buf = vec![0.0f32; 4 * padded];

        for (i, s) in spheres.iter().enumerate() {
            buf[i] = s.center.x;
            buf[padded + i] = s.center.y;
            buf[2 * padded + i] = s.center.z;
            buf[3 * padded + i] = s.radius;
        }

        Self { buf, padded, len }
    }

    pub fn from_bounded<T: Bounded>(items: &[T]) -> Self {
        let len = items.len();
        let padded = pad(len);
        let mut buf = vec![0.0f32; 4 * padded];

        for (i, item) in items.iter().enumerate() {
            let bp = item.broadphase();
            buf[i] = bp.center.x;
            buf[padded + i] = bp.center.y;
            buf[2 * padded + i] = bp.center.z;
            buf[3 * padded + i] = bp.radius;
        }

        Self { buf, padded, len }
    }

    #[inline]
    pub fn x(&self) -> &[f32] {
        &self.buf[..self.len]
    }

    #[inline]
    pub fn y(&self) -> &[f32] {
        &self.buf[self.padded..self.padded + self.len]
    }

    #[inline]
    pub fn z(&self) -> &[f32] {
        &self.buf[2 * self.padded..2 * self.padded + self.len]
    }

    #[inline]
    pub fn r(&self) -> &[f32] {
        &self.buf[3 * self.padded..3 * self.padded + self.len]
    }

    #[inline]
    pub fn slices_mut(&mut self) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        let p = self.padded;
        let len = self.len;
        let (x, rest) = self.buf.split_at_mut(p);
        let (y, rest) = rest.split_at_mut(p);
        let (z, r) = rest.split_at_mut(p);
        (&mut x[..len], &mut y[..len], &mut z[..len], &mut r[..len])
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, sphere: Sphere) {
        if self.len == self.padded {
            self.grow();
        }
        let p = self.padded;
        self.buf[self.len] = sphere.center.x;
        self.buf[p + self.len] = sphere.center.y;
        self.buf[2 * p + self.len] = sphere.center.z;
        self.buf[3 * p + self.len] = sphere.radius;
        self.len += 1;
    }

    fn grow(&mut self) {
        let old = self.padded;
        let new = old + PAD;
        let mut buf = vec![0.0f32; 4 * new];
        if old > 0 {
            buf[..old].copy_from_slice(&self.buf[..old]);
            buf[new..new + old].copy_from_slice(&self.buf[old..2 * old]);
            buf[2 * new..2 * new + old].copy_from_slice(&self.buf[2 * old..3 * old]);
            buf[3 * new..3 * new + old].copy_from_slice(&self.buf[3 * old..4 * old]);
        }
        self.buf = buf;
        self.padded = new;
    }

    /// Moves all real entries from `other` into `self`, re-padding once.
    pub fn append(&mut self, other: &mut Self) {
        if other.len == 0 {
            return;
        }
        let new_len = self.len + other.len;
        let new_padded = pad(new_len);
        let mut buf = vec![0.0f32; 4 * new_padded];

        let sp = self.padded;
        let op = other.padded;
        let sl = self.len;
        let ol = other.len;

        buf[..sl].copy_from_slice(&self.buf[..sl]);
        buf[sl..sl + ol].copy_from_slice(&other.buf[..ol]);

        buf[new_padded..new_padded + sl].copy_from_slice(&self.buf[sp..sp + sl]);
        buf[new_padded + sl..new_padded + sl + ol].copy_from_slice(&other.buf[op..op + ol]);

        buf[2 * new_padded..2 * new_padded + sl].copy_from_slice(&self.buf[2 * sp..2 * sp + sl]);
        buf[2 * new_padded + sl..2 * new_padded + sl + ol]
            .copy_from_slice(&other.buf[2 * op..2 * op + ol]);

        buf[3 * new_padded..3 * new_padded + sl].copy_from_slice(&self.buf[3 * sp..3 * sp + sl]);
        buf[3 * new_padded + sl..3 * new_padded + sl + ol]
            .copy_from_slice(&other.buf[3 * op..3 * op + ol]);

        self.buf = buf;
        self.padded = new_padded;
        self.len = new_len;
        other.clear();
    }

    pub fn extend_from(&mut self, other: &Self) {
        if other.len == 0 {
            return;
        }
        let new_len = self.len + other.len;
        let new_padded = pad(new_len);
        let mut buf = vec![0.0f32; 4 * new_padded];

        let sp = self.padded;
        let op = other.padded;
        let sl = self.len;
        let ol = other.len;

        buf[..sl].copy_from_slice(&self.buf[..sl]);
        buf[sl..sl + ol].copy_from_slice(&other.buf[..ol]);

        buf[new_padded..new_padded + sl].copy_from_slice(&self.buf[sp..sp + sl]);
        buf[new_padded + sl..new_padded + sl + ol].copy_from_slice(&other.buf[op..op + ol]);

        buf[2 * new_padded..2 * new_padded + sl].copy_from_slice(&self.buf[2 * sp..2 * sp + sl]);
        buf[2 * new_padded + sl..2 * new_padded + sl + ol]
            .copy_from_slice(&other.buf[2 * op..2 * op + ol]);

        buf[3 * new_padded..3 * new_padded + sl].copy_from_slice(&self.buf[3 * sp..3 * sp + sl]);
        buf[3 * new_padded + sl..3 * new_padded + sl + ol]
            .copy_from_slice(&other.buf[3 * op..3 * op + ol]);

        self.buf = buf;
        self.padded = new_padded;
        self.len = new_len;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Replaces the contents of `self` with those of `other`, reusing the
    /// existing allocation when the current capacity permits.
    #[inline]
    pub fn clone_from(&mut self, other: &Self) {
        self.buf.clone_from(&other.buf);
        self.padded = other.padded;
        self.len = other.len;
    }

    #[inline]
    pub fn get(&self, index: usize) -> Sphere {
        debug_assert!(index < self.len);
        let p = self.padded;
        Sphere::new(
            Vec3::new(
                self.buf[index],
                self.buf[p + index],
                self.buf[2 * p + index],
            ),
            self.buf[3 * p + index],
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = Sphere> + '_ {
        (0..self.len).map(|i| self.get(i))
    }

    /// Test if any sphere in this SoA collides with the given sphere.
    #[inline]
    pub fn any_collides_sphere(&self, sphere: &Sphere) -> bool {
        if self.is_empty() {
            return false;
        }
        any_collides_sphere_k(self.x(), self.y(), self.z(), self.r(), sphere.center, sphere.radius)
    }

    /// SIMD count of stored bounding spheres overlapping `query`.
    pub fn count_overlaps(&self, query: &Sphere) -> usize {
        if self.is_empty() {
            return 0;
        }
        count_overlaps_k(self.x(), self.y(), self.z(), self.r(), query.center, query.radius, self.len) as usize
    }

    /// SIMD broadphase: mark which spheres overlap `query`.
    ///
    /// Sets `out[i] = true` for each sphere `i` (up to `self.len`) whose
    /// bounding sphere overlaps `query`. Clears all entries first.
    /// Returns `true` if any hit was found.
    /// SIMD broadphase: mark which spheres overlap `query`.
    ///
    /// Sets `out[i]` to `true` for each sphere `i` (up to `self.len`) whose
    /// bounding sphere overlaps `query`. Clears all entries first.
    /// Returns `true` if any hit was found.
    #[inline]
    pub fn broadphase_collect(&self, query: &Sphere, out: &mut [bool]) -> bool {
        debug_assert!(out.len() >= self.len);
        out[..self.len].fill(false);

        if self.is_empty() {
            return false;
        }
        broadphase_collect_k(self.x(), self.y(), self.z(), self.r(), query.center, query.radius, self.len, out)
    }

    /// Test if any sphere in `self` collides with any sphere in `other`.
    ///
    /// For each sphere in `self`, broadcasts its position across all chunks
    /// of `other` — O(n*m) but with no per-chunk transpose overhead.
    pub fn any_collides_soa(&self, other: &SpheresSoA) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        any_collides_soa_k(
            self.x(),
            self.y(),
            self.z(),
            self.r(),
            other.x(),
            other.y(),
            other.z(),
            other.r(),
        )
    }
}

impl Transformable for SpheresSoA {
    fn translate(&mut self, offset: glam::Vec3A) {
        let (xs, ys, zs, _) = self.slices_mut();
        translate_k(xs, ys, zs, Vec3::from(offset));
    }

    fn rotate_mat(&mut self, mat: glam::Mat3A) {
        let (xs, ys, zs, _) = self.slices_mut();
        rotate_mat_k(xs, ys, zs, Mat3::from(mat));
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        self.rotate_mat(glam::Mat3A::from_quat(quat));
    }

    fn transform(&mut self, mat: glam::Affine3A) {
        let (xs, ys, zs, _) = self.slices_mut();
        transform_k(xs, ys, zs, Mat3::from(mat.matrix3), Vec3::from(mat.translation));
    }
}

impl Scalable for SpheresSoA {
    /// Scale all radii by a factor.
    fn scale(&mut self, factor: f32) {
        let (_, _, _, rs) = self.slices_mut();
        scale_k(rs, factor);
    }
}

/// Sphere-vs-SoA broadphase: is the query within `(query.r + r[i])` of any stored centre?
#[kernel]
fn any_collides_sphere_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    center: Vec3,
    radius: f32,
) -> bool {
    let c = ctx.splat_vec3(center);
    let sr = ctx.splat(radius);
    ctx.any_n([xs, ys, zs, rs], |[x, y, z, r]| {
        let rsum = sr + r;
        (c - Vec3Wide::from([x, y, z])).length_squared().le(rsum * rsum)
    })
}

/// Count how many stored bounding spheres overlap the query — the density signal that
/// chooses between the broad-gated scalar scan and the SIMD batch narrowphase.
#[kernel]
fn count_overlaps_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    center: Vec3,
    radius: f32,
    len: usize,
) -> f32 {
    let c = ctx.splat_vec3(center);
    let sr = ctx.splat(radius);
    ctx.count_n([&xs[..len], &ys[..len], &zs[..len], &rs[..len]], |[x, y, z, r]| {
        let rsum = sr + r;
        (c - Vec3Wide::from([x, y, z])).length_squared().le(rsum * rsum)
    }) as f32
}

/// Like [`any_collides_sphere_k`] but records every overlapping index into `out`.
#[kernel]
fn broadphase_collect_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    center: Vec3,
    radius: f32,
    len: usize,
    out: &'a mut [bool],
) -> bool {
    let c = ctx.splat_vec3(center);
    let sr = ctx.splat(radius);
    ctx.for_each_hit_n(
        [&xs[..len], &ys[..len], &zs[..len], &rs[..len]],
        |[x, y, z, r]| {
            let rsum = sr + r;
            (c - Vec3Wide::from([x, y, z])).length_squared().le(rsum * rsum)
        },
        |idx| out[idx] = true,
    )
}

/// O(n·m) SoA-vs-SoA: each `a` centre broadcast across every `b` chunk.
#[kernel]
fn any_collides_soa_k<'a>(
    ctx: Gang,
    axs: &'a [f32],
    ays: &'a [f32],
    azs: &'a [f32],
    ars: &'a [f32],
    bxs: &'a [f32],
    bys: &'a [f32],
    bzs: &'a [f32],
    brs: &'a [f32],
) -> bool {
    for i in 0..axs.len() {
        let c = ctx.splat_vec3(Vec3::new(axs[i], ays[i], azs[i]));
        let sr = ctx.splat(ars[i]);
        let hit = ctx.any_n([bxs, bys, bzs, brs], |[x, y, z, r]| {
            let rsum = sr + r;
            (c - Vec3Wide::from([x, y, z])).length_squared().le(rsum * rsum)
        });
        if hit {
            return true;
        }
    }
    false
}

#[kernel]
fn translate_k<'a>(ctx: Gang, xs: &'a mut [f32], ys: &'a mut [f32], zs: &'a mut [f32], off: Vec3) {
    let off = ctx.splat_vec3(off);
    ctx.map_n::<f32, 3>([xs, ys, zs], 0.0, |[x, y, z]| (Vec3Wide::from([x, y, z]) + off).0);
}

#[kernel]
fn rotate_mat_k<'a>(ctx: Gang, xs: &'a mut [f32], ys: &'a mut [f32], zs: &'a mut [f32], m: Mat3) {
    let m = ctx.splat_mat3(m);
    ctx.map_n::<f32, 3>([xs, ys, zs], 0.0, |[x, y, z]| m.mul_vec3(Vec3Wide::from([x, y, z])).0);
}

#[kernel]
fn transform_k<'a>(ctx: Gang, xs: &'a mut [f32], ys: &'a mut [f32], zs: &'a mut [f32], m: Mat3, t: Vec3) {
    let m = ctx.splat_mat3(m);
    let t = ctx.splat_vec3(t);
    ctx.map_n::<f32, 3>([xs, ys, zs], 0.0, |[x, y, z]| m.mul_add(Vec3Wide::from([x, y, z]), t).0);
}

#[kernel]
fn scale_k<'a>(ctx: Gang, rs: &'a mut [f32], factor: f32) {
    let f = ctx.splat(factor);
    ctx.map_n::<f32, 1>([rs], 0.0, |[r]| [r * f]);
}

impl Default for SpheresSoA {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&[Sphere]> for SpheresSoA {
    fn from(spheres: &[Sphere]) -> Self {
        Self::from_slice(spheres)
    }
}

impl From<Vec<Sphere>> for SpheresSoA {
    fn from(spheres: Vec<Sphere>) -> Self {
        Self::from_slice(&spheres)
    }
}

#[derive(Debug, Clone)]
pub struct BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    pub(crate) items: Vec<T>,
    pub(crate) broad: SpheresSoA,
}

impl<T> Default for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    fn default() -> Self {
        Self {
            items: Vec::new(),
            broad: SpheresSoA::new(),
        }
    }
}

impl<T> BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    pub fn new(items: Vec<T>) -> Self {
        let broad = SpheresSoA::from_bounded(&items);
        Self { items, broad }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            items: Vec::with_capacity(cap),
            broad: SpheresSoA::with_capacity(cap),
        }
    }

    pub fn extend_from_slice(&mut self, items: &[T]) {
        self.items.reserve(items.len());
        for item in items {
            self.push(item.clone());
        }
    }

    pub fn push(&mut self, item: T) {
        self.broad.push(item.broadphase());
        self.items.push(item);
    }

    pub fn extend(&mut self, items: impl IntoIterator<Item = T>) {
        for item in items {
            self.push(item);
        }
    }

    /// Moves all items from `other` into `self` in bulk.
    pub fn append(&mut self, other: &mut Self) {
        self.items.append(&mut other.items);
        self.broad.append(&mut other.broad);
    }

    /// Replaces the contents of `self` with those of `other`, reusing the
    /// existing allocations when capacity allows.
    #[inline]
    pub fn clone_from(&mut self, other: &Self) {
        self.items.clone_from(&other.items);
        self.broad.clone_from(&other.broad);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn items(&self) -> &[T] {
        &self.items
    }

    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.items.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.items.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter_mut()
    }
}

impl<T> Display for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BroadCollection(len: {})", self.items.len())
    }
}

impl<T> BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    /// SIMD broadphase + scalar narrowphase.
    ///
    /// Quick-rejects the whole collection via bounding-sphere overlap, then
    /// runs narrowphase only on items whose own broadphase sphere overlaps
    /// `shape` — the per-item reject is a few FLOPs and skips the far more
    /// expensive narrowphase on shapes that cannot touch.
    pub fn collides<U>(&self, shape: &U) -> bool
    where
        U: Collides<T> + Bounded,
    {
        let bp = shape.broadphase();
        if !self.broad.any_collides_sphere(&bp) {
            return false;
        }
        let (qx, qy, qz, qr) = (bp.center.x, bp.center.y, bp.center.z, bp.radius);
        let (bx, by, bz, br) = (self.broad.x(), self.broad.y(), self.broad.z(), self.broad.r());
        self.items.iter().enumerate().any(|(i, item)| {
            let dx = bx[i] - qx;
            let dy = by[i] - qy;
            let dz = bz[i] - qz;
            let rad = qr + br[i];
            dx * dx + dy * dy + dz * dz <= rad * rad && shape.test::<false>(item)
        })
    }

    /// SIMD broadphase only — no narrowphase.
    ///
    /// Returns `true` if any stored bounding sphere overlaps `shape`'s.
    pub fn collides_only_broadphase<U: Bounded>(&self, shape: &U) -> bool {
        let bp = shape.broadphase();
        self.broad.any_collides_sphere(&bp)
    }
}

impl<T> Transformable for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    fn translate(&mut self, offset: glam::Vec3A) {
        for item in &mut self.items {
            item.translate(offset);
        }
        self.broad.translate(offset);
    }

    fn rotate_mat(&mut self, mat: glam::Mat3A) {
        for item in &mut self.items {
            item.rotate_mat(mat);
        }
        self.broad.rotate_mat(mat);
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        for item in &mut self.items {
            item.rotate_quat(quat);
        }
        self.broad.rotate_quat(quat);
    }

    fn transform(&mut self, mat: glam::Affine3A) {
        for item in &mut self.items {
            item.transform(mat);
        }
        self.broad.transform(mat);
    }
}

impl<T> Scalable for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    fn scale(&mut self, factor: f32) {
        for item in &mut self.items {
            item.scale(factor);
        }
        self.broad.scale(factor);
    }
}

impl<T> From<Vec<T>> for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    fn from(items: Vec<T>) -> Self {
        Self::new(items)
    }
}

impl<T> From<&[T]> for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    fn from(items: &[T]) -> Self {
        Self::new(items.to_vec())
    }
}

/// Columnar counterpart to [`BroadCollection`] for shapes that implement [`SoaShape`]
/// (capsule, cuboid, cylinder). Stores the shapes' fields as flat `f32` columns in a
/// [`ShapeSoa`] plus a bounding-sphere [`SpheresSoA`] for broadphase — so the collision batch
/// kernels read each field as a contiguous slice (no AoS gather). Reconstructing whole shapes
/// (`iter`/`to_vec`) and mutating (`transform`/`scale`, which rebuild) are the deliberate cost.
#[derive(Clone, PartialEq)]
pub struct ShapeCollection<S: SoaShape> {
    pub(crate) shapes: ShapeSoa<S>,
    pub(crate) broad: SpheresSoA,
}

impl<S> ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    pub fn new(items: Vec<S>) -> Self {
        let broad = SpheresSoA::from_bounded(&items);
        Self {
            shapes: items.iter().copied().collect(),
            broad,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            shapes: ShapeSoa::with_capacity(cap),
            broad: SpheresSoA::with_capacity(cap),
        }
    }

    pub fn push(&mut self, item: S) {
        self.broad.push(item.broadphase());
        self.shapes.push(&item);
    }

    pub fn extend(&mut self, items: impl IntoIterator<Item = S>) {
        for item in items {
            self.push(item);
        }
    }

    pub fn extend_from_slice(&mut self, items: &[S]) {
        self.extend(items.iter().copied());
    }

    pub fn append(&mut self, other: &mut Self) {
        self.shapes.append(&mut other.shapes);
        self.broad.append(&mut other.broad);
    }

    #[inline]
    pub fn clone_from(&mut self, other: &Self) {
        self.shapes = other.shapes.clone();
        self.broad.clone_from(&other.broad);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Reconstruct each stored shape, by value (the columnar "accessor" cost).
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = S> + '_ {
        self.shapes.iter()
    }

    pub fn to_vec(&self) -> Vec<S> {
        self.shapes.iter().collect()
    }

    fn remap(&mut self, mut f: impl FnMut(&mut S)) {
        let v: Vec<S> = self
            .shapes
            .iter()
            .map(|mut s| {
                f(&mut s);
                s
            })
            .collect();
        *self = Self::new(v);
    }

    /// SIMD broadphase reject, then narrowphase over the columns.
    ///
    /// Only shapes whose own broadphase sphere overlaps `shape` are
    /// reconstructed from the columns and narrowphased; the per-shape reject
    /// reads the broad columns directly and avoids the reconstruction cost on
    /// shapes that cannot touch.
    pub fn collides<U>(&self, shape: &U) -> bool
    where
        U: Collides<S> + Bounded,
    {
        let bp = shape.broadphase();
        if !self.broad.any_collides_sphere(&bp) {
            return false;
        }
        let (qx, qy, qz, qr) = (bp.center.x, bp.center.y, bp.center.z, bp.radius);
        let (bx, by, bz, br) = (self.broad.x(), self.broad.y(), self.broad.z(), self.broad.r());
        (0..self.shapes.len()).any(|i| {
            let dx = bx[i] - qx;
            let dy = by[i] - qy;
            let dz = bz[i] - qz;
            let rad = qr + br[i];
            dx * dx + dy * dy + dz * dz <= rad * rad && shape.test::<false>(&self.shapes.get(i))
        })
    }

    pub fn collides_only_broadphase<U: Bounded>(&self, shape: &U) -> bool {
        self.broad.any_collides_sphere(&shape.broadphase())
    }
}

impl<S> Default for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    fn default() -> Self {
        Self {
            shapes: ShapeSoa::new(),
            broad: SpheresSoA::new(),
        }
    }
}

impl<S> Transformable for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    fn translate(&mut self, offset: glam::Vec3A) {
        self.remap(|s| s.translate(offset));
    }
    fn rotate_mat(&mut self, mat: glam::Mat3A) {
        self.remap(|s| s.rotate_mat(mat));
    }
    fn rotate_quat(&mut self, quat: glam::Quat) {
        self.remap(|s| s.rotate_quat(quat));
    }
    fn transform(&mut self, mat: glam::Affine3A) {
        self.remap(|s| s.transform(mat));
    }
}

impl<S> Scalable for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    fn scale(&mut self, factor: f32) {
        self.remap(|s| s.scale(factor));
    }
}

impl<S> Display for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShapeCollection(len: {})", self.len())
    }
}

impl<S> Debug for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShapeCollection")
            .field("len", &self.len())
            .finish()
    }
}

impl<S> From<Vec<S>> for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug,
{
    fn from(items: Vec<S>) -> Self {
        Self::new(items)
    }
}

#[cfg(feature = "serde")]
impl<S> serde::Serialize for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug + serde::Serialize,
{
    fn serialize<Z: serde::Serializer>(&self, serializer: Z) -> Result<Z::Ok, Z::Error> {
        serializer.collect_seq(self.iter())
    }
}

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for ShapeCollection<S>
where
    S: SoaShape + Bounded + Transformable + Scalable + Debug + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(Vec::<S>::deserialize(deserializer)?))
    }
}

pub(crate) mod batch {
    #[cfg(not(feature = "std"))]
    #[allow(unused_imports)]
    use crate::F32Ext;
    use glam::Vec3;
    use hydroplane::{Backend, Gang, GangGlamExt, Mask, Varying, Vec3Wide, kernel};

    /// Density cutoff (percent of stored shapes surviving the query's broadphase) above which
    /// the SIMD batch narrowphase beats the broad-gated scalar scan. Below it most lanes are
    /// broad-rejected, so SIMD wastes work on them and the per-shape scalar scan wins.
    const SIMD_DENSITY_PCT: usize = 35;

    enum BatchPlan {
        Reject,
        Scalar,
        Simd,
    }

    /// Decide how a query meets a stored collection: reject outright, run the broad-gated scalar
    /// scan, or run the SIMD batch narrowphase.
    #[inline]
    fn batch_plan(len: usize, broad: &SpheresSoA, query_bp: &Sphere) -> BatchPlan {
        if len == 0 {
            return BatchPlan::Reject;
        }
        let surv = broad.count_overlaps(query_bp);
        if surv == 0 {
            return BatchPlan::Reject;
        }
        if surv * 100 < len * SIMD_DENSITY_PCT {
            return BatchPlan::Scalar;
        }
        BatchPlan::Simd
    }

    use super::{ShapeCollection, SpheresSoA};
    use crate::Collides;
    use crate::capsule::Capsule;
    use crate::cuboid::Cuboid;
    use crate::cylinder::Cylinder;
    use crate::line::{Line, LineSegment, Ray};
    use crate::plane::Plane;
    use crate::sphere::Sphere;

    pub fn plane_vs_spheres_soa(plane: &Plane, soa: &SpheresSoA) -> bool {
        if soa.is_empty() {
            return false;
        }
        plane_vs_spheres_k(soa.x(), soa.y(), soa.z(), soa.r(), plane.normal, plane.d)
    }

    #[kernel]
    fn plane_vs_spheres_k<'a>(
        ctx: Gang,
        xs: &'a [f32],
        ys: &'a [f32],
        zs: &'a [f32],
        rs: &'a [f32],
        normal: Vec3,
        d: f32,
    ) -> bool {
        let n = ctx.splat_vec3(normal);
        let zero = ctx.splat(0.0);
        ctx.any_n([xs, ys, zs, rs], |[x, y, z, r]| {
            let proj = n.dot(Vec3Wide::from([x, y, z]));
            (proj - r - d).le(zero)
        })
    }

    #[inline]
    fn line_vs_spheres_soa_inner(
        origin: Vec3,
        dir: Vec3,
        rdv: f32,
        t_min: f32,
        t_max: f32,
        soa: &SpheresSoA,
    ) -> bool {
        if soa.is_empty() {
            return false;
        }
        line_vs_spheres_k(soa.x(), soa.y(), soa.z(), soa.r(), origin, dir, rdv, t_min, t_max)
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn line_vs_spheres_k<'a>(
        ctx: Gang,
        xs: &'a [f32],
        ys: &'a [f32],
        zs: &'a [f32],
        rs: &'a [f32],
        origin: Vec3,
        dir: Vec3,
        rdv: f32,
        t_min: f32,
        t_max: f32,
    ) -> bool {
        let o = ctx.splat_vec3(origin);
        let d = ctx.splat_vec3(dir);
        let lo = ctx.splat(t_min);
        let hi = ctx.splat(t_max);

        ctx.any_n([xs, ys, zs, rs], |[x, y, z, r]| {
            let c = Vec3Wide::from([x, y, z]);
            let t = ((c - o).dot(d) * rdv).max(lo).min(hi);
            (c - o.add_scaled(d, t)).length_squared().le(r * r)
        })
    }

    #[inline]
    pub fn line_vs_spheres_soa(line: &Line, soa: &SpheresSoA) -> bool {
        line_vs_spheres_soa_inner(
            line.origin,
            line.dir,
            crate::line::rdv(line.dir),
            f32::NEG_INFINITY,
            f32::INFINITY,
            soa,
        )
    }

    #[inline]
    pub fn ray_vs_spheres_soa(ray: &Ray, soa: &SpheresSoA) -> bool {
        line_vs_spheres_soa_inner(
            ray.origin,
            ray.dir,
            crate::line::rdv(ray.dir),
            0.0,
            f32::INFINITY,
            soa,
        )
    }

    #[inline]
    pub fn segment_vs_spheres_soa(seg: &LineSegment, soa: &SpheresSoA) -> bool {
        let dir = seg.dir();
        line_vs_spheres_soa_inner(seg.start, dir, crate::line::rdv(dir), 0.0, 1.0, soa)
    }

    /// True narrowphase sphere-OBB test: one cuboid against many spheres.
    ///
    /// Padding lanes use `r = NaN`, so `dist_sq <= r²` is always false for
    /// them and they cannot produce a false positive.
    pub fn cuboid_vs_spheres_soa(cuboid: &Cuboid, soa: &SpheresSoA) -> bool {
        if soa.is_empty() {
            return false;
        }
        cuboid_vs_spheres_k(soa.x(), soa.y(), soa.z(), soa.r(), cuboid.center, cuboid.axes, cuboid.half_extents)
    }

    #[kernel]
    fn cuboid_vs_spheres_k<'a>(
        ctx: Gang,
        xs: &'a [f32],
        ys: &'a [f32],
        zs: &'a [f32],
        rs: &'a [f32],
        center: Vec3,
        axes: [Vec3; 3],
        he: [f32; 3],
    ) -> bool {
        let c = ctx.splat_vec3(center);
        let axes = axes.map(|a| ctx.splat_vec3(a));
        let zero = ctx.splat(0.0);

        ctx.any_n([xs, ys, zs, rs], |[x, y, z, sr]| {
            let df = Vec3Wide::from([x, y, z]) - c;
            let mut dist_sq = zero;
            for a in 0..3 {
                let proj = df.dot(axes[a]);
                let ex = (proj.abs() - he[a]).max(zero);
                dist_sq = dist_sq + ex * ex;
            }
            dist_sq.le(sr * sr)
        })
    }

    /// True narrowphase capsule-sphere test: one capsule against many spheres.
    ///
    /// Computes the squared distance from each sphere center to the capsule's
    /// line segment and compares against `(sphere.radius + capsule.radius)²`.
    pub fn capsule_vs_spheres_soa(capsule: &Capsule, soa: &SpheresSoA) -> bool {
        if soa.is_empty() {
            return false;
        }
        capsule_vs_spheres_k(soa.x(), soa.y(), soa.z(), soa.r(), capsule.p1, capsule.dir, capsule.rdv, capsule.radius)
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn capsule_vs_spheres_k<'a>(
        ctx: Gang,
        xs: &'a [f32],
        ys: &'a [f32],
        zs: &'a [f32],
        rs: &'a [f32],
        p1: Vec3,
        dir: Vec3,
        rdv: f32,
        cr: f32,
    ) -> bool {
        let p1v = ctx.splat_vec3(p1);
        let dv = ctx.splat_vec3(dir);
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);

        ctx.any_n([xs, ys, zs, rs], |[x, y, z, sr]| {
            let s = Vec3Wide::from([x, y, z]);
            let t = ((s - p1v).dot(dv) * rdv).max(zero).min(one);
            let dist_sq = (s - p1v.add_scaled(dv, t)).length_squared();

            let rsum = sr + cr;
            dist_sq.le(rsum * rsum)
        })
    }

    /// True narrowphase cylinder-sphere test: one cylinder against many spheres.
    ///
    /// Uses the same barrel / end-cap decomposition as `sphere_vs_cylinders_broad`,
    /// but splats cylinder state across the SIMD lanes and iterates the sphere SoA.
    pub fn cylinder_vs_spheres_soa(cyl: &Cylinder, soa: &SpheresSoA) -> bool {
        if soa.is_empty() {
            return false;
        }
        cylinder_vs_spheres_k(
            soa.x(),
            soa.y(),
            soa.z(),
            soa.r(),
            cyl.p1,
            cyl.dir,
            cyl.rdv,
            cyl.radius,
            cyl.dir.dot(cyl.dir),
        )
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn cylinder_vs_spheres_k<'a>(
        ctx: Gang,
        xs: &'a [f32],
        ys: &'a [f32],
        zs: &'a [f32],
        rs: &'a [f32],
        p1: Vec3,
        dir: Vec3,
        rdv: f32,
        cyl_r: f32,
        dir_sq_s: f32,
    ) -> bool {
        let p1v = ctx.splat_vec3(p1);
        let dv = ctx.splat_vec3(dir);
        let cr = ctx.splat(cyl_r);
        let dir_sq = ctx.splat(dir_sq_s);
        let cr_sq = cr * cr;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);

        ctx.any_n([xs, ys, zs, rs], |[x, y, z, sr]| {
            let w = Vec3Wide::from([x, y, z]) - p1v;

            let t = w.dot(dv) * rdv;
            let t_c = t.max(zero).min(one);

            let perp = w - dv * t;
            let r_sq = perp.length_squared();

            let sr_sq = sr * sr;
            let in_barrel = zero.le(t) & t.le(one);
            let combined = cr + sr;
            let barrel_hit = in_barrel & r_sq.le(combined * combined);

            let t_excess = t - t_c;
            let d_axial_sq = t_excess * t_excess * dir_sq;

            let inside_r = r_sq.le(cr_sq);
            let endcap_inside = inside_r & d_axial_sq.le(sr_sq);

            let l = r_sq + cr_sq + d_axial_sq - sr_sq;
            let endcap_outside = l.le(zero) | (l * l).le(cr_sq * r_sq * 4.0);

            let not_barrel = !in_barrel;
            barrel_hit | (not_barrel & (endcap_inside | endcap_outside))
        })
    }

    // ── Broadphase-filtered batch functions (Collider paths) ─────────────
    //
    // Each lane-chunk first reads the matching bounding spheres from the
    // BroadCollection's SpheresSoA and skips the chunk's narrowphase when no
    // bounding sphere overlaps the query. Inactive tail lanes are removed from
    // the final reduction with an explicit `lane < cnt` mask, so a short final
    // chunk can never produce a false positive.

    pub fn sphere_vs_capsules_broad(sphere: &Sphere, col: &ShapeCollection<Capsule>) -> bool {
        match batch_plan(col.len(), &col.broad, &sphere.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(sphere),
            BatchPlan::Simd => {}
        }
        sphere_vs_capsules_broad_k(
            col,
            [sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius],
        )
    }

    /// Capsule columns: `p1{x,y,z}=0..2`, `dir{x,y,z}=3..5`, `radius=6`, `rdv=7`. Read straight
    /// from the [`ShapeSoa`] — no AoS gather.
    #[kernel]
    fn sphere_vs_capsules_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Capsule>,
        q: [f32; 4],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let [cx, cy, cz, sr] = ctx.splat_n([q[0], q[1], q[2], q[3]]);
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = cx - bx;
            let bdy = cy - by;
            let bdz = cz - bz;
            let bmax = sr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [p1xv, p1yv, p1zv, dxv, dyv, dzv, crv, rdvv] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                    &s.col(7)[r.clone()],
                ],
                0.0,
            );

            let dfx = cx - p1xv;
            let dfy = cy - p1yv;
            let dfz = cz - p1zv;
            let t = ((dfx * dxv + dfy * dyv + dfz * dzv) * rdvv).max(zero).min(one);

            let ex = cx - (p1xv + dxv * t);
            let ey = cy - (p1yv + dyv * t);
            let ez = cz - (p1zv + dzv * t);
            let dist_sq = ex * ex + ey * ey + ez * ez;

            let rsum = sr + crv;
            if (dist_sq.le(rsum * rsum) & active).any() {
                return true;
            }
        }
        false
    }

    pub fn sphere_vs_cuboids_broad(sphere: &Sphere, col: &ShapeCollection<Cuboid>) -> bool {
        match batch_plan(col.len(), &col.broad, &sphere.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(sphere),
            BatchPlan::Simd => {}
        }
        sphere_vs_cuboids_broad_k(
            col,
            [sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius],
        )
    }

    /// Cuboid columns: `center=0..2`, `axes[a]=3+3a..5+3a`, `half_extents[a]=12+a`.
    #[kernel]
    fn sphere_vs_cuboids_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cuboid>,
        q: [f32; 4],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let cx = ctx.splat(q[0]);
        let cy = ctx.splat(q[1]);
        let cz = ctx.splat(q[2]);
        let sr = ctx.splat(q[3]);
        let r_sq = ctx.splat(q[3] * q[3]);
        let zero = ctx.splat(0.0);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = cx - bx;
            let bdy = cy - by;
            let bdz = cz - bz;
            let bmax = sr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [c0, c1, c2] =
                ctx.load_partial_n([&s.col(0)[r.clone()], &s.col(1)[r.clone()], &s.col(2)[r.clone()]], 0.0);
            let dfx = cx - c0;
            let dfy = cy - c1;
            let dfz = cz - c2;

            let mut dist_sq = zero;
            for a in 0..3 {
                let proj = dfx * ctx.load_partial(&s.col(3 + a * 3)[r.clone()], 0.0)
                    + dfy * ctx.load_partial(&s.col(4 + a * 3)[r.clone()], 0.0)
                    + dfz * ctx.load_partial(&s.col(5 + a * 3)[r.clone()], 0.0);
                let excess = (proj.abs() - ctx.load_partial(&s.col(12 + a)[r.clone()], 0.0)).max(zero);
                dist_sq = dist_sq + excess * excess;
            }

            if (dist_sq.le(r_sq) & active).any() {
                return true;
            }
        }
        false
    }

    pub fn sphere_vs_cylinders_broad(sphere: &Sphere, col: &ShapeCollection<Cylinder>) -> bool {
        match batch_plan(col.len(), &col.broad, &sphere.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(sphere),
            BatchPlan::Simd => {}
        }
        sphere_vs_cylinders_broad_k(
            col,
            [sphere.center.x, sphere.center.y, sphere.center.z, sphere.radius],
        )
    }

    /// Cylinder columns: `p1=0..2`, `dir=3..5`, `radius=6`, `rdv=7`; `dir²` recomputed from `dir`.
    #[kernel]
    fn sphere_vs_cylinders_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cylinder>,
        q: [f32; 4],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let cx = ctx.splat(q[0]);
        let cy = ctx.splat(q[1]);
        let cz = ctx.splat(q[2]);
        let sr = ctx.splat(q[3]);
        let sr_sq = ctx.splat(q[3] * q[3]);
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = cx - bx;
            let bdy = cy - by;
            let bdz = cz - bz;
            let bmax = sr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [p1xv, p1yv, p1zv, dxv, dyv, dzv, crv, rdvv] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                    &s.col(7)[r.clone()],
                ],
                0.0,
            );
            let dir_sq = dxv * dxv + dyv * dyv + dzv * dzv;

            let wx = cx - p1xv;
            let wy = cy - p1yv;
            let wz = cz - p1zv;

            let t = (wx * dxv + wy * dyv + wz * dzv) * rdvv;
            let t_c = t.max(zero).min(one);

            let perpx = wx - dxv * t;
            let perpy = wy - dyv * t;
            let perpz = wz - dzv * t;
            let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;

            let in_barrel = zero.le(t) & t.le(one);
            let combined = crv + sr;
            let barrel_hit = in_barrel & r_sq.le(combined * combined);

            let t_excess = t - t_c;
            let d_axial_sq = t_excess * t_excess * dir_sq;
            let cr_sq = crv * crv;

            let inside_r = r_sq.le(cr_sq);
            let endcap_inside = inside_r & d_axial_sq.le(sr_sq);

            let l = r_sq + cr_sq + d_axial_sq - sr_sq;
            let endcap_outside = l.le(zero) | (l * l).le(cr_sq * r_sq * 4.0);

            let not_barrel = !in_barrel;
            let hit = barrel_hit | (not_barrel & (endcap_inside | endcap_outside));
            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    pub fn plane_vs_capsules_broad(plane: &Plane, col: &ShapeCollection<Capsule>) -> bool {
        if col.is_empty() {
            return false;
        }
        plane_vs_capsules_broad_k(col, [plane.normal.x, plane.normal.y, plane.normal.z, plane.d])
    }

    #[kernel]
    fn plane_vs_capsules_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Capsule>,
        plane: [f32; 4],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let nx = ctx.splat(plane[0]);
        let ny = ctx.splat(plane[1]);
        let nz = ctx.splat(plane[2]);
        let d = plane[3];
        let zero = ctx.splat(0.0);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let bproj = nx * ctx.load_partial(&b.x()[r.clone()], 0.0)
                + ny * ctx.load_partial(&b.y()[r.clone()], 0.0)
                + nz * ctx.load_partial(&b.z()[r.clone()], 0.0);
            let bsep = bproj - ctx.load_partial(&b.r()[r.clone()], 0.0) - d;
            if !(bsep.le(zero) & active).any() {
                continue;
            }

            let [p1x, p1y, p1z, dx, dy, dz, cr] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                ],
                0.0,
            );

            let proj1 = nx * p1x + ny * p1y + nz * p1z;
            let proj2 = nx * (p1x + dx) + ny * (p1y + dy) + nz * (p1z + dz);
            let sep = proj1.min(proj2) - cr - d;
            if (sep.le(zero) & active).any() {
                return true;
            }
        }
        false
    }

    pub fn plane_vs_cuboids_broad(plane: &Plane, col: &ShapeCollection<Cuboid>) -> bool {
        if col.is_empty() {
            return false;
        }
        plane_vs_cuboids_broad_k(col, [plane.normal.x, plane.normal.y, plane.normal.z, plane.d])
    }

    #[kernel]
    fn plane_vs_cuboids_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cuboid>,
        plane: [f32; 4],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let nx = ctx.splat(plane[0]);
        let ny = ctx.splat(plane[1]);
        let nz = ctx.splat(plane[2]);
        let d = plane[3];
        let zero = ctx.splat(0.0);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let bproj = nx * ctx.load_partial(&b.x()[r.clone()], 0.0)
                + ny * ctx.load_partial(&b.y()[r.clone()], 0.0)
                + nz * ctx.load_partial(&b.z()[r.clone()], 0.0);
            let bsep = bproj - ctx.load_partial(&b.r()[r.clone()], 0.0) - d;
            if !(bsep.le(zero) & active).any() {
                continue;
            }

            let center_proj = nx * ctx.load_partial(&s.col(0)[r.clone()], 0.0)
                + ny * ctx.load_partial(&s.col(1)[r.clone()], 0.0)
                + nz * ctx.load_partial(&s.col(2)[r.clone()], 0.0);

            let mut ext = zero;
            for a in 0..3 {
                let np = nx * ctx.load_partial(&s.col(3 + a * 3)[r.clone()], 0.0)
                    + ny * ctx.load_partial(&s.col(4 + a * 3)[r.clone()], 0.0)
                    + nz * ctx.load_partial(&s.col(5 + a * 3)[r.clone()], 0.0);
                ext = ext + np.abs() * ctx.load_partial(&s.col(12 + a)[r.clone()], 0.0);
            }
            let sep = center_proj - ext - d;
            if (sep.le(zero) & active).any() {
                return true;
            }
        }
        false
    }

    pub fn plane_vs_cylinders_broad(plane: &Plane, col: &ShapeCollection<Cylinder>) -> bool {
        if col.is_empty() {
            return false;
        }
        plane_vs_cylinders_broad_k(col, [plane.normal.x, plane.normal.y, plane.normal.z, plane.d])
    }

    #[kernel]
    fn plane_vs_cylinders_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cylinder>,
        plane: [f32; 4],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let nx = ctx.splat(plane[0]);
        let ny = ctx.splat(plane[1]);
        let nz = ctx.splat(plane[2]);
        let d = plane[3];
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let eps = ctx.splat(f32::EPSILON);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let bproj = nx * ctx.load_partial(&b.x()[r.clone()], 0.0)
                + ny * ctx.load_partial(&b.y()[r.clone()], 0.0)
                + nz * ctx.load_partial(&b.z()[r.clone()], 0.0);
            let bsep = bproj - ctx.load_partial(&b.r()[r.clone()], 0.0) - d;
            if !(bsep.le(zero) & active).any() {
                continue;
            }

            let [p1x, p1y, p1z, dx, dy, dz, cr] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                ],
                0.0,
            );

            let proj1 = nx * p1x + ny * p1y + nz * p1z;
            let proj2 = nx * (p1x + dx) + ny * (p1y + dy) + nz * (p1z + dz);
            let min_proj = proj1.min(proj2);

            let dir_sq = dx * dx + dy * dy + dz * dz;
            let n_dot_dir = nx * dx + ny * dy + nz * dz;
            // (1 - (n·dir)²/dir²) clamped to ≥0; degenerate (dir²≈0) lanes use 1.
            let perp = (one - n_dot_dir * n_dot_dir / dir_sq).max(zero);
            let n_perp_sq = perp.select(dir_sq.gt(eps), one);
            let sep = min_proj - cr * n_perp_sq.sqrt() - d;
            if (sep.le(zero) & active).any() {
                return true;
            }
        }
        false
    }

    pub fn capsule_vs_capsules_broad(q: &Capsule, col: &ShapeCollection<Capsule>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let a = q.dir.dot(q.dir);
        if a <= f32::EPSILON {
            // zero-length query capsule degenerates to a sphere
            return sphere_vs_capsules_broad(&Sphere::new(q.p1, q.radius), col);
        }
        let (bc, br) = q.bounding_sphere();
        capsule_vs_capsules_broad_k(
            col,
            [bc.x, bc.y, bc.z, br],
            q.p1.to_array(),
            q.dir.to_array(),
            q.radius,
            a,
        )
    }

    /// Query capsule (segment `p1q + s·d1q`, radius `rq`) vs every stored capsule
    /// (`p1=cols 0..2`, `dir=3..5`, `radius=6`): branchless segment-segment closest distance,
    /// a SIMD port of `segment_segment_dist_sq` (the query is non-degenerate; `a = |d1q|² > eps`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn capsule_vs_capsules_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Capsule>,
        qbp: [f32; 4],
        p1q: [f32; 3],
        d1q: [f32; 3],
        rq: f32,
        a: f32,
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qr = ctx.splat(qbp[3]);
        let p1x = ctx.splat(p1q[0]);
        let p1y = ctx.splat(p1q[1]);
        let p1z = ctx.splat(p1q[2]);
        let d1x = ctx.splat(d1q[0]);
        let d1y = ctx.splat(d1q[1]);
        let d1z = ctx.splat(d1q[2]);
        let rqv = ctx.splat(rq);
        let av = ctx.splat(a);
        let inv_a = ctx.splat(1.0 / a);
        let eps = ctx.splat(f32::EPSILON);
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [p2x, p2y, p2z, d2x, d2y, d2z, cr] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                ],
                0.0,
            );

            let rx = p1x - p2x;
            let ry = p1y - p2y;
            let rz = p1z - p2z;
            let e = d2x * d2x + d2y * d2y + d2z * d2z;
            let f = d2x * rx + d2y * ry + d2z * rz;
            let cc = d1x * rx + d1y * ry + d1z * rz;
            let bdot = d1x * d2x + d1y * d2y + d1z * d2z;
            let denom = av * e - bdot * bdot;

            let s_gen = (((bdot * f - cc * e) / denom).max(zero).min(one)).select(denom.abs().gt(eps), zero);
            let t_n = (bdot * s_gen + f) / e;
            let below = t_n.lt(zero);
            let above = t_n.gt(one);
            let s_lt = ((zero - cc) * inv_a).max(zero).min(one);
            let s_gt = ((bdot - cc) * inv_a).max(zero).min(one);
            let mut sv = s_gt.select(above, s_lt.select(below, s_gen));
            let mut tv = t_n.max(zero).min(one);
            let edeg = e.le(eps);
            sv = s_lt.select(edeg, sv);
            tv = zero.select(edeg, tv);

            let dfx = (p1x + d1x * sv) - (p2x + d2x * tv);
            let dfy = (p1y + d1y * sv) - (p2y + d2y * tv);
            let dfz = (p1z + d1z * sv) - (p2z + d2z * tv);
            let dist_sq = dfx * dfx + dfy * dfy + dfz * dfz;
            let rsum = rqv + cr;
            if (dist_sq.le(rsum * rsum) & active).any() {
                return true;
            }
        }
        false
    }

    /// Single-dispatch capsule-SoA vs capsule-SoA: the query capsules are walked scalar-outer, each
    /// run through the segment-segment narrowphase over `b` via the `_on` companion — one ISA
    /// dispatch for the whole n×m rather than one per query capsule (the per-query path's cost).
    pub fn capsules_vs_capsules_soa(a: &ShapeCollection<Capsule>, b: &ShapeCollection<Capsule>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        capsules_vs_capsules_soa_k(a, b)
    }

    #[kernel]
    fn capsules_vs_capsules_soa_k<'a>(
        ctx: Gang,
        a: &'a ShapeCollection<Capsule>,
        b: &'a ShapeCollection<Capsule>,
    ) -> bool {
        let s = &a.shapes;
        for i in 0..a.len() {
            let p1 = [s.col(0)[i], s.col(1)[i], s.col(2)[i]];
            let dir = [s.col(3)[i], s.col(4)[i], s.col(5)[i]];
            let rad = s.col(6)[i];
            let aa = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            let hit = if aa > f32::EPSILON {
                let bc = [p1[0] + dir[0] * 0.5, p1[1] + dir[1] * 0.5, p1[2] + dir[2] * 0.5];
                let br = aa.sqrt() * 0.5 + rad;
                capsule_vs_capsules_broad_k_on(ctx, b, [bc[0], bc[1], bc[2], br], p1, dir, rad, aa)
            } else {
                sphere_vs_capsules_broad_k_on(ctx, b, [p1[0], p1[1], p1[2], rad])
            };
            if hit {
                return true;
            }
        }
        false
    }

    /// Single-dispatch cylinder-SoA vs cylinder-SoA — query cylinders walked scalar-outer, each run
    /// through the barrel/endcap narrowphase over `b` via the `_on` companion (one dispatch total).
    /// The kernel assumes non-degenerate queries; if any query cylinder is zero-length the whole
    /// call defers to the per-query path (which handles the degenerate case).
    pub fn cylinders_vs_cylinders_soa(a: &ShapeCollection<Cylinder>, b: &ShapeCollection<Cylinder>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        let s = &a.shapes;
        for i in 0..a.len() {
            let (dx, dy, dz) = (s.col(3)[i], s.col(4)[i], s.col(5)[i]);
            if dx * dx + dy * dy + dz * dz <= f32::EPSILON {
                return a.shapes.iter().any(|q| cylinder_vs_cylinders_broad(&q, b));
            }
        }
        cylinders_vs_cylinders_soa_k(a, b)
    }

    #[kernel]
    fn cylinders_vs_cylinders_soa_k<'a>(
        ctx: Gang,
        a: &'a ShapeCollection<Cylinder>,
        b: &'a ShapeCollection<Cylinder>,
    ) -> bool {
        let s = &a.shapes;
        for i in 0..a.len() {
            let p1 = [s.col(0)[i], s.col(1)[i], s.col(2)[i]];
            let dir = [s.col(3)[i], s.col(4)[i], s.col(5)[i]];
            let rad = s.col(6)[i];
            let rdv = s.col(7)[i];
            let aa = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            let bc = [p1[0] + dir[0] * 0.5, p1[1] + dir[1] * 0.5, p1[2] + dir[2] * 0.5];
            let br = aa.sqrt() * 0.5 + rad;
            if cylinder_vs_cylinders_broad_k_on(ctx, b, [bc[0], bc[1], bc[2], br], p1, dir, rad, aa, rdv) {
                return true;
            }
        }
        false
    }

    pub fn cylinder_vs_cylinders_broad(q: &Cylinder, col: &ShapeCollection<Cylinder>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let a = q.dir.dot(q.dir);
        if a <= f32::EPSILON {
            // zero-length query cylinder is degenerate; fall back to scalar (rare).
            return col.shapes.iter().any(|c| q.test::<false>(&c));
        }
        let (bc, br) = q.bounding_sphere();
        cylinder_vs_cylinders_broad_k(
            col,
            [bc.x, bc.y, bc.z, br],
            q.p1.to_array(),
            q.dir.to_array(),
            q.radius,
            a,
            q.rdv,
        )
    }

    /// Query cylinder vs every stored cylinder (`p1=0..2`, `dir=3..5`, `radius=6`, `rdv=7`):
    /// barrel-barrel via the branchless segment-segment distance, plus 5+5 axis-sample endcap
    /// point-cylinder tests — a SIMD port of `Collides<Cylinder> for Cylinder`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn cylinder_vs_cylinders_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cylinder>,
        qbp: [f32; 4],
        p1q: [f32; 3],
        d1q: [f32; 3],
        rq: f32,
        a: f32,
        rdvq: f32,
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let eps = ctx.splat(f32::EPSILON);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let p1x = ctx.splat(p1q[0]);
        let p1y = ctx.splat(p1q[1]);
        let p1z = ctx.splat(p1q[2]);
        let d1x = ctx.splat(d1q[0]);
        let d1y = ctx.splat(d1q[1]);
        let d1z = ctx.splat(d1q[2]);
        let rqv = ctx.splat(rq);
        let av = ctx.splat(a);
        let inv_a = ctx.splat(1.0 / a);
        let rdvqv = ctx.splat(rdvq);

        // point–cylinder squared distance, all-varying (`cdsq` = cyl.dir·cyl.dir).
        let pcyl = |px: Varying<f32, _>,
                    py: Varying<f32, _>,
                    pz: Varying<f32, _>,
                    cp1x: Varying<f32, _>,
                    cp1y: Varying<f32, _>,
                    cp1z: Varying<f32, _>,
                    cdx: Varying<f32, _>,
                    cdy: Varying<f32, _>,
                    cdz: Varying<f32, _>,
                    crdv: Varying<f32, _>,
                    cr: Varying<f32, _>,
                    cdsq: Varying<f32, _>| {
            let wx = px - cp1x;
            let wy = py - cp1y;
            let wz = pz - cp1z;
            let t = (wx * cdx + wy * cdy + wz * cdz) * crdv;
            let perpx = wx - cdx * t;
            let perpy = wy - cdy * t;
            let perpz = wz - cdz * t;
            let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;
            let t_c = t.max(zero).min(one);
            let te = t - t_c;
            let d_axial = te * te * cdsq;
            let inside = r_sq.le(cr * cr);
            let radial = r_sq.sqrt() - cr;
            d_axial + zero.select(inside, radial * radial)
        };

        const SAMPLES: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [p2x, p2y, p2z, d2x, d2y, d2z, cr, rdv2] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                    &s.col(7)[r.clone()],
                ],
                0.0,
            );
            let cdsq = d2x * d2x + d2y * d2y + d2z * d2z;

            // barrel-barrel: branchless segment-segment closest distance (query non-degenerate).
            let rx = p1x - p2x;
            let ry = p1y - p2y;
            let rz = p1z - p2z;
            let e = cdsq;
            let f = d2x * rx + d2y * ry + d2z * rz;
            let cc = d1x * rx + d1y * ry + d1z * rz;
            let bdot = d1x * d2x + d1y * d2y + d1z * d2z;
            let denom = av * e - bdot * bdot;
            let s_gen = (((bdot * f - cc * e) / denom).max(zero).min(one)).select(denom.abs().gt(eps), zero);
            let t_n = (bdot * s_gen + f) / e;
            let below = t_n.lt(zero);
            let above = t_n.gt(one);
            let s_lt = ((zero - cc) * inv_a).max(zero).min(one);
            let s_gt = ((bdot - cc) * inv_a).max(zero).min(one);
            let mut sv = s_gt.select(above, s_lt.select(below, s_gen));
            let mut tv = t_n.max(zero).min(one);
            let edeg = e.le(eps);
            sv = s_lt.select(edeg, sv);
            tv = zero.select(edeg, tv);
            let dfx = (p1x + d1x * sv) - (p2x + d2x * tv);
            let dfy = (p1y + d1y * sv) - (p2y + d2y * tv);
            let dfz = (p1z + d1z * sv) - (p2z + d2z * tv);
            let seg_sq = dfx * dfx + dfy * dfy + dfz * dfz;
            let combined = rqv + cr;
            let mut hit = seg_sq.le(combined * combined);

            // endcaps: 5 samples of the query axis vs each stored cylinder
            let rq_sq = rqv * rqv;
            for &ts in &SAMPLES {
                let sx = ctx.splat(p1q[0] + d1q[0] * ts);
                let sy = ctx.splat(p1q[1] + d1q[1] * ts);
                let sz = ctx.splat(p1q[2] + d1q[2] * ts);
                let pd = pcyl(sx, sy, sz, p2x, p2y, p2z, d2x, d2y, d2z, rdv2, cr, cdsq);
                hit = hit | pd.le(rq_sq);
            }
            // endcaps: 5 samples of each stored axis vs the query cylinder
            for &ts in &SAMPLES {
                let sx = p2x + d2x * ts;
                let sy = p2y + d2y * ts;
                let sz = p2z + d2z * ts;
                let pd = pcyl(sx, sy, sz, p1x, p1y, p1z, d1x, d1y, d1z, rdvqv, rqv, av);
                hit = hit | pd.le(cr * cr);
            }

            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    pub fn cuboid_vs_cuboids_broad(q: &Cuboid, col: &ShapeCollection<Cuboid>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let br = q.bounding_sphere_radius();
        cuboid_vs_cuboids_broad_k(
            col,
            [q.center.x, q.center.y, q.center.z, br],
            [
                q.axes[0].to_array(),
                q.axes[1].to_array(),
                q.axes[2].to_array(),
            ],
            q.half_extents,
        )
    }

    /// Query cuboid vs every stored cuboid (`center=0..2`, `axes=3..11`, `half_extents=12..14`):
    /// the 15-axis separating-axis test run branchlessly across a lane of stored boxes — a SIMD
    /// port of `Collides<Cuboid> for Cuboid`. The query is `self`, each stored box is `other`.
    #[kernel]
    fn cuboid_vs_cuboids_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cuboid>,
        qbp: [f32; 4],
        qaxes: [[f32; 3]; 3],
        qhe: [f32; 3],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let eps = ctx.splat(1e-6);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let acx = qcx;
        let acy = qcy;
        let acz = qcz;
        let aax: [[Varying<f32, _>; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|k| ctx.splat(qaxes[i][k])));
        let ea: [Varying<f32, _>; 3] = core::array::from_fn(|i| ctx.splat(qhe[i]));

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bmax = qbr + br;
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [bcx, bcy, bcz] =
                ctx.load_partial_n([&s.col(0)[r.clone()], &s.col(1)[r.clone()], &s.col(2)[r.clone()]], 0.0);
            let bax: [[Varying<f32, _>; 3]; 3] = core::array::from_fn(|j| {
                core::array::from_fn(|k| ctx.load_partial(&s.col(3 + j * 3 + k)[r.clone()], 0.0))
            });
            let eb: [Varying<f32, _>; 3] =
                core::array::from_fn(|j| ctx.load_partial(&s.col(12 + j)[r.clone()], 0.0));

            let rm: [[Varying<f32, _>; 3]; 3] = core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    aax[i][0] * bax[j][0] + aax[i][1] * bax[j][1] + aax[i][2] * bax[j][2]
                })
            });
            let ar: [[Varying<f32, _>; 3]; 3] =
                core::array::from_fn(|i| core::array::from_fn(|j| rm[i][j].abs() + eps));

            let tvx = bcx - acx;
            let tvy = bcy - acy;
            let tvz = bcz - acz;
            let t: [Varying<f32, _>; 3] = core::array::from_fn(|i| {
                tvx * aax[i][0] + tvy * aax[i][1] + tvz * aax[i][2]
            });

            let mut collide = zero.le(zero);
            let mut sat = |sep: Varying<f32, _>, ra: Varying<f32, _>, rb: Varying<f32, _>| {
                collide = collide & sep.abs().le(ra + rb);
            };

            // L = A0, A1, A2
            for i in 0..3 {
                let rb = eb[0] * ar[i][0] + eb[1] * ar[i][1] + eb[2] * ar[i][2];
                sat(t[i], ea[i], rb);
            }
            // L = B0, B1, B2
            for j in 0..3 {
                let ra = ea[0] * ar[0][j] + ea[1] * ar[1][j] + ea[2] * ar[2][j];
                let sep = t[0] * rm[0][j] + t[1] * rm[1][j] + t[2] * rm[2][j];
                sat(sep, ra, eb[j]);
            }
            // 9 cross-product axes Ai x Bj
            sat(
                t[2] * rm[1][0] - t[1] * rm[2][0],
                ea[1] * ar[2][0] + ea[2] * ar[1][0],
                eb[1] * ar[0][2] + eb[2] * ar[0][1],
            );
            sat(
                t[2] * rm[1][1] - t[1] * rm[2][1],
                ea[1] * ar[2][1] + ea[2] * ar[1][1],
                eb[0] * ar[0][2] + eb[2] * ar[0][0],
            );
            sat(
                t[2] * rm[1][2] - t[1] * rm[2][2],
                ea[1] * ar[2][2] + ea[2] * ar[1][2],
                eb[0] * ar[0][1] + eb[1] * ar[0][0],
            );
            sat(
                t[0] * rm[2][0] - t[2] * rm[0][0],
                ea[0] * ar[2][0] + ea[2] * ar[0][0],
                eb[1] * ar[1][2] + eb[2] * ar[1][1],
            );
            sat(
                t[0] * rm[2][1] - t[2] * rm[0][1],
                ea[0] * ar[2][1] + ea[2] * ar[0][1],
                eb[0] * ar[1][2] + eb[2] * ar[1][0],
            );
            sat(
                t[0] * rm[2][2] - t[2] * rm[0][2],
                ea[0] * ar[2][2] + ea[2] * ar[0][2],
                eb[0] * ar[1][1] + eb[1] * ar[1][0],
            );
            sat(
                t[1] * rm[0][0] - t[0] * rm[1][0],
                ea[0] * ar[1][0] + ea[1] * ar[0][0],
                eb[1] * ar[2][2] + eb[2] * ar[2][1],
            );
            sat(
                t[1] * rm[0][1] - t[0] * rm[1][1],
                ea[0] * ar[1][1] + ea[1] * ar[0][1],
                eb[0] * ar[2][2] + eb[2] * ar[2][0],
            );
            sat(
                t[1] * rm[0][2] - t[0] * rm[1][2],
                ea[0] * ar[1][2] + ea[1] * ar[0][2],
                eb[0] * ar[2][1] + eb[1] * ar[2][0],
            );

            if (collide & active).any() {
                return true;
            }
        }
        false
    }

    /// Single-dispatch capsule-SoA vs cuboid-SoA — query capsules walked scalar-outer, each run
    /// through the breakpoint-sampling narrowphase over `b` via the `_on` companion.
    pub fn capsules_vs_cuboids_soa(a: &ShapeCollection<Capsule>, b: &ShapeCollection<Cuboid>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        capsules_vs_cuboids_soa_k(a, b)
    }

    #[kernel]
    fn capsules_vs_cuboids_soa_k<'a>(
        ctx: Gang,
        a: &'a ShapeCollection<Capsule>,
        b: &'a ShapeCollection<Cuboid>,
    ) -> bool {
        let s = &a.shapes;
        for i in 0..a.len() {
            let p1 = [s.col(0)[i], s.col(1)[i], s.col(2)[i]];
            let dir = [s.col(3)[i], s.col(4)[i], s.col(5)[i]];
            let rad = s.col(6)[i];
            let aa = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            let bc = [p1[0] + dir[0] * 0.5, p1[1] + dir[1] * 0.5, p1[2] + dir[2] * 0.5];
            let br = aa.sqrt() * 0.5 + rad;
            if capsule_vs_cuboids_broad_k_on(ctx, b, [bc[0], bc[1], bc[2], br], p1, dir, rad * rad) {
                return true;
            }
        }
        false
    }

    /// Single-dispatch sphere-SoA queries over a stored shape collection: each sphere `(c, r)` is
    /// run through the per-query narrowphase over `b` via the `_on` companion (one dispatch total).
    pub fn spheres_vs_capsules_soa(a: &SpheresSoA, b: &ShapeCollection<Capsule>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        spheres_vs_capsules_soa_k(a, b)
    }

    #[kernel]
    fn spheres_vs_capsules_soa_k<'a>(ctx: Gang, a: &'a SpheresSoA, b: &'a ShapeCollection<Capsule>) -> bool {
        let (xs, ys, zs, rs) = (a.x(), a.y(), a.z(), a.r());
        for i in 0..xs.len() {
            if sphere_vs_capsules_broad_k_on(ctx, b, [xs[i], ys[i], zs[i], rs[i]]) {
                return true;
            }
        }
        false
    }

    pub fn spheres_vs_cuboids_soa(a: &SpheresSoA, b: &ShapeCollection<Cuboid>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        spheres_vs_cuboids_soa_k(a, b)
    }

    #[kernel]
    fn spheres_vs_cuboids_soa_k<'a>(ctx: Gang, a: &'a SpheresSoA, b: &'a ShapeCollection<Cuboid>) -> bool {
        let (xs, ys, zs, rs) = (a.x(), a.y(), a.z(), a.r());
        for i in 0..xs.len() {
            if sphere_vs_cuboids_broad_k_on(ctx, b, [xs[i], ys[i], zs[i], rs[i]]) {
                return true;
            }
        }
        false
    }

    pub fn spheres_vs_cylinders_soa(a: &SpheresSoA, b: &ShapeCollection<Cylinder>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        spheres_vs_cylinders_soa_k(a, b)
    }

    #[kernel]
    fn spheres_vs_cylinders_soa_k<'a>(ctx: Gang, a: &'a SpheresSoA, b: &'a ShapeCollection<Cylinder>) -> bool {
        let (xs, ys, zs, rs) = (a.x(), a.y(), a.z(), a.r());
        for i in 0..xs.len() {
            if sphere_vs_cylinders_broad_k_on(ctx, b, [xs[i], ys[i], zs[i], rs[i]]) {
                return true;
            }
        }
        false
    }

    /// Single-dispatch capsule-SoA vs cylinder-SoA (query capsules scalar-outer via the `_on`).
    pub fn capsules_vs_cylinders_soa(a: &ShapeCollection<Capsule>, b: &ShapeCollection<Cylinder>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        capsules_vs_cylinders_soa_k(a, b)
    }

    #[kernel]
    fn capsules_vs_cylinders_soa_k<'a>(
        ctx: Gang,
        a: &'a ShapeCollection<Capsule>,
        b: &'a ShapeCollection<Cylinder>,
    ) -> bool {
        let s = &a.shapes;
        for i in 0..a.len() {
            let p1 = [s.col(0)[i], s.col(1)[i], s.col(2)[i]];
            let dir = [s.col(3)[i], s.col(4)[i], s.col(5)[i]];
            let rad = s.col(6)[i];
            let aa = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            let bc = [p1[0] + dir[0] * 0.5, p1[1] + dir[1] * 0.5, p1[2] + dir[2] * 0.5];
            let br = aa.sqrt() * 0.5 + rad;
            if capsule_vs_cylinders_broad_k_on(ctx, b, [bc[0], bc[1], bc[2], br], p1, dir, rad) {
                return true;
            }
        }
        false
    }

    /// Single-dispatch cylinder-SoA vs cuboid-SoA (query cylinders scalar-outer via the `_on`).
    pub fn cylinders_vs_cuboids_soa(a: &ShapeCollection<Cylinder>, b: &ShapeCollection<Cuboid>) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }
        cylinders_vs_cuboids_soa_k(a, b)
    }

    #[kernel]
    fn cylinders_vs_cuboids_soa_k<'a>(
        ctx: Gang,
        a: &'a ShapeCollection<Cylinder>,
        b: &'a ShapeCollection<Cuboid>,
    ) -> bool {
        let s = &a.shapes;
        for i in 0..a.len() {
            let p1 = [s.col(0)[i], s.col(1)[i], s.col(2)[i]];
            let dir = [s.col(3)[i], s.col(4)[i], s.col(5)[i]];
            let rad = s.col(6)[i];
            let rdv = s.col(7)[i];
            let aa = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            let bc = [p1[0] + dir[0] * 0.5, p1[1] + dir[1] * 0.5, p1[2] + dir[2] * 0.5];
            let br = aa.sqrt() * 0.5 + rad;
            if cylinder_vs_cuboids_broad_k_on(ctx, b, [bc[0], bc[1], bc[2], br], p1, dir, rdv, rad * rad) {
                return true;
            }
        }
        false
    }

    pub fn capsule_vs_cuboids_broad(q: &Capsule, col: &ShapeCollection<Cuboid>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let (bc, br) = q.bounding_sphere();
        capsule_vs_cuboids_broad_k(
            col,
            [bc.x, bc.y, bc.z, br],
            q.p1.to_array(),
            q.dir.to_array(),
            q.radius * q.radius,
        )
    }

    /// Query capsule vs every stored cuboid (`center=0..2`, `axes=3..11`, `half_extents=12..14`):
    /// the capsule axis is projected into each cuboid's local frame and sampled at the 8 convex
    /// breakpoints (2 endpoints + 6 slab crossings); a hit if any sample lies within the capsule
    /// radius. A SIMD port of `capsule_cuboid_collides` over a lane of stored cuboids.
    #[kernel]
    fn capsule_vs_cuboids_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cuboid>,
        qbp: [f32; 4],
        qp1: [f32; 3],
        qdir: [f32; 3],
        rs_sq: f32,
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let eps = ctx.splat(f32::EPSILON);
        let big = ctx.splat(f32::MAX);
        let rs = ctx.splat(rs_sq);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let p1x = ctx.splat(qp1[0]);
        let p1y = ctx.splat(qp1[1]);
        let p1z = ctx.splat(qp1[2]);
        let dqx = ctx.splat(qdir[0]);
        let dqy = ctx.splat(qdir[1]);
        let dqz = ctx.splat(qdir[2]);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [cx, cy, cz] =
                ctx.load_partial_n([&s.col(0)[r.clone()], &s.col(1)[r.clone()], &s.col(2)[r.clone()]], 0.0);
            let ax: [[Varying<f32, _>; 3]; 3] = core::array::from_fn(|i| {
                core::array::from_fn(|k| ctx.load_partial(&s.col(3 + i * 3 + k)[r.clone()], 0.0))
            });
            let he: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| ctx.load_partial(&s.col(12 + i)[r.clone()], 0.0));

            let wx = p1x - cx;
            let wy = p1y - cy;
            let wz = p1z - cz;
            let p0: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| wx * ax[i][0] + wy * ax[i][1] + wz * ax[i][2]);
            let dir: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| dqx * ax[i][0] + dqy * ax[i][1] + dqz * ax[i][2]);
            let inv: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| (one / dir[i]).select(dir[i].abs().gt(eps), big));

            let dist_sq = |t: Varying<f32, _>| {
                let mut d = zero;
                for i in 0..3 {
                    let pos = t * dir[i] + p0[i];
                    let ex = (pos.abs() - he[i]).max(zero);
                    d = d + ex * ex;
                }
                d
            };

            let mut hit = dist_sq(zero).le(rs) | dist_sq(one).le(rs);
            for i in 0..3 {
                let lo = (((zero - he[i]) - p0[i]) * inv[i]).max(zero).min(one);
                let hi = ((he[i] - p0[i]) * inv[i]).max(zero).min(one);
                hit = hit | dist_sq(lo).le(rs) | dist_sq(hi).le(rs);
            }

            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    pub fn cuboid_vs_capsules_broad(q: &Cuboid, col: &ShapeCollection<Capsule>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let br = q.bounding_sphere_radius();
        cuboid_vs_capsules_broad_k(
            col,
            [q.center.x, q.center.y, q.center.z, br],
            [
                q.axes[0].to_array(),
                q.axes[1].to_array(),
                q.axes[2].to_array(),
            ],
            q.half_extents,
        )
    }

    /// Query cuboid vs every stored capsule (`p1=0..2`, `dir=3..5`, `radius=6`): each capsule axis
    /// is projected into the query cuboid's local frame and sampled at the 8 convex breakpoints.
    /// The companion of `capsule_vs_cuboids_broad` with the roles (and the per-lane radius) flipped.
    #[kernel]
    fn cuboid_vs_capsules_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Capsule>,
        qbp: [f32; 4],
        qaxes: [[f32; 3]; 3],
        qhe: [f32; 3],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let eps = ctx.splat(f32::EPSILON);
        let big = ctx.splat(f32::MAX);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let ax: [[Varying<f32, _>; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|k| ctx.splat(qaxes[i][k])));
        let he: [Varying<f32, _>; 3] = core::array::from_fn(|i| ctx.splat(qhe[i]));

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [px, py, pz, ddx, ddy, ddz, crad] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                ],
                0.0,
            );
            let rs = crad * crad;

            let wx = px - qcx;
            let wy = py - qcy;
            let wz = pz - qcz;
            let p0: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| wx * ax[i][0] + wy * ax[i][1] + wz * ax[i][2]);
            let dir: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| ddx * ax[i][0] + ddy * ax[i][1] + ddz * ax[i][2]);
            let inv: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| (one / dir[i]).select(dir[i].abs().gt(eps), big));

            let dist_sq = |t: Varying<f32, _>| {
                let mut d = zero;
                for i in 0..3 {
                    let pos = t * dir[i] + p0[i];
                    let ex = (pos.abs() - he[i]).max(zero);
                    d = d + ex * ex;
                }
                d
            };

            let mut hit = dist_sq(zero).le(rs) | dist_sq(one).le(rs);
            for i in 0..3 {
                let lo = (((zero - he[i]) - p0[i]) * inv[i]).max(zero).min(one);
                let hi = ((he[i] - p0[i]) * inv[i]).max(zero).min(one);
                hit = hit | dist_sq(lo).le(rs) | dist_sq(hi).le(rs);
            }

            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    struct CylCap<S: Backend<f32>> {
        zero: Varying<f32, S>,
        one: Varying<f32, S>,
        half: Varying<f32, S>,
        four: Varying<f32, S>,
        eps: Varying<f32, S>,
        e: Varying<f32, S>,
        rdvv: Varying<f32, S>,
        ap1x: Varying<f32, S>,
        ap1y: Varying<f32, S>,
        ap1z: Varying<f32, S>,
        adx: Varying<f32, S>,
        ady: Varying<f32, S>,
        adz: Varying<f32, S>,
        cap_r: Varying<f32, S>,
        cap_r_sq: Varying<f32, S>,
        cp1x: Varying<f32, S>,
        cp1y: Varying<f32, S>,
        cp1z: Varying<f32, S>,
        cdx: Varying<f32, S>,
        cdy: Varying<f32, S>,
        cdz: Varying<f32, S>,
        cyl_r: Varying<f32, S>,
        cyl_r_sq: Varying<f32, S>,
    }

    /// Shared cylinder-capsule sampled overlap, vectorized over a lane of shapes. Computes the 3
    /// data-dependent capsule-axis samples (closest approach + the two cylinder end-cap crossings)
    /// branchlessly, then ORs the barrel / inside-cap / outside-cap test at all 8 samples. Both
    /// query directions feed it the same operands as `Varying`s, so the math lives here once.
    fn cyl_cap_eval<S: Backend<f32>>(v: CylCap<S>) -> Mask<f32, S> {
        let CylCap {
            zero, one, half, four, eps, e, rdvv,
            ap1x, ap1y, ap1z, adx, ady, adz, cap_r, cap_r_sq,
            cp1x, cp1y, cp1z, cdx, cdy, cdz, cyl_r, cyl_r_sq,
        } = v;

        let rx = ap1x - cp1x;
        let ry = ap1y - cp1y;
        let rz = ap1z - cp1z;
        let a = adx * adx + ady * ady + adz * adz;
        let c_val = adx * rx + ady * ry + adz * rz;
        let bdot = adx * cdx + ady * cdy + adz * cdz;
        let f = cdx * rx + cdy * ry + cdz * rz;
        let denom = a * e - bdot * bdot;
        let s_gen = ((bdot * f - c_val * e) / denom)
            .max(zero)
            .min(one)
            .select(denom.abs().gt(eps), half);
        let s_ea = ((zero - c_val) / a).max(zero).min(one);
        let s_inner = s_gen.select(e.gt(eps), s_ea);
        let s_closest = s_inner.select(a.gt(eps), zero);
        let inv = one / bdot;
        let s_t0 = ((zero - f) * inv).max(zero).min(one).select(bdot.abs().gt(eps), zero);
        let s_t1 = ((e - f) * inv).max(zero).min(one).select(bdot.abs().gt(eps), one);

        let combined = cyl_r + cap_r;
        let combined_sq = combined * combined;
        let quarter = half * half;
        let three_q = one - quarter;

        let eval = |sx: Varying<f32, S>| {
            let wx = (ap1x + adx * sx) - cp1x;
            let wy = (ap1y + ady * sx) - cp1y;
            let wz = (ap1z + adz * sx) - cp1z;
            let t = (wx * cdx + wy * cdy + wz * cdz) * rdvv;
            let tc = t.max(zero).min(one);
            let perpx = wx - cdx * t;
            let perpy = wy - cdy * t;
            let perpz = wz - cdz * t;
            let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;
            let te = t - tc;
            let d_axial = te * te * e;
            let in_barrel = zero.le(t) & t.le(one);
            let barrel = in_barrel & r_sq.le(combined_sq);
            let ein = r_sq.le(cyl_r_sq) & d_axial.le(cap_r_sq);
            let l = r_sq + cyl_r_sq + d_axial - cap_r_sq;
            let eout = l.le(zero) | (l * l).le(cyl_r_sq * r_sq * four);
            barrel | ein | eout
        };

        eval(zero)
            | eval(one)
            | eval(s_closest)
            | eval(s_t0)
            | eval(s_t1)
            | eval(quarter)
            | eval(half)
            | eval(three_q)
    }

    pub fn cylinder_vs_capsules_broad(q: &Cylinder, col: &ShapeCollection<Capsule>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let (bc, br) = q.bounding_sphere();
        cylinder_vs_capsules_broad_k(
            col,
            [bc.x, bc.y, bc.z, br],
            q.p1.to_array(),
            q.dir.to_array(),
            q.rdv,
            q.radius,
            q.dir.dot(q.dir),
        )
    }

    /// Query cylinder vs every stored capsule (`p1=0..2`, `dir=3..5`, `radius=6`): the capsule axis
    /// is sampled at 8 data-dependent points and each is tested against the query cylinder's barrel
    /// and end-cap regions — a SIMD-over-shapes port of `cylinder_capsule_collides`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn cylinder_vs_capsules_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Capsule>,
        qbp: [f32; 4],
        cylp1: [f32; 3],
        cyld: [f32; 3],
        rdv: f32,
        cyl_radius: f32,
        e_s: f32,
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let half = ctx.splat(0.5);
        let four = ctx.splat(4.0);
        let eps = ctx.splat(f32::EPSILON);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let cp1x = ctx.splat(cylp1[0]);
        let cp1y = ctx.splat(cylp1[1]);
        let cp1z = ctx.splat(cylp1[2]);
        let cdx = ctx.splat(cyld[0]);
        let cdy = ctx.splat(cyld[1]);
        let cdz = ctx.splat(cyld[2]);
        let rdvv = ctx.splat(rdv);
        let cyl_r = ctx.splat(cyl_radius);
        let cyl_r_sq = cyl_r * cyl_r;
        let e = ctx.splat(e_s);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [ap1x, ap1y, ap1z, adx, ady, adz, cap_r] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                ],
                0.0,
            );
            let cap_r_sq = cap_r * cap_r;

            let hit = cyl_cap_eval(
                CylCap {
                    zero, one, half, four, eps, e, rdvv,
                    ap1x, ap1y, ap1z, adx, ady, adz, cap_r, cap_r_sq,
                    cp1x, cp1y, cp1z, cdx, cdy, cdz, cyl_r, cyl_r_sq,
                },
            );

            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    pub fn capsule_vs_cylinders_broad(q: &Capsule, col: &ShapeCollection<Cylinder>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let (bc, br) = q.bounding_sphere();
        capsule_vs_cylinders_broad_k(
            col,
            [bc.x, bc.y, bc.z, br],
            q.p1.to_array(),
            q.dir.to_array(),
            q.radius,
        )
    }

    /// Query capsule vs every stored cylinder (`p1=0..2`, `dir=3..5`, `radius=6`, `rdv=7`): the
    /// companion of `cylinder_vs_capsules_broad` with the cylinder side read per-lane.
    #[kernel]
    fn capsule_vs_cylinders_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cylinder>,
        qbp: [f32; 4],
        capp1: [f32; 3],
        capd: [f32; 3],
        cap_radius: f32,
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let half = ctx.splat(0.5);
        let four = ctx.splat(4.0);
        let eps = ctx.splat(f32::EPSILON);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let ap1x = ctx.splat(capp1[0]);
        let ap1y = ctx.splat(capp1[1]);
        let ap1z = ctx.splat(capp1[2]);
        let adx = ctx.splat(capd[0]);
        let ady = ctx.splat(capd[1]);
        let adz = ctx.splat(capd[2]);
        let cap_r = ctx.splat(cap_radius);
        let cap_r_sq = cap_r * cap_r;

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let [cp1x, cp1y, cp1z, cdx, cdy, cdz, cyl_r, rdvv] = ctx.load_partial_n(
                [
                    &s.col(0)[r.clone()],
                    &s.col(1)[r.clone()],
                    &s.col(2)[r.clone()],
                    &s.col(3)[r.clone()],
                    &s.col(4)[r.clone()],
                    &s.col(5)[r.clone()],
                    &s.col(6)[r.clone()],
                    &s.col(7)[r.clone()],
                ],
                0.0,
            );
            let cyl_r_sq = cyl_r * cyl_r;
            let e = cdx * cdx + cdy * cdy + cdz * cdz;

            let hit = cyl_cap_eval(
                CylCap {
                    zero, one, half, four, eps, e, rdvv,
                    ap1x, ap1y, ap1z, adx, ady, adz, cap_r, cap_r_sq,
                    cp1x, cp1y, cp1z, cdx, cdy, cdz, cyl_r, cyl_r_sq,
                },
            );

            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    /// Squared-distance breakpoint test of a segment (in a cuboid's local frame: `p0`, `dir`)
    /// against the cuboid `[-he, he]` expanded by `rs` (radius²). Samples the 8 convex
    /// breakpoints (2 endpoints + 6 slab crossings). Shared by the cylinder-cuboid barrel test.
    #[allow(clippy::too_many_arguments)]
    fn cuboid_seg_hit<S: Backend<f32>>(
        zero: Varying<f32, S>,
        one: Varying<f32, S>,
        eps: Varying<f32, S>,
        big: Varying<f32, S>,
        p0: [Varying<f32, S>; 3],
        dir: [Varying<f32, S>; 3],
        he: [Varying<f32, S>; 3],
        rs: Varying<f32, S>,
    ) -> Mask<f32, S> {
        let inv: [Varying<f32, S>; 3] =
            core::array::from_fn(|i| (one / dir[i]).select(dir[i].abs().gt(eps), big));
        let dist_sq = |t: Varying<f32, S>| {
            let mut d = zero;
            for i in 0..3 {
                let ex = ((t * dir[i] + p0[i]).abs() - he[i]).max(zero);
                d = d + ex * ex;
            }
            d
        };
        let mut hit = dist_sq(zero).le(rs) | dist_sq(one).le(rs);
        for i in 0..3 {
            let lo = (((zero - he[i]) - p0[i]) * inv[i]).max(zero).min(one);
            let hi = ((he[i] - p0[i]) * inv[i]).max(zero).min(one);
            hit = hit | dist_sq(lo).le(rs) | dist_sq(hi).le(rs);
        }
        hit
    }

    /// Is a point `(cx,cy,cz)` inside a cylinder's barrel (within the axis slab and radius² `rs`)?
    #[allow(clippy::too_many_arguments)]
    fn corner_in_cyl<S: Backend<f32>>(
        zero: Varying<f32, S>,
        one: Varying<f32, S>,
        c: [Varying<f32, S>; 3],
        p1: [Varying<f32, S>; 3],
        d: [Varying<f32, S>; 3],
        rdv: Varying<f32, S>,
        rs: Varying<f32, S>,
    ) -> Mask<f32, S> {
        let wx = c[0] - p1[0];
        let wy = c[1] - p1[1];
        let wz = c[2] - p1[2];
        let t = (wx * d[0] + wy * d[1] + wz * d[2]) * rdv;
        let in_slab = zero.le(t) & t.le(one);
        let perpx = wx - d[0] * t;
        let perpy = wy - d[1] * t;
        let perpz = wz - d[2] * t;
        let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;
        in_slab & r_sq.le(rs)
    }

    const CUBOID_SIGNS: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ];

    pub fn cylinder_vs_cuboids_broad(q: &Cylinder, col: &ShapeCollection<Cuboid>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let (bc, br) = q.bounding_sphere();
        cylinder_vs_cuboids_broad_k(
            col,
            [bc.x, bc.y, bc.z, br],
            q.p1.to_array(),
            q.dir.to_array(),
            q.rdv,
            q.radius * q.radius,
        )
    }

    /// Query cylinder vs every stored cuboid: cylinder axis sampled against the cuboid faces
    /// (barrel), plus the 8 cuboid corners tested against the cylinder barrel. A SIMD-over-shapes
    /// port of `cylinder_cuboid_collides`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    fn cylinder_vs_cuboids_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cuboid>,
        qbp: [f32; 4],
        cylp1: [f32; 3],
        cyld: [f32; 3],
        rdv: f32,
        rs_sq: f32,
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let eps = ctx.splat(f32::EPSILON);
        let big = ctx.splat(f32::MAX);
        let rs = ctx.splat(rs_sq);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let p1 = [ctx.splat(cylp1[0]), ctx.splat(cylp1[1]), ctx.splat(cylp1[2])];
        let cd = [ctx.splat(cyld[0]), ctx.splat(cyld[1]), ctx.splat(cyld[2])];
        let rdvv = ctx.splat(rdv);

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let cen = ctx.load_partial_n([&s.col(0)[r.clone()], &s.col(1)[r.clone()], &s.col(2)[r.clone()]], 0.0);
            let ax: [[Varying<f32, _>; 3]; 3] = core::array::from_fn(|i| {
                core::array::from_fn(|k| ctx.load_partial(&s.col(3 + i * 3 + k)[r.clone()], 0.0))
            });
            let he: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| ctx.load_partial(&s.col(12 + i)[r.clone()], 0.0));

            let w = [p1[0] - cen[0], p1[1] - cen[1], p1[2] - cen[2]];
            let p0: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| w[0] * ax[i][0] + w[1] * ax[i][1] + w[2] * ax[i][2]);
            let dl: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| cd[0] * ax[i][0] + cd[1] * ax[i][1] + cd[2] * ax[i][2]);

            let mut hit = cuboid_seg_hit(zero, one, eps, big, p0, dl, he, rs);
            for sg in CUBOID_SIGNS {
                let off: [Varying<f32, _>; 3] = core::array::from_fn(|k| {
                    ax[0][k] * (he[0] * sg[0]) + ax[1][k] * (he[1] * sg[1]) + ax[2][k] * (he[2] * sg[2])
                });
                let corner = [cen[0] + off[0], cen[1] + off[1], cen[2] + off[2]];
                hit = hit | corner_in_cyl(zero, one, corner, p1, cd, rdvv, rs);
            }

            if (hit & active).any() {
                return true;
            }
        }
        false
    }

    pub fn cuboid_vs_cylinders_broad(q: &Cuboid, col: &ShapeCollection<Cylinder>) -> bool {
        match batch_plan(col.len(), &col.broad, &q.broadphase()) {
            BatchPlan::Reject => return false,
            BatchPlan::Scalar => return col.collides(q),
            BatchPlan::Simd => {}
        }
        let br = q.bounding_sphere_radius();
        let he = q.half_extents;
        let mut corners = [[0.0f32; 3]; 8];
        for (ci, sg) in CUBOID_SIGNS.iter().enumerate() {
            let v = q.center
                + q.axes[0] * (he[0] * sg[0])
                + q.axes[1] * (he[1] * sg[1])
                + q.axes[2] * (he[2] * sg[2]);
            corners[ci] = [v.x, v.y, v.z];
        }
        cuboid_vs_cylinders_broad_k(
            col,
            [q.center.x, q.center.y, q.center.z, br],
            [q.axes[0].to_array(), q.axes[1].to_array(), q.axes[2].to_array()],
            he,
            corners,
        )
    }

    /// Query cuboid vs every stored cylinder: companion of `cylinder_vs_cuboids_broad` with the
    /// cuboid splatted (its 8 corners precomputed on the host) and the cylinder read per-lane.
    #[kernel]
    fn cuboid_vs_cylinders_broad_k<'a>(
        ctx: Gang,
        col: &'a ShapeCollection<Cylinder>,
        qbp: [f32; 4],
        qaxes: [[f32; 3]; 3],
        qhe: [f32; 3],
        qcorners: [[f32; 3]; 8],
    ) -> bool {
        let len = col.len();
        let s = &col.shapes;
        let b = &col.broad;
        let zero = ctx.splat(0.0);
        let one = ctx.splat(1.0);
        let eps = ctx.splat(f32::EPSILON);
        let big = ctx.splat(f32::MAX);

        let qcx = ctx.splat(qbp[0]);
        let qcy = ctx.splat(qbp[1]);
        let qcz = ctx.splat(qbp[2]);
        let qbr = ctx.splat(qbp[3]);
        let qc = [qcx, qcy, qcz];
        let ax: [[Varying<f32, _>; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|k| ctx.splat(qaxes[i][k])));
        let he: [Varying<f32, _>; 3] = core::array::from_fn(|i| ctx.splat(qhe[i]));
        let corners: [[Varying<f32, _>; 3]; 8] =
            core::array::from_fn(|ci| core::array::from_fn(|k| ctx.splat(qcorners[ci][k])));

        for (off, cnt, active) in ctx.masked_chunks::<f32>(len) {
            let r = off..off + cnt;

            let [bx, by, bz, br] =
                ctx.load_partial_n([&b.x()[r.clone()], &b.y()[r.clone()], &b.z()[r.clone()], &b.r()[r.clone()]], 0.0);
            let bdx = qcx - bx;
            let bdy = qcy - by;
            let bdz = qcz - bz;
            let bmax = qbr + br;
            if !((bdx * bdx + bdy * bdy + bdz * bdz).le(bmax * bmax) & active).any() {
                continue;
            }

            let p1 = ctx.load_partial_n([&s.col(0)[r.clone()], &s.col(1)[r.clone()], &s.col(2)[r.clone()]], 0.0);
            let cd = ctx.load_partial_n([&s.col(3)[r.clone()], &s.col(4)[r.clone()], &s.col(5)[r.clone()]], 0.0);
            let crad = ctx.load_partial(&s.col(6)[r.clone()], 0.0);
            let rs = crad * crad;
            let rdvv = ctx.load_partial(&s.col(7)[r.clone()], 0.0);

            let w = [p1[0] - qc[0], p1[1] - qc[1], p1[2] - qc[2]];
            let p0: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| w[0] * ax[i][0] + w[1] * ax[i][1] + w[2] * ax[i][2]);
            let dl: [Varying<f32, _>; 3] =
                core::array::from_fn(|i| cd[0] * ax[i][0] + cd[1] * ax[i][1] + cd[2] * ax[i][2]);

            let mut hit = cuboid_seg_hit(zero, one, eps, big, p0, dl, he, rs);
            for corner in corners {
                hit = hit | corner_in_cyl(zero, one, corner, p1, cd, rdvv, rs);
            }

            if (hit & active).any() {
                return true;
            }
        }
        false
    }
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized + serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.items.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T> serde::Deserialize<'de> for BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let items = Vec::<T>::deserialize(deserializer)?;
        Ok(Self::new(items))
    }
}
