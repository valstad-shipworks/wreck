use alloc::vec::Vec;
use core::fmt::Debug;

use glam::Vec3;
use wide::{CmpLe, f32x8};

use crate::{Bounded, Collides, Scalable, Sphere, Transformable};

/// Structure-of-Arrays storage for spheres.
///
/// Backed by a single contiguous allocation laid out as `[x; padded][y; padded][z; padded][r; padded]`
/// so that all four channels share one cache-line neighbourhood.
/// Each channel is padded to a multiple of 16 so SIMD loops never need a scalar
/// remainder path (works for both 8-wide and 16-wide).
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
        Ok(Self { buf, padded, len: h.len })
    }
}

const PAD: usize = 16;
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

        for i in len..padded {
            buf[3 * padded + i] = f32::NAN;
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

        for i in len..padded {
            buf[3 * padded + i] = f32::NAN;
        }

        Self { buf, padded, len }
    }

    #[inline]
    pub fn x(&self) -> &[f32] {
        &self.buf[..self.padded]
    }

    #[inline]
    pub fn y(&self) -> &[f32] {
        &self.buf[self.padded..2 * self.padded]
    }

    #[inline]
    pub fn z(&self) -> &[f32] {
        &self.buf[2 * self.padded..3 * self.padded]
    }

    #[inline]
    pub fn r(&self) -> &[f32] {
        &self.buf[3 * self.padded..4 * self.padded]
    }

    #[inline]
    pub fn slices_mut(&mut self) -> (&mut [f32], &mut [f32], &mut [f32], &mut [f32]) {
        let p = self.padded;
        let (x, rest) = self.buf.split_at_mut(p);
        let (y, rest) = rest.split_at_mut(p);
        let (z, r) = rest.split_at_mut(p);
        (x, y, z, r)
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
        for i in old..new {
            buf[3 * new + i] = f32::NAN;
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
        buf[2 * new_padded + sl..2 * new_padded + sl + ol].copy_from_slice(&other.buf[2 * op..2 * op + ol]);

        buf[3 * new_padded..3 * new_padded + sl].copy_from_slice(&self.buf[3 * sp..3 * sp + sl]);
        buf[3 * new_padded + sl..3 * new_padded + sl + ol].copy_from_slice(&other.buf[3 * op..3 * op + ol]);

        for i in new_len..new_padded {
            buf[3 * new_padded + i] = f32::NAN;
        }

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
        buf[2 * new_padded + sl..2 * new_padded + sl + ol].copy_from_slice(&other.buf[2 * op..2 * op + ol]);

        buf[3 * new_padded..3 * new_padded + sl].copy_from_slice(&self.buf[3 * sp..3 * sp + sl]);
        buf[3 * new_padded + sl..3 * new_padded + sl + ol].copy_from_slice(&other.buf[3 * op..3 * op + ol]);

        for i in new_len..new_padded {
            buf[3 * new_padded + i] = f32::NAN;
        }

        self.buf = buf;
        self.padded = new_padded;
        self.len = new_len;
    }

    pub fn clear(&mut self) {
        let p = self.padded;
        for i in 0..self.len {
            self.buf[3 * p + i] = f32::NAN;
        }
        self.len = 0;
    }

    #[inline]
    pub fn get(&self, index: usize) -> Sphere {
        debug_assert!(index < self.len);
        let p = self.padded;
        Sphere::new(
            Vec3::new(self.buf[index], self.buf[p + index], self.buf[2 * p + index]),
            self.buf[3 * p + index],
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = Sphere> + '_ {
        (0..self.len).map(|i| self.get(i))
    }

    #[inline]
    pub(crate) fn chunk_count_8(&self) -> usize {
        self.padded / 8
    }

    /// Test if any sphere in this SoA collides with the given sphere.
    #[inline]
    pub fn any_collides_sphere(&self, sphere: &Sphere) -> bool {
        if self.is_empty() {
            return false;
        }

        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::any_collides_sphere_avx512(self, sphere) }
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    return unsafe { avx512::any_collides_sphere_avx512(self, sphere) };
                }
                self.any_collides_sphere_f32x8(sphere)
            } else {
                self.any_collides_sphere_f32x8(sphere)
            }
        }
    }

    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn any_collides_sphere_f32x8(&self, sphere: &Sphere) -> bool {
        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let sr = f32x8::splat(sphere.radius);

        let xp = self.x().as_ptr();
        let yp = self.y().as_ptr();
        let zp = self.z().as_ptr();
        let rp = self.r().as_ptr();

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let ox;
            let oy;
            let oz;
            let or;
            unsafe {
                ox = f32x8::new(*xp.add(base).cast::<[f32; 8]>());
                oy = f32x8::new(*yp.add(base).cast::<[f32; 8]>());
                oz = f32x8::new(*zp.add(base).cast::<[f32; 8]>());
                or = f32x8::new(*rp.add(base).cast::<[f32; 8]>());
            }

            let dx = cx - ox;
            let dy = cy - oy;
            let dz = cz - oz;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let rs = sr + or;
            if dist_sq.simd_le(rs * rs).any() {
                return true;
            }
        }

        false
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

        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::broadphase_collect_avx512(self, query, out) }
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    return unsafe { avx512::broadphase_collect_avx512(self, query, out) };
                }
                self.broadphase_collect_f32x8(query, out)
            } else {
                self.broadphase_collect_f32x8(query, out)
            }
        }
    }

    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn broadphase_collect_f32x8(&self, query: &Sphere, out: &mut [bool]) -> bool {
        let cx = f32x8::splat(query.center.x);
        let cy = f32x8::splat(query.center.y);
        let cz = f32x8::splat(query.center.z);
        let sr = f32x8::splat(query.radius);
        let mut any_hit = false;

        let xs = self.x();
        let ys = self.y();
        let zs = self.z();
        let rs = self.r();

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let ox = f32x8::new(xs[base..base + 8].try_into().unwrap());
            let oy = f32x8::new(ys[base..base + 8].try_into().unwrap());
            let oz = f32x8::new(zs[base..base + 8].try_into().unwrap());
            let or = f32x8::new(rs[base..base + 8].try_into().unwrap());

            let dx = cx - ox;
            let dy = cy - oy;
            let dz = cz - oz;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let rsum = sr + or;
            let mask = dist_sq.simd_le(rsum * rsum).to_bitmask();
            if mask != 0 {
                any_hit = true;
                let end = (base + 8).min(self.len);
                for j in base..end {
                    if (mask >> (j - base)) & 1 != 0 {
                        out[j] = true;
                    }
                }
            }
        }

        any_hit
    }

    /// Test if any sphere in `self` collides with any sphere in `other`.
    ///
    /// For each sphere in `self`, broadcasts its position across all chunks
    /// of `other` — O(n*m) but with no per-chunk transpose overhead.
    pub fn any_collides_soa(&self, other: &SpheresSoA) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }

        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::any_collides_soa_avx512(self, other) }
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    return unsafe { avx512::any_collides_soa_avx512(self, other) };
                }
                self.any_collides_soa_f32x8(other)
            } else {
                self.any_collides_soa_f32x8(other)
            }
        }
    }

    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn any_collides_soa_f32x8(&self, other: &SpheresSoA) -> bool {
        let other_chunks = other.chunk_count_8();
        let oxp = other.x().as_ptr();
        let oyp = other.y().as_ptr();
        let ozp = other.z().as_ptr();
        let orp = other.r().as_ptr();

        let sxp = self.x().as_ptr();
        let syp = self.y().as_ptr();
        let szp = self.z().as_ptr();
        let srp = self.r().as_ptr();

        for i in 0..self.len {
            let cx;
            let cy;
            let cz;
            let sr;
            unsafe {
                cx = f32x8::splat(*sxp.add(i));
                cy = f32x8::splat(*syp.add(i));
                cz = f32x8::splat(*szp.add(i));
                sr = f32x8::splat(*srp.add(i));
            }

            for j in 0..other_chunks {
                let base = j * 8;
                let (ox, oy, oz, or);
                unsafe {
                    ox = f32x8::new(*oxp.add(base).cast::<[f32; 8]>());
                    oy = f32x8::new(*oyp.add(base).cast::<[f32; 8]>());
                    oz = f32x8::new(*ozp.add(base).cast::<[f32; 8]>());
                    or = f32x8::new(*orp.add(base).cast::<[f32; 8]>());
                }

                let dx = cx - ox;
                let dy = cy - oy;
                let dz = cz - oz;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let rs = sr + or;
                if dist_sq.simd_le(rs * rs).any() {
                    return true;
                }
            }
        }

        false
    }
}

impl Transformable for SpheresSoA {
    fn translate(&mut self, offset: glam::Vec3A) {
        let offset = Vec3::from(offset);
        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::translate_avx512(self, offset) };
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    unsafe { avx512::translate_avx512(self, offset) };
                    return;
                }
                self.translate_f32x8(offset);
            } else {
                self.translate_f32x8(offset);
            }
        }
    }

    fn rotate_mat(&mut self, mat: glam::Mat3A) {
        let mat = glam::Mat3::from(mat);
        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::rotate_mat_avx512(self, mat) };
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    unsafe { avx512::rotate_mat_avx512(self, mat) };
                    return;
                }
                self.rotate_mat_f32x8(mat);
            } else {
                self.rotate_mat_f32x8(mat);
            }
        }
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        self.rotate_mat(glam::Mat3A::from_quat(quat));
    }

    fn transform(&mut self, mat: glam::Affine3A) {
        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::transform_avx512(self, mat) };
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    unsafe { avx512::transform_avx512(self, mat) };
                    return;
                }
                self.transform_f32x8(mat);
            } else {
                self.transform_f32x8(mat);
            }
        }
    }
}

impl SpheresSoA {
    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn translate_f32x8(&mut self, offset: Vec3) {
        let ox = f32x8::splat(offset.x);
        let oy = f32x8::splat(offset.y);
        let oz = f32x8::splat(offset.z);
        let chunks = self.chunk_count_8();
        let (xs, ys, zs, _) = self.slices_mut();
        let xp = xs.as_mut_ptr();
        let yp = ys.as_mut_ptr();
        let zp = zs.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 8;
            unsafe {
                let x = xp.add(base).cast::<[f32; 8]>();
                let y = yp.add(base).cast::<[f32; 8]>();
                let z = zp.add(base).cast::<[f32; 8]>();
                *x = *(f32x8::new(*x) + ox).as_array();
                *y = *(f32x8::new(*y) + oy).as_array();
                *z = *(f32x8::new(*z) + oz).as_array();
            }
        }
    }

    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn rotate_mat_f32x8(&mut self, mat: glam::Mat3) {
        let m00 = f32x8::splat(mat.x_axis.x);
        let m01 = f32x8::splat(mat.y_axis.x);
        let m02 = f32x8::splat(mat.z_axis.x);
        let m10 = f32x8::splat(mat.x_axis.y);
        let m11 = f32x8::splat(mat.y_axis.y);
        let m12 = f32x8::splat(mat.z_axis.y);
        let m20 = f32x8::splat(mat.x_axis.z);
        let m21 = f32x8::splat(mat.y_axis.z);
        let m22 = f32x8::splat(mat.z_axis.z);
        let chunks = self.chunk_count_8();
        let (xs, ys, zs, _) = self.slices_mut();
        let xp = xs.as_mut_ptr();
        let yp = ys.as_mut_ptr();
        let zp = zs.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 8;
            unsafe {
                let xsl = xp.add(base).cast::<[f32; 8]>();
                let ysl = yp.add(base).cast::<[f32; 8]>();
                let zsl = zp.add(base).cast::<[f32; 8]>();
                let x = f32x8::new(*xsl);
                let y = f32x8::new(*ysl);
                let z = f32x8::new(*zsl);
                *xsl = *(m00 * x + m01 * y + m02 * z).as_array();
                *ysl = *(m10 * x + m11 * y + m12 * z).as_array();
                *zsl = *(m20 * x + m21 * y + m22 * z).as_array();
            }
        }
    }

    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn transform_f32x8(&mut self, mat: glam::Affine3A) {
        let rot = mat.matrix3;
        let m00 = f32x8::splat(rot.x_axis.x);
        let m01 = f32x8::splat(rot.y_axis.x);
        let m02 = f32x8::splat(rot.z_axis.x);
        let m10 = f32x8::splat(rot.x_axis.y);
        let m11 = f32x8::splat(rot.y_axis.y);
        let m12 = f32x8::splat(rot.z_axis.y);
        let m20 = f32x8::splat(rot.x_axis.z);
        let m21 = f32x8::splat(rot.y_axis.z);
        let m22 = f32x8::splat(rot.z_axis.z);
        let tx = f32x8::splat(mat.translation.x);
        let ty = f32x8::splat(mat.translation.y);
        let tz = f32x8::splat(mat.translation.z);
        let chunks = self.chunk_count_8();
        let (xs, ys, zs, _) = self.slices_mut();
        let xp = xs.as_mut_ptr();
        let yp = ys.as_mut_ptr();
        let zp = zs.as_mut_ptr();

        for i in 0..chunks {
            let base = i * 8;
            unsafe {
                let xsl = xp.add(base).cast::<[f32; 8]>();
                let ysl = yp.add(base).cast::<[f32; 8]>();
                let zsl = zp.add(base).cast::<[f32; 8]>();
                let x = f32x8::new(*xsl);
                let y = f32x8::new(*ysl);
                let z = f32x8::new(*zsl);
                *xsl = *(m00 * x + m01 * y + m02 * z + tx).as_array();
                *ysl = *(m10 * x + m11 * y + m12 * z + ty).as_array();
                *zsl = *(m20 * x + m21 * y + m22 * z + tz).as_array();
            }
        }
    }
}

impl Scalable for SpheresSoA {
    /// Scale all radii by a factor.
    fn scale(&mut self, factor: f32) {
        cfg_if::cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
                unsafe { avx512::scale_avx512(self, factor) };
            } else if #[cfg(all(target_arch = "x86_64", feature = "std"))] {
                if is_x86_feature_detected!("avx512f") {
                    unsafe { avx512::scale_avx512(self, factor) };
                    return;
                }
                self.scale_f32x8(factor);
            } else {
                self.scale_f32x8(factor);
            }
        }
    }
}

impl SpheresSoA {
    #[cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f"), allow(dead_code))]
    fn scale_f32x8(&mut self, factor: f32) {
        let f = f32x8::splat(factor);
        let chunks = self.chunk_count_8();
        let (_, _, _, rs) = self.slices_mut();

        for i in 0..chunks {
            let base = i * 8;
            let r: [f32; 8] = rs[base..base + 8].try_into().unwrap();
            let scaled = f32x8::new(r) * f;
            rs[base..base + 8].copy_from_slice(scaled.as_array());
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 implementations (16 floats per register)
// ---------------------------------------------------------------------------
#[cfg(all(
    target_arch = "x86_64",
    any(feature = "std", target_feature = "avx512f")
))]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx512 {
    use super::SpheresSoA;
    use crate::Sphere;
    use glam::Vec3;
    use core::arch::x86_64::*;

    #[inline]
    fn chunk_count_16(soa: &SpheresSoA) -> usize {
        soa.padded / 16
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn any_collides_sphere_avx512(soa: &SpheresSoA, sphere: &Sphere) -> bool {
        let cx = _mm512_set1_ps(sphere.center.x);
        let cy = _mm512_set1_ps(sphere.center.y);
        let cz = _mm512_set1_ps(sphere.center.z);
        let sr = _mm512_set1_ps(sphere.radius);

        let xs = soa.x();
        let ys = soa.y();
        let zs = soa.z();
        let rs = soa.r();

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let ox = _mm512_loadu_ps(xs[base..].as_ptr());
            let oy = _mm512_loadu_ps(ys[base..].as_ptr());
            let oz = _mm512_loadu_ps(zs[base..].as_ptr());
            let or = _mm512_loadu_ps(rs[base..].as_ptr());

            let dx = _mm512_sub_ps(cx, ox);
            let dy = _mm512_sub_ps(cy, oy);
            let dz = _mm512_sub_ps(cz, oz);

            let dist_sq = _mm512_fmadd_ps(dz, dz, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dx, dx)));

            let rsum = _mm512_add_ps(sr, or);
            let rs_sq = _mm512_mul_ps(rsum, rsum);

            let mask = _mm512_cmp_ps_mask::<0x12>(dist_sq, rs_sq);
            if mask != 0 {
                return true;
            }
        }

        false
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn broadphase_collect_avx512(
        soa: &SpheresSoA,
        query: &Sphere,
        out: &mut [bool],
    ) -> bool {
        let cx = _mm512_set1_ps(query.center.x);
        let cy = _mm512_set1_ps(query.center.y);
        let cz = _mm512_set1_ps(query.center.z);
        let sr = _mm512_set1_ps(query.radius);
        let mut any_hit = false;

        let xs = soa.x();
        let ys = soa.y();
        let zs = soa.z();
        let rs = soa.r();

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let ox = _mm512_loadu_ps(xs[base..].as_ptr());
            let oy = _mm512_loadu_ps(ys[base..].as_ptr());
            let oz = _mm512_loadu_ps(zs[base..].as_ptr());
            let or = _mm512_loadu_ps(rs[base..].as_ptr());

            let dx = _mm512_sub_ps(cx, ox);
            let dy = _mm512_sub_ps(cy, oy);
            let dz = _mm512_sub_ps(cz, oz);

            let dist_sq = _mm512_fmadd_ps(dz, dz, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dx, dx)));

            let rsum = _mm512_add_ps(sr, or);
            let rs_sq = _mm512_mul_ps(rsum, rsum);

            let mask = _mm512_cmp_ps_mask::<0x12>(dist_sq, rs_sq);
            if mask != 0 {
                any_hit = true;
                let end = (base + 16).min(soa.len);
                for j in base..end {
                    if (mask >> (j - base)) & 1 != 0 {
                        out[j] = true;
                    }
                }
            }
        }

        any_hit
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn any_collides_soa_avx512(a: &SpheresSoA, b: &SpheresSoA) -> bool {
        let b_chunks = chunk_count_16(b);
        let axs = a.x();
        let ays = a.y();
        let azs = a.z();
        let ars = a.r();
        let bxs = b.x();
        let bys = b.y();
        let bzs = b.z();
        let brs = b.r();

        for i in 0..a.len {
            let cx = _mm512_set1_ps(axs[i]);
            let cy = _mm512_set1_ps(ays[i]);
            let cz = _mm512_set1_ps(azs[i]);
            let sr = _mm512_set1_ps(ars[i]);

            for j in 0..b_chunks {
                let base = j * 16;
                let ox = _mm512_loadu_ps(bxs[base..].as_ptr());
                let oy = _mm512_loadu_ps(bys[base..].as_ptr());
                let oz = _mm512_loadu_ps(bzs[base..].as_ptr());
                let or = _mm512_loadu_ps(brs[base..].as_ptr());

                let dx = _mm512_sub_ps(cx, ox);
                let dy = _mm512_sub_ps(cy, oy);
                let dz = _mm512_sub_ps(cz, oz);

                let dist_sq =
                    _mm512_fmadd_ps(dz, dz, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dx, dx)));

                let rsum = _mm512_add_ps(sr, or);
                let rs_sq = _mm512_mul_ps(rsum, rsum);

                let mask = _mm512_cmp_ps_mask::<0x12>(dist_sq, rs_sq);
                if mask != 0 {
                    return true;
                }
            }
        }

        false
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn translate_avx512(soa: &mut SpheresSoA, offset: Vec3) {
        let ox = _mm512_set1_ps(offset.x);
        let oy = _mm512_set1_ps(offset.y);
        let oz = _mm512_set1_ps(offset.z);
        let chunks = chunk_count_16(soa);
        let (xs, ys, zs, _) = soa.slices_mut();

        for i in 0..chunks {
            let base = i * 16;
            let xp = xs[base..].as_mut_ptr();
            let yp = ys[base..].as_mut_ptr();
            let zp = zs[base..].as_mut_ptr();
            _mm512_storeu_ps(xp, _mm512_add_ps(_mm512_loadu_ps(xp), ox));
            _mm512_storeu_ps(yp, _mm512_add_ps(_mm512_loadu_ps(yp), oy));
            _mm512_storeu_ps(zp, _mm512_add_ps(_mm512_loadu_ps(zp), oz));
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn scale_avx512(soa: &mut SpheresSoA, factor: f32) {
        let f = _mm512_set1_ps(factor);
        let chunks = chunk_count_16(soa);
        let (_, _, _, rs) = soa.slices_mut();

        for i in 0..chunks {
            let base = i * 16;
            let rp = rs[base..].as_mut_ptr();
            _mm512_storeu_ps(rp, _mm512_mul_ps(_mm512_loadu_ps(rp), f));
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn rotate_mat_avx512(soa: &mut SpheresSoA, mat: glam::Mat3) {
        let m00 = _mm512_set1_ps(mat.x_axis.x);
        let m01 = _mm512_set1_ps(mat.y_axis.x);
        let m02 = _mm512_set1_ps(mat.z_axis.x);
        let m10 = _mm512_set1_ps(mat.x_axis.y);
        let m11 = _mm512_set1_ps(mat.y_axis.y);
        let m12 = _mm512_set1_ps(mat.z_axis.y);
        let m20 = _mm512_set1_ps(mat.x_axis.z);
        let m21 = _mm512_set1_ps(mat.y_axis.z);
        let m22 = _mm512_set1_ps(mat.z_axis.z);
        let chunks = chunk_count_16(soa);
        let (xs, ys, zs, _) = soa.slices_mut();

        for i in 0..chunks {
            let base = i * 16;
            let xp = xs[base..].as_mut_ptr();
            let yp = ys[base..].as_mut_ptr();
            let zp = zs[base..].as_mut_ptr();
            let x = _mm512_loadu_ps(xp);
            let y = _mm512_loadu_ps(yp);
            let z = _mm512_loadu_ps(zp);

            let nx = _mm512_fmadd_ps(m02, z, _mm512_fmadd_ps(m01, y, _mm512_mul_ps(m00, x)));
            let ny = _mm512_fmadd_ps(m12, z, _mm512_fmadd_ps(m11, y, _mm512_mul_ps(m10, x)));
            let nz = _mm512_fmadd_ps(m22, z, _mm512_fmadd_ps(m21, y, _mm512_mul_ps(m20, x)));

            _mm512_storeu_ps(xp, nx);
            _mm512_storeu_ps(yp, ny);
            _mm512_storeu_ps(zp, nz);
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn transform_avx512(soa: &mut SpheresSoA, mat: glam::Affine3A) {
        let rot = mat.matrix3;
        let m00 = _mm512_set1_ps(rot.x_axis.x);
        let m01 = _mm512_set1_ps(rot.y_axis.x);
        let m02 = _mm512_set1_ps(rot.z_axis.x);
        let m10 = _mm512_set1_ps(rot.x_axis.y);
        let m11 = _mm512_set1_ps(rot.y_axis.y);
        let m12 = _mm512_set1_ps(rot.z_axis.y);
        let m20 = _mm512_set1_ps(rot.x_axis.z);
        let m21 = _mm512_set1_ps(rot.y_axis.z);
        let m22 = _mm512_set1_ps(rot.z_axis.z);
        let tx = _mm512_set1_ps(mat.translation.x);
        let ty = _mm512_set1_ps(mat.translation.y);
        let tz = _mm512_set1_ps(mat.translation.z);
        let chunks = chunk_count_16(soa);
        let (xs, ys, zs, _) = soa.slices_mut();

        for i in 0..chunks {
            let base = i * 16;
            let xp = xs[base..].as_mut_ptr();
            let yp = ys[base..].as_mut_ptr();
            let zp = zs[base..].as_mut_ptr();
            let x = _mm512_loadu_ps(xp);
            let y = _mm512_loadu_ps(yp);
            let z = _mm512_loadu_ps(zp);

            let nx = _mm512_fmadd_ps(m02, z, _mm512_fmadd_ps(m01, y, _mm512_fmadd_ps(m00, x, tx)));
            let ny = _mm512_fmadd_ps(m12, z, _mm512_fmadd_ps(m11, y, _mm512_fmadd_ps(m10, x, ty)));
            let nz = _mm512_fmadd_ps(m22, z, _mm512_fmadd_ps(m21, y, _mm512_fmadd_ps(m20, x, tz)));

            _mm512_storeu_ps(xp, nx);
            _mm512_storeu_ps(yp, ny);
            _mm512_storeu_ps(zp, nz);
        }
    }
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

impl<T> BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    /// SIMD broadphase + scalar narrowphase.
    ///
    /// Quick-rejects via bounding sphere overlap, then runs narrowphase
    /// on all items (accepting some wasted work on non-overlapping items
    /// to avoid scratch-buffer bookkeeping).
    pub fn collides<U>(&self, shape: &U) -> bool
    where
        U: Collides<T> + Bounded,
    {
        let bp = shape.broadphase();
        if !self.broad.any_collides_sphere(&bp) {
            return false;
        }
        self.items.iter().any(|item| shape.test::<false>(item))
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

pub(crate) mod batch {
    use glam::Vec3;
    use wide::{CmpLe, f32x8};
    #[cfg(not(feature = "std"))]
    #[allow(unused_imports)]
    use crate::F32Ext;

    use super::{BroadCollection, SpheresSoA};
    use crate::Collides;
    use crate::capsule::Capsule;
    use crate::cuboid::Cuboid;
    use crate::cylinder::Cylinder;
    use crate::line::{Line, LineSegment, Ray};
    use crate::plane::Plane;
    use crate::sphere::Sphere;

    pub fn plane_vs_spheres_soa(plane: &Plane, soa: &SpheresSoA) -> bool {
        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;
        let xp = soa.x().as_ptr();
        let yp = soa.y().as_ptr();
        let zp = soa.z().as_ptr();
        let rp = soa.r().as_ptr();

        for i in 0..soa.chunk_count_8() {
            let base = i * 8;
            let (cx, cy, cz, r);
            unsafe {
                cx = f32x8::new(*xp.add(base).cast::<[f32; 8]>());
                cy = f32x8::new(*yp.add(base).cast::<[f32; 8]>());
                cz = f32x8::new(*zp.add(base).cast::<[f32; 8]>());
                r = f32x8::new(*rp.add(base).cast::<[f32; 8]>());
            }
            let proj = nx * cx + ny * cy + nz * cz;
            let sep = proj - d - r;
            if sep.simd_le(zero).any() {
                return true;
            }
        }
        false
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
        let ox = f32x8::splat(origin.x);
        let oy = f32x8::splat(origin.y);
        let oz = f32x8::splat(origin.z);
        let dx = f32x8::splat(dir.x);
        let dy = f32x8::splat(dir.y);
        let dz = f32x8::splat(dir.z);
        let rdv8 = f32x8::splat(rdv);
        let lo = f32x8::splat(t_min);
        let hi = f32x8::splat(t_max);
        let xp = soa.x().as_ptr();
        let yp = soa.y().as_ptr();
        let zp = soa.z().as_ptr();
        let rp = soa.r().as_ptr();

        for i in 0..soa.chunk_count_8() {
            let base = i * 8;
            let (cx, cy, cz, r);
            unsafe {
                cx = f32x8::new(*xp.add(base).cast::<[f32; 8]>());
                cy = f32x8::new(*yp.add(base).cast::<[f32; 8]>());
                cz = f32x8::new(*zp.add(base).cast::<[f32; 8]>());
                r = f32x8::new(*rp.add(base).cast::<[f32; 8]>());
            }

            let dfx = cx - ox;
            let dfy = cy - oy;
            let dfz = cz - oz;
            let t = ((dfx * dx + dfy * dy + dfz * dz) * rdv8).max(lo).min(hi);

            let px = ox + dx * t;
            let py = oy + dy * t;
            let pz = oz + dz * t;

            let ex = cx - px;
            let ey = cy - py;
            let ez = cz - pz;
            let dist_sq = ex * ex + ey * ey + ez * ez;

            if dist_sq.simd_le(r * r).any() {
                return true;
            }
        }
        false
    }

    #[inline]
    pub fn line_vs_spheres_soa(line: &Line, soa: &SpheresSoA) -> bool {
        line_vs_spheres_soa_inner(
            line.origin,
            line.dir,
            line.rdv,
            f32::NEG_INFINITY,
            f32::INFINITY,
            soa,
        )
    }

    #[inline]
    pub fn ray_vs_spheres_soa(ray: &Ray, soa: &SpheresSoA) -> bool {
        line_vs_spheres_soa_inner(ray.origin, ray.dir, ray.rdv, 0.0, f32::INFINITY, soa)
    }

    #[inline]
    pub fn segment_vs_spheres_soa(seg: &LineSegment, soa: &SpheresSoA) -> bool {
        line_vs_spheres_soa_inner(seg.p1, seg.dir, seg.rdv, 0.0, 1.0, soa)
    }

    // ── Broadphase-filtered batch functions (Collider paths) ─────────────
    //
    // These read per-chunk bounding spheres from the BroadCollection's
    // SpheresSoA and skip chunks where no bounding sphere overlaps the query.

    #[inline]
    fn broad_sphere_overlaps_chunk(
        cx: f32x8, cy: f32x8, cz: f32x8, sr: f32x8,
        bxp: *const f32, byp: *const f32, bzp: *const f32, brp: *const f32,
        base: usize,
    ) -> bool {
        let (bx, by, bz, br);
        unsafe {
            bx = f32x8::new(*bxp.add(base).cast::<[f32; 8]>());
            by = f32x8::new(*byp.add(base).cast::<[f32; 8]>());
            bz = f32x8::new(*bzp.add(base).cast::<[f32; 8]>());
            br = f32x8::new(*brp.add(base).cast::<[f32; 8]>());
        }
        let dx = cx - bx;
        let dy = cy - by;
        let dz = cz - bz;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let max_r = sr + br;
        dist_sq.simd_le(max_r * max_r).any()
    }

    #[inline]
    fn broad_plane_overlaps_chunk(
        nx: f32x8, ny: f32x8, nz: f32x8, d: f32x8,
        bxp: *const f32, byp: *const f32, bzp: *const f32, brp: *const f32,
        base: usize,
    ) -> bool {
        let (bx, by, bz, br);
        unsafe {
            bx = f32x8::new(*bxp.add(base).cast::<[f32; 8]>());
            by = f32x8::new(*byp.add(base).cast::<[f32; 8]>());
            bz = f32x8::new(*bzp.add(base).cast::<[f32; 8]>());
            br = f32x8::new(*brp.add(base).cast::<[f32; 8]>());
        }
        let proj = nx * bx + ny * by + nz * bz;
        let sep = proj - d - br;
        sep.simd_le(f32x8::ZERO).any()
    }

    pub fn sphere_vs_capsules_broad(sphere: &Sphere, col: &BroadCollection<Capsule>) -> bool {
        let items = col.items();
        let broad = &col.broad;
        let bxp = broad.x().as_ptr();
        let byp = broad.y().as_ptr();
        let bzp = broad.z().as_ptr();
        let brp = broad.r().as_ptr();

        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let sr = f32x8::splat(sphere.radius);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = items.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base = chunk_idx * 8;
            if !broad_sphere_overlaps_chunk(cx, cy, cz, sr, bxp, byp, bzp, brp, base) {
                continue;
            }

            let mut p1x = [0.0f32; 8];
            let mut p1y = [0.0f32; 8];
            let mut p1z = [0.0f32; 8];
            let mut dx = [0.0f32; 8];
            let mut dy = [0.0f32; 8];
            let mut dz = [0.0f32; 8];
            let mut rdv = [0.0f32; 8];
            let mut cr = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                p1x[i] = c.p1.x;
                p1y[i] = c.p1.y;
                p1z[i] = c.p1.z;
                dx[i] = c.dir.x;
                dy[i] = c.dir.y;
                dz[i] = c.dir.z;
                rdv[i] = c.rdv;
                cr[i] = c.radius;
            }
            let p1x = f32x8::new(p1x);
            let p1y = f32x8::new(p1y);
            let p1z = f32x8::new(p1z);
            let dx = f32x8::new(dx);
            let dy = f32x8::new(dy);
            let dz = f32x8::new(dz);
            let rdv = f32x8::new(rdv);
            let cr = f32x8::new(cr);

            let dfx = cx - p1x;
            let dfy = cy - p1y;
            let dfz = cz - p1z;
            let t = ((dfx * dx + dfy * dy + dfz * dz) * rdv).max(zero).min(one);

            let clx = p1x + dx * t;
            let cly = p1y + dy * t;
            let clz = p1z + dz * t;

            let ex = cx - clx;
            let ey = cy - cly;
            let ez = cz - clz;
            let dist_sq = ex * ex + ey * ey + ez * ez;

            let rs = sr + cr;
            if dist_sq.simd_le(rs * rs).any() {
                return true;
            }
        }

        remainder.iter().any(|c| sphere.collides(c))
    }

    pub fn sphere_vs_cuboids_broad(sphere: &Sphere, col: &BroadCollection<Cuboid>) -> bool {
        let items = col.items();
        let broad = &col.broad;
        let bxp = broad.x().as_ptr();
        let byp = broad.y().as_ptr();
        let bzp = broad.z().as_ptr();
        let brp = broad.r().as_ptr();

        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let r_sq = f32x8::splat(sphere.radius * sphere.radius);
        let sr = f32x8::splat(sphere.radius);
        let zero = f32x8::splat(0.0);

        let chunks = items.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base = chunk_idx * 8;
            if !broad_sphere_overlaps_chunk(cx, cy, cz, sr, bxp, byp, bzp, brp, base) {
                continue;
            }

            let mut ocx = [0.0f32; 8];
            let mut ocy = [0.0f32; 8];
            let mut ocz = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                ocx[i] = c.center.x;
                ocy[i] = c.center.y;
                ocz[i] = c.center.z;
            }
            let dfx = cx - f32x8::new(ocx);
            let dfy = cy - f32x8::new(ocy);
            let dfz = cz - f32x8::new(ocz);

            let mut dist_sq = zero;
            for axis_idx in 0..3 {
                let mut ax = [0.0f32; 8];
                let mut ay = [0.0f32; 8];
                let mut az = [0.0f32; 8];
                let mut he = [0.0f32; 8];
                for (i, c) in chunk.iter().enumerate() {
                    ax[i] = c.axes[axis_idx].x;
                    ay[i] = c.axes[axis_idx].y;
                    az[i] = c.axes[axis_idx].z;
                    he[i] = c.half_extents[axis_idx];
                }
                let proj = dfx * f32x8::new(ax) + dfy * f32x8::new(ay) + dfz * f32x8::new(az);
                let abs_proj = proj.max(-proj);
                let excess = (abs_proj - f32x8::new(he)).max(zero);
                dist_sq = dist_sq + excess * excess;
            }

            if dist_sq.simd_le(r_sq).any() {
                return true;
            }
        }

        remainder.iter().any(|c| sphere.collides(c))
    }

    pub fn sphere_vs_cylinders_broad(sphere: &Sphere, col: &BroadCollection<Cylinder>) -> bool {
        let items = col.items();
        let broad = &col.broad;
        let bxp = broad.x().as_ptr();
        let byp = broad.y().as_ptr();
        let bzp = broad.z().as_ptr();
        let brp = broad.r().as_ptr();

        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let sr_sq = f32x8::splat(sphere.radius * sphere.radius);
        let sr = f32x8::splat(sphere.radius);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let four = f32x8::splat(4.0);

        let chunks = items.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base = chunk_idx * 8;
            if !broad_sphere_overlaps_chunk(cx, cy, cz, sr, bxp, byp, bzp, brp, base) {
                continue;
            }

            let mut p1x = [0.0f32; 8];
            let mut p1y = [0.0f32; 8];
            let mut p1z = [0.0f32; 8];
            let mut dx = [0.0f32; 8];
            let mut dy = [0.0f32; 8];
            let mut dz = [0.0f32; 8];
            let mut rdv = [0.0f32; 8];
            let mut cr = [0.0f32; 8];
            let mut dir_sq = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                p1x[i] = c.p1.x;
                p1y[i] = c.p1.y;
                p1z[i] = c.p1.z;
                dx[i] = c.dir.x;
                dy[i] = c.dir.y;
                dz[i] = c.dir.z;
                rdv[i] = c.rdv;
                cr[i] = c.radius;
                dir_sq[i] = c.dir.dot(c.dir);
            }
            let p1x = f32x8::new(p1x);
            let p1y = f32x8::new(p1y);
            let p1z = f32x8::new(p1z);
            let dx = f32x8::new(dx);
            let dy = f32x8::new(dy);
            let dz = f32x8::new(dz);
            let rdv = f32x8::new(rdv);
            let cr = f32x8::new(cr);
            let dir_sq = f32x8::new(dir_sq);

            let wx = cx - p1x;
            let wy = cy - p1y;
            let wz = cz - p1z;

            let t = (wx * dx + wy * dy + wz * dz) * rdv;
            let t_c = t.max(zero).min(one);

            let perpx = wx - dx * t;
            let perpy = wy - dy * t;
            let perpz = wz - dz * t;
            let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;

            let in_barrel = zero.simd_le(t) & t.simd_le(one);
            let combined = cr + sr;
            let barrel_hit = in_barrel & r_sq.simd_le(combined * combined);

            let t_excess = t - t_c;
            let d_axial_sq = t_excess * t_excess * dir_sq;
            let cr_sq = cr * cr;

            let inside_r = r_sq.simd_le(cr_sq);
            let endcap_inside = inside_r & d_axial_sq.simd_le(sr_sq);

            let l = r_sq + cr_sq + d_axial_sq - sr_sq;
            let endcap_outside = l.simd_le(zero) | (l * l).simd_le(four * cr_sq * r_sq);

            let not_barrel = !in_barrel;
            let hit = barrel_hit | (not_barrel & (endcap_inside | endcap_outside));
            if hit.any() {
                return true;
            }
        }

        remainder.iter().any(|c| sphere.collides(c))
    }

    pub fn plane_vs_capsules_broad(plane: &Plane, col: &BroadCollection<Capsule>) -> bool {
        let items = col.items();
        let broad = &col.broad;
        let bxp = broad.x().as_ptr();
        let byp = broad.y().as_ptr();
        let bzp = broad.z().as_ptr();
        let brp = broad.r().as_ptr();

        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;

        let chunks = items.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base = chunk_idx * 8;
            if !broad_plane_overlaps_chunk(nx, ny, nz, d, bxp, byp, bzp, brp, base) {
                continue;
            }

            let mut p1x = [0.0f32; 8];
            let mut p1y = [0.0f32; 8];
            let mut p1z = [0.0f32; 8];
            let mut p2x = [0.0f32; 8];
            let mut p2y = [0.0f32; 8];
            let mut p2z = [0.0f32; 8];
            let mut cr = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                let p2 = c.p2();
                p1x[i] = c.p1.x;
                p1y[i] = c.p1.y;
                p1z[i] = c.p1.z;
                p2x[i] = p2.x;
                p2y[i] = p2.y;
                p2z[i] = p2.z;
                cr[i] = c.radius;
            }
            let proj1 = nx * f32x8::new(p1x) + ny * f32x8::new(p1y) + nz * f32x8::new(p1z);
            let proj2 = nx * f32x8::new(p2x) + ny * f32x8::new(p2y) + nz * f32x8::new(p2z);
            let min_proj = proj1.min(proj2);
            let sep = min_proj - d - f32x8::new(cr);
            if sep.simd_le(zero).any() {
                return true;
            }
        }

        remainder.iter().any(|c| plane.collides(c))
    }

    pub fn plane_vs_cuboids_broad(plane: &Plane, col: &BroadCollection<Cuboid>) -> bool {
        let items = col.items();
        let broad = &col.broad;
        let bxp = broad.x().as_ptr();
        let byp = broad.y().as_ptr();
        let bzp = broad.z().as_ptr();
        let brp = broad.r().as_ptr();

        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;

        let chunks = items.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base = chunk_idx * 8;
            if !broad_plane_overlaps_chunk(nx, ny, nz, d, bxp, byp, bzp, brp, base) {
                continue;
            }

            let mut ccx = [0.0f32; 8];
            let mut ccy = [0.0f32; 8];
            let mut ccz = [0.0f32; 8];
            let mut ext = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                ccx[i] = c.center.x;
                ccy[i] = c.center.y;
                ccz[i] = c.center.z;
                ext[i] = plane.normal.dot(c.axes[0]).abs() * c.half_extents[0]
                    + plane.normal.dot(c.axes[1]).abs() * c.half_extents[1]
                    + plane.normal.dot(c.axes[2]).abs() * c.half_extents[2];
            }
            let center_proj = nx * f32x8::new(ccx) + ny * f32x8::new(ccy) + nz * f32x8::new(ccz);
            let sep = center_proj - f32x8::new(ext) - d;
            if sep.simd_le(zero).any() {
                return true;
            }
        }

        remainder.iter().any(|c| plane.collides(c))
    }

    pub fn plane_vs_cylinders_broad(plane: &Plane, col: &BroadCollection<Cylinder>) -> bool {
        let items = col.items();
        let broad = &col.broad;
        let bxp = broad.x().as_ptr();
        let byp = broad.y().as_ptr();
        let bzp = broad.z().as_ptr();
        let brp = broad.r().as_ptr();

        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;

        let chunks = items.chunks_exact(8);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let base = chunk_idx * 8;
            if !broad_plane_overlaps_chunk(nx, ny, nz, d, bxp, byp, bzp, brp, base) {
                continue;
            }

            let mut sep_arr = [0.0f32; 8];
            for (i, c) in chunk.iter().enumerate() {
                let proj1 = plane.normal.dot(c.p1);
                let proj2 = plane.normal.dot(c.p1 + c.dir);
                let min_proj = proj1.min(proj2);

                let dir_sq = c.dir.dot(c.dir);
                let n_dot_dir = plane.normal.dot(c.dir);
                let n_perp_sq = if dir_sq > f32::EPSILON {
                    (1.0 - n_dot_dir * n_dot_dir / dir_sq).max(0.0)
                } else {
                    1.0
                };
                let disc_extent = c.radius * n_perp_sq.sqrt();
                sep_arr[i] = min_proj - disc_extent - plane.d;
            }
            if f32x8::new(sep_arr).simd_le(zero).any() {
                return true;
            }
        }

        remainder.iter().any(|c| plane.collides(c))
    }
}
