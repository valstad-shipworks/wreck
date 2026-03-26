use std::fmt::Debug;

use glam::Vec3;
use wide::{CmpLe, f32x8};

use crate::{Bounded, Collides, Scalable, Sphere, Transformable};

/// Structure-of-Arrays storage for spheres.
///
/// Stores x, y, z, r in separate contiguous arrays, padded to a multiple of 16
/// so SIMD loops never need a scalar remainder path (works for both 8-wide and 16-wide).
/// Padding slots use `r = -1.0` which guarantees `dist_sq <= (self.r + (-1))^2`
/// is false for any reasonable query radius, preventing false positives.
#[derive(Debug, Clone)]
pub struct SpheresSoA {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub r: Vec<f32>,
    len: usize,
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
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            r: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        let padded = pad(cap);
        Self {
            x: Vec::with_capacity(padded),
            y: Vec::with_capacity(padded),
            z: Vec::with_capacity(padded),
            r: Vec::with_capacity(padded),
            len: 0,
        }
    }

    pub fn from_slice(spheres: &[Sphere]) -> Self {
        let len = spheres.len();
        let padded = pad(len);
        let mut x = Vec::with_capacity(padded);
        let mut y = Vec::with_capacity(padded);
        let mut z = Vec::with_capacity(padded);
        let mut r = Vec::with_capacity(padded);

        for s in spheres {
            x.push(s.center.x);
            y.push(s.center.y);
            z.push(s.center.z);
            r.push(s.radius);
        }

        for _ in len..padded {
            x.push(0.0);
            y.push(0.0);
            z.push(0.0);
            r.push(-1.0);
        }

        Self { x, y, z, r, len }
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
        if self.len == self.x.len() {
            self.x.extend_from_slice(&[0.0; PAD]);
            self.y.extend_from_slice(&[0.0; PAD]);
            self.z.extend_from_slice(&[0.0; PAD]);
            self.r.extend_from_slice(&[-1.0; PAD]);
        }
        self.x[self.len] = sphere.center.x;
        self.y[self.len] = sphere.center.y;
        self.z[self.len] = sphere.center.z;
        self.r[self.len] = sphere.radius;
        self.len += 1;
    }

    /// Moves all real entries from `other` into `self`, re-padding once.
    pub fn append(&mut self, other: &mut Self) {
        if other.len == 0 {
            return;
        }
        // Strip self's padding back to real length
        self.x.truncate(self.len);
        self.y.truncate(self.len);
        self.z.truncate(self.len);
        self.r.truncate(self.len);
        // Move real entries from other
        self.x.extend_from_slice(&other.x[..other.len]);
        self.y.extend_from_slice(&other.y[..other.len]);
        self.z.extend_from_slice(&other.z[..other.len]);
        self.r.extend_from_slice(&other.r[..other.len]);
        self.len += other.len;
        // Re-pad to SIMD boundary
        let padded = pad(self.len);
        self.x.resize(padded, 0.0);
        self.y.resize(padded, 0.0);
        self.z.resize(padded, 0.0);
        self.r.resize(padded, -1.0);
        // Clear other
        other.clear();
    }

    pub fn clear(&mut self) {
        for i in 0..self.len {
            self.r[i] = -1.0;
        }
        self.len = 0;
    }

    #[inline]
    pub fn get(&self, index: usize) -> Sphere {
        debug_assert!(index < self.len);
        Sphere::new(
            Vec3::new(self.x[index], self.y[index], self.z[index]),
            self.r[index],
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = Sphere> + '_ {
        (0..self.len).map(|i| self.get(i))
    }

    #[inline]
    fn chunk_count_8(&self) -> usize {
        self.x.len() / 8
    }

    /// Test if any sphere in this SoA collides with the given sphere.
    #[inline]
    pub fn any_collides_sphere(&self, sphere: &Sphere) -> bool {
        if self.is_empty() {
            return false;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { avx512::any_collides_sphere_avx512(self, sphere) };
            }
        }

        self.any_collides_sphere_f32x8(sphere)
    }

    fn any_collides_sphere_f32x8(&self, sphere: &Sphere) -> bool {
        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let sr = f32x8::splat(sphere.radius);

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let ox = f32x8::new(self.x[base..base + 8].try_into().unwrap());
            let oy = f32x8::new(self.y[base..base + 8].try_into().unwrap());
            let oz = f32x8::new(self.z[base..base + 8].try_into().unwrap());
            let or = f32x8::new(self.r[base..base + 8].try_into().unwrap());

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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { avx512::broadphase_collect_avx512(self, query, out) };
            }
        }

        self.broadphase_collect_f32x8(query, out)
    }

    fn broadphase_collect_f32x8(&self, query: &Sphere, out: &mut [bool]) -> bool {
        let cx = f32x8::splat(query.center.x);
        let cy = f32x8::splat(query.center.y);
        let cz = f32x8::splat(query.center.z);
        let sr = f32x8::splat(query.radius);
        let mut any_hit = false;

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let ox = f32x8::new(self.x[base..base + 8].try_into().unwrap());
            let oy = f32x8::new(self.y[base..base + 8].try_into().unwrap());
            let oz = f32x8::new(self.z[base..base + 8].try_into().unwrap());
            let or = f32x8::new(self.r[base..base + 8].try_into().unwrap());

            let dx = cx - ox;
            let dy = cy - oy;
            let dz = cz - oz;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let rs = sr + or;
            let mask = dist_sq.simd_le(rs * rs).to_bitmask();
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { avx512::any_collides_soa_avx512(self, other) };
            }
        }

        self.any_collides_soa_f32x8(other)
    }

    fn any_collides_soa_f32x8(&self, other: &SpheresSoA) -> bool {
        let other_chunks = other.chunk_count_8();

        for i in 0..self.len {
            let cx = f32x8::splat(self.x[i]);
            let cy = f32x8::splat(self.y[i]);
            let cz = f32x8::splat(self.z[i]);
            let sr = f32x8::splat(self.r[i]);

            for j in 0..other_chunks {
                let base = j * 8;
                let ox = f32x8::new(other.x[base..base + 8].try_into().unwrap());
                let oy = f32x8::new(other.y[base..base + 8].try_into().unwrap());
                let oz = f32x8::new(other.z[base..base + 8].try_into().unwrap());
                let or = f32x8::new(other.r[base..base + 8].try_into().unwrap());

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
    /// Translate all spheres by an offset.
    fn translate(&mut self, offset: Vec3) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { avx512::translate_avx512(self, offset) };
                return;
            }
        }

        self.translate_f32x8(offset);
    }

    /// Apply a 3x3 rotation matrix to all sphere centers.
    fn rotate_mat(&mut self, mat: glam::Mat3) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { avx512::rotate_mat_avx512(self, mat) };
                return;
            }
        }

        self.rotate_mat_f32x8(mat);
    }

    fn rotate_quat(&mut self, quat: glam::Quat) {
        self.rotate_mat(glam::Mat3::from_quat(quat));
    }

    /// Apply an affine transform (rotation + translation) to all sphere centers.
    fn transform(&mut self, mat: glam::Affine3) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { avx512::transform_avx512(self, mat) };
                return;
            }
        }

        self.transform_f32x8(mat);
    }
}

impl SpheresSoA {
    fn translate_f32x8(&mut self, offset: Vec3) {
        let ox = f32x8::splat(offset.x);
        let oy = f32x8::splat(offset.y);
        let oz = f32x8::splat(offset.z);

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let x: [f32; 8] = self.x[base..base + 8].try_into().unwrap();
            let y: [f32; 8] = self.y[base..base + 8].try_into().unwrap();
            let z: [f32; 8] = self.z[base..base + 8].try_into().unwrap();
            let rx = f32x8::new(x) + ox;
            let ry = f32x8::new(y) + oy;
            let rz = f32x8::new(z) + oz;
            self.x[base..base + 8].copy_from_slice(rx.as_array());
            self.y[base..base + 8].copy_from_slice(ry.as_array());
            self.z[base..base + 8].copy_from_slice(rz.as_array());
        }
    }

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

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let x = f32x8::new(self.x[base..base + 8].try_into().unwrap());
            let y = f32x8::new(self.y[base..base + 8].try_into().unwrap());
            let z = f32x8::new(self.z[base..base + 8].try_into().unwrap());

            let nx = m00 * x + m01 * y + m02 * z;
            let ny = m10 * x + m11 * y + m12 * z;
            let nz = m20 * x + m21 * y + m22 * z;

            self.x[base..base + 8].copy_from_slice(nx.as_array());
            self.y[base..base + 8].copy_from_slice(ny.as_array());
            self.z[base..base + 8].copy_from_slice(nz.as_array());
        }
    }

    fn transform_f32x8(&mut self, mat: glam::Affine3) {
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

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let x = f32x8::new(self.x[base..base + 8].try_into().unwrap());
            let y = f32x8::new(self.y[base..base + 8].try_into().unwrap());
            let z = f32x8::new(self.z[base..base + 8].try_into().unwrap());

            let nx = m00 * x + m01 * y + m02 * z + tx;
            let ny = m10 * x + m11 * y + m12 * z + ty;
            let nz = m20 * x + m21 * y + m22 * z + tz;

            self.x[base..base + 8].copy_from_slice(nx.as_array());
            self.y[base..base + 8].copy_from_slice(ny.as_array());
            self.z[base..base + 8].copy_from_slice(nz.as_array());
        }
    }
}

impl Scalable for SpheresSoA {
    /// Scale all radii by a factor.
    fn scale(&mut self, factor: f32) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { avx512::scale_avx512(self, factor) };
                return;
            }
        }

        self.scale_f32x8(factor);
    }
}

impl SpheresSoA {
    fn scale_f32x8(&mut self, factor: f32) {
        let f = f32x8::splat(factor);

        for i in 0..self.chunk_count_8() {
            let base = i * 8;
            let r: [f32; 8] = self.r[base..base + 8].try_into().unwrap();
            let scaled = f32x8::new(r) * f;
            self.r[base..base + 8].copy_from_slice(scaled.as_array());
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 implementations (16 floats per register)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx512 {
    use super::SpheresSoA;
    use crate::Sphere;
    use glam::Vec3;
    use std::arch::x86_64::*;

    #[inline]
    fn chunk_count_16(soa: &SpheresSoA) -> usize {
        soa.x.len() / 16
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn any_collides_sphere_avx512(soa: &SpheresSoA, sphere: &Sphere) -> bool {
        let cx = _mm512_set1_ps(sphere.center.x);
        let cy = _mm512_set1_ps(sphere.center.y);
        let cz = _mm512_set1_ps(sphere.center.z);
        let sr = _mm512_set1_ps(sphere.radius);

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let ox = _mm512_loadu_ps(soa.x[base..].as_ptr());
            let oy = _mm512_loadu_ps(soa.y[base..].as_ptr());
            let oz = _mm512_loadu_ps(soa.z[base..].as_ptr());
            let or = _mm512_loadu_ps(soa.r[base..].as_ptr());

            let dx = _mm512_sub_ps(cx, ox);
            let dy = _mm512_sub_ps(cy, oy);
            let dz = _mm512_sub_ps(cz, oz);

            let dist_sq = _mm512_fmadd_ps(dz, dz, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dx, dx)));

            let rs = _mm512_add_ps(sr, or);
            let rs_sq = _mm512_mul_ps(rs, rs);

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

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let ox = _mm512_loadu_ps(soa.x[base..].as_ptr());
            let oy = _mm512_loadu_ps(soa.y[base..].as_ptr());
            let oz = _mm512_loadu_ps(soa.z[base..].as_ptr());
            let or = _mm512_loadu_ps(soa.r[base..].as_ptr());

            let dx = _mm512_sub_ps(cx, ox);
            let dy = _mm512_sub_ps(cy, oy);
            let dz = _mm512_sub_ps(cz, oz);

            let dist_sq = _mm512_fmadd_ps(dz, dz, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dx, dx)));

            let rs = _mm512_add_ps(sr, or);
            let rs_sq = _mm512_mul_ps(rs, rs);

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

        for i in 0..a.len {
            let cx = _mm512_set1_ps(a.x[i]);
            let cy = _mm512_set1_ps(a.y[i]);
            let cz = _mm512_set1_ps(a.z[i]);
            let sr = _mm512_set1_ps(a.r[i]);

            for j in 0..b_chunks {
                let base = j * 16;
                let ox = _mm512_loadu_ps(b.x[base..].as_ptr());
                let oy = _mm512_loadu_ps(b.y[base..].as_ptr());
                let oz = _mm512_loadu_ps(b.z[base..].as_ptr());
                let or = _mm512_loadu_ps(b.r[base..].as_ptr());

                let dx = _mm512_sub_ps(cx, ox);
                let dy = _mm512_sub_ps(cy, oy);
                let dz = _mm512_sub_ps(cz, oz);

                let dist_sq =
                    _mm512_fmadd_ps(dz, dz, _mm512_fmadd_ps(dy, dy, _mm512_mul_ps(dx, dx)));

                let rs = _mm512_add_ps(sr, or);
                let rs_sq = _mm512_mul_ps(rs, rs);

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

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let xp = soa.x[base..].as_mut_ptr();
            let yp = soa.y[base..].as_mut_ptr();
            let zp = soa.z[base..].as_mut_ptr();
            _mm512_storeu_ps(xp, _mm512_add_ps(_mm512_loadu_ps(xp), ox));
            _mm512_storeu_ps(yp, _mm512_add_ps(_mm512_loadu_ps(yp), oy));
            _mm512_storeu_ps(zp, _mm512_add_ps(_mm512_loadu_ps(zp), oz));
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn scale_avx512(soa: &mut SpheresSoA, factor: f32) {
        let f = _mm512_set1_ps(factor);

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let rp = soa.r[base..].as_mut_ptr();
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

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let xp = soa.x[base..].as_mut_ptr();
            let yp = soa.y[base..].as_mut_ptr();
            let zp = soa.z[base..].as_mut_ptr();
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
    pub(super) unsafe fn transform_avx512(soa: &mut SpheresSoA, mat: glam::Affine3) {
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

        for i in 0..chunk_count_16(soa) {
            let base = i * 16;
            let xp = soa.x[base..].as_mut_ptr();
            let yp = soa.y[base..].as_mut_ptr();
            let zp = soa.z[base..].as_mut_ptr();
            let x = _mm512_loadu_ps(xp);
            let y = _mm512_loadu_ps(yp);
            let z = _mm512_loadu_ps(zp);

            let nx = _mm512_add_ps(
                _mm512_fmadd_ps(m02, z, _mm512_fmadd_ps(m01, y, _mm512_mul_ps(m00, x))),
                tx,
            );
            let ny = _mm512_add_ps(
                _mm512_fmadd_ps(m12, z, _mm512_fmadd_ps(m11, y, _mm512_mul_ps(m10, x))),
                ty,
            );
            let nz = _mm512_add_ps(
                _mm512_fmadd_ps(m22, z, _mm512_fmadd_ps(m21, y, _mm512_mul_ps(m20, x))),
                tz,
            );

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
        let broad = SpheresSoA::from_slice(
            &items
                .iter()
                .map(|item| item.broadphase())
                .collect::<Vec<_>>(),
        );
        Self { items, broad }
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
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.items.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.items.iter_mut()
    }
}

impl<'a, T> IntoIterator for &'a BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut BroadCollection<T>
where
    T: Bounded + Transformable + Scalable + Debug + Clone + Sized,
{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
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
    fn translate(&mut self, offset: glam::Vec3) {
        for item in &mut self.items {
            item.translate(offset);
        }
        self.broad.translate(offset);
    }

    fn rotate_mat(&mut self, mat: glam::Mat3) {
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

    fn transform(&mut self, mat: glam::Affine3) {
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

    use crate::Collides;
    use crate::capsule::Capsule;
    use crate::cuboid::Cuboid;
    use crate::cylinder::Cylinder;
    use crate::line::{Line, LineSegment, Ray};
    use crate::plane::Plane;
    use crate::sphere::Sphere;

    // ── Sphere vs Capsule ────────────────────────────────────────────────

    pub fn sphere_vs_capsules(sphere: &Sphere, others: &[Capsule]) -> bool {
        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let sr = f32x8::splat(sphere.radius);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
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

    // ── Sphere vs Cuboid ─────────────────────────────────────────────────

    pub fn sphere_vs_cuboids(sphere: &Sphere, others: &[Cuboid]) -> bool {
        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let r_sq = f32x8::splat(sphere.radius * sphere.radius);
        let zero = f32x8::splat(0.0);

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
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

    // ── Plane vs Sphere ──────────────────────────────────────────────────

    pub fn plane_vs_spheres(plane: &Plane, others: &[Sphere]) -> bool {
        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut cx = [0.0f32; 8];
            let mut cy = [0.0f32; 8];
            let mut cz = [0.0f32; 8];
            let mut r = [0.0f32; 8];
            for (i, s) in chunk.iter().enumerate() {
                cx[i] = s.center.x;
                cy[i] = s.center.y;
                cz[i] = s.center.z;
                r[i] = s.radius;
            }
            let proj = nx * f32x8::new(cx) + ny * f32x8::new(cy) + nz * f32x8::new(cz);
            let sep = proj - d - f32x8::new(r);
            if sep.simd_le(zero).any() {
                return true;
            }
        }

        remainder.iter().any(|s| plane.collides(s))
    }

    // ── Plane vs Capsule ─────────────────────────────────────────────────

    pub fn plane_vs_capsules(plane: &Plane, others: &[Capsule]) -> bool {
        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
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

    // ── Plane vs Cuboid ──────────────────────────────────────────────────

    pub fn plane_vs_cuboids(plane: &Plane, others: &[Cuboid]) -> bool {
        let nx = f32x8::splat(plane.normal.x);
        let ny = f32x8::splat(plane.normal.y);
        let nz = f32x8::splat(plane.normal.z);
        let d = f32x8::splat(plane.d);
        let zero = f32x8::ZERO;

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
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

    // ── Sphere vs Cylinder ────────────────────────────────────────────────

    pub fn sphere_vs_cylinders(sphere: &Sphere, others: &[Cylinder]) -> bool {
        let cx = f32x8::splat(sphere.center.x);
        let cy = f32x8::splat(sphere.center.y);
        let cz = f32x8::splat(sphere.center.z);
        let sr_sq = f32x8::splat(sphere.radius * sphere.radius);
        let sr = f32x8::splat(sphere.radius);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);
        let four = f32x8::splat(4.0);

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
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

            // Perpendicular distance (using unclamped t)
            let perpx = wx - dx * t;
            let perpy = wy - dy * t;
            let perpz = wz - dz * t;
            let r_sq = perpx * perpx + perpy * perpy + perpz * perpz;

            // Barrel: t in [0,1], r_sq <= (cyl_r + sphere_r)^2
            let in_barrel = zero.simd_le(t) & t.simd_le(one);
            let combined = cr + sr;
            let barrel_hit = in_barrel & r_sq.simd_le(combined * combined);

            // End cap: axial and radial distance
            let t_excess = t - t_c;
            let d_axial_sq = t_excess * t_excess * dir_sq;
            let cr_sq = cr * cr;

            // End cap, radially inside
            let inside_r = r_sq.simd_le(cr_sq);
            let endcap_inside = inside_r & d_axial_sq.simd_le(sr_sq);

            // End cap, radially outside: sqrt-free
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

    // ── Plane vs Cylinder ───────────────────────────────────────────────

    pub fn plane_vs_cylinders(plane: &Plane, others: &[Cylinder]) -> bool {
        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();
        let zero = f32x8::ZERO;

        for chunk in chunks {
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

    // ── Line/Ray/Segment vs Sphere ───────────────────────────────────────

    fn line_vs_spheres_inner(
        origin: Vec3,
        dir: Vec3,
        rdv: f32,
        t_min: f32,
        t_max: f32,
        others: &[Sphere],
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

        let chunks = others.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut cx = [0.0f32; 8];
            let mut cy = [0.0f32; 8];
            let mut cz = [0.0f32; 8];
            let mut r = [0.0f32; 8];
            for (i, s) in chunk.iter().enumerate() {
                cx[i] = s.center.x;
                cy[i] = s.center.y;
                cz[i] = s.center.z;
                r[i] = s.radius;
            }
            let cx = f32x8::new(cx);
            let cy = f32x8::new(cy);
            let cz = f32x8::new(cz);
            let r = f32x8::new(r);

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

        for s in remainder {
            let t = crate::line::closest_t_to_point(origin, dir, rdv, s.center, t_min, t_max);
            let closest = origin + dir * t;
            let d = closest - s.center;
            if d.dot(d) <= s.radius * s.radius {
                return true;
            }
        }
        false
    }

    pub fn line_vs_spheres(line: &Line, others: &[Sphere]) -> bool {
        line_vs_spheres_inner(
            line.origin,
            line.dir,
            line.rdv,
            f32::NEG_INFINITY,
            f32::INFINITY,
            others,
        )
    }

    pub fn ray_vs_spheres(ray: &Ray, others: &[Sphere]) -> bool {
        line_vs_spheres_inner(ray.origin, ray.dir, ray.rdv, 0.0, f32::INFINITY, others)
    }

    pub fn segment_vs_spheres(seg: &LineSegment, others: &[Sphere]) -> bool {
        line_vs_spheres_inner(seg.p1, seg.dir, seg.rdv, 0.0, 1.0, others)
    }
}
