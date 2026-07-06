// Column scatter/gather indexes a strided buffer and a row in lockstep — the index is intrinsic.
#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use glam::Vec3;

use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::cylinder::Cylinder;

/// Widest column count of any [`SoaShape`] (`Cuboid` = 15). Sizes the stack buffer used to
/// scatter/gather one shape's row, so `to_row`/`from_row` never allocate.
pub(crate) const MAX_SHAPE_COLS: usize = 16;

/// A shape representable as a fixed number of `f32` columns, for columnar (structure-of-arrays)
/// storage in a [`Collider`](crate::Collider).
///
/// Storing shapes column-wise lets the SIMD collision kernels read each field as a contiguous
/// `&[f32]` and feed it straight to `hydroplane`'s `any_n` — no per-chunk AoS gather. The trade-off
/// is that reconstructing a whole shape ([`ShapeSoa::get`]/[`ShapeSoa::iter`]) reads one value from
/// every column, so random per-shape access is slower than an `&[Shape]`. For collision batches —
/// where the access is always columnar — this is the ideal layout.
///
/// Derived/cached fields (a capsule's `rdv`, a cuboid's `axis_aligned`, …) need not occupy a column
/// when `from_row` can recompute them through the shape's constructor.
pub trait SoaShape: Copy {
    /// Number of `f32` columns; equals the length of the slice passed to [`to_row`](Self::to_row)
    /// and [`from_row`](Self::from_row). Must not exceed [`MAX_SHAPE_COLS`].
    const COLS: usize;

    /// Write this shape's field values into `row` (`row.len() == COLS`).
    fn to_row(&self, row: &mut [f32]);

    /// Reconstruct a shape from a row of `COLS` field values written by [`to_row`](Self::to_row).
    fn from_row(row: &[f32]) -> Self;
}

/// Columnar storage for a [`SoaShape`]: one contiguous buffer laid out as
/// `[col0; cap][col1; cap] … [col(COLS-1); cap]`, each logical column exactly `len` long.
///
/// Unlike [`SpheresSoA`](crate::soa::SpheresSoA) there is no SIMD padding — the collision kernels
/// iterate the real length via `hydroplane`'s `chunks`/`any_n`, which mask the final partial
/// register, so a column is stored at exactly `len` with spare capacity `cap - len` for growth.
pub struct ShapeSoa<S> {
    buf: Vec<f32>,
    cap: usize,
    len: usize,
    _marker: PhantomData<S>,
}

impl<S: SoaShape> ShapeSoa<S> {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            cap: 0,
            len: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: vec![0.0; S::COLS * cap],
            cap,
            len: 0,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Column `c` (`0..COLS`) as a contiguous `&[f32]` of length `len` — the slice the SIMD
    /// kernels load from.
    #[inline]
    pub fn col(&self, c: usize) -> &[f32] {
        debug_assert!(c < S::COLS);
        &self.buf[c * self.cap..c * self.cap + self.len]
    }

    pub fn push(&mut self, shape: &S) {
        if self.len == self.cap {
            self.grow();
        }
        let mut row = [0.0f32; MAX_SHAPE_COLS];
        shape.to_row(&mut row[..S::COLS]);
        for c in 0..S::COLS {
            self.buf[c * self.cap + self.len] = row[c];
        }
        self.len += 1;
    }

    #[inline]
    pub fn get(&self, i: usize) -> S {
        debug_assert!(i < self.len);
        let mut row = [0.0f32; MAX_SHAPE_COLS];
        for c in 0..S::COLS {
            row[c] = self.buf[c * self.cap + i];
        }
        S::from_row(&row[..S::COLS])
    }

    pub fn iter(&self) -> impl Iterator<Item = S> + '_ {
        (0..self.len).map(|i| self.get(i))
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append every shape from `other`, leaving `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        for i in 0..other.len {
            let s = other.get(i);
            self.push(&s);
        }
        other.clear();
    }

    /// Push every shape yielded by `iter`.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = S>) {
        for s in iter {
            self.push(&s);
        }
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 8 } else { self.cap * 2 };
        let mut buf = vec![0.0f32; S::COLS * new_cap];
        for c in 0..S::COLS {
            let src = c * self.cap;
            let dst = c * new_cap;
            buf[dst..dst + self.len].copy_from_slice(&self.buf[src..src + self.len]);
        }
        self.buf = buf;
        self.cap = new_cap;
    }
}

impl<S: SoaShape> Default for ShapeSoa<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: SoaShape> Clone for ShapeSoa<S> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            cap: self.cap,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

impl<S: SoaShape + core::fmt::Debug> core::fmt::Debug for ShapeSoa<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<S: SoaShape + PartialEq> PartialEq for ShapeSoa<S> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<S: SoaShape> FromIterator<S> for ShapeSoa<S> {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let it = iter.into_iter();
        let mut soa = Self::with_capacity(it.size_hint().0);
        for s in it {
            soa.push(&s);
        }
        soa
    }
}

#[cfg(feature = "serde")]
impl<S: SoaShape + serde::Serialize> serde::Serialize for ShapeSoa<S> {
    fn serialize<Z: serde::Serializer>(&self, serializer: Z) -> Result<Z::Ok, Z::Error> {
        serializer.collect_seq(self.iter())
    }
}

#[cfg(feature = "serde")]
impl<'de, S: SoaShape + serde::Deserialize<'de>> serde::Deserialize<'de> for ShapeSoa<S> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Vec::<S>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

// ── Per-shape column layouts ─────────────────────────────────────────────────
// Derived fields (rdv, z_aligned, axis_aligned) are recomputed by the constructor in `from_row`
// rather than stored — except a capsule/cylinder `rdv`, which gets its own column because the
// narrowphase kernels read it per lane.

impl SoaShape for Capsule {
    const COLS: usize = 8;

    fn to_row(&self, r: &mut [f32]) {
        r[0] = self.p1.x;
        r[1] = self.p1.y;
        r[2] = self.p1.z;
        r[3] = self.dir.x;
        r[4] = self.dir.y;
        r[5] = self.dir.z;
        r[6] = self.radius;
        r[7] = self.rdv;
    }

    fn from_row(r: &[f32]) -> Self {
        let dir = Vec3::new(r[3], r[4], r[5]);
        Capsule {
            p1: Vec3::new(r[0], r[1], r[2]),
            dir,
            radius: r[6],
            rdv: r[7],
            z_aligned: dir.x == 0.0 && dir.y == 0.0,
        }
    }
}

impl SoaShape for Cylinder {
    const COLS: usize = 8;

    fn to_row(&self, r: &mut [f32]) {
        r[0] = self.p1.x;
        r[1] = self.p1.y;
        r[2] = self.p1.z;
        r[3] = self.dir.x;
        r[4] = self.dir.y;
        r[5] = self.dir.z;
        r[6] = self.radius;
        r[7] = self.rdv;
    }

    fn from_row(r: &[f32]) -> Self {
        let dir = Vec3::new(r[3], r[4], r[5]);
        Cylinder {
            p1: Vec3::new(r[0], r[1], r[2]),
            dir,
            radius: r[6],
            rdv: r[7],
            z_aligned: dir.x == 0.0 && dir.y == 0.0,
        }
    }
}

impl SoaShape for Cuboid {
    const COLS: usize = 15;

    fn to_row(&self, r: &mut [f32]) {
        r[0] = self.center.x;
        r[1] = self.center.y;
        r[2] = self.center.z;
        for a in 0..3 {
            r[3 + a * 3] = self.axes[a].x;
            r[4 + a * 3] = self.axes[a].y;
            r[5 + a * 3] = self.axes[a].z;
        }
        r[12] = self.half_extents[0];
        r[13] = self.half_extents[1];
        r[14] = self.half_extents[2];
    }

    fn from_row(r: &[f32]) -> Self {
        Cuboid::new(
            Vec3::new(r[0], r[1], r[2]),
            [
                Vec3::new(r[3], r[4], r[5]),
                Vec3::new(r[6], r[7], r[8]),
                Vec3::new(r[9], r[10], r[11]),
            ],
            [r[12], r[13], r[14]],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(i: f32) -> Capsule {
        Capsule::new(Vec3::new(i, i + 1.0, i + 2.0), Vec3::new(i + 3.0, i, i + 1.0), 0.5 + i * 0.1)
    }

    #[test]
    fn capsule_roundtrip_and_columns() {
        let shapes: Vec<Capsule> = (0..20).map(|i| cap(i as f32)).collect();
        let soa: ShapeSoa<Capsule> = shapes.iter().copied().collect();
        assert_eq!(soa.len(), 20);
        // get/iter reconstruct each shape exactly (fields set directly, no lossy recompute)
        for (i, want) in shapes.iter().enumerate() {
            assert_eq!(soa.get(i), *want);
        }
        assert!(soa.iter().eq(shapes.iter().copied()));
        // columns expose the raw fields
        for (i, want) in shapes.iter().enumerate() {
            assert_eq!(soa.col(0)[i], want.p1.x);
            assert_eq!(soa.col(5)[i], want.dir.z);
            assert_eq!(soa.col(6)[i], want.radius);
            assert_eq!(soa.col(7)[i], want.rdv);
        }
    }

    #[test]
    fn cuboid_roundtrip() {
        let c = Cuboid::new(
            Vec3::new(1.0, 2.0, 3.0),
            [Vec3::X, Vec3::Y, Vec3::Z],
            [0.5, 1.0, 1.5],
        );
        let mut soa: ShapeSoa<Cuboid> = ShapeSoa::new();
        soa.push(&c);
        assert_eq!(soa.get(0), c);
        assert_eq!(soa.col(12)[0], 0.5);
    }
}
