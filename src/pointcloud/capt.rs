//! # Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Collision Checking
//!
//! This is copied from https://github.com/oh-yes-0-fps/capt/tree/non-nightly
//!
//! This is a Rust implementation of the _collision-affording point tree_ (CAPT), a data structure
//! for SIMD-parallel collision-checking between spheres and point clouds.
//!
//! You may also want to look at the following other sources:
//!
//! - [The paper](https://arxiv.org/abs/2406.02807)
//! - [C++ implementation](https://github.com/KavrakiLab/vamp)
//! - [Blog post about it](https://www.claytonwramsey.com/blog/captree)
//! - [Demo video](https://youtu.be/BzDKdrU1VpM)
//!
//! If you use this in an academic work, please cite it as follows:
//!
//! ```bibtex
//! @InProceedings{capt,
//!   title = {Collision-Affording Point Trees: {SIMD}-Amenable Nearest Neighbors for Fast Collision Checking},
//!   author = {Ramsey, Clayton W. and Kingston, Zachary and Thomason, Wil and Kavraki, Lydia E.},
//!   booktitle = {Robotics: Science and Systems},
//!   date = {2024},
//!   url = {http://arxiv.org/abs/2406.02807},
//!   note = {To Appear.}
//! }
//! ```
//!
//! ## Usage
//!
//! The core data structure in this library is the [`Capt`], which is a search tree used for
//! collision checking. [`Capt`]s are polymorphic over dimension and data type. On construction,
//! they take in a list of points in a point cloud and a _radius range_: a tuple of the minimum and
//! maximum radius used for querying.
//!
//! ```ignore
//! use capt::Capt;
//!
//! // list of points in cloud
//! let points = [[0.0, 1.1], [0.2, 3.1]];
//! let r_min = 0.05;
//! let r_max = 2.0;
//!
//! let capt = Capt::<2>::new(&points, (r_min, r_max), 1);
//! ```
//!
//! Once you have a `Capt`, you can use it for collision-checking against spheres.
//! Correct answers are only guaranteed if you collision-check against spheres with a radius inside
//! the radius range.
//!
//! ```ignore
//! # use capt::Capt;
//! # let points = [[0.0, 1.1], [0.2, 3.1]];
//! # let capt = Capt::<2>::new(&points, (0.05, 2.0), 1);
//! let center = [0.0, 0.0]; // center of sphere
//! let radius0 = 1.0; // radius of sphere
//! assert!(!capt.collides(&center, radius0));
//!
//! let radius1 = 1.5;
//! assert!(capt.collides(&center, radius1));
//! ```
//!
//! ## Querying
//!
//! [`Capt::query_simd`] is the f32 collision checker: a scalar tree descent followed by a
//! `hydroplane` SIMD scan of the leaf's affordance buffer.
//!
//! ## License
//!
//! This work is licensed to you under the Apache 2.0 license.
#![warn(clippy::pedantic, clippy::cargo, clippy::nursery, missing_docs)]

use aligned_vec::{ABox, AVec, RuntimeAlign};
use alloc::{boxed::Box, vec, vec::Vec};

use core::{
    alloc::Layout,
    array,
    cmp::max,
    fmt::Debug,
    ops::{Add, Sub},
};

use glam::Vec3;
use hydroplane::{Gang, GangGlamExt, Vec3Wide, kernel};

/// A generic trait representing values which may be used as an "axis;" that is, elements of a
/// vector representing a point.
///
/// An array of `Axis` values is a point which can be stored in a [`Capt`].
/// Accordingly, this trait specifies nearly all the requirements for points that [`Capt`]s require.
/// The only exception is that [`Axis`] values really ought to be [`Ord`] instead of [`PartialOrd`];
/// however, due to the disaster that is IEE 754 floating point numbers, `f32` and `f64` are not
/// totally ordered. As a compromise, we relax the `Ord` requirement so that you can use floats in a
/// `Capt`.
///
/// # Examples
///
/// ```ignore
/// #[derive(Clone, Copy, PartialOrd, PartialEq)]
/// enum HyperInt {
///     MinusInf,
///     Real(i32),
///     PlusInf,
/// }
///
/// impl std::ops::Add for HyperInt {
/// // ...
/// #    type Output = Self;
/// #
/// #    fn add(self, rhs: Self) -> Self {
/// #        match (self, rhs) {
/// #            (Self::MinusInf, Self::PlusInf) => Self::Real(0), // evil, but who cares?
/// #            (Self::MinusInf, _) | (_, Self::MinusInf) => Self::MinusInf,
/// #            (Self::PlusInf, _) | (_, Self::PlusInf) => Self::PlusInf,
/// #            (Self::Real(x), Self::Real(y)) => Self::Real(x + y),
/// #        }
/// #    }
/// }
///
///
/// impl std::ops::Sub for HyperInt {
/// // ...
/// #    type Output = Self;
/// #
/// #    fn sub(self, rhs: Self) -> Self {
/// #        match (self, rhs) {
/// #            (Self::MinusInf, Self::MinusInf) | (Self::PlusInf, Self::PlusInf) => Self::Real(0), // evil, but who cares?
/// #            (Self::MinusInf, _) | (_, Self::PlusInf) => Self::MinusInf,
/// #            (Self::PlusInf, _) | (_, Self::MinusInf) => Self::PlusInf,
/// #            (Self::Real(x), Self::Real(y)) => Self::Real(x - y),
/// #        }
/// #    }
/// }
///
/// impl capt::Axis for HyperInt {
///     const ZERO: Self = Self::Real(0);
///     const INFINITY: Self = Self::PlusInf;
///     const NEG_INFINITY: Self = Self::MinusInf;
///
///     fn is_finite(self) -> bool {
///         matches!(self, Self::Real(_))
///     }
///
///     fn in_between(self, rhs: Self) -> Self {
///         match (self, rhs) {
///             (Self::PlusInf, Self::MinusInf) | (Self::MinusInf, Self::PlusInf) => Self::Real(0),
///             (Self::MinusInf, _) | (_, Self::MinusInf) => Self::MinusInf,
///             (Self::PlusInf, _) | (_, Self::PlusInf) => Self::PlusInf,
///             (Self::Real(a), Self::Real(b)) => Self::Real((a + b) / 2)
///         }
///     }
///
///     fn square(self) -> Self {
///         match self {
///             Self::PlusInf | Self::MinusInf => Self::PlusInf,
///             Self::Real(a) => Self::Real(a * a),
///         }
///     }
/// }
/// ```
pub trait Axis: PartialOrd + Copy + Sub<Output = Self> + Add<Output = Self> {
    /// A zero value.
    const ZERO: Self;
    /// A value which is larger than any finite value.
    const INFINITY: Self;
    /// A value which is smaller than any finite value.
    const NEG_INFINITY: Self;

    #[must_use]
    /// Determine whether this value is finite or infinite.
    fn is_finite(self) -> bool;

    #[must_use]
    /// Compute a value of `Self` which is halfway between `self` and `rhs`.
    /// If there are no legal values between `self` and `rhs`, it is acceptable to return `self`
    /// instead.
    fn in_between(self, rhs: Self) -> Self;

    #[must_use]
    /// Compute the square of this value.
    fn square(self) -> Self;
}

/// An index type used for lookups into and out of arrays.
///
/// This is implemented so that [`Capt`]s can use smaller index sizes (such as [`u32`] or [`u16`])
/// for improved memory performance.
pub trait Index: TryFrom<usize> + TryInto<usize> + Copy {
    /// The zero index. This must be equal to `(0usize).try_into().unwrap()`.
    const ZERO: Self;
}

macro_rules! impl_axis {
    ($t: ty, $tm: ty) => {
        impl Axis for $t {
            const ZERO: Self = 0.0;
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }

            fn in_between(self, rhs: Self) -> Self {
                (self + rhs) / 2.0
            }

            fn square(self) -> Self {
                self * self
            }
        }
    };
}

macro_rules! impl_idx {
    ($t: ty) => {
        impl Index for $t {
            const ZERO: Self = 0;
        }
    };
}

impl_axis!(f32, i32);
impl_axis!(f64, i64);

impl_idx!(u8);
impl_idx!(u16);
impl_idx!(u32);
impl_idx!(u64);
impl_idx!(usize);

/// Clamp a floating-point number.
fn clamp<A: PartialOrd>(x: A, min: A, max: A) -> A {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// Squared distance from a fixed query point to the afforded points stored as the three
/// columns `xs`/`ys`/`zs`; `true` if any squared distance is `<= rsq`. A `lane < cnt` mask
/// drops the tail lanes a short final chunk leaves inactive.
#[kernel]
fn afforded3_hit<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    center: Vec3,
    rsq: f32,
) -> bool {
    let c = ctx.splat_vec3(center);
    let rs = ctx.splat(rsq);
    ctx.any_n([xs, ys, zs], |[x, y, z]| {
        (c - Vec3Wide::from([x, y, z])).length_squared().le(rs)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::module_name_repetitions)]
/// A collision-affording point tree (CAPT), which allows for efficient collision-checking in a
/// SIMD-parallel manner between spheres and point clouds.
///
/// # Generic parameters
///
/// - `K`: The dimension of the space.
/// - `L`: The lane size of this tree. Internally, this is the upper bound on the width of a SIMD
///   lane that can be used in this data structure. The alignment of this structure must be a power
///   of two.
/// - `A`: The value of the axes of each point. This should typically be `f32` or `f64`. This should
///   implement [`Axis`].
/// - `I`: The index integer. This should generally be an unsigned integer, such as `usize` or
///   `u32`. This should implement [`Index`].
///
/// # Examples
///
/// ```ignore
/// // list of points in cloud
/// let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
///
/// // query radii must be between 0.0 and 0.2
/// let t = capt::Capt::<2>::new(&points, (0.0, 0.2), 1);
///
/// assert!(!t.collides(&[0.0, 0.3], 0.1));
/// assert!(t.collides(&[0.0, 0.2], 0.15));
/// ```
pub struct Capt<const K: usize, A = f32, I = usize> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than
    /// `tests[idx]`, we advance to `2 * idx + 1`; otherwise, we go to `2 * idx + 2`.
    ///
    /// The length of `tests` must be `N`, rounded up to the next power of 2, minus one.
    tests: Box<[A]>,
    /// Axis-aligned bounding boxes containing the set of afforded points for each cell.
    aabbs: Box<[Aabb<A, K>]>,
    /// Indexes for the starts of the affordance buffer subsequence of `points` corresponding to
    /// each leaf cell in the tree.
    /// This buffer is padded with one extra `usize` at the end with the maximum length of `points`
    /// for the sake of branchless computation.
    starts: Box<[I]>,
    /// The sets of afforded points for each cell.
    afforded: [ABox<[A], RuntimeAlign>; K],
    r_point: A,
    /// Log-base-2 of the number of lanes in this tree.
    lanes_log2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct Aabb<A, const K: usize> {
    /// The lower bound on the volume.
    pub lo: [A; K],
    /// The upper bound on the volume.
    pub hi: [A; K],
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
/// The errors which can occur when calling [`Capt::try_new`].
pub enum NewCaptError {
    /// There were too many points in the provided cloud to be represented without integer
    /// overflow.
    TooManyPoints,
    /// At least one of the points had a non-finite value.
    NonFinite,
    /// The CAPT was constructed with a non-power-of-two lane count.
    InvalidLaneCount,
}

impl<A, I, const K: usize> Capt<K, A, I>
where
    A: Axis,
    I: Index,
{
    /// Construct a new CAPT containing all the points in `points`.
    ///
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `n_lanes` is the number of SIMD lanes to be used by this tree, and may be 1 for
    /// single-element batch requests.
    ///
    /// # Panics
    ///
    /// This function will panic if there are too many points in the tree to be addressed by `I`, or
    /// if any points contain non-finite non-real value. This can even be the case if there are
    /// fewer points in `points` than can be addressed by `I` as the CAPT may duplicate points
    /// for efficiency.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let points = [[0.0]];
    ///
    /// let capt = capt::Capt::<1>::new(&points, (0.0, f32::INFINITY), 1);
    ///
    /// assert!(capt.collides(&[1.0], 1.5));
    /// assert!(!capt.collides(&[1.0], 0.5));
    /// ```
    ///
    /// If there are too many points in `points`, this could cause a panic!
    ///
    /// ```ignore
    /// let points = [[0.0]; 256];
    ///
    /// // note that we are using `u8` as our index type
    /// let capt = capt::Capt::<1, f32, u8>::new(&points, (0.0, f32::INFINITY), 1);
    /// ```
    #[allow(dead_code)]
    pub fn new(points: &[[A; K]], r_range: (A, A), n_lanes: usize) -> Self {
        Self::try_new(points, r_range, n_lanes)
            .expect("index type I must be able to support all points in CAPT during construction")
    }

    /// Construct a new CAPT containing all the points in `points` with a point radius `r_point`.
    ///
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `n_lanes` is the number of SIMD lanes to be used by this tree, and may be 1 for
    /// single-element batch requests.
    ///
    /// # Panics
    ///
    /// This function will panic if there are too many points in the tree to be addressed by `I`, or
    /// if any points contain non-finite non-real value. This can even be the case if there are
    /// fewer points in `points` than can be addressed by `I` as the CAPT may duplicate points
    /// for efficiency.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let points = [[0.0]];
    ///
    /// let capt = capt::Capt::<1>::with_point_radius(&points, (0.0, f32::INFINITY), 0.2, 1);
    ///
    /// assert!(capt.collides(&[1.0], 1.5));
    /// assert!(!capt.collides(&[1.0], 0.5));
    /// ```
    ///
    /// If there are too many points in `points`, this could cause a panic!
    ///
    /// ```ignore
    /// let points = [[0.0]; 256];
    ///
    /// // note that we are using `u8` as our index type
    /// let capt = capt::Capt::<1, f32, u8>::with_point_radius(&points, (0.0, f32::INFINITY), 0.2, 1);
    /// ```
    #[allow(dead_code)]
    pub fn with_point_radius(
        points: &[[A; K]],
        r_range: (A, A),
        r_point: A,
        n_lanes: usize,
    ) -> Self {
        Self::try_with_point_radius(points, r_range, r_point, n_lanes)
            .expect("index type I must be able to support all points in CAPT during construction")
    }

    /// Construct a new CAPT containing all the points in `points`, checking for index overflow.
    ///
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `n_lanes` is the number of SIMD lanes to be used by this tree, and may be 1 for
    /// single-element batch requests.
    ///
    /// # Errors
    ///
    /// This function will return `Err(NewCaptError::TooManyPoints)` if there are too many points to
    /// be indexed by `I`. It will return `Err(NewCaptError::NonFinite)` if any element of
    /// `points` is non-finite.
    ///
    /// # Examples
    ///
    /// Unwrapping the output from this function is equivalent to calling [`Capt::new`].
    ///
    /// ```ignore
    /// let points = [[0.0]];
    ///
    /// let capt = capt::Capt::<1>::try_new(&points, (0.0, f32::INFINITY), 1).unwrap();
    /// ```
    ///
    /// In failure, we get an `Err`.
    ///
    /// ```ignore
    /// let points = [[0.0]; 256];
    ///
    /// // note that we are using `u8` as our index type
    /// let opt = capt::Capt::<1, f32, u8>::try_new(&points, (0.0, f32::INFINITY), 8);
    ///
    /// assert!(opt.is_err());
    /// ```
    #[allow(dead_code)]
    pub fn try_new(
        points: &[[A; K]],
        r_range: (A, A),
        n_lanes: usize,
    ) -> Result<Self, NewCaptError> {
        Self::try_with_point_radius(points, r_range, A::ZERO, n_lanes)
    }

    /// Construct a new CAPT containing all the points in `points` with a point radius `r_point`,
    /// checking for index overflow.
    ///
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `n_lanes` is the number of SIMD lanes to be used by this tree, and may be 1 for
    /// single-element batch requests.
    ///
    /// # Errors
    ///
    /// This function will return `Err(NewCaptError::TooManyPoints)` if there are too many points to
    /// be indexed by `I`. It will return `Err(NewCaptError::NonFinite)` if any element of
    /// `points` is non-finite.
    ///
    /// # Examples
    ///
    /// Unwrapping the output from this function is equivalent to calling
    /// [`Capt::with_point_radius`].
    ///
    /// ```ignore
    /// let points = [[0.0]];
    ///
    /// let capt =
    ///     capt::Capt::<1>::try_with_point_radius(&points, (0.0, f32::INFINITY), 0.01, 1).unwrap();
    /// ```
    ///
    /// In failure, we get an `Err`.
    ///
    /// ```ignore
    /// let points = [[0.0]; 256];
    ///
    /// // note that we are using `u8` as our index type
    /// let opt =
    ///     capt::Capt::<1, f32, u8>::try_with_point_radius(&points, (0.0, f32::INFINITY), 0.01, 1);
    ///
    /// assert!(opt.is_err());
    /// ```
    #[allow(dead_code)]
    pub fn try_with_point_radius(
        points: &[[A; K]],
        r_range: (A, A),
        r_point: A,
        n_lanes: usize,
    ) -> Result<Self, NewCaptError> {
        let n2 = points.len().next_power_of_two();

        if !n_lanes.is_power_of_two() {
            return Err(NewCaptError::InvalidLaneCount);
        }
        let lanes_log2 = n_lanes.ilog2();

        if points.iter().any(|p| p.iter().any(|x| !x.is_finite())) {
            return Err(NewCaptError::NonFinite);
        }

        let mut tests = vec![A::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut points2 = vec![[A::INFINITY; K]; n2].into_boxed_slice();
        points2[..points.len()].copy_from_slice(points);

        let layout = Layout::array::<A>(n_lanes).map_err(|_| NewCaptError::InvalidLaneCount)?;
        let alignment = max(layout.size(), layout.align());
        // hack - reduce number of reallocations by allocating a lot of points from the start
        let mut afforded = array::from_fn(|_| AVec::with_capacity(alignment, n2 * 100));
        let mut starts = vec![I::ZERO; n2 + 1].into_boxed_slice();

        let mut aabbs = vec![
            Aabb {
                lo: [A::NEG_INFINITY; K],
                hi: [A::INFINITY; K],
            };
            n2
        ]
        .into_boxed_slice();

        unsafe {
            // SAFETY: We tested that `points` contains no `NaN` values.
            Self::new_help(
                &mut points2,
                &mut tests,
                &mut aabbs,
                &mut afforded,
                &mut starts,
                0,
                0,
                r_range,
                lanes_log2,
                Vec::new(),
                Aabb::ALL,
            )?;
        }

        Ok(Self {
            tests,
            starts,
            afforded: afforded.map(AVec::into_boxed_slice),
            aabbs,
            r_point,
            lanes_log2,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    /// # Safety
    ///
    /// This function will contain undefined behavior if `points` contains any `NaN` values.
    unsafe fn new_help(
        points: &mut [[A; K]],
        tests: &mut [A],
        aabbs: &mut [Aabb<A, K>],
        afforded: &mut [AVec<A, RuntimeAlign>; K],
        starts: &mut [I],
        k: usize,
        i: usize,
        r_range: (A, A),
        lanes_log2: u32,
        in_range: Vec<[A; K]>,
        cell: Aabb<A, K>,
    ) -> Result<(), NewCaptError> {
        let lanes_mask = (1 << lanes_log2) - 1;
        unsafe {
            let rsq_min = r_range.0.square();
            if let [rep] = *points {
                let z = i - tests.len();
                let aabb = &mut aabbs[z];
                *aabb = Aabb { lo: rep, hi: rep };
                if rep[0].is_finite() {
                    // lanes for afforded points
                    for k in 0..K {
                        afforded[k].push(rep[k]);
                    }
                    let mut n_points_added = 1;

                    // populate affordance buffer if the representative doesn't cover everything
                    if !cell.contained_by_ball(&rep, rsq_min) {
                        for ak in afforded.iter_mut() {
                            ak.reserve(ak.len() + in_range.len());
                        }
                        for p in in_range {
                            aabb.insert(&p);

                            // add this point to the lane
                            for k in 0..K {
                                afforded[k].push(p[k]);
                            }
                            n_points_added += 1;
                        }
                    }
                    // fill out the last block with infinities
                    while n_points_added & lanes_mask != 0 {
                        for aff in afforded.iter_mut() {
                            aff.push(A::INFINITY);
                        }
                        n_points_added += 1;
                    }

                    assert!(afforded[0].len() & lanes_mask == 0);
                    for k in 0..(K - 1) {
                        assert_eq!(afforded[k].len(), afforded[k + 1].len());
                    }
                }

                starts[z + 1] = afforded[0]
                    .len()
                    .try_into()
                    .map_err(|_| NewCaptError::TooManyPoints)?;
                let is_misaligned = (starts[z + 1])
                    .try_into()
                    .is_ok_and(|start| (start & lanes_mask) != 0);
                assert!(
                    !is_misaligned,
                    "start indices for affordance bufs must be aligned"
                );
                return Ok(());
            }

            let test = median_partition(points, k);
            tests[i] = test;

            let (lhs, rhs) = points.split_at_mut(points.len() / 2);
            let (lo_vol, hi_vol) = cell.split(test, k);

            let lo_too_small = distsq(lo_vol.lo, lo_vol.hi) <= rsq_min;
            let hi_too_small = distsq(hi_vol.lo, hi_vol.hi) <= rsq_min;

            // retain only points which might be in the affordance buffer for the split-out cells
            let (lo_afford, hi_afford) = match (lo_too_small, hi_too_small) {
                (false, false) => {
                    let mut lo_afford = in_range;
                    let mut hi_afford = lo_afford.clone();
                    lo_afford.retain(|pt| pt[k] <= test + r_range.1);
                    lo_afford.extend(rhs.iter().filter(|pt| pt[k] <= test + r_range.1));
                    hi_afford.retain(|pt| pt[k].is_finite() && test - r_range.1 <= pt[k]);
                    hi_afford.extend(
                        lhs.iter()
                            .filter(|pt| pt[k].is_finite() && test - r_range.1 <= pt[k]),
                    );

                    (lo_afford, hi_afford)
                }
                (false, true) => {
                    let mut lo_afford = in_range;
                    lo_afford.retain(|pt| pt[k] <= test + r_range.1);
                    lo_afford.extend(rhs.iter().filter(|pt| pt[k] <= test + r_range.1));

                    (lo_afford, Vec::new())
                }
                (true, false) => {
                    let mut hi_afford = in_range;
                    hi_afford.retain(|pt| test - r_range.1 <= pt[k]);
                    hi_afford.extend(
                        lhs.iter()
                            .filter(|pt| pt[k].is_finite() && test - r_range.1 <= pt[k]),
                    );

                    (Vec::new(), hi_afford)
                }
                (true, true) => (Vec::new(), Vec::new()),
            };

            let next_k = (k + 1) % K;
            Self::new_help(
                lhs,
                tests,
                aabbs,
                afforded,
                starts,
                next_k,
                2 * i + 1,
                r_range,
                lanes_log2,
                lo_afford,
                lo_vol,
            )?;
            Self::new_help(
                rhs,
                tests,
                aabbs,
                afforded,
                starts,
                next_k,
                2 * i + 2,
                r_range,
                lanes_log2,
                hi_afford,
                hi_vol,
            )?;

            Ok(())
        }
    }

    #[must_use]
    /// Determine whether a point in this tree is within a distance of `radius` to `center`.
    ///
    /// Note that this function will accept query radii outside of the range `r_range` passed to the
    /// construction for this CAPT in [`Capt::new`] or [`Capt::try_new`]. However, if the query
    /// radius is outside this range, the tree may erroneously return `false` (that is, erroneously
    /// report non-collision).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let points = [[0.0; 3], [1.0; 3], [0.1, 0.5, 1.0]];
    /// let capt = capt::Capt::<3>::new(&points, (0.0, 1.0), 1);
    ///
    /// assert!(capt.collides(&[1.1; 3], 0.2));
    /// assert!(!capt.collides(&[2.0; 3], 1.0));
    ///
    /// // no guarantees about what this is, since the radius is greater than the construction range
    /// println!(
    ///     "collision check result is {:?}",
    ///     capt.collides(&[100.0; 3], 100.0)
    /// );
    /// ```
    pub fn collides(&self, center: &[A; K], mut radius: A) -> bool {
        radius = radius + self.r_point;
        // forward pass through the tree
        let mut test_idx = 0;
        let mut k = 0;
        for _ in 0..self.tests.len().trailing_ones() {
            test_idx = 2 * test_idx
                + 1
                + usize::from(unsafe { *self.tests.get_unchecked(test_idx) } <= center[k]);
            k = (k + 1) % K;
        }

        // retrieve affordance buffer location
        let rsq = radius.square();
        let i = test_idx - self.tests.len();
        let aabb = unsafe { self.aabbs.get_unchecked(i) };
        if aabb.closest_distsq_to(center) > rsq {
            return false;
        }

        let mut range = unsafe {
            // SAFETY: The conversion worked the first way.
            self.starts[i].try_into().unwrap_unchecked()
                ..self.starts[i + 1].try_into().unwrap_unchecked()
        };

        // check affordance buffer
        range.any(|i| {
            let aff_pt = array::from_fn(|k| self.afforded[k][i]);
            distsq(aff_pt, *center) <= rsq
        })
    }
}

impl<I, const K: usize> Capt<K, f32, I>
where
    I: Index,
{
    #[must_use]
    /// Like [`Capt::collides`] but vectorises the affordance-buffer distance scan with a
    /// `hydroplane` kernel for the common `K = 3` case: the tree descent stays scalar, the
    /// per-leaf point scan runs as SIMD.
    pub fn query_simd(&self, center: &[f32; K], radius: f32) -> bool {
        let radius = radius + self.r_point;
        let mut test_idx = 0;
        let mut k = 0;
        for _ in 0..self.tests.len().trailing_ones() {
            test_idx = 2 * test_idx
                + 1
                + usize::from(unsafe { *self.tests.get_unchecked(test_idx) } <= center[k]);
            k = (k + 1) % K;
        }

        let rsq = radius * radius;
        let i = test_idx - self.tests.len();
        let aabb = unsafe { self.aabbs.get_unchecked(i) };
        if aabb.closest_distsq_to(center) > rsq {
            return false;
        }

        let start: usize = unsafe { self.starts[i].try_into().unwrap_unchecked() };
        let end: usize = unsafe { self.starts[i + 1].try_into().unwrap_unchecked() };

        if K == 3 {
            afforded3_hit(
                &self.afforded[0][start..end],
                &self.afforded[1][start..end],
                &self.afforded[2][start..end],
                glam::Vec3::new(center[0], center[1], center[2]),
                rsq,
            )
        } else {
            (start..end).any(|i| {
                let aff_pt = array::from_fn(|k| self.afforded[k][i]);
                distsq(aff_pt, *center) <= rsq
            })
        }
    }
}

fn distsq<A: Axis, const K: usize>(a: [A; K], b: [A; K]) -> A {
    let mut total = A::ZERO;
    for i in 0..K {
        total = total + (a[i] - b[i]).square();
    }
    total
}

impl<A, const K: usize> Aabb<A, K>
where
    A: Axis,
{
    const ALL: Self = Self {
        lo: [A::NEG_INFINITY; K],
        hi: [A::INFINITY; K],
    };

    /// Split this volume by a test plane with value `test` along `dim`.
    const fn split(mut self, test: A, dim: usize) -> (Self, Self) {
        let mut rhs = self;
        self.hi[dim] = test;
        rhs.lo[dim] = test;

        (self, rhs)
    }

    fn contained_by_ball(&self, center: &[A; K], rsq: A) -> bool {
        let mut dist = A::ZERO;

        #[allow(clippy::needless_range_loop)]
        for k in 0..K {
            let lo_diff = (self.lo[k] - center[k]).square();
            let hi_diff = (self.hi[k] - center[k]).square();

            dist = dist + if lo_diff < hi_diff { hi_diff } else { lo_diff };
        }

        dist <= rsq
    }

    #[doc(hidden)]
    pub fn closest_distsq_to(&self, pt: &[A; K]) -> A {
        let mut dist = A::ZERO;

        #[allow(clippy::needless_range_loop)]
        for d in 0..K {
            let clamped = clamp(pt[d], self.lo[d], self.hi[d]);
            dist = dist + (pt[d] - clamped).square();
        }

        dist
    }

    fn insert(&mut self, point: &[A; K]) {
        self.lo
            .iter_mut()
            .zip(&mut self.hi)
            .zip(point)
            .for_each(|((l, h), &x)| {
                if *l > x {
                    *l = x;
                }
                if x > *h {
                    *h = x;
                }
            });
    }
}

#[inline]
/// Calculate the "true" median (halfway between two midpoints) and partition `points` about said
/// median along axis `d`.
///
/// # Safety
///
/// This function will result in undefined behavior if `points` contains any `NaN` values.
unsafe fn median_partition<A: Axis, const K: usize>(points: &mut [[A; K]], k: usize) -> A {
    unsafe {
        let (lh, med_hi, _) = points.select_nth_unstable_by(points.len() / 2, |a, b| {
            a[k].partial_cmp(&b[k]).unwrap_unchecked()
        });
        let med_lo = lh
            .iter_mut()
            .map(|p| p[k])
            .max_by(|a, b| a.partial_cmp(b).unwrap_unchecked())
            .unwrap();
        A::in_between(med_lo, med_hi[k])
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use alloc::vec::Vec;
    use core::alloc::Layout;
    use core::cmp::max;
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, SerializeStruct, Serializer};

    impl<A: Serialize, const K: usize> Serialize for Aabb<A, K> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut state = serializer.serialize_struct("Aabb", 2)?;
            state.serialize_field("lo", self.lo.as_slice())?;
            state.serialize_field("hi", self.hi.as_slice())?;
            state.end()
        }
    }

    #[derive(serde::Deserialize)]
    struct AabbData<A> {
        lo: Vec<A>,
        hi: Vec<A>,
    }

    impl<'de, A: Deserialize<'de> + Copy, const K: usize> Deserialize<'de> for Aabb<A, K> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let data = AabbData::<A>::deserialize(deserializer)?;
            let lo: [A; K] = data.lo.try_into().map_err(|_| {
                serde::de::Error::invalid_length(0, &"expected array of correct length")
            })?;
            let hi: [A; K] = data.hi.try_into().map_err(|_| {
                serde::de::Error::invalid_length(0, &"expected array of correct length")
            })?;
            Ok(Aabb { lo, hi })
        }
    }

    impl<const K: usize, A, I> Serialize for Capt<K, A, I>
    where
        A: Serialize + Axis,
        I: Serialize,
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut state = serializer.serialize_struct("Capt", 6)?;
            state.serialize_field("tests", &*self.tests)?;
            state.serialize_field("aabbs", &*self.aabbs)?;
            state.serialize_field("starts", &*self.starts)?;
            let afforded_slices: Vec<&[A]> = self.afforded.iter().map(|a| &**a).collect();
            state.serialize_field("afforded", &afforded_slices)?;
            state.serialize_field("r_point", &self.r_point)?;
            state.serialize_field("lanes_log2", &self.lanes_log2)?;
            state.end()
        }
    }

    #[derive(serde::Deserialize)]
    #[serde(bound(deserialize = "A: serde::Deserialize<'de> + Copy, I: serde::Deserialize<'de>"))]
    struct CaptData<const K: usize, A, I> {
        tests: Vec<A>,
        aabbs: Vec<Aabb<A, K>>,
        starts: Vec<I>,
        afforded: Vec<Vec<A>>,
        r_point: A,
        lanes_log2: u32,
    }

    impl<'de, const K: usize, A, I> Deserialize<'de> for Capt<K, A, I>
    where
        A: Deserialize<'de> + Axis + Copy,
        I: Deserialize<'de>,
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let data = CaptData::<K, A, I>::deserialize(deserializer)?;

            let n_lanes = 1usize << data.lanes_log2;
            let layout = Layout::array::<A>(n_lanes).map_err(serde::de::Error::custom)?;
            let alignment = max(layout.size(), layout.align());

            let afforded_arr: [ABox<[A], RuntimeAlign>; K] = {
                let afforded_vecs = data.afforded;
                if afforded_vecs.len() != K {
                    return Err(serde::de::Error::custom(alloc::format!(
                        "expected {K} afforded arrays, got {}",
                        afforded_vecs.len()
                    )));
                }
                let mut iter = afforded_vecs.into_iter();
                array::from_fn(|_| {
                    let v = iter.next().unwrap();
                    let mut avec = AVec::with_capacity(alignment, v.len());
                    for val in v {
                        avec.push(val);
                    }
                    avec.into_boxed_slice()
                })
            };

            Ok(Self {
                tests: data.tests.into_boxed_slice(),
                aabbs: data.aabbs.into_boxed_slice(),
                starts: data.starts.into_boxed_slice(),
                afforded: afforded_arr,
                r_point: data.r_point,
                lanes_log2: data.lanes_log2,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    use super::*;

    #[test]
    fn build_simple() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Capt::<2>::new(&points, (0.0, 0.2), 1);
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Capt::<2>::new(&points, (0.0, 0.2), 1);

        println!("{t:?}");

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, 0.12));
    }

    #[test]
    fn another_one() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Capt::<2>::new(&points, (0.0, 0.2), 1);

        println!("{t:?}");

        let q0 = [0.003_265_380_9, 0.106_527_805];
        assert!(t.collides(&q0, 0.02));
    }

    #[test]
    fn three_d() {
        let points = [
            [0.0; 3],
            [0.1, -1.1, 0.5],
            [-0.2, -0.3, 0.25],
            [0.1, -1.1, 0.5],
        ];

        let t = Capt::<3>::new(&points, (0.0, 0.2), 1);

        println!("{t:?}");
        assert!(t.collides(&[0.0, 0.1, 0.0], 0.11));
        assert!(!t.collides(&[0.0, 0.1, 0.0], 0.05));
    }

    #[test]
    fn fuzz() {
        const R: f32 = 0.04;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = SmallRng::seed_from_u64(1234);
        let t = Capt::<2>::new(&points, (0.0, R), 1);

        for _ in 0..10_000 {
            let p = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
            let collides = points.iter().any(|a| distsq(*a, p) <= R * R);
            println!("{p:?}; {collides}");
            assert_eq!(collides, t.collides(&p, R));
        }
    }

    #[test]
    /// This test _should_ fail, but it doesn't somehow?
    fn weird_bounds() {
        const R_SQ: f32 = 1.0;
        let points = [
            [-1.0, 0.0],
            [0.001, 0.0],
            [0.0, 0.5],
            [-1.0, 10.0],
            [-2.0, 10.0],
            [-3.0, 10.0],
            [-0.5, 0.0],
            [-11.0, 1.0],
            [-1.0, -0.5],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ];
        let rsq_range = (R_SQ - f32::EPSILON, R_SQ + f32::EPSILON);
        let t = Capt::<2>::new(&points, rsq_range, 1);
        println!("{t:?}");

        assert!(t.collides(&[-0.001, -0.2], 1.0));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn does_it_partition() {
        let mut points = vec![[1.0], [2.0], [1.5], [2.1], [-0.5]];
        let median = unsafe { median_partition(&mut points, 0) };
        assert_eq!(median, 1.25);
        for p0 in &points[..points.len() / 2] {
            assert!(p0[0] <= median);
        }

        for p0 in &points[points.len() / 2..] {
            assert!(p0[0] >= median);
        }
    }

    #[test]
    fn point_radius() {
        let points = [[0.0, 0.0], [0.0, 1.0]];
        let r_range = (0.0, 1.0);

        let capt: Capt<_, _, u32> = Capt::with_point_radius(&points, r_range, 0.5, 8);
        assert!(capt.collides(&[0.6, 0.0], 0.2));
        assert!(!capt.collides(&[0.6, 0.0], 0.05));
    }
}
