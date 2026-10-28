# wreck

A 3D collision detection library for Rust. Built on top of `glam` for math, [`squiggle`](https://crates.io/crates/squiggle) for the line types, and [`hydroplane`](../hydroplane) for runtime-dispatched SIMD acceleration (SSE4 / AVX2 / AVX-512 / NEON, selected per host CPU).

`no_std` compatible: disable default features and enable `libm`.

## Traits

- `Collides<T>` - boolean collision test between two shapes. Every pair of built-in shapes has an implementation in both directions.
- `Raycast<T>` - where a `Line`, `Ray` or `LineSegment` first meets a shape, as a point and a parameter along the line. Implemented for every volume, `Plane`, `ConvexPolygon`, `Pointcloud` and `Collider`, in both directions.
- `Scalable` - uniform scaling of a shape.
- `Transformable` - translate, rotate (via `Mat3A` or `Quat`), or apply a full `Affine3A` transform.
- `Stretchable` - sweep a shape along a translation vector, producing the convex hull of the motion.
- `Bounded` - compute bounding volumes: `broadphase()` (bounding sphere), `obb()` (oriented bounding box), and `aabb()` (axis-aligned bounding box). Implemented for all volume types, `ConvexPolygon`, `LineSegment`, `Pointcloud`, and `Collider`. For `Collider`, returns infinite bounds if the collider contains any `Plane`, `Line`, or `Ray`.

Every method taking `f32` types also has a `_d` variant taking the `f64` equivalents (`translate_d`, `stretch_d`, `Sphere::new_d`, ...).

## Shapes

### Volumes

- **Sphere** - center + radius
- **Capsule** - line segment + radius, with a fast path for Z-aligned capsules
- **Cylinder** - flat-capped cylinder between two endpoints, with a fast path for Z-aligned cylinders
- **Cuboid** - oriented bounding box (center, 3 axes, half-extents), with fast paths for axis-aligned cases
- **ConvexPolytope** - arbitrary convex shape defined by half-planes and vertices, with an OBB broadphase
- **ArrayConvexPolytope<P, V>** - fixed-size, `const`-constructible `ConvexPolytope`

### Surfaces

- **Plane** - infinite half-space defined by a normal and offset (`normal · x <= d`)
- **ConvexPolygon** - bounded convex polygon embedded in 3D, defined by a center, normal, and 2D vertices
- **ArrayConvexPolygon<V>** - fixed-size, `const`-constructible `ConvexPolygon`

### Lines

Re-exported from `squiggle`.

- **Line** - infinite line through a point with a direction
- **Ray** - half-line with an origin and direction
- **LineSegment** - finite segment between two endpoints

### Primitives

- **Point** - a single point in space

### Spatial

- **Pointcloud** - point cloud backed by a CAPT spatial index for fast broadphase queries

## Collider

`Collider` is a heterogeneous collection of shapes that can be tested against any single shape or another `Collider`. Spheres, capsules, cuboids and cylinders are stored column-wise so the SIMD kernels can test a query against all of them at once. A cached bounding sphere rejects distant queries before any narrowphase runs.

`Transformable` and `Scalable` apply to every contained shape. `Stretchable` is only implemented for `Collider<NoPcl>`, since a point cloud can't be stretched; `Collider::try_stretch` returns `None` if any point clouds are present.

```rust
use glam::Vec3;
use wreck::*;

let mut collider: Collider = Collider::new();
collider.add(Sphere::new(Vec3::ZERO, 1.0));
collider.add(Cuboid::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0)));
collider.add_vec(vec![Capsule::new(Vec3::ZERO, Vec3::Z, 0.1)]);

// test against a single shape
let probe = Sphere::new(Vec3::new(0.5, 0.0, 0.0), 0.2);
if collider.collides(&probe) {
    // hit
}

// test against another collider
let mut other: Collider = Collider::new();
other.add(Cylinder::new(Vec3::ZERO, Vec3::Y, 0.5));
collider.collides_other(&other);

// merge colliders
let merged = Collider::union_of(&[&collider, &other]);
collider.include(other);
```

## Shapes API

### Creating shapes

```rust
use glam::Vec3;
use wreck::*;

// Sphere
let sphere = Sphere::new(Vec3::ZERO, 1.0);

// Capsule — defined by two endpoints and a radius
let capsule = Capsule::new(
    Vec3::new(-2.0, 0.0, 0.0),
    Vec3::new(2.0, 0.0, 0.0),
    0.3,
);

// Cylinder — same as capsule, with flat caps
let cylinder = Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0), 0.5);

// Cuboid — oriented bounding box
let cuboid = Cuboid::new(
    Vec3::ZERO,                                          // center
    [Vec3::X, Vec3::Y, Vec3::Z],                         // axes
    [1.0, 2.0, 0.5],                                     // half-extents
);
// or from an axis-aligned bounding box
let aabb = Cuboid::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0));

// Point
let point = Point::new(1.0, 2.0, 3.0);

// Line — infinite, through a point with a direction
let line = Line::new(Vec3::ZERO, Vec3::Y);
let line = Line::from_points(Vec3::ZERO, Vec3::ONE);

// Ray — half-line from an origin
let ray = Ray::new(Vec3::ZERO, Vec3::X);

// LineSegment — between two endpoints
let seg = LineSegment::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0));

// Plane — infinite half-space (normal · x <= d)
let plane = Plane::new(Vec3::Y, 0.0);
let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);

// ConvexPolygon — bounded polygon in 3D, vertices in CCW order
let polygon = ConvexPolygon::new(
    Vec3::ZERO,                                          // center
    Vec3::Z,                                             // normal
    vec![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],  // 2D vertices
);

// ConvexPolytope — arbitrary convex hull
let polytope = ConvexPolytope::new(
    vec![(Vec3::X, 1.0), (-Vec3::X, 1.0),               // half-planes (normal, d)
         (Vec3::Y, 1.0), (-Vec3::Y, 1.0),
         (Vec3::Z, 1.0), (-Vec3::Z, 1.0)],
    vec![                                                // vertices
        Vec3::new( 1.0,  1.0,  1.0), Vec3::new(-1.0,  1.0,  1.0),
        Vec3::new( 1.0, -1.0,  1.0), Vec3::new(-1.0, -1.0,  1.0),
        Vec3::new( 1.0,  1.0, -1.0), Vec3::new(-1.0,  1.0, -1.0),
        Vec3::new( 1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0),
    ],
);

// Pointcloud — points, (min, max) query radius range, radius of each point
let pcl = Pointcloud::new(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], (0.0, 1.0), 0.01);
```

### Collision testing

```rust
use wreck::Collides;

// pairwise — works in both directions
sphere.collides(&capsule);
capsule.collides(&sphere);
```

### Raycasting

`Raycast` finds where a `Line`, `Ray` or `LineSegment` first meets a shape.

```rust
use wreck::Raycast;

let ray = Ray::new(Vec3::ZERO, Vec3::X);

if let Some(hit) = ray.raycast(&sphere) {
    hit.point; // Vec3 — the intersection point
    hit.t;     // parameter along the ray: origin + dir * t
}

// works in both directions, and against a whole Collider (nearest shape wins)
let same = sphere.raycast(&ray);
let nearest = ray.raycast(&collider);
```

`t` is in the line's own parameter space — `origin + dir * t` for `Line` and `Ray`,
`start + (end - start) * t` for `LineSegment`. Directions are not normalized, so `t` is a
distance only when the direction happens to be unit length.

The hit is the *first* point of the line that is inside the shape, searching forward over that
line type's domain: all of `(-∞, ∞)` for `Line`, `[0, ∞)` for `Ray`, `[0, 1]` for `LineSegment`.
So `raycast` returns `Some` exactly when `collides` returns `true`, and a ray that starts inside
a shape reports its own origin (`t = 0`) rather than the far surface.

Implemented for every volume, `Plane`, `ConvexPolygon`, `Pointcloud` and `Collider`. Line-vs-line
and line-vs-`Point` pairs are left out, the same pairs `Collides` always answers `false` for.

### Transforming shapes

```rust
use glam::{Affine3A, Quat, Vec3A};
use wreck::{Transformable, Scalable};

let mut sphere = Sphere::new(Vec3::ZERO, 1.0);

// in-place
sphere.translate(Vec3A::X);
sphere.scale(2.0);

// or get a new copy
let moved = sphere.translated(Vec3A::new(5.0, 0.0, 0.0));
let rotated = sphere.rotated_quat(Quat::from_rotation_y(1.0));
let transformed = sphere.transformed(Affine3A::IDENTITY);
```

### Bounding volumes

```rust
use wreck::Bounded;

// bounding sphere
let bsphere = capsule.broadphase();

// axis-aligned bounding box
let aabb = capsule.aabb();

// oriented bounding box
let obb = capsule.obb();

// works on Collider too — returns infinite bounds if it contains a Plane, Line, or Ray
let collider_aabb = collider.aabb();
```

### Stretching (swept volumes)

```rust
use wreck::Stretchable;

// stretching a sphere along a direction produces a capsule
let capsule = Sphere::new(Vec3::ZERO, 1.0).stretch(Vec3::X);

// stretching a cuboid produces either a larger cuboid (axis-aligned case)
// or a convex polytope (general case)
let swept = Cuboid::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0))
    .stretch(Vec3::new(1.0, 1.0, 0.0));
```

Stretch outputs are enums in `wreck::stretched` (`SphereStretch`, `CuboidStretch`, ...) holding whichever shape the sweep produced.

## Features

- `std` (default) — use `std` for float math
- `libm` — use `libm` for float math in `no_std` builds; one of `std` or `libm` is required
- `serde` — `Serialize` / `Deserialize` for all shapes and `Collider`
- `approx` — `AbsDiffEq` / `RelativeEq` for shapes
- `quote` — `From<Shape> for TokenStream`, for embedding shapes in generated code
- `valuable` — `Valuable` for shapes
- `wreck-assert` — enables internal assertions in all builds
- `debug-wreck-assert` — enables internal assertions in debug builds only
