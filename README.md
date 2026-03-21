# wreck

A 3D collision detection library for Rust. Built on top of `glam` for math and `wide` for SIMD acceleration.

## Traits

- `Collides<T>` - boolean collision test between two shapes. Every pair of built-in shapes has an implementation in both directions.
- `Scalable` - uniform scaling of a shape.
- `Transformable` - translate, rotate (via `Mat3` or `Quat`), or apply a full `Affine3` transform.
- `Stretchable` - sweep a shape along a translation vector, producing the convex hull of the motion.

## Shapes

### Volumes

- **Sphere** - center + radius
- **Capsule** - line segment + radius, with a fast path for Z-aligned capsules
- **Cuboid** - oriented bounding box (center, 3 axes, half-extents), with fast paths for axis-aligned cases
- **ConvexPolytope** - arbitrary convex shape defined by half-planes and vertices, with an OBB broadphase

### Surfaces

- **Plane** - infinite half-space defined by a normal and offset (`normal · x <= d`)
- **ConvexPolygon** - bounded convex polygon embedded in 3D, defined by a center, normal, and 2D vertices

### Lines

- **Line** - infinite line through a point with a direction
- **Ray** - half-line with an origin and direction
- **LineSegment** - finite segment between two endpoints

### Primitives

- **Point** - a single point in space

### Spatial

- **Pointcloud** - point cloud backed by a CAPT spatial index for fast broadphase queries

All shape pairs implement `Collides<T>` for boolean intersection tests. The trait also provides `collides_many` which can be more performant.

## Collider

`Collider` is a heterogeneous collection of shapes that can be tested against any single shape or another `Collider`. It supports `Transformable`, `Scalable`, applying the operation to every contained shape.

`Stretchable` is only implemented for Collider<NoPcl> since there's no way to stretch a point cloud.

```rust
use glam::Vec3;
use wreck::*;

let mut collider: Collider = Collider {
    spheres: vec![Sphere::new(Vec3::ZERO, 1.0)],
    ..Default::default()
};

// add shapes dynamically
collider.add(Cuboid::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0)));

// test against a single shape
let probe = Sphere::new(Vec3::new(0.5, 0.0, 0.0), 0.2);
if collider.collides(&probe) {
    // hit
}
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

// ConvexPolygon — bounded polygon in 3D
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
```

### Collision testing

```rust
use wreck::Collides;

// pairwise — works in both directions
sphere.collides(&capsule);
capsule.collides(&sphere);

// batch — SIMD-accelerated for some shapes
let others: Vec<Sphere> = vec![/* ... */];
sphere.collides_many(&others);
```

### Transforming shapes

```rust
use wreck::{Transformable, Scalable};

let mut sphere = Sphere::new(Vec3::ZERO, 1.0);

// in-place
sphere.translate(Vec3::new(1.0, 0.0, 0.0));
sphere.scale(2.0);

// or get a new copy
let moved = sphere.clone().translated([5.0, 0.0, 0.0]);
let rotated = sphere.clone().rotated_quat(glam::Quat::from_rotation_y(1.0));
let transformed = sphere.clone().transformed(glam::Affine3::IDENTITY);
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

## Features

- `convex-polytope` (default) - enables `ConvexPolytope` and the `Stretchable` trait
- `capt` - enables `Pointcloud` for collision testing against point cloud data using the CAPT spatial index

## Performance notes

Most collision routines have multiple tiers of early-outs. For example capsule-cuboid does a bounding sphere check first, then tries a Z-aligned + axis-aligned fast path, and only falls through to the general SIMD path if neither applies. Cuboid-cuboid uses SAT with 15 separating axes but short-circuits to simple AABB overlap when both are axis aligned.

The `collides_many` methods on Sphere use `wide::f32x8` to test 8 candidates per iteration with no branching in the hot loop.

There is a benchmark suite under `benches/` you can run with `cargo bench`.
