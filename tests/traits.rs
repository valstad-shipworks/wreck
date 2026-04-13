use static_assertions::{assert_impl_all, assert_not_impl_any};
use wreck::*;

// Every bounded shape must implement CollidesWithEverything<Pointcloud>.
assert_impl_all!(Sphere: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Capsule: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Cuboid: CollidesWithEverything<Pointcloud>);
assert_impl_all!(ConvexPolytope: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Point: CollidesWithEverything<Pointcloud>);
assert_impl_all!(ConvexPolygon: CollidesWithEverything<Pointcloud>);
assert_impl_all!(LineSegment: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Pointcloud: CollidesWithEverything<Pointcloud>);

// Unbounded types (Plane, Line, Ray) do not implement CollidesWithEverything
// because they lack Bounded.
assert_not_impl_any!(Plane: CollidesWithEverything<Pointcloud>);
assert_not_impl_any!(Line: CollidesWithEverything<Pointcloud>);
assert_not_impl_any!(Ray: CollidesWithEverything<Pointcloud>);

assert_impl_all!(Sphere: CollidesWithEverything<NoPcl>);
assert_impl_all!(Capsule: CollidesWithEverything<NoPcl>);
assert_impl_all!(Cuboid: CollidesWithEverything<NoPcl>);
assert_impl_all!(ConvexPolytope: CollidesWithEverything<NoPcl>);
assert_impl_all!(Point: CollidesWithEverything<NoPcl>);
assert_impl_all!(ConvexPolygon: CollidesWithEverything<NoPcl>);
assert_impl_all!(LineSegment: CollidesWithEverything<NoPcl>);
assert_impl_all!(Pointcloud: CollidesWithEverything<NoPcl>);
