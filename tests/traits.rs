use static_assertions::{assert_impl_all, assert_not_impl_any};
use wreck::*;

// Every shape except Pointcloud must implement CollidesWithEverything<Pointcloud>.
assert_impl_all!(Sphere: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Capsule: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Cuboid: CollidesWithEverything<Pointcloud>);
assert_impl_all!(ConvexPolytope: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Point: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Plane: CollidesWithEverything<Pointcloud>);
assert_impl_all!(ConvexPolygon: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Line: CollidesWithEverything<Pointcloud>);
assert_impl_all!(Ray: CollidesWithEverything<Pointcloud>);
assert_impl_all!(LineSegment: CollidesWithEverything<Pointcloud>);
assert_not_impl_any!(Pointcloud: CollidesWithEverything<Pointcloud>);


assert_impl_all!(Sphere: CollidesWithEverything<NoPcl>);
assert_impl_all!(Capsule: CollidesWithEverything<NoPcl>);
assert_impl_all!(Cuboid: CollidesWithEverything<NoPcl>);
assert_impl_all!(ConvexPolytope: CollidesWithEverything<NoPcl>);
assert_impl_all!(Point: CollidesWithEverything<NoPcl>);
assert_impl_all!(Plane: CollidesWithEverything<NoPcl>);
assert_impl_all!(ConvexPolygon: CollidesWithEverything<NoPcl>);
assert_impl_all!(Line: CollidesWithEverything<NoPcl>);
assert_impl_all!(Ray: CollidesWithEverything<NoPcl>);
assert_impl_all!(LineSegment: CollidesWithEverything<NoPcl>);
assert_impl_all!(Pointcloud: CollidesWithEverything<NoPcl>);
