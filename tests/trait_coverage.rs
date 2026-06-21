//! Compile-time guarantee that the line types (now re-exported from `squiggle`)
//! implement every wreck-native trait they are expected to. If an impl is ever
//! dropped, this test fails to compile.

use wreck::{
    ArrayConvexPolytope, Bounded, Capsule, Collides, ColliderComponent, ConvexPolygon,
    ConvexPolytope, Cuboid, Cylinder, Line, LineSegment, Plane, Point, Pointcloud, Ray, Scalable,
    Sphere, Stretchable, Transformable,
};

fn assert_scalable<T: Scalable>() {}
fn assert_transformable<T: Transformable>() {}
fn assert_stretchable<T: Stretchable>() {}
fn assert_bounded<T: Bounded>() {}
fn assert_component<T: ColliderComponent>() {}
fn assert_collides<A: Collides<B>, B: Sized + Clone>() {}

macro_rules! assert_collides_with_everything {
    ($t:ty) => {{
        assert_collides::<$t, Sphere>();
        assert_collides::<$t, Capsule>();
        assert_collides::<$t, Cuboid>();
        assert_collides::<$t, Cylinder>();
        assert_collides::<$t, Plane>();
        assert_collides::<$t, ConvexPolygon>();
        assert_collides::<$t, ConvexPolytope>();
        assert_collides::<$t, ArrayConvexPolytope<6, 8>>();
        assert_collides::<$t, Point>();
        assert_collides::<$t, Pointcloud>();
        assert_collides::<$t, Line>();
        assert_collides::<$t, Ray>();
        assert_collides::<$t, LineSegment>();
    }};
}

#[test]
fn line_implements_native_traits() {
    assert_scalable::<Line>();
    assert_transformable::<Line>();
    assert_stretchable::<Line>();
    assert_component::<Line>();
    assert_collides_with_everything!(Line);
}

#[test]
fn ray_implements_native_traits() {
    assert_scalable::<Ray>();
    assert_transformable::<Ray>();
    assert_stretchable::<Ray>();
    assert_component::<Ray>();
    assert_collides_with_everything!(Ray);
}

#[test]
fn segment_implements_native_traits() {
    assert_scalable::<LineSegment>();
    assert_transformable::<LineSegment>();
    assert_stretchable::<LineSegment>();
    assert_component::<LineSegment>();
    // Only the bounded segment carries `Bounded`; `Line`/`Ray` are infinite.
    assert_bounded::<LineSegment>();
    assert_collides_with_everything!(LineSegment);
}
