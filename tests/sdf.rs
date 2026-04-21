#![cfg(feature = "sdf")]

use glam::Vec3;
use wreck::sdf::gjk_epa_signed_distance;
use wreck::{
    Capsule, ConvexPolygon, ConvexPolytope, Cuboid, Cylinder, Line, LineSegment, Plane, Point,
    Pointcloud, Ray, SignedDistance, Sphere,
};

const EPS: f32 = 1e-5;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn sphere_sphere_separated() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), 3.0, EPS));
    assert!(approx_eq(b.signed_distance(&a), 3.0, EPS));
}

#[test]
fn sphere_sphere_touching() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), 0.0, EPS));
}

#[test]
fn sphere_sphere_interpenetrating() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), -0.5, EPS));
}

#[test]
fn sphere_sphere_concentric() {
    let a = Sphere::new(Vec3::ZERO, 2.0);
    let b = Sphere::new(Vec3::ZERO, 1.0);
    assert!(approx_eq(a.signed_distance(&b), -3.0, EPS));
}

#[test]
fn sphere_point_outside() {
    let s = Sphere::new(Vec3::ZERO, 1.0);
    let p = Point::new(3.0, 0.0, 0.0);
    assert!(approx_eq(s.signed_distance(&p), 2.0, EPS));
    assert!(approx_eq(p.signed_distance(&s), 2.0, EPS));
}

#[test]
fn sphere_point_on_surface() {
    let s = Sphere::new(Vec3::ZERO, 1.0);
    let p = Point::new(1.0, 0.0, 0.0);
    assert!(approx_eq(s.signed_distance(&p), 0.0, EPS));
}

#[test]
fn sphere_point_interior() {
    let s = Sphere::new(Vec3::ZERO, 2.0);
    let p = Point::new(0.5, 0.0, 0.0);
    assert!(approx_eq(s.signed_distance(&p), -1.5, EPS));
}

#[test]
fn point_point_same() {
    let a = Point::new(1.0, 2.0, 3.0);
    let b = Point::new(1.0, 2.0, 3.0);
    assert!(approx_eq(a.signed_distance(&b), 0.0, EPS));
}

#[test]
fn point_point_apart() {
    let a = Point::new(0.0, 0.0, 0.0);
    let b = Point::new(3.0, 4.0, 0.0);
    assert!(approx_eq(a.signed_distance(&b), 5.0, EPS));
}

#[test]
fn sphere_capsule_separated() {
    let s = Sphere::new(Vec3::new(0.0, 5.0, 0.0), 1.0);
    let c = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&c), 3.0, EPS));
    assert!(approx_eq(c.signed_distance(&s), 3.0, EPS));
}

#[test]
fn sphere_capsule_touching_side() {
    let s = Sphere::new(Vec3::new(0.0, 3.0, 0.0), 1.0);
    let c = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&c), 1.0, EPS));
}

#[test]
fn sphere_capsule_touching_end() {
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 2.0);
    let c = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&c), 0.0, EPS));
}

#[test]
fn sphere_capsule_interpenetrating() {
    let s = Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0);
    let c = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&c), -1.0, EPS));
}

#[test]
fn sphere_cuboid_separated_axis_aligned() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&cube), 3.0, EPS));
    assert!(approx_eq(cube.signed_distance(&s), 3.0, EPS));
}

#[test]
fn sphere_cuboid_corner() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let s = Sphere::new(Vec3::new(4.0, 4.0, 4.0), 1.0);
    let corner_dist = (Vec3::new(3.0, 3.0, 3.0)).length();
    assert!(approx_eq(s.signed_distance(&cube), corner_dist - 1.0, EPS));
}

#[test]
fn sphere_cuboid_touching_face() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let s = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&cube), 0.0, EPS));
}

#[test]
fn sphere_cuboid_interior() {
    let cube = Cuboid::from_aabb(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));
    let s = Sphere::new(Vec3::ZERO, 0.5);
    assert!(approx_eq(s.signed_distance(&cube), -2.5, EPS));
}

#[test]
fn sphere_cuboid_overlapping_face() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let s = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&cube), -0.5, EPS));
}

#[test]
fn sphere_plane_outside() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let s = Sphere::new(Vec3::new(0.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&plane), 4.0, EPS));
    assert!(approx_eq(plane.signed_distance(&s), 4.0, EPS));
}

#[test]
fn sphere_plane_touching() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let s = Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&plane), 0.0, EPS));
}

#[test]
fn sphere_plane_interpenetrating() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let s = Sphere::new(Vec3::new(0.0, 0.5, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&plane), -0.5, EPS));
}

#[test]
fn sphere_plane_fully_inside_halfspace() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let s = Sphere::new(Vec3::new(0.0, -5.0, 0.0), 1.0);
    assert!(approx_eq(s.signed_distance(&plane), -6.0, EPS));
}

#[test]
fn capsule_capsule_parallel_separated() {
    let a = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    let b = Capsule::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), 3.0, EPS));
    assert!(approx_eq(b.signed_distance(&a), 3.0, EPS));
}

#[test]
fn capsule_capsule_parallel_touching() {
    let a = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    let b = Capsule::new(Vec3::new(-2.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), 0.0, EPS));
}

#[test]
fn capsule_capsule_parallel_overlapping() {
    let a = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    let b = Capsule::new(Vec3::new(-2.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), -1.0, EPS));
}

#[test]
fn capsule_capsule_end_separated() {
    let a = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    let b = Capsule::new(Vec3::new(6.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(a.signed_distance(&b), 2.0, EPS));
}

#[test]
fn capsule_capsule_crossing_perpendicular() {
    let a = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.5);
    let b = Capsule::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 0.5);
    assert!(approx_eq(a.signed_distance(&b), -1.0, EPS));
}

#[test]
fn capsule_plane_outside() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 7.0, 0.0), 1.0);
    assert!(approx_eq(c.signed_distance(&plane), 4.0, EPS));
    assert!(approx_eq(plane.signed_distance(&c), 4.0, EPS));
}

#[test]
fn capsule_plane_touching() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(-2.0, 1.0, 0.0), Vec3::new(2.0, 3.0, 0.0), 1.0);
    assert!(approx_eq(c.signed_distance(&plane), 0.0, EPS));
}

#[test]
fn capsule_plane_interpenetrating() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(-2.0, -0.5, 0.0), Vec3::new(2.0, 2.0, 0.0), 1.0);
    assert!(approx_eq(c.signed_distance(&plane), -1.5, EPS));
}

#[test]
fn capsule_cuboid_separated_side() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let c = Capsule::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(c.signed_distance(&cube), 3.0, EPS));
    assert!(approx_eq(cube.signed_distance(&c), 3.0, EPS));
}

#[test]
fn capsule_cuboid_touching_face() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let c = Capsule::new(Vec3::new(-2.0, 2.0, 0.0), Vec3::new(2.0, 2.0, 0.0), 1.0);
    assert!(approx_eq(c.signed_distance(&cube), 0.0, EPS));
}

#[test]
fn capsule_cuboid_passing_through() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let c = Capsule::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), 0.5);
    assert!(approx_eq(c.signed_distance(&cube), -1.5, EPS));
}

#[test]
fn capsule_cuboid_endpoint_inside() {
    let cube = Cuboid::from_aabb(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));
    let c = Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 0.3);
    assert!(approx_eq(c.signed_distance(&cube), -2.3, EPS));
}

#[test]
fn cuboid_cuboid_aa_separated() {
    let a = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let b = Cuboid::from_aabb(Vec3::new(4.0, -1.0, -1.0), Vec3::new(6.0, 1.0, 1.0));
    assert!(approx_eq(a.signed_distance(&b), 3.0, EPS));
    assert!(approx_eq(b.signed_distance(&a), 3.0, EPS));
}

#[test]
fn cuboid_cuboid_aa_touching() {
    let a = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let b = Cuboid::from_aabb(Vec3::new(1.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0));
    assert!(approx_eq(a.signed_distance(&b), 0.0, EPS));
}

#[test]
fn cuboid_cuboid_aa_overlapping() {
    let a = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let b = Cuboid::from_aabb(Vec3::new(0.5, -0.5, -0.5), Vec3::new(2.5, 0.5, 0.5));
    assert!(approx_eq(a.signed_distance(&b), -0.5, EPS));
}

#[test]
fn cuboid_cuboid_aa_fully_inside() {
    let outer = Cuboid::from_aabb(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0));
    let inner = Cuboid::from_aabb(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    assert!(approx_eq(outer.signed_distance(&inner), -2.5, EPS));
}

#[test]
fn cuboid_cuboid_rotated_separated() {
    let a = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let angle = core::f32::consts::FRAC_PI_4;
    let (s, co) = angle.sin_cos();
    let b = Cuboid::new(
        Vec3::new(5.0, 0.0, 0.0),
        [
            Vec3::new(co, s, 0.0),
            Vec3::new(-s, co, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        [1.0, 1.0, 1.0],
    );
    let sd = a.signed_distance(&b);
    assert!(sd > 0.0);
    let sd2 = b.signed_distance(&a);
    assert!(approx_eq(sd, sd2, EPS));
}

#[test]
fn cuboid_plane_outside() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, 3.0, -1.0), Vec3::new(1.0, 5.0, 1.0));
    assert!(approx_eq(cube.signed_distance(&plane), 3.0, EPS));
    assert!(approx_eq(plane.signed_distance(&cube), 3.0, EPS));
}

#[test]
fn cuboid_plane_interpenetrating() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -0.5, -1.0), Vec3::new(1.0, 1.5, 1.0));
    assert!(approx_eq(cube.signed_distance(&plane), -0.5, EPS));
}

#[test]
fn gjk_sphere_sphere_separated() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(gjk_epa_signed_distance(&a, &b), 3.0, 1e-3));
}

#[test]
fn gjk_sphere_sphere_touching() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(gjk_epa_signed_distance(&a, &b), 0.0, 1e-3));
}

#[test]
fn gjk_sphere_sphere_interpenetrating() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(approx_eq(gjk_epa_signed_distance(&a, &b), -0.5, 1e-3));
}

#[test]
fn gjk_cuboid_cuboid_aa_separated() {
    let a = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let b = Cuboid::from_aabb(Vec3::new(4.0, -1.0, -1.0), Vec3::new(6.0, 1.0, 1.0));
    assert!(approx_eq(gjk_epa_signed_distance(&a, &b), 3.0, 1e-3));
}

#[test]
fn gjk_cuboid_cuboid_overlapping() {
    let a = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let b = Cuboid::from_aabb(Vec3::new(0.5, -0.5, -0.5), Vec3::new(2.5, 0.5, 0.5));
    assert!(approx_eq(gjk_epa_signed_distance(&a, &b), -0.5, 1e-2));
}

#[test]
fn gjk_sphere_cuboid_diagonal() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let s = Sphere::new(Vec3::new(4.0, 4.0, 4.0), 1.0);
    let corner_dist = Vec3::new(3.0, 3.0, 3.0).length();
    assert!(approx_eq(
        gjk_epa_signed_distance(&s, &cube),
        corner_dist - 1.0,
        1e-2
    ));
}

#[test]
fn gjk_capsule_capsule_parallel() {
    let a = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
    let b = Capsule::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(gjk_epa_signed_distance(&a, &b), 3.0, 1e-3));
}

#[test]
fn gjk_matches_closed_form_sphere_pairs() {
    let s1 = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.2);
    let s2 = Sphere::new(Vec3::new(3.0, 0.5, -0.7), 0.8);
    let closed = s1.signed_distance(&s2);
    let generic = gjk_epa_signed_distance(&s1, &s2);
    assert!(approx_eq(closed, generic, 1e-3));
}

#[test]
fn gjk_matches_closed_form_sphere_cuboid_pairs() {
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let s = Sphere::new(Vec3::new(2.5, 0.3, -0.2), 0.7);
    let closed = s.signed_distance(&cube);
    let generic = gjk_epa_signed_distance(&s, &cube);
    assert!(approx_eq(closed, generic, 1e-2));
}

fn unit_cube_polytope() -> ConvexPolytope {
    let planes = vec![
        (Vec3::X, 1.0),
        (Vec3::NEG_X, 1.0),
        (Vec3::Y, 1.0),
        (Vec3::NEG_Y, 1.0),
        (Vec3::Z, 1.0),
        (Vec3::NEG_Z, 1.0),
    ];
    let vertices = vec![
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    ConvexPolytope::new(planes, vertices)
}

fn unit_square_polygon() -> ConvexPolygon {
    ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    )
}

#[test]
fn polytope_sphere_separated() {
    let p = unit_cube_polytope();
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(p.signed_distance(&s), 3.0, 1e-2));
    assert!(approx_eq(s.signed_distance(&p), 3.0, 1e-2));
}

#[test]
fn polytope_sphere_overlapping() {
    let p = unit_cube_polytope();
    let s = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(approx_eq(p.signed_distance(&s), -0.5, 1e-2));
}

#[test]
fn polytope_polytope_separated() {
    let a = unit_cube_polytope();
    let b = {
        let planes = vec![
            (Vec3::X, 5.0),
            (Vec3::NEG_X, -3.0),
            (Vec3::Y, 1.0),
            (Vec3::NEG_Y, 1.0),
            (Vec3::Z, 1.0),
            (Vec3::NEG_Z, 1.0),
        ];
        let verts = vec![
            Vec3::new(3.0, -1.0, -1.0),
            Vec3::new(3.0, -1.0, 1.0),
            Vec3::new(3.0, 1.0, -1.0),
            Vec3::new(3.0, 1.0, 1.0),
            Vec3::new(5.0, -1.0, -1.0),
            Vec3::new(5.0, -1.0, 1.0),
            Vec3::new(5.0, 1.0, -1.0),
            Vec3::new(5.0, 1.0, 1.0),
        ];
        ConvexPolytope::new(planes, verts)
    };
    assert!(approx_eq(a.signed_distance(&b), 2.0, 1e-2));
}

#[test]
fn polytope_cuboid_overlapping() {
    let p = unit_cube_polytope();
    let cube = Cuboid::from_aabb(Vec3::new(0.5, -0.5, -0.5), Vec3::new(2.5, 0.5, 0.5));
    assert!(approx_eq(p.signed_distance(&cube), -0.5, 1e-2));
    assert!(approx_eq(cube.signed_distance(&p), -0.5, 1e-2));
}

#[test]
fn polytope_capsule_separated() {
    let p = unit_cube_polytope();
    let c = Capsule::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(p.signed_distance(&c), 3.0, 1e-2));
}

#[test]
fn cylinder_sphere_separated() {
    let cyl = Cylinder::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 1.0);
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(cyl.signed_distance(&s), 3.0, 1e-2));
}

#[test]
fn cylinder_sphere_touching_side() {
    let cyl = Cylinder::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 1.0);
    let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(cyl.signed_distance(&s), 1.0, 1e-2));
}

#[test]
fn cylinder_cylinder_parallel() {
    let a = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.5);
    let b = Cylinder::new(Vec3::new(3.0, -1.0, 0.0), Vec3::new(3.0, 1.0, 0.0), 0.5);
    assert!(approx_eq(a.signed_distance(&b), 2.0, 1e-2));
}

#[test]
fn polygon_sphere_above() {
    let poly = unit_square_polygon();
    let s = Sphere::new(Vec3::new(0.0, 3.0, 0.0), 0.5);
    assert!(approx_eq(poly.signed_distance(&s), 2.5, 1e-2));
}

#[test]
fn polygon_sphere_touching() {
    let poly = unit_square_polygon();
    let s = Sphere::new(Vec3::new(0.0, 0.5, 0.0), 0.5);
    assert!(approx_eq(poly.signed_distance(&s), 0.0, 1e-2));
}

#[test]
fn line_segment_sphere_separated() {
    let seg = LineSegment::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0));
    let s = Sphere::new(Vec3::ZERO, 1.0);
    assert!(approx_eq(seg.signed_distance(&s), 4.0, 1e-2));
    assert!(approx_eq(s.signed_distance(&seg), 4.0, 1e-2));
}

#[test]
fn line_segment_cuboid_through() {
    let seg = LineSegment::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(approx_eq(seg.signed_distance(&cube), -1.0, 1e-2));
}

#[test]
fn line_sphere_tangent() {
    let line = Line::new(Vec3::new(0.0, 3.0, 0.0), Vec3::X);
    let s = Sphere::new(Vec3::ZERO, 1.0);
    assert!(approx_eq(line.signed_distance(&s), 2.0, EPS));
    assert!(approx_eq(s.signed_distance(&line), 2.0, EPS));
}

#[test]
fn line_sphere_passing_through() {
    let line = Line::new(Vec3::new(0.0, 0.3, 0.0), Vec3::X);
    let s = Sphere::new(Vec3::ZERO, 1.0);
    assert!(approx_eq(line.signed_distance(&s), -0.7, EPS));
}

#[test]
fn line_point() {
    let line = Line::new(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
    let p = Point::new(0.0, 3.0, 4.0);
    assert!(approx_eq(line.signed_distance(&p), 5.0, EPS));
}

#[test]
fn line_line_skew() {
    let a = Line::new(Vec3::ZERO, Vec3::X);
    let b = Line::new(Vec3::new(0.0, 0.0, 3.0), Vec3::Y);
    assert!(approx_eq(a.signed_distance(&b), 3.0, EPS));
}

#[test]
fn line_line_parallel() {
    let a = Line::new(Vec3::ZERO, Vec3::X);
    let b = Line::new(Vec3::new(0.0, 3.0, 4.0), Vec3::X);
    assert!(approx_eq(a.signed_distance(&b), 5.0, EPS));
}

#[test]
fn line_capsule_through() {
    let line = Line::new(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
    let cap = Capsule::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 0.5);
    assert!(approx_eq(line.signed_distance(&cap), -0.5, EPS));
}

#[test]
fn line_plane_parallel_above() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let line = Line::new(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
    assert!(approx_eq(line.signed_distance(&plane), 5.0, EPS));
}

#[test]
fn line_plane_crossing() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let line = Line::new(Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
    assert_eq!(line.signed_distance(&plane), f32::NEG_INFINITY);
}

#[test]
fn line_cuboid_tangent_outside() {
    let line = Line::new(Vec3::new(0.0, 3.0, 0.0), Vec3::X);
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(approx_eq(line.signed_distance(&cube), 2.0, EPS));
}

#[test]
fn line_cuboid_through() {
    let line = Line::new(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(approx_eq(line.signed_distance(&cube), -1.0, EPS));
}

#[test]
fn ray_sphere_behind() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
    let s = Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(ray.signed_distance(&s), 4.0, EPS));
}

#[test]
fn ray_sphere_ahead() {
    let ray = Ray::new(Vec3::new(0.0, 3.0, 0.0), Vec3::X);
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(ray.signed_distance(&s), 2.0, EPS));
}

#[test]
fn ray_plane_away_from_solid() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let ray = Ray::new(Vec3::new(0.0, 3.0, 0.0), Vec3::Y);
    assert!(approx_eq(ray.signed_distance(&plane), 3.0, EPS));
}

#[test]
fn ray_plane_into_solid() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let ray = Ray::new(Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y);
    assert_eq!(ray.signed_distance(&plane), f32::NEG_INFINITY);
}

#[test]
fn segment_segment_parallel() {
    let a = LineSegment::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
    let b = LineSegment::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(2.0, 3.0, 0.0));
    assert!(approx_eq(a.signed_distance(&b), 3.0, EPS));
}

#[test]
fn segment_plane_above() {
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    let seg = LineSegment::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(2.0, 7.0, 0.0));
    assert!(approx_eq(seg.signed_distance(&plane), 5.0, EPS));
}

#[test]
fn segment_line_crossing_perpendicular() {
    let line = Line::new(Vec3::ZERO, Vec3::Y);
    let seg = LineSegment::new(Vec3::new(-2.0, 0.5, 3.0), Vec3::new(2.0, 0.5, 3.0));
    assert!(approx_eq(seg.signed_distance(&line), 3.0, EPS));
}

fn make_cloud(pts: &[[f32; 3]], r: f32) -> Pointcloud {
    Pointcloud::new(pts, (r, r), r)
}

#[test]
fn pcl_sphere_separated() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.1);
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(pcl.signed_distance(&s), 2.9, 1e-4));
    assert!(approx_eq(s.signed_distance(&pcl), 2.9, 1e-4));
}

#[test]
fn pcl_sphere_touching() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0]], 0.5);
    let s = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(approx_eq(pcl.signed_distance(&s), 0.0, 1e-4));
}

#[test]
fn pcl_sphere_overlapping() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], 0.5);
    let s = Sphere::new(Vec3::new(0.2, 0.0, 0.0), 1.0);
    assert!(approx_eq(pcl.signed_distance(&s), -1.3, 1e-4));
}

#[test]
fn pcl_cuboid_separated() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0]], 0.5);
    let cube = Cuboid::from_aabb(Vec3::new(4.0, -1.0, -1.0), Vec3::new(6.0, 1.0, 1.0));
    assert!(approx_eq(pcl.signed_distance(&cube), 3.5, 1e-3));
}

#[test]
fn pcl_plane_above() {
    let pcl = make_cloud(&[[0.0, 5.0, 0.0], [1.0, 6.0, 0.0]], 0.25);
    let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::Y);
    assert!(approx_eq(pcl.signed_distance(&plane), 4.75, 1e-4));
}

#[test]
fn pcl_pcl_separated() {
    let a = make_cloud(&[[0.0, 0.0, 0.0]], 0.5);
    let b = make_cloud(&[[5.0, 0.0, 0.0]], 0.5);
    assert!(approx_eq(a.signed_distance(&b), 4.0, 1e-4));
    assert!(approx_eq(b.signed_distance(&a), 4.0, 1e-4));
}

#[test]
fn symmetry_sphere_pairs() {
    let s = Sphere::new(Vec3::new(1.0, 2.0, 3.0), 0.7);
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let cap = Capsule::new(Vec3::new(-2.0, -3.0, 0.0), Vec3::new(2.0, 1.0, 0.0), 0.5);
    let plane = Plane::from_point_normal(Vec3::new(0.0, -1.0, 0.0), Vec3::Y);
    let p = Point::new(4.0, -2.0, 1.5);

    assert!(approx_eq(s.signed_distance(&cube), cube.signed_distance(&s), EPS));
    assert!(approx_eq(s.signed_distance(&cap), cap.signed_distance(&s), EPS));
    assert!(approx_eq(s.signed_distance(&plane), plane.signed_distance(&s), EPS));
    assert!(approx_eq(s.signed_distance(&p), p.signed_distance(&s), EPS));
}
