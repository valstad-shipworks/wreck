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
fn capsule_capsule_z_aligned_separated() {
    let a = Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0), 0.5);
    let b = Capsule::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 2.0), 0.5);
    assert!(approx_eq(a.signed_distance(&b), 2.0, EPS));
}

#[test]
fn capsule_capsule_z_aligned_z_gap() {
    let a = Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5);
    let b = Capsule::new(Vec3::new(0.0, 0.0, 4.0), Vec3::new(0.0, 0.0, 5.0), 0.5);
    assert!(approx_eq(a.signed_distance(&b), 2.0, EPS));
}

#[test]
fn capsule_capsule_z_aligned_overlap() {
    let a = Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0), 0.5);
    let b = Capsule::new(Vec3::new(0.5, 0.0, 1.0), Vec3::new(0.5, 0.0, 3.0), 0.5);
    assert!(approx_eq(a.signed_distance(&b), -0.5, EPS));
}

#[test]
fn capsule_capsule_batch_matches_scalar() {
    use wreck::capsule_capsule_sdf_batch;
    let a = vec![
        Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0), 0.5),
        Capsule::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 2.0), 0.3),
        Capsule::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.4),
    ];
    let b = vec![
        Capsule::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 2.0), 0.5),
        Capsule::new(Vec3::new(5.5, 0.0, 0.0), Vec3::new(5.5, 0.0, 2.0), 0.4),
        Capsule::new(Vec3::new(0.0, 7.0, 0.0), Vec3::new(2.0, 7.0, 0.0), 0.3),
    ];
    let mut out = vec![0.0f32; 3];
    capsule_capsule_sdf_batch(&a, &b, &mut out);
    for i in 0..3 {
        assert!(approx_eq(out[i], a[i].signed_distance(&b[i]), 1e-5));
    }
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

// ---------------------------------------------------------------------------
// Ray × rest
// ---------------------------------------------------------------------------

#[test]
fn ray_point() {
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let p = Point::new(5.0, 3.0, 4.0);
    assert!(approx_eq(ray.signed_distance(&p), 5.0, EPS));
    assert!(approx_eq(p.signed_distance(&ray), 5.0, EPS));
}

#[test]
fn ray_point_behind_origin() {
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let p = Point::new(-3.0, 4.0, 0.0);
    assert!(approx_eq(ray.signed_distance(&p), 5.0, EPS));
}

#[test]
fn ray_capsule_separated() {
    let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
    let cap = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(ray.signed_distance(&cap), 4.0, EPS));
    assert!(approx_eq(cap.signed_distance(&ray), 4.0, EPS));
}

#[test]
fn ray_capsule_through() {
    // Ray passes exactly through the capsule's axis segment at (0, 0.5, 0);
    // distance from ray to segment = 0, so SDF = 0 − radius = −1.
    let ray = Ray::new(Vec3::new(-3.0, 0.5, 0.0), Vec3::X);
    let cap = Capsule::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 1.0);
    assert!(approx_eq(ray.signed_distance(&cap), -1.0, EPS));
}

#[test]
fn ray_cuboid_ahead() {
    // Ray at y=z=0 going +X passes through the cuboid interior — negative SDF.
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let cube = Cuboid::from_aabb(Vec3::new(3.0, -1.0, -1.0), Vec3::new(5.0, 1.0, 1.0));
    assert!(approx_eq(ray.signed_distance(&cube), -1.0, EPS));
    assert!(approx_eq(cube.signed_distance(&ray), -1.0, EPS));
}

#[test]
fn ray_cuboid_through() {
    let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X);
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    assert!(approx_eq(ray.signed_distance(&cube), -1.0, EPS));
}

#[test]
fn ray_cuboid_behind() {
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let cube = Cuboid::from_aabb(Vec3::new(-5.0, -1.0, -1.0), Vec3::new(-3.0, 1.0, 1.0));
    assert!(approx_eq(ray.signed_distance(&cube), 3.0, EPS));
}

#[test]
fn ray_ray_skew() {
    let a = Ray::new(Vec3::ZERO, Vec3::X);
    let b = Ray::new(Vec3::new(0.0, 0.0, 4.0), Vec3::Y);
    assert!(approx_eq(a.signed_distance(&b), 4.0, EPS));
}

#[test]
fn ray_ray_parallel() {
    let a = Ray::new(Vec3::ZERO, Vec3::X);
    let b = Ray::new(Vec3::new(0.0, 3.0, 4.0), Vec3::X);
    assert!(approx_eq(a.signed_distance(&b), 5.0, EPS));
}

#[test]
fn ray_line_skew() {
    // Ray: (1+t, 0, 0), t≥0. Line: (0, s, 3). Closest: (1,0,0)↔(0,0,3), dist=√10.
    let r = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::X);
    let l = Line::new(Vec3::new(0.0, 0.0, 3.0), Vec3::Y);
    let expected = (10.0_f32).sqrt();
    assert!(approx_eq(r.signed_distance(&l), expected, EPS));
    assert!(approx_eq(l.signed_distance(&r), expected, EPS));
}

#[test]
fn ray_segment_separated() {
    let r = Ray::new(Vec3::ZERO, Vec3::X);
    let s = LineSegment::new(Vec3::new(5.0, 3.0, 0.0), Vec3::new(7.0, 3.0, 0.0));
    assert!(approx_eq(r.signed_distance(&s), 3.0, EPS));
    assert!(approx_eq(s.signed_distance(&r), 3.0, EPS));
}

// ---------------------------------------------------------------------------
// LineSegment × rest
// ---------------------------------------------------------------------------

#[test]
fn segment_point() {
    let s = LineSegment::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
    let p = Point::new(1.0, 3.0, 4.0);
    assert!(approx_eq(s.signed_distance(&p), 5.0, 1e-2));
    assert!(approx_eq(p.signed_distance(&s), 5.0, 1e-2));
}

#[test]
fn segment_capsule_perpendicular() {
    let seg = LineSegment::new(Vec3::new(0.0, -2.0, 3.0), Vec3::new(0.0, 2.0, 3.0));
    let cap = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.5);
    assert!(approx_eq(seg.signed_distance(&cap), 2.5, 1e-2));
    assert!(approx_eq(cap.signed_distance(&seg), 2.5, 1e-2));
}

// ---------------------------------------------------------------------------
// Cylinder × rest (GJK)
// ---------------------------------------------------------------------------

#[test]
fn cylinder_point_outside() {
    let cyl = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0);
    let p = Point::new(4.0, 0.0, 0.0);
    assert!(approx_eq(cyl.signed_distance(&p), 3.0, 1e-2));
    assert!(approx_eq(p.signed_distance(&cyl), 3.0, 1e-2));
}

#[test]
fn cylinder_capsule_separated() {
    let cyl = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0);
    let cap = Capsule::new(Vec3::new(4.0, -1.0, 0.0), Vec3::new(4.0, 1.0, 0.0), 0.5);
    assert!(approx_eq(cyl.signed_distance(&cap), 2.5, 1e-2));
}

#[test]
fn cylinder_cuboid_separated() {
    let cyl = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0);
    let cube = Cuboid::from_aabb(Vec3::new(3.0, -1.0, -1.0), Vec3::new(5.0, 1.0, 1.0));
    assert!(approx_eq(cyl.signed_distance(&cube), 2.0, 1e-2));
    assert!(approx_eq(cube.signed_distance(&cyl), 2.0, 1e-2));
}

#[test]
fn cylinder_segment_separated() {
    let cyl = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0);
    let seg = LineSegment::new(Vec3::new(4.0, -1.0, 0.0), Vec3::new(4.0, 1.0, 0.0));
    assert!(approx_eq(cyl.signed_distance(&seg), 3.0, 1e-2));
}

// ---------------------------------------------------------------------------
// ConvexPolytope × rest (GJK)
// ---------------------------------------------------------------------------

#[test]
fn polytope_point_outside() {
    let p = unit_cube_polytope();
    let pt = Point::new(4.0, 0.0, 0.0);
    assert!(approx_eq(p.signed_distance(&pt), 3.0, 1e-2));
    assert!(approx_eq(pt.signed_distance(&p), 3.0, 1e-2));
}

#[test]
fn polytope_cylinder_separated() {
    let p = unit_cube_polytope();
    let cyl = Cylinder::new(Vec3::new(4.0, -1.0, 0.0), Vec3::new(4.0, 1.0, 0.0), 1.0);
    assert!(approx_eq(p.signed_distance(&cyl), 2.0, 1e-2));
}

#[test]
fn polytope_segment_separated() {
    let p = unit_cube_polytope();
    let seg = LineSegment::new(Vec3::new(4.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));
    assert!(approx_eq(p.signed_distance(&seg), 3.0, 1e-2));
    assert!(approx_eq(seg.signed_distance(&p), 3.0, 1e-2));
}

// Note: ConvexPolytope × Line/Ray was deferred (requires bounded-segment
// approximation for GJK on unbounded shapes) and is not part of the current
// SDF pair matrix.

// ---------------------------------------------------------------------------
// ConvexPolygon × rest
// ---------------------------------------------------------------------------

#[test]
fn polygon_polygon_separated() {
    let a = unit_square_polygon();
    let b = ConvexPolygon::with_axes(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    assert!(approx_eq(a.signed_distance(&b), 3.0, 1e-2));
}

#[test]
fn polygon_cuboid_above() {
    let poly = unit_square_polygon();
    let cube = Cuboid::from_aabb(Vec3::new(-1.0, 3.0, -1.0), Vec3::new(1.0, 5.0, 1.0));
    assert!(approx_eq(poly.signed_distance(&cube), 3.0, 1e-2));
    assert!(approx_eq(cube.signed_distance(&poly), 3.0, 1e-2));
}

#[test]
fn polygon_capsule_above() {
    let poly = unit_square_polygon();
    let cap = Capsule::new(Vec3::new(-0.5, 5.0, 0.0), Vec3::new(0.5, 5.0, 0.0), 1.0);
    assert!(approx_eq(poly.signed_distance(&cap), 4.0, 1e-2));
}

#[test]
fn polygon_point_above() {
    let poly = unit_square_polygon();
    let pt = Point::new(0.0, 4.0, 0.0);
    assert!(approx_eq(poly.signed_distance(&pt), 4.0, 1e-2));
    assert!(approx_eq(pt.signed_distance(&poly), 4.0, 1e-2));
}

#[test]
fn polygon_segment_above() {
    let poly = unit_square_polygon();
    let seg = LineSegment::new(Vec3::new(-1.0, 3.0, 0.0), Vec3::new(1.0, 3.0, 0.0));
    assert!(approx_eq(poly.signed_distance(&seg), 3.0, 1e-2));
}

#[test]
fn polygon_polytope_above() {
    let poly = unit_square_polygon();
    let cube = unit_cube_polytope();
    // unit_cube_polytope is centered at origin (he=1), polygon is at y=0 inside.
    // So they overlap — sdf should be negative.
    let d = poly.signed_distance(&cube);
    assert!(d < 0.0, "expected overlap, got {d}");
}

// ---------------------------------------------------------------------------
// Pointcloud × rest (remaining pairs)
// ---------------------------------------------------------------------------

#[test]
fn pcl_point_separated() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.25);
    let p = Point::new(5.0, 0.0, 0.0);
    assert!(approx_eq(pcl.signed_distance(&p), 3.75, 1e-4));
    assert!(approx_eq(p.signed_distance(&pcl), 3.75, 1e-4));
}

#[test]
fn pcl_capsule_separated() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0]], 0.25);
    let cap = Capsule::new(Vec3::new(-1.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(pcl.signed_distance(&cap), 3.75, 1e-3));
}

#[test]
fn pcl_cylinder_separated() {
    // Cloud at origin with point radius 0.25. Cylinder starts at (0, 3, 0);
    // closest cylinder point to origin is on the bottom cap at (0, 3, 0)
    // — 3 units away — so SDF = 3 − 0.25 = 2.75.
    let pcl = make_cloud(&[[0.0, 0.0, 0.0]], 0.25);
    let cyl = Cylinder::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(pcl.signed_distance(&cyl), 2.75, 1e-2));
}

#[test]
fn pcl_polytope_separated() {
    let pcl = make_cloud(&[[0.0, 0.0, 0.0]], 0.25);
    let p = {
        let planes = vec![
            (Vec3::X, 6.0),
            (Vec3::NEG_X, -4.0),
            (Vec3::Y, 1.0),
            (Vec3::NEG_Y, 1.0),
            (Vec3::Z, 1.0),
            (Vec3::NEG_Z, 1.0),
        ];
        let verts = vec![
            Vec3::new(4.0, -1.0, -1.0),
            Vec3::new(4.0, -1.0, 1.0),
            Vec3::new(4.0, 1.0, -1.0),
            Vec3::new(4.0, 1.0, 1.0),
            Vec3::new(6.0, -1.0, -1.0),
            Vec3::new(6.0, -1.0, 1.0),
            Vec3::new(6.0, 1.0, -1.0),
            Vec3::new(6.0, 1.0, 1.0),
        ];
        ConvexPolytope::new(planes, verts)
    };
    assert!(approx_eq(pcl.signed_distance(&p), 3.75, 1e-2));
}

#[test]
fn pcl_polygon_above() {
    let pcl = make_cloud(&[[0.0, 5.0, 0.0]], 0.25);
    let poly = unit_square_polygon();
    assert!(approx_eq(pcl.signed_distance(&poly), 4.75, 1e-2));
}

#[test]
fn pcl_line_above() {
    let pcl = make_cloud(&[[0.0, 5.0, 0.0]], 0.25);
    let line = Line::new(Vec3::ZERO, Vec3::X);
    assert!(approx_eq(pcl.signed_distance(&line), 4.75, 1e-3));
}

#[test]
fn pcl_ray_separated() {
    let pcl = make_cloud(&[[0.0, 5.0, 0.0]], 0.25);
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    assert!(approx_eq(pcl.signed_distance(&ray), 4.75, 1e-3));
}

#[test]
fn pcl_segment_separated() {
    let pcl = make_cloud(&[[0.0, 5.0, 0.0]], 0.25);
    let seg = LineSegment::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
    assert!(approx_eq(pcl.signed_distance(&seg), 4.75, 1e-3));
}

// ---------------------------------------------------------------------------
// ArrayConvexPolytope × rest
// ---------------------------------------------------------------------------

#[test]
fn array_polytope_sphere_separated() {
    use wreck::ArrayConvexPolytope;
    let planes = [
        (Vec3::X, 1.0),
        (Vec3::NEG_X, 1.0),
        (Vec3::Y, 1.0),
        (Vec3::NEG_Y, 1.0),
        (Vec3::Z, 1.0),
        (Vec3::NEG_Z, 1.0),
    ];
    let verts = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let obb = Cuboid::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0));
    let p = ArrayConvexPolytope::<6, 8>::new(planes, verts, obb);
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(p.signed_distance(&s), 3.0, 1e-2));
    assert!(approx_eq(s.signed_distance(&p), 3.0, 1e-2));
}

// ---------------------------------------------------------------------------
// ArrayConvexPolygon × rest
// ---------------------------------------------------------------------------

#[test]
fn array_polygon_sphere_above() {
    use wreck::ArrayConvexPolygon;
    let p = ArrayConvexPolygon::<4>::new(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    let s = Sphere::new(Vec3::new(0.0, 5.0, 0.0), 1.0);
    assert!(approx_eq(p.signed_distance(&s), 4.0, 1e-2));
}
