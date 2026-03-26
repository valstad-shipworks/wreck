use glam::Vec3;
use wreck::{Capsule, Collides, ConvexPolygon, ConvexPolytope, Cuboid, Plane, Sphere};

fn unit_square() -> ConvexPolygon {
    // 2x2 square on the XZ plane at y=0, normal = Y
    ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    )
}

#[test]
fn sphere_polygon_hit_above() {
    let poly = unit_square();
    // Sphere directly above center, close enough
    let s = Sphere::new(Vec3::new(0.0, 0.3, 0.0), 0.5);
    assert!(poly.collides(&s));
    assert!(s.collides(&poly));
}

#[test]
fn sphere_polygon_miss_above() {
    let poly = unit_square();
    let s = Sphere::new(Vec3::new(0.0, 2.0, 0.0), 0.5);
    assert!(!poly.collides(&s));
}

#[test]
fn sphere_polygon_miss_lateral() {
    let poly = unit_square();
    // Sphere at same height but far from polygon boundary
    let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5);
    assert!(!poly.collides(&s));
}

#[test]
fn sphere_polygon_hit_edge() {
    let poly = unit_square();
    // Sphere near edge
    let s = Sphere::new(Vec3::new(1.2, 0.0, 0.0), 0.5);
    assert!(poly.collides(&s));
}

#[test]
fn sphere_polygon_hit_corner() {
    let poly = unit_square();
    // Sphere near corner (1, 0, -1) in world = vertex (1, 1) in 2D
    let s = Sphere::new(Vec3::new(1.1, 0.1, -1.1), 0.3);
    // Distance to corner = sqrt(0.1^2 + 0.1^2 + 0.1^2) ~ 0.173
    assert!(poly.collides(&s));
}

#[test]
fn capsule_polygon_through() {
    let poly = unit_square();
    // Capsule passing through the polygon
    let c = Capsule::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 0.1);
    assert!(poly.collides(&c));
    assert!(c.collides(&poly));
}

#[test]
fn capsule_polygon_parallel_above() {
    let poly = unit_square();
    // Capsule parallel to polygon, too far above
    let c = Capsule::new(Vec3::new(-1.0, 2.0, 0.0), Vec3::new(1.0, 2.0, 0.0), 0.1);
    assert!(!poly.collides(&c));
}

#[test]
fn capsule_polygon_near_edge() {
    let poly = unit_square();
    // Capsule near edge but within radius
    let c = Capsule::new(Vec3::new(-0.5, 0.1, 0.0), Vec3::new(0.5, 0.1, 0.0), 0.2);
    assert!(poly.collides(&c));
}

#[test]
fn cuboid_polygon_overlapping() {
    let poly = unit_square();
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [0.5, 0.5, 0.5]);
    assert!(poly.collides(&c));
    assert!(c.collides(&poly));
}

#[test]
fn cuboid_polygon_above() {
    let poly = unit_square();
    let c = Cuboid::new(
        Vec3::new(0.0, 3.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    assert!(!poly.collides(&c));
}

#[test]
fn cuboid_polygon_touching() {
    let poly = unit_square();
    let c = Cuboid::new(
        Vec3::new(0.0, 1.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.5, 1.0, 0.5],
    );
    assert!(poly.collides(&c));
}

#[test]
fn cuboid_polygon_lateral_miss() {
    let poly = unit_square();
    // Cuboid beside the polygon, straddling the plane but not overlapping in 2D
    let c = Cuboid::new(
        Vec3::new(5.0, 0.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.5, 0.5, 0.5],
    );
    assert!(!poly.collides(&c));
}

#[test]
fn polytope_polygon_overlapping() {
    let poly = unit_square();
    let cube = ConvexPolytope::new(
        vec![
            (Vec3::X, 0.5),
            (Vec3::NEG_X, 0.5),
            (Vec3::Y, 0.5),
            (Vec3::NEG_Y, 0.5),
            (Vec3::Z, 0.5),
            (Vec3::NEG_Z, 0.5),
        ],
        vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(0.5, -0.5, -0.5),
        ],
    );
    assert!(poly.collides(&cube));
    assert!(cube.collides(&poly));
}

#[test]
fn polytope_polygon_miss() {
    let poly = unit_square();
    let cube = ConvexPolytope::new(
        vec![
            (Vec3::X, 5.5),
            (Vec3::NEG_X, -4.5),
            (Vec3::Y, 5.5),
            (Vec3::NEG_Y, -4.5),
            (Vec3::Z, 5.5),
            (Vec3::NEG_Z, -4.5),
        ],
        vec![Vec3::new(4.5, 4.5, 4.5), Vec3::new(5.5, 5.5, 5.5)],
    );
    assert!(!poly.collides(&cube));
}

#[test]
fn infinite_plane_polygon() {
    let poly = unit_square();
    // Infinite plane y <= -0.5: polygon is at y=0, so no collision
    let ip = Plane::new(Vec3::Y, -0.5);
    assert!(!poly.collides(&ip));
    assert!(!ip.collides(&poly));

    // Infinite plane y <= 0.5: polygon at y=0 is inside
    let ip2 = Plane::new(Vec3::Y, 0.5);
    assert!(poly.collides(&ip2));
    assert!(ip2.collides(&poly));
}

#[test]
fn polygon_polygon_coplanar_overlapping() {
    let a = unit_square();
    let b = ConvexPolygon::with_axes(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    assert!(a.collides(&b));
}

#[test]
fn polygon_polygon_coplanar_separated() {
    let a = unit_square();
    let b = ConvexPolygon::with_axes(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    assert!(!a.collides(&b));
}

#[test]
fn polygon_polygon_perpendicular_crossing() {
    let a = unit_square(); // XZ plane at y=0
    // Polygon on XY plane at z=0
    let b = ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Z,
        Vec3::X,
        Vec3::Y,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    assert!(a.collides(&b));
}

#[test]
fn polygon_polygon_parallel_separated() {
    let a = unit_square();
    // Same orientation but offset in Y
    let b = ConvexPolygon::with_axes(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    assert!(!a.collides(&b));
}

#[test]
fn polygon_translate() {
    let mut poly = unit_square();
    poly.translate(Vec3::new(0.0, 5.0, 0.0));
    assert_eq!(poly.center, Vec3::new(0.0, 5.0, 0.0));
    // Sphere at origin should miss now
    let s = Sphere::new(Vec3::ZERO, 0.5);
    assert!(!poly.collides(&s));
}

#[test]
fn polygon_scale() {
    let mut poly = unit_square();
    poly.scale(2.0);
    // Should now extend from -2 to 2
    let s = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 0.1);
    assert!(poly.collides(&s));
}

#[test]
fn polygon_cw_winding_corrected() {
    // Provide CW vertices -- should auto-correct to CCW
    let poly = ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
    );
    // Should still work correctly
    let s = Sphere::new(Vec3::new(0.0, 0.3, 0.0), 0.5);
    assert!(poly.collides(&s));

    let s2 = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5);
    assert!(!poly.collides(&s2));
}

#[test]
fn triangle_polygon() {
    let tri = ConvexPolygon::new(
        Vec3::ZERO,
        Vec3::Y,
        vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
    );
    // Sphere above the triangle center
    let s = Sphere::new(Vec3::new(0.0, 0.3, 0.0), 0.5);
    assert!(tri.collides(&s));
}
