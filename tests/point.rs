use glam::Vec3;
use wreck::{Capsule, Collides, ConvexPolytope, Cuboid, Point, Sphere};

#[test]
fn point_sphere_inside() {
    let p = Point(Vec3::ZERO);
    let s = Sphere::new(Vec3::new(0.5, 0.0, 0.0), 1.0);
    assert!(p.collides(&s));
    assert!(s.collides(&p));
}

#[test]
fn point_sphere_outside() {
    let p = Point(Vec3::new(5.0, 0.0, 0.0));
    let s = Sphere::new(Vec3::ZERO, 1.0);
    assert!(!p.collides(&s));
}

#[test]
fn point_capsule_hit() {
    let p = Point(Vec3::new(1.0, 0.1, 0.0));
    let c = Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.2);
    assert!(p.collides(&c));
    assert!(c.collides(&p));
}

#[test]
fn point_capsule_miss() {
    let p = Point(Vec3::new(1.0, 1.0, 0.0));
    let c = Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.2);
    assert!(!p.collides(&c));
}

#[test]
fn point_cuboid_inside() {
    let p = Point(Vec3::ZERO);
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(p.collides(&c));
    assert!(c.collides(&p));
}

#[test]
fn point_cuboid_outside() {
    let p = Point(Vec3::new(5.0, 0.0, 0.0));
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(!p.collides(&c));
}

#[test]
fn point_cuboid_on_surface() {
    let p = Point(Vec3::new(1.0, 0.0, 0.0));
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(p.collides(&c));
}

#[test]
fn point_polytope_inside() {
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
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ];
    let poly = ConvexPolytope::new(planes, vertices);
    let p = Point(Vec3::ZERO);
    assert!(p.collides(&poly));
    assert!(poly.collides(&p));
}

#[test]
fn point_polytope_outside() {
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
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ];
    let poly = ConvexPolytope::new(planes, vertices);
    let p = Point(Vec3::new(5.0, 0.0, 0.0));
    assert!(!p.collides(&poly));
}
