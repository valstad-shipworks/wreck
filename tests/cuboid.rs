use glam::Vec3;
use wreck::{Capsule, Collides, Cuboid, Sphere};

fn axis_aligned(center: Vec3, half: [f32; 3]) -> Cuboid {
    Cuboid::new(center, [Vec3::X, Vec3::Y, Vec3::Z], half)
}

#[test]
fn sphere_cuboid_inside() {
    let s = Sphere::new(Vec3::ZERO, 0.1);
    let c = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    assert!(s.collides(&c));
    assert!(c.collides(&s));
}

#[test]
fn sphere_cuboid_separated() {
    let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5);
    let c = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    assert!(!s.collides(&c));
}

#[test]
fn sphere_cuboid_corner() {
    // Sphere near corner of cuboid
    let c = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let s = Sphere::new(Vec3::new(1.5, 1.5, 1.5), 1.0);
    // Distance from (1.5,1.5,1.5) to corner (1,1,1) = sqrt(0.75) ~ 0.866
    assert!(s.collides(&c));

    let s2 = Sphere::new(Vec3::new(1.5, 1.5, 1.5), 0.5);
    assert!(!s2.collides(&c));
}

#[test]
fn capsule_cuboid_through() {
    let cap = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.1);
    let cub = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    assert!(cap.collides(&cub));
    assert!(cub.collides(&cap));
}

#[test]
fn capsule_cuboid_separated() {
    let cap = Capsule::new(Vec3::new(-2.0, 3.0, 0.0), Vec3::new(2.0, 3.0, 0.0), 0.1);
    let cub = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    assert!(!cap.collides(&cub));
}

#[test]
fn cuboid_cuboid_overlapping() {
    let a = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let b = axis_aligned(Vec3::new(1.5, 0.0, 0.0), [1.0, 1.0, 1.0]);
    assert!(a.collides(&b));
}

#[test]
fn cuboid_cuboid_separated() {
    let a = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let b = axis_aligned(Vec3::new(3.0, 0.0, 0.0), [1.0, 1.0, 1.0]);
    assert!(!a.collides(&b));
}

#[test]
fn cuboid_cuboid_rotated() {
    let a = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    // Rotated 45 degrees around Z
    let angle = std::f32::consts::FRAC_PI_4;
    let c = angle.cos();
    let s = angle.sin();
    let b = Cuboid::new(
        Vec3::new(2.0, 0.0, 0.0),
        [Vec3::new(c, s, 0.0), Vec3::new(-s, c, 0.0), Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    // Gap between them should be ~0.586, rotated box corner reaches closer
    assert!(a.collides(&b));
}

#[test]
fn cuboid_cuboid_touching() {
    let a = axis_aligned(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let b = axis_aligned(Vec3::new(2.0, 0.0, 0.0), [1.0, 1.0, 1.0]);
    assert!(a.collides(&b));
}
