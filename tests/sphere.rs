use glam::Vec3;
use wreck::{Collides, Sphere};

#[test]
fn sphere_sphere_overlapping() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert!(a.collides(&b));
}

#[test]
fn sphere_sphere_touching() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(2.0, 0.0, 0.0), 1.0);
    assert!(a.collides(&b));
}

#[test]
fn sphere_sphere_separated() {
    let a = Sphere::new(Vec3::ZERO, 1.0);
    let b = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0);
    assert!(!a.collides(&b));
}
