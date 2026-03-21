use glam::Vec3;
use wreck::{Capsule, Collides, Sphere};

#[test]
fn sphere_capsule_overlapping() {
    let s = Sphere::new(Vec3::new(0.5, 0.5, 0.0), 0.5);
    let c = Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.3);
    assert!(s.collides(&c));
    assert!(c.collides(&s));
}

#[test]
fn sphere_capsule_separated() {
    let s = Sphere::new(Vec3::new(0.0, 2.0, 0.0), 0.3);
    let c = Capsule::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), 0.3);
    assert!(!s.collides(&c));
}

#[test]
fn capsule_capsule_crossing() {
    let a = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.2);
    let b = Capsule::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.2);
    assert!(a.collides(&b));
}

#[test]
fn capsule_capsule_parallel_separated() {
    let a = Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.1);
    let b = Capsule::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0), 0.1);
    assert!(!a.collides(&b));
}

#[test]
fn capsule_capsule_touching() {
    let a = Capsule::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.5);
    let b = Capsule::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0), 0.5);
    assert!(a.collides(&b));
}

#[test]
fn degenerate_capsule_is_sphere() {
    let s = Sphere::new(Vec3::ZERO, 1.0);
    let c = Capsule::new(Vec3::ZERO, Vec3::ZERO, 1.0);
    let target = Sphere::new(Vec3::new(1.5, 0.0, 0.0), 1.0);
    assert_eq!(s.collides(&target), c.collides(&target));
}
