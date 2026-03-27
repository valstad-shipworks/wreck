use glam::Vec3;
use wreck::{Bounded, Collider, Collides, Sphere};

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

#[test]
fn refine_bounding_opposed_spheres() {
    let mut c = Collider::<wreck::Pointcloud>::new();
    c.add(Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 0.5));
    c.add(Sphere::new(Vec3::new(5.0, 0.0, 0.0), 0.5));
    let loose = c.broadphase().radius;
    c.refine_bounding();
    let tight = c.broadphase().radius;
    assert!(tight <= loose, "refined {tight} should be <= loose {loose}");
    assert!(tight < 5.5 + 0.1, "refined {tight} should be near optimal 5.5");
}

#[test]
fn refine_bounding_cluster() {
    let mut c = Collider::<wreck::Pointcloud>::new();
    for x in [-1.0, 0.0, 1.0] {
        for y in [-1.0, 0.0, 1.0] {
            c.add(Sphere::new(Vec3::new(x, y, 0.0), 0.2));
        }
    }
    let loose = c.broadphase().radius;
    c.refine_bounding();
    let tight = c.broadphase().radius;
    assert!(tight <= loose, "refined {tight} should be <= loose {loose}");
}

#[test]
fn refine_bounding_empty() {
    let mut c = Collider::<wreck::Pointcloud>::new();
    c.refine_bounding();
    assert_eq!(c.broadphase().radius, 0.0);
}

#[test]
fn refine_bounding_single() {
    let mut c = Collider::<wreck::Pointcloud>::new();
    c.add(Sphere::new(Vec3::new(3.0, 4.0, 0.0), 2.0));
    c.refine_bounding();
    let b = c.broadphase();
    assert!((b.center - Vec3::new(3.0, 4.0, 0.0)).length() < 0.01);
    assert!((b.radius - 2.0).abs() < 0.01);
}

#[test]
fn refine_bounding_encloses_all() {
    let mut c = Collider::<wreck::Pointcloud>::new();
    let spheres = vec![
        Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 0.5),
        Sphere::new(Vec3::new(5.0, 0.0, 0.0), 0.5),
        Sphere::new(Vec3::new(0.0, 3.0, 0.0), 1.0),
        Sphere::new(Vec3::new(0.0, 0.0, -4.0), 0.3),
    ];
    for s in &spheres {
        c.add(*s);
    }
    c.refine_bounding();
    let b = c.broadphase();
    for s in &spheres {
        let dist = (s.center - b.center).length() + s.radius;
        assert!(
            dist <= b.radius + 1e-5,
            "sphere at {:?} r={} not enclosed: dist={dist} > bounding={}",
            s.center, s.radius, b.radius
        );
    }
}
