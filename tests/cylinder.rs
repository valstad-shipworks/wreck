use glam::Vec3;
use wreck::{Capsule, Collides, Cuboid, Cylinder, Line, LineSegment, Plane, Point, Ray, Sphere};

fn cyl() -> Cylinder {
    Cylinder::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 4.0), 1.0)
}

#[test]
fn cylinder_sphere_barrel_hit() {
    let c = cyl();
    let s = Sphere::new(Vec3::new(1.5, 0.0, 2.0), 1.0);
    assert!(c.collides(&s));
    assert!(s.collides(&c));
}

#[test]
fn cylinder_sphere_barrel_miss() {
    let c = cyl();
    let s = Sphere::new(Vec3::new(3.0, 0.0, 2.0), 0.5);
    assert!(!c.collides(&s));
}

#[test]
fn cylinder_sphere_endcap_hit() {
    let c = cyl();
    let s = Sphere::new(Vec3::new(0.0, 0.0, 4.5), 1.0);
    assert!(c.collides(&s));
}

#[test]
fn cylinder_sphere_endcap_miss() {
    let c = cyl();
    let s = Sphere::new(Vec3::new(0.0, 0.0, 6.0), 0.5);
    assert!(!c.collides(&s));
}

#[test]
fn cylinder_sphere_rim_hit() {
    let c = cyl();
    let s = Sphere::new(Vec3::new(1.0, 0.0, 4.5), 1.0);
    assert!(c.collides(&s));
}

#[test]
fn cylinder_sphere_rim_miss() {
    let c = cyl();
    let s = Sphere::new(Vec3::new(2.0, 0.0, 5.5), 0.3);
    assert!(!c.collides(&s));
}

#[test]
fn cylinder_point_inside() {
    let c = cyl();
    let p = Point::new(0.5, 0.0, 2.0);
    assert!(c.collides(&p));
    assert!(p.collides(&c));
}

#[test]
fn cylinder_point_outside() {
    let c = cyl();
    let p = Point::new(2.0, 0.0, 2.0);
    assert!(!c.collides(&p));
}

#[test]
fn cylinder_point_past_endcap() {
    let c = cyl();
    let p = Point::new(0.0, 0.0, 5.0);
    assert!(!c.collides(&p));
}

#[test]
fn cylinder_capsule_barrel_hit() {
    let c = cyl();
    let cap = Capsule::new(Vec3::new(2.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 3.0), 1.5);
    assert!(c.collides(&cap));
    assert!(cap.collides(&c));
}

#[test]
fn cylinder_capsule_miss() {
    let c = cyl();
    let cap = Capsule::new(Vec3::new(5.0, 0.0, 1.0), Vec3::new(5.0, 0.0, 3.0), 0.5);
    assert!(!c.collides(&cap));
}

#[test]
fn cylinder_cuboid_axis_through() {
    let c = cyl();
    let cb = Cuboid::new(
        Vec3::new(0.0, 0.0, 2.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.5, 0.5, 0.5],
    );
    assert!(c.collides(&cb));
    assert!(cb.collides(&c));
}

#[test]
fn cylinder_cuboid_corner_inside() {
    let c = cyl();
    let cb = Cuboid::new(
        Vec3::new(0.8, 0.0, 2.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.5, 0.5, 0.5],
    );
    assert!(c.collides(&cb));
}

#[test]
fn cylinder_cuboid_miss() {
    let c = cyl();
    let cb = Cuboid::new(
        Vec3::new(5.0, 0.0, 2.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.5, 0.5, 0.5],
    );
    assert!(!c.collides(&cb));
}

#[test]
fn cylinder_cylinder_parallel_overlap() {
    let c1 = Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 4.0), 1.0);
    let c2 = Cylinder::new(Vec3::new(1.5, 0.0, 0.0), Vec3::new(1.5, 0.0, 4.0), 1.0);
    assert!(c1.collides(&c2));
}

#[test]
fn cylinder_cylinder_separated() {
    let c1 = Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 4.0), 1.0);
    let c2 = Cylinder::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 4.0), 1.0);
    assert!(!c1.collides(&c2));
}

#[test]
fn cylinder_plane_hit() {
    let c = cyl();
    let p = Plane::new(Vec3::Y, 0.5);
    assert!(c.collides(&p));
    assert!(p.collides(&c));
}

#[test]
fn cylinder_plane_miss() {
    let c = cyl();
    let p = Plane::new(Vec3::Y, -2.0);
    assert!(!c.collides(&p));
}

#[test]
fn cylinder_line_through_barrel() {
    let c = cyl();
    let l = Line::new(Vec3::new(-5.0, 0.0, 2.0), Vec3::new(1.0, 0.0, 0.0));
    assert!(c.collides(&l));
    assert!(l.collides(&c));
}

#[test]
fn cylinder_line_miss() {
    let c = cyl();
    let l = Line::new(Vec3::new(5.0, 5.0, 2.0), Vec3::new(0.0, 0.0, 1.0));
    assert!(!c.collides(&l));
}

#[test]
fn cylinder_line_through_endcap() {
    let c = cyl();
    let l = Line::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
    assert!(c.collides(&l));
}

#[test]
fn cylinder_ray_hit() {
    let c = cyl();
    let r = Ray::new(Vec3::new(-5.0, 0.0, 2.0), Vec3::new(1.0, 0.0, 0.0));
    assert!(c.collides(&r));
}

#[test]
fn cylinder_ray_miss_behind() {
    let c = cyl();
    let r = Ray::new(Vec3::new(5.0, 0.0, 2.0), Vec3::new(1.0, 0.0, 0.0));
    assert!(!c.collides(&r));
}

#[test]
fn cylinder_segment_hit() {
    let c = cyl();
    let s = LineSegment::new(Vec3::new(-2.0, 0.0, 2.0), Vec3::new(2.0, 0.0, 2.0));
    assert!(c.collides(&s));
    assert!(s.collides(&c));
}

#[test]
fn cylinder_segment_miss_short() {
    let c = cyl();
    let s = LineSegment::new(Vec3::new(-5.0, 0.0, 2.0), Vec3::new(-3.0, 0.0, 2.0));
    assert!(!c.collides(&s));
}

#[test]
fn cylinder_symmetry() {
    let c = cyl();
    let shapes: Vec<Box<dyn Fn() -> bool>> = vec![
        Box::new(|| {
            let s = Sphere::new(Vec3::new(1.5, 0.0, 2.0), 1.0);
            c.collides(&s) == s.collides(&c)
        }),
        Box::new(|| {
            let p = Point::new(0.5, 0.0, 2.0);
            c.collides(&p) == p.collides(&c)
        }),
        Box::new(|| {
            let cap = Capsule::new(Vec3::new(2.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 3.0), 1.5);
            c.collides(&cap) == cap.collides(&c)
        }),
        Box::new(|| {
            let cb = Cuboid::new(
                Vec3::new(0.0, 0.0, 2.0),
                [Vec3::X, Vec3::Y, Vec3::Z],
                [0.5, 0.5, 0.5],
            );
            c.collides(&cb) == cb.collides(&c)
        }),
    ];
    for check in &shapes {
        assert!(check());
    }
}

#[test]
fn cylinder_bounded() {
    let c = cyl();
    let bp = c.broadphase();
    assert_eq!(bp.center, Vec3::new(0.0, 0.0, 2.0));
    assert!((bp.radius - 3.0).abs() < 0.01);
}

#[test]
fn cylinder_transform() {
    let mut c = cyl();
    c.translate(glam::Vec3A::new(1.0, 0.0, 0.0));
    assert_eq!(c.p1, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(c.p2(), Vec3::new(1.0, 0.0, 4.0));
}

#[test]
fn cylinder_scale() {
    let mut c = cyl();
    c.scale(2.0);
    assert_eq!(c.radius, 2.0);
    assert_eq!(c.dir, Vec3::new(0.0, 0.0, 8.0));
}
