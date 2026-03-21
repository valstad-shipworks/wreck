use glam::Vec3;
use wreck::{Capsule, Collides, ConvexPolytope, Cuboid, Plane, Sphere};

fn ground_plane() -> Plane {
    // Y-up ground plane at y=0: everything with y <= 0 is solid
    Plane::new(Vec3::Y, 0.0)
}

#[test]
fn sphere_above_plane() {
    let plane = ground_plane();
    let s = Sphere::new(Vec3::new(0.0, 2.0, 0.0), 0.5);
    assert!(!plane.collides(&s));
    assert!(!s.collides(&plane));
}

#[test]
fn sphere_touching_plane() {
    let plane = ground_plane();
    let s = Sphere::new(Vec3::new(0.0, 0.5, 0.0), 0.5);
    assert!(plane.collides(&s));
    assert!(s.collides(&plane));
}

#[test]
fn sphere_below_plane() {
    let plane = ground_plane();
    let s = Sphere::new(Vec3::new(0.0, -1.0, 0.0), 0.5);
    assert!(plane.collides(&s));
}

#[test]
fn capsule_above_plane() {
    let plane = ground_plane();
    let c = Capsule::new(
        Vec3::new(-1.0, 2.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
        0.3,
    );
    assert!(!plane.collides(&c));
    assert!(!c.collides(&plane));
}

#[test]
fn capsule_crossing_plane() {
    let plane = ground_plane();
    let c = Capsule::new(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        0.1,
    );
    assert!(plane.collides(&c));
    assert!(c.collides(&plane));
}

#[test]
fn capsule_near_plane() {
    let plane = ground_plane();
    let c = Capsule::new(
        Vec3::new(-1.0, 0.2, 0.0),
        Vec3::new(1.0, 0.2, 0.0),
        0.3,
    );
    assert!(plane.collides(&c));
}

#[test]
fn cuboid_above_plane() {
    let plane = ground_plane();
    let c = Cuboid::new(
        Vec3::new(0.0, 3.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    assert!(!plane.collides(&c));
    assert!(!c.collides(&plane));
}

#[test]
fn cuboid_touching_plane() {
    let plane = ground_plane();
    let c = Cuboid::new(
        Vec3::new(0.0, 1.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    assert!(plane.collides(&c));
    assert!(c.collides(&plane));
}

#[test]
fn cuboid_rotated_touching_plane() {
    let plane = ground_plane();
    // Cuboid rotated 45 degrees around Z, hovering at y=1.5
    let angle = std::f32::consts::FRAC_PI_4;
    let c = angle.cos();
    let s = angle.sin();
    let cub = Cuboid::new(
        Vec3::new(0.0, 1.5, 0.0),
        [Vec3::new(c, s, 0.0), Vec3::new(-s, c, 0.0), Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    // Extent projection onto Y = |s|*1 + |c|*1 + 0*1 ~ 1.414
    // center_proj - extent ~ 1.5 - 1.414 ~ 0.086 > 0 -> no collision
    assert!(!plane.collides(&cub));
}

#[test]
fn polytope_above_plane() {
    let plane = ground_plane();
    let poly = ConvexPolytope::new(
        vec![
            (Vec3::X, 6.0),
            (Vec3::NEG_X, -4.0),
            (Vec3::Y, 6.0),
            (Vec3::NEG_Y, -4.0),
            (Vec3::Z, 6.0),
            (Vec3::NEG_Z, -4.0),
        ],
        vec![
            Vec3::new(4.0, 4.0, 4.0),
            Vec3::new(6.0, 6.0, 6.0),
            Vec3::new(4.0, 4.0, 6.0),
            Vec3::new(6.0, 6.0, 4.0),
            Vec3::new(4.0, 6.0, 4.0),
            Vec3::new(6.0, 4.0, 6.0),
            Vec3::new(4.0, 6.0, 6.0),
            Vec3::new(6.0, 4.0, 4.0),
        ],
    );
    assert!(!plane.collides(&poly));
    assert!(!poly.collides(&plane));
}

#[test]
fn polytope_crossing_plane() {
    let plane = ground_plane();
    let poly = ConvexPolytope::new(
        vec![
            (Vec3::X, 1.0),
            (Vec3::NEG_X, 1.0),
            (Vec3::Y, 1.0),
            (Vec3::NEG_Y, 1.0),
            (Vec3::Z, 1.0),
            (Vec3::NEG_Z, 1.0),
        ],
        vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
        ],
    );
    assert!(plane.collides(&poly));
    assert!(poly.collides(&plane));
}

#[test]
fn plane_plane_perpendicular() {
    let a = Plane::new(Vec3::Y, 0.0);
    let b = Plane::new(Vec3::X, 0.0);
    assert!(a.collides(&b));
}

#[test]
fn plane_plane_parallel_same_dir() {
    let a = Plane::new(Vec3::Y, 0.0);
    let b = Plane::new(Vec3::Y, 5.0);
    assert!(a.collides(&b));
}

#[test]
fn plane_plane_facing_overlapping() {
    // Antiparallel, d1+d2 = 2 >= 0 -> overlap
    let a = Plane::new(Vec3::Y, 1.0);
    let b = Plane::new(Vec3::NEG_Y, 1.0);
    assert!(a.collides(&b));
}

#[test]
fn plane_plane_facing_separated() {
    // Antiparallel, d1+d2 = -2 < 0 -> no overlap
    let a = Plane::new(Vec3::Y, -1.0);
    let b = Plane::new(Vec3::NEG_Y, -1.0);
    assert!(!a.collides(&b));
}

#[test]
fn plane_from_point_normal() {
    let p = Plane::from_point_normal(Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
    assert_eq!(p.d, 5.0);
    // Sphere at y=4 is fully inside the half-space y<=5
    let s = Sphere::new(Vec3::new(0.0, 4.0, 0.0), 0.5);
    assert!(p.collides(&s));
    // Sphere at y=6 with r=0.5: 6-5=1 > 0.5 -> outside
    let s2 = Sphere::new(Vec3::new(0.0, 6.0, 0.0), 0.5);
    assert!(!p.collides(&s2));
    // Sphere at y=5.3 with r=0.5: 5.3-5=0.3 <= 0.5 -> touching
    let s3 = Sphere::new(Vec3::new(0.0, 5.3, 0.0), 0.5);
    assert!(p.collides(&s3));
}

#[test]
fn plane_translate() {
    let mut p = Plane::new(Vec3::Y, 0.0);
    p.translate(Vec3::new(0.0, 3.0, 0.0));
    assert_eq!(p.d, 3.0);
}

#[test]
fn plane_collides_many_spheres() {
    let plane = ground_plane();
    let spheres: Vec<Sphere> = (0..20)
        .map(|i| {
            let y = -5.0 + i as f32 * 0.6;
            Sphere::new(Vec3::new(0.0, y, 0.0), 0.1)
        })
        .collect();

    let simd_result = plane.collides_many(&spheres);
    let scalar_result = spheres.iter().any(|s| plane.collides(s));
    assert_eq!(simd_result, scalar_result);
}

#[test]
fn plane_collides_many_capsules() {
    let plane = ground_plane();
    let capsules: Vec<Capsule> = (0..20)
        .map(|i| {
            let y = -5.0 + i as f32 * 0.6;
            Capsule::new(
                Vec3::new(-1.0, y, 0.0),
                Vec3::new(1.0, y, 0.0),
                0.1,
            )
        })
        .collect();

    let simd_result = plane.collides_many(&capsules);
    let scalar_result = capsules.iter().any(|c| plane.collides(c));
    assert_eq!(simd_result, scalar_result);
}

#[test]
fn plane_collides_many_cuboids() {
    let plane = ground_plane();
    let cuboids: Vec<Cuboid> = (0..20)
        .map(|i| {
            let y = -5.0 + i as f32 * 0.6;
            Cuboid::new(
                Vec3::new(0.0, y, 0.0),
                [Vec3::X, Vec3::Y, Vec3::Z],
                [0.3, 0.3, 0.3],
            )
        })
        .collect();

    let simd_result = plane.collides_many(&cuboids);
    let scalar_result = cuboids.iter().any(|c| plane.collides(c));
    assert_eq!(simd_result, scalar_result);
}
