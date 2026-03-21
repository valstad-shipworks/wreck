use glam::Vec3;
use wreck::{Capsule, Collides, ConvexPolytope, Cuboid, Plane, Pointcloud, Sphere};

fn test_cloud() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
}

#[test]
fn sphere_pcl_hit() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    let s = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 0.1);
    assert!(pcl.collides(&s));
    assert!(s.collides(&pcl));
}

#[test]
fn sphere_pcl_miss() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    let s = Sphere::new(Vec3::new(5.0, 5.0, 5.0), 0.1);
    assert!(!pcl.collides(&s));
}

#[test]
fn capsule_pcl_hit() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    let c = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.2);
    assert!(pcl.collides(&c));
    assert!(c.collides(&pcl));
}

#[test]
fn capsule_pcl_miss() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    let c = Capsule::new(Vec3::new(-1.0, 5.0, 0.0), Vec3::new(2.0, 5.0, 0.0), 0.2);
    assert!(!pcl.collides(&c));
}

#[test]
fn cuboid_pcl_hit() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    let c = Cuboid::new(
        Vec3::new(0.5, 0.5, 0.5),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.6, 0.6, 0.6],
    );
    assert!(pcl.collides(&c));
    assert!(c.collides(&pcl));
}

#[test]
fn cuboid_pcl_miss() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    let c = Cuboid::new(
        Vec3::new(5.0, 5.0, 5.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [0.5, 0.5, 0.5],
    );
    assert!(!pcl.collides(&c));
}

#[test]
fn pcl_with_point_radius() {
    let pts = vec![[0.0, 0.0, 0.0]];
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.5);
    let s = Sphere::new(Vec3::new(0.4, 0.0, 0.0), 0.0);
    assert!(pcl.collides(&s));

    let s_far = Sphere::new(Vec3::new(0.6, 0.0, 0.0), 0.0);
    assert!(!pcl.collides(&s_far));
}

#[test]
fn sphere_pcl_collides_many_simd() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
    let pts: Vec<[f32; 3]> = (0..200)
        .map(|_| {
            [
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);

    // 20 spheres -> exercises both the 8-wide SIMD path and the scalar remainder
    let spheres: Vec<Sphere> = (0..20)
        .map(|_| {
            Sphere::new(
                Vec3::new(
                    rng.random_range(-6.0..6.0),
                    rng.random_range(-6.0..6.0),
                    rng.random_range(-6.0..6.0),
                ),
                rng.random_range(0.1..1.0),
            )
        })
        .collect();

    let simd_result = pcl.collides_many(&spheres);
    let scalar_result = spheres.iter().any(|s| pcl.collides(s));
    assert_eq!(simd_result, scalar_result);
}

#[test]
fn capsule_pcl_large_cloud() {
    // Test SIMD path with >8 points
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let pts: Vec<[f32; 3]> = (0..100)
        .map(|_| {
            [
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);

    let cap = Capsule::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 0.5);
    let simd_result = pcl.collides(&cap);

    // Brute-force check
    let brute = pts.iter().any(|pt| {
        let p = Vec3::from(*pt);
        let closest = cap.closest_point_to(p);
        let d = p - closest;
        d.dot(d) <= 0.5 * 0.5
    });
    assert_eq!(simd_result, brute);
}

#[test]
fn polytope_pcl_hit() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    // Unit cube centered at origin as ConvexPolytope
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
    assert!(pcl.collides(&poly));
    assert!(poly.collides(&pcl));
}

#[test]
fn polytope_pcl_miss() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    // Cube far away
    let poly = ConvexPolytope::new(
        vec![
            (Vec3::X, 11.0),
            (Vec3::NEG_X, -9.0),
            (Vec3::Y, 11.0),
            (Vec3::NEG_Y, -9.0),
            (Vec3::Z, 11.0),
            (Vec3::NEG_Z, -9.0),
        ],
        vec![
            Vec3::new(9.0, 9.0, 9.0),
            Vec3::new(11.0, 11.0, 11.0),
            Vec3::new(9.0, 9.0, 11.0),
            Vec3::new(11.0, 11.0, 9.0),
            Vec3::new(9.0, 11.0, 9.0),
            Vec3::new(11.0, 9.0, 11.0),
            Vec3::new(9.0, 11.0, 11.0),
            Vec3::new(11.0, 9.0, 9.0),
        ],
    );
    assert!(!pcl.collides(&poly));
}

#[test]
fn polytope_pcl_large_cloud() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let pts: Vec<[f32; 3]> = (0..100)
        .map(|_| {
            [
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);

    let poly = ConvexPolytope::new(
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
    let simd_result = pcl.collides(&poly);

    // Brute-force: point inside if n.p <= d for all planes
    let brute = pts.iter().any(|pt| {
        let p = Vec3::from(*pt);
        poly.planes.iter().all(|&(n, d)| n.dot(p) <= d)
    });
    assert_eq!(simd_result, brute);
}

#[test]
fn cuboid_pcl_large_cloud() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let pts: Vec<[f32; 3]> = (0..100)
        .map(|_| {
            [
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);

    let cub = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [0.5, 0.5, 0.5]);
    let simd_result = pcl.collides(&cub);

    let brute = pts.iter().any(|pt| {
        let p = Vec3::from(*pt);
        // point_dist_sq is pub(crate), so use the Collides trait instead
        let point = wreck::Point(p);
        point.collides(&cub)
    });
    assert_eq!(simd_result, brute);
}

#[test]
fn plane_pcl_hit() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    // Ground plane at y=0.5: half-space y <= 0.5
    let plane = Plane::new(Vec3::Y, 0.5);
    assert!(pcl.collides(&plane));
    assert!(plane.collides(&pcl));
}

#[test]
fn plane_pcl_miss() {
    let pts = test_cloud();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.0);
    // Plane y <= -1: no point has y <= -1
    let plane = Plane::new(Vec3::Y, -1.0);
    assert!(!pcl.collides(&plane));
    assert!(!plane.collides(&pcl));
}

#[test]
fn plane_pcl_with_radius() {
    let pts = vec![[0.0, 1.0, 0.0]];
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.5);
    // Plane y <= 0.4: point at y=1.0, radius 0.5 -> 1.0 <= 0.4+0.5=0.9 -> no
    let plane = Plane::new(Vec3::Y, 0.4);
    assert!(!pcl.collides(&plane));
    // Plane y <= 0.5: point at y=1.0, radius 0.5 -> 1.0 <= 0.5+0.5=1.0 -> yes
    let plane2 = Plane::new(Vec3::Y, 0.5);
    assert!(pcl.collides(&plane2));
}

#[test]
fn plane_pcl_large_cloud() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let pts: Vec<[f32; 3]> = (0..100)
        .map(|_| {
            [
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();
    let pcl = Pointcloud::new(&pts, (0.0, 2.0), 0.1);

    let plane = Plane::new(Vec3::Y, -4.5);
    let simd_result = pcl.collides(&plane);

    let brute = pts.iter().any(|pt| {
        let p = Vec3::from(*pt);
        plane.normal.dot(p) <= plane.d + 0.1
    });
    assert_eq!(simd_result, brute);
}
