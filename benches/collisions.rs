use criterion::{Criterion, criterion_group, criterion_main};
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use std::hint::black_box;

use wreck::{
    Capsule, Collides, ConvexPolygon, Cuboid, Line, LineSegment, Plane, Point, Ray, Sphere,
};

fn rand_vec3(rng: &mut SmallRng, range: f32) -> Vec3 {
    Vec3::new(
        rng.random_range(-range..range),
        rng.random_range(-range..range),
        rng.random_range(-range..range),
    )
}

fn rand_sphere(rng: &mut SmallRng) -> Sphere {
    Sphere::new(rand_vec3(rng, 5.0), rng.random_range(0.1..1.0))
}

fn rand_capsule(rng: &mut SmallRng) -> Capsule {
    let p1 = rand_vec3(rng, 5.0);
    let p2 = p1 + rand_vec3(rng, 2.0);
    Capsule::new(p1, p2, rng.random_range(0.1..0.5))
}

fn rand_cuboid(rng: &mut SmallRng) -> Cuboid {
    let center = rand_vec3(rng, 5.0);
    let he = [
        rng.random_range(0.2..1.5),
        rng.random_range(0.2..1.5),
        rng.random_range(0.2..1.5),
    ];
    // Random rotation via quaternion
    let quat = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
    );
    let axes = [quat * Vec3::X, quat * Vec3::Y, quat * Vec3::Z];
    Cuboid::new(center, axes, he)
}

fn rand_z_capsule(rng: &mut SmallRng) -> Capsule {
    let p1 = rand_vec3(rng, 5.0);
    let z_len = rng.random_range(-2.0..2.0);
    let p2 = p1 + Vec3::new(0.0, 0.0, z_len);
    Capsule::new(p1, p2, rng.random_range(0.1..0.5))
}

fn rand_aabb(rng: &mut SmallRng) -> Cuboid {
    let center = rand_vec3(rng, 5.0);
    let he = [
        rng.random_range(0.2..1.5),
        rng.random_range(0.2..1.5),
        rng.random_range(0.2..1.5),
    ];
    Cuboid::new(center, [Vec3::X, Vec3::Y, Vec3::Z], he)
}

fn rand_convex_polygon(rng: &mut SmallRng) -> ConvexPolygon {
    let center = rand_vec3(rng, 5.0);
    let normal = rand_vec3(rng, 1.0).normalize_or_zero();
    let normal = if normal.length_squared() < 0.5 {
        Vec3::Y
    } else {
        normal
    };
    // Vary vertex count: 3-8 sided regular polygons with random perturbation
    let n_verts = rng.random_range(3..=8usize);
    let radius = rng.random_range(0.5..2.0);
    let verts: Vec<[f32; 2]> = (0..n_verts)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / n_verts as f32;
            let r = radius * rng.random_range(0.7..1.0);
            [r * angle.cos(), r * angle.sin()]
        })
        .collect();
    ConvexPolygon::new(center, normal, verts)
}

fn rand_infinite_plane(rng: &mut SmallRng) -> Plane {
    let normal = rand_vec3(rng, 1.0).normalize_or_zero();
    let normal = if normal.length_squared() < 0.5 {
        Vec3::Y
    } else {
        normal
    };
    let d = rng.random_range(-3.0..3.0);
    Plane::new(normal, d)
}

fn rand_line(rng: &mut SmallRng) -> Line {
    Line::new(rand_vec3(rng, 5.0), rand_vec3(rng, 2.0))
}

fn rand_ray(rng: &mut SmallRng) -> Ray {
    Ray::new(rand_vec3(rng, 5.0), rand_vec3(rng, 2.0))
}

fn rand_line_segment(rng: &mut SmallRng) -> LineSegment {
    let p1 = rand_vec3(rng, 5.0);
    let p2 = p1 + rand_vec3(rng, 2.0);
    LineSegment::new(p1, p2)
}

/// Generate a convex polytope with many faces: a randomly rotated beveled cube
/// (6 face planes + 12 edge-bevel planes + 8 corner-bevel planes = 26 planes, 24 vertices).
fn rand_polytope(rng: &mut SmallRng) -> wreck::ConvexPolytope {
    let center = rand_vec3(rng, 3.0);
    let quat = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
    );
    let half = rng.random_range(0.5..1.5);
    let bevel = half * 0.3; // bevel amount

    let axes = [quat * Vec3::X, quat * Vec3::Y, quat * Vec3::Z];
    let mut planes = Vec::with_capacity(26);

    // 6 face planes
    for i in 0..3 {
        planes.push((axes[i], axes[i].dot(center) + half));
        planes.push((-axes[i], (-axes[i]).dot(center) + half));
    }

    // 12 edge-bevel planes (pairs of axes)
    let edge_dirs = [(0, 1), (0, 2), (1, 2)];
    for &(a, b) in &edge_dirs {
        for sa in [-1.0f32, 1.0] {
            for sb in [-1.0f32, 1.0] {
                let n = (axes[a] * sa + axes[b] * sb).normalize();
                let d = n.dot(center) + half - bevel;
                planes.push((n, d));
            }
        }
    }

    // 8 corner-bevel planes
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let n = (axes[0] * sx + axes[1] * sy + axes[2] * sz).normalize();
                let d = n.dot(center) + half - bevel * 1.5;
                planes.push((n, d));
            }
        }
    }

    // Generate vertices by intersecting triplets of face+bevel planes.
    // Easier: sample vertices on the beveled shape.
    let cut = half - bevel;
    let mut vertices = Vec::with_capacity(24);
    // Face vertices (6 faces × 4 verts each, but shared at edges)
    for &sa in &[-1.0f32, 1.0] {
        for &sb in &[-1.0f32, 1.0] {
            // Vertices on the ±X faces (at the bevel cut)
            vertices.push(center + axes[0] * half * sa + axes[1] * cut * sb + axes[2] * cut);
            vertices.push(center + axes[0] * half * sa + axes[1] * cut * sb - axes[2] * cut);
            // Vertices on the ±Y faces
            vertices.push(center + axes[0] * cut * sa + axes[1] * half * sb + axes[2] * cut);
            vertices.push(center + axes[0] * cut * sa + axes[1] * half * sb - axes[2] * cut);
            // Vertices on the ±Z faces
            vertices.push(center + axes[0] * cut * sa + axes[1] * cut * sb + axes[2] * half);
            vertices.push(center + axes[0] * cut * sa + axes[1] * cut * sb - axes[2] * half);
        }
    }

    wreck::ConvexPolytope::new(planes, vertices)
}

/// Generate a random pointcloud with clustered structure for realism.
fn rand_pointcloud(rng: &mut SmallRng, n_points: usize) -> wreck::Pointcloud {
    let mut pts = Vec::with_capacity(n_points);

    // Generate several clusters of points
    let n_clusters = 5 + rng.random_range(0..5usize);
    let pts_per_cluster = n_points / n_clusters;

    for _ in 0..n_clusters {
        let cluster_center: [f32; 3] = [
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ];
        let spread = rng.random_range(0.5..2.0);
        for _ in 0..pts_per_cluster {
            pts.push([
                cluster_center[0] + rng.random_range(-spread..spread),
                cluster_center[1] + rng.random_range(-spread..spread),
                cluster_center[2] + rng.random_range(-spread..spread),
            ]);
        }
    }

    // Fill remainder with uniform random
    while pts.len() < n_points {
        pts.push([
            rng.random_range(-6.0..6.0),
            rng.random_range(-6.0..6.0),
            rng.random_range(-6.0..6.0),
        ]);
    }

    wreck::Pointcloud::new(&pts, (0.0, 3.0), 0.02)
}

const N_PAIRS: usize = 256;

fn bench_sphere_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_sphere(&mut rng), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("sphere_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (a, b_s) in &pairs {
                if black_box(a).collides(black_box(b_s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_sphere_capsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_sphere(&mut rng), rand_capsule(&mut rng)))
        .collect();

    c.bench_function("sphere_capsule", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (s, cap) in &pairs {
                if black_box(s).collides(black_box(cap)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_sphere_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_sphere(&mut rng), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("sphere_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (s, cub) in &pairs {
                if black_box(s).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_capsule_capsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_capsule(&mut rng), rand_capsule(&mut rng)))
        .collect();

    c.bench_function("capsule_capsule", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (a, b_c) in &pairs {
                if black_box(a).collides(black_box(b_c)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_capsule_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_capsule(&mut rng), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("capsule_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (cap, cub) in &pairs {
                if black_box(cap).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_cuboid_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_cuboid(&mut rng), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("cuboid_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (a, b_c) in &pairs {
                if black_box(a).collides(black_box(b_c)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_batch_sphere_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let query = rand_sphere(&mut rng);
    let targets: Vec<_> = (0..1024).map(|_| rand_sphere(&mut rng)).collect();

    c.bench_function("batch_sphere_sphere_1024", |b| {
        b.iter(|| black_box(&query).collides_many(black_box(&targets)))
    });
}

fn bench_batch_sphere_capsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let query = rand_sphere(&mut rng);
    let targets: Vec<_> = (0..1024).map(|_| rand_capsule(&mut rng)).collect();

    c.bench_function("batch_sphere_capsule_1024", |b| {
        b.iter(|| black_box(&query).collides_many(black_box(&targets)))
    });
}

fn bench_batch_sphere_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let query = rand_sphere(&mut rng);
    let targets: Vec<_> = (0..1024).map(|_| rand_cuboid(&mut rng)).collect();

    c.bench_function("batch_sphere_cuboid_1024", |b| {
        b.iter(|| black_box(&query).collides_many(black_box(&targets)))
    });
}

fn bench_sphere_polytope(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let polytopes: Vec<_> = (0..N_PAIRS).map(|_| rand_polytope(&mut rng)).collect();
    let spheres: Vec<_> = (0..N_PAIRS).map(|_| rand_sphere(&mut rng)).collect();

    c.bench_function("sphere_polytope_26p", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (s, p) in spheres.iter().zip(polytopes.iter()) {
                if black_box(s).collides(black_box(p)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_cuboid_polytope(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let polytopes: Vec<_> = (0..N_PAIRS).map(|_| rand_polytope(&mut rng)).collect();
    let cuboids: Vec<_> = (0..N_PAIRS).map(|_| rand_cuboid(&mut rng)).collect();

    c.bench_function("cuboid_polytope_26p", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (cub, p) in cuboids.iter().zip(polytopes.iter()) {
                if black_box(cub).collides(black_box(p)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_sphere_pcl(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pcl = rand_pointcloud(&mut rng, 50_000);
    let spheres: Vec<_> = (0..N_PAIRS).map(|_| rand_sphere(&mut rng)).collect();

    c.bench_function("sphere_pcl_50k", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for s in &spheres {
                if pcl.collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_capsule_pcl(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pcl = rand_pointcloud(&mut rng, 50_000);
    let capsules: Vec<_> = (0..N_PAIRS).map(|_| rand_capsule(&mut rng)).collect();

    c.bench_function("capsule_pcl_50k", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for cap in &capsules {
                if pcl.collides(black_box(cap)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_cuboid_pcl(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pcl = rand_pointcloud(&mut rng, 50_000);
    let cuboids: Vec<_> = (0..N_PAIRS).map(|_| rand_cuboid(&mut rng)).collect();

    c.bench_function("cuboid_pcl_50k", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for cub in &cuboids {
                if pcl.collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_polygon_pcl(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pcl = rand_pointcloud(&mut rng, 50_000);
    let polygons: Vec<_> = (0..N_PAIRS)
        .map(|_| rand_convex_polygon(&mut rng))
        .collect();

    c.bench_function("polygon_pcl_50k", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for poly in &polygons {
                if pcl.collides(black_box(poly)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_plane_pcl(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pcl = rand_pointcloud(&mut rng, 50_000);
    let planes: Vec<_> = (0..N_PAIRS)
        .map(|_| rand_infinite_plane(&mut rng))
        .collect();

    c.bench_function("plane_pcl_50k", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for plane in &planes {
                if pcl.collides(black_box(plane)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_sphere_pcl_small(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pcl = rand_pointcloud(&mut rng, 2_500);
    let spheres: Vec<_> = (0..N_PAIRS).map(|_| rand_sphere(&mut rng)).collect();

    c.bench_function("sphere_pcl_2500", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for s in &spheres {
                if pcl.collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

// Axis-aligned benchmarks
fn bench_sphere_aabb(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_sphere(&mut rng), rand_aabb(&mut rng)))
        .collect();

    c.bench_function("sphere_aabb", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (s, cub) in &pairs {
                if black_box(s).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_capsule_aabb(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_capsule(&mut rng), rand_aabb(&mut rng)))
        .collect();

    c.bench_function("capsule_aabb", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (cap, cub) in &pairs {
                if black_box(cap).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_zcapsule_aabb(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_z_capsule(&mut rng), rand_aabb(&mut rng)))
        .collect();

    c.bench_function("zcapsule_aabb", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (cap, cub) in &pairs {
                if black_box(cap).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_aabb_aabb(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_aabb(&mut rng), rand_aabb(&mut rng)))
        .collect();

    c.bench_function("aabb_aabb", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (a, b_c) in &pairs {
                if black_box(a).collides(black_box(b_c)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_sphere_zcapsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_sphere(&mut rng), rand_z_capsule(&mut rng)))
        .collect();

    c.bench_function("sphere_zcapsule", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (s, cap) in &pairs {
                if black_box(s).collides(black_box(cap)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_point_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (Point(rand_vec3(&mut rng, 5.0)), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("point_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (p, s) in &pairs {
                if black_box(p).collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_point_capsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (Point(rand_vec3(&mut rng, 5.0)), rand_capsule(&mut rng)))
        .collect();

    c.bench_function("point_capsule", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (p, cap) in &pairs {
                if black_box(p).collides(black_box(cap)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_point_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (Point(rand_vec3(&mut rng, 5.0)), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("point_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (p, cub) in &pairs {
                if black_box(p).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_point_aabb(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (Point(rand_vec3(&mut rng, 5.0)), rand_aabb(&mut rng)))
        .collect();

    c.bench_function("point_aabb", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (p, cub) in &pairs {
                if black_box(p).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_point_polytope(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let polytopes: Vec<_> = (0..N_PAIRS).map(|_| rand_polytope(&mut rng)).collect();
    let points: Vec<_> = (0..N_PAIRS)
        .map(|_| Point(rand_vec3(&mut rng, 5.0)))
        .collect();

    c.bench_function("point_polytope_26p", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (p, poly) in points.iter().zip(polytopes.iter()) {
                if black_box(p).collides(black_box(poly)) {
                    count += 1;
                }
            }
            count
        })
    });
}

// Stretch benchmarks (require convex-polytope)

fn bench_stretch_sphere(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_sphere(&mut rng), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_sphere", |b| {
        b.iter(|| {
            for (s, t) in &pairs {
                black_box(black_box(s).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_capsule_aligned(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| {
            let cap = rand_capsule(&mut rng);
            let t = cap.dir.normalize_or_zero() * rng.random_range(0.5..2.0);
            (cap, t)
        })
        .collect();

    c.bench_function("stretch_capsule_aligned", |b| {
        b.iter(|| {
            for (cap, t) in &pairs {
                black_box(black_box(cap).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_capsule_unaligned(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| {
            let cap = rand_capsule(&mut rng);
            // Translation perpendicular to capsule dir
            let perp = if cap.dir.length_squared() > f32::EPSILON {
                cap.dir.cross(Vec3::Y).normalize_or_zero()
            } else {
                Vec3::X
            };
            let t = perp * rng.random_range(0.5..2.0);
            (cap, t)
        })
        .collect();

    c.bench_function("stretch_capsule_unaligned", |b| {
        b.iter(|| {
            for (cap, t) in &pairs {
                black_box(black_box(cap).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_cuboid_aligned(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| {
            let cub = rand_cuboid(&mut rng);
            let axis_idx = rng.random_range(0..3usize);
            let t = cub.axes[axis_idx] * rng.random_range(0.5..2.0);
            (cub, t)
        })
        .collect();

    c.bench_function("stretch_cuboid_aligned", |b| {
        b.iter(|| {
            for (cub, t) in &pairs {
                black_box(black_box(cub).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_cuboid_unaligned(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| {
            let cub = rand_cuboid(&mut rng);
            let t = rand_vec3(&mut rng, 2.0);
            (cub, t)
        })
        .collect();

    c.bench_function("stretch_cuboid_unaligned", |b| {
        b.iter(|| {
            for (cub, t) in &pairs {
                black_box(black_box(cub).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_polytope(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| {
            let poly = rand_polytope(&mut rng);
            let t = rand_vec3(&mut rng, 2.0);
            (poly, t)
        })
        .collect();

    c.bench_function("stretch_polytope_26p", |b| {
        b.iter(|| {
            for (poly, t) in &pairs {
                black_box(black_box(poly).stretch(black_box(*t)));
            }
        })
    });
}

// --- ConvexPolygon collision benchmarks ---

fn bench_polygon_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_convex_polygon(&mut rng), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("polygon_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (poly, s) in &pairs {
                if black_box(poly).collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_polygon_capsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_convex_polygon(&mut rng), rand_capsule(&mut rng)))
        .collect();

    c.bench_function("polygon_capsule", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (poly, cap) in &pairs {
                if black_box(poly).collides(black_box(cap)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_polygon_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_convex_polygon(&mut rng), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("polygon_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (poly, cub) in &pairs {
                if black_box(poly).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_polygon_polygon(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_convex_polygon(&mut rng), rand_convex_polygon(&mut rng)))
        .collect();

    c.bench_function("polygon_polygon", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (a, b_p) in &pairs {
                if black_box(a).collides(black_box(b_p)) {
                    count += 1;
                }
            }
            count
        })
    });
}

// --- InfinitePlane collision benchmarks ---

fn bench_plane_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_infinite_plane(&mut rng), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("plane_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (plane, s) in &pairs {
                if black_box(plane).collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_plane_capsule(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_infinite_plane(&mut rng), rand_capsule(&mut rng)))
        .collect();

    c.bench_function("plane_capsule", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (plane, cap) in &pairs {
                if black_box(plane).collides(black_box(cap)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_plane_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_infinite_plane(&mut rng), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("plane_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (plane, cub) in &pairs {
                if black_box(plane).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

// --- Line/Ray/LineSegment collision benchmarks ---

fn bench_line_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_line(&mut rng), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("line_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (l, s) in &pairs {
                if black_box(l).collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_ray_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_ray(&mut rng), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("ray_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (r, s) in &pairs {
                if black_box(r).collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_segment_sphere(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_line_segment(&mut rng), rand_sphere(&mut rng)))
        .collect();

    c.bench_function("segment_sphere", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (seg, s) in &pairs {
                if black_box(seg).collides(black_box(s)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_ray_cuboid(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_ray(&mut rng), rand_cuboid(&mut rng)))
        .collect();

    c.bench_function("ray_cuboid", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (r, cub) in &pairs {
                if black_box(r).collides(black_box(cub)) {
                    count += 1;
                }
            }
            count
        })
    });
}

fn bench_segment_polygon(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_line_segment(&mut rng), rand_convex_polygon(&mut rng)))
        .collect();

    c.bench_function("segment_polygon", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for (seg, poly) in &pairs {
                if black_box(seg).collides(black_box(poly)) {
                    count += 1;
                }
            }
            count
        })
    });
}

// --- New stretch benchmarks ---

fn bench_stretch_point(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (Point(rand_vec3(&mut rng, 5.0)), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_point", |b| {
        b.iter(|| {
            for (p, t) in &pairs {
                black_box(black_box(p).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_line_segment(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_line_segment(&mut rng), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_line_segment", |b| {
        b.iter(|| {
            for (seg, t) in &pairs {
                black_box(black_box(seg).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_ray(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_ray(&mut rng), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_ray", |b| {
        b.iter(|| {
            for (r, t) in &pairs {
                black_box(black_box(r).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_line(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_line(&mut rng), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_line", |b| {
        b.iter(|| {
            for (l, t) in &pairs {
                black_box(black_box(l).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_polygon(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_convex_polygon(&mut rng), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_polygon", |b| {
        b.iter(|| {
            for (poly, t) in &pairs {
                black_box(black_box(poly).stretch(black_box(*t)));
            }
        })
    });
}

fn bench_stretch_infinite_plane(c: &mut Criterion) {
    use wreck::Stretchable;

    let mut rng = SmallRng::seed_from_u64(42);
    let pairs: Vec<_> = (0..N_PAIRS)
        .map(|_| (rand_infinite_plane(&mut rng), rand_vec3(&mut rng, 2.0)))
        .collect();

    c.bench_function("stretch_infinite_plane", |b| {
        b.iter(|| {
            for (plane, t) in &pairs {
                black_box(black_box(plane).stretch(black_box(*t)));
            }
        })
    });
}

// Primitive collision benchmarks (always available)
criterion_group!(
    primitive_benches,
    bench_sphere_sphere,
    bench_sphere_capsule,
    bench_sphere_cuboid,
    bench_capsule_capsule,
    bench_capsule_cuboid,
    bench_cuboid_cuboid,
    bench_batch_sphere_sphere,
    bench_batch_sphere_capsule,
    bench_batch_sphere_cuboid,
    bench_sphere_aabb,
    bench_capsule_aabb,
    bench_zcapsule_aabb,
    bench_aabb_aabb,
    bench_sphere_zcapsule,
    bench_point_sphere,
    bench_point_capsule,
    bench_point_cuboid,
    bench_point_aabb,
    bench_polygon_sphere,
    bench_polygon_capsule,
    bench_polygon_cuboid,
    bench_polygon_polygon,
    bench_plane_sphere,
    bench_plane_capsule,
    bench_plane_cuboid,
    bench_line_sphere,
    bench_ray_sphere,
    bench_segment_sphere,
    bench_ray_cuboid,
    bench_segment_polygon,
);

// Polytope benchmarks (behind feature flag)
criterion_group!(
    polytope_benches,
    bench_sphere_polytope,
    bench_cuboid_polytope,
    bench_point_polytope,
    bench_stretch_sphere,
    bench_stretch_capsule_aligned,
    bench_stretch_capsule_unaligned,
    bench_stretch_cuboid_aligned,
    bench_stretch_cuboid_unaligned,
    bench_stretch_polytope,
    bench_stretch_point,
    bench_stretch_line_segment,
    bench_stretch_ray,
    bench_stretch_line,
    bench_stretch_polygon,
    bench_stretch_infinite_plane,
);

// Pointcloud benchmarks
criterion_group!(
    pcl_benches,
    bench_sphere_pcl,
    bench_capsule_pcl,
    bench_cuboid_pcl,
    bench_polygon_pcl,
    bench_plane_pcl,
    bench_sphere_pcl_small,
);

criterion_main!(primitive_benches, polytope_benches, pcl_benches);
