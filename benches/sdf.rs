#![cfg(feature = "sdf")]

use criterion::{Criterion, criterion_group, criterion_main};
use glam::Vec3;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use std::hint::black_box;

use wreck::sdf::gjk_epa_signed_distance;
use wreck::{
    Capsule, ConvexPolygon, ConvexPolytope, Cuboid, Cylinder, Line, LineSegment, Plane, Point,
    Pointcloud, Ray, SignedDistance, Sphere, capsule_capsule_sdf_batch,
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
    let quat = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
        rng.random_range(0.0..std::f32::consts::TAU),
    );
    Cuboid::new(
        center,
        [quat * Vec3::X, quat * Vec3::Y, quat * Vec3::Z],
        he,
    )
}

fn rand_aabb(rng: &mut SmallRng) -> Cuboid {
    let center = rand_vec3(rng, 5.0);
    Cuboid::new(
        center,
        [Vec3::X, Vec3::Y, Vec3::Z],
        [
            rng.random_range(0.2..1.5),
            rng.random_range(0.2..1.5),
            rng.random_range(0.2..1.5),
        ],
    )
}

const N: usize = 128;

fn generate<T>(seed: u64, mut f: impl FnMut(&mut SmallRng) -> T) -> Vec<T> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..N).map(|_| f(&mut rng)).collect()
}

fn rand_cylinder(rng: &mut SmallRng) -> Cylinder {
    let p1 = rand_vec3(rng, 5.0);
    let p2 = p1 + rand_vec3(rng, 2.0);
    Cylinder::new(p1, p2, rng.random_range(0.1..0.5))
}

fn rand_point(rng: &mut SmallRng) -> Point {
    Point(rand_vec3(rng, 5.0))
}

fn rand_plane(rng: &mut SmallRng) -> Plane {
    let n = rand_vec3(rng, 1.0).normalize_or_zero();
    let n = if n.length_squared() < 0.5 { Vec3::Y } else { n };
    Plane::new(n, rng.random_range(-2.0..2.0))
}

fn rand_line(rng: &mut SmallRng) -> Line {
    Line::new(rand_vec3(rng, 5.0), rand_vec3(rng, 1.0))
}

fn rand_ray(rng: &mut SmallRng) -> Ray {
    Ray::new(rand_vec3(rng, 5.0), rand_vec3(rng, 1.0))
}

fn rand_segment(rng: &mut SmallRng) -> LineSegment {
    let p1 = rand_vec3(rng, 5.0);
    LineSegment::new(p1, p1 + rand_vec3(rng, 2.0))
}

fn rand_polytope(rng: &mut SmallRng) -> ConvexPolytope {
    let center = rand_vec3(rng, 5.0);
    let he = rng.random_range(0.5..1.5);
    let planes = vec![
        (Vec3::X, center.x + he),
        (Vec3::NEG_X, -center.x + he),
        (Vec3::Y, center.y + he),
        (Vec3::NEG_Y, -center.y + he),
        (Vec3::Z, center.z + he),
        (Vec3::NEG_Z, -center.z + he),
    ];
    let verts = vec![
        center + Vec3::new(-he, -he, -he),
        center + Vec3::new(-he, -he, he),
        center + Vec3::new(-he, he, -he),
        center + Vec3::new(-he, he, he),
        center + Vec3::new(he, -he, -he),
        center + Vec3::new(he, -he, he),
        center + Vec3::new(he, he, -he),
        center + Vec3::new(he, he, he),
    ];
    ConvexPolytope::new(planes, verts)
}

fn rand_polygon(rng: &mut SmallRng) -> ConvexPolygon {
    let center = rand_vec3(rng, 5.0);
    let normal = rand_vec3(rng, 1.0).normalize_or_zero();
    let normal = if normal.length_squared() < 0.5 {
        Vec3::Y
    } else {
        normal
    };
    let n_verts = rng.random_range(3..=6usize);
    let radius = rng.random_range(0.5..1.5);
    let verts: Vec<[f32; 2]> = (0..n_verts)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / n_verts as f32;
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect();
    ConvexPolygon::new(center, normal, verts)
}

fn make_pointcloud(seed: u64, n_points: usize) -> Pointcloud {
    let mut rng = SmallRng::seed_from_u64(seed);
    let pts: Vec<[f32; 3]> = (0..n_points)
        .map(|_| {
            let v = rand_vec3(&mut rng, 3.0);
            [v.x, v.y, v.z]
        })
        .collect();
    Pointcloud::new(&pts, (0.1, 0.1), 0.1)
}

fn bench_sphere_sphere(c: &mut Criterion) {
    let a = generate(1, rand_sphere);
    let b = generate(2, rand_sphere);
    c.bench_function("sdf/sphere-sphere/closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/sphere-sphere/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
}

fn bench_sphere_cuboid(c: &mut Criterion) {
    let a = generate(3, rand_sphere);
    let b = generate(4, rand_cuboid);
    c.bench_function("sdf/sphere-cuboid/closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/sphere-cuboid/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
}

fn rand_z_capsule(rng: &mut SmallRng) -> Capsule {
    let p1 = rand_vec3(rng, 5.0);
    let z_len = rng.random_range(-2.0..2.0);
    Capsule::new(
        p1,
        Vec3::new(p1.x, p1.y, p1.z + z_len),
        rng.random_range(0.1..0.5),
    )
}

fn bench_capsule_capsule(c: &mut Criterion) {
    let a = generate(5, rand_capsule);
    let b = generate(6, rand_capsule);
    c.bench_function("sdf/capsule-capsule/closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });

    let za = generate(51, rand_z_capsule);
    let zb = generate(52, rand_z_capsule);
    c.bench_function("sdf/capsule-capsule/z-aligned-closed", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in za.iter().zip(zb.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });

    let mut out = vec![0.0f32; N];
    c.bench_function("sdf/capsule-capsule/z-aligned-batch", |bencher| {
        bencher.iter(|| {
            capsule_capsule_sdf_batch(black_box(&za), black_box(&zb), &mut out);
            out[0]
        })
    });

    c.bench_function("sdf/capsule-capsule/batch-general", |bencher| {
        bencher.iter(|| {
            capsule_capsule_sdf_batch(black_box(&a), black_box(&b), &mut out);
            out[0]
        })
    });
}

fn bench_cuboid_cuboid(c: &mut Criterion) {
    let a = generate(7, rand_cuboid);
    let b = generate(8, rand_cuboid);
    c.bench_function("sdf/cuboid-cuboid/sat", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
    c.bench_function("sdf/cuboid-cuboid/gjk", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                acc += gjk_epa_signed_distance(black_box(x), black_box(y));
            }
            acc
        })
    });
    let aa = generate(9, rand_aabb);
    let bb = generate(10, rand_aabb);
    c.bench_function("sdf/aabb-aabb/sat", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (x, y) in aa.iter().zip(bb.iter()) {
                acc += black_box(x).signed_distance(black_box(y));
            }
            acc
        })
    });
}

fn bench_parametric(c: &mut Criterion) {
    let lines: Vec<Line> = {
        let mut rng = SmallRng::seed_from_u64(11);
        (0..N)
            .map(|_| Line::new(rand_vec3(&mut rng, 5.0), rand_vec3(&mut rng, 1.0)))
            .collect()
    };
    let spheres = generate(12, rand_sphere);
    let cubes = generate(13, rand_cuboid);
    let points: Vec<Point> = {
        let mut rng = SmallRng::seed_from_u64(14);
        (0..N).map(|_| Point(rand_vec3(&mut rng, 5.0))).collect()
    };
    let planes: Vec<Plane> = {
        let mut rng = SmallRng::seed_from_u64(15);
        (0..N)
            .map(|_| {
                let n = rand_vec3(&mut rng, 1.0).normalize_or_zero();
                let n = if n.length_squared() < 0.5 { Vec3::Y } else { n };
                Plane::new(n, rng.random_range(-2.0..2.0))
            })
            .collect()
    };

    c.bench_function("sdf/line-sphere", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, s) in lines.iter().zip(spheres.iter()) {
                acc += black_box(l).signed_distance(black_box(s));
            }
            acc
        })
    });
    c.bench_function("sdf/line-cuboid", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, cube) in lines.iter().zip(cubes.iter()) {
                acc += black_box(l).signed_distance(black_box(cube));
            }
            acc
        })
    });
    c.bench_function("sdf/line-point", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, p) in lines.iter().zip(points.iter()) {
                acc += black_box(l).signed_distance(black_box(p));
            }
            acc
        })
    });
    c.bench_function("sdf/line-plane", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (l, p) in lines.iter().zip(planes.iter()) {
                acc += black_box(l).signed_distance(black_box(p));
            }
            acc
        })
    });

    let rays: Vec<Ray> = {
        let mut rng = SmallRng::seed_from_u64(16);
        (0..N)
            .map(|_| Ray::new(rand_vec3(&mut rng, 5.0), rand_vec3(&mut rng, 1.0)))
            .collect()
    };
    c.bench_function("sdf/ray-sphere", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (r, s) in rays.iter().zip(spheres.iter()) {
                acc += black_box(r).signed_distance(black_box(s));
            }
            acc
        })
    });

    let segs: Vec<LineSegment> = {
        let mut rng = SmallRng::seed_from_u64(17);
        (0..N)
            .map(|_| {
                let p1 = rand_vec3(&mut rng, 5.0);
                LineSegment::new(p1, p1 + rand_vec3(&mut rng, 2.0))
            })
            .collect()
    };
    c.bench_function("sdf/segment-segment", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for (s1, s2) in segs.iter().zip(segs.iter().rev()) {
                acc += black_box(s1).signed_distance(black_box(s2));
            }
            acc
        })
    });
}

/// Helper: bench `a[i].signed_distance(&b[i])` for each i.
macro_rules! pair_bench {
    ($c:expr, $name:literal, $a:expr, $b:expr) => {{
        let a = &$a;
        let b = &$b;
        $c.bench_function($name, |bencher| {
            bencher.iter(|| {
                let mut acc = 0.0f32;
                for (x, y) in a.iter().zip(b.iter()) {
                    acc += black_box(x).signed_distance(black_box(y));
                }
                acc
            })
        });
    }};
}

fn bench_sphere_rest(c: &mut Criterion) {
    let s = generate(100, rand_sphere);
    let caps = generate(101, rand_capsule);
    let plns = generate(102, rand_plane);
    let pts = generate(103, rand_point);
    let cyls = generate(104, rand_cylinder);
    let polys = generate(105, rand_polygon);
    let pts2 = generate(106, rand_polytope);
    pair_bench!(c, "sdf/sphere-capsule", s, caps);
    pair_bench!(c, "sdf/sphere-plane", s, plns);
    pair_bench!(c, "sdf/sphere-point", s, pts);
    pair_bench!(c, "sdf/sphere-cylinder/gjk", s, cyls);
    pair_bench!(c, "sdf/sphere-polygon/gjk", s, polys);
    pair_bench!(c, "sdf/sphere-polytope/gjk", s, pts2);
}

fn bench_capsule_rest(c: &mut Criterion) {
    let caps = generate(110, rand_capsule);
    let cubes = generate(111, rand_cuboid);
    let plns = generate(112, rand_plane);
    let cyls = generate(113, rand_cylinder);
    let polys = generate(114, rand_polygon);
    let pts = generate(115, rand_polytope);
    pair_bench!(c, "sdf/capsule-cuboid", caps, cubes);
    pair_bench!(c, "sdf/capsule-plane", caps, plns);
    pair_bench!(c, "sdf/capsule-cylinder/gjk", caps, cyls);
    pair_bench!(c, "sdf/capsule-polygon/gjk", caps, polys);
    pair_bench!(c, "sdf/capsule-polytope/gjk", caps, pts);
}

fn bench_cuboid_rest(c: &mut Criterion) {
    let cubes = generate(120, rand_cuboid);
    let plns = generate(121, rand_plane);
    let cyls = generate(122, rand_cylinder);
    let polys = generate(123, rand_polygon);
    let pts = generate(124, rand_polytope);
    pair_bench!(c, "sdf/cuboid-plane", cubes, plns);
    pair_bench!(c, "sdf/cuboid-cylinder/gjk", cubes, cyls);
    pair_bench!(c, "sdf/cuboid-polygon/gjk", cubes, polys);
    pair_bench!(c, "sdf/cuboid-polytope/gjk", cubes, pts);
}

fn bench_cylinder_pairs(c: &mut Criterion) {
    let cyls1 = generate(130, rand_cylinder);
    let cyls2 = generate(131, rand_cylinder);
    let pts = generate(132, rand_point);
    let segs = generate(133, rand_segment);
    let polys = generate(134, rand_polygon);
    let polyt = generate(135, rand_polytope);
    pair_bench!(c, "sdf/cylinder-cylinder/gjk", cyls1, cyls2);
    pair_bench!(c, "sdf/cylinder-point/gjk", cyls1, pts);
    pair_bench!(c, "sdf/cylinder-segment/gjk", cyls1, segs);
    pair_bench!(c, "sdf/cylinder-polygon/gjk", cyls1, polys);
    pair_bench!(c, "sdf/cylinder-polytope/gjk", cyls1, polyt);
}

fn bench_polytope_pairs(c: &mut Criterion) {
    let pts1 = generate(140, rand_polytope);
    let pts2 = generate(141, rand_polytope);
    let polys = generate(142, rand_polygon);
    let pts = generate(143, rand_point);
    let segs = generate(144, rand_segment);
    pair_bench!(c, "sdf/polytope-polytope/gjk", pts1, pts2);
    pair_bench!(c, "sdf/polytope-polygon/gjk", pts1, polys);
    pair_bench!(c, "sdf/polytope-point/gjk", pts1, pts);
    pair_bench!(c, "sdf/polytope-segment/gjk", pts1, segs);
}

fn bench_polygon_pairs(c: &mut Criterion) {
    let p1 = generate(150, rand_polygon);
    let p2 = generate(151, rand_polygon);
    let pts = generate(152, rand_point);
    let segs = generate(153, rand_segment);
    pair_bench!(c, "sdf/polygon-polygon/gjk", p1, p2);
    pair_bench!(c, "sdf/polygon-point/gjk", p1, pts);
    pair_bench!(c, "sdf/polygon-segment/gjk", p1, segs);
}

fn bench_ray_rest(c: &mut Criterion) {
    let rays = generate(160, rand_ray);
    let caps = generate(161, rand_capsule);
    let cubes = generate(162, rand_cuboid);
    let plns = generate(163, rand_plane);
    let pts = generate(164, rand_point);
    let segs = generate(165, rand_segment);
    let rays2 = generate(166, rand_ray);
    let lines = generate(167, rand_line);
    pair_bench!(c, "sdf/ray-capsule", rays, caps);
    pair_bench!(c, "sdf/ray-cuboid", rays, cubes);
    pair_bench!(c, "sdf/ray-plane", rays, plns);
    pair_bench!(c, "sdf/ray-point", rays, pts);
    pair_bench!(c, "sdf/ray-segment", rays, segs);
    pair_bench!(c, "sdf/ray-ray", rays, rays2);
    pair_bench!(c, "sdf/ray-line", rays, lines);
}

fn bench_line_rest(c: &mut Criterion) {
    let lines = generate(170, rand_line);
    let lines2 = generate(171, rand_line);
    let caps = generate(172, rand_capsule);
    let segs = generate(173, rand_segment);
    pair_bench!(c, "sdf/line-line", lines, lines2);
    pair_bench!(c, "sdf/line-capsule", lines, caps);
    pair_bench!(c, "sdf/line-segment", lines, segs);
}

fn bench_segment_rest(c: &mut Criterion) {
    let segs = generate(180, rand_segment);
    let sphs = generate(181, rand_sphere);
    let caps = generate(182, rand_capsule);
    let cubes = generate(183, rand_cuboid);
    let plns = generate(184, rand_plane);
    let pts = generate(185, rand_point);
    pair_bench!(c, "sdf/segment-sphere/gjk", segs, sphs);
    pair_bench!(c, "sdf/segment-capsule/gjk", segs, caps);
    pair_bench!(c, "sdf/segment-cuboid/gjk", segs, cubes);
    pair_bench!(c, "sdf/segment-plane", segs, plns);
    pair_bench!(c, "sdf/segment-point/gjk", segs, pts);
}

fn bench_pointcloud_pairs(c: &mut Criterion) {
    const PCL_SIZE: usize = 1024;
    let pcl = make_pointcloud(190, PCL_SIZE);
    let pcl2 = make_pointcloud(191, 64);

    let sphere = Sphere::new(Vec3::new(1.0, 1.0, 1.0), 0.5);
    let point = Point::new(1.0, 1.0, 1.0);
    let plane = Plane::new(Vec3::Y, 0.0);
    let cube = Cuboid::from_aabb(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
    let cap = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.3);
    let cyl = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.5);
    let poly = ConvexPolygon::new(
        Vec3::ZERO,
        Vec3::Y,
        vec![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
    );
    let line = Line::new(Vec3::ZERO, Vec3::X);
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let seg = LineSegment::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

    c.bench_function("sdf/pcl-sphere/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&sphere)))
    });
    c.bench_function("sdf/pcl-point/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&point)))
    });
    c.bench_function("sdf/pcl-plane/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&plane)))
    });
    c.bench_function("sdf/pcl-cuboid/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&cube)))
    });
    c.bench_function("sdf/pcl-capsule/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&cap)))
    });
    c.bench_function("sdf/pcl-cylinder/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&cyl)))
    });
    c.bench_function("sdf/pcl-polygon/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&poly)))
    });
    c.bench_function("sdf/pcl-line/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&line)))
    });
    c.bench_function("sdf/pcl-ray/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&ray)))
    });
    c.bench_function("sdf/pcl-segment/1024pts", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&seg)))
    });
    c.bench_function("sdf/pcl-pcl/1024x64", |bencher| {
        bencher.iter(|| black_box(&pcl).signed_distance(black_box(&pcl2)))
    });
}

criterion_group!(
    sdf_benches,
    bench_sphere_sphere,
    bench_sphere_cuboid,
    bench_capsule_capsule,
    bench_cuboid_cuboid,
    bench_parametric,
    bench_sphere_rest,
    bench_capsule_rest,
    bench_cuboid_rest,
    bench_cylinder_pairs,
    bench_polytope_pairs,
    bench_polygon_pairs,
    bench_ray_rest,
    bench_line_rest,
    bench_segment_rest,
    bench_pointcloud_pairs,
);
criterion_main!(sdf_benches);
