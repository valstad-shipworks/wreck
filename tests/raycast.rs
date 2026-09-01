use glam::{Quat, Vec3};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::f32::consts::PI;
use wreck::{
    ArrayConvexPolygon, ArrayConvexPolytope, Capsule, Collider, Collides, ConvexPolygon,
    ConvexPolytope, Cuboid, Cylinder, Hit, Line, LineSegment, Plane, Point, Pointcloud, Ray,
    Raycast, Sphere,
};

fn approx(a: Vec3, b: Vec3, tol: f32) -> bool {
    (a - b).length() <= tol
}

#[test]
fn ray_sphere_reports_near_surface() {
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let sphere = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    let hit = ray.raycast(&sphere).expect("ray should hit");
    assert!((hit.t - 4.0).abs() < 1e-4, "t = {}", hit.t);
    assert!(approx(hit.point, Vec3::new(4.0, 0.0, 0.0), 1e-4));
    // Same query from the shape's side.
    assert_eq!(sphere.raycast(&ray), Some(hit));
}

#[test]
fn ray_direction_scales_t() {
    let sphere = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    let hit = Ray::new(Vec3::ZERO, Vec3::X * 2.0)
        .raycast(&sphere)
        .expect("ray should hit");
    assert!((hit.t - 2.0).abs() < 1e-4, "t = {}", hit.t);
    assert!(approx(hit.point, Vec3::new(4.0, 0.0, 0.0), 1e-4));
}

#[test]
fn ray_starting_inside_reports_its_origin() {
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let sphere = Sphere::new(Vec3::ZERO, 1.0);
    let hit = ray.raycast(&sphere).expect("origin is inside");
    assert_eq!(hit.t, 0.0);
    assert!(approx(hit.point, Vec3::ZERO, 1e-6));
}

#[test]
fn ray_pointing_away_misses() {
    let ray = Ray::new(Vec3::ZERO, -Vec3::X);
    let sphere = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(ray.raycast(&sphere).is_none());
    // The infinite line through the same origin still finds it.
    let hit = Line::new(Vec3::ZERO, -Vec3::X)
        .raycast(&sphere)
        .expect("line extends both ways");
    assert!((hit.t + 6.0).abs() < 1e-4, "t = {}", hit.t);
}

#[test]
fn segment_stops_at_its_end() {
    let sphere = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(
        LineSegment::new(Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0))
            .raycast(&sphere)
            .is_none()
    );
    let hit = LineSegment::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0))
        .raycast(&sphere)
        .expect("segment reaches the sphere");
    assert!((hit.t - 0.4).abs() < 1e-4, "t = {}", hit.t);
    assert!(approx(hit.point, Vec3::new(4.0, 0.0, 0.0), 1e-4));
}

#[test]
fn ray_cuboid_hits_nearest_face() {
    let cuboid = Cuboid::from_aabb(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
    let hit = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z)
        .raycast(&cuboid)
        .expect("ray should hit");
    assert!(approx(hit.point, Vec3::new(0.0, 0.0, -1.0), 1e-5));
}

#[test]
fn ray_plane_crosses_the_surface() {
    // Half-space y <= 0, entered from above.
    let plane = Plane::new(Vec3::Y, 0.0);
    let hit = Ray::new(Vec3::new(0.0, 4.0, 0.0), Vec3::new(0.0, -2.0, 0.0))
        .raycast(&plane)
        .expect("ray should reach the plane");
    assert!((hit.t - 2.0).abs() < 1e-5, "t = {}", hit.t);
    assert!(approx(hit.point, Vec3::ZERO, 1e-5));

    // Parallel and outside the half-space: never enters.
    assert!(
        Ray::new(Vec3::new(0.0, 4.0, 0.0), Vec3::X)
            .raycast(&plane)
            .is_none()
    );
}

#[test]
fn line_inside_a_half_space_reports_a_finite_point() {
    let plane = Plane::new(Vec3::Y, 0.0);
    // The whole line sits at y = -1, inside the half-space and unbounded both ways.
    let hit = Line::new(Vec3::new(3.0, -1.0, 0.0), Vec3::X)
        .raycast(&plane)
        .expect("the line lies inside");
    assert_eq!(hit.t, 0.0);
    assert!(approx(hit.point, Vec3::new(3.0, -1.0, 0.0), 1e-6));
}

#[test]
fn ray_polygon_pierces_it() {
    let poly = ConvexPolygon::new(
        Vec3::ZERO,
        Vec3::Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    let hit = Ray::new(Vec3::new(0.2, 0.3, -5.0), Vec3::Z)
        .raycast(&poly)
        .expect("ray should pierce the polygon");
    assert!(approx(hit.point, Vec3::new(0.2, 0.3, 0.0), 1e-4));

    // Through the polygon's plane but outside its boundary.
    assert!(
        Ray::new(Vec3::new(5.0, 0.0, -5.0), Vec3::Z)
            .raycast(&poly)
            .is_none()
    );
}

#[test]
fn ray_capsule_hits_the_cap() {
    let capsule = Capsule::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 6.0), 1.0);
    let hit = Ray::new(Vec3::ZERO, Vec3::Z)
        .raycast(&capsule)
        .expect("ray should hit the lower cap");
    assert!((hit.t - 1.0).abs() < 1e-4, "t = {}", hit.t);
}

#[test]
fn ray_cylinder_hits_the_end_cap() {
    let cylinder = Cylinder::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 6.0), 1.0);
    let hit = Ray::new(Vec3::ZERO, Vec3::Z)
        .raycast(&cylinder)
        .expect("ray should hit the end cap");
    assert!((hit.t - 2.0).abs() < 1e-4, "t = {}", hit.t);
}

#[test]
fn array_polytope_matches_its_heap_form() {
    const UNIT_BOX: ArrayConvexPolytope<6, 8> = ArrayConvexPolytope::new(
        [
            (Vec3::X, 1.0),
            (Vec3::NEG_X, 1.0),
            (Vec3::Y, 1.0),
            (Vec3::NEG_Y, 1.0),
            (Vec3::Z, 1.0),
            (Vec3::NEG_Z, 1.0),
        ],
        [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ],
        Cuboid::from_aabb(Vec3::splat(-1.0), Vec3::splat(1.0)),
    );

    let ray = Ray::new(Vec3::new(-5.0, 0.25, 0.25), Vec3::X);
    let hit = ray.raycast(&UNIT_BOX).expect("ray should hit");
    assert!(approx(hit.point, Vec3::new(-1.0, 0.25, 0.25), 1e-4));
    assert_eq!(UNIT_BOX.raycast(&ray), Some(hit));

    let heap = ConvexPolytope::from(UNIT_BOX);
    assert_eq!(ray.raycast(&heap).map(|h| h.t), Some(hit.t));
}

#[test]
fn array_polygon_matches_its_heap_form() {
    const QUAD: ArrayConvexPolygon<4> = ArrayConvexPolygon::new(
        Vec3::ZERO,
        Vec3::Z,
        Vec3::X,
        Vec3::Y,
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );

    let ray = Ray::new(Vec3::new(0.4, -0.2, -3.0), Vec3::Z);
    let hit = ray.raycast(&QUAD).expect("ray should pierce the quad");
    assert!(approx(hit.point, Vec3::new(0.4, -0.2, 0.0), 1e-4));
    assert_eq!(QUAD.raycast(&ray), Some(hit));

    let heap = ConvexPolygon::new(
        Vec3::ZERO,
        Vec3::Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    assert_eq!(ray.raycast(&heap).map(|h| h.t), Some(hit.t));
}

#[test]
fn collider_reports_the_nearest_shape() {
    let mut collider: Collider = Collider::new();
    collider.add(Sphere::new(Vec3::new(9.0, 0.0, 0.0), 1.0));
    collider.add(Sphere::new(Vec3::new(4.0, 0.0, 0.0), 1.0));
    collider.add(Cuboid::from_aabb(
        Vec3::new(19.0, -1.0, -1.0),
        Vec3::new(21.0, 1.0, 1.0),
    ));

    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let hit = ray.raycast(&collider).expect("ray should hit");
    assert!((hit.t - 3.0).abs() < 1e-4, "t = {}", hit.t);
    assert_eq!(collider.raycast(&ray), Some(hit));

    // Facing away, only the infinite line can reach anything.
    let away = Ray::new(Vec3::ZERO, -Vec3::X);
    assert!(away.raycast(&collider).is_none());
}

#[test]
fn collider_includes_pointclouds() {
    let cloud = Pointcloud::new(&[[6.0, 0.0, 0.0], [12.0, 0.0, 0.0]], (0.0, 4.0), 0.5);
    let mut collider: Collider = Collider::new();
    collider.add(cloud);

    let hit = Ray::new(Vec3::ZERO, Vec3::X)
        .raycast(&collider)
        .expect("ray should hit the near point");
    assert!((hit.t - 5.5).abs() < 1e-4, "t = {}", hit.t);
}

#[test]
fn empty_collider_never_hits() {
    let collider: Collider = Collider::new();
    assert!(Ray::new(Vec3::ZERO, Vec3::X).raycast(&collider).is_none());
}

/// Checks a reported hit against the shape it came from: the hit is on the line, on or inside
/// the shape, agrees with the boolean narrowphase, and nothing earlier along the line was
/// already inside.
///
/// `bounded` says whether the shape has finite extent. Unbounded shapes can contain the whole
/// negative half of an infinite line, in which case there is nothing to be "first".
#[allow(clippy::too_many_arguments)]
fn verify<S>(
    shape: &S,
    origin: Vec3,
    dir: Vec3,
    t_min: f32,
    t_max: f32,
    collides: bool,
    hit: Option<Hit>,
    bounded: bool,
    label: &str,
) where
    S: Collides<Sphere> + Collides<Point>,
{
    assert_eq!(
        hit.is_some(),
        collides,
        "{label}: raycast and collides disagree"
    );
    let Some(hit) = hit else { return };

    assert!(
        hit.t >= t_min && hit.t <= t_max,
        "{label}: t = {} outside [{t_min}, {t_max}]",
        hit.t
    );
    assert!(hit.t.is_finite(), "{label}: t = {}", hit.t);
    // A line nearly parallel to a plane crosses it a long way out, where f32 has only a few
    // decimals left, so what counts as "on the shape" has to grow with the distance travelled.
    let tol = 1e-3 * (1.0 + hit.point.length());
    assert!(
        approx(hit.point, origin + dir * hit.t, tol),
        "{label}: point is not on the line"
    );
    assert!(
        shape.collides(&Sphere::new(hit.point, tol)),
        "{label}: hit point is not on the shape (t = {})",
        hit.t
    );

    let check_earlier = if t_min.is_finite() {
        hit.t > t_min
    } else {
        bounded
    };
    if check_earlier {
        for frac in [1e-3f32, 1e-2, 1e-1, 1.0] {
            let t = hit.t - frac * (1.0 + hit.t.abs());
            if t < t_min {
                continue;
            }
            let p = origin + dir * t;
            assert!(
                !shape.collides(&Point::new(p.x, p.y, p.z)),
                "{label}: already inside at t = {t}, before the reported {}",
                hit.t
            );
        }
    }
}

macro_rules! check_shape {
    ($shape:expr, $origin:expr, $dir:expr, $bounded:expr, $label:expr) => {{
        let shape = $shape;
        let (o, d) = ($origin, $dir);

        let line = Line::new(o, d);
        verify(
            &shape,
            o,
            d,
            f32::NEG_INFINITY,
            f32::INFINITY,
            line.collides(&shape),
            line.raycast(&shape),
            $bounded,
            &format!("{} / line", $label),
        );

        let ray = Ray::new(o, d);
        verify(
            &shape,
            o,
            d,
            0.0,
            f32::INFINITY,
            ray.collides(&shape),
            ray.raycast(&shape),
            $bounded,
            &format!("{} / ray", $label),
        );

        let segment = LineSegment::new(o, o + d);
        verify(
            &shape,
            o,
            d,
            0.0,
            1.0,
            segment.collides(&shape),
            segment.raycast(&shape),
            $bounded,
            &format!("{} / segment", $label),
        );

        // Both directions of the pair are the same query.
        assert_eq!(shape.raycast(&line), line.raycast(&shape));
        assert_eq!(shape.raycast(&ray), ray.raycast(&shape));
        assert_eq!(shape.raycast(&segment), segment.raycast(&shape));

        line.raycast(&shape).is_some()
    }};
}

fn rand_vec(rng: &mut SmallRng, range: f32) -> Vec3 {
    Vec3::new(
        rng.random_range(-range..range),
        rng.random_range(-range..range),
        rng.random_range(-range..range),
    )
}

fn rand_quat(rng: &mut SmallRng) -> Quat {
    Quat::from_euler(
        glam::EulerRot::XYZ,
        rng.random_range(-PI..PI),
        rng.random_range(-PI..PI),
        rng.random_range(-PI..PI),
    )
}

/// A line aimed near the origin, so a shape sitting there is hit about half the time.
fn rand_probe(rng: &mut SmallRng) -> (Vec3, Vec3) {
    let origin = rand_vec(rng, 6.0);
    let target = rand_vec(rng, 2.5);
    let dir = (target - origin) * rng.random_range(0.2..2.0);
    (origin, dir)
}

fn box_polytope(rng: &mut SmallRng) -> ConvexPolytope {
    let center = rand_vec(rng, 1.5);
    let he = Vec3::new(
        rng.random_range(0.3..1.5),
        rng.random_range(0.3..1.5),
        rng.random_range(0.3..1.5),
    );
    let q = rand_quat(rng);
    let axes = [q * Vec3::X, q * Vec3::Y, q * Vec3::Z];

    let mut planes = Vec::with_capacity(6);
    for (axis, h) in axes.into_iter().zip([he.x, he.y, he.z]) {
        planes.push((axis, axis.dot(center) + h));
        planes.push((-axis, -axis.dot(center) + h));
    }

    let mut vertices = Vec::with_capacity(8);
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                vertices.push(
                    center + axes[0] * (sx * he.x) + axes[1] * (sy * he.y) + axes[2] * (sz * he.z),
                );
            }
        }
    }
    ConvexPolytope::new(planes, vertices)
}

/// A cube with its eight corners cut off: fourteen planes and twenty-four vertices, enough to
/// run the polytope clip past its first step and leave it a short tail.
fn beveled_polytope(rng: &mut SmallRng) -> ConvexPolytope {
    let center = rand_vec(rng, 1.5);
    let h = rng.random_range(0.4..1.5);
    let bevel = h * 0.3;
    let q = rand_quat(rng);
    let axes = [q * Vec3::X, q * Vec3::Y, q * Vec3::Z];
    let local = |v: Vec3| center + axes[0] * v.x + axes[1] * v.y + axes[2] * v.z;

    let mut planes = Vec::with_capacity(14);
    for axis in axes {
        planes.push((axis, axis.dot(center) + h));
        planes.push((-axis, -axis.dot(center) + h));
    }
    // The corner plane meets each of the corner's three edges `bevel * sqrt(3)` along it.
    let cut = bevel * 3.0f32.sqrt();
    let mut vertices = Vec::with_capacity(24);
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let n = (axes[0] * sx + axes[1] * sy + axes[2] * sz) / 3.0f32.sqrt();
                planes.push((n, n.dot(center) + h * 3.0f32.sqrt() - bevel));
                let corner = Vec3::new(sx, sy, sz) * h;
                vertices.push(local(corner - Vec3::new(sx * cut, 0.0, 0.0)));
                vertices.push(local(corner - Vec3::new(0.0, sy * cut, 0.0)));
                vertices.push(local(corner - Vec3::new(0.0, 0.0, sz * cut)));
            }
        }
    }
    ConvexPolytope::new(planes, vertices)
}

fn rand_polygon(rng: &mut SmallRng) -> ConvexPolygon {
    let q = rand_quat(rng);
    let half = rng.random_range(0.5..2.0);
    ConvexPolygon::new(
        rand_vec(rng, 1.5),
        q * Vec3::Z,
        vec![[-half, -half], [half, -half], [half, half], [-half, half]],
    )
}

#[test]
fn fuzz_matches_narrowphase_and_reports_the_first_point() {
    let mut rng = SmallRng::seed_from_u64(0x9EC7);
    let mut hits = [0u32; 8];

    for _ in 0..20_000 {
        let (o, d) = rand_probe(&mut rng);

        let sphere = Sphere::new(rand_vec(&mut rng, 1.5), rng.random_range(0.2..2.0));
        hits[0] += check_shape!(sphere, o, d, true, "sphere") as u32;

        let capsule = Capsule::new(
            rand_vec(&mut rng, 1.5),
            rand_vec(&mut rng, 1.5),
            rng.random_range(0.2..1.5),
        );
        hits[1] += check_shape!(capsule, o, d, true, "capsule") as u32;

        let cylinder = Cylinder::new(
            rand_vec(&mut rng, 1.5),
            rand_vec(&mut rng, 1.5),
            rng.random_range(0.2..1.5),
        );
        hits[2] += check_shape!(cylinder, o, d, true, "cylinder") as u32;

        let q = rand_quat(&mut rng);
        let cuboid = Cuboid::new(
            rand_vec(&mut rng, 1.5),
            [q * Vec3::X, q * Vec3::Y, q * Vec3::Z],
            [
                rng.random_range(0.3..1.5),
                rng.random_range(0.3..1.5),
                rng.random_range(0.3..1.5),
            ],
        );
        hits[3] += check_shape!(cuboid, o, d, true, "cuboid") as u32;

        let polytope = box_polytope(&mut rng);
        hits[4] += check_shape!(polytope, o, d, true, "polytope") as u32;

        let beveled = beveled_polytope(&mut rng);
        hits[7] += check_shape!(beveled, o, d, true, "beveled polytope") as u32;

        let polygon = rand_polygon(&mut rng);
        hits[5] += check_shape!(polygon, o, d, true, "polygon") as u32;

        let plane =
            Plane::from_point_normal(rand_vec(&mut rng, 1.5), rand_quat(&mut rng) * Vec3::Z);
        hits[6] += check_shape!(plane, o, d, false, "plane") as u32;
    }

    for (i, count) in hits.iter().enumerate() {
        assert!(*count > 500, "shape {i} was hit only {count} times");
    }
}

#[test]
fn fuzz_collider_returns_the_nearest_of_its_shapes() {
    let mut rng = SmallRng::seed_from_u64(0x11CE);
    let mut hits = 0u32;

    for iter in 0..5_000 {
        let spheres: Vec<Sphere> = (0..rng.random_range(0..6))
            .map(|_| Sphere::new(rand_vec(&mut rng, 3.0), rng.random_range(0.2..1.2)))
            .collect();
        let cuboids: Vec<Cuboid> = (0..rng.random_range(0..4))
            .map(|_| {
                Cuboid::from_aabb(
                    rand_vec(&mut rng, 3.0) - Vec3::splat(0.6),
                    rand_vec(&mut rng, 3.0) + Vec3::splat(0.6),
                )
            })
            .collect();
        let capsules: Vec<Capsule> = (0..rng.random_range(0..3))
            .map(|_| {
                Capsule::new(
                    rand_vec(&mut rng, 3.0),
                    rand_vec(&mut rng, 3.0),
                    rng.random_range(0.2..1.0),
                )
            })
            .collect();

        let mut collider: Collider = Collider::new();
        for s in &spheres {
            collider.add(*s);
        }
        for c in &cuboids {
            collider.add(*c);
        }
        for c in &capsules {
            collider.add(*c);
        }

        let (o, d) = rand_probe(&mut rng);
        let ray = Ray::new(o, d);

        let want = spheres
            .iter()
            .filter_map(|s| ray.raycast(s))
            .chain(cuboids.iter().filter_map(|c| ray.raycast(c)))
            .chain(capsules.iter().filter_map(|c| ray.raycast(c)))
            .map(|h| h.t)
            .fold(f32::INFINITY, f32::min);

        match ray.raycast(&collider) {
            Some(hit) => {
                assert!(
                    (hit.t - want).abs() < 1e-4,
                    "iter {iter}: collider reported t = {}, nearest shape is at {want}",
                    hit.t
                );
                hits += 1;
            }
            None => assert!(
                want.is_infinite(),
                "iter {iter}: collider missed but a shape is at t = {want}"
            ),
        }
    }

    assert!(hits > 1000, "too few positive cases ({hits})");
}

#[test]
fn fuzz_pointcloud_matches_a_scalar_scan() {
    let mut rng = SmallRng::seed_from_u64(0xC10D);
    let mut hits = 0u32;

    for iter in 0..2_000 {
        let count = rng.random_range(1..40);
        let points: Vec<[f32; 3]> = (0..count)
            .map(|_| {
                let p = rand_vec(&mut rng, 3.0);
                [p.x, p.y, p.z]
            })
            .collect();
        let radius = rng.random_range(0.1..0.8);
        let cloud = Pointcloud::new(&points, (0.0, 4.0), radius);

        let (o, d) = rand_probe(&mut rng);
        let ray = Ray::new(o, d);

        let want = points
            .iter()
            .filter_map(|p| ray.raycast(&Sphere::new(Vec3::from(*p), radius)))
            .map(|h| h.t)
            .fold(f32::INFINITY, f32::min);

        match ray.raycast(&cloud) {
            Some(hit) => {
                assert!(
                    (hit.t - want).abs() < 1e-4,
                    "iter {iter}: cloud reported t = {}, nearest point is at {want}",
                    hit.t
                );
                hits += 1;
            }
            None => assert!(
                want.is_infinite(),
                "iter {iter}: cloud missed but a point is at t = {want}"
            ),
        }
    }

    assert!(hits > 200, "too few positive cases ({hits})");
}
