use glam::Vec3;
use wreck::{
    Capsule, Collides, ConvexPolygon, ConvexPolytope, Cuboid, Line, LineSegment, Plane, Ray,
    Scalable, Sphere, Transformable,
};

// ---------------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------------

#[test]
fn line_sphere_hit() {
    let l = Line::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::new(5.0, 0.5, 0.0), 1.0);
    assert!(l.collides(&s));
    assert!(s.collides(&l));
}

#[test]
fn line_sphere_miss() {
    let l = Line::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::new(0.0, 3.0, 0.0), 1.0);
    assert!(!l.collides(&s));
}

#[test]
fn line_sphere_behind_origin() {
    // Line extends infinitely, so it should hit even "behind" origin
    let l = Line::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 1.0);
    assert!(l.collides(&s));
}

#[test]
fn line_capsule_hit() {
    let l = Line::new(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(-1.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0), 0.5);
    assert!(l.collides(&c));
}

#[test]
fn line_capsule_miss() {
    let l = Line::new(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(3.0, 2.0, 0.0), 0.5);
    assert!(!l.collides(&c));
}

#[test]
fn line_cuboid_hit() {
    let l = Line::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(l.collides(&c));
}

#[test]
fn line_cuboid_miss() {
    let l = Line::new(Vec3::new(5.0, 0.0, -5.0), Vec3::Z);
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(!l.collides(&c));
}

#[test]
fn line_polytope_hit() {
    let l = Line::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
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
        ],
    );
    assert!(l.collides(&poly));
}

#[test]
fn line_infinite_plane() {
    let l = Line::new(Vec3::new(0.0, 5.0, 0.0), Vec3::X);
    // Half-space y <= 0: line at y=5 parallel and outside
    let ip = Plane::new(Vec3::Y, 0.0);
    assert!(!l.collides(&ip));

    // Non-parallel line always enters the half-space
    let l2 = Line::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, -1.0, 0.0));
    assert!(l2.collides(&ip));
}

#[test]
fn line_convex_polygon() {
    let poly = ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    // Line through center of polygon
    let l = Line::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
    assert!(l.collides(&poly));

    // Line parallel, offset
    let l2 = Line::new(Vec3::new(0.0, 1.0, 0.0), Vec3::X);
    assert!(!l2.collides(&poly));

    // Line through polygon edge
    let l3 = Line::new(Vec3::new(0.5, -5.0, 0.0), Vec3::Y);
    assert!(l3.collides(&poly));
}

#[test]
fn line_collides_many_spheres() {
    let l = Line::new(Vec3::ZERO, Vec3::X);
    let spheres: Vec<Sphere> = (0..20)
        .map(|i| Sphere::new(Vec3::new(0.0, i as f32 * 0.5, 0.0), 0.1))
        .collect();

    let simd_result = spheres.iter().any(|x| l.collides(x));
    let scalar_result = spheres.iter().any(|s| l.collides(s));
    assert_eq!(simd_result, scalar_result);
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

#[test]
fn ray_sphere_hit_forward() {
    let r = Ray::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(r.collides(&s));
    assert!(s.collides(&r));
}

#[test]
fn ray_sphere_miss_behind() {
    // Ray only goes forward -- sphere behind origin should miss
    let r = Ray::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 1.0);
    assert!(!r.collides(&s));
}

#[test]
fn ray_sphere_hit_at_origin() {
    let r = Ray::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::ZERO, 1.0);
    assert!(r.collides(&s));
}

#[test]
fn ray_sphere_miss_lateral() {
    let r = Ray::new(Vec3::ZERO, Vec3::X);
    let s = Sphere::new(Vec3::new(5.0, 3.0, 0.0), 1.0);
    assert!(!r.collides(&s));
}

#[test]
fn ray_capsule_hit() {
    let r = Ray::new(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(-1.0, 5.0, 0.0), Vec3::new(1.0, 5.0, 0.0), 0.5);
    assert!(r.collides(&c));
}

#[test]
fn ray_capsule_miss_behind() {
    let r = Ray::new(Vec3::ZERO, Vec3::Y);
    let c = Capsule::new(Vec3::new(-1.0, -5.0, 0.0), Vec3::new(1.0, -5.0, 0.0), 0.5);
    assert!(!r.collides(&c));
}

#[test]
fn ray_cuboid_hit() {
    let r = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(r.collides(&c));
}

#[test]
fn ray_cuboid_miss_behind() {
    let r = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::Z);
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(!r.collides(&c));
}

#[test]
fn ray_infinite_plane() {
    // Origin above half-space, pointing down
    let r = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
    let ip = Plane::new(Vec3::Y, 0.0);
    assert!(r.collides(&ip));

    // Origin above, pointing away
    let r2 = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
    assert!(!r2.collides(&ip));

    // Origin inside half-space
    let r3 = Ray::new(Vec3::new(0.0, -1.0, 0.0), Vec3::Y);
    assert!(r3.collides(&ip));
}

#[test]
fn ray_convex_polygon() {
    let poly = ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    // Ray from above, hitting polygon
    let r = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
    assert!(r.collides(&poly));

    // Ray from above, missing polygon
    let r2 = Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_Y);
    assert!(!r2.collides(&poly));

    // Ray pointing away
    let r3 = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
    assert!(!r3.collides(&poly));
}

#[test]
fn ray_collides_many_spheres() {
    let r = Ray::new(Vec3::ZERO, Vec3::X);
    let spheres: Vec<Sphere> = (0..20)
        .map(|i| Sphere::new(Vec3::new(i as f32 - 5.0, 0.0, 0.0), 0.1))
        .collect();

    let simd_result = spheres.iter().any(|x| r.collides(x));
    let scalar_result = spheres.iter().any(|s| r.collides(s));
    assert_eq!(simd_result, scalar_result);
}

// ---------------------------------------------------------------------------
// LineSegment
// ---------------------------------------------------------------------------

#[test]
fn segment_sphere_hit() {
    let s = LineSegment::new(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
    let sp = Sphere::new(Vec3::new(5.0, 0.5, 0.0), 1.0);
    assert!(s.collides(&sp));
    assert!(sp.collides(&s));
}

#[test]
fn segment_sphere_miss_beyond() {
    let s = LineSegment::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
    let sp = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
    assert!(!s.collides(&sp));
}

#[test]
fn segment_sphere_miss_behind() {
    let s = LineSegment::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
    let sp = Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 1.0);
    assert!(!s.collides(&sp));
}

#[test]
fn segment_capsule_hit() {
    let s = LineSegment::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
    let c = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
    assert!(s.collides(&c));
}

#[test]
fn segment_capsule_miss() {
    let s = LineSegment::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 10.0, 0.0));
    let c = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.5);
    assert!(!s.collides(&c));
}

#[test]
fn segment_cuboid_through() {
    let s = LineSegment::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 5.0));
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(s.collides(&c));
}

#[test]
fn segment_cuboid_miss_short() {
    let s = LineSegment::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, -3.0));
    let c = Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(!s.collides(&c));
}

#[test]
fn segment_infinite_plane() {
    // Segment crossing the plane
    let s = LineSegment::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    let ip = Plane::new(Vec3::Y, 0.0);
    assert!(s.collides(&ip));

    // Segment entirely above
    let s2 = LineSegment::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 3.0, 0.0));
    assert!(!s2.collides(&ip));
}

#[test]
fn segment_convex_polygon() {
    let poly = ConvexPolygon::with_axes(
        Vec3::ZERO,
        Vec3::Y,
        Vec3::X,
        Vec3::NEG_Z,
        vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    );
    // Segment through polygon
    let s = LineSegment::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
    assert!(s.collides(&poly));

    // Segment too short
    let s2 = LineSegment::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 3.0, 0.0));
    assert!(!s2.collides(&poly));
}

#[test]
fn segment_collides_many_spheres() {
    let s = LineSegment::new(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0));
    let spheres: Vec<Sphere> = (0..20)
        .map(|i| Sphere::new(Vec3::new(i as f32 - 5.0, 0.0, 0.0), 0.1))
        .collect();

    let simd_result = spheres.iter().any(|x| s.collides(x));
    let scalar_result = spheres.iter().any(|sp| s.collides(sp));
    assert_eq!(simd_result, scalar_result);
}

#[test]
fn segment_translate() {
    let mut s = LineSegment::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    s.translate(glam::Vec3A::new(0.0, 5.0, 0.0));
    assert_eq!(s.start, Vec3::new(0.0, 5.0, 0.0));
}

#[test]
fn segment_scale() {
    let mut s = LineSegment::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    s.scale(3.0);
    assert!((s.dir().x - 3.0).abs() < 1e-6);
}

#[test]
fn segment_scale_anchors_at_start() {
    // Scaling is size-only about the segment's own start, not about the world origin.
    let start = Vec3::new(2.0, 0.0, 0.0);
    let mut s = LineSegment::new(start, Vec3::new(3.0, 0.0, 0.0));
    s.scale(4.0);
    assert_eq!(s.start, start);
    assert_eq!(s.end, Vec3::new(6.0, 0.0, 0.0));
}

#[test]
fn segment_affine_transform_maps_both_endpoints() {
    let start = Vec3::new(1.0, -2.0, 0.5);
    let end = Vec3::new(3.0, 1.0, 2.0);
    let mut s = LineSegment::new(start, end);
    let affine = glam::Affine3A::from_scale_rotation_translation(
        Vec3::new(2.0, 0.5, 1.5),
        glam::Quat::from_rotation_y(0.7),
        Vec3::new(1.0, 1.0, 1.0),
    );
    s.transform(affine);
    assert!((s.start - affine.transform_point3(start)).length() < 1e-5);
    assert!((s.end - affine.transform_point3(end)).length() < 1e-5);
}

#[test]
fn segment_rotate_then_translate_maps_endpoints() {
    let start = Vec3::new(1.0, 2.0, 3.0);
    let end = Vec3::new(4.0, 0.0, -1.0);
    let mut s = LineSegment::new(start, end);
    let rot = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let offset = Vec3::new(5.0, -1.0, 2.0);
    s.rotate_quat(rot);
    s.translate(glam::Vec3A::from(offset));
    assert!((s.start - (rot * start + offset)).length() < 1e-5);
    assert!((s.end - (rot * end + offset)).length() < 1e-5);
}

#[test]
fn line_affine_transform_point_and_vector() {
    // origin moves as a point (picks up translation); dir moves as a free vector.
    let origin = Vec3::new(1.0, 2.0, 3.0);
    let dir = Vec3::new(0.0, 1.0, 0.0);
    let mut l = Line::new(origin, dir);
    let affine = glam::Affine3A::from_scale_rotation_translation(
        Vec3::new(2.0, 0.5, 1.5),
        glam::Quat::from_rotation_x(0.5),
        Vec3::new(4.0, 0.0, -2.0),
    );
    l.transform(affine);
    assert!((l.origin - affine.transform_point3(origin)).length() < 1e-5);
    assert!((l.dir - affine.transform_vector3(dir)).length() < 1e-5);
}
