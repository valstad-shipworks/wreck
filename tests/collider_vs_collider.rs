//! Regression tests for `Collider::collides_other` and `Collider::collides`
//! specifically targeting cases where the bounding spheres of two shapes
//! overlap but the actual shapes do not — the bug was that several code
//! paths only consulted the per-shape broadphase (bounding sphere) and
//! returned a false positive (or missed cross-shape pairs entirely).

use glam::Vec3;
use wreck::{
    Capsule, Collider, Collides, ConvexPolytope, Cuboid, Cylinder, LineSegment, NoPcl, Point,
    Pointcloud, Sphere,
};

fn aa_cuboid(center: Vec3, half: [f32; 3]) -> Cuboid {
    Cuboid::new(center, [Vec3::X, Vec3::Y, Vec3::Z], half)
}

/// Sphere placed just past a cuboid corner: the two bounding spheres overlap,
/// but narrowphase must return false (the corner distance exceeds the radius).
#[test]
fn sphere_outside_cuboid_corner() {
    let cuboid = aa_cuboid(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let sphere = Sphere::new(Vec3::new(2.0, 2.0, 2.0), 1.0);
    assert!(
        !sphere.collides(&cuboid),
        "narrowphase sanity: sphere must sit outside cuboid"
    );

    let mut a = Collider::<NoPcl>::new();
    a.add(cuboid);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert!(
        !a.collides_other(&b),
        "collider(cuboid) vs collider(sphere): broadphase-only result leaks false positive"
    );
    assert!(!b.collides_other(&a), "inverse orientation must agree");
    assert!(!a.collides(&sphere), "Collider::collides(Sphere) path");
    assert!(
        !b.collides(&cuboid),
        "Collider::collides(Cuboid) — this was the original reported bug"
    );
}

#[test]
fn sphere_inside_cuboid_collides() {
    let cuboid = aa_cuboid(Vec3::ZERO, [2.0, 2.0, 2.0]);
    let sphere = Sphere::new(Vec3::new(0.5, 0.0, 0.0), 0.25);

    let mut a = Collider::<NoPcl>::new();
    a.add(cuboid);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert!(a.collides_other(&b));
    assert!(b.collides_other(&a));
    assert!(a.collides(&sphere));
    assert!(b.collides(&cuboid));
}

#[test]
fn sphere_outside_capsule() {
    let capsule = Capsule::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.25);
    let sphere = Sphere::new(Vec3::new(0.0, 1.5, 0.0), 0.25);
    assert!(!sphere.collides(&capsule));

    let mut a = Collider::<NoPcl>::new();
    a.add(capsule);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert!(!a.collides_other(&b));
    assert!(!b.collides_other(&a));
    assert!(!b.collides(&capsule));
}

#[test]
fn sphere_inside_capsule() {
    let capsule = Capsule::new(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 0.5);
    let sphere = Sphere::new(Vec3::new(0.0, 0.1, 0.0), 0.25);

    let mut a = Collider::<NoPcl>::new();
    a.add(capsule);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert!(a.collides_other(&b));
    assert!(b.collides_other(&a));
    assert!(b.collides(&capsule));
}

#[test]
fn sphere_outside_cylinder() {
    let cyl = Cylinder::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.5);
    let sphere = Sphere::new(Vec3::new(1.2, 0.0, 0.0), 0.25);
    assert!(!sphere.collides(&cyl));

    let mut a = Collider::<NoPcl>::new();
    a.add(cyl);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert!(!a.collides_other(&b));
    assert!(!b.collides_other(&a));
    assert!(!b.collides(&cyl));
}

#[test]
fn sphere_inside_cylinder() {
    let cyl = Cylinder::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(0.0, 2.0, 0.0), 1.0);
    let sphere = Sphere::new(Vec3::new(0.2, 0.0, 0.0), 0.25);

    let mut a = Collider::<NoPcl>::new();
    a.add(cyl);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert!(a.collides_other(&b));
    assert!(b.collides_other(&a));
    assert!(b.collides(&cyl));
}

/// Many spheres in `other` where all sit outside the cuboid but their bounding
/// spheres overlap the cuboid's bounding sphere. Previously a broadphase-only
/// check short-circuited to `true` for all of them.
#[test]
fn many_spheres_near_cuboid_corner_all_miss() {
    let cuboid = aa_cuboid(Vec3::ZERO, [1.0, 1.0, 1.0]);

    let mut a = Collider::<NoPcl>::new();
    a.add(cuboid);

    let mut b = Collider::<NoPcl>::new();
    for i in 0..16 {
        let t = i as f32 * 0.1;
        b.add(Sphere::new(Vec3::new(2.0 + t, 2.0 + t, 2.0 + t), 0.4));
    }

    assert!(
        !a.collides_other(&b),
        "cuboid vs many spheres: all corner-clear, but broadphase-only reported collision"
    );
    assert!(!b.collides_other(&a));
}

#[test]
fn sphere_between_two_cuboids_misses_both() {
    let mut a = Collider::<NoPcl>::new();
    a.add(aa_cuboid(Vec3::new(-2.0, 0.0, 0.0), [0.5, 0.5, 0.5]));
    a.add(aa_cuboid(Vec3::new(2.0, 0.0, 0.0), [0.5, 0.5, 0.5]));

    let mut b = Collider::<NoPcl>::new();
    b.add(Sphere::new(Vec3::ZERO, 0.8));

    assert!(!a.collides_other(&b));
    assert!(!b.collides_other(&a));
}

/// Narrow cross-shape case: self holds only cuboids, other holds only
/// spheres. Before the fix, `self.spheres.any_collides_soa(&other.spheres)`
/// would return false because self has no spheres, and the remaining branches
/// only iterate non-sphere types in `other` — so a sphere actually colliding
/// with a cuboid was missed entirely.
#[test]
fn cross_shape_cuboid_vs_sphere_is_checked() {
    let mut a = Collider::<NoPcl>::new();
    a.add(aa_cuboid(Vec3::ZERO, [1.0, 1.0, 1.0]));

    let mut b = Collider::<NoPcl>::new();
    b.add(Sphere::new(Vec3::new(0.5, 0.0, 0.0), 0.25));

    assert!(a.collides_other(&b));
    assert!(b.collides_other(&a));
}

#[test]
fn cross_shape_capsule_vs_sphere_is_checked() {
    let mut a = Collider::<NoPcl>::new();
    a.add(Capsule::new(
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        0.5,
    ));

    let mut b = Collider::<NoPcl>::new();
    b.add(Sphere::new(Vec3::new(0.0, 0.2, 0.0), 0.1));

    assert!(a.collides_other(&b));
    assert!(b.collides_other(&a));
}

#[test]
fn cross_shape_cylinder_vs_sphere_is_checked() {
    let mut a = Collider::<NoPcl>::new();
    a.add(Cylinder::new(
        Vec3::new(0.0, -2.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        0.5,
    ));

    let mut b = Collider::<NoPcl>::new();
    b.add(Sphere::new(Vec3::new(0.2, 0.0, 0.0), 0.2));

    assert!(a.collides_other(&b));
    assert!(b.collides_other(&a));
}

#[test]
fn rotated_cuboid_corner_miss() {
    use glam::Quat;
    let axes = {
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        [q * Vec3::X, q * Vec3::Y, Vec3::Z]
    };
    let cuboid = Cuboid::new(Vec3::ZERO, axes, [1.0, 1.0, 1.0]);

    let bounding_r = (3.0f32).sqrt();
    let dir = Vec3::new(1.0, 1.0, 0.0).normalize();
    let sphere_center = dir * (bounding_r - 0.1);
    let sphere = Sphere::new(sphere_center, 0.1);

    let hits_narrow = sphere.collides(&cuboid);

    let mut a = Collider::<NoPcl>::new();
    a.add(cuboid);
    let mut b = Collider::<NoPcl>::new();
    b.add(sphere);

    assert_eq!(a.collides_other(&b), hits_narrow);
    assert_eq!(b.collides_other(&a), hits_narrow);
    assert_eq!(a.collides(&sphere), hits_narrow);
    assert_eq!(b.collides(&cuboid), hits_narrow);
}

#[test]
fn pointcloud_vs_sphere_outside_bounding() {
    let pts = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]];
    let pc = Pointcloud::new(&pts, (0.0, 0.0), 0.0);

    let bp = pc.broadphase();
    let sphere_miss = Sphere::new(bp.center + Vec3::X * (bp.radius + 1.0), 0.05);
    let sphere_hit = Sphere::new(Vec3::new(0.05, 0.0, 0.0), 0.1);

    let mut a = Collider::<Pointcloud>::new();
    a.add(pc);

    let mut miss = Collider::<Pointcloud>::new();
    miss.add(sphere_miss);
    let mut hit = Collider::<Pointcloud>::new();
    hit.add(sphere_hit);

    assert!(!a.collides_other(&miss));
    assert!(!miss.collides_other(&a));
    assert!(a.collides_other(&hit));
    assert!(hit.collides_other(&a));
}

#[test]
fn segment_outside_sphere_narrowphase() {
    let mut spheres = Collider::<NoPcl>::new();
    spheres.add(Sphere::new(Vec3::ZERO, 0.5));

    let seg_miss = LineSegment::new(Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, -1.0, 0.0));
    let seg_hit = LineSegment::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

    assert!(!spheres.collides(&seg_miss));
    assert!(spheres.collides(&seg_hit));

    let mut a = Collider::<NoPcl>::new();
    a.add(seg_miss);
    let mut b = Collider::<NoPcl>::new();
    b.add(seg_hit);

    assert!(!spheres.collides_other(&a));
    assert!(spheres.collides_other(&b));
}

#[test]
fn point_outside_cuboid_corner() {
    let cuboid = aa_cuboid(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let p = Point(Vec3::new(1.5, 1.5, 1.5));

    let mut a = Collider::<NoPcl>::new();
    a.add(cuboid);
    let mut b = Collider::<NoPcl>::new();
    b.add(p);

    assert!(!a.collides_other(&b));
    assert!(!b.collides_other(&a));
}

#[test]
fn sphere_outside_polytope_bounding() {
    let cuboid = aa_cuboid(Vec3::ZERO, [1.0, 1.0, 1.0]);
    let poly = ConvexPolytope::from(cuboid);

    let bp = poly.broadphase();
    let sphere_miss = Sphere::new(bp.center + Vec3::X * (bp.radius + 0.5), 0.1);
    let sphere_hit = Sphere::new(Vec3::ZERO, 0.1);

    let narrow_miss = sphere_miss.collides(&poly);
    let narrow_hit = sphere_hit.collides(&poly);

    let mut a = Collider::<NoPcl>::new();
    a.add(poly);

    let mut m = Collider::<NoPcl>::new();
    m.add(sphere_miss);
    let mut h = Collider::<NoPcl>::new();
    h.add(sphere_hit);

    assert_eq!(a.collides_other(&m), narrow_miss);
    assert_eq!(m.collides_other(&a), narrow_miss);
    assert_eq!(a.collides_other(&h), narrow_hit);
    assert_eq!(h.collides_other(&a), narrow_hit);
}
