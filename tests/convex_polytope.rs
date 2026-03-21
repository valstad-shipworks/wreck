use glam::Vec3;
use wreck::{ArrayConvexPolytope, Collides, ConvexPolytope, Cuboid, Sphere};

// ---------------------------------------------------------------------------
// Heap (ConvexPolytope)
// ---------------------------------------------------------------------------

fn unit_cube_polytope() -> ConvexPolytope {
    // Unit cube centered at origin: halfplanes n.x <= 1 for all 6 face normals
    let planes = vec![
        (Vec3::X, 1.0),
        (Vec3::NEG_X, 1.0),
        (Vec3::Y, 1.0),
        (Vec3::NEG_Y, 1.0),
        (Vec3::Z, 1.0),
        (Vec3::NEG_Z, 1.0),
    ];
    let vertices = vec![
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    ConvexPolytope::new(planes, vertices)
}

#[test]
fn sphere_polytope_inside() {
    let p = unit_cube_polytope();
    let s = Sphere::new(Vec3::ZERO, 0.5);
    assert!(s.collides(&p));
}

#[test]
fn sphere_polytope_separated() {
    let p = unit_cube_polytope();
    let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5);
    assert!(!s.collides(&p));
}

#[test]
fn cuboid_polytope_overlapping() {
    let p = unit_cube_polytope();
    let c = Cuboid::new(Vec3::new(1.5, 0.0, 0.0), [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(c.collides(&p));
}

#[test]
fn cuboid_polytope_separated() {
    let p = unit_cube_polytope();
    let c = Cuboid::new(Vec3::new(5.0, 0.0, 0.0), [Vec3::X, Vec3::Y, Vec3::Z], [1.0, 1.0, 1.0]);
    assert!(!c.collides(&p));
}

// ---------------------------------------------------------------------------
// Array (ArrayConvexPolytope)
// ---------------------------------------------------------------------------

const UNIT_CUBE: ArrayConvexPolytope<6, 8> = ArrayConvexPolytope::new(
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
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ],
    Cuboid::new(Vec3::ZERO, [Vec3::X, Vec3::Y, Vec3::Z], [1.0; 3]),
);

#[test]
fn const_sphere_inside() {
    let s = Sphere::new(Vec3::ZERO, 0.5);
    assert!(s.collides(&UNIT_CUBE));
}

#[test]
fn const_sphere_separated() {
    let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 0.5);
    assert!(!s.collides(&UNIT_CUBE));
}

#[test]
fn const_cuboid_overlapping() {
    let c = Cuboid::new(
        Vec3::new(1.5, 0.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    assert!(c.collides(&UNIT_CUBE));
}

#[test]
fn const_cuboid_separated() {
    let c = Cuboid::new(
        Vec3::new(5.0, 0.0, 0.0),
        [Vec3::X, Vec3::Y, Vec3::Z],
        [1.0, 1.0, 1.0],
    );
    assert!(!c.collides(&UNIT_CUBE));
}

#[test]
fn symmetry() {
    let s = Sphere::new(Vec3::ZERO, 0.5);
    assert!(UNIT_CUBE.collides(&s));
}
