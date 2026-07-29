#![cfg(feature = "valuable")]
//! `valuable::Valuable` impls for the public shape types. Glam fields are
//! bridged through `glam_traits_ext::GlamValuable`.

use ::valuable::{
    EnumDef, Enumerable, Fields, Listable, NamedField, NamedValues, StructDef, Structable,
    Valuable, Value, Variant, VariantDef, Visit,
};
use glam::Vec3;
use glam_traits_ext::GlamValuable;

use crate::{
    ArrayConvexPolygon, ArrayConvexPolytope, Capsule, ConvexPolygon, ConvexPolytope, Cuboid,
    Cylinder, NoPcl, Plane, Point, Sphere,
    capsule::CapsuleStretch,
    cuboid::CuboidStretch,
    cylinder::CylinderStretch,
    line::{LineSegmentStretch, LineStretch, RayStretch},
    plane::ConvexPolygonStretch,
    sphere::SphereStretch,
};

// ---------------------------------------------------------------------------
// `Listable` adapter for slices of types implementing `GlamValuable`
// ---------------------------------------------------------------------------

struct GlamSlice<'a, T: GlamValuable>(&'a [T]);

impl<T: GlamValuable> Valuable for GlamSlice<'_, T> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Listable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        for item in self.0 {
            visit.visit_value(item.as_value());
        }
    }
}

impl<T: GlamValuable> Listable for GlamSlice<'_, T> {
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

// ---------------------------------------------------------------------------
// `Listable` adapter for `(Vec3, f32)` plane definitions, rendered as
// `{ normal, d }` structs.
// ---------------------------------------------------------------------------

struct PlanesSlice<'a>(&'a [(Vec3, f32)]);

impl Valuable for PlanesSlice<'_> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Listable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        for (n, d) in self.0 {
            let pair = PlaneTuple { normal: *n, d: *d };
            visit.visit_value(Value::Structable(&pair));
        }
    }
}

impl Listable for PlanesSlice<'_> {
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len(), Some(self.0.len()))
    }
}

struct PlaneTuple {
    normal: Vec3,
    d: f32,
}

const PLANE_TUPLE_FIELDS: &[NamedField<'static>] =
    &[NamedField::new("normal"), NamedField::new("d")];

impl Valuable for PlaneTuple {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [GlamValuable::as_value(&self.normal), Value::F32(self.d)];
        visit.visit_named_fields(&NamedValues::new(PLANE_TUPLE_FIELDS, &values));
    }
}

impl Structable for PlaneTuple {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("Plane", Fields::Named(PLANE_TUPLE_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// Macro for the simple structs.
// ---------------------------------------------------------------------------

macro_rules! field_binding {
    ($self:ident, $field:ident, glam_array) => {
        let $field = GlamSlice(&$self.$field[..]);
    };
    ($self:ident, $field:ident, glam_vec) => {
        let $field = GlamSlice($self.$field.as_slice());
    };
    ($self:ident, $field:ident, planes_vec) => {
        let $field = PlanesSlice($self.$field.as_slice());
    };
    ($self:ident, $field:ident, glam) => {
        /* none */
    };
    ($self:ident, $field:ident, plain) => {
        /* none */
    };
}

macro_rules! field_value {
    ($self:ident, $field:ident, glam) => {
        GlamValuable::as_value(&$self.$field)
    };
    ($self:ident, $field:ident, glam_array) => {
        $field.as_value()
    };
    ($self:ident, $field:ident, glam_vec) => {
        $field.as_value()
    };
    ($self:ident, $field:ident, planes_vec) => {
        $field.as_value()
    };
    ($self:ident, $field:ident, plain) => {
        Valuable::as_value(&$self.$field)
    };
}

macro_rules! impl_struct {
    (
        $ty:ident { $( $field:ident : $kind:tt ),* $(,)? }
    ) => {
        impl Valuable for $ty {
            #[inline]
            fn as_value(&self) -> Value<'_> {
                Value::Structable(self)
            }
            fn visit(&self, visit: &mut dyn Visit) {
                $( field_binding!(self, $field, $kind); )*
                const FIELDS: &[NamedField<'static>] = &[
                    $( NamedField::new(stringify!($field)) ),*
                ];
                let values = [
                    $( field_value!(self, $field, $kind) ),*
                ];
                visit.visit_named_fields(&NamedValues::new(FIELDS, &values));
            }
        }

        impl Structable for $ty {
            fn definition(&self) -> StructDef<'_> {
                const FIELDS: &[NamedField<'static>] = &[
                    $( NamedField::new(stringify!($field)) ),*
                ];
                StructDef::new_static(stringify!($ty), Fields::Named(FIELDS))
            }
        }
    };
}

impl_struct!(Sphere {
    center: glam,
    radius: plain
});
impl_struct!(Capsule {
    p1: glam,
    dir: glam,
    radius: plain,
    rdv: plain,
    z_aligned: plain,
});
impl_struct!(Cylinder {
    p1: glam,
    dir: glam,
    radius: plain,
    rdv: plain,
    z_aligned: plain,
});
impl_struct!(Cuboid {
    center: glam,
    axes: glam_array,
    half_extents: plain,
    axis_aligned: plain,
});
impl_struct!(Plane {
    normal: glam,
    d: plain
});
impl_struct!(ConvexPolygon {
    center: glam,
    normal: glam,
    u_axis: glam,
    v_axis: glam,
    vertices_2d: plain,
    vertices_3d: glam_vec,
    edge_normals_2d: plain,
    edge_offsets_2d: plain,
});
impl_struct!(ConvexPolytope {
    planes: planes_vec,
    vertices: glam_vec,
    obb: plain,
});

// ---------------------------------------------------------------------------
// Tuple-struct / unit-struct / const-generic types
// ---------------------------------------------------------------------------

impl Valuable for Point {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let values = [GlamValuable::as_value(&self.0)];
        visit.visit_unnamed_fields(&values);
    }
}

impl Structable for Point {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("Point", Fields::Unnamed(1))
    }
}

impl Valuable for NoPcl {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        visit.visit_unnamed_fields(&[]);
    }
}

impl Structable for NoPcl {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("NoPcl", Fields::Unnamed(0))
    }
}

const ARR_POLYGON_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("center"),
    NamedField::new("normal"),
    NamedField::new("u_axis"),
    NamedField::new("v_axis"),
    NamedField::new("vertices_2d"),
    NamedField::new("vertices_3d"),
];

impl<const V: usize> Valuable for ArrayConvexPolygon<V> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let vertices_3d = GlamSlice(&self.vertices_3d[..]);
        let values = [
            GlamValuable::as_value(&self.center),
            GlamValuable::as_value(&self.normal),
            GlamValuable::as_value(&self.u_axis),
            GlamValuable::as_value(&self.v_axis),
            self.vertices_2d.as_value(),
            vertices_3d.as_value(),
        ];
        visit.visit_named_fields(&NamedValues::new(ARR_POLYGON_FIELDS, &values));
    }
}

impl<const V: usize> Structable for ArrayConvexPolygon<V> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("ArrayConvexPolygon", Fields::Named(ARR_POLYGON_FIELDS))
    }
}

const ARR_POLYTOPE_FIELDS: &[NamedField<'static>] = &[
    NamedField::new("planes"),
    NamedField::new("vertices"),
    NamedField::new("obb"),
];

impl<const P: usize, const V: usize> Valuable for ArrayConvexPolytope<P, V> {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Structable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        let planes = PlanesSlice(&self.planes[..]);
        let vertices = GlamSlice(&self.vertices[..]);
        let values = [planes.as_value(), vertices.as_value(), self.obb.as_value()];
        visit.visit_named_fields(&NamedValues::new(ARR_POLYTOPE_FIELDS, &values));
    }
}

impl<const P: usize, const V: usize> Structable for ArrayConvexPolytope<P, V> {
    fn definition(&self) -> StructDef<'_> {
        StructDef::new_static("ArrayConvexPolytope", Fields::Named(ARR_POLYTOPE_FIELDS))
    }
}

// ---------------------------------------------------------------------------
// Stretch enums
// ---------------------------------------------------------------------------

const SPHERE_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("NoStretch", Fields::Unnamed(1)),
    VariantDef::new("Stretch", Fields::Unnamed(1)),
];

impl Valuable for SphereStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            SphereStretch::NoStretch(s) => visit.visit_unnamed_fields(&[s.as_value()]),
            SphereStretch::Stretch(c) => visit.visit_unnamed_fields(&[c.as_value()]),
        }
    }
}

impl Enumerable for SphereStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("SphereStretch", SPHERE_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            SphereStretch::NoStretch(_) => 0,
            SphereStretch::Stretch(_) => 1,
        };
        Variant::Static(&SPHERE_STRETCH_VARIANTS[idx])
    }
}

const CAPSULE_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Aligned", Fields::Unnamed(1)),
    VariantDef::new("Unaligned", Fields::Unnamed(2)),
];

impl Valuable for CapsuleStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            CapsuleStretch::Aligned(c) => visit.visit_unnamed_fields(&[c.as_value()]),
            CapsuleStretch::Unaligned(arr, poly) => {
                let arr_slice: &[Capsule] = &arr[..];
                visit.visit_unnamed_fields(&[arr_slice.as_value(), poly.as_value()]);
            }
        }
    }
}

impl Enumerable for CapsuleStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("CapsuleStretch", CAPSULE_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            CapsuleStretch::Aligned(_) => 0,
            CapsuleStretch::Unaligned(_, _) => 1,
        };
        Variant::Static(&CAPSULE_STRETCH_VARIANTS[idx])
    }
}

const CYLINDER_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Aligned", Fields::Unnamed(1)),
    VariantDef::new("Unaligned", Fields::Unnamed(2)),
];

impl Valuable for CylinderStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            CylinderStretch::Aligned(c) => visit.visit_unnamed_fields(&[c.as_value()]),
            CylinderStretch::Unaligned(arr, poly) => {
                let arr_slice: &[Capsule] = &arr[..];
                visit.visit_unnamed_fields(&[arr_slice.as_value(), poly.as_value()]);
            }
        }
    }
}

impl Enumerable for CylinderStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("CylinderStretch", CYLINDER_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            CylinderStretch::Aligned(_) => 0,
            CylinderStretch::Unaligned(_, _) => 1,
        };
        Variant::Static(&CYLINDER_STRETCH_VARIANTS[idx])
    }
}

const CUBOID_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Aligned", Fields::Unnamed(1)),
    VariantDef::new("Unaligned", Fields::Unnamed(1)),
];

impl Valuable for CuboidStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            CuboidStretch::Aligned(c) => visit.visit_unnamed_fields(&[c.as_value()]),
            CuboidStretch::Unaligned(p) => visit.visit_unnamed_fields(&[p.as_value()]),
        }
    }
}

impl Enumerable for CuboidStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("CuboidStretch", CUBOID_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            CuboidStretch::Aligned(_) => 0,
            CuboidStretch::Unaligned(_) => 1,
        };
        Variant::Static(&CUBOID_STRETCH_VARIANTS[idx])
    }
}

const LINE_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Parallel", Fields::Unnamed(1)),
    VariantDef::new("Polygon", Fields::Unnamed(1)),
];

impl Valuable for LineStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            LineStretch::Parallel(l) => visit.visit_unnamed_fields(&[l.as_value()]),
            LineStretch::Polygon(p) => visit.visit_unnamed_fields(&[p.as_value()]),
        }
    }
}

impl Enumerable for LineStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("LineStretch", LINE_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            LineStretch::Parallel(_) => 0,
            LineStretch::Polygon(_) => 1,
        };
        Variant::Static(&LINE_STRETCH_VARIANTS[idx])
    }
}

const SEGMENT_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Parallel", Fields::Unnamed(1)),
    VariantDef::new("Polygon", Fields::Unnamed(1)),
];

impl Valuable for LineSegmentStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            LineSegmentStretch::Parallel(l) => visit.visit_unnamed_fields(&[l.as_value()]),
            LineSegmentStretch::Polygon(p) => visit.visit_unnamed_fields(&[p.as_value()]),
        }
    }
}

impl Enumerable for LineSegmentStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("LineSegmentStretch", SEGMENT_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            LineSegmentStretch::Parallel(_) => 0,
            LineSegmentStretch::Polygon(_) => 1,
        };
        Variant::Static(&SEGMENT_STRETCH_VARIANTS[idx])
    }
}

const RAY_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("Parallel", Fields::Unnamed(1)),
    VariantDef::new("Polygon", Fields::Unnamed(1)),
];

impl Valuable for RayStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            RayStretch::Parallel(r) => visit.visit_unnamed_fields(&[r.as_value()]),
            RayStretch::Polygon(p) => visit.visit_unnamed_fields(&[p.as_value()]),
        }
    }
}

impl Enumerable for RayStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("RayStretch", RAY_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            RayStretch::Parallel(_) => 0,
            RayStretch::Polygon(_) => 1,
        };
        Variant::Static(&RAY_STRETCH_VARIANTS[idx])
    }
}

const POLYGON_STRETCH_VARIANTS: &[VariantDef<'static>] = &[
    VariantDef::new("InPlane", Fields::Unnamed(1)),
    VariantDef::new("OutOfPlane", Fields::Unnamed(1)),
];

impl Valuable for ConvexPolygonStretch {
    #[inline]
    fn as_value(&self) -> Value<'_> {
        Value::Enumerable(self)
    }
    fn visit(&self, visit: &mut dyn Visit) {
        match self {
            ConvexPolygonStretch::InPlane(p) => visit.visit_unnamed_fields(&[p.as_value()]),
            ConvexPolygonStretch::OutOfPlane(p) => visit.visit_unnamed_fields(&[p.as_value()]),
        }
    }
}

impl Enumerable for ConvexPolygonStretch {
    fn definition(&self) -> EnumDef<'_> {
        EnumDef::new_static("ConvexPolygonStretch", POLYGON_STRETCH_VARIANTS)
    }
    fn variant(&self) -> Variant<'_> {
        let idx = match self {
            ConvexPolygonStretch::InPlane(_) => 0,
            ConvexPolygonStretch::OutOfPlane(_) => 1,
        };
        Variant::Static(&POLYGON_STRETCH_VARIANTS[idx])
    }
}
