use alloc::vec::Vec;
use proc_macro2::TokenStream;
use quote::quote;

use crate::capsule::Capsule;
use crate::cuboid::Cuboid;
use crate::cylinder::Cylinder;
use crate::plane::Plane;
use crate::plane::array_convex::ArrayConvexPolygon;
use crate::convex_polytope::array::ArrayConvexPolytope;
use crate::sphere::Sphere;

fn vec3_tokens(v: glam::Vec3) -> TokenStream {
    let x = v.x;
    let y = v.y;
    let z = v.z;
    quote! { glam::Vec3::new(#x, #y, #z) }
}

impl From<Sphere> for TokenStream {
    fn from(s: Sphere) -> TokenStream {
        let center = vec3_tokens(s.center);
        let radius = s.radius;
        quote! { wreck::Sphere::new(#center, #radius) }
    }
}

impl From<Capsule> for TokenStream {
    fn from(c: Capsule) -> TokenStream {
        let p1 = vec3_tokens(c.p1);
        let p2 = vec3_tokens(c.p1 + c.dir);
        let radius = c.radius;
        quote! { wreck::Capsule::new(#p1, #p2, #radius) }
    }
}

impl From<Cylinder> for TokenStream {
    fn from(c: Cylinder) -> TokenStream {
        let p1 = vec3_tokens(c.p1);
        let p2 = vec3_tokens(c.p1 + c.dir);
        let radius = c.radius;
        quote! { wreck::Cylinder::new(#p1, #p2, #radius) }
    }
}

impl From<Cuboid> for TokenStream {
    fn from(c: Cuboid) -> TokenStream {
        let center = vec3_tokens(c.center);
        let a0 = vec3_tokens(c.axes[0]);
        let a1 = vec3_tokens(c.axes[1]);
        let a2 = vec3_tokens(c.axes[2]);
        let h0 = c.half_extents[0];
        let h1 = c.half_extents[1];
        let h2 = c.half_extents[2];
        quote! {
            wreck::Cuboid::new(
                #center,
                [#a0, #a1, #a2],
                [#h0, #h1, #h2],
            )
        }
    }
}

impl From<Plane> for TokenStream {
    fn from(p: Plane) -> TokenStream {
        let normal = vec3_tokens(p.normal);
        let d = p.d;
        quote! { wreck::Plane::new(#normal, #d) }
    }
}

impl<const V: usize> From<ArrayConvexPolygon<V>> for TokenStream {
    fn from(p: ArrayConvexPolygon<V>) -> TokenStream {
        let center = vec3_tokens(p.center);
        let normal = vec3_tokens(p.normal);
        let u_axis = vec3_tokens(p.u_axis);
        let v_axis = vec3_tokens(p.v_axis);
        let verts: Vec<TokenStream> = p.vertices_2d.iter().map(|v| {
            let a = v[0];
            let b = v[1];
            quote! { [#a, #b] }
        }).collect();
        quote! {
            wreck::ArrayConvexPolygon::<#V>::new(
                #center,
                #normal,
                #u_axis,
                #v_axis,
                [#(#verts),*],
            )
        }
    }
}

impl<const P: usize, const V: usize> From<ArrayConvexPolytope<P, V>> for TokenStream {
    fn from(p: ArrayConvexPolytope<P, V>) -> TokenStream {
        let planes: Vec<TokenStream> = p.planes.iter().map(|(n, d)| {
            let nt = vec3_tokens(*n);
            quote! { (#nt, #d) }
        }).collect();
        let vertices: Vec<TokenStream> = p.vertices.iter().map(|v| vec3_tokens(*v)).collect();
        let obb: TokenStream = p.obb.into();
        quote! {
            wreck::ArrayConvexPolytope::<#P, #V>::new(
                [#(#planes),*],
                [#(#vertices),*],
                #obb,
            )
        }
    }
}
