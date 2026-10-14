//! Shape conversion utilities for GPU-accelerated collision detection.
//!
//! Converts shapes from CPU-side representations (like parry shapes) to the
//! GPU-compatible [`Shape`] type (defined in the shader crate, re-exported from
//! the crate root). Complex shapes (polylines, triangle meshes) reference
//! external vertex data held in [`ShapeBuffers`] rather than storing it inline.

use crate::math::Point;
use crate::parry::bounding_volume::BoundingVolume;
use crate::parry::shape::{Shape as ParryShape, TypedShape};
use crate::shaders::PaddedVector;
use crate::shaders::shapes::{Shape, DEFAULT_POLYLINE_RADIUS};

#[cfg(all(test, feature = "dim2", feature = "cpu"))]
mod tests {
    use super::*;
    use crate::math::{Pose, Vec2};
    use crate::shaders::glamx::UVec3;
    use crate::shaders::shapes::gpu_refit_polylines;

    #[test]
    fn refitted_polyline_bounds_follow_edited_vertices_and_radius() {
        let polyline = crate::parry::shape::Polyline::new(
            (0..5).map(|x| Vec2::new(x as f32, 0.)).collect(), None);
        let mut buffers = ShapeBuffers::default();
        let mut shape = shape_from_parry(&polyline, &mut buffers).unwrap().with_polyline_radius(0.2);
        let first = shape.polyline_vertex_start() as usize;
        for (i, point) in buffers.vertices[first..].iter_mut().enumerate() {
            *point = PaddedVector::new(Vec2::new(i as f32, 2. + 0.5 * i as f32));
        }
        gpu_refit_polylines(UVec3::ZERO, core::slice::from_mut(&mut shape), &mut buffers.vertices, &buffers.indices);
        let bounds = shape.compute_aabb(Pose::IDENTITY, &buffers.vertices);
        assert!(bounds.mins.distance(Vec2::new(-0.2, 1.8)) < 1e-5);
        assert!(bounds.maxs.distance(Vec2::new(4.2, 4.2)) < 1e-5);
        let polyline = shape.to_polyline();
        for index in 0..polyline.bvh_node_len {
            let node = polyline.bvh_node_idx(index, &buffers.indices);
            let bounds = polyline.bvh_node_aabb(index, &buffers.vertices);
            let start = if node.entry_index == u32::MAX { index } else { node.entry_index };
            for child in start..node.exit_index {
                let leaf = polyline.bvh_node_idx(child, &buffers.indices);
                if leaf.entry_index != u32::MAX { continue; }
                let segment = polyline.segment(leaf.shape_index, &buffers.vertices, &buffers.indices);
                for point in [segment.a, segment.b] {
                    assert!(point.cmpge(bounds.mins).all() && point.cmple(bounds.maxs).all());
                }
            }
        }
    }
}

/// Auxiliary buffers for complex shape types like polylines and triangle meshes.
///
/// Holds the external vertex data that polyline/trimesh shapes reference rather
/// than storing inline.
#[derive(Default, Clone, Debug)]
pub struct ShapeBuffers {
    /// Vertex buffer for polylines and triangle meshes. Polyline and TriMesh
    /// shapes reference ranges within this buffer.
    pub vertices: Vec<PaddedVector>,
    /// Index buffers for polylines, triangle meshes, and convex polyhedrons.
    pub indices: Vec<u32>,
}

/// Converts a parry shape to a GPU [`Shape`]. For complex shapes like polylines
/// and triangle meshes, vertex data is appended to `buffers`.
///
/// Returns `None` if the shape type is not yet supported on GPU.
pub fn shape_from_parry(
    shape: &(impl ParryShape + ?Sized),
    buffers: &mut ShapeBuffers,
) -> Option<Shape> {
    match shape.as_typed_shape() {
        TypedShape::Ball(shape) => Some(Shape::ball(shape.radius)),
        TypedShape::Cuboid(shape) => Some(Shape::cuboid(shape.half_extents)),
        TypedShape::Capsule(shape) => Some(Shape::capsule(
            shape.segment.a,
            shape.segment.b,
            shape.radius,
        )),
        TypedShape::Polyline(shape) => {
            let bvh_vtx_root_id = buffers.vertices.len();
            let bvh_idx_root_id = buffers.indices.len();

            struct BvhObject {
                aabb: bvh::aabb::AABB,
                node_index: usize,
            }

            impl bvh::aabb::Bounded for BvhObject {
                fn aabb(&self) -> bvh::aabb::AABB {
                    self.aabb
                }
            }

            impl bvh::bounding_hierarchy::BHShape for BvhObject {
                fn set_bh_node_index(&mut self, index: usize) {
                    self.node_index = index;
                }

                fn bh_node_index(&self) -> usize {
                    self.node_index
                }
            }

            let mut objects: Vec<_> = shape
                .segments()
                .map(|seg| {
                    // Bound the swept segment so collinear polylines have nonzero SAH area.
                    let aabb = seg.local_aabb().loosened(DEFAULT_POLYLINE_RADIUS);
                    BvhObject {
                        aabb: bvh::aabb::AABB::with_bounds(
                            parry_to_bvh_point(aabb.mins),
                            parry_to_bvh_point(aabb.maxs),
                        ),
                        node_index: 0,
                    }
                })
                .collect();

            let bvh = bvh::bvh::BVH::build(&mut objects);
            let flat_bvh = bvh.flatten();
            buffers.vertices.extend(
                flat_bvh
                    .iter()
                    .flat_map(|n| [bvh_to_point(n.aabb.min), bvh_to_point(n.aabb.max)])
                    .map(PaddedVector::new),
            );
            let bvh_node_len = flat_bvh.len();
            buffers.indices.extend(
                flat_bvh
                    .iter()
                    .flat_map(|n| [n.entry_index, n.exit_index, n.shape_index]),
            );

            buffers
                .vertices
                .extend(shape.vertices().iter().copied().map(PaddedVector::new));
            buffers
                .indices
                .extend(shape.indices().iter().flat_map(|seg| seg.iter().copied()));

            let aabb = shape.local_aabb();
            Some(Shape::polyline(
                bvh_vtx_root_id as u32,
                bvh_idx_root_id as u32,
                bvh_node_len as u32,
                aabb.mins,
                aabb.maxs,
            ))
        }
        TypedShape::TriMesh(shape) => {
            let bvh_vtx_root_id = buffers.vertices.len();
            let bvh_idx_root_id = buffers.indices.len();

            struct BvhObject {
                aabb: bvh::aabb::AABB,
                node_index: usize,
            }

            impl bvh::aabb::Bounded for BvhObject {
                fn aabb(&self) -> bvh::aabb::AABB {
                    self.aabb
                }
            }

            impl bvh::bounding_hierarchy::BHShape for BvhObject {
                fn set_bh_node_index(&mut self, index: usize) {
                    self.node_index = index;
                }

                fn bh_node_index(&self) -> usize {
                    self.node_index
                }
            }

            let mut objects: Vec<_> = shape
                .triangles()
                .map(|tri| {
                    let aabb = tri.local_aabb();
                    BvhObject {
                        aabb: bvh::aabb::AABB::with_bounds(
                            parry_to_bvh_point(aabb.mins),
                            parry_to_bvh_point(aabb.maxs),
                        ),
                        node_index: 0,
                    }
                })
                .collect();

            let bvh = bvh::bvh::BVH::build(&mut objects);
            let flat_bvh = bvh.flatten();
            buffers.vertices.extend(
                flat_bvh
                    .iter()
                    .flat_map(|n| [bvh_to_point(n.aabb.min), bvh_to_point(n.aabb.max)])
                    .map(PaddedVector::new),
            );
            let bvh_node_len = flat_bvh.len();
            buffers.indices.extend(
                flat_bvh
                    .iter()
                    .flat_map(|n| [n.entry_index, n.exit_index, n.shape_index]),
            );

            buffers
                .vertices
                .extend(shape.vertices().iter().copied().map(PaddedVector::new));

            // Append pseudo-normals (vertex + edge) needed by project_local_point
            // to determine inside/outside status.
            #[cfg(feature = "dim3")]
            {
                let pn = shape
                    .pseudo_normals()
                    .expect("trimeshes without pseudo-normals are not supported");
                buffers.vertices.extend(
                    pn.vertices_pseudo_normal
                        .iter()
                        .copied()
                        .map(PaddedVector::new),
                );
                buffers.vertices.extend(
                    pn.edges_pseudo_normal
                        .iter()
                        .flat_map(|n| *n)
                        .map(PaddedVector::new),
                );
            }

            buffers
                .indices
                .extend(shape.indices().iter().flat_map(|tri| tri.iter().copied()));

            let aabb = shape.local_aabb();
            Some(Shape::trimesh(
                bvh_vtx_root_id as u32,
                bvh_idx_root_id as u32,
                bvh_node_len as u32,
                shape.indices().len() as u32,
                shape.vertices().len() as u32,
                aabb.mins,
                aabb.maxs,
            ))
        }
        #[cfg(feature = "dim2")]
        TypedShape::ConvexPolygon(poly) => {
            let first_vtx_id = buffers.vertices.len() as u32;
            buffers
                .vertices
                .extend(poly.points().iter().copied().map(PaddedVector::new));
            let end_vtx_id = buffers.vertices.len() as u32;
            Some(Shape::convex_poly(first_vtx_id, end_vtx_id, 0, 0))
        }
        #[cfg(feature = "dim3")]
        TypedShape::ConvexPolyhedron(poly) => {
            let first_vtx_id = buffers.vertices.len();
            let first_face_id = buffers.indices.len();
            let all_idx = poly.vertices_adj_to_face();

            buffers
                .vertices
                .extend(poly.points().iter().copied().map(PaddedVector::new));
            for face in poly.faces() {
                let id = face.first_vertex_or_edge as usize;

                if face.num_vertices_or_edges >= 3 {
                    buffers.indices.push(all_idx[id]);
                    buffers.indices.push(all_idx[id + 1]);
                    buffers.indices.push(all_idx[id + 2]);
                }
            }

            let end_vtx_id = buffers.vertices.len();
            let end_face_id = buffers.indices.len();
            Some(Shape::convex_poly(
                first_vtx_id as u32,
                end_vtx_id as u32,
                first_face_id as u32,
                end_face_id as u32,
            ))
        }
        #[cfg(feature = "dim2")]
        TypedShape::HeightField(_shape) => {
            todo!()
        }
        #[cfg(feature = "dim3")]
        TypedShape::HeightField(_shape) => {
            todo!()
        }
        #[cfg(feature = "dim3")]
        TypedShape::Cone(shape) => Some(Shape::cone(shape.half_height, shape.radius)),
        #[cfg(feature = "dim3")]
        TypedShape::Cylinder(shape) => Some(Shape::cylinder(shape.half_height, shape.radius)),
        _ => None,
    }
}

/// Convert parry point to bvh Point3 (glam::Vec3)
fn parry_to_bvh_point(p: Point) -> bvh::Point3 {
    #[cfg(feature = "dim2")]
    return bvh::Point3::new(p.x, p.y, 0.0);
    #[cfg(feature = "dim3")]
    return bvh::Point3::new(p.x, p.y, p.z);
}

/// Convert bvh Point3 (glam::Vec3) to our Point type
fn bvh_to_point(p: bvh::Point3) -> Point {
    #[cfg(feature = "dim2")]
    return glamx::Vec2::new(p.x, p.y);
    #[cfg(feature = "dim3")]
    return glamx::Vec3::new(p.x, p.y, p.z);
}
