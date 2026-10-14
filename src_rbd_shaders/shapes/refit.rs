use crate::{PaddedVector, Vector};
use crate::bounding_volumes::Aabb;
use crate::shapes::{Shape, SHAPE_TYPE_POLYLINE};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Refit each polyline's existing BVH after editing its vertices in place.
/// Each polyline must own its vertex/BVH storage; topology remains unchanged.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_refit_polylines(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] shapes: &mut [Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertices: &mut [PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] indices: &[u32],
) {
    let id = invocation_id.x as usize;
    if id >= shapes.len() || shapes.read(id).shape_type() != SHAPE_TYPE_POLYLINE {
        return;
    }
    let shape = shapes.read(id);
    let polyline = shape.to_polyline();
    let mut root = Aabb::new(Vector::splat(f32::MAX), Vector::splat(-f32::MAX));
    for offset in 0..polyline.bvh_node_len {
        let node_id = polyline.bvh_node_len - 1 - offset;
        let node = polyline.bvh_node_idx(node_id, indices);
        let aabb = if node.entry_index == u32::MAX {
            let segment = polyline.segment(node.shape_index, vertices, indices);
            Aabb::new(segment.a.min(segment.b), segment.a.max(segment.b))
        } else {
            let left = node.entry_index;
            let right = polyline.bvh_node_idx(left, indices).exit_index;
            let bounds = polyline.bvh_node_aabb(left, vertices);
            if right < node.exit_index {
                bounds.merged(&polyline.bvh_node_aabb(right, vertices))
            } else {
                bounds
            }
        };
        let vertex = (polyline.bvh_vtx_root_id + node_id * 2) as usize;
        vertices.write(vertex, PaddedVector::new(aabb.mins));
        vertices.write(vertex + 1, PaddedVector::new(aabb.maxs));
        root = root.merged(&aabb);
    }
    shapes.write(id, Shape::polyline(polyline.bvh_vtx_root_id, polyline.bvh_idx_root_id,
        polyline.bvh_node_len, root.mins, root.maxs).with_polyline_radius(polyline.radius));
}
