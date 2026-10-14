use crate::{PaddedVector, Vector};
use crate::bounding_volumes::Aabb;
use crate::shapes::Shape;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Constructs polylines in reserved slots. Requests contain collider index + 1;
/// zero means idle. Each slot owns storage and a segment-index/BVH topology.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_create_polylines(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] requests: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] points: &[PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] radii: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] shapes: &mut [Shape],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] vertices: &mut [PaddedVector],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] indices: &[u32],
) {
    let request = invocation_id.x as usize;
    if request >= requests.len() || requests.read(request) == 0 { return; }
    let id = (requests.read(request) - 1) as usize;
    let slot = shapes.read(id).to_polyline();
    let count = points.len() / requests.len();
    let first = (slot.bvh_vtx_root_id + 2 * slot.bvh_node_len) as usize;
    for i in 0..count {
        vertices.write(first + i, PaddedVector::new(*points.read(request * count + i)));
    }
    let mut root = Aabb::new(Vector::splat(f32::MAX), Vector::splat(-f32::MAX));
    for offset in 0..slot.bvh_node_len {
        let node_id = slot.bvh_node_len - 1 - offset;
        let node = slot.bvh_node_idx(node_id, indices);
        let bounds = if node.entry_index == u32::MAX {
            let segment = slot.segment(node.shape_index, vertices, indices);
            Aabb::new(segment.a.min(segment.b), segment.a.max(segment.b))
        } else {
            let left = node.entry_index;
            let right = slot.bvh_node_idx(left, indices).exit_index;
            let left_bounds = slot.bvh_node_aabb(left, vertices);
            if right < node.exit_index {
                left_bounds.merged(&slot.bvh_node_aabb(right, vertices))
            } else { left_bounds }
        };
        let vertex = (slot.bvh_vtx_root_id + node_id * 2) as usize;
        vertices.write(vertex, PaddedVector::new(bounds.mins));
        vertices.write(vertex + 1, PaddedVector::new(bounds.maxs));
        root = root.merged(&bounds);
    }
    shapes.write(id, Shape::polyline(slot.bvh_vtx_root_id, slot.bvh_idx_root_id,
        slot.bvh_node_len, root.mins, root.maxs).with_polyline_radius(radii.read(request)));
    requests.write(request, 0);
}
