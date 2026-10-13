//! Native joint-space reset and immediate rigid-body state publication.

use glamx::Vec4;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::compute_dynamics_pre::{_forward_kinematics, _propagate_velocities};
use super::{
    MultibodyInfo, MultibodyLinkStatic, WS_JOINT_ROT, WS_KIN_ACC, WS_RB_VELS, WsAddr, ws_coord,
    ws_set_coord, ws_set_rot, ws_set_vel, ws_vel,
};
use crate::dynamics::{SPATIAL_DIM, Velocity};
use crate::utils::BatchIndices;
use crate::{DIM, Pose};

/// One invocation per articulation. Inputs use the live batch interleave;
/// a nonzero batch mask selects all articulations of that batch.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_reset_state(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] selected: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] roots: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] coordinates: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] velocities: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] infos: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] links: &mut [MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let nb = batch_ids.num_batches;
    if id.x >= batch_ids.multibodies_len * nb {
        return;
    }
    let batch = id.x % nb;
    if selected.read(batch as usize) == 0 {
        return;
    }
    let mb = infos.read(id.x as usize);
    let wa = WsAddr::new(mb.first_link as usize, nb, batch);
    for d in 0..mb.ndofs {
        let index = batch_ids.mbi(batch, (mb.first_dof + d) as usize);
        dof_state.write(index, velocities.read(index));
    }
    for k in 0..mb.num_links {
        let index = batch_ids.mbi(batch, (mb.first_link + k) as usize);
        let mut link = links.read(index);
        let mut free = 0;
        for axis in 0..SPATIAL_DIM as u32 {
            let value = if link.data.locked_axes & (1 << axis) == 0 {
                let value = coordinates
                    .read(batch_ids.mbi(batch, (mb.first_dof + link.assembly_id + free) as usize));
                free += 1;
                value
            } else {
                0.0
            };
            ws_set_coord(workspace, wa, k, axis, value);
            link.data.motors[axis as usize].impulse = 0.0;
            link.data.limits[axis as usize].impulse = 0.0;
        }
        links.write(index, link);
        #[cfg(feature = "dim2")]
        let rotation = crate::rotation_from_angle(ws_coord(workspace, wa, k, DIM));
        #[cfg(feature = "dim3")]
        let rotation = crate::rotation_from_scaled_axis(crate::Vector::new(
            ws_coord(workspace, wa, k, 3),
            ws_coord(workspace, wa, k, 4),
            ws_coord(workspace, wa, k, 5),
        ));
        ws_set_rot(workspace, wa, k, WS_JOINT_ROT, rotation);
        ws_set_vel(workspace, wa, k, WS_KIN_ACC, Velocity::default());
        if k == 0 {
            // Invert body_to_parent's frame composition, keeping the supplied
            // root pose independent of the generalized coordinate input.
            let joint_pose = link.data.local_frame_a.inverse()
                * roots.read(id.x as usize)
                * link.data.local_frame_b;
            for axis in 0..DIM {
                #[cfg(feature = "dim2")]
                let value = if axis == 0 {
                    joint_pose.translation.x
                } else {
                    joint_pose.translation.y
                };
                #[cfg(feature = "dim3")]
                let value = if axis == 0 {
                    joint_pose.translation.x
                } else if axis == 1 {
                    joint_pose.translation.y
                } else {
                    joint_pose.translation.z
                };
                ws_set_coord(workspace, wa, 0, axis, value);
            }
            #[cfg(feature = "dim2")]
            ws_set_coord(
                workspace,
                wa,
                0,
                DIM,
                crate::rotation_angle(joint_pose.rotation),
            );
            #[cfg(feature = "dim3")]
            {
                let angular = joint_pose.rotation.to_scaled_axis();
                ws_set_coord(workspace, wa, 0, DIM, angular.x);
                ws_set_coord(workspace, wa, 0, DIM + 1, angular.y);
                ws_set_coord(workspace, wa, 0, DIM + 2, angular.z);
            }
            ws_set_rot(workspace, wa, 0, WS_JOINT_ROT, joint_pose.rotation);
        }
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_publish_reset_state(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] selected: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] roots: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] infos: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] velocities: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let nb = batch_ids.num_batches;
    if id.x >= batch_ids.multibodies_len * nb {
        return;
    }
    let batch = id.x % nb;
    if selected.read(batch as usize) == 0 {
        return;
    }
    let mb = infos.read(id.x as usize);
    let stat = batch_ids.ib(batch, links).offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, nb, batch);
    let mut body_poses = batch_ids.ib_mut(batch, poses);
    body_poses[stat[0].rb_id as usize] = roots.read(id.x as usize);
    _forward_kinematics(&mb, &stat, &mut body_poses, workspace, wa, mb.num_links);
    let dofs = batch_ids.ib(batch, dof_state).offset(mb.first_dof as usize);
    _propagate_velocities(mb.num_links, &stat, &dofs, workspace, wa);
    for k in 0..mb.num_links {
        velocities.write(
            batch_ids.mbi(batch, stat[k as usize].rb_id as usize),
            ws_vel(workspace, wa, k, WS_RB_VELS),
        );
    }
}
