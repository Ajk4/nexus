//! Integrate kernel (semi-implicit Euler).
//!
//! Advances generalized velocities, then each link's `coords` / `joint_rot`.
//! After this pass, callers are expected to re-run forward kinematics to
//! refresh link poses.

use glamx::Vec4;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

#[cfg(feature = "dim2")]
use crate::rotation_from_angle;
use crate::utils::BatchIndices;
use crate::{ANG_DIM, DIM, Pose, Vector};
#[cfg(feature = "dim3")]
use crate::{rotation_from_scaled_axis, rotation_renormalize_fast};
use parry::math::VectorExt;

use super::compute_dynamics_pre::{_forward_kinematics, _propagate_velocities};
use super::types::{MultibodyInfo, MultibodyLinkStatic};
#[cfg(feature = "dim3")]
use super::ws_soa::ws_rot;
use super::ws_soa::{WS_JOINT_ROT, WsAddr, ws_coord, ws_set_coord, ws_set_rot};
use super::ws_soa::{
    WS_LTP, WS_LTW, WS_RB_VELS, WS_WORLD_COM, ws_ext_wrench, ws_pose, ws_set_pose, ws_set_vec,
    ws_vec, ws_vel,
};
use crate::dynamics::Velocity;
use crate::utils::ISlice;

fn _centroid(
    links: &ISlice<MultibodyLinkStatic>,
    workspace: &[Vec4],
    wa: WsAddr,
    count: u32,
) -> (f32, Vector, Vector) {
    let mut mass = 0.0;
    let mut center = Vector::ZERO;
    let mut momentum = Vector::ZERO;
    let mut center_error = Vector::ZERO;
    let mut momentum_error = Vector::ZERO;
    for k in 0..count {
        let inverse_mass = links[k as usize].local_mprops.inv_mass.x;
        if inverse_mass == 0.0 {
            continue;
        }
        mass += 1.0 / inverse_mass;
        let term = ws_vec(workspace, wa, k, WS_WORLD_COM) / inverse_mass - center_error;
        let next = center + term;
        center_error = (next - center) - term;
        center = next;
        let term = ws_vel(workspace, wa, k, WS_RB_VELS).linear / inverse_mass - momentum_error;
        let next = momentum + term;
        momentum_error = (next - momentum) - term;
        momentum = next;
    }
    (mass, center / mass, momentum)
}

fn _shift_base_velocity(
    root: &MultibodyLinkStatic,
    velocities: &mut crate::utils::ISliceMut<f32>,
    delta: Vector,
) {
    let delta = root.data.local_frame_a.rotation.inverse() * delta;
    for axis in 0..DIM {
        velocities[axis as usize] += delta.vget(axis as usize);
    }
}

/// Update generalized velocities: `v += a · dt`.
///
/// Split out from the position-update half so that joint-limit / motor
/// constraints can run in between (rapier's order: velocity update → constraint
/// solver → position update).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_integrate_velocities(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] gen_accelerations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] links_workspace: &mut [Vec4],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] gravity: &Vec4,
) {
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / num_mb;
    let mb_idx = invocation_id.x % num_mb;
    let dt = *dt_uniform;

    let mb = batch_ids.ib(batch_id, multibody_info).read(mb_idx as usize);

    let links = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);
    let floating = mb.root_is_dynamic != 0 && links[0].data.locked_axes & ((1 << DIM) - 1) == 0;
    let mut expected = Vector::ZERO;
    let mut mass = 0.0;
    if floating {
        (mass, _, expected) = _centroid(&links, links_workspace, wa, mb.num_links);
        #[cfg(feature = "dim2")]
        let gravity = Vector::new(gravity.x, gravity.y);
        #[cfg(feature = "dim3")]
        let gravity = Vector::new(gravity.x, gravity.y, gravity.z);
        for k in 0..mb.num_links {
            let inverse_mass = links[k as usize].local_mprops.inv_mass.x;
            if inverse_mass == 0.0 {
                continue;
            }
            let (force, _, scale) = ws_ext_wrench(links_workspace, wa, k);
            expected += (force + gravity * (scale / inverse_mass)) * dt;
        }
    }

    let mut dof_vel = batch_ids
        .ib_mut(batch_id, dof_state)
        .offset(mb.first_dof as usize);
    // The accelerations live in the per-multibody-region `gen_forces` buffer
    // (dense region per (multibody, batch); `dof_state` stays interleaved).
    let acc0 = batch_ids.mb_region(batch_id, mb.first_dof, mb.ndofs);

    for d in 0..mb.ndofs {
        let di = d as usize;
        dof_vel[di] += gen_accelerations.read(acc0 + di) * dt;
    }
    if floating {
        let velocities = batch_ids
            .ib(batch_id, dof_state)
            .offset(mb.first_dof as usize);
        _propagate_velocities(mb.num_links, &links, &velocities, links_workspace, wa);
        let (_, _, current) = _centroid(&links, links_workspace, wa, mb.num_links);
        let mut velocities = batch_ids
            .ib_mut(batch_id, dof_state)
            .offset(mb.first_dof as usize);
        _shift_base_velocity(&links[0], &mut velocities, (expected - current) / mass);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_integrate(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] poses: &mut [Pose],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] dt_uniform: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / num_mb;
    let mb_idx = invocation_id.x % num_mb;
    let dt = *dt_uniform;

    let mb = batch_ids.ib(batch_id, multibody_info).read(mb_idx as usize);
    let num_links = mb.num_links;

    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);
    let dof_vel = batch_ids
        .ib(batch_id, dof_state)
        .offset(mb.first_dof as usize);

    let floating =
        mb.root_is_dynamic != 0 && stat_slice[0].data.locked_axes & ((1 << DIM) - 1) == 0;
    let center = if floating {
        _propagate_velocities(num_links, &stat_slice, &dof_vel, links_workspace, wa);
        let (mass, center, momentum) = _centroid(&stat_slice, links_workspace, wa, num_links);
        center + momentum * (dt / mass)
    } else {
        Vector::ZERO
    };

    // Per-link coord / joint_rot update (uses the already-corrected `dof_velocities`).
    //
    // Only `coords` (≤ 24 B) and `joint_rot` (16 B) are modified. We mutate them
    // in place through `&mut ws_slice[k]` so SPIR-V emits field-targeted stores
    // instead of a whole `MultibodyLinkWorkspace` round-trip (~240 B in 3D).
    for k in 0..num_links {
        let stat = stat_slice[k as usize];
        let locked = stat.data.locked_axes;
        let aid = stat.assembly_id as usize;

        // Free linear DOFs first, in axis order.
        let mut curr_free = 0u32;
        for i in 0..DIM {
            if (locked & (1 << i)) == 0 {
                let v = dof_vel[aid + curr_free as usize];
                let cur = ws_coord(links_workspace, wa, k, i);
                ws_set_coord(links_workspace, wa, k, i, cur + v * dt);
                curr_free += 1;
            }
        }

        // Free angular DOFs.
        let ang_locked = (locked >> DIM) & ((1 << ANG_DIM) - 1);
        let num_ang = ANG_DIM - ang_locked.count_ones();
        if num_ang == 1 {
            #[cfg(feature = "dim3")]
            {
                let dof_id = (!ang_locked & 0x7).trailing_zeros();
                let v = dof_vel[aid + curr_free as usize];
                let idx = 3 + dof_id;
                let new = ws_coord(links_workspace, wa, k, idx) + v * dt;
                ws_set_coord(links_workspace, wa, k, idx, new);
                ws_set_rot(
                    links_workspace,
                    wa,
                    k,
                    WS_JOINT_ROT,
                    rotation_from_scaled_axis(Vector::ith(dof_id as usize, new)),
                );
            }
            #[cfg(feature = "dim2")]
            {
                let v = dof_vel[aid + curr_free as usize];
                let new = ws_coord(links_workspace, wa, k, DIM) + v * dt;
                ws_set_coord(links_workspace, wa, k, DIM, new);
                ws_set_rot(
                    links_workspace,
                    wa,
                    k,
                    WS_JOINT_ROT,
                    rotation_from_angle(new),
                );
            }
        } else if num_ang == 3 {
            #[cfg(feature = "dim3")]
            {
                let vx = dof_vel[aid + curr_free as usize];
                let vy = dof_vel[aid + (curr_free + 1) as usize];
                let vz = dof_vel[aid + (curr_free + 2) as usize];
                let ang = Vector::new(vx, vy, vz);
                let disp = rotation_from_scaled_axis(ang * dt);
                let jr = ws_rot(links_workspace, wa, k, WS_JOINT_ROT);
                ws_set_rot(
                    links_workspace,
                    wa,
                    k,
                    WS_JOINT_ROT,
                    rotation_renormalize_fast(disp * jr),
                );
                let c3 = ws_coord(links_workspace, wa, k, 3);
                ws_set_coord(links_workspace, wa, k, 3, c3 + vx * dt);
                let c4 = ws_coord(links_workspace, wa, k, 4);
                ws_set_coord(links_workspace, wa, k, 4, c4 + vy * dt);
                let c5 = ws_coord(links_workspace, wa, k, 5);
                ws_set_coord(links_workspace, wa, k, 5, c5 + vz * dt);
            }
        }
        // num_ang == 0: no-op.
    }

    if floating {
        let mut body_poses = batch_ids.ib_mut(batch_id, poses);
        _forward_kinematics(
            &mb,
            &stat_slice,
            &mut body_poses,
            links_workspace,
            wa,
            num_links,
        );
        _propagate_velocities(num_links, &stat_slice, &dof_vel, links_workspace, wa);
        let (_, new_center, _) = _centroid(&stat_slice, links_workspace, wa, num_links);
        // Advance the system COM using its linear momentum. Joint motion
        // changes the body's offset from that COM, not its translation rate.
        let displacement = center - new_center;
        let local = stat_slice[0].data.local_frame_a.rotation.inverse() * displacement;
        for axis in 0..DIM {
            let coordinate = ws_coord(links_workspace, wa, 0, axis);
            ws_set_coord(
                links_workspace,
                wa,
                0,
                axis,
                coordinate + local.vget(axis as usize),
            );
        }
        let mut root_pose = ws_pose(links_workspace, wa, 0, WS_LTP);
        root_pose.translation += displacement;
        ws_set_pose(links_workspace, wa, 0, WS_LTP, root_pose);
        for k in 0..num_links {
            let mut pose = ws_pose(links_workspace, wa, k, WS_LTW);
            pose.translation += displacement;
            ws_set_pose(links_workspace, wa, k, WS_LTW, pose);
            let com = ws_vec(links_workspace, wa, k, WS_WORLD_COM) + displacement;
            ws_set_vec(links_workspace, wa, k, WS_WORLD_COM, com);
            body_poses[stat_slice[k as usize].rb_id as usize] = pose;
        }
    }
}

/// Carry velocity through the configuration change after both constraint
/// sweeps. The retained tree transforms describe the impulses just solved.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_transport_velocities(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] factors: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let count = batch_ids.multibodies_len;
    if id.x >= count * batch_ids.num_batches {
        return;
    }
    let batch = id.x % batch_ids.num_batches;
    let mb = batch_ids
        .ib(batch, multibody_info)
        .read((id.x / batch_ids.num_batches) as usize);
    let links = batch_ids
        .ib(batch, links_static)
        .offset(mb.first_link as usize);
    if mb.root_is_dynamic == 0 || links[0].data.locked_axes & ((1 << DIM) - 1) != 0 {
        return;
    }
    let velocities = batch_ids.ib(batch, dof_state).offset(mb.first_dof as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch);
    let factors = batch_ids
        .ib(batch, factors)
        .offset(mb.first_link as usize * super::articulated_body::ARTICULATED_FACTOR_WORDS);
    let momentum = super::articulated_body::_tree_momentum(
        mb.num_links,
        &links,
        &factors,
        &velocities,
        links_workspace,
        wa,
    );
    _propagate_velocities(mb.num_links, &links, &velocities, links_workspace, wa);
    let (mass, _, current) = _centroid(&links, links_workspace, wa, mb.num_links);
    let mut velocities = batch_ids
        .ib_mut(batch, dof_state)
        .offset(mb.first_dof as usize);
    _shift_base_velocity(&links[0], &mut velocities, (momentum - current) / mass);
    let velocities = batch_ids.ib(batch, dof_state).offset(mb.first_dof as usize);
    _propagate_velocities(mb.num_links, &links, &velocities, links_workspace, wa);
}

/// Publish the final articulation state after the last constraint sweep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_publish_state(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] velocities: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let count = batch_ids.multibodies_len;
    if id.x >= count * batch_ids.num_batches {
        return;
    }
    let batch = id.x / count;
    let mb = batch_ids
        .ib(batch, multibody_info)
        .read((id.x % count) as usize);
    let links = batch_ids
        .ib(batch, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch);
    let mut body_poses = batch_ids.ib_mut(batch, poses);
    _forward_kinematics(
        &mb,
        &links,
        &mut body_poses,
        links_workspace,
        wa,
        mb.num_links,
    );
    let dofs = batch_ids.ib(batch, dof_state).offset(mb.first_dof as usize);
    _propagate_velocities(mb.num_links, &links, &dofs, links_workspace, wa);
    let mut body_velocities = batch_ids.ib_mut(batch, velocities);
    for k in 0..mb.num_links {
        body_velocities[links[k as usize].rb_id as usize] =
            ws_vel(links_workspace, wa, k, WS_RB_VELS);
    }
}
