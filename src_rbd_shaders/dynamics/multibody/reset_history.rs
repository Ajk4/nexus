//! Selectively invalidate persistent solver history after a joint-space reset.

use super::{
    MAX_CONTACT_SENSORS, MbImpulseJointConstraint, MultibodyContactConstraint, MultibodyInfo,
    MultibodyJointConstraint,
};
use crate::Pose;
use crate::dynamics::{LocalMassProperties, TwoBodyConstraint, Velocity};
use crate::utils::BatchIndices;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_reset_history(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] selected: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] infos: &mut [MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    joints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contacts: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    old_contacts: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    external: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] delay: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] sensors: &mut [f32],
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
    let mut mb = infos.read(id.x as usize);
    for s in 0..mb.max_constraints {
        joints.write(
            batch_ids.mbi(batch, (mb.first_constraint + s) as usize),
            MultibodyJointConstraint::default(),
        );
    }
    for s in 0..mb.contact_constraint_count {
        contacts.write(
            (mb.contact_constraint_start + s) as usize,
            MultibodyContactConstraint::default(),
        );
    }
    for s in 0..mb.old_contact_constraint_count {
        old_contacts.write(
            (mb.old_contact_constraint_start + s) as usize,
            MultibodyContactConstraint::default(),
        );
    }
    mb.contact_constraint_count = 0;
    mb.old_contact_constraint_count = 0;
    mb.contact_index_len = 0;
    infos.write(id.x as usize, mb);
    for s in 0..MAX_CONTACT_SENSORS {
        sensors.write(
            id.x as usize * MAX_CONTACT_SENSORS as usize + s as usize,
            0.0,
        );
    }
    // These banks belong to the whole batch. Only its first articulation
    // clears them, so independent articulation invocations never race.
    if id.x / nb == 0 {
        for s in 0..external.len() as u32 / nb {
            external.write(
                batch_ids.mbi(batch, s as usize),
                MbImpulseJointConstraint::default(),
            );
        }
        let base = batch as usize * (2 + batch_ids.links_batch_capacity) as usize;
        delay.write(base, delay.read(base + 1) + 1.0);
        for k in 0..batch_ids.links_batch_capacity {
            delay.write(base + 2 + k as usize, 0.0);
        }
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_reset_rigid_contact_history(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] selected: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts: &mut [TwoBodyConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    if id.x as usize >= contacts.len() {
        return;
    }
    let mut contact = contacts.read(id.x as usize);
    if contact.len == 0 {
        return;
    }
    let batch = batch_ids.collider_batch(contact.solver_body_a);
    if selected.read(batch as usize) != 0 {
        contact = TwoBodyConstraint::default();
        // Prefix sums and old body-index lists remain valid, but can no
        // longer match this entry for impulse transfer or color reuse.
        contact.solver_body_a = u32::MAX;
        contact.solver_body_b = u32::MAX;
        contacts.write(id.x as usize, contact);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_reset_solver_state(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] selected: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] velocities: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] local_mass: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] solver_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] solver_velocities: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] solver_increments: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
) {
    if id.x >= batch_ids.bodies_len * batch_ids.num_batches {
        return;
    }
    if selected.read(batch_ids.collider_batch(id.x) as usize) == 0 {
        return;
    }
    let index = id.x as usize;
    let pose = poses.read(index);
    solver_poses.write(
        index,
        Pose::from_parts(pose * local_mass.read(index).com, pose.rotation),
    );
    solver_velocities.write(index, velocities.read(index));
    solver_increments.write(index, Velocity::default());
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_reset_collider_poses(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] selected: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] local_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] parents: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] world_poses: &mut [Pose],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    if id.x >= batch_ids.colliders_len * batch_ids.num_batches {
        return;
    }
    if selected.read(batch_ids.collider_batch(id.x) as usize) == 0 {
        return;
    }
    let index = id.x as usize;
    let parent = parents.read(index);
    let local = local_poses.read(index);
    let pose = if parent == u32::MAX {
        local
    } else {
        body_poses.read(parent as usize) * local
    };
    world_poses.write(index, pose);
}
