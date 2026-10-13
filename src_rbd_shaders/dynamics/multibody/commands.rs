//! GPU-resident articulation commands. Selection records must be unique.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::{MbImpulseJointBuilder, MultibodyLinkStatic};
use crate::Pose;
use crate::dynamics::SPATIAL_DIM;
use crate::utils::BatchIndices;

/// Position and velocity targets of an already configured motor.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ArticulationMotorTarget {
    pub position: f32,
    pub velocity: f32,
}

/// One selected batch/link, with a mask of motor axes to update.
/// Link ids use the articulation traversal order, not rigid-body ids.
/// `axes == 0` marks an inactive slot in a persistent command buffer.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ArticulationMotorCommand {
    pub batch: u32,
    pub link: u32,
    pub axes: u32,
    pub _padding: u32,
    pub targets: [ArticulationMotorTarget; SPATIAL_DIM],
}

/// Update an existing external joint, preserving topology and solver coloring.
/// `joint` is the source impulse-joint id (`MbImpulseJointBuilder::joint_id`).
/// This id is assigned before color sorting; it is not a Rapier handle or a
/// sorted builder slot. `joint == u32::MAX` marks an inactive command slot.
/// Frame conventions match the resident builders: body-origin local space
/// for an articulation link, COM-local space for a free body, and world space
/// for a fixed side (whose body transform is already folded into the frame).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ArticulationJointCommand {
    pub batch: u32,
    pub joint: u32,
    pub locked_axes: u32,
    pub _padding: u32,
    pub frame_a: Pose,
    pub frame_b: Pose,
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_motor_commands(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    commands: &[ArticulationMotorCommand],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links: &mut [MultibodyLinkStatic],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    if id.x as usize >= commands.len() {
        return;
    }
    let command = commands.read(id.x as usize);
    if command.axes == 0 {
        return;
    }
    let index = batch_ids.mbi(command.batch, command.link as usize);
    let mut link = links.read(index);
    for axis in 0..SPATIAL_DIM {
        if command.axes & (1 << axis) != 0 {
            let target = command.targets.read(axis);
            link.data.motors[axis].target_pos = target.position;
            link.data.motors[axis].target_vel = target.velocity;
        }
    }
    links.write(index, link);
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_joint_commands(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    commands: &[ArticulationJointCommand],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    builders: &mut [MbImpulseJointBuilder],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    if id.x as usize >= commands.len() {
        return;
    }
    let command = commands.read(id.x as usize);
    if command.joint == u32::MAX {
        return;
    }
    for joint in 0..batch_ids.mb_imp_joints_batch_capacity {
        let index = batch_ids.mbi(command.batch, joint as usize);
        let mut builder = builders.read(index);
        if builder.joint_id == command.joint {
            builder.joint.local_frame_a = command.frame_a;
            builder.joint.local_frame_b = command.frame_b;
            builder.joint.locked_axes = command.locked_axes;
            builders.write(index, builder);
        }
    }
}
