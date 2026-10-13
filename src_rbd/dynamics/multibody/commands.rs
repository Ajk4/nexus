//! Persistent kernels for commands produced on the GPU.

use super::GpuMultibodySet;
use crate::math::Pose;
use crate::shaders::dynamics::{
    GpuMbJointCommands, GpuMbMotorCommands, GpuMbPublishResetState, GpuMbResetState, Velocity,
};
use crate::shaders::utils::BatchIndices;
use khal::Shader;
use khal::backend::{Encoder, GpuBackend, GpuBackendError, GpuEncoder};
use vortx::tensor::Tensor;

pub use crate::shaders::dynamics::{
    ArticulationJointCommand, ArticulationMotorCommand, ArticulationMotorTarget,
};

/// Load once at setup, then record commands from persistent GPU input buffers.
/// Recording does not upload, allocate tensors, submit, or read back state.
#[derive(Shader)]
pub struct GpuArticulationCommands {
    motors: GpuMbMotorCommands,
    joints: GpuMbJointCommands,
    reset: GpuMbResetState,
    publish: GpuMbPublishResetState,
}

impl GpuArticulationCommands {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn _encode_reset_state(
        &self,
        encoder: &mut GpuEncoder,
        mb: &mut GpuMultibodySet,
        batch_indices: &Tensor<BatchIndices>,
        selected: &Tensor<u32>,
        roots: &Tensor<Pose>,
        coordinates: &Tensor<f32>,
        velocities: &Tensor<f32>,
        poses: &mut Tensor<Pose>,
        body_velocities: &mut Tensor<Velocity>,
    ) -> Result<(), GpuBackendError> {
        let threads = mb.num_active_multibodies * mb.num_batches;
        let mut pass = encoder.begin_pass("[RBD] mb/reset-state", None);
        self.reset.call(
            &mut pass,
            threads,
            selected,
            roots,
            coordinates,
            velocities,
            &mb.multibody_info,
            &mut mb.links_static,
            &mut mb.links_workspace,
            &mut mb.dof_state,
            batch_indices,
        )?;
        self.publish.call(
            &mut pass,
            threads,
            selected,
            roots,
            &mb.multibody_info,
            &mb.links_static,
            &mut mb.links_workspace,
            &mb.dof_state,
            poses,
            body_velocities,
            batch_indices,
        )
    }
    /// Load the command kernels before recording simulation work.
    pub fn new(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Self::from_backend(backend)
    }

    /// Each (batch, link) must occur once. Only targets of configured motors
    /// change; gains, enabled axes and accumulated impulses are preserved.
    /// Writes bypass the host motor mirror; do not mix with host motor setters.
    pub fn encode_motor_targets(
        &self,
        encoder: &mut GpuEncoder,
        multibodies: &mut GpuMultibodySet,
        batch_indices: &Tensor<BatchIndices>,
        commands: &Tensor<ArticulationMotorCommand>,
    ) -> Result<(), GpuBackendError> {
        let mut pass = encoder.begin_pass("[RBD] mb/motor-commands", None);
        self.motors.call(
            &mut pass,
            commands.len() as u32,
            commands,
            &mut multibodies.links_static,
            batch_indices,
        )
    }

    /// Each (batch, source joint id) must occur once. Joint endpoints remain
    /// unchanged, so existing capacities and graph coloring remain valid.
    pub fn encode_impulse_joints(
        &self,
        encoder: &mut GpuEncoder,
        multibodies: &mut GpuMultibodySet,
        batch_indices: &Tensor<BatchIndices>,
        commands: &Tensor<ArticulationJointCommand>,
    ) -> Result<(), GpuBackendError> {
        let mut pass = encoder.begin_pass("[RBD] mb/joint-commands", None);
        self.joints.call(
            &mut pass,
            commands.len() as u32,
            commands,
            &mut multibodies.mb_imp_joint_builders,
            batch_indices,
        )
    }
}
