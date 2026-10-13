//! Persistent kernels for commands produced on the GPU.

use super::GpuMultibodySet;
use crate::shaders::dynamics::{GpuMbJointCommands, GpuMbMotorCommands};
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
}

impl GpuArticulationCommands {
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
