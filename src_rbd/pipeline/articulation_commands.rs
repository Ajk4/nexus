use crate::dynamics::{
    ArticulationJointCommand, ArticulationMotorCommand, GpuArticulationCommands,
};
use crate::pipeline::RbdState;
use khal::backend::{GpuBackendError, GpuEncoder};
use vortx::tensor::Tensor;

impl RbdState {
    /// Record GPU-resident motor target updates. See
    /// [`GpuArticulationCommands::encode_motor_targets`] for input semantics.
    pub fn encode_articulation_motor_targets(
        &mut self,
        encoder: &mut GpuEncoder,
        kernels: &GpuArticulationCommands,
        commands: &Tensor<ArticulationMotorCommand>,
    ) -> Result<(), GpuBackendError> {
        kernels.encode_motor_targets(
            encoder,
            &mut self.multibodies,
            &self.batch_indices,
            commands,
        )
    }

    /// Record GPU-resident frame and locked-axis updates for external joints.
    pub fn encode_articulation_impulse_joints(
        &mut self,
        encoder: &mut GpuEncoder,
        kernels: &GpuArticulationCommands,
        commands: &Tensor<ArticulationJointCommand>,
    ) -> Result<(), GpuBackendError> {
        kernels.encode_impulse_joints(
            encoder,
            &mut self.multibodies,
            &self.batch_indices,
            commands,
        )
    }
}
