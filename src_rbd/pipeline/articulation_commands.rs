use crate::dynamics::{
    ArticulationJointCommand, ArticulationMotorCommand, GpuArticulationCommands,
};
use crate::pipeline::RbdState;
use khal::backend::{GpuBackendError, GpuEncoder};
use vortx::tensor::Tensor;

impl RbdState {
    /// Set joint-space state in selected batches and immediately publish link
    /// poses and COM velocities, without advancing time. All inputs reside on
    /// the GPU. `selected` has one u32 per batch (nonzero means reset).
    /// `roots` contains one native world pose per multibody, batch-interleaved.
    /// Coordinates and velocities contain `dofs_per_batch * num_batches`
    /// floats in the live DOF interleave, with free axes in spatial axis order.
    /// Root coordinates are ignored in favor of `roots`. In 3D, three free
    /// angular coordinates specify a scaled-axis rotation. Recording uses the
    /// persistent kernels and existing workspaces, with no uploads/readbacks.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_articulation_reset(
        &mut self,
        encoder: &mut GpuEncoder,
        kernels: &GpuArticulationCommands,
        selected: &Tensor<u32>,
        roots: &Tensor<crate::math::Pose>,
        coordinates: &Tensor<f32>,
        velocities: &Tensor<f32>,
    ) -> Result<(), GpuBackendError> {
        assert_eq!(selected.len(), self.num_batches as u64);
        assert_eq!(
            roots.len(),
            self.multibodies.multibodies_per_batch() as u64 * self.num_batches as u64
        );
        let dofs = self.multibodies.dofs_per_batch() as u64 * self.num_batches as u64;
        assert_eq!(coordinates.len(), dofs);
        assert_eq!(velocities.len(), dofs);
        kernels._encode_reset_state(
            encoder,
            &mut self.multibodies,
            &self.batch_indices,
            selected,
            roots,
            coordinates,
            velocities,
            &mut self.body_poses,
            &mut self.vels,
        )
    }
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
