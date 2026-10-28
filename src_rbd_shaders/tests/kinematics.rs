use crate::dynamics::{gpu_integrate_linearized, gpu_solver_cleanup, RbdSimParams, Velocity};
use crate::utils::BatchIndices;
use crate::{Pose, Vector};
use glamx::UVec3;

#[test]
fn zero_mass_body_integrates_prescribed_velocity() {
    let indices = BatchIndices { num_batches: 1, bodies_len: 2, colliders_batch_capacity: 2, ..Default::default() };
    let mut velocities = [Velocity::default(); 2];
    velocities[0].linear = Vector::X * 2.;
    let mut solver = [Velocity::default(); 2];
    let mut counts = [0; 2];
    let mut poses = [Pose::identity(); 2];
    let params = RbdSimParams { dt: 0.1, ..Default::default() };
    for _ in 0..10 {
        for i in 0..2 {
            gpu_solver_cleanup(UVec3::new(i, 0, 0), UVec3::ONE, &mut counts, &mut solver, &velocities, &indices);
            gpu_integrate_linearized(UVec3::new(i, 0, 0), &mut poses, &mut solver, &params, &indices);
        }
    }
    assert!((poses[0].translation - Vector::X * 2.).length() < 1e-5);
    assert_eq!(poses[1].translation, Vector::ZERO);
}
