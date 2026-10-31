use super::{RbdCapacities, RbdPipeline, RbdState};
use crate::rapier::prelude::*;
use crate::shaders::dynamics::RbdSimParams;
use khal::backend::GpuBackend;

#[test]
fn angular_motor_takes_a_path_allowed_by_limits() {
    let limit = std::f32::consts::TAU / 3.0;
    for bounded in [false, true] {
        for sign in [-1.0, 1.0] {
            let backend = GpuBackend::Cpu;
            let mut bodies = RigidBodySet::new();
            let mut colliders = ColliderSet::new();
            let mut joints = ImpulseJointSet::new();
            let multibodies = MultibodyJointSet::new();
            let anchor = bodies.insert(RigidBodyBuilder::fixed());
            colliders.insert_with_parent(
                ColliderBuilder::ball(0.1).collision_groups(InteractionGroups::none()),
                anchor,
                &mut bodies,
            );
            let rotor = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .rotation(-sign * limit)
                    .can_sleep(false),
            );
            colliders.insert_with_parent(ColliderBuilder::ball(0.5).mass(1.0), rotor, &mut bodies);
            let mut joint = RevoluteJointBuilder::new()
                .motor_model(MotorModel::ForceBased)
                .motor_position(sign * limit, 10.0, 2.0)
                .motor_max_force(40.0);
            if bounded {
                joint = joint.limits([-limit, limit]);
            }
            joints.insert(anchor, rotor, joint, true);
            let params = RbdSimParams {
                dt: 1.0 / 60.0,
                num_solver_iterations: 6,
                ..RbdSimParams::tgs_soft()
            };
            let mut state = RbdState::from_rapier(
                &backend,
                &[(&bodies, &colliders, &joints, &multibodies, &params)],
                RbdCapacities {
                    batches: 1,
                    body_capacity: 2,
                    collisions_capacity: 16,
                    ..Default::default()
                },
            );
            state.set_gravity(&backend, [0.0; 3]);
            let pipeline = RbdPipeline::new(&backend).unwrap();
            let mut travelled = 0.0;
            let mut previous = -sign * limit;
            for _ in 0..300 {
                pipeline.step(&backend, &mut state, None).unwrap();
                let angle = state.body_poses().buffer().unwrap_slice()[1]
                    .rotation
                    .angle();
                let delta = angle - previous;
                travelled += delta.sin().atan2(delta.cos());
                previous = angle;
                if bounded {
                    assert!(angle.abs() <= limit + 0.01, "crossed a stop: {angle}");
                }
            }
            assert!(
                (previous - sign * limit).abs() < 0.01,
                "missed target: {previous}"
            );
            let expected = if bounded {
                sign * 2.0 * limit
            } else {
                -sign * limit
            };
            assert!(
                (travelled - expected).abs() < 0.01,
                "wrong path: {travelled} expected {expected}"
            );
        }
    }
}
