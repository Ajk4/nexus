//! Native articulation probes, with the same planar fixture run through 3D.
//!
//! Run `cargo test -p nexus_rbd2d --features cpu articulation_ -- --nocapture`;
//! set `NEXUS_TEST_GPU=1` for WebGPU. Substitute `nexus_rbd3d` for the planar
//! baseline. Run the ignored `articulation_benchmark` with `--release` for
//! throughput measurements; it separates host submission and GPU pass times.
//!
//! The floating fixture has three unit-mass balls (radius 0.15), a revolute
//! joint and a prismatic joint. The zero-gravity conservation probe rotates
//! and telescopes simultaneously for 240 steps at dt=1/600, retaining the
//! 0.001 m COM-drift and 0.001 kg*m/s linear-momentum limits. The smaller
//! timestep is needed by the strict bound in the 3D baseline too.
//!
//! Release measurements, 2026-10-13: RTX 5070, NVIDIA 580.173.02, Vulkan.
//! Floating chains alternate revolute/prismatic motors, with contacts disabled
//! and minimal contact-bank capacities. Warm up ten steps; report the median
//! of three synchronized twenty-step windows. Host-call times include encoding
//! and submission. GPU pass times are separate medians of five profiled steps;
//! the articulation subtotal sums passes whose labels contain `mb`.
//!
//! | Links | Batches | 2D DOFs | 2D wall ms | 3D wall ms | 2D articulation GPU ms | 3D articulation GPU ms |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | 3 | 4096 | 5 | 3.415 | 4.168 | 1.073 | 1.844 |
//! | 12 | 4096 | 14 | 8.540 | 13.695 | 5.620 | 10.599 |
//! | 24 | 4096 | 26 | 30.203 | 42.101 | 27.402 | 38.821 |
//! | 48 | 1024 | 50 | 53.641 | 64.550 | 53.016 | 63.823 |
//!
//! Larger models become GPU-bound: the 24-link case spends 27.4 ms in
//! articulation passes, versus about 0.64 ms in host calls. Dense Jacobian and
//! mass-matrix assembly scales strongly with the number of links and DOFs.
//! The three-link case peaks near 1.2M end-to-end model-steps/s; the 24-link
//! case near 136k. Core buffers use 1880/4256 bytes per three-link 2D/3D model,
//! or 30272/58772 bytes per 24-link model. These counts exclude constraint banks,
//! uniforms, and rigid-body buffers. Planar link workspaces are 144 bytes/link.
//!
//! Disassembling the final 2D FK, gravity, integration,
//! contact, and external-joint SPIR-V modules showed only scalar, Vec2, and
//! aligned Vec4 floating-point types, with no Vec3 or matrix types. UVec3
//! remains necessary for GPU invocation IDs.
use crate::math::{Pose, Vector};

#[cfg(feature = "dim2")]
fn _v(x: f32, y: f32) -> Vector {
    Vector::new(x, y)
}
#[cfg(feature = "dim3")]
fn _v(x: f32, y: f32) -> Vector {
    Vector::new(x, y, 0.0)
}
fn _hinge() -> RevoluteJointBuilder {
    #[cfg(feature = "dim2")]
    {
        RevoluteJointBuilder::new()
    }
    #[cfg(feature = "dim3")]
    {
        RevoluteJointBuilder::new(Vector::Z)
    }
}
#[cfg(feature = "dim2")]
const ANG_AXIS: usize = 2;
#[cfg(feature = "dim3")]
const ANG_AXIS: usize = 3;
use crate::pipeline::{RbdCapacities, RbdPipeline, RbdResizePolicy, RbdState};
use crate::rapier::prelude::*;
use crate::shaders::dynamics::{JointMotor, RbdSimParams, Velocity};
use khal::backend::{Backend, GpuBackend};

async fn _backend() -> GpuBackend {
    if std::env::var_os("NEXUS_TEST_GPU").is_some() {
        GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap())
    } else {
        GpuBackend::Cpu
    }
}

struct Scene {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    mb: MultibodyJointSet,
    joints: ImpulseJointSet,
    handles: Vec<RigidBodyHandle>,
}

fn _scene(fixed: bool, motors: bool, ground: bool, external: bool) -> Scene {
    let mut s = Scene {
        bodies: RigidBodySet::new(),
        colliders: ColliderSet::new(),
        mb: MultibodyJointSet::new(),
        joints: ImpulseJointSet::new(),
        handles: Vec::new(),
    };
    for i in 0..3 {
        let builder = if fixed && i == 0 {
            RigidBodyBuilder::fixed()
        } else {
            RigidBodyBuilder::dynamic()
        };
        let h = s.bodies.insert(builder.translation(_v(i as f32, 2.0)));
        let co = ColliderBuilder::ball(0.15)
            .mass(1.0)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                if ground { Group::GROUP_2 } else { Group::NONE },
                InteractionTestMode::And,
            ));
        s.colliders.insert_with_parent(co, h, &mut s.bodies);
        s.handles.push(h);
    }
    let mut hinge = _hinge()
        .local_anchor1(_v(0.5, 0.0))
        .local_anchor2(_v(-0.5, 0.0))
        .limits([-0.5, 0.5]);
    let mut slider = PrismaticJointBuilder::new(Vector::X)
        .local_anchor1(_v(0.5, 0.0))
        .local_anchor2(_v(-0.5, 0.0))
        .limits([-0.2, 0.2]);
    if motors {
        hinge = hinge.motor_position(0.35, 40.0, 5.0).motor_max_force(10.0);
        slider = slider.motor_position(0.15, 40.0, 5.0).motor_max_force(10.0);
    }
    s.mb.insert(s.handles[0], s.handles[1], hinge, true)
        .unwrap();
    s.mb.insert(s.handles[1], s.handles[2], slider, true)
        .unwrap();
    if ground {
        let h = s
            .bodies
            .insert(RigidBodyBuilder::fixed().translation(_v(0.0, -0.5)));
        s.colliders.insert_with_parent(
            _ground().collision_groups(InteractionGroups::new(
                Group::GROUP_2,
                Group::GROUP_1,
                InteractionTestMode::And,
            )),
            h,
            &mut s.bodies,
        );
    }
    if external {
        let h = s
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(_v(2.0, 1.5)));
        s.colliders.insert_with_parent(
            ColliderBuilder::ball(0.1)
                .mass(0.5)
                .collision_groups(InteractionGroups::new(
                    Group::NONE,
                    Group::NONE,
                    InteractionTestMode::And,
                )),
            h,
            &mut s.bodies,
        );
        s.joints
            .insert(s.handles[2], h, _hinge().local_anchor2(_v(0.0, 0.5)), true);
        s.handles.push(h);
    }
    for (_, b) in s.bodies.iter_mut() {
        b.recompute_mass_properties_from_colliders(&s.colliders);
    }
    s
}

fn _state(
    backend: &GpuBackend,
    scenes: &[Scene],
    gravity: f32,
    contact_capacity: u32,
) -> (RbdState, RbdPipeline) {
    let params = RbdSimParams {
        dt: 1.0 / 600.0,
        ..RbdSimParams::tgs_soft()
    };
    let refs: Vec<_> = scenes
        .iter()
        .map(|s| (&s.bodies, &s.colliders, &s.joints, &s.mb, &params))
        .collect();
    let mut state = RbdState::from_rapier(
        backend,
        &refs,
        RbdCapacities {
            batches: scenes.len() as u32,
            collisions_capacity: contact_capacity,
            mb_contact_constraints_capacity: contact_capacity,
            collisions_resize_policy: RbdResizePolicy::Fixed,
            solver_colors_resize_policy: RbdResizePolicy::Fixed,
            ..Default::default()
        },
    );
    state.set_gravity(backend, [0.0, gravity, 0.0]);
    (state, RbdPipeline::new(backend).unwrap())
}

async fn _poses(backend: &GpuBackend, state: &RbdState) -> Vec<Pose> {
    backend.synchronize().unwrap();
    backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap()
}

#[futures_test::test]
#[should_panic(expected = "articulation topology differs")]
async fn articulation_batch_rejects_different_connections() {
    let backend = _backend().await;
    let a = _scene(false, false, false, false);
    let mut b = _scene(false, false, false, false);
    b.mb = MultibodyJointSet::new();
    b.mb.insert(b.handles[0], b.handles[1], _hinge(), true)
        .unwrap();
    b.mb.insert(
        b.handles[0],
        b.handles[2],
        PrismaticJointBuilder::new(Vector::X),
        true,
    )
    .unwrap();
    _state(&backend, &[a, b], 0., 32);
}

#[futures_test::test]
#[should_panic(expected = "articulation topology differs")]
async fn articulation_batch_rejects_different_joint_axes() {
    let backend = _backend().await;
    let a = _scene(false, false, false, false);
    let mut b = _scene(false, false, false, false);
    b.mb = MultibodyJointSet::new();
    b.mb.insert(
        b.handles[0],
        b.handles[1],
        PrismaticJointBuilder::new(Vector::X),
        true,
    )
    .unwrap();
    b.mb.insert(b.handles[1], b.handles[2], _hinge(), true)
        .unwrap();
    _state(&backend, &[a, b], 0., 32);
}

#[futures_test::test]
async fn articulation_batch_accepts_different_physical_parameters() {
    let backend = _backend().await;
    let a = _scene(false, true, false, false);
    let mut b = _scene(false, false, false, false);
    for (_, body) in b.bodies.iter_mut() {
        body.set_translation(body.translation() + _v(2., 3.), false);
        body.set_linvel(_v(0.2, -0.3), false);
        body.set_additional_mass(0.7, false);
    }
    let (mut state, pipeline) = _state(&backend, &[a, b], 0., 32);
    pipeline.step(&backend, &mut state, None).unwrap();
    assert!(
        _poses(&backend, &state)
            .await
            .iter()
            .all(|p| p.translation.is_finite())
    );
}

#[futures_test::test]
async fn articulation_batches_match_independent_steps() {
    let backend = _backend().await;
    let make_scene = |index: usize| {
        let mut scene = _scene(false, true, true, true);
        let offset = _v(0., if index % 2 == 0 { -2. } else { 0. });
        for &handle in &scene.handles {
            let body = &mut scene.bodies[handle];
            body.set_translation(body.translation() + offset, false);
            body.set_linvel(_v(index as f32 * 0.001, 0.), false);
        }
        scene
    };
    let scenes: Vec<_> = (0..65).map(make_scene).collect();
    let (mut batched, pipeline) = _state(&backend, &scenes, -9.81, 4096);
    for _ in 0..12 {
        pipeline.step(&backend, &mut batched, None).unwrap();
    }
    let poses = _poses(&backend, &batched).await;
    for index in [0usize, 1, 32, 64] {
        let (mut single, pipeline) = _state(&backend, &[make_scene(index)], -9.81, 64);
        for _ in 0..12 {
            pipeline.step(&backend, &mut single, None).unwrap();
        }
        let reference = _poses(&backend, &single).await;
        for (body, pose) in reference.iter().enumerate() {
            assert!(
                poses[body * 65 + index]
                    .translation
                    .distance(pose.translation)
                    < 0.0001,
                "batch {index}, body {body}: batched pose differs from independent simulation"
            );
        }
    }
}

#[futures_test::test]
async fn articulation_gpu_control_commands() {
    use crate::dynamics::{
        ArticulationJointCommand, ArticulationMotorCommand, ArticulationMotorTarget,
        GpuArticulationCommands,
    };
    use khal::BufferUsages;
    use vortx::tensor::Tensor;
    let backend = _backend().await;
    let scenes = [
        _scene(false, true, false, true),
        _scene(false, true, false, true),
    ];
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
    let kernels = GpuArticulationCommands::new(&backend).unwrap();
    let mut motor = ArticulationMotorCommand {
        batch: 1,
        link: 1,
        axes: 1 << ANG_AXIS,
        ..Default::default()
    };
    motor.targets[ANG_AXIS] = ArticulationMotorTarget {
        position: -0.25,
        velocity: -0.4,
    };
    let motors = Tensor::vector(
        &backend,
        &[
            motor,
            ArticulationMotorCommand {
                batch: u32::MAX,
                link: u32::MAX,
                ..Default::default()
            },
        ],
        BufferUsages::STORAGE,
    )
    .unwrap();
    let old = _read(&backend, state.multibodies().impulse_joint_builders()).await;
    let joints = Tensor::vector(
        &backend,
        &[
            ArticulationJointCommand {
                batch: 1,
                joint: old[1].joint_id,
                locked_axes: 0,
                frame_a: Pose::from_translation(_v(0.1, 0.2)),
                frame_b: Pose::from_translation(_v(-0.1, -0.2)),
                _padding: 0,
            },
            ArticulationJointCommand {
                batch: u32::MAX,
                joint: u32::MAX,
                ..Default::default()
            },
        ],
        BufferUsages::STORAGE,
    )
    .unwrap();
    let mut encoder = backend.begin_encoding();
    state
        .encode_articulation_motor_targets(&mut encoder, &kernels, &motors)
        .unwrap();
    state
        .encode_articulation_impulse_joints(&mut encoder, &kernels, &joints)
        .unwrap();
    backend.submit(encoder).unwrap();
    let links = _read(&backend, state.multibodies().links_static()).await;
    assert_eq!(links[2].data.motors[ANG_AXIS].target_pos, 0.35);
    assert_eq!(links[3].data.motors[ANG_AXIS].target_pos, -0.25);
    assert_eq!(links[3].data.motors[ANG_AXIS].target_vel, -0.4);
    let updated = _read(&backend, state.multibodies().impulse_joint_builders()).await;
    assert_eq!(updated[0].joint.local_frame_a, old[0].joint.local_frame_a);
    assert_eq!(updated[0].joint.locked_axes, old[0].joint.locked_axes);
    assert_eq!(updated[1].joint.local_frame_a.translation, _v(0.1, 0.2));
    assert_eq!(updated[1].joint.local_frame_b.translation, _v(-0.1, -0.2));
    assert_eq!(updated[1].joint.locked_axes, 0);
    for _ in 0..12 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let poses = _poses(&backend, &state).await;
    assert!(poses.iter().all(|pose| pose.translation.is_finite()));
}

async fn _read<T: bytemuck::Pod + Send + Sync>(
    backend: &GpuBackend,
    tensor: &vortx::tensor::Tensor<T>,
) -> Vec<T> {
    let mut values = bytemuck::zeroed_vec(tensor.len() as usize);
    backend
        .slow_read_buffer(tensor.buffer(), &mut values)
        .await
        .unwrap();
    values
}

fn _rotation(angle: f32) -> crate::math::Rotation {
    #[cfg(feature = "dim2")]
    {
        crate::math::Rotation::from_angle(angle)
    }
    #[cfg(feature = "dim3")]
    {
        crate::math::Rotation::from_rotation_z(angle)
    }
}

fn _branched_scene() -> Scene {
    let mut scene = Scene {
        bodies: RigidBodySet::new(),
        colliders: ColliderSet::new(),
        mb: MultibodyJointSet::new(),
        joints: ImpulseJointSet::new(),
        handles: Vec::new(),
    };
    for body in 0..51 {
        let position = if body == 0 {
            _v(0.0, 3.0)
        } else {
            let branch = (body - 1) / 10;
            let depth = (body - 1) % 10;
            _v((depth + 1) as f32 * 0.4, 3.0 + (branch as f32 - 2.0) * 0.6)
        };
        let handle = scene
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(position));
        scene.colliders.insert_with_parent(
            ColliderBuilder::ball(0.07)
                .mass(1.0)
                .collision_groups(InteractionGroups::none()),
            handle,
            &mut scene.bodies,
        );
        scene.handles.push(handle);
        if body != 0 {
            let depth = (body - 1) % 10;
            let parent = if depth == 0 { 0 } else { body - 1 };
            let offset = if depth == 0 {
                ((body - 1) / 10) as f32 * 0.6 - 1.2
            } else {
                0.0
            };
            let joint: GenericJoint = if depth % 2 == 0 {
                _hinge()
                    .local_anchor1(_v(0.2, offset))
                    .local_anchor2(_v(-0.2, 0.0))
                    .limits([-0.5, 0.5])
                    .motor_position(0.0, 20.0, 4.0)
                    .motor_max_force(10.0)
                    .into()
            } else {
                PrismaticJointBuilder::new(Vector::X)
                    .local_anchor1(_v(0.2, offset))
                    .local_anchor2(_v(-0.2, 0.0))
                    .limits([-0.2, 0.2])
                    .motor_position(0.0, 20.0, 4.0)
                    .motor_max_force(10.0)
                    .into()
            };
            scene
                .mb
                .insert(scene.handles[parent], handle, joint, true)
                .unwrap();
        }
    }
    for branch in 0..5 {
        let tip = scene.handles[10 + branch * 10];
        let anchor = scene.bodies.insert(
            RigidBodyBuilder::fixed().translation(scene.bodies[tip].position().translation),
        );
        scene.colliders.insert_with_parent(
            ColliderBuilder::ball(0.02).collision_groups(InteractionGroups::none()),
            anchor,
            &mut scene.bodies,
        );
        scene.joints.insert(
            tip,
            anchor,
            GenericJointBuilder::new(JointAxesMask::empty()),
            true,
        );
    }
    for (_, body) in scene.bodies.iter_mut() {
        body.recompute_mass_properties_from_colliders(&scene.colliders);
    }
    scene
}

/// Independent planar branch kinematics, evaluated twice for a velocity
/// finite difference. Inputs are indexed by physical body, not GPU link order.
fn _branched_poses(root: Pose, coordinates: &[f32]) -> Vec<Pose> {
    let mut poses = vec![root; 51];
    for branch in 0..5 {
        let mut parent = root;
        for depth in 0..10 {
            let body = 1 + branch * 10 + depth;
            let offset = if depth == 0 {
                branch as f32 * 0.6 - 1.2
            } else {
                0.0
            };
            let joint = if depth % 2 == 0 {
                Pose::from_parts(Vector::ZERO, _rotation(coordinates[body]))
            } else {
                Pose::from_translation(_v(coordinates[body], 0.0))
            };
            parent = parent
                * Pose::from_translation(_v(0.2, offset))
                * joint
                * Pose::from_translation(_v(0.2, 0.0));
            poses[body] = parent;
        }
    }
    poses
}

#[futures_test::test]
async fn articulation_gpu_branched_control_and_reset() {
    use crate::dynamics::{
        ArticulationJointCommand, ArticulationMotorCommand, ArticulationMotorTarget,
        GpuArticulationCommands,
    };
    use crate::shaders::dynamics::SPATIAL_DIM;
    use khal::BufferUsages;
    use vortx::tensor::Tensor;
    let backend = _backend().await;
    let kernels = GpuArticulationCommands::new(&backend).unwrap();
    let scenes = [_branched_scene(), _branched_scene(), _branched_scene()];
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
    assert_eq!(state.multibodies().links_per_batch(), 51);
    assert_eq!(
        state.multibodies().dofs_per_batch(),
        SPATIAL_DIM as u32 + 50
    );
    for _ in 0..3 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let before = _poses(&backend, &state).await;
    let before_vels = _read(&backend, state.body_velocities()).await;
    let before_dofs = _read(&backend, state.multibodies().dof_state()).await;
    let links = _read(&backend, state.multibodies().links_static()).await;
    let builders = _read(&backend, state.multibodies().impulse_joint_builders()).await;
    let dpb = state.multibodies().dofs_per_batch() as usize;
    let mut coordinates = vec![0.0; dpb * 3];
    let mut velocities = vec![0.0; dpb * 3];
    velocities[1] = 0.2;
    velocities[4] = -0.1;
    velocities[(SPATIAL_DIM - 1) * 3 + 1] = 0.07;
    let mut body_q = vec![0.0; 51];
    let mut body_qd = vec![0.0; 51];
    let mut motors = Vec::new();
    for link in 1..51 {
        let descriptor = links[link * 3 + 1];
        let body = descriptor.rb_id as usize;
        let revolute = (body - 1) % 10 % 2 == 0;
        let q = if revolute { 0.03 } else { 0.01 };
        let qd = if revolute { 0.08 } else { 0.02 };
        coordinates[descriptor.assembly_id as usize * 3 + 1] = q;
        velocities[descriptor.assembly_id as usize * 3 + 1] = qd;
        body_q[body] = q;
        body_qd[body] = qd;
        let axis =
            (!descriptor.data.locked_axes & ((1 << SPATIAL_DIM) - 1)).trailing_zeros() as usize;
        let mut command = ArticulationMotorCommand {
            batch: 1,
            link: link as u32,
            axes: 1 << axis,
            ..Default::default()
        };
        command.targets[axis] = ArticulationMotorTarget {
            position: q + 0.04,
            velocity: qd + 0.1,
        };
        motors.push(command);
    }
    let root = Pose::from_parts(_v(0.0, 3.0), _rotation(0.2));
    let expected = _branched_poses(root, &body_q);
    let epsilon = 0.001;
    let next_q: Vec<_> = body_q
        .iter()
        .zip(&body_qd)
        .map(|(q, qd)| q + qd * epsilon)
        .collect();
    let next = _branched_poses(
        Pose::from_parts(
            root.translation + _v(0.2, -0.1) * epsilon,
            _rotation(0.2 + 0.07 * epsilon),
        ),
        &next_q,
    );
    let mut joint_commands = Vec::new();
    for builder in builders.iter().skip(1).step_by(3) {
        let body = links[builder.side_a_link as usize * 3 + 1].rb_id as usize;
        // The anchor remains fixed. Move its local frame to meet the reset
        // tip pose, then lock translation while every branch motor is active.
        let frame_b = expected[body];
        joint_commands.push(ArticulationJointCommand {
            batch: 1,
            joint: builder.joint_id,
            locked_axes: (1 << crate::math::DIM) - 1,
            frame_a: Pose::IDENTITY,
            frame_b,
            _padding: 0,
        });
    }
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    let selected = Tensor::vector(&backend, &[0u32, 1, 0], storage).unwrap();
    let roots = Tensor::vector(&backend, &[root; 3], storage).unwrap();
    let coords = Tensor::vector(&backend, &coordinates, storage).unwrap();
    let vels = Tensor::vector(&backend, &velocities, storage).unwrap();
    let motors = Tensor::vector(&backend, &motors, storage).unwrap();
    let joints = Tensor::vector(&backend, &joint_commands, storage).unwrap();
    let mut encoder = backend.begin_encoding();
    state
        .encode_articulation_motor_targets(&mut encoder, &kernels, &motors)
        .unwrap();
    state
        .encode_articulation_impulse_joints(&mut encoder, &kernels, &joints)
        .unwrap();
    state
        .encode_articulation_reset(&mut encoder, &kernels, &selected, &roots, &coords, &vels)
        .unwrap();
    backend.submit(encoder).unwrap();
    let reset = _poses(&backend, &state).await;
    let reset_vels = _read(&backend, state.body_velocities()).await;
    let reset_dofs = _read(&backend, state.multibodies().dof_state()).await;
    for body in 0..51 {
        assert!(
            reset[body * 3 + 1]
                .translation
                .distance(expected[body].translation)
                < 2.0e-5,
            "body {body} pose"
        );
        let expected_velocity = (next[body].translation - expected[body].translation) / epsilon;
        assert!(
            reset_vels[body * 3 + 1].linear.distance(expected_velocity) < 0.002,
            "body {body} velocity: {:?} vs {expected_velocity:?}",
            reset_vels[body * 3 + 1].linear
        );
        for batch in [0, 2] {
            assert_eq!(reset[body * 3 + batch], before[body * 3 + batch]);
            assert_eq!(
                bytemuck::bytes_of(&reset_vels[body * 3 + batch]),
                bytemuck::bytes_of(&before_vels[body * 3 + batch])
            );
        }
    }
    for (index, &value) in reset_dofs.iter().enumerate() {
        if index < dpb * 3 && index % 3 == 1 {
            assert_eq!(value, velocities[index]);
        } else {
            assert_eq!(value, before_dofs[index]);
        }
    }
    // Reuse the same command tensors and kernels in a step encoder. Only the
    // selection mask changes; no API performs staging or reads during encode.
    for step in 0..12 {
        let mut encoder = backend.begin_encoding();
        state
            .encode_articulation_motor_targets(&mut encoder, &kernels, &motors)
            .unwrap();
        state
            .encode_articulation_impulse_joints(&mut encoder, &kernels, &joints)
            .unwrap();
        pipeline
            .step_encoded(&backend, &mut state, None, &mut encoder)
            .unwrap();
        backend.submit(encoder).unwrap();
        let poses = _poses(&backend, &state).await;
        assert!(
            poses.iter().all(|pose| pose.translation.is_finite()),
            "nonfinite poses at controlled step {step}: {:?}",
            poses
                .iter()
                .enumerate()
                .filter(|(_, pose)| !pose.translation.is_finite())
                .map(|(index, _)| index)
                .collect::<Vec<_>>()
        );
    }
    let solved = _poses(&backend, &state).await;
    let solved_vels = _read(&backend, state.body_velocities()).await;
    assert!(solved.iter().all(|pose| pose.translation.is_finite()));
    assert!(solved_vels.iter().all(|vel| vel.linear.is_finite()));
    for branch in 0..5 {
        let tip = 10 + branch * 10;
        assert!(
            solved[tip * 3 + 1]
                .translation
                .distance(expected[tip].translation)
                < 0.01,
            "branch {branch} external lock drift"
        );
    }
    assert!(
        solved[4].translation.distance(reset[4].translation) > 1.0e-5,
        "active articulation did not move"
    );
    // Select two batches in a second reset and leave the controlled batch
    // byte-for-byte unchanged, exercising mask updates on persistent inputs.
    let mut selected = selected;
    backend
        .write_buffer(selected.buffer_mut(), 0, &[1u32, 0, 1])
        .unwrap();
    let mut encoder = backend.begin_encoding();
    state
        .encode_articulation_reset(&mut encoder, &kernels, &selected, &roots, &coords, &vels)
        .unwrap();
    backend.submit(encoder).unwrap();
    let final_poses = _poses(&backend, &state).await;
    for body in 0..state.num_active_bodies as usize {
        assert_eq!(final_poses[body * 3 + 1], solved[body * 3 + 1]);
    }
}

#[futures_test::test]
async fn articulation_gpu_state_reset() {
    use crate::dynamics::GpuArticulationCommands;
    use khal::BufferUsages;
    use vortx::tensor::Tensor;
    let backend = _backend().await;
    let kernels = GpuArticulationCommands::new(&backend).unwrap();
    for fixed in [false, true] {
        let scenes = [
            _scene(fixed, true, false, true),
            _scene(fixed, true, false, true),
            _scene(fixed, true, false, true),
        ];
        let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
        pipeline.step(&backend, &mut state, None).unwrap();
        let before = _poses(&backend, &state).await;
        let before_vels = _read(&backend, state.body_velocities()).await;
        let before_joints = _read(&backend, state.multibodies().joint_constraints()).await;
        let links = _read(&backend, state.multibodies().links_static()).await;
        let dpb = state.multibodies().dofs_per_batch() as usize;
        let mut coords = vec![0.0; dpb * 3];
        coords[links[4].assembly_id as usize * 3 + 1] = 0.2;
        coords[links[7].assembly_id as usize * 3 + 1] = -0.1;
        let mut vels = vec![0.0; dpb * 3];
        if !fixed {
            vels[1] = 0.3;
            vels[4] = -0.2;
        }
        #[cfg(feature = "dim2")]
        let root = Pose::from_parts(_v(3.0, 4.0), _rotation(0.7));
        #[cfg(feature = "dim3")]
        let root = Pose::from_parts(
            _v(3.0, 4.0),
            crate::math::Rotation::from_euler(glamx::EulerRot::XYZ, 0.2, 0.3, 0.7),
        );
        let storage = BufferUsages::STORAGE;
        let selected = Tensor::vector(&backend, &[0u32, 1, 0], storage).unwrap();
        let roots = Tensor::vector(&backend, &[root; 3], storage).unwrap();
        let coords = Tensor::vector(&backend, coords, storage).unwrap();
        let vels = Tensor::vector(&backend, vels, storage).unwrap();
        let mut encoder = backend.begin_encoding();
        state
            .encode_articulation_reset(&mut encoder, &kernels, &selected, &roots, &coords, &vels)
            .unwrap();
        backend.submit(encoder).unwrap();
        let after = _poses(&backend, &state).await;
        let after_vels = _read(&backend, state.body_velocities()).await;
        let after_joints = _read(&backend, state.multibodies().joint_constraints()).await;
        for (index, (before, after)) in before_joints.iter().zip(&after_joints).enumerate() {
            if index % 3 == 1 {
                assert_eq!(after.kind, 0);
                assert_eq!(after.impulse, 0.0);
            } else {
                assert_eq!(bytemuck::bytes_of(before), bytemuck::bytes_of(after));
            }
        }
        let child = root
            * Pose::from_translation(_v(0.5, 0.0))
            * Pose::from_parts(Vector::ZERO, _rotation(0.2))
            * Pose::from_translation(_v(0.5, 0.0));
        let tip = child * Pose::from_translation(_v(0.9, 0.0));
        for (body, expected) in [root, child, tip].into_iter().enumerate() {
            assert!(
                after[body * 3 + 1]
                    .translation
                    .distance(expected.translation)
                    < 1.0e-5
            );
            assert!(
                (after[body * 3 + 1].rotation * Vector::X).distance(expected.rotation * Vector::X)
                    < 1.0e-5
            );
            assert!(
                after_vels[body * 3 + 1].linear.distance(if fixed {
                    Vector::ZERO
                } else {
                    _v(0.3, -0.2)
                }) < 1.0e-5
            );
        }
        for body in 0..state.num_colliders_per_batch() as usize {
            for batch in [0, 2] {
                assert_eq!(after[body * 3 + batch], before[body * 3 + batch]);
                assert_eq!(
                    after_vels[body * 3 + batch].linear,
                    before_vels[body * 3 + batch].linear
                );
            }
        }
        let solver_vels = _read(&backend, &state.solver_vels).await;
        let increments = _read(&backend, &state.solver_vels_inc).await;
        for body in 0..state.num_active_bodies as usize {
            assert_eq!(
                solver_vels[body * 3 + 1].linear,
                after_vels[body * 3 + 1].linear
            );
            assert_eq!(increments[body * 3 + 1].linear, Vector::ZERO);
        }
    }
}

#[futures_test::test]
async fn articulation_gpu_contact_reset() {
    use crate::dynamics::GpuArticulationCommands;
    use khal::BufferUsages;
    use vortx::tensor::Tensor;
    let backend = _backend().await;
    let kernels = GpuArticulationCommands::new(&backend).unwrap();
    let mut scenes = [
        _scene(false, true, true, false),
        _scene(false, true, true, false),
    ];
    for scene in &mut scenes {
        for &handle in &scene.handles {
            let pose = *scene.bodies[handle].position();
            scene.bodies[handle].set_position(
                Pose::from_parts(pose.translation - _v(0.0, 1.88), pose.rotation),
                true,
            );
        }
        // Exercise the regular rigid-body contact bank alongside articulation
        // contacts. This body remains in place when the articulation resets.
        let free = scene
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(_v(4.0, 0.12)));
        scene.colliders.insert_with_parent(
            ColliderBuilder::ball(0.15)
                .mass(1.0)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_1,
                    Group::GROUP_2,
                    InteractionTestMode::And,
                )),
            free,
            &mut scene.bodies,
        );
        scene.bodies[free].recompute_mass_properties_from_colliders(&scene.colliders);
    }
    let (mut state, pipeline) = _state(&backend, &scenes, -9.81, 32);
    for _ in 0..4 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let before = _read(&backend, state.multibodies().contact_constraints()).await;
    assert!(
        before
            .iter()
            .any(|contact| contact.kind != 0 && contact.impulse > 0.0)
    );
    let infos = _read(&backend, state.multibodies().multibody_info()).await;
    let rigid_before = _read(&backend, &state.old_constraints).await;
    assert!(
        rigid_before
            .iter()
            .any(|contact| contact.len != 0 && contact.solver_body_a % 2 == 1)
    );
    let storage = BufferUsages::STORAGE;
    let mask = Tensor::vector(&backend, &[0u32, 1], storage).unwrap();
    let root = Pose::from_translation(_v(0.0, 3.0));
    let roots = Tensor::vector(&backend, &[root; 2], storage).unwrap();
    let dofs = state.multibodies().dofs_per_batch() as usize * 2;
    let coords = Tensor::vector(&backend, vec![0.0; dofs], storage).unwrap();
    let vels = Tensor::vector(&backend, vec![0.0; dofs], storage).unwrap();
    let mut encoder = backend.begin_encoding();
    state
        .encode_articulation_reset(&mut encoder, &kernels, &mask, &roots, &coords, &vels)
        .unwrap();
    backend.submit(encoder).unwrap();
    let after = _read(&backend, state.multibodies().contact_constraints()).await;
    let rigid_after = _read(&backend, &state.old_constraints).await;
    for (before, after) in rigid_before.iter().zip(&rigid_after) {
        if before.len != 0 && before.solver_body_a % 2 == 1 {
            assert_eq!(after.len, 0);
            assert_eq!(after.solver_body_a, u32::MAX);
            assert_eq!(after.solver_body_b, u32::MAX);
        } else {
            assert_eq!(bytemuck::bytes_of(before), bytemuck::bytes_of(after));
        }
    }
    for batch in 0..2 {
        let mb = infos[batch];
        for index in
            mb.contact_constraint_start..mb.contact_constraint_start + mb.contact_constraint_count
        {
            if batch == 1 {
                assert_eq!(after[index as usize].kind, 0);
                assert_eq!(after[index as usize].impulse, 0.0);
            } else {
                assert_eq!(
                    bytemuck::bytes_of(&after[index as usize]),
                    bytemuck::bytes_of(&before[index as usize])
                );
            }
        }
    }
    for _ in 0..2 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let contacts = _read(&backend, state.multibodies().contact_constraints()).await;
    let infos = _read(&backend, state.multibodies().multibody_info()).await;
    assert_eq!(infos[1].contact_constraint_count, 0);
    assert!(
        contacts
            .iter()
            .any(|contact| contact.kind != 0 && contact.impulse > 0.0)
    );
}

#[futures_test::test]
async fn articulation_roots_kinematics_and_dynamics() {
    let backend = _backend().await;
    for fixed in [false, true] {
        let scenes = [_scene(fixed, false, false, false)];
        let (mut state, pipeline) = _state(&backend, &scenes, -9.81, 32);
        pipeline.step(&backend, &mut state, None).unwrap();
        let poses = _poses(&backend, &state).await;
        for i in 0..3 {
            assert!((poses[i].translation.x - i as f32).abs() < 1.0e-4);
        }
        assert!((poses[1].translation.distance(poses[0].translation) - 1.0).abs() < 1.0e-4);
        let coords = state
            .multibodies()
            .read_dof_coords(&backend, 0)
            .await
            .unwrap();
        assert_eq!(
            coords.len(),
            if fixed {
                2
            } else {
                crate::shaders::dynamics::SPATIAL_DIM + 2
            }
        );
        if fixed {
            assert_eq!(poses[0].translation, _v(0.0, 2.0));
        } else {
            let dofs: Vec<f32> = backend
                .slow_read_vec(state.multibodies().dof_state().buffer())
                .await
                .unwrap();
            assert!(
                (dofs[1] + 9.81 / 600.0).abs() < 1.0e-4,
                "root velocity {}",
                dofs[1]
            );
        }
    }
}

#[futures_test::test]
async fn articulation_tree_inverse_matches_dense_operator() {
    use crate::shaders::dynamics::{
        ARTICULATED_FACTOR_WORDS, GpuMbAssembleMassMatrix, GpuMbBuildJointResponseColumns,
        GpuMbComputeDynamicsPre, GpuMbFactorArticulatedBody, GpuMbInitializeArticulatedBody,
        GpuMbSolveArticulatedBody, MB_JOINT_KIND_COUPLING, MB_JOINT_KIND_MOTOR,
        MultibodyJointConstraint, TREE_SCRATCH_WIDTH,
    };
    use khal::backend::Encoder;
    use khal::{BufferUsages, Shader};
    use vortx::tensor::Tensor;
    let backend = _backend().await;
    for scene in [
        _scene(false, false, false, false),
        _scene(true, false, false, false),
        _branched_scene(),
    ] {
        let (mut state, _) = _state(&backend, &[scene], 0., 1);
        let indices = backend
            .slow_read_vec(state.batch_indices.buffer())
            .await
            .unwrap()[0];
        let infos = backend
            .slow_read_vec(state.multibodies().multibody_info().buffer())
            .await
            .unwrap();
        let mb = infos[0];
        let n = mb.ndofs as usize;
        let velocities: Vec<_> = (0..state.multibodies().dofs_per_batch())
            .map(|d| (d as f32 * 0.37).sin() * 2.)
            .collect();
        backend
            .write_buffer(
                state.multibodies_mut().dof_state_mut().buffer_mut(),
                0,
                &velocities,
            )
            .unwrap();
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut ws = Tensor::vector(
            &backend,
            backend
                .slow_read_vec(state.multibodies().links_workspace().buffer())
                .await
                .unwrap(),
            storage,
        )
        .unwrap();
        let mut poses = Tensor::vector(
            &backend,
            backend
                .slow_read_vec(state.body_poses().buffer())
                .await
                .unwrap(),
            storage,
        )
        .unwrap();
        let mut jac = Tensor::vector(
            &backend,
            vec![0.; state.multibodies().body_jacobians().len() as usize],
            storage,
        )
        .unwrap();
        let mut cor = Tensor::vector(
            &backend,
            vec![
                0.;
                (indices.coriolis_batch_capacity
                    + indices.coriolis_batch_capacity / crate::shaders::DIM
                        * crate::shaders::ANG_DIM) as usize
            ],
            storage,
        )
        .unwrap();
        let mut dense = Tensor::vector(
            &backend,
            vec![0.; 2 * indices.mass_matrix_acc_section_offset as usize],
            storage,
        )
        .unwrap();
        let mut factors = Tensor::vector(
            &backend,
            vec![0u32; state.multibodies().links_per_batch() as usize * ARTICULATED_FACTOR_WORDS],
            storage,
        )
        .unwrap();
        let mut scratch = Tensor::vector(
            &backend,
            vec![0.; indices.mb_max_links as usize * TREE_SCRATCH_WIDTH],
            storage,
        )
        .unwrap();
        let dt = Tensor::scalar(
            &backend,
            0.017f32,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
        #[derive(Shader)]
        struct Kernels {
            pre: GpuMbComputeDynamicsPre,
            assemble: GpuMbAssembleMassMatrix,
            initialize: GpuMbInitializeArticulatedBody,
            factor: GpuMbFactorArticulatedBody,
            joint_responses: GpuMbBuildJointResponseColumns,
            solve: GpuMbSolveArticulatedBody,
        }
        let Kernels {
            pre,
            assemble,
            initialize,
            factor,
            joint_responses,
            solve,
        } = Kernels::from_backend(&backend).unwrap();
        let mut encoder = backend.begin_encoding();
        {
            let mut pass = encoder.begin_pass("tree oracle", None);
            pre.call(
                &mut pass,
                [64, 1, 1],
                state.multibodies().multibody_info(),
                state.multibodies().links_static(),
                &mut ws,
                &mut poses,
                &mut jac,
                &mut cor,
                state.multibodies().dof_state(),
                &state.batch_indices,
            )
            .unwrap();
            assemble
                .call(
                    &mut pass,
                    [n as u32 * n as u32, 1, 1],
                    state.multibodies().multibody_info(),
                    state.multibodies().links_static(),
                    &ws,
                    &jac,
                    &cor,
                    state.multibodies().dof_state(),
                    &mut dense,
                    &dt,
                    &state.batch_indices,
                )
                .unwrap();
            initialize
                .call(
                    &mut pass,
                    state.multibodies().links_per_batch(),
                    state.multibodies().multibody_info(),
                    state.multibodies().links_static(),
                    &ws,
                    state.multibodies().dof_state(),
                    &mut factors,
                    &dt,
                    &state.batch_indices,
                )
                .unwrap();
            factor
                .call(
                    &mut pass,
                    1,
                    state.multibodies().multibody_info(),
                    state.multibodies().dof_state(),
                    &mut factors,
                    &dt,
                    &state.batch_indices,
                )
                .unwrap();
        }
        backend.submit(encoder).unwrap();
        let matrix = backend.slow_read_vec(dense.buffer()).await.unwrap();
        let mut links: Vec<crate::shaders::dynamics::MultibodyLinkStatic> =
            vec![bytemuck::Zeroable::zeroed(); state.multibodies().links_per_batch() as usize];
        backend
            .slow_read_buffer(state.multibodies().links_static().buffer(), &mut links)
            .await
            .unwrap();
        let source_link = |d: usize| {
            links
                .iter()
                .position(|link| {
                    d >= link.assembly_id as usize && d < (link.assembly_id + link.ndofs) as usize
                })
                .unwrap() as u32
        };
        let constraints: Vec<_> = (0..2 * n)
            .map(|s| {
                let d = s % n;
                let paired = (d + n / 2) % n;
                MultibodyJointConstraint {
                    dof_id: d as u32,
                    dof2_id: paired as u32,
                    kind: if s < n {
                        MB_JOINT_KIND_MOTOR
                    } else {
                        MB_JOINT_KIND_COUPLING
                    },
                    _kind_extra: source_link(d),
                    _kind_extra2: source_link(paired),
                    coupling_coeff: if s < n { 0. } else { 0.7 },
                    response_slot: s as u32,
                    ..Default::default()
                }
            })
            .collect();
        let constraints = Tensor::vector(&backend, constraints.clone(), storage).unwrap();
        let mut joint_columns = Tensor::vector(&backend, vec![0.; 2 * n * n], storage).unwrap();
        let mut joint_scratch = Tensor::vector(
            &backend,
            vec![f32::NAN; 2 * n * indices.mb_max_links as usize * TREE_SCRATCH_WIDTH],
            storage,
        )
        .unwrap();
        let joint_info = Tensor::vector(
            &backend,
            vec![crate::shaders::dynamics::MultibodyInfo {
                first_constraint: 0,
                max_constraints: (2 * n) as u32,
                ..mb
            }],
            storage,
        )
        .unwrap();
        let mut encoder = backend.begin_encoding();
        {
            let mut pass = encoder.begin_pass("sparse joint oracle", None);
            joint_responses
                .call(
                    &mut pass,
                    [1, (2 * n) as u32, 1],
                    &joint_info,
                    &constraints,
                    &mut joint_columns,
                    &factors,
                    &mut joint_scratch,
                    &state.batch_indices,
                )
                .unwrap();
        }
        backend.submit(encoder).unwrap();
        let joint_columns = backend.slow_read_vec(joint_columns.buffer()).await.unwrap();
        for s in 0..2 * n {
            let d = s % n;
            let paired = (d + n / 2) % n;
            for row in 0..n {
                let product: f32 = (0..n)
                    .map(|k| matrix[k * n + row] * joint_columns[s * n + k])
                    .sum();
                let expected =
                    if row == d { 1. } else { 0. } - if s >= n && row == paired { 0.7 } else { 0. };
                assert!(
                    (product - expected).abs() < 0.0003,
                    "sparse n={n} row={row} constraint={s}: residual {}",
                    product - expected
                );
            }
        }
        for implicit in [0u32, 1] {
            let mode = Tensor::scalar(
                &backend,
                implicit,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
            .unwrap();
            for column in 0..n {
                let mut rhs = vec![0.; n];
                rhs[column] = 1.;
                let mut rhs = Tensor::vector(&backend, rhs, storage).unwrap();
                let mut encoder = backend.begin_encoding();
                {
                    let mut pass = encoder.begin_pass("tree solve", None);
                    solve
                        .call(
                            &mut pass,
                            1,
                            state.multibodies().multibody_info(),
                            &factors,
                            &mut rhs,
                            &mut scratch,
                            &state.batch_indices,
                            &mode,
                        )
                        .unwrap();
                }
                backend.submit(encoder).unwrap();
                let x = backend.slow_read_vec(rhs.buffer()).await.unwrap();
                let base = if implicit != 0 {
                    indices.mass_matrix_acc_section_offset as usize
                } else {
                    0
                };
                for row in 0..n {
                    let product: f32 = (0..n).map(|d| matrix[base + d * n + row] * x[d]).sum();
                    let expected = if row == column { 1. } else { 0. };
                    assert!(
                        (product - expected).abs() < 0.0003,
                        "n={n} implicit={implicit} row={row} column={column}: residual {}",
                        product - expected
                    );
                }
            }
        }
    }
}

#[futures_test::test]
async fn articulation_limits_recover_out_of_range_joints() {
    let backend = _backend().await;
    let mut scenes = [_scene(true, false, false, false)];
    let mut displacement = vec![0.0; crate::shaders::dynamics::SPATIAL_DIM + 2];
    let n = displacement.len();
    displacement[n - 2] = 0.7;
    displacement[n - 1] = 0.3;
    let handle = scenes[0].mb.iter().next().unwrap().0;
    scenes[0]
        .mb
        .get_mut(handle)
        .unwrap()
        .0
        .apply_displacements(&displacement);
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
    let before = state
        .multibodies()
        .read_dof_coords(&backend, 0)
        .await
        .unwrap();
    assert!(before[0] > 0.5 && before[1] > 0.2);
    for _ in 0..90 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let after = state
        .multibodies()
        .read_dof_coords(&backend, 0)
        .await
        .unwrap();
    assert!(after[0].abs() < 0.52, "hinge limit recovery: {}", after[0]);
    assert!(after[1].abs() < 0.22, "slider limit recovery: {}", after[1]);
}

#[futures_test::test]
async fn articulation_motors_limits_and_momentum() {
    let backend = _backend().await;
    for fixed in [false, true] {
        let scenes = [_scene(fixed, true, false, false)];
        let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
        let initial = _poses(&backend, &state).await;
        let initial_com =
            (initial[0].translation + initial[1].translation + initial[2].translation) / 3.0;
        let mut max_drift = 0.0_f32;
        let mut max_momentum = 0.0_f32;
        for step in 0..240 {
            if step % 40 == 0 {
                let sign = if step % 80 == 0 { 1.0 } else { -1.0 };
                for (link, axis, target) in [(1, ANG_AXIS, 0.8 * sign), (2, 0, 0.35 * sign)] {
                    state
                        .multibodies_mut()
                        .set_motor(
                            &backend,
                            0,
                            link,
                            axis,
                            JointMotor {
                                target_pos: target,
                                stiffness: 40.0,
                                damping: 5.0,
                                max_force: 10.0,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
            }
            pipeline.step(&backend, &mut state, None).unwrap();
            let poses = _poses(&backend, &state).await;
            let coords = state
                .multibodies()
                .read_dof_coords(&backend, 0)
                .await
                .unwrap();
            let n = coords.len();
            if step == 39 {
                assert!(
                    coords[n - 2].abs() > 0.005,
                    "hinge did not move: {coords:?}"
                );
                assert!(
                    coords[n - 1].abs() > 0.001,
                    "slider did not move: {coords:?}"
                );
            }
            assert!(
                coords[n - 2].abs() < 0.52,
                "hinge at step {step}: {}",
                coords[n - 2]
            );
            assert!(
                coords[n - 1].abs() < 0.22,
                "slider at step {step}: {}",
                coords[n - 1]
            );
            if !fixed {
                let com =
                    (poses[0].translation + poses[1].translation + poses[2].translation) / 3.0;
                let velocities = _velocities(&backend, &state).await;
                let momentum = velocities.iter().take(3).map(|v| v.linear).sum::<Vector>();
                max_drift = max_drift.max(com.distance(initial_com));
                max_momentum = max_momentum.max(momentum.length());
                assert!(max_drift < 0.001, "COM drift {max_drift} at step {step}");
                assert!(
                    max_momentum < 0.001,
                    "momentum {max_momentum} at step {step}"
                );
            }
        }
        eprintln!("fixed={fixed}: COM drift={max_drift}, momentum={max_momentum}");
    }
}

#[futures_test::test]
async fn articulation_contacts_and_external_joints() {
    let backend = _backend().await;
    for external in [false, true] {
        let scenes = [_scene(external, false, !external, external)];
        let (mut state, pipeline) = _state(&backend, &scenes, -9.81, 32);
        for _ in 0..1200 {
            pipeline.step(&backend, &mut state, None).unwrap();
        }
        let poses = _poses(&backend, &state).await;
        if external {
            let anchor = poses[3].translation + poses[3].rotation * _v(0.0, 0.5);
            assert!(
                anchor.distance(poses[2].translation) < 0.03,
                "external joint separation {}",
                anchor.distance(poses[2].translation)
            );
        } else {
            for p in poses.iter().take(3) {
                assert!(
                    (0.12..0.2).contains(&p.translation.y),
                    "contact height {}",
                    p.translation.y
                );
            }
            // Sliding must produce opposing friction on the articulation.
            backend
                .write_buffer(
                    state.multibodies_mut().dof_state_mut().buffer_mut(),
                    0,
                    &[2.0_f32],
                )
                .unwrap();
            for _ in 0..90 {
                pipeline.step(&backend, &mut state, None).unwrap();
            }
            backend.synchronize().unwrap();
            let dofs: Vec<f32> = backend
                .slow_read_vec(state.multibodies().dof_state().buffer())
                .await
                .unwrap();
            assert!(
                (0.0..1.8).contains(&dofs[0]),
                "friction left root velocity {}",
                dofs[0]
            );
        }
    }
}

#[cfg(feature = "dim2")]
#[futures_test::test]
async fn articulation_planar_rotation_and_wrench() {
    let backend = _backend().await;
    let mut scenes = [_scene(false, false, false, false)];
    let root = scenes[0].handles[0];
    scenes[0].bodies[root].set_rotation(glamx::Rot2::from_angle(0.4), true);
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
    pipeline.step(&backend, &mut state, None).unwrap();
    let poses = _poses(&backend, &state).await;
    for p in poses.iter().take(3) {
        assert!((p.rotation.angle() - 0.4).abs() < 1.0e-5);
    }
    state
        .multibodies_mut()
        .set_link_external_wrench(&backend, 0, 0, 0, _v(3.0, 0.0), 1.0, 0.0)
        .unwrap();
    for _ in 0..30 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let moved = _poses(&backend, &state).await;
    assert!(moved[0].rotation.angle() > poses[0].rotation.angle());
    let com = |p: &[Pose]| p[..3].iter().map(|p| p.translation).sum::<Vector>() / 3.0;
    assert!(com(&moved).x > com(&poses).x + 0.0005);
}

#[futures_test::test]
#[ignore = "short throughput and allocation probe"]
async fn articulation_benchmark() {
    let backend = _backend().await;
    if let GpuBackend::WebGpu(webgpu) = &backend {
        eprintln!("adapter={:?}", webgpu.device().adapter_info());
    }
    for (links, batches) in [
        (3, 1),
        (3, 256),
        (3, 1024),
        (3, 4096),
        (3, 16384),
        (12, 4096),
        (24, 1),
        (24, 64),
        (24, 256),
        (24, 1024),
        (24, 4096),
        (48, 1024),
    ] {
        _measure_articulation(&backend, links, batches).await;
    }
}

#[derive(Clone, Copy)]
struct BenchmarkSample {
    envs: usize,
    wall_ms: f64,
    gpu_ms: f64,
    articulation_ms: f64,
}

#[futures_test::test]
#[ignore = "bounded environment-count scaling probe"]
async fn articulation_env_scaling() {
    let backend = _backend().await;
    if let GpuBackend::WebGpu(webgpu) = &backend {
        eprintln!("adapter={:?}", webgpu.device().adapter_info());
    }
    for links in [3, 24] {
        let mut samples = Vec::new();
        // The 24-collider brute-force dispatch uses nine workgroups per env;
        // 8192 envs exceed WebGPU's 65535 workgroup limit on its X dimension.
        let largest = if links == 3 { 8192 } else { 6144 };
        for envs in [1, 64, 256, 512, 1024, 2048, 4096, largest] {
            samples.push(_measure_articulation(&backend, links, envs).await);
        }
        for (range, min_envs) in [("all", 1), ("large", 1024)] {
            let selected: Vec<_> = samples.iter().filter(|s| s.envs >= min_envs).collect();
            _report_linear_fit(links, range, "wall", &selected, |s| s.wall_ms);
            _report_linear_fit(links, range, "all_gpu", &selected, |s| s.gpu_ms);
            _report_linear_fit(links, range, "articulation_gpu", &selected, |s| {
                s.articulation_ms
            });
        }
        // Repeat the GPU-bound range in reverse order to check clock/order effects.
        let mut reverse = Vec::new();
        for envs in [largest, 4096, 2048, 1024] {
            reverse.push(_measure_articulation(&backend, links, envs).await);
        }
        let selected: Vec<_> = reverse.iter().collect();
        _report_linear_fit(links, "reverse_large", "wall", &selected, |s| s.wall_ms);
        _report_linear_fit(links, "reverse_large", "all_gpu", &selected, |s| s.gpu_ms);
        _report_linear_fit(links, "reverse_large", "articulation_gpu", &selected, |s| {
            s.articulation_ms
        });
    }
}

fn _report_linear_fit(
    links: usize,
    range: &str,
    metric: &str,
    samples: &[&BenchmarkSample],
    value: impl Fn(&BenchmarkSample) -> f64,
) {
    let n = samples.len() as f64;
    let mean_x = samples.iter().map(|s| s.envs as f64).sum::<f64>() / n;
    let mean_y = samples.iter().map(|s| value(s)).sum::<f64>() / n;
    let xx = samples
        .iter()
        .map(|s| (s.envs as f64 - mean_x).powi(2))
        .sum::<f64>();
    let xy = samples
        .iter()
        .map(|s| (s.envs as f64 - mean_x) * (value(s) - mean_y))
        .sum::<f64>();
    let slope = xy / xx;
    let intercept = mean_y - slope * mean_x;
    let residual = samples
        .iter()
        .map(|s| (value(s) - intercept - slope * s.envs as f64).powi(2))
        .sum::<f64>();
    let total = samples
        .iter()
        .map(|s| (value(s) - mean_y).powi(2))
        .sum::<f64>();
    let max_relative = samples
        .iter()
        .map(|s| (value(s) - intercept - slope * s.envs as f64).abs() / value(s))
        .fold(0.0, f64::max);
    eprintln!(
        "fit links={links} range={range} metric={metric} intercept_ms={intercept:.6} slope_us_per_env={:.6} r_squared={:.6} max_residual_percent={:.2}",
        slope * 1000.0,
        1.0 - residual / total,
        max_relative * 100.0
    );
}

async fn _measure_articulation(
    backend: &GpuBackend,
    links: usize,
    batches: usize,
) -> BenchmarkSample {
    let scenes: Vec<_> = (0..batches).map(|_| _benchmark_chain(links)).collect();
    // Contact-free benchmark: keep the unused contact banks minimal.
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 1);
    for _ in 0..10 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();
    let mut elapsed = Vec::new();
    let mut host = Vec::new();
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let mut host_seconds = 0.0;
        for _ in 0..20 {
            let host_start = std::time::Instant::now();
            pipeline.step(&backend, &mut state, None).unwrap();
            host_seconds += host_start.elapsed().as_secs_f64();
        }
        backend.synchronize().unwrap();
        elapsed.push(start.elapsed().as_secs_f64() / 20.0);
        host.push(host_seconds / 20.0);
    }
    elapsed.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    let seconds = elapsed[1];
    let mut timestamps = khal::backend::GpuTimestamps::new(&backend, 256);
    assert!(
        timestamps.is_enabled(),
        "benchmark requires pass timestamps"
    );
    let mut gpu_ms = Vec::new();
    let mut articulation_ms = Vec::new();
    for _ in 0..5 {
        timestamps.reset();
        pipeline
            .step(&backend, &mut state, Some(&mut timestamps))
            .unwrap();
        backend.synchronize().unwrap();
        let passes = timestamps.read(&backend).await.unwrap();
        gpu_ms.push(passes.iter().map(|p| p.duration_ms).sum::<f64>());
        articulation_ms.push(
            passes
                .iter()
                .filter(|p| p.label.contains("mb"))
                .map(|p| p.duration_ms)
                .sum::<f64>(),
        );
    }
    gpu_ms.sort_by(f64::total_cmp);
    articulation_ms.sort_by(f64::total_cmp);
    let allocations = state.multibodies()._core_allocation_sizes();
    let bytes: usize = allocations.iter().map(|(_, bytes)| bytes).sum();
    let infos: Vec<crate::shaders::dynamics::MultibodyInfo> = backend
        .slow_read_vec(state.multibodies().multibody_info().buffer())
        .await
        .unwrap();
    let ndofs = infos[0].ndofs;
    let poses = _poses(&backend, &state).await;
    assert!(poses.iter().all(|p| p.translation.is_finite()));
    eprintln!(
        "dim={} links={links} batches={batches} dofs={ndofs} core_bytes_per_env={} wall_ms={:.3} host_call_ms={:.3} all_gpu_ms={:.3} articulation_gpu_ms={:.3} env_steps_per_second={:.0}",
        crate::math::DIM,
        bytes / batches,
        seconds * 1000.0,
        host[1] * 1000.0,
        gpu_ms[2],
        articulation_ms[2],
        batches as f64 / seconds
    );
    BenchmarkSample {
        envs: batches,
        wall_ms: seconds * 1000.0,
        gpu_ms: gpu_ms[2],
        articulation_ms: articulation_ms[2],
    }
}

fn _benchmark_chain(links: usize) -> Scene {
    let mut s = Scene {
        bodies: RigidBodySet::new(),
        colliders: ColliderSet::new(),
        mb: MultibodyJointSet::new(),
        joints: ImpulseJointSet::new(),
        handles: Vec::new(),
    };
    for i in 0..links {
        let h = s
            .bodies
            .insert(RigidBodyBuilder::dynamic().translation(_v(i as f32, 2.0)));
        s.colliders.insert_with_parent(
            ColliderBuilder::ball(0.15)
                .mass(1.0)
                .collision_groups(InteractionGroups::new(
                    Group::NONE,
                    Group::NONE,
                    InteractionTestMode::And,
                )),
            h,
            &mut s.bodies,
        );
        if i > 0 {
            let joint: GenericJoint = if i % 2 == 1 {
                _hinge()
                    .local_anchor1(_v(0.5, 0.0))
                    .local_anchor2(_v(-0.5, 0.0))
                    .limits([-0.5, 0.5])
                    .motor_position(0.15, 40.0, 5.0)
                    .motor_max_force(10.0)
                    .build()
                    .into()
            } else {
                PrismaticJointBuilder::new(Vector::X)
                    .local_anchor1(_v(0.5, 0.0))
                    .local_anchor2(_v(-0.5, 0.0))
                    .limits([-0.2, 0.2])
                    .motor_position(0.1, 40.0, 5.0)
                    .motor_max_force(10.0)
                    .build()
                    .into()
            };
            s.mb.insert(s.handles[i - 1], h, joint, true).unwrap();
        }
        s.handles.push(h);
    }
    for (_, b) in s.bodies.iter_mut() {
        b.recompute_mass_properties_from_colliders(&s.colliders);
    }
    s
}

fn _ground() -> ColliderBuilder {
    #[cfg(feature = "dim2")]
    {
        ColliderBuilder::cuboid(10.0, 0.5)
    }
    #[cfg(feature = "dim3")]
    {
        ColliderBuilder::cuboid(10.0, 0.5, 10.0)
    }
}

async fn _velocities(backend: &GpuBackend, state: &RbdState) -> Vec<Velocity> {
    use crate::shaders::dynamics::{WS_RB_VELS, WsAddr, ws_vel};
    let ws: Vec<glamx::Vec4> = backend
        .slow_read_vec(state.multibodies().links_workspace().buffer())
        .await
        .unwrap();
    (0..3)
        .map(|k| ws_vel(&ws, WsAddr::new(0, 1, 0), k, WS_RB_VELS))
        .collect()
}

#[futures_test::test]
async fn articulation_selective_reset() {
    let backend = _backend().await;
    let template_scene = [_scene(false, true, false, false)];
    let (mut template, template_pipeline) = _state(&backend, &template_scene, 0.0, 32);
    for _ in 0..30 {
        template_pipeline
            .step(&backend, &mut template, None)
            .unwrap();
    }
    let snapshot = template.snapshot(&backend).await;
    let template_poses = _poses(&backend, &template).await;
    let template_coords = template
        .multibodies()
        .read_dof_coords(&backend, 0)
        .await
        .unwrap();
    let scenes = [
        _scene(false, true, false, false),
        _scene(false, true, false, false),
        _scene(false, true, false, false),
    ];
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0, 32);
    for _ in 0..10 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let before = _poses(&backend, &state).await;
    let offset = _v(4.0, -3.0);
    state.reset_env_from_snapshot_offset(&backend, 1, &snapshot, offset);
    let after = _poses(&backend, &state).await;
    let mut expected_coords = template_coords.clone();
    expected_coords[0] += offset.x;
    expected_coords[1] += offset.y;
    assert_eq!(
        state
            .multibodies()
            .read_dof_coords(&backend, 1)
            .await
            .unwrap(),
        expected_coords
    );
    for body in 0..3 {
        assert_eq!(before[body * 3], after[body * 3]);
        assert_eq!(before[body * 3 + 2], after[body * 3 + 2]);
        assert!(
            after[body * 3 + 1]
                .translation
                .distance(template_poses[body].translation + offset)
                < 1.0e-6
        );
    }
    state.publish_reset_templates(&backend, &[&snapshot]);
    state.reset_envs_from_templates(
        &backend,
        &[(0, 0), (2, 0)],
        &[offset, -offset],
        &vec![0.0; 2 * state.multibodies().dofs_per_batch() as usize],
    );
    let reset = _poses(&backend, &state).await;
    for (batch, translation) in [(0, offset), (2, -offset)] {
        let mut expected_coords = template_coords.clone();
        expected_coords[0] += translation.x;
        expected_coords[1] += translation.y;
        assert_eq!(
            state
                .multibodies()
                .read_dof_coords(&backend, batch)
                .await
                .unwrap(),
            expected_coords
        );
    }
    for body in 0..3 {
        assert_eq!(reset[body * 3 + 1], after[body * 3 + 1]);
        assert!(
            reset[body * 3]
                .translation
                .distance(template_poses[body].translation + offset)
                < 1.0e-6
        );
        assert!(
            reset[body * 3 + 2]
                .translation
                .distance(template_poses[body].translation - offset)
                < 1.0e-6
        );
    }
    pipeline.step(&backend, &mut state, None).unwrap();
    let stepped = _poses(&backend, &state).await;
    for body in 0..3 {
        assert!(
            stepped[body * 3]
                .translation
                .distance(reset[body * 3].translation)
                < 0.01
        );
        assert!(
            stepped[body * 3 + 2]
                .translation
                .distance(reset[body * 3 + 2].translation)
                < 0.01
        );
    }
}

#[futures_test::test]
async fn articulation_rapier_comparison() {
    let backend = _backend().await;
    for fixed in [false, true] {
        let mut scenes = [_scene(fixed, false, false, false)];
        let (mut state, pipeline) = _state(&backend, &scenes, -9.81, 32);
        let mut rapier = PhysicsPipeline::new();
        let params = IntegrationParameters {
            dt: 1.0 / 600.0,
            ..Default::default()
        };
        let mut islands = IslandManager::new();
        let mut broad = BroadPhaseBvh::new();
        let mut narrow = NarrowPhase::new();
        let mut ccd = CCDSolver::new();
        for _ in 0..30 {
            pipeline.step(&backend, &mut state, None).unwrap();
            let s = &mut scenes[0];
            rapier.step(
                _v(0.0, -9.81),
                &params,
                &mut islands,
                &mut broad,
                &mut narrow,
                &mut s.bodies,
                &mut s.colliders,
                &mut s.joints,
                &mut s.mb,
                &mut ccd,
                &(),
                &(),
            );
        }
        let poses = _poses(&backend, &state).await;
        for (i, h) in scenes[0].handles.iter().enumerate() {
            let rp = scenes[0].bodies[*h].position();
            assert!(
                poses[i].translation.distance(rp.translation) < 0.02,
                "fixed={fixed}, body={i}: Nexus {:?}, Rapier {:?}",
                poses[i].translation,
                rp.translation
            );
            assert!((poses[i].rotation * Vector::X).distance(rp.rotation * Vector::X) < 0.02);
        }
    }
}
