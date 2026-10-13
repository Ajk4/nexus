//! Native articulation probes. Set NEXUS_TEST_GPU=1 to run on WebGPU.
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

fn _state(backend: &GpuBackend, scenes: &[Scene], gravity: f32) -> (RbdState, RbdPipeline) {
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
            collisions_capacity: 32,
            mb_contact_constraints_capacity: 32,
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
async fn articulation_roots_kinematics_and_dynamics() {
    let backend = _backend().await;
    for fixed in [false, true] {
        let scenes = [_scene(fixed, false, false, false)];
        let (mut state, pipeline) = _state(&backend, &scenes, -9.81);
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
async fn articulation_motors_limits_and_momentum() {
    let backend = _backend().await;
    for fixed in [false, true] {
        let scenes = [_scene(fixed, true, false, false)];
        let (mut state, pipeline) = _state(&backend, &scenes, 0.0);
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
        let (mut state, pipeline) = _state(&backend, &scenes, -9.81);
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
        }
    }
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
    let (mut template, template_pipeline) = _state(&backend, &template_scene, 0.0);
    for _ in 0..30 {
        template_pipeline
            .step(&backend, &mut template, None)
            .unwrap();
    }
    let snapshot = template.snapshot(&backend).await;
    let template_poses = _poses(&backend, &template).await;
    let scenes = [
        _scene(false, true, false, false),
        _scene(false, true, false, false),
        _scene(false, true, false, false),
    ];
    let (mut state, pipeline) = _state(&backend, &scenes, 0.0);
    for _ in 0..10 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let before = _poses(&backend, &state).await;
    let offset = _v(4.0, -3.0);
    state.reset_env_from_snapshot_offset(&backend, 1, &snapshot, offset);
    let after = _poses(&backend, &state).await;
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
        let (mut state, pipeline) = _state(&backend, &scenes, -9.81);
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
