//! Rigid-body dynamics: forces, velocities, constraints, and solvers.

pub use crate::shaders::dynamics::RbdSimParams;
pub use body::{BodyCoupling, BodyCouplingEntry, BodyDesc, GpuBodySet};
pub use coloring::{ColorBucketsArgs, ColoringArgs, GpuColoring};
pub use joint::{GpuImpulseJointSet, GpuJointSolver, JointSolverArgs, convert_joint_motor};
pub use mprops_update::{GpuMpropsUpdate, GpuSyncColliderPosesShader};
pub use multibody::{
    ArticulationJointCommand, ArticulationMotorCommand, ArticulationMotorTarget,
    GpuArticulationCommands, GpuMultibodySet, GpuMultibodySnapshot, GpuMultibodySolver,
    MultibodySolverArgs,
};
pub use prep_render::{RbdInstanceDesc, WgRbdPrepRender};
pub use solver::{GpuSolver, SolverArgs};
pub use warmstart::{GpuWarmstart, WarmstartArgs};

pub mod body;
mod coloring;
mod joint;
mod mprops_update;
pub(crate) mod multibody;
mod prep_render;
mod solver;
pub mod warmstart;
