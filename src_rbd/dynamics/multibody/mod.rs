//! Host-side GPU multibody: buffer packing and kernel dispatch.
//!
//! A single `GpuMultibodySet` packs N rapier `Multibody`s across multiple simulation
//! batches into flat GPU tensors. `GpuMultibodySolver::step` advances them one step,
//! dispatched one workgroup per multibody.
//!
//! Supports native 2D and 3D articulation dynamics, contacts, and external joints.

mod env_reset;
mod loop_closing_joints;
mod multibody_from_rapier;
mod multibody_set;
mod multibody_solver;

pub use env_reset::GpuMultibodySnapshot;
pub use multibody_set::GpuMultibodySet;
pub use multibody_solver::{GpuMultibodySolver, MultibodySolverArgs};
