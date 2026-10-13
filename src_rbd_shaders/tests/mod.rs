//! Tests for collision detection algorithms.
//!
//! These tests run on the CPU and call the shader functions directly.

mod epa;
mod gjk;
#[cfg(feature = "dim3")]
mod linalg;
#[cfg(feature = "dim2")]
mod linalg2d;
mod pfm_pfm;
