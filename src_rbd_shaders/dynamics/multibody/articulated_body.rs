//! Articulated-body factorization and inverse-mass application.
//!
//! Spatial vectors are ordered (linear, angular). The implicit operator
//! carries the Coriolis derivative as a second spatial vector. Its force
//! remains spatial, so every elimination uses joint-sized matrices.

use super::{
    types::{MultibodyInfo, MultibodyLinkStatic},
    ws_soa::*,
};
use crate::dynamics::{SPATIAL_DIM, Velocity};
use crate::utils::{BatchIndices, ISlice, ISliceMut, linalg::VSlice};
#[cfg(feature = "dim3")]
use crate::{ANG_DIM, DIM};
use crate::{AngVector, Pose, Vector, gcross_av};
use glamx::Vec4;
use khal_std::{
    glamx::UVec3,
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};
#[cfg(feature = "dim3")]
use parry::math::VectorExt;

const S: usize = SPATIAL_DIM;
pub const TREE_SCRATCH_WIDTH: usize = 2 * S;
type Matrix = [f32; S * S];
type Spatial = [f32; S];

#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TreeSolveFactor {
    inverse: Matrix,
    reaction: Matrix,
    parent_velocity: Matrix,
    parent_coriolis: Matrix,
}

#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ArticulatedBodyFactor {
    parent: u32,
    dof: u32,
    ndofs: u32,
    _padding: u32,
    transform: Matrix,
    motion: Matrix,
    coriolis_parent: Matrix,
    coriolis_motion: Matrix,
    plain: TreeSolveFactor,
    implicit: TreeSolveFactor,
    plain_inertia: Matrix,
    implicit_inertia: Matrix,
    implicit_derivative: Matrix,
}

/// Scalar fields are interleaved across batches. Integer metadata and float
/// bit patterns share one allocation, preserving the shader binding budget.
pub const ARTICULATED_FACTOR_WORDS: usize = 4 + 15 * S * S;

fn _read_matrix(words: &ISlice<u32>, offset: usize) -> Matrix {
    let mut matrix = [0.; S * S];
    for i in 0..S * S {
        matrix[i] = f32::from_bits(words[offset + i]);
    }
    matrix
}

fn _read_factor(words: &ISlice<u32>, node: usize) -> ArticulatedBodyFactor {
    let offset = node * ARTICULATED_FACTOR_WORDS;
    ArticulatedBodyFactor {
        parent: words[offset],
        dof: words[offset + 1],
        ndofs: words[offset + 2],
        _padding: 0,
        transform: _read_matrix(words, offset + 4),
        motion: _read_matrix(words, offset + 4 + S * S),
        coriolis_parent: _read_matrix(words, offset + 4 + 2 * S * S),
        coriolis_motion: _read_matrix(words, offset + 4 + 3 * S * S),
        plain: TreeSolveFactor {
            inverse: _read_matrix(words, offset + 4 + 4 * S * S),
            reaction: _read_matrix(words, offset + 4 + 5 * S * S),
            parent_velocity: _read_matrix(words, offset + 4 + 6 * S * S),
            parent_coriolis: _read_matrix(words, offset + 4 + 7 * S * S),
        },
        implicit: TreeSolveFactor {
            inverse: _read_matrix(words, offset + 4 + 8 * S * S),
            reaction: _read_matrix(words, offset + 4 + 9 * S * S),
            parent_velocity: _read_matrix(words, offset + 4 + 10 * S * S),
            parent_coriolis: _read_matrix(words, offset + 4 + 11 * S * S),
        },
        plain_inertia: _read_matrix(words, offset + 4 + 12 * S * S),
        implicit_inertia: _read_matrix(words, offset + 4 + 13 * S * S),
        implicit_derivative: _read_matrix(words, offset + 4 + 14 * S * S),
    }
}

fn _write_factor(words: &mut ISliceMut<u32>, node: usize, factor: ArticulatedBodyFactor) {
    let offset = node * ARTICULATED_FACTOR_WORDS;
    words[offset] = factor.parent;
    words[offset + 1] = factor.dof;
    words[offset + 2] = factor.ndofs;
    words[offset + 3] = 0;
    for i in 0..S * S {
        words[offset + 4 + i] = factor.transform[i].to_bits();
        words[offset + 4 + S * S + i] = factor.motion[i].to_bits();
        words[offset + 4 + 2 * S * S + i] = factor.coriolis_parent[i].to_bits();
        words[offset + 4 + 3 * S * S + i] = factor.coriolis_motion[i].to_bits();
        words[offset + 4 + 4 * S * S + i] = factor.plain.inverse[i].to_bits();
        words[offset + 4 + 5 * S * S + i] = factor.plain.reaction[i].to_bits();
        words[offset + 4 + 6 * S * S + i] = factor.plain.parent_velocity[i].to_bits();
        words[offset + 4 + 7 * S * S + i] = factor.plain.parent_coriolis[i].to_bits();
        words[offset + 4 + 8 * S * S + i] = factor.implicit.inverse[i].to_bits();
        words[offset + 4 + 9 * S * S + i] = factor.implicit.reaction[i].to_bits();
        words[offset + 4 + 10 * S * S + i] = factor.implicit.parent_velocity[i].to_bits();
        words[offset + 4 + 11 * S * S + i] = factor.implicit.parent_coriolis[i].to_bits();
        words[offset + 4 + 12 * S * S + i] = factor.plain_inertia[i].to_bits();
        words[offset + 4 + 13 * S * S + i] = factor.implicit_inertia[i].to_bits();
        words[offset + 4 + 14 * S * S + i] = factor.implicit_derivative[i].to_bits();
    }
}

fn _transpose(a: Matrix) -> Matrix {
    let mut out = [0.; S * S];
    for r in 0..S {
        for c in 0..S {
            out[r * S + c] = a[c * S + r];
        }
    }
    out
}
fn _multiply(a: Matrix, b: Matrix) -> Matrix {
    let mut out = [0.; S * S];
    for r in 0..S {
        for c in 0..S {
            let mut error = 0.;
            for k in 0..S {
                let term = a[r * S + k] * b[k * S + c] - error;
                let next = out[r * S + c] + term;
                error = (next - out[r * S + c]) - term;
                out[r * S + c] = next;
            }
        }
    }
    out
}
fn _apply(a: Matrix, b: Spatial) -> Spatial {
    let mut out = [0.; S];
    for r in 0..S {
        let mut error = 0.;
        for c in 0..S {
            let term = a[r * S + c] * b[c] - error;
            let next = out[r] + term;
            error = (next - out[r]) - term;
            out[r] = next;
        }
    }
    out
}
fn _add(a: Matrix, b: Matrix) -> Matrix {
    let mut out = a;
    for i in 0..S * S {
        out[i] += b[i];
    }
    out
}
fn _subtract(a: Matrix, b: Matrix) -> Matrix {
    let mut out = a;
    for i in 0..S * S {
        out[i] -= b[i];
    }
    out
}
fn _inverse(mut a: Matrix, n: usize) -> Matrix {
    let mut out = [0.; S * S];
    for i in 0..n {
        out[i * S + i] = 1.;
    }
    for k in 0..n {
        let mut pivot = k;
        for r in k + 1..n {
            if a[r * S + k].abs() > a[pivot * S + k].abs() {
                pivot = r;
            }
        }
        for c in 0..n {
            let value = a[k * S + c];
            a[k * S + c] = a[pivot * S + c];
            a[pivot * S + c] = value;
            let value = out[k * S + c];
            out[k * S + c] = out[pivot * S + c];
            out[pivot * S + c] = value;
        }
        let reciprocal = 1. / a[k * S + k];
        for c in 0..n {
            a[k * S + c] *= reciprocal;
            out[k * S + c] *= reciprocal;
        }
        for r in 0..n {
            if r == k {
                continue;
            }
            let scale = a[r * S + k];
            for c in 0..n {
                a[r * S + c] -= scale * a[k * S + c];
                out[r * S + c] -= scale * out[k * S + c];
            }
        }
    }
    out
}

fn _spatial(linear: Vector, angular: AngVector) -> Spatial {
    #[cfg(feature = "dim2")]
    {
        [linear.x, linear.y, angular]
    }
    #[cfg(feature = "dim3")]
    {
        [
            linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
        ]
    }
}
fn _parts(v: Spatial) -> (Vector, AngVector) {
    #[cfg(feature = "dim2")]
    {
        (Vector::new(v[0], v[1]), v[2])
    }
    #[cfg(feature = "dim3")]
    {
        (Vector::new(v[0], v[1], v[2]), Vector::new(v[3], v[4], v[5]))
    }
}
fn _basis(c: usize) -> Spatial {
    let mut v = [0.; S];
    v[c] = 1.;
    v
}
fn _column(a: &mut Matrix, c: usize, v: Spatial) {
    for r in 0..S {
        a[r * S + c] = v[r];
    }
}
fn _angular_cross(a: AngVector, b: AngVector) -> AngVector {
    #[cfg(feature = "dim2")]
    {
        let _ = (a, b);
        0.
    }
    #[cfg(feature = "dim3")]
    {
        a.cross(b)
    }
}

fn _eliminate(
    a: Matrix,
    b: Matrix,
    motion: Matrix,
    derivative: Matrix,
    diagonal: Spatial,
    ndofs: usize,
) -> (TreeSolveFactor, Matrix, Matrix) {
    let projection = _transpose(motion);
    let u = _add(_multiply(a, motion), _multiply(b, derivative));
    let va = _multiply(projection, a);
    let vb = _multiply(projection, b);
    let mut d = _multiply(projection, u);
    for i in 0..ndofs {
        d[i * S + i] += diagonal[i];
    }
    let inverse = _inverse(d, ndofs);
    let reaction = _multiply(u, inverse);
    let reduced_a = _subtract(a, _multiply(reaction, va));
    let reduced_b = _subtract(b, _multiply(reaction, vb));
    (
        TreeSolveFactor {
            inverse,
            reaction,
            parent_velocity: _multiply(inverse, va),
            parent_coriolis: _multiply(inverse, vb),
        },
        reduced_a,
        reduced_b,
    )
}

/// Backward elimination of each link's joint, followed by a forward solve.
/// Scratch is caller-owned and separate for each simultaneous right-hand side.
pub(super) fn _solve_tree(
    mb: &MultibodyInfo,
    factors: &ISlice<u32>,
    implicit: bool,
    rhs: &mut [f32],
    row: VSlice,
    scratch: &mut [f32],
    scratch_base: usize,
    scratch_stride: usize,
) {
    for i in 0..mb.num_links as usize * TREE_SCRATCH_WIDTH {
        scratch.write(scratch_base + i * scratch_stride, 0.);
    }
    for reverse in 0..mb.num_links {
        let k = mb.num_links - 1 - reverse;
        let node = _read_factor(factors, k as usize);
        let factor = if implicit { node.implicit } else { node.plain };
        let mut force = [0.; S];
        for i in 0..S {
            force[i] =
                scratch.read(scratch_base + (k as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride);
        }
        let projection = _apply(_transpose(node.motion), force);
        let mut u = [0.; S];
        for i in 0..node.ndofs as usize {
            u[i] = rhs.read(row.at(node.dof + i as u32)) - projection[i];
        }
        let q = _apply(factor.inverse, u);
        for i in 0..node.ndofs as usize {
            rhs.write(row.at(node.dof + i as u32), q[i]);
        }
        if k != 0 {
            let reaction = _apply(factor.reaction, u);
            for i in 0..S {
                force[i] += reaction[i];
            }
            let parent = _apply(_transpose(node.transform), force);
            for i in 0..S {
                let at =
                    scratch_base + (node.parent as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride;
                scratch.write(at, scratch.read(at) + parent[i]);
            }
        }
    }
    for k in 0..mb.num_links {
        let node = _read_factor(factors, k as usize);
        let factor = if implicit { node.implicit } else { node.plain };
        let mut parent_v = [0.; S];
        let mut parent_c = [0.; S];
        if k != 0 {
            for i in 0..S {
                parent_v[i] = scratch.read(
                    scratch_base + (node.parent as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride,
                );
                if implicit {
                    parent_c[i] = scratch.read(
                        scratch_base
                            + (node.parent as usize * TREE_SCRATCH_WIDTH + S + i) * scratch_stride,
                    );
                }
            }
        }
        let mut velocity = _apply(node.transform, parent_v);
        let mut coriolis = _apply(node.transform, parent_c);
        if implicit {
            let change = _apply(node.coriolis_parent, parent_v);
            for i in 0..S {
                coriolis[i] += change[i];
            }
        }
        let from_v = _apply(factor.parent_velocity, velocity);
        let from_c = if implicit {
            _apply(factor.parent_coriolis, coriolis)
        } else {
            [0.; S]
        };
        let mut q = [0.; S];
        for i in 0..node.ndofs as usize {
            q[i] = rhs.read(row.at(node.dof + i as u32)) - from_v[i] - from_c[i];
            rhs.write(row.at(node.dof + i as u32), q[i]);
        }
        let joint_v = _apply(node.motion, q);
        let joint_c = if implicit {
            _apply(node.coriolis_motion, q)
        } else {
            [0.; S]
        };
        for i in 0..S {
            velocity[i] += joint_v[i];
            coriolis[i] += joint_c[i];
            scratch.write(
                scratch_base + (k as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride,
                velocity[i],
            );
            if implicit {
                scratch.write(
                    scratch_base + (k as usize * TREE_SCRATCH_WIDTH + S + i) * scratch_stride,
                    coriolis[i],
                );
            }
        }
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_factor_articulated_body(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] infos: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ws: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dofs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] factors: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] dt: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batches: &BatchIndices,
) {
    if id.x >= batches.num_batches * batches.multibodies_len {
        return;
    }
    let batch = id.x % batches.num_batches;
    let mb = batches
        .ib(batch, infos)
        .read((id.x / batches.num_batches) as usize);
    let stat = batches.ib(batch, links).offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batches.num_batches, batch);
    let state = batches.ib(batch, dofs).offset(mb.first_dof as usize);
    let stride = batches.dof_batch_capacity as usize;
    let mut tree = batches
        .ib_mut(batch, factors)
        .offset(mb.first_link as usize * ARTICULATED_FACTOR_WORDS);
    for k in 0..mb.num_links {
        let link = stat[k as usize];
        let parent_pose = if k == 0 {
            Pose::default()
        } else {
            ws_pose(ws, wa, link.parent_link_id, WS_LTW)
        };
        let parent_w = if k == 0 {
            AngVector::default()
        } else {
            ws_vel_ang(ws, wa, link.parent_link_id, WS_RB_VELS)
        };
        let w = ws_vel_ang(ws, wa, k, WS_RB_VELS);
        let joint = ws_vel(ws, wa, k, WS_JOINT_VEL);
        let r02 = ws_vec(ws, wa, k, WS_SHIFT02);
        let r23 = ws_vec(ws, wa, k, WS_SHIFT23);
        let mut x = [0.; S * S];
        let mut motion = [0.; S * S];
        let mut f = [0.; S * S];
        let mut t = [0.; S * S];
        for c in 0..S {
            let (v, dw) = _parts(_basis(c));
            _column(&mut x, c, _spatial(v + gcross_av(dw, r02 + r23), dw));
            if k != 0 {
                let cw = -_angular_cross(joint.angular, dw);
                let cv = gcross_av(dw, gcross_av(w, r02) + joint.linear * 3.)
                    + gcross_av(parent_w, gcross_av(dw, r02))
                    + gcross_av(cw, r23)
                    + gcross_av(dw, gcross_av(w, r23))
                    + gcross_av(w, gcross_av(dw, r23));
                _column(&mut f, c, _spatial(cv, cw));
            }
        }
        for c in 0..link.ndofs as usize {
            let (v, dw) = link.joint_jacobian_column(
                parent_pose.rotation * link.data.local_frame_a.rotation,
                c as u32,
            );
            if state[5 * stride + link.assembly_id as usize + c] == 0. {
                _column(&mut motion, c, _spatial(v + gcross_av(dw, r23), dw));
                let cw = if k == 0 {
                    AngVector::default()
                } else {
                    _angular_cross(parent_w, dw)
                };
                let cv = if k == 0 {
                    Vector::ZERO
                } else {
                    gcross_av(parent_w, v) * 2.
                } + gcross_av(cw, r23)
                    + gcross_av(dw, gcross_av(w, r23))
                    + gcross_av(w, gcross_av(dw, r23));
                _column(&mut t, c, _spatial(cv, cw));
            }
        }
        let mut inertia = [0.; S * S];
        if link.local_mprops.inv_mass.x != 0. {
            let mass = 1. / link.local_mprops.inv_mass.x;
            let angular_inertia = ws_world_inertia(ws, wa, k, &link.local_mprops);
            for c in 0..S {
                let (v, dw) = _parts(_basis(c));
                _column(&mut inertia, c, _spatial(v * mass, angular_inertia * dw));
            }
        }
        let implicit_inertia = inertia;
        #[cfg(feature = "dim3")]
        let implicit_inertia = {
            let mut implicit_inertia = implicit_inertia;
            use crate::utils::linalg::skew;
            let i = ws_world_inertia(ws, wa, k, &link.local_mprops);
            let gyro = skew(w) * i - skew(i * w);
            for c in 0..ANG_DIM as usize {
                let column = gyro * Vector::ith(c, *dt);
                for r in 0..ANG_DIM as usize {
                    implicit_inertia[(DIM as usize + r) * S + DIM as usize + c] += column.vget(r);
                }
            }
            implicit_inertia
        };
        let mut derivative = inertia;
        for i in 0..S * S {
            derivative[i] *= *dt;
        }
        let zero = TreeSolveFactor {
            inverse: [0.; S * S],
            reaction: [0.; S * S],
            parent_velocity: [0.; S * S],
            parent_coriolis: [0.; S * S],
        };
        _write_factor(
            &mut tree,
            k as usize,
            ArticulatedBodyFactor {
                parent: link.parent_link_id,
                dof: link.assembly_id,
                ndofs: link.ndofs,
                _padding: 0,
                transform: x,
                motion,
                coriolis_parent: f,
                coriolis_motion: t,
                plain: zero,
                implicit: zero,
                plain_inertia: inertia,
                implicit_inertia,
                implicit_derivative: derivative,
            },
        );
    }
    for reverse in 0..mb.num_links {
        let k = mb.num_links - 1 - reverse;
        let mut node = _read_factor(&tree.as_ref(), k as usize);
        let mut diagonal = [0.; S];
        for c in 0..node.ndofs as usize {
            let d = node.dof as usize + c;
            diagonal[c] = if state[5 * stride + d] != 0. {
                1.
            } else {
                state[stride + d] * *dt + state[2 * stride + d] + state[3 * stride + d] * *dt * *dt
            };
        }
        let (plain, pa, _) = _eliminate(
            node.plain_inertia,
            [0.; S * S],
            node.motion,
            [0.; S * S],
            diagonal,
            node.ndofs as usize,
        );
        let (implicit, ia, ib) = _eliminate(
            node.implicit_inertia,
            node.implicit_derivative,
            node.motion,
            node.coriolis_motion,
            diagonal,
            node.ndofs as usize,
        );
        node.plain = plain;
        node.implicit = implicit;
        _write_factor(&mut tree, k as usize, node);
        if k != 0 {
            let mut parent = _read_factor(&tree.as_ref(), node.parent as usize);
            let xt = _transpose(node.transform);
            parent.plain_inertia = _add(
                parent.plain_inertia,
                _multiply(xt, _multiply(pa, node.transform)),
            );
            parent.implicit_inertia = _add(
                parent.implicit_inertia,
                _multiply(
                    xt,
                    _add(
                        _multiply(ia, node.transform),
                        _multiply(ib, node.coriolis_parent),
                    ),
                ),
            );
            parent.implicit_derivative = _add(
                parent.implicit_derivative,
                _multiply(xt, _multiply(ib, node.transform)),
            );
            _write_factor(&mut tree, node.parent as usize, parent);
        }
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_articulated_body(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] infos: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] factors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] rhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] scratch: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batches: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] implicit: &u32,
) {
    if id.x >= batches.num_batches * batches.multibodies_len {
        return;
    }
    let batch = id.x / batches.multibodies_len;
    let mb = batches
        .ib(batch, infos)
        .read((id.x % batches.multibodies_len) as usize);
    let tree = batches
        .ib(batch, factors)
        .offset(mb.first_link as usize * ARTICULATED_FACTOR_WORDS);
    let row = VSlice::dense(batches.mb_region(batch, mb.first_dof, mb.ndofs));
    _solve_tree(
        &mb,
        &tree,
        *implicit != 0,
        rhs,
        row,
        scratch,
        id.x as usize * batches.mb_max_links as usize * TREE_SCRATCH_WIDTH,
        1,
    );
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_apply_articulated_body_forces(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] infos: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ws: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dofs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] factors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] rhs: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] scratch: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dt: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] gravity: &Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] batches: &BatchIndices,
) {
    if id.x >= batches.num_batches * batches.multibodies_len {
        return;
    }
    let batch = id.x / batches.multibodies_len;
    let mb = batches
        .ib(batch, infos)
        .read((id.x % batches.multibodies_len) as usize);
    let stat = batches.ib(batch, links).offset(mb.first_link as usize);
    let tree = batches
        .ib(batch, factors)
        .offset(mb.first_link as usize * ARTICULATED_FACTOR_WORDS);
    let state = batches.ib(batch, dofs).offset(mb.first_dof as usize);
    let stride = batches.dof_batch_capacity as usize;
    let base = id.x as usize * batches.mb_max_links as usize * TREE_SCRATCH_WIDTH;
    let scratch_stride = 1;
    let row = VSlice::dense(batches.mb_region(batch, mb.first_dof, mb.ndofs));
    let wa = WsAddr::new(mb.first_link as usize, batches.num_batches, batch);
    #[cfg(feature = "dim2")]
    let gravity = Vector::new(gravity.x, gravity.y);
    #[cfg(feature = "dim3")]
    let gravity = Vector::new(gravity.x, gravity.y, gravity.z);
    for k in 0..mb.num_links {
        let node = _read_factor(&tree, k as usize);
        let properties = stat[k as usize].local_mprops;
        let joint = ws_vel(ws, wa, k, WS_JOINT_VEL);
        let r02 = ws_vec(ws, wa, k, WS_SHIFT02);
        let r23 = ws_vec(ws, wa, k, WS_SHIFT23);
        let omega = ws_vel_ang(ws, wa, k, WS_RB_VELS);
        let mut acceleration = Velocity::new(Vector::ZERO, AngVector::default());
        if k != 0 {
            let parent = ws_vel(ws, wa, node.parent, WS_KIN_ACC);
            let parent_w = ws_vel_ang(ws, wa, node.parent, WS_RB_VELS);
            acceleration.linear = parent.linear
                + gcross_av(parent_w, joint.linear) * 2.
                + gcross_av(parent_w, gcross_av(parent_w, r02))
                + gcross_av(parent.angular, r02);
            acceleration.angular = parent.angular + _angular_cross(parent_w, joint.angular);
        }
        acceleration.linear +=
            gcross_av(omega, gcross_av(omega, r23)) + gcross_av(acceleration.angular, r23);
        ws_set_vel(ws, wa, k, WS_KIN_ACC, acceleration);
        let mut force = [0.; S];
        if properties.inv_mass.x != 0. {
            let mass = 1. / properties.inv_mass.x;
            let inertia = ws_world_inertia(ws, wa, k, &properties);
            let (external, torque, scale) = ws_ext_wrench(ws, wa, k);
            force = _spatial(
                gravity * (mass * scale) + external - acceleration.linear * mass,
                torque - _angular_cross(omega, inertia * omega) - inertia * acceleration.angular,
            );
        }
        for i in 0..S {
            scratch.write(
                base + (k as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride,
                force[i],
            );
        }
    }
    for reverse in 0..mb.num_links {
        let k = mb.num_links - 1 - reverse;
        let node = _read_factor(&tree, k as usize);
        let link = stat[k as usize];
        let mut force = [0.; S];
        for i in 0..S {
            force[i] = scratch.read(base + (k as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride);
        }
        let projected = _apply(_transpose(node.motion), force);
        let mut c = 0usize;
        for axis in 0..S {
            if link.data.locked_axes & (1 << axis) != 0 {
                continue;
            }
            let d = node.dof as usize + c;
            let velocity = state[d];
            let stiffness = state[3 * stride + d];
            let spring = stiffness * (ws_coord(ws, wa, k, axis as u32) - state[4 * stride + d])
                + stiffness * *dt * velocity;
            let value = if state[5 * stride + d] == 0. {
                projected[c] - state[stride + d] * velocity - spring
            } else {
                0.
            };
            rhs.write(row.at(d as u32), value);
            c += 1;
        }
        if k != 0 {
            let parent = _apply(_transpose(node.transform), force);
            for i in 0..S {
                let at = base + (node.parent as usize * TREE_SCRATCH_WIDTH + i) * scratch_stride;
                scratch.write(at, scratch.read(at) + parent[i]);
            }
        }
    }
    _solve_tree(
        &mb,
        &tree,
        batches.mass_matrix_acc_section_offset != 0,
        rhs,
        row,
        scratch,
        base,
        scratch_stride,
    );
}
