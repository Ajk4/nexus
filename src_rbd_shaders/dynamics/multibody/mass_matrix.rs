//! Whole-multibody matrix assembly across the simulation batch.

#[cfg(feature = "dim3")]
use super::ws_soa::{WS_RB_VELS, ws_vel_ang};
use super::{
    types::{MultibodyInfo, MultibodyLinkStatic},
    ws_soa::{WsAddr, ws_world_inertia},
};
use crate::{
    ANG_DIM, AngVector, DIM, Vector,
    dynamics::SPATIAL_DIM,
    gdot,
    utils::{BatchIndices, linalg::MatSlice},
};
use glamx::Vec4;
use khal_std::{
    glamx::UVec3,
    index::MaybeIndexUnchecked,
    macros::{spirv, spirv_bindgen},
};

fn _linear(data: &[f32], matrix: MatSlice, column: u32) -> Vector {
    #[cfg(feature = "dim2")]
    {
        Vector::new(
            data.read(matrix.idx(0, column)),
            data.read(matrix.idx(1, column)),
        )
    }
    #[cfg(feature = "dim3")]
    {
        Vector::new(
            data.read(matrix.idx(0, column)),
            data.read(matrix.idx(1, column)),
            data.read(matrix.idx(2, column)),
        )
    }
}

#[cfg(all(test, feature = "dim2", not(target_arch_is_gpu)))]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn batched_mass_and_coriolis_match_spatial_products() {
        let n = 7u32;
        let count = 4u32;
        let nb = 3u32;
        let dt = 0.01;
        let batches = BatchIndices {
            num_batches: nb,
            multibodies_len: 1,
            dof_batch_capacity: n,
            coriolis_batch_capacity: count * DIM * n,
            mass_matrix_acc_section_offset: n * n,
            ..Default::default()
        };
        let infos = vec![
            MultibodyInfo {
                ndofs: n,
                num_links: count,
                ..Default::default()
            };
            nb as usize
        ];
        let mut links = vec![MultibodyLinkStatic::zeroed(); (count * nb) as usize];
        for k in 0..count {
            for b in 0..nb {
                let properties = &mut links[(k * nb + b) as usize].local_mprops;
                properties.inv_mass.x = if k == count - 1 {
                    0.0
                } else {
                    1.0 / (k + b + 1) as f32
                };
                properties.inv_inertia = 1.0 / (0.3 + k as f32 * 0.1);
            }
        }
        let jac: Vec<_> = (0..count * SPATIAL_DIM as u32 * n * nb)
            .map(|i| (i as f32 * 0.23).sin())
            .collect();
        let cor: Vec<_> = (0..count * SPATIAL_DIM as u32 * n * nb)
            .map(|i| (i as f32 * 0.19).cos())
            .collect();
        let mut dofs = vec![0.0; (6 * n * nb) as usize];
        for b in 0..nb {
            for d in 0..n {
                dofs[((n + d) * nb + b) as usize] = 0.2;
                dofs[((2 * n + d) * nb + b) as usize] = 0.1;
                dofs[((3 * n + d) * nb + b) as usize] = 0.4;
            }
            dofs[((5 * n + b) * nb + b) as usize] = 1.0;
        }
        let mut matrices = vec![0.0; (2 * n * n * nb) as usize];
        for b in 0..nb {
            for c in 0..n {
                for r in 0..n {
                    gpu_mb_assemble_mass_matrix(
                        UVec3::new(c * n + r, b, 0),
                        &infos,
                        &links,
                        &[],
                        &jac,
                        &cor,
                        &dofs,
                        &mut matrices,
                        &dt,
                        &batches,
                    );
                    let mut expected = 0.0;
                    let mut correction = 0.0;
                    for k in 0..count - 1 {
                        let mass = (k + b + 1) as f32;
                        let inertia = 0.3 + k as f32 * 0.1;
                        for axis in 0..3 {
                            let j =
                                |d| jac[(b * count * 3 * n + k * 3 * n + d * 3 + axis) as usize];
                            let weight = if axis < 2 { mass } else { inertia };
                            let ci = if axis < 2 {
                                b * count * 2 * n + k * 2 * n + c * 2 + axis
                            } else {
                                count * 2 * n * nb + b * count * n + k * n + c
                            };
                            expected += j(r) * weight * j(c);
                            correction += j(r) * weight * cor[ci as usize] * dt;
                        }
                    }
                    if r == c {
                        expected += 0.2 * dt + 0.1 + 0.4 * dt * dt;
                    }
                    if r == b || c == b {
                        expected = if r == c { 1.0 } else { 0.0 };
                        correction = 0.0;
                    }
                    let index = (b * n * n + c * n + r) as usize;
                    assert!((matrices[index] - expected).abs() < 1e-5);
                    assert!(
                        (matrices[(n * n * nb) as usize + index] - expected - correction).abs()
                            < 1e-5
                    );
                }
            }
        }
    }
}

fn _angular(data: &[f32], matrix: MatSlice, column: u32) -> AngVector {
    #[cfg(feature = "dim2")]
    {
        data.read(matrix.idx(0, column))
    }
    #[cfg(feature = "dim3")]
    {
        Vector::new(
            data.read(matrix.idx(0, column)),
            data.read(matrix.idx(1, column)),
            data.read(matrix.idx(2, column)),
        )
    }
}

/// Each invocation accumulates one matrix entry in registers. Adjacent lanes
/// write adjacent rows, without a matrix read/write for every link.
#[spirv_bindgen]
#[spirv(compute(threads(128)))]
pub fn gpu_mb_assemble_mass_matrix(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] infos: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] workspace: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] coriolis: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] dofs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] matrices: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] dt: &f32,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batches: &BatchIndices,
) {
    let batch = id.y / batches.multibodies_len;
    let mb = batches
        .ib(batch, infos)
        .read((id.y % batches.multibodies_len) as usize);
    let n = mb.ndofs;
    if id.x >= n * n {
        return;
    }
    let row = id.x % n;
    let column = id.x / n;
    let wa = WsAddr::new(mb.first_link as usize, batches.num_batches, batch);
    let links = batches.ib(batch, links).offset(mb.first_link as usize);
    let split = batches.mass_matrix_acc_section_offset != 0;
    let jac = batches.mb_region(
        batch,
        mb.jacobian_offset,
        mb.num_links * SPATIAL_DIM as u32 * n,
    );
    let cv = batches.mb_region(batch, mb.coriolis_offset, mb.num_links * DIM * n);
    let cw = batches.mb_region(
        batch,
        batches.coriolis_batch_capacity + mb.coriolis_offset / DIM * ANG_DIM,
        mb.num_links * ANG_DIM * n,
    );
    let mut mass = 0.0;
    let mut correction = 0.0;
    for k in 0..mb.num_links {
        let properties = links[k as usize].local_mprops;
        if properties.inv_mass.x == 0.0 {
            continue;
        }
        let weight = 1.0 / properties.inv_mass.x;
        let inertia = ws_world_inertia(workspace, wa, k, &properties);
        let j = MatSlice::dense(
            jac + k as usize * SPATIAL_DIM * n as usize,
            SPATIAL_DIM as u32,
            n,
        );
        let jv = j.fixed_rows(0, DIM);
        let jw = j.fixed_rows(DIM, ANG_DIM);
        let vr = _linear(jacobians, jv, row);
        let wr = _angular(jacobians, jw, row);
        let vc = _linear(jacobians, jv, column);
        let wc = _angular(jacobians, jw, column);
        mass += vr.dot(vc * weight) + gdot(wr, inertia * wc);
        if split {
            let v = MatSlice::dense(cv + k as usize * DIM as usize * n as usize, DIM, n);
            let w = MatSlice::dense(cw + k as usize * ANG_DIM as usize * n as usize, ANG_DIM, n);
            let linear = _linear(coriolis, v, column) * (weight * *dt);
            let angular = (inertia * _angular(coriolis, w, column)) * *dt;
            #[cfg(feature = "dim3")]
            let angular = {
                use crate::utils::linalg::skew;
                let omega = ws_vel_ang(workspace, wa, k, WS_RB_VELS);
                let gyro = skew(omega) * inertia - skew(inertia * omega);
                angular + (gyro * wc) * *dt
            };
            correction += vr.dot(linear) + gdot(wr, angular);
        }
    }
    let state = batches.ib(batch, dofs).offset(mb.first_dof as usize);
    let stride = batches.dof_batch_capacity as usize;
    if row == column {
        let d = row as usize;
        mass += state[stride + d] * *dt + state[2 * stride + d] + state[3 * stride + d] * *dt * *dt;
    }
    if state[5 * stride + row as usize] != 0.0 || state[5 * stride + column as usize] != 0.0 {
        mass = if row == column { 1.0 } else { 0.0 };
        correction = 0.0;
    }
    let base = batches.mb_region(batch, mb.mass_matrix_offset, n * n);
    matrices.write(base + id.x as usize, mass);
    if split {
        let base = batches.mb_region(
            batch,
            batches.mass_matrix_acc_section_offset + mb.mass_matrix_offset,
            n * n,
        );
        matrices.write(base + id.x as usize, correction + mass);
    }
}
