//! LU decomposition + solve.
//!
//! Split into two kernels so the factorization can be reused across multiple
//! right-hand sides within a frame (e.g. gravity τ, contact impulses, …).
//! Dispatched one workgroup per `(multibody, batch)` pair.

use khal_std::index::MaybeIndexUnchecked;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::utils::linalg::{MAX_MB_DOFS, VSlice};
use crate::utils::shared::{_read as _broadcast_read, _write as _broadcast_write};

/// Workgroup width for the parallelised LU kernels. Must match the
/// `threads(N, 1, 1)` attribute and `MB_LU_LANES` on the host side.
pub(super) const LANES: u32 = 64;

/// Side length of the workgroup-shared matrix tile. Must equal both the lane
/// count and the maximum supported `ndofs`.
const MAX_DOFS_U32: u32 = MAX_MB_DOFS as u32;

#[cfg(all(test, feature = "cpu", not(target_arch_is_gpu)))]
mod tests {
    use super::*;
    use std::cell::UnsafeCell;

    #[test]
    fn parallel_lu_solves_coupled_dofs() {
        let n = 53;
        let matrix = UnsafeCell::new([0.; MAX_MB_DOFS * MAX_MB_DOFS]);
        let pivots = UnsafeCell::new([0u32; MAX_MB_DOFS]);
        let rhs = UnsafeCell::new([0.; MAX_MB_DOFS]);
        let partial = UnsafeCell::new([0.; LANES as usize]);
        let pivot_row = UnsafeCell::new(0u32);
        let inverse_pivot = UnsafeCell::new(0f32);
        for c in 0..n {
            for r in 0..n {
                let value = if r == c { 2.0 + r as f32 * 0.01 } else { 0.1 };
                unsafe {
                    (*matrix.get())[sm_idx(r, c)] = value;
                    (*rhs.get())[r as usize] += value;
                }
            }
        }
        let pointers = [
            matrix.get() as usize,
            pivots.get() as usize,
            rhs.get() as usize,
            partial.get() as usize,
            pivot_row.get() as usize,
            inverse_pivot.get() as usize,
        ];
        khal_std::arch::cpu::dispatch_workgroup_threads(LANES as usize, |lane| unsafe {
            let matrix = &mut *(pointers[0] as *mut [f32; MAX_MB_DOFS * MAX_MB_DOFS]);
            let pivots = &mut *(pointers[1] as *mut [u32; MAX_MB_DOFS]);
            let rhs = &mut *(pointers[2] as *mut [f32; MAX_MB_DOFS]);
            let partial = &mut *(pointers[3] as *mut [f32; LANES as usize]);
            lu_factor_in_shared(
                n,
                n,
                lane,
                matrix,
                pivots,
                VSlice::dense(0),
                &mut *(pointers[4] as *mut u32),
                &mut *(pointers[5] as *mut f32),
            );
            lu_apply_pivots(n, lane, pivots, VSlice::dense(0), rhs);
            lu_triangular_solve_in_place(n, n, lane, matrix, rhs, partial);
        });
        let solution = unsafe { &*rhs.get() };
        for &value in &solution[..n as usize] {
            assert!((value - 1.0).abs() < 1e-4, "incorrect LU solution: {value}");
        }
    }

    #[test]
    fn packed_lu_solves_coupled_dofs() {
        let n = 5;
        let matrix = UnsafeCell::new([0.; 512]);
        let pivots = UnsafeCell::new([0u32; MAX_MB_DOFS]);
        let rhs = UnsafeCell::new([0.; MAX_MB_DOFS]);
        let partial = UnsafeCell::new([0.; LANES as usize]);
        let pivot_row = UnsafeCell::new([0u32; 8]);
        let inverse_pivot = UnsafeCell::new([0f32; 8]);
        for c in 0..n {
            for r in 0..n {
                let value = if r == c { 2.0 + r as f32 * 0.01 } else { 0.1 };
                unsafe {
                    (*matrix.get())[sm_idx_packed::<8>(0, r, c)] = value;
                    (*rhs.get())[r as usize] += value;
                }
            }
        }
        let pointers = [
            matrix.get() as usize,
            pivots.get() as usize,
            rhs.get() as usize,
            partial.get() as usize,
            pivot_row.get() as usize,
            inverse_pivot.get() as usize,
        ];
        khal_std::arch::cpu::dispatch_workgroup_threads(LANES as usize, |thread| unsafe {
            let matrix = &mut *(pointers[0] as *mut [f32; 512]);
            let pivots = &mut *(pointers[1] as *mut [u32; MAX_MB_DOFS]);
            let rhs = &mut *(pointers[2] as *mut [f32; MAX_MB_DOFS]);
            let partial = &mut *(pointers[3] as *mut [f32; LANES as usize]);
            let slot = thread / 8;
            let lane = thread % 8;
            lu_factor_in_shared_packed::<8, 512, 8>(
                n,
                n,
                slot,
                lane,
                slot == 0,
                matrix,
                pivots,
                VSlice::dense(0),
                &mut *(pointers[4] as *mut [u32; 8]),
                &mut *(pointers[5] as *mut [f32; 8]),
            );
            lu_apply_pivots_packed::<8>(n, slot, lane, slot == 0, pivots, VSlice::dense(0), rhs);
            lu_triangular_solve_in_place_packed::<8, 512>(
                n,
                n,
                slot,
                lane,
                slot == 0,
                matrix,
                rhs,
                partial,
            );
        });
        let solution = unsafe { &*rhs.get() };
        for &value in &solution[..n as usize] {
            assert!(
                (value - 1.0).abs() < 1e-4,
                "incorrect packed LU solution: {value}"
            );
        }
    }
}

/// Index helper for the shared `mat[col * MAX_MB_DOFS + row]` tile.
#[inline]
pub(super) fn sm_idx(r: u32, c: u32) -> usize {
    (c * MAX_DOFS_U32 + r) as usize
}

/// Workgroup-parallel LU factorization on the shared `mat` tile in place.
#[inline]
pub(super) fn lu_factor_in_shared(
    n: u32,
    max_n: u32,
    lane: u32,
    mat: &mut [f32; MAX_MB_DOFS * MAX_MB_DOFS],
    pivots_dst: &mut [u32],
    piv: VSlice,
    pivot_row_shared: &mut u32,
    inv_akk_shared: &mut f32,
) {
    for k in 0..max_n {
        let active = k < n;
        if active && lane == 0 {
            let mut pivot_row = k;
            let mut pivot_val = {
                let v = mat.read(sm_idx(k, k));
                if v >= 0.0 { v } else { -v }
            };
            for i in (k + 1)..n {
                let v = mat.read(sm_idx(i, k));
                let av = if v >= 0.0 { v } else { -v };
                if av > pivot_val {
                    pivot_val = av;
                    pivot_row = i;
                }
            }
            _broadcast_write(pivot_row_shared, pivot_row);
            pivots_dst.write(piv.at(k), pivot_row);
        }
        workgroup_memory_barrier_with_group_sync();
        let pivot_row = _broadcast_read(pivot_row_shared);

        if active && pivot_row != k && lane < n {
            let c = lane;
            let a = mat.read(sm_idx(k, c));
            let b = mat.read(sm_idx(pivot_row, c));
            mat.write(sm_idx(k, c), b);
            mat.write(sm_idx(pivot_row, c), a);
        }
        workgroup_memory_barrier_with_group_sync();

        if active && lane == 0 {
            let akk = mat.read(sm_idx(k, k));
            _broadcast_write(inv_akk_shared, if akk != 0.0 { 1.0 / akk } else { 0.0 });
        }
        workgroup_memory_barrier_with_group_sync();
        let inv_akk = _broadcast_read(inv_akk_shared);

        if active {
            let r = k + 1 + lane;
            if r < n {
                let v = mat.read(sm_idx(r, k)) * inv_akk;
                mat.write(sm_idx(r, k), v);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        if active {
            let j = k + 1 + lane;
            if j < n {
                let akj = mat.read(sm_idx(k, j));
                for r in (k + 1)..n {
                    let lik = mat.read(sm_idx(r, k));
                    let v = mat.read(sm_idx(r, j)) - lik * akj;
                    mat.write(sm_idx(r, j), v);
                }
            }
        }
        workgroup_memory_barrier_with_group_sync();
    }
}

/// Inner of the workgroup-parallel triangular solves used by the LU
/// solve/factor-and-solve kernels. Preconditions: `mat`
/// already holds the LU factors and `x` already holds the permuted rhs.
#[inline]
pub(super) fn lu_triangular_solve_in_place(
    n: u32,
    max_n: u32,
    lane: u32,
    mat: &[f32; MAX_MB_DOFS * MAX_MB_DOFS],
    x: &mut [f32; MAX_MB_DOFS],
    partial: &mut [f32; LANES as usize],
) {
    for i in 0..max_n {
        let active = i < n;
        let s = if active && lane < i {
            mat.read(sm_idx(i, lane)) * _broadcast_read(x.at(lane as usize))
        } else {
            0.0f32
        };
        _broadcast_write(partial.at_mut(lane as usize), s);
        workgroup_memory_barrier_with_group_sync();
        for step in 0..6u32 {
            let stride = 1u32 << (5 - step);
            if lane < stride {
                let v = _broadcast_read(partial.at(lane as usize))
                    + _broadcast_read(partial.at((lane + stride) as usize));
                _broadcast_write(partial.at_mut(lane as usize), v);
            }
            workgroup_memory_barrier_with_group_sync();
        }
        if active && lane == 0 {
            let cur = _broadcast_read(x.at(i as usize));
            _broadcast_write(x.at_mut(i as usize), cur - _broadcast_read(partial.at(0)));
        }
        workgroup_memory_barrier_with_group_sync();
    }

    for step in 0..max_n {
        let active = step < n;
        // For dummy iterations (step >= n), `i` is not meaningful; guard
        // every use of it behind `active`.
        let i = if active { n - 1 - step } else { 0 };
        let s = if active && lane > i && lane < n {
            mat.read(sm_idx(i, lane)) * _broadcast_read(x.at(lane as usize))
        } else {
            0.0f32
        };
        _broadcast_write(partial.at_mut(lane as usize), s);
        workgroup_memory_barrier_with_group_sync();
        for r in 0..6u32 {
            let stride = 1u32 << (5 - r);
            if lane < stride {
                let v = _broadcast_read(partial.at(lane as usize))
                    + _broadcast_read(partial.at((lane + stride) as usize));
                _broadcast_write(partial.at_mut(lane as usize), v);
            }
            workgroup_memory_barrier_with_group_sync();
        }
        if active && lane == 0 {
            let u = mat.read(sm_idx(i, i));
            let cur = _broadcast_read(x.at(i as usize)) - _broadcast_read(partial.at(0));
            _broadcast_write(x.at_mut(i as usize), if u != 0.0 { cur / u } else { 0.0 });
        }
        workgroup_memory_barrier_with_group_sync();
    }
}

/*
 * Packed variants: `SLOTS = 64 / T` multibodies per 64-lane workgroup, each
 * owning `T` lanes and a `T×T` shared tile. Shrinking the tile from
 * `MAX_MB_DOFS² = 16 KB` to `64·T` floats lifts the shared-memory occupancy
 * cap that made the one-multibody-per-workgroup layout latency-bound when
 * every environment holds one small robot. All barriers stay at uniform
 * points: every slot executes the same `max_n`-bounded loops in lock-step,
 * inactive slots simply skip their stores.
 */

/// Index into the packed shared tile: slot-`slot`'s `T×T` column-major tile.
#[inline]
pub(super) fn sm_idx_packed<const T: u32>(slot: u32, r: u32, c: u32) -> usize {
    (slot * T * T + c * T + r) as usize
}

/// Packed [`lu_factor_in_shared`]: factor each slot's `T×T` tile in place.
/// `lane` is slot-relative (`0..T`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn lu_factor_in_shared_packed<const T: u32, const MATN: usize, const SLOTS: usize>(
    n: u32,
    max_n: u32,
    slot: u32,
    lane: u32,
    active_slot: bool,
    mat: &mut [f32; MATN],
    pivots_dst: &mut [u32],
    piv: VSlice,
    pivot_row_shared: &mut [u32; SLOTS],
    inv_akk_shared: &mut [f32; SLOTS],
) {
    for k in 0..max_n {
        let active = active_slot && k < n;
        if active && lane == 0 {
            let mut pivot_row = k;
            let mut pivot_val = {
                let v = mat.read(sm_idx_packed::<T>(slot, k, k));
                if v >= 0.0 { v } else { -v }
            };
            for i in (k + 1)..n {
                let v = mat.read(sm_idx_packed::<T>(slot, i, k));
                let av = if v >= 0.0 { v } else { -v };
                if av > pivot_val {
                    pivot_val = av;
                    pivot_row = i;
                }
            }
            pivot_row_shared.write(slot as usize, pivot_row);
            pivots_dst.write(piv.at(k), pivot_row);
        }
        workgroup_memory_barrier_with_group_sync();
        let pivot_row = pivot_row_shared.read(slot as usize);

        if active && pivot_row != k && lane < n {
            let c = lane;
            let a = mat.read(sm_idx_packed::<T>(slot, k, c));
            let b = mat.read(sm_idx_packed::<T>(slot, pivot_row, c));
            mat.write(sm_idx_packed::<T>(slot, k, c), b);
            mat.write(sm_idx_packed::<T>(slot, pivot_row, c), a);
        }
        workgroup_memory_barrier_with_group_sync();

        if active && lane == 0 {
            let akk = mat.read(sm_idx_packed::<T>(slot, k, k));
            inv_akk_shared.write(slot as usize, if akk != 0.0 { 1.0 / akk } else { 0.0 });
        }
        workgroup_memory_barrier_with_group_sync();
        let inv_akk = inv_akk_shared.read(slot as usize);

        if active {
            let r = k + 1 + lane;
            if r < n {
                let v = mat.read(sm_idx_packed::<T>(slot, r, k)) * inv_akk;
                mat.write(sm_idx_packed::<T>(slot, r, k), v);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        if active {
            let j = k + 1 + lane;
            if j < n {
                let akj = mat.read(sm_idx_packed::<T>(slot, k, j));
                for r in (k + 1)..n {
                    let lik = mat.read(sm_idx_packed::<T>(slot, r, k));
                    let v = mat.read(sm_idx_packed::<T>(slot, r, j)) - lik * akj;
                    mat.write(sm_idx_packed::<T>(slot, r, j), v);
                }
            }
        }
        workgroup_memory_barrier_with_group_sync();
    }
}

/// Packed [`lu_triangular_solve_in_place`]: per-slot `x`/`partial` segments
/// live at `slot·T`; the tree reduction runs `log2(T)` levels within each
/// slot's segment, all slots in lock-step.
#[inline]
pub(super) fn lu_triangular_solve_in_place_packed<const T: u32, const MATN: usize>(
    n: u32,
    max_n: u32,
    slot: u32,
    lane: u32,
    active_slot: bool,
    mat: &[f32; MATN],
    x: &mut [f32; 64],
    partial: &mut [f32; 64],
) {
    let seg = (slot * T) as usize;
    let log2_t = T.trailing_zeros();

    for i in 0..max_n {
        let active = active_slot && i < n;
        let s = if active && lane < i {
            mat.read(sm_idx_packed::<T>(slot, i, lane)) * x.read(seg + lane as usize)
        } else {
            0.0f32
        };
        partial.write(seg + lane as usize, s);
        workgroup_memory_barrier_with_group_sync();
        for step in 0..log2_t {
            let stride = T >> (step + 1);
            if lane < stride {
                let v = partial.read(seg + lane as usize)
                    + partial.read(seg + (lane + stride) as usize);
                partial.write(seg + lane as usize, v);
            }
            workgroup_memory_barrier_with_group_sync();
        }
        if active && lane == 0 {
            let cur = x.read(seg + i as usize);
            x.write(seg + i as usize, cur - partial.read(seg));
        }
        workgroup_memory_barrier_with_group_sync();
    }

    for step in 0..max_n {
        let active = active_slot && step < n;
        // For dummy iterations (step >= n), `i` is not meaningful; guard
        // every use of it behind `active`.
        let i = if active { n - 1 - step } else { 0 };
        let s = if active && lane > i && lane < n {
            mat.read(sm_idx_packed::<T>(slot, i, lane)) * x.read(seg + lane as usize)
        } else {
            0.0f32
        };
        partial.write(seg + lane as usize, s);
        workgroup_memory_barrier_with_group_sync();
        for r in 0..log2_t {
            let stride = T >> (r + 1);
            if lane < stride {
                let v = partial.read(seg + lane as usize)
                    + partial.read(seg + (lane + stride) as usize);
                partial.write(seg + lane as usize, v);
            }
            workgroup_memory_barrier_with_group_sync();
        }
        if active && lane == 0 {
            let u = mat.read(sm_idx_packed::<T>(slot, i, i));
            let cur = x.read(seg + i as usize) - partial.read(seg);
            x.write(seg + i as usize, if u != 0.0 { cur / u } else { 0.0 });
        }
        workgroup_memory_barrier_with_group_sync();
    }
}

/// Packed [`lu_apply_pivots`]: sequential on each slot's lane 0.
#[inline]
pub(super) fn lu_apply_pivots_packed<const T: u32>(
    n: u32,
    slot: u32,
    lane: u32,
    active_slot: bool,
    buf_pivots: &[u32],
    piv: VSlice,
    x: &mut [f32; 64],
) {
    let seg = (slot * T) as usize;
    if active_slot && lane == 0 {
        for k in 0..n {
            let p = buf_pivots.read(piv.at(k));
            if p != k {
                let a = x.read(seg + k as usize);
                let b = x.read(seg + p as usize);
                x.write(seg + k as usize, b);
                x.write(seg + p as usize, a);
            }
        }
    }
    workgroup_memory_barrier_with_group_sync();
}

/// Apply the recorded pivots (sequential — lane 0 only). `n` is small so the
/// extra parallelism wouldn't pay for the barrier.
#[inline]
pub(super) fn lu_apply_pivots(
    n: u32,
    lane: u32,
    buf_pivots: &[u32],
    piv: VSlice,
    x: &mut [f32; MAX_MB_DOFS],
) {
    if lane == 0 {
        for k in 0..n {
            let p = buf_pivots.read(piv.at(k));
            if p != k {
                let a = x.read(k as usize);
                let b = x.read(p as usize);
                x.write(k as usize, b);
                x.write(p as usize, a);
            }
        }
    }
    workgroup_memory_barrier_with_group_sync();
}
