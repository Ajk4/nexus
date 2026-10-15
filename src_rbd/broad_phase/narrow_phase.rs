//! Narrow-phase collision detection: generates contact manifolds from broad-phase pairs.

use crate::math::Pose;
use crate::queries::GpuIndexedContact;
use crate::shaders::PaddedVector;
#[cfg(feature = "dim3")]
use crate::shaders::broad_phase::GpuReduceContacts;
use crate::shaders::broad_phase::{
    CollisionPair, ContactPlan, GpuContactPlan, GpuNarrowPhasePfmPfm, GpuNarrowPhaseShapeShape,
    GpuNarrowPhaseShapeShapeDeferred, GpuPfmSortKeys, GpuResetNarrowPhase, NarrowPhasePfmPair,
};
use crate::shaders::shapes::Shape;
use crate::utils::{RadixSort, RadixSortWorkspace};
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

#[cfg(all(test, feature = "dim2", feature = "cpu"))]
mod tests {
    use super::*;
    use crate::math::Vec2;
    use crate::pipeline::{RbdCapacities, RbdPipeline, RbdResizePolicy, RbdState};
    use crate::rapier::prelude::*;
    use crate::shaders::dynamics::RbdSimParams;

    #[test]
    fn polyline_contacts_use_subshape_capacity() {
        let backend = GpuBackend::Cpu;
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let wall = bodies.insert(RigidBodyBuilder::fixed());
        colliders.insert_with_parent(
            ColliderBuilder::polyline(
                (-3..=3).map(|i| Vec2::new(i as f32 * 0.1, 0.)).collect(),
                None,
            ),
            wall,
            &mut bodies,
        );
        let body = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec2::new(0., 0.04)));
        colliders.insert_with_parent(ColliderBuilder::capsule_x(0.25, 0.05), body, &mut bodies);
        bodies[body].recompute_mass_properties_from_colliders(&colliders);
        let joints = ImpulseJointSet::new();
        let multibodies = MultibodyJointSet::new();
        let params = RbdSimParams::default();
        let mut state = RbdState::from_rapier(
            &backend,
            &[(&bodies, &colliders, &joints, &multibodies, &params)],
            RbdCapacities {
                collisions_capacity: 1,
                subshape_pairs_per_collision: 8,
                collisions_resize_policy: RbdResizePolicy::Fixed,
                solver_colors_resize_policy: RbdResizePolicy::Fixed,
                ..Default::default()
            },
        );
        state.set_gravity(&backend, [0.; 3]);
        RbdPipeline::new(&backend)
            .unwrap()
            .step(&backend, &mut state, None)
            .unwrap();
        let contacts = state.contacts().buffer().unwrap_slice();
        let bound = state.contact_plan().buffer().unwrap_slice()[0].bound as usize;
        let mut left = false;
        let mut right = false;
        for manifold in &contacts[..bound] {
            for point in &manifold.contact.points_a[..manifold.contact.len as usize] {
                if point.dist <= 0. {
                    left |= point.pt.x < -0.1;
                    right |= point.pt.x > 0.1;
                }
            }
        }
        assert!(
            left && right,
            "capsule must contact both sides of the segmented wall"
        );
    }
}

/// Narrow-phase kernel bundle (see [`GpuNarrowPhase`]).
#[derive(Shader)]
struct GpuNarrowPhaseShaders {
    reset_narrow_phase: GpuResetNarrowPhase,
    narrow_phase: GpuNarrowPhaseShapeShape,
    /// Defers complex shape pairs (PFM / trimesh / polyline) into the
    /// `pfm_pairs` work-list. Split from `narrow_phase` to fit 8 storage buffers.
    narrow_phase_deferred: GpuNarrowPhaseShapeShapeDeferred,
    narrow_phase_pfm_pfm: GpuNarrowPhasePfmPfm,
    #[cfg(feature = "dim3")]
    reduce_contacts: GpuReduceContacts,
    /// Publishes the clamped list totals + every derived dispatch grid.
    contact_plan: GpuContactPlan,
    /// Extracts each PFM entry's pair index into the radix-sort key buffer.
    pfm_sort_keys: GpuPfmSortKeys,
}

/// GPU shader for narrow-phase collision detection.
pub struct GpuNarrowPhase {
    shaders: GpuNarrowPhaseShaders,
    /// Groups the PFM entries per originating pair (contact-reduction only).
    sort: RadixSort,
}

impl GpuNarrowPhase {
    /// Loads the narrow-phase kernels (and the radix sort they use) from the
    /// given backend.
    pub fn from_backend(backend: &GpuBackend) -> Result<Self, GpuBackendError> {
        Ok(Self {
            shaders: GpuNarrowPhaseShaders::from_backend(backend)?,
            sort: RadixSort::from_backend(backend)?,
        })
    }
}

/// Buffers backing the per-pair PFM sort of the contact-reduction path. All
/// sized by the PFM work-list capacity; `resize` must be called whenever that
/// capacity changes.
pub struct PfmSortState {
    /// Per-entry pair-index keys, written by `gpu_pfm_sort_keys`.
    keys: Tensor<u32>,
    /// The identity permutation `0..capacity`: the sort's value input, and
    /// the `pfm_order` indirection of the unsorted path.
    identity: Tensor<u32>,
    /// Sort outputs (pair-grouped keys + entry permutation).
    sorted_keys: Tensor<u32>,
    sorted_values: Tensor<u32>,
    /// Clamped PFM entry count (the sort's GPU-side `n_sort`), written by
    /// `gpu_contact_plan`.
    sort_len: Tensor<u32>,
    workspace: RadixSortWorkspace,
}

impl PfmSortState {
    /// Allocates the sort buffers for a PFM list of `capacity` entries.
    pub fn new(backend: &GpuBackend, capacity: u32) -> Self {
        let storage = BufferUsages::STORAGE;
        let identity: Vec<u32> = (0..capacity).collect();
        Self {
            keys: Tensor::vector_uninit(backend, capacity.max(1), storage).unwrap(),
            identity: Tensor::vector(backend, &identity, storage).unwrap(),
            sorted_keys: Tensor::vector_uninit(backend, capacity.max(1), storage).unwrap(),
            sorted_values: Tensor::vector_uninit(backend, capacity.max(1), storage).unwrap(),
            sort_len: Tensor::vector(backend, [0u32], storage).unwrap(),
            workspace: RadixSortWorkspace::new(backend),
        }
    }

    /// Regrow after a collision-pair capacity change.
    pub fn resize(&mut self, backend: &GpuBackend, capacity: u32) {
        *self = Self::new(backend, capacity);
    }

    /// The sorted-keys tensor consumed by `gpu_reduce_contacts`.
    #[cfg(feature = "dim3")]
    pub fn sorted_keys(&self) -> &Tensor<u32> {
        &self.sorted_keys
    }
}

impl GpuNarrowPhase {
    /// Dispatches the narrow-phase collision detection pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        backend: &GpuBackend,
        pass: &mut GpuPass,
        poses: &Tensor<Pose>,
        shapes: &Tensor<Shape>,
        vertices: &Tensor<PaddedVector>,
        indices: &Tensor<u32>,
        collision_pairs: &Tensor<CollisionPair>,
        collision_pairs_len: &mut Tensor<u32>,
        contacts: &mut Tensor<GpuIndexedContact>,
        contacts_indirect: &mut Tensor<[u32; 3]>,
        contact_plan: &mut Tensor<ContactPlan>,
        mb_sweep_indirect: &mut Tensor<[u32; 3]>,
        pfm_pairs: &mut Tensor<NarrowPhasePfmPair>,
        pfm_pairs_len: &mut Tensor<u32>,
        pfm_pairs_indirect: &mut Tensor<[u32; 3]>,
        pfm_sort: &mut PfmSortState,
        batch_indices: &Tensor<crate::shaders::utils::BatchIndices>,
        collider_parent: &Tensor<u32>,
        collider_materials: &Tensor<crate::shaders::queries::ColliderMaterial>,
        sim_params: &Tensor<crate::shaders::dynamics::RbdSimParams>,
        // Optional: merge each collider pair's manifolds into one before the
        // solvers see them. Enables the per-pair PFM sort (the manifolds of a
        // pair must land in contiguous slots for the per-run reduction).
        reduce_contacts: bool,
        // The `[total/64, 1, 1]` grid written by the broad phase from the
        // single global pair counter.
        collision_pairs_indirect: &Tensor<[u32; 3]>,
        // The flat pair/PFM buffer capacity, bounding the sort-key width.
        collision_pairs_capacity: u32,
    ) -> Result<(), GpuBackendError> {
        // The per-run reduction kernel is 3D-only; without it the sort would
        // group entries nobody consumes.
        let reduce_contacts = reduce_contacts && cfg!(feature = "dim3");

        self.shaders
            .reset_narrow_phase
            .call(pass, 1u32, pfm_pairs_len)?;

        // Defer the complex shape pairs into `pfm_pairs` FIRST: both list
        // counters must be final before the plan is published.
        self.shaders.narrow_phase_deferred.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            collision_pairs_len,
            poses,
            shapes,
            pfm_pairs,
            pfm_pairs_len,
            batch_indices,
            sim_params,
            vertices,
            indices,
        )?;

        // Clamped totals + every derived grid (contacts sweep, PFM sweep,
        // multibody contact sweep) in one serial thread.
        self.shaders.contact_plan.call(
            pass,
            1u32,
            collision_pairs_len,
            pfm_pairs_len,
            contact_plan,
            &mut pfm_sort.sort_len,
            contacts_indirect,
            pfm_pairs_indirect,
            mb_sweep_indirect,
            batch_indices,
        )?;

        // Analytic pairs: pair `t` writes contact slot `t` (inert on a miss).
        self.shaders.narrow_phase.call(
            pass,
            collision_pairs_indirect,
            collision_pairs,
            &*contact_plan,
            poses,
            shapes,
            contacts,
            collider_parent,
            collider_materials,
            sim_params,
        )?;

        // Contact reduction needs each pair's manifolds contiguous: group the
        // PFM entries per originating pair with a stable radix sort (which
        // also makes the PFM contact slots deterministic). Without reduction
        // the entries are consumed in emission order through the identity
        // permutation.
        if reduce_contacts {
            self.shaders.pfm_sort_keys.call(
                pass,
                &*pfm_pairs_indirect,
                pfm_pairs,
                &*contact_plan,
                &mut pfm_sort.keys,
            )?;
            // Keys are flat pair indices: bounded by the pair capacity.
            let sorting_bits =
                (32 - collision_pairs_capacity.saturating_sub(1).leading_zeros()).max(1);
            // Split borrows: the sort reads `keys`/`identity` and writes the
            // `sorted_*` pair.
            let PfmSortState {
                keys,
                identity,
                sorted_keys,
                sorted_values,
                sort_len,
                workspace,
            } = pfm_sort;
            self.sort.dispatch(
                backend,
                pass,
                workspace,
                keys,
                identity,
                sort_len,
                sorting_bits,
                1,
                sorted_keys,
                sorted_values,
            )?;
        }

        // PFM entry `i` (in sorted order when the sort ran) writes contact
        // slot `pairs_total + i` (inert on a miss).
        let pfm_order = if reduce_contacts {
            &pfm_sort.sorted_values
        } else {
            &pfm_sort.identity
        };
        self.shaders.narrow_phase_pfm_pfm.call(
            pass,
            &*pfm_pairs_indirect,
            contacts,
            pfm_pairs,
            pfm_order,
            &*contact_plan,
            vertices,
            indices,
            collider_parent,
            collider_materials,
            sim_params,
        )?;

        #[cfg(feature = "dim3")]
        if reduce_contacts {
            self.shaders.reduce_contacts.call(
                pass,
                &*pfm_pairs_indirect,
                contacts,
                &pfm_sort.sorted_keys,
                &*contact_plan,
                sim_params,
            )?;
        }

        Ok(())
    }
}
