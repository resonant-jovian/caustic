//! MPI domain decomposition for distributed Vlasov–Poisson simulations.
//!
//! Partitions the spatial domain across MPI ranks while replicating velocity-space
//! data. Communication patterns:
//! - All-reduce for density moments (Poisson solve)
//! - Ghost-zone exchange for spatial advection stencils
//!
//! Requires the `mpi` feature and a working MPI installation (OpenMPI or MPICH).
//!
//! # Architecture
//!
//! The HT tensor structure naturally supports spatial domain decomposition:
//! spatial leaf factors are distributed (each rank owns N/P rows), while
//! velocity leaf factors are replicated across all ranks. Transfer tensors
//! are small and replicated.
//!
//! Reference: caustic.md §7 (MPI 3.1 bindings via the `mpi` crate).

use mpi::traits::*;

/// Spatial domain decomposition across MPI ranks.
///
/// Each rank owns a contiguous slab of the spatial grid along the x₁ axis.
/// Velocity dimensions are fully replicated on every rank.
pub struct DomainDecomposition {
    /// This rank's index.
    pub rank: usize,
    /// Total number of ranks.
    pub size: usize,
    /// Global spatial shape [nx, ny, nz].
    pub global_shape: [usize; 3],
    /// Local x₁ range: [start, end) indices into the global grid.
    pub local_x1_range: (usize, usize),
    /// Number of ghost zones on each side for stencil operations.
    pub ghost_width: usize,
}

impl DomainDecomposition {
    /// Create a slab decomposition along x₁ from an already-initialized MPI universe.
    ///
    /// The x₁ axis is split evenly across ranks. Remainder cells go to the last rank.
    pub fn new(global_shape: [usize; 3], ghost_width: usize, rank: usize, size: usize) -> Self {
        let nx = global_shape[0];
        let chunk = nx / size;
        let start = rank * chunk;
        let end = if rank == size - 1 { nx } else { start + chunk };

        Self {
            rank,
            size,
            global_shape,
            local_x1_range: (start, end),
            ghost_width,
        }
    }

    /// Number of x₁ cells owned by this rank (excluding ghost zones).
    pub fn local_nx1(&self) -> usize {
        self.local_x1_range.1 - self.local_x1_range.0
    }
}

/// MPI context wrapper for use by the simulation.
///
/// Owns the MPI universe and provides domain decomposition and
/// collective communication routines.
pub struct MpiContext {
    pub decomposition: DomainDecomposition,
}

impl MpiContext {
    /// Initialize MPI and create a slab decomposition.
    pub fn new(global_shape: [usize; 3], ghost_width: usize) -> Self {
        let universe = mpi::initialize().expect("MPI initialization failed");
        let world = universe.world();
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        Self {
            decomposition: DomainDecomposition::new(global_shape, ghost_width, rank, size),
        }
    }

    pub fn is_root(&self) -> bool {
        self.decomposition.rank == 0
    }
}
