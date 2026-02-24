//! `IOManager` — handles HDF5 snapshot I/O and checkpoint/restart.
//! Requires the `hdf5` feature flag for full HDF5 support.

use super::types::{PhaseSpaceSnapshot};
use super::diagnostics::{Diagnostics, GlobalDiagnostics};

/// Snapshot serialisation format.
pub enum OutputFormat {
    Hdf5,
    Npy,
    Binary,
}

/// Manages all file I/O: snapshots, diagnostics, and checkpoints.
pub struct IOManager {
    pub output_dir: std::path::PathBuf,
    pub format: OutputFormat,
}

impl IOManager {
    pub fn new(output_dir: &str, format: OutputFormat) -> Self {
        todo!()
    }

    /// Save a full `PhaseSpaceSnapshot` to disk at the given path.
    pub fn save_snapshot(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        filename: &str,
    ) -> anyhow::Result<()> {
        todo!("write 6D array + metadata via HDF5/NPY")
    }

    /// Load a snapshot from disk and reconstruct. Used for restarts.
    pub fn load_snapshot(&self, filename: &str) -> anyhow::Result<PhaseSpaceSnapshot> {
        todo!()
    }

    /// Append one row to the running diagnostics CSV / HDF5 dataset.
    pub fn append_diagnostics(&self, row: &GlobalDiagnostics) -> anyhow::Result<()> {
        todo!()
    }

    /// Save a full checkpoint (snapshot + diagnostics history + config).
    /// Used for wall-clock restarts.
    pub fn save_checkpoint(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        diagnostics: &Diagnostics,
    ) -> anyhow::Result<()> {
        todo!()
    }
}
