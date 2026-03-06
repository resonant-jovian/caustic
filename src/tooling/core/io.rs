//! `IOManager` — handles HDF5 snapshot I/O and checkpoint/restart.
//! Requires the `hdf5` feature flag for full HDF5 support.

use super::types::PhaseSpaceSnapshot;
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
        Self {
            output_dir: std::path::PathBuf::from(output_dir),
            format,
        }
    }

    /// Save a full `PhaseSpaceSnapshot` to disk at the given path.
    pub fn save_snapshot(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        filename: &str,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join(filename);
        let mut file = std::fs::File::create(path)?;
        // Simple binary format: 6 u64 shape values + f64 data
        for &s in &snapshot.shape {
            file.write_all(&(s as u64).to_le_bytes())?;
        }
        file.write_all(&snapshot.time.to_le_bytes())?;
        for &v in &snapshot.data {
            file.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load a snapshot from disk and reconstruct. Used for restarts.
    pub fn load_snapshot(&self, filename: &str) -> anyhow::Result<PhaseSpaceSnapshot> {
        todo!("load snapshot from binary file")
    }

    /// Append one row to the running diagnostics CSV.
    pub fn append_diagnostics(&self, row: &GlobalDiagnostics) -> anyhow::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join("diagnostics.csv");
        let file_exists = path.exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        if !file_exists {
            writeln!(file, "t,E,T,W,vir,C2,S,M")?;
        }
        writeln!(file, "{},{},{},{},{},{},{},{}",
            row.time, row.total_energy, row.kinetic_energy, row.potential_energy,
            row.virial_ratio, row.casimir_c2, row.entropy, row.mass_in_box
        )?;
        Ok(())
    }

    /// Save a full checkpoint (snapshot + diagnostics history + config).
    pub fn save_checkpoint(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        diagnostics: &Diagnostics,
    ) -> anyhow::Result<()> {
        todo!("save checkpoint for wall-clock restart")
    }
}
