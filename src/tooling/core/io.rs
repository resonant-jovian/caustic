//! Snapshot I/O, checkpoint/restart, and diagnostics output.
//!
//! [`IOManager`] is the single entry point for all file I/O in the solver.
//! It writes and reads phase-space snapshots (binary or HDF5), appends
//! per-step diagnostics rows to a CSV, and saves/loads Hierarchical Tucker
//! checkpoints so that a simulation can be restarted without expanding the
//! full N^6 dense array.
//!
//! All paths are resolved relative to the configured `output_dir`.

use super::diagnostics::{Diagnostics, GlobalDiagnostics};
use super::types::PhaseSpaceSnapshot;

/// Serializable HT tensor checkpoint: stores the tree structure for
/// save/restore without expanding to the full N⁶ dense array.
///
/// Format: binary header + leaf frames + transfer tensors.
/// Each node stores its type, dimensions, and flattened f64 data.
#[derive(Clone)]
pub struct HtCheckpoint {
    /// Shape of the 6D grid [n1, n2, n3, n4, n5, n6].
    pub shape: [usize; 6],
    /// Ranks at each of the 11 HT nodes.
    pub ranks: Vec<usize>,
    /// Leaf frame data: 6 entries, each a flattened column-major matrix.
    pub leaf_frames: Vec<Vec<f64>>,
    /// Transfer tensor data: 5 entries (interior nodes), each flattened.
    pub transfer_tensors: Vec<Vec<f64>>,
    /// Simulation time at checkpoint.
    pub time: f64,
    /// Truncation tolerance used.
    pub tolerance: f64,
    /// Maximum rank setting.
    pub max_rank: usize,
}

/// Snapshot serialisation format selector.
pub enum OutputFormat {
    /// HDF5 (requires the `hdf5` feature). Supports yt-compatible layout.
    Hdf5,
    /// NumPy `.npy` — flat little-endian f64 with a standard NumPy header.
    Npy,
    /// Raw little-endian binary: 6 x u64 shape + f64 time + f64 data.
    Binary,
}

/// Manages all file I/O: snapshots, diagnostics, and checkpoints.
///
/// All files are written into (or read from) [`output_dir`](Self::output_dir).
/// The directory is created on demand by each write method.
pub struct IOManager {
    /// Base directory for all output files.
    pub output_dir: std::path::PathBuf,
    /// Serialisation format used for snapshot writes.
    pub format: OutputFormat,
}

impl IOManager {
    /// Create a new I/O manager writing to `output_dir` in the given format.
    pub fn new(output_dir: &str, format: OutputFormat) -> Self {
        Self {
            output_dir: std::path::PathBuf::from(output_dir),
            format,
        }
    }

    /// Save a full [`PhaseSpaceSnapshot`] to disk as raw little-endian binary.
    pub fn save_snapshot(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        filename: &str,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join(filename);
        let mut file = std::fs::File::create(path)?;
        // Simple binary format: 6 u64 shape values + f64 time + f64 data
        for &s in &snapshot.shape {
            file.write_all(&(s as u64).to_le_bytes())?;
        }
        file.write_all(&snapshot.time.to_le_bytes())?;
        for &v in &snapshot.data {
            file.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load a snapshot from disk and reconstruct a [`PhaseSpaceSnapshot`]. Used for restarts.
    pub fn load_snapshot(&self, filename: &str) -> anyhow::Result<PhaseSpaceSnapshot> {
        use std::io::Read;
        let path = self.output_dir.join(filename);
        let mut file = std::fs::File::open(path)?;

        // Read 6 u64 shape values
        let mut shape = [0usize; 6];
        for s in shape.iter_mut() {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            *s = u64::from_le_bytes(buf) as usize;
        }

        // Read f64 time
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        let time = f64::from_le_bytes(buf);

        // Read f64 data
        let n: usize = shape.iter().product();
        let mut data = vec![0.0f64; n];
        for v in data.iter_mut() {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            *v = f64::from_le_bytes(buf);
        }

        Ok(PhaseSpaceSnapshot { data, shape, time })
    }

    /// Append one row to the running `diagnostics.csv` file (creates header on first call).
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
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            row.time,
            row.total_energy,
            row.kinetic_energy,
            row.potential_energy,
            row.virial_ratio,
            row.casimir_c2,
            row.entropy,
            row.mass_in_box
        )?;
        Ok(())
    }

    /// Save an HT tensor checkpoint (tree structure without dense expansion).
    ///
    /// Binary format:
    /// - 6 × u64: shape
    /// - u64: time (as f64 bits)
    /// - u64: tolerance (as f64 bits)
    /// - u64: max_rank
    /// - u64: n_nodes (11 for 6D HT)
    /// - For each node: u64 rank
    /// - u64: n_leaves (6)
    /// - For each leaf: u64 len, then len × f64 data
    /// - u64: n_transfers (5)
    /// - For each transfer: u64 len, then len × f64 data
    pub fn save_ht_checkpoint(&self, ht: &HtCheckpoint, filename: &str) -> anyhow::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join(filename);
        let mut file = std::fs::File::create(path)?;

        // Shape
        for &s in &ht.shape {
            file.write_all(&(s as u64).to_le_bytes())?;
        }
        // Metadata
        file.write_all(&ht.time.to_le_bytes())?;
        file.write_all(&ht.tolerance.to_le_bytes())?;
        file.write_all(&(ht.max_rank as u64).to_le_bytes())?;

        // Ranks
        file.write_all(&(ht.ranks.len() as u64).to_le_bytes())?;
        for &r in &ht.ranks {
            file.write_all(&(r as u64).to_le_bytes())?;
        }

        // Leaf frames
        file.write_all(&(ht.leaf_frames.len() as u64).to_le_bytes())?;
        for frame in &ht.leaf_frames {
            file.write_all(&(frame.len() as u64).to_le_bytes())?;
            for &v in frame {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        // Transfer tensors
        file.write_all(&(ht.transfer_tensors.len() as u64).to_le_bytes())?;
        for tensor in &ht.transfer_tensors {
            file.write_all(&(tensor.len() as u64).to_le_bytes())?;
            for &v in tensor {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load an HT tensor checkpoint from disk.
    pub fn load_ht_checkpoint(&self, filename: &str) -> anyhow::Result<HtCheckpoint> {
        use std::io::Read;
        let path = self.output_dir.join(filename);
        let mut file = std::fs::File::open(path)?;

        let read_u64 = |f: &mut std::fs::File| -> anyhow::Result<u64> {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        };
        let read_f64 = |f: &mut std::fs::File| -> anyhow::Result<f64> {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)?;
            Ok(f64::from_le_bytes(buf))
        };

        // Shape
        let mut shape = [0usize; 6];
        for s in shape.iter_mut() {
            *s = read_u64(&mut file)? as usize;
        }

        // Metadata
        let time = read_f64(&mut file)?;
        let tolerance = read_f64(&mut file)?;
        let max_rank = read_u64(&mut file)? as usize;

        // Ranks
        let n_nodes = read_u64(&mut file)? as usize;
        let mut ranks = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            ranks.push(read_u64(&mut file)? as usize);
        }

        // Leaf frames
        let n_leaves = read_u64(&mut file)? as usize;
        let mut leaf_frames = Vec::with_capacity(n_leaves);
        for _ in 0..n_leaves {
            let len = read_u64(&mut file)? as usize;
            let mut data = vec![0.0f64; len];
            for v in data.iter_mut() {
                *v = read_f64(&mut file)?;
            }
            leaf_frames.push(data);
        }

        // Transfer tensors
        let n_transfers = read_u64(&mut file)? as usize;
        let mut transfer_tensors = Vec::with_capacity(n_transfers);
        for _ in 0..n_transfers {
            let len = read_u64(&mut file)? as usize;
            let mut data = vec![0.0f64; len];
            for v in data.iter_mut() {
                *v = read_f64(&mut file)?;
            }
            transfer_tensors.push(data);
        }

        Ok(HtCheckpoint {
            shape,
            ranks,
            leaf_frames,
            transfer_tensors,
            time,
            tolerance,
            max_rank,
        })
    }

    /// Save a snapshot in HDF5 format. Requires the `hdf5` feature.
    ///
    /// HDF5 layout:
    /// - `/shape` — 6-element u64 dataset
    /// - `/time` — scalar f64 attribute
    /// - `/data` — flat f64 dataset of length ∏shape
    #[cfg(feature = "hdf5")]
    pub fn save_snapshot_hdf5(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        filename: &str,
    ) -> anyhow::Result<()> {
        use hdf5_metno as hdf5;

        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join(filename);
        let file = hdf5::File::create(path)?;

        // Write shape
        let shape_u64: Vec<u64> = snapshot.shape.iter().map(|&s| s as u64).collect();
        file.new_dataset::<u64>()
            .shape([6])
            .create("shape")?
            .write(&shape_u64)?;

        // Write time as attribute on root
        file.new_attr::<f64>()
            .create("time")?
            .write_scalar(&snapshot.time)?;

        // Write data
        file.new_dataset::<f64>()
            .shape([snapshot.data.len()])
            .create("data")?
            .write(&snapshot.data)?;

        Ok(())
    }

    /// Load a snapshot from HDF5 format. Requires the `hdf5` feature.
    #[cfg(feature = "hdf5")]
    pub fn load_snapshot_hdf5(&self, filename: &str) -> anyhow::Result<PhaseSpaceSnapshot> {
        use hdf5_metno as hdf5;

        let path = self.output_dir.join(filename);
        let file = hdf5::File::open(path)?;

        let shape_data: Vec<u64> = file.dataset("shape")?.read_raw()?;
        let mut shape = [0usize; 6];
        for (i, &s) in shape_data.iter().enumerate() {
            shape[i] = s as usize;
        }

        let time: f64 = file.attr("time")?.read_scalar()?;
        let data: Vec<f64> = file.dataset("data")?.read_raw()?;

        Ok(PhaseSpaceSnapshot { data, shape, time })
    }

    /// Save an HT checkpoint in HDF5 format. Requires the `hdf5` feature.
    ///
    /// Stores the tree structure (leaf frames + transfer tensors) as separate
    /// datasets within a group hierarchy, following caustic.md Section 7.
    #[cfg(feature = "hdf5")]
    pub fn save_ht_checkpoint_hdf5(&self, ht: &HtCheckpoint, filename: &str) -> anyhow::Result<()> {
        use hdf5_metno as hdf5;

        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join(filename);
        let file = hdf5::File::create(path)?;

        // Metadata
        let shape_u64: Vec<u64> = ht.shape.iter().map(|&s| s as u64).collect();
        file.new_dataset::<u64>()
            .shape([6])
            .create("shape")?
            .write(&shape_u64)?;
        file.new_attr::<f64>()
            .create("time")?
            .write_scalar(&ht.time)?;
        file.new_attr::<f64>()
            .create("tolerance")?
            .write_scalar(&ht.tolerance)?;
        file.new_attr::<u64>()
            .create("max_rank")?
            .write_scalar(&(ht.max_rank as u64))?;

        // Ranks
        let ranks_u64: Vec<u64> = ht.ranks.iter().map(|&r| r as u64).collect();
        file.new_dataset::<u64>()
            .shape([ranks_u64.len()])
            .create("ranks")?
            .write(&ranks_u64)?;

        // Leaf frames group
        let leaves = file.create_group("leaf_frames")?;
        for (i, frame) in ht.leaf_frames.iter().enumerate() {
            leaves
                .new_dataset::<f64>()
                .shape([frame.len()])
                .create(format!("leaf_{i}").as_str())?
                .write(frame)?;
        }

        // Transfer tensors group
        let transfers = file.create_group("transfer_tensors")?;
        for (i, tensor) in ht.transfer_tensors.iter().enumerate() {
            transfers
                .new_dataset::<f64>()
                .shape([tensor.len()])
                .create(format!("transfer_{i}").as_str())?
                .write(tensor)?;
        }

        Ok(())
    }

    /// Load an HT checkpoint from HDF5 format. Requires the `hdf5` feature.
    #[cfg(feature = "hdf5")]
    pub fn load_ht_checkpoint_hdf5(&self, filename: &str) -> anyhow::Result<HtCheckpoint> {
        use hdf5_metno as hdf5;

        let path = self.output_dir.join(filename);
        let file = hdf5::File::open(path)?;

        let shape_data: Vec<u64> = file.dataset("shape")?.read_raw()?;
        let mut shape = [0usize; 6];
        for (i, &s) in shape_data.iter().enumerate() {
            shape[i] = s as usize;
        }

        let time: f64 = file.attr("time")?.read_scalar()?;
        let tolerance: f64 = file.attr("tolerance")?.read_scalar()?;
        let max_rank: u64 = file.attr("max_rank")?.read_scalar()?;

        let ranks_data: Vec<u64> = file.dataset("ranks")?.read_raw()?;
        let ranks: Vec<usize> = ranks_data.iter().map(|&r| r as usize).collect();

        let leaves_group = file.group("leaf_frames")?;
        let n_leaves = leaves_group.len();
        let mut leaf_frames = Vec::with_capacity(n_leaves as usize);
        for i in 0..n_leaves {
            let data: Vec<f64> = leaves_group
                .dataset(format!("leaf_{i}").as_str())?
                .read_raw()?;
            leaf_frames.push(data);
        }

        let transfers_group = file.group("transfer_tensors")?;
        let n_transfers = transfers_group.len();
        let mut transfer_tensors = Vec::with_capacity(n_transfers as usize);
        for i in 0..n_transfers {
            let data: Vec<f64> = transfers_group
                .dataset(format!("transfer_{i}").as_str())?
                .read_raw()?;
            transfer_tensors.push(data);
        }

        Ok(HtCheckpoint {
            shape,
            ranks,
            leaf_frames,
            transfer_tensors,
            time,
            tolerance,
            max_rank: max_rank as usize,
        })
    }

    /// Save a snapshot in yt-compatible HDF5 format.
    ///
    /// Writes HDF5 with groups following yt's generic format:
    /// - /simulation_parameters: time, domain edges, dimensions
    /// - /field_data/density: 3D density field
    /// - /field_data/potential: 3D potential field (if provided)
    /// - /units: length, time, mass, velocity units
    #[cfg(feature = "hdf5")]
    pub fn save_snapshot_yt(
        &self,
        density: &super::types::DensityField,
        potential: Option<&super::types::PotentialField>,
        time: f64,
        g: f64,
        filename: &str,
    ) -> anyhow::Result<()> {
        use hdf5_metno as hdf5;

        std::fs::create_dir_all(&self.output_dir)?;
        let path = self.output_dir.join(filename);
        let file = hdf5::File::create(path)?;
        let [nx, ny, nz] = density.shape;

        // Simulation parameters
        let params = file.create_group("simulation_parameters")?;
        params
            .new_attr::<f64>()
            .create("time")?
            .write_scalar(&time)?;
        params
            .new_attr::<[f64; 3]>()
            .create("domain_left_edge")?
            .write_scalar(&[-1.0, -1.0, -1.0])?;
        params
            .new_attr::<[f64; 3]>()
            .create("domain_right_edge")?
            .write_scalar(&[1.0, 1.0, 1.0])?;
        params
            .new_attr::<[u64; 3]>()
            .create("domain_dimensions")?
            .write_scalar(&[nx as u64, ny as u64, nz as u64])?;
        params
            .new_attr::<i32>()
            .create("cosmological_simulation")?
            .write_scalar(&0)?;
        params
            .new_attr::<f64>()
            .create("gravitational_constant")?
            .write_scalar(&g)?;

        // Field data
        let fields = file.create_group("field_data")?;
        fields
            .new_dataset::<f64>()
            .shape([nx, ny, nz])
            .create("density")?
            .write_raw(&density.data)?;

        if let Some(pot) = potential {
            fields
                .new_dataset::<f64>()
                .shape([nx, ny, nz])
                .create("potential")?
                .write_raw(&pot.data)?;
        }

        // Units (dimensionless by default)
        let units = file.create_group("units")?;
        units
            .new_attr::<f64>()
            .create("length_unit")?
            .write_scalar(&1.0)?;
        units
            .new_attr::<f64>()
            .create("time_unit")?
            .write_scalar(&1.0)?;
        units
            .new_attr::<f64>()
            .create("mass_unit")?
            .write_scalar(&1.0)?;
        units
            .new_attr::<f64>()
            .create("velocity_unit")?
            .write_scalar(&1.0)?;

        Ok(())
    }

    /// Save a full checkpoint (snapshot binary + diagnostics history as JSON).
    pub fn save_checkpoint(
        &self,
        snapshot: &PhaseSpaceSnapshot,
        diagnostics: &Diagnostics,
    ) -> anyhow::Result<()> {
        use std::io::Write;

        // Save the snapshot binary
        self.save_snapshot(snapshot, "checkpoint.bin")?;

        // Save diagnostics history as JSON
        let path = self.output_dir.join("checkpoint_diagnostics.json");
        let json = serde_json::to_string_pretty(&diagnostics.history)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_binary() {
        let tmp = std::env::temp_dir().join("caustic_io_test_roundtrip");
        let _ = std::fs::remove_dir_all(&tmp);
        let io = IOManager::new(tmp.to_str().unwrap(), OutputFormat::Binary);
        let snap = PhaseSpaceSnapshot {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: [2, 2, 2, 1, 1, 1],
            time: 0.5,
        };
        io.save_snapshot(&snap, "test.bin").unwrap();
        let loaded = io.load_snapshot("test.bin").unwrap();
        assert_eq!(loaded.shape, snap.shape);
        assert!((loaded.time - 0.5).abs() < 1e-14);
        assert_eq!(loaded.data.len(), snap.data.len());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn save_snapshot_yt_creates_groups() {
        use super::super::types::{DensityField, PotentialField};
        use hdf5_metno as hdf5;

        let tmp = std::env::temp_dir().join("caustic_io_test_yt");
        let _ = std::fs::remove_dir_all(&tmp);
        let io = IOManager::new(tmp.to_str().unwrap(), OutputFormat::Hdf5);

        let nx = 4;
        let density = DensityField {
            data: vec![1.0; nx * nx * nx],
            shape: [nx, nx, nx],
        };
        let potential = PotentialField {
            data: vec![-0.5; nx * nx * nx],
            shape: [nx, nx, nx],
        };

        io.save_snapshot_yt(&density, Some(&potential), 1.5, 1.0, "snap_yt.h5")
            .unwrap();

        let path = tmp.join("snap_yt.h5");
        assert!(path.exists(), "yt HDF5 file should exist");

        // Re-open and verify structure
        let file = hdf5::File::open(&path).unwrap();

        // Check groups exist
        assert!(file.group("simulation_parameters").is_ok());
        assert!(file.group("field_data").is_ok());
        assert!(file.group("units").is_ok());

        // Check simulation_parameters attributes
        let params = file.group("simulation_parameters").unwrap();
        let time: f64 = params.attr("time").unwrap().read_scalar().unwrap();
        assert!((time - 1.5).abs() < 1e-14, "time should be 1.5");

        let g_val: f64 = params
            .attr("gravitational_constant")
            .unwrap()
            .read_scalar()
            .unwrap();
        assert!((g_val - 1.0).abs() < 1e-14, "G should be 1.0");

        // Check field datasets
        let fields = file.group("field_data").unwrap();
        assert!(fields.dataset("density").is_ok());
        assert!(fields.dataset("potential").is_ok());

        // Check units attributes
        let units = file.group("units").unwrap();
        let length: f64 = units.attr("length_unit").unwrap().read_scalar().unwrap();
        assert!((length - 1.0).abs() < 1e-14);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
