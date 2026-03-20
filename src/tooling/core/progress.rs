//! Shared atomic progress state for intra-step/intra-build visibility.
//!
//! `StepProgress` uses atomics for lock-free, allocation-free updates.
//! The simulation thread writes phases; the TUI thread reads snapshots
//! on each ~33ms render frame.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Enumeration of all possible progress phases.
///
/// Build phases (1–6) are set by phasma before caustic starts stepping.
/// Step phases (10+) are set by integrators and `Simulation::step()`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepPhase {
    Idle = 0,
    // Build phases (set by phasma, not caustic)
    BuildDomain = 1,
    BuildIC = 2,
    BuildICSampling = 7,
    BuildICCompression = 8,
    BuildPoisson = 3,
    BuildIntegrator = 4,
    BuildExitConditions = 5,
    BuildAssembly = 6,
    // Strang sub-phases
    DriftHalf1 = 10,
    ComputeDensity = 11,
    PoissonSolve = 12,
    ComputeAcceleration = 13,
    Kick = 14,
    DriftHalf2 = 15,
    // Post-advance (Simulation::step)
    LoMaC = 16,
    PostDensity = 17,
    Diagnostics = 18,
    StepComplete = 19,
    // Yoshida (7 sub-steps)
    YoshidaDrift1 = 20,
    YoshidaKick1 = 21,
    YoshidaDrift2 = 22,
    YoshidaKick2 = 23,
    YoshidaDrift3 = 24,
    YoshidaKick3 = 25,
    YoshidaDrift4 = 26,
    // RKEI (3 stages)
    RkeiStage1 = 30,
    RkeiStage2 = 31,
    RkeiStage3 = 32,
    // Unsplit RK stages
    UnsplitStage1 = 40,
    UnsplitStage2 = 41,
    UnsplitStage3 = 42,
    UnsplitStage4 = 43,
}

impl StepPhase {
    /// Convert from raw u8, returning `Idle` for unknown values.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Idle,
            1 => Self::BuildDomain,
            2 => Self::BuildIC,
            3 => Self::BuildPoisson,
            4 => Self::BuildIntegrator,
            5 => Self::BuildExitConditions,
            6 => Self::BuildAssembly,
            7 => Self::BuildICSampling,
            8 => Self::BuildICCompression,
            10 => Self::DriftHalf1,
            11 => Self::ComputeDensity,
            12 => Self::PoissonSolve,
            13 => Self::ComputeAcceleration,
            14 => Self::Kick,
            15 => Self::DriftHalf2,
            16 => Self::LoMaC,
            17 => Self::PostDensity,
            18 => Self::Diagnostics,
            19 => Self::StepComplete,
            20 => Self::YoshidaDrift1,
            21 => Self::YoshidaKick1,
            22 => Self::YoshidaDrift2,
            23 => Self::YoshidaKick2,
            24 => Self::YoshidaDrift3,
            25 => Self::YoshidaKick3,
            26 => Self::YoshidaDrift4,
            30 => Self::RkeiStage1,
            31 => Self::RkeiStage2,
            32 => Self::RkeiStage3,
            40 => Self::UnsplitStage1,
            41 => Self::UnsplitStage2,
            42 => Self::UnsplitStage3,
            43 => Self::UnsplitStage4,
            _ => Self::Idle,
        }
    }

    /// Human-readable label for display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::BuildDomain => "Domain",
            Self::BuildIC => "IC generation",
            Self::BuildICSampling => "IC sampling",
            Self::BuildICCompression => "IC compression",
            Self::BuildPoisson => "Poisson solver",
            Self::BuildIntegrator => "Integrator",
            Self::BuildExitConditions => "Exit conditions",
            Self::BuildAssembly => "Assembly",
            Self::DriftHalf1 => "Drift ½",
            Self::ComputeDensity => "Density",
            Self::PoissonSolve => "Poisson solve",
            Self::ComputeAcceleration => "Acceleration",
            Self::Kick => "Kick",
            Self::DriftHalf2 => "Drift ½",
            Self::LoMaC => "LoMaC",
            Self::PostDensity => "Post density",
            Self::Diagnostics => "Diagnostics",
            Self::StepComplete => "Complete",
            Self::YoshidaDrift1 => "Drift 1",
            Self::YoshidaKick1 => "Kick 1",
            Self::YoshidaDrift2 => "Drift 2",
            Self::YoshidaKick2 => "Kick 2",
            Self::YoshidaDrift3 => "Drift 3",
            Self::YoshidaKick3 => "Kick 3",
            Self::YoshidaDrift4 => "Drift 4",
            Self::RkeiStage1 => "Stage 1",
            Self::RkeiStage2 => "Stage 2",
            Self::RkeiStage3 => "Stage 3",
            Self::UnsplitStage1 => "Stage 1",
            Self::UnsplitStage2 => "Stage 2",
            Self::UnsplitStage3 => "Stage 3",
            Self::UnsplitStage4 => "Stage 4",
        }
    }

    /// Whether this is a build phase (< 10).
    pub fn is_build(self) -> bool {
        (self as u8) >= 1 && (self as u8) <= 6
    }
}

/// Shared atomic progress state. Written by the sim thread, read by the TUI.
pub struct StepProgress {
    phase: AtomicU8,
    sub_step: AtomicU8,
    sub_step_total: AtomicU8,
    sub_done: AtomicU64,
    sub_total: AtomicU64,
    step_start_ns: AtomicU64,
}

/// A consistent snapshot of progress at one instant.
pub struct ProgressSnapshot {
    pub phase: StepPhase,
    pub sub_step: u8,
    pub sub_step_total: u8,
    pub sub_done: u64,
    pub sub_total: u64,
    pub elapsed_ms: f64,
}

impl StepProgress {
    /// Create a new `StepProgress` wrapped in `Arc` for sharing.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            phase: AtomicU8::new(0),
            sub_step: AtomicU8::new(0),
            sub_step_total: AtomicU8::new(0),
            sub_done: AtomicU64::new(0),
            sub_total: AtomicU64::new(0),
            step_start_ns: AtomicU64::new(now_ns()),
        })
    }

    /// Set the current phase.
    pub fn set_phase(&self, phase: StepPhase) {
        self.phase.store(phase as u8, Ordering::Relaxed);
        // Reset intra-phase counters
        self.sub_done.store(0, Ordering::Relaxed);
        self.sub_total.store(0, Ordering::Relaxed);
    }

    /// Set sub-step index and total (e.g. 2 of 5 for Strang).
    pub fn set_sub_step(&self, current: u8, total: u8) {
        self.sub_step.store(current, Ordering::Relaxed);
        self.sub_step_total.store(total, Ordering::Relaxed);
    }

    /// Set intra-phase work units (e.g. grid cells processed).
    pub fn set_intra_progress(&self, done: u64, total: u64) {
        self.sub_done.store(done, Ordering::Relaxed);
        self.sub_total.store(total, Ordering::Relaxed);
    }

    /// Mark the start of a new step (resets timer).
    pub fn start_step(&self) {
        self.step_start_ns.store(now_ns(), Ordering::Relaxed);
        self.sub_step.store(0, Ordering::Relaxed);
        self.sub_step_total.store(0, Ordering::Relaxed);
        self.sub_done.store(0, Ordering::Relaxed);
        self.sub_total.store(0, Ordering::Relaxed);
    }

    /// Read a consistent-enough snapshot (all Relaxed loads).
    pub fn read(&self) -> ProgressSnapshot {
        let start = self.step_start_ns.load(Ordering::Relaxed);
        let now = now_ns();
        let elapsed_ns = now.saturating_sub(start);
        ProgressSnapshot {
            phase: StepPhase::from_u8(self.phase.load(Ordering::Relaxed)),
            sub_step: self.sub_step.load(Ordering::Relaxed),
            sub_step_total: self.sub_step_total.load(Ordering::Relaxed),
            sub_done: self.sub_done.load(Ordering::Relaxed),
            sub_total: self.sub_total.load(Ordering::Relaxed),
            elapsed_ms: elapsed_ns as f64 / 1_000_000.0,
        }
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
