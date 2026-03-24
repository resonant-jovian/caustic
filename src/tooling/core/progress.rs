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
/// Build phases (1–8) are set by phasma before caustic starts stepping.
/// Step phases (10+) are set by integrators and `Simulation::step()`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepPhase {
    /// No computation in progress; the simulation is idle.
    Idle = 0,
    // Build phases (set by phasma, not caustic)
    /// Constructing the computational domain (spatial/velocity extents, resolution, boundaries).
    BuildDomain = 1,
    /// Generating initial conditions (sampling the distribution function).
    BuildIC = 2,
    /// Sampling phase-space points for the initial distribution function.
    BuildICSampling = 7,
    /// Compressing the initial conditions into a low-rank representation (HT/TT).
    BuildICCompression = 8,
    /// Initializing the Poisson solver (FFT plans, Green's function, tree structure).
    BuildPoisson = 3,
    /// Constructing the time integrator (splitting scheme, RK tableaux).
    BuildIntegrator = 4,
    /// Registering exit/termination conditions (energy drift, wall clock, etc.).
    BuildExitConditions = 5,
    /// Final assembly of the simulation object from all constructed components.
    BuildAssembly = 6,
    // Strang sub-phases
    /// First half-step spatial drift in Strang splitting (advance x by v*dt/2).
    DriftHalf1 = 10,
    /// Computing the density field rho(x) by integrating f over velocity space.
    ComputeDensity = 11,
    /// Solving the Poisson equation nabla^2 Phi = 4*pi*G*rho for the gravitational potential.
    PoissonSolve = 12,
    /// Computing the acceleration field g = -nabla(Phi) from the potential.
    ComputeAcceleration = 13,
    /// Full velocity kick in Strang splitting (advance v by g*dt).
    Kick = 14,
    /// Second half-step spatial drift in Strang splitting (advance x by v*dt/2).
    DriftHalf2 = 15,
    // Post-advance (Simulation::step)
    /// Applying the LoMaC conservative projection to restore mass/momentum/energy conservation.
    LoMaC = 16,
    /// Recomputing the density field after the time step for diagnostics.
    PostDensity = 17,
    /// Computing global diagnostics (energy, momentum, Casimir invariants, etc.).
    Diagnostics = 18,
    /// The time step has completed; all sub-phases finished.
    StepComplete = 19,
    // Yoshida (7 sub-steps)
    /// Yoshida 4th-order splitting: first drift sub-step (c1*dt).
    YoshidaDrift1 = 20,
    /// Yoshida 4th-order splitting: first kick sub-step (d1*dt).
    YoshidaKick1 = 21,
    /// Yoshida 4th-order splitting: second drift sub-step (c2*dt).
    YoshidaDrift2 = 22,
    /// Yoshida 4th-order splitting: second kick sub-step (d2*dt).
    YoshidaKick2 = 23,
    /// Yoshida 4th-order splitting: third drift sub-step (c3*dt).
    YoshidaDrift3 = 24,
    /// Yoshida 4th-order splitting: third kick sub-step (d3*dt).
    YoshidaKick3 = 25,
    /// Yoshida 4th-order splitting: fourth (final) drift sub-step (c4*dt).
    YoshidaDrift4 = 26,
    // RKEI (3 stages)
    /// RKEI (SSP-RK3 exponential integrator): first stage evaluation.
    RkeiStage1 = 30,
    /// RKEI (SSP-RK3 exponential integrator): second stage evaluation.
    RkeiStage2 = 31,
    /// RKEI (SSP-RK3 exponential integrator): third stage evaluation and combination.
    RkeiStage3 = 32,
    // Unsplit RK stages
    /// Unsplit Runge-Kutta integrator: first stage (used by RK2, RK3, and RK4).
    UnsplitStage1 = 40,
    /// Unsplit Runge-Kutta integrator: second stage (used by RK2, RK3, and RK4).
    UnsplitStage2 = 41,
    /// Unsplit Runge-Kutta integrator: third stage (used by RK3 and RK4).
    UnsplitStage3 = 42,
    /// Unsplit Runge-Kutta integrator: fourth stage (used by RK4 only).
    UnsplitStage4 = 43,
    // Adaptive error estimation
    /// Adaptive time-stepping: computing the Lie (first-order) estimate for error control.
    AdaptiveLie = 44,
    /// Adaptive time-stepping: applying the velocity kick in the Lie estimate.
    AdaptiveLieKick = 45,
    /// Adaptive time-stepping: computing the error norm between splitting orders.
    AdaptiveError = 46,
    // BM4 (6-stage, 11 sub-steps)
    /// Blanes-Moan 4th-order splitting: first drift sub-step (a1*dt).
    Bm4Sub0 = 50,
    /// Blanes-Moan 4th-order splitting: first kick sub-step (b1*dt).
    Bm4Sub1 = 51,
    /// Blanes-Moan 4th-order splitting: second drift sub-step (a2*dt).
    Bm4Sub2 = 52,
    /// Blanes-Moan 4th-order splitting: second kick sub-step (b2*dt).
    Bm4Sub3 = 53,
    /// Blanes-Moan 4th-order splitting: third drift sub-step (a3*dt).
    Bm4Sub4 = 54,
    /// Blanes-Moan 4th-order splitting: third kick sub-step (b3*dt).
    Bm4Sub5 = 55,
    /// Blanes-Moan 4th-order splitting: fourth drift sub-step (a4*dt).
    Bm4Sub6 = 56,
    /// Blanes-Moan 4th-order splitting: fourth kick sub-step (b4*dt).
    Bm4Sub7 = 57,
    /// Blanes-Moan 4th-order splitting: fifth drift sub-step (a5*dt).
    Bm4Sub8 = 58,
    /// Blanes-Moan 4th-order splitting: fifth kick sub-step (b5*dt).
    Bm4Sub9 = 59,
    // RKN6 (triple-jump composition)
    /// 6th-order Runge-Kutta-Nystrom triple-jump composition: first Strang block.
    Rkn6Phase1 = 60,
    /// 6th-order Runge-Kutta-Nystrom triple-jump composition: second (scaled) Strang block.
    Rkn6Phase2 = 61,
    /// 6th-order Runge-Kutta-Nystrom triple-jump composition: third Strang block.
    Rkn6Phase3 = 62,
    // BUG integrator
    /// BUG integrator: K-step (kinetic/drift sub-step).
    BugKStep = 80,
    /// BUG integrator: L-step (Lie/interaction sub-step).
    BugLStep = 81,
    /// BUG integrator: S-step (Strang-like combined sub-step).
    BugSStep = 82,
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
            44 => Self::AdaptiveLie,
            45 => Self::AdaptiveLieKick,
            46 => Self::AdaptiveError,
            50 => Self::Bm4Sub0,
            51 => Self::Bm4Sub1,
            52 => Self::Bm4Sub2,
            53 => Self::Bm4Sub3,
            54 => Self::Bm4Sub4,
            55 => Self::Bm4Sub5,
            56 => Self::Bm4Sub6,
            57 => Self::Bm4Sub7,
            58 => Self::Bm4Sub8,
            59 => Self::Bm4Sub9,
            60 => Self::Rkn6Phase1,
            61 => Self::Rkn6Phase2,
            62 => Self::Rkn6Phase3,
            80 => Self::BugKStep,
            81 => Self::BugLStep,
            82 => Self::BugSStep,
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
            Self::AdaptiveLie => "Lie estimate",
            Self::AdaptiveLieKick => "Lie kick",
            Self::AdaptiveError => "Error estimate",
            Self::Bm4Sub0 => "BM4 drift 1",
            Self::Bm4Sub1 => "BM4 kick 1",
            Self::Bm4Sub2 => "BM4 drift 2",
            Self::Bm4Sub3 => "BM4 kick 2",
            Self::Bm4Sub4 => "BM4 drift 3",
            Self::Bm4Sub5 => "BM4 kick 3",
            Self::Bm4Sub6 => "BM4 drift 4",
            Self::Bm4Sub7 => "BM4 kick 4",
            Self::Bm4Sub8 => "BM4 drift 5",
            Self::Bm4Sub9 => "BM4 kick 5",
            Self::Rkn6Phase1 => "RKN6 phase 1",
            Self::Rkn6Phase2 => "RKN6 phase 2",
            Self::Rkn6Phase3 => "RKN6 phase 3",
            Self::BugKStep => "BUG K-step",
            Self::BugLStep => "BUG L-step",
            Self::BugSStep => "BUG S-step",
        }
    }

    /// Whether this is a build phase (set by phasma before caustic starts stepping).
    pub fn is_build(self) -> bool {
        matches!(
            self,
            Self::BuildDomain
                | Self::BuildIC
                | Self::BuildICSampling
                | Self::BuildICCompression
                | Self::BuildPoisson
                | Self::BuildIntegrator
                | Self::BuildExitConditions
                | Self::BuildAssembly
        )
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
