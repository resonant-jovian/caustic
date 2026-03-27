//! Unified simulation event system for structured observability.
//!
//! `SimEvent` is the single source of truth for all computational decisions,
//! progress, warnings, and diagnostics in caustic. Components emit events through
//! an `EventEmitter`; consumers (e.g. phasma TUI) drain events from an `EventReceiver`.
//!
//! The channel is always-on (~5% wall-time budget). When the bounded channel is full,
//! events are dropped (non-blocking sender) to maintain real-time performance.

use crossbeam_channel::{Receiver, Sender, TrySendError};

use super::conditions::ExitReason;
use super::diagnostics::GlobalDiagnostics;
use super::integrator::StepTimings;
use super::progress::StepPhase;

// ─── EventEmitter / EventReceiver ────────────────────────────────────────────

/// Sends `SimEvent`s into a bounded channel. Thread-safe, cloneable.
///
/// All simulation components emit events through this. The `emit()` method is
/// non-blocking: if the channel is full the event is silently dropped.
#[derive(Clone)]
pub struct EventEmitter {
    tx: Sender<SimEvent>,
}

/// Receives `SimEvent`s from the simulation. Typically owned by the TUI or batch consumer.
pub struct EventReceiver {
    rx: Receiver<SimEvent>,
}

impl EventEmitter {
    /// Create a paired emitter + receiver with the given bounded capacity.
    ///
    /// Default capacity is 4096 events, configurable via
    /// `SimulationBuilder::event_channel_capacity()`.
    pub fn channel(capacity: usize) -> (Self, EventReceiver) {
        let (tx, rx) = crossbeam_channel::bounded(capacity);
        (Self { tx }, EventReceiver { rx })
    }

    /// Emit an event. Non-blocking. Returns `false` if the channel is full
    /// (event dropped) or the receiver has been dropped.
    #[inline]
    pub fn emit(&self, event: SimEvent) -> bool {
        match self.tx.try_send(event) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    /// Create a no-op emitter whose events are immediately discarded.
    /// Use in tests and headless/batch runs where no consumer is attached.
    pub fn sink() -> Self {
        // Channel capacity 0 means sends always fail (try_send → Full).
        // This is cheaper than capacity 1 because no buffer is allocated.
        let (tx, _rx) = crossbeam_channel::bounded(0);
        Self { tx }
    }
}

impl EventReceiver {
    /// Drain all pending events without blocking. Call from a TUI frame loop
    /// or batch consumer poll.
    pub fn drain(&self) -> Vec<SimEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    /// Blocking receive. Use in batch/headless consumers that process events
    /// sequentially.
    pub fn recv(&self) -> Result<SimEvent, crossbeam_channel::RecvError> {
        self.rx.recv()
    }

    /// Non-blocking receive. Returns `None` if no event is available.
    pub fn try_recv(&self) -> Option<SimEvent> {
        self.rx.try_recv().ok()
    }
}

// ─── Supporting enums ────────────────────────────────────────────────────────

/// Which CFL / dynamical constraint determined the timestep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TimestepConstraint {
    CflSpatial,
    CflVelocity,
    Dynamical,
    Orbital,
    UserLimit,
}

/// Poisson solver implementation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SolverKind {
    Fft,
    FftIsolated,
    Multigrid,
    HtPoisson,
    TensorPoisson,
    SphericalHarmonics,
    Tree,
    RangeSeparated,
    Spherical1D,
    Vgf,
}

/// Phase-space representation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReprKind {
    UniformGrid6D,
    HtTensor,
    TensorTrain,
    SheetTracker,
    SpectralV,
    AmrGrid,
    HybridRepr,
    FlowMapRepr,
    MacroMicroRepr,
    SphericalRepr,
}

/// Time integrator kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IntegratorKind {
    Strang,
    Yoshida,
    Lie,
    Unsplit,
    Rkei,
    InstrumentedStrang,
    AdaptiveStrang,
    BlanesMoan,
    Cosmological,
    LawsonRk,
    Rkn6,
    Bug,
    ParallelBug,
    RkBug,
}

/// Spatial (drift) or velocity (kick) advection direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AdvectDirection {
    Spatial,
    Velocity,
}

/// Conserved physical quantity for drift monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConservedQuantity {
    Energy,
    Mass,
    CasimirC2,
    Entropy,
    Momentum,
    AngularMomentum,
}

/// Exit condition type for distance-to-threshold reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExitConditionKind {
    TimeLimit,
    EnergyDrift,
    MassLoss,
    CasimirDrift,
    WallClock,
    SteadyState,
    CflViolation,
    VirialRelaxed,
    CausticFormation,
}

/// Pre-simulation build phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BuildPhase {
    Domain,
    IC,
    ICSampling,
    ICCompression,
    Poisson,
    Integrator,
    ExitConditions,
    Assembly,
}

/// SLAR interpolation method for HT tensor advection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SlarMethod {
    SparsePolynomial,
    TricubicCatmullRom,
}

/// FFT transform direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FftDirection {
    Forward,
    Inverse,
}

/// Component category for memory tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComponentKind {
    Repr,
    Poisson,
    LoMaC,
    Scratch,
    Diagnostics,
}

/// Structured warning (non-fatal but noteworthy).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SimWarning {
    /// HT rank budget nearly exhausted.
    RankBudgetSaturated { fraction: f64, max_rank: u32 },
    /// Exponential rank growth detected.
    RankExplosion { growth_rate: f64, doubling_steps: f64 },
    /// Mass changed more than expected during a phase.
    MassImbalance { relative_change: f64, phase: String },
    /// Mass collapsed to near-zero after tensor reconstruction.
    MassCollapsed { phase: String, pre_mass: f64, post_mass: f64 },
    /// NaN detected in a field or tensor component.
    NaNDetected { component: String },
    /// Iterative solver stalled before convergence.
    ConvergenceStalled { solver: String, iterations: u32, residual: f64 },
    /// Positivity limiter activated on many cells.
    PositivityLimitActive { cells_limited: u64 },
}

// ─── SimEvent ────────────────────────────────────────────────────────────────

/// Unified simulation event — the single source of truth for all observability.
///
/// Components emit events through [`EventEmitter::emit()`]. Consumers drain them
/// via [`EventReceiver::drain()`]. Events are grouped by category:
///
/// - **Lifecycle**: simulation start/complete
/// - **Step**: per-step phases, timing, progress
/// - **Timestep control**: CFL, adaptive dt accept/reject
/// - **Poisson solve**: convergence, timing
/// - **Phase-space**: advection, density, positivity
/// - **HT tensor**: rank, truncation, SLAR, ACA, frame quality
/// - **TT tensor**: recompression
/// - **Conservation (LoMaC)**: correction magnitude, skip reasons
/// - **Exit conditions**: distance-to-threshold monitoring
/// - **Diagnostics**: full snapshots, conservation drift
/// - **Warnings**: structured non-fatal alerts
/// - **Build phases**: pre-simulation construction progress
/// - **Performance**: memory, FFT plans, rayon, operation timing
#[derive(Debug, Clone)]
pub enum SimEvent {
    // ═══ Simulation Lifecycle ═════════════════════════════════════════════════

    /// Emitted once at simulation start with full configuration summary.
    SimStarted {
        repr_kind: ReprKind,
        solver_kind: SolverKind,
        integrator_kind: IntegratorKind,
        grid_shape: [usize; 6],
        memory_bytes: usize,
        g: f64,
        t_final: f64,
    },
    /// Emitted once when simulation terminates.
    SimComplete {
        reason: ExitReason,
        total_steps: u64,
        wall_secs: f64,
    },

    // ═══ Step Lifecycle ═══════════════════════════════════════════════════════

    /// Beginning of a new timestep.
    StepStarted { step: u64, time: f64, dt: f64 },
    /// Integrator entered a new sub-phase within the step.
    PhaseEntered { phase: StepPhase, step: u64 },
    /// Intra-phase work-unit progress (e.g. grid cells processed).
    SubStepProgress { done: u64, total: u64 },
    /// Step completed with full timing breakdown.
    StepComplete {
        step: u64,
        dt: f64,
        wall_ms: f64,
        timings: StepTimings,
    },

    // ═══ Timestep Control ════════════════════════════════════════════════════

    /// Timestep determined from CFL / dynamical constraints.
    TimestepComputed {
        dt: f64,
        constraint: TimestepConstraint,
        rho_max: f64,
        cfl_factor: f64,
    },
    /// Adaptive integrator rejected a timestep (error too large).
    AdaptiveDtRejected {
        attempted_dt: f64,
        error_estimate: f64,
        new_dt: f64,
        rejection_count: u32,
    },
    /// Adaptive integrator accepted a timestep.
    AdaptiveDtAccepted { dt: f64, error_estimate: f64 },

    // ═══ Poisson Solve ═══════════════════════════════════════════════════════

    /// Poisson solve completed (any solver type).
    PoissonSolveComplete { solver: SolverKind, wall_us: u64 },
    /// Multigrid V-cycle converged.
    MultigridConverged {
        iterations: u32,
        initial_residual: f64,
        final_residual: f64,
        convergence_rate: f64,
    },
    /// Multigrid failed to converge within max iterations.
    MultigridDiverged { iterations: u32, residual: f64 },

    // ═══ Phase-Space Operations ══════════════════════════════════════════════

    /// Density field computed (emitted by the caller of compute_density, not the repr).
    DensityComputed {
        total_mass: f64,
        rho_max: f64,
        rho_min: f64,
    },
    /// Advection sub-step completed with mass balance.
    AdvectionComplete {
        direction: AdvectDirection,
        mass_before: f64,
        mass_after: f64,
        wall_us: u64,
    },
    /// Positivity limiter activated during advection.
    PositivityViolations { count: u64, max_magnitude: f64 },

    // ═══ HT Tensor Deep Observability ════════════════════════════════════════

    /// Per-step HT rank snapshot across all tensor tree nodes.
    HtRankSnapshot {
        ranks: Vec<u32>,
        total_rank: u32,
        memory_bytes: usize,
        compression_ratio: f64,
    },
    /// Single node truncated during HSVD.
    HtTruncation {
        node: u32,
        old_rank: u32,
        new_rank: u32,
        max_sv: f64,
        min_kept_sv: f64,
        discarded_count: u32,
    },
    /// SLAR interpolation path selected for advection.
    HtSlarPath {
        interpolation: SlarMethod,
        nodes_visited: u32,
        wall_us: u64,
    },
    /// ACA (Adaptive Cross Approximation) evaluation completed.
    HtAcaEvaluation {
        node: u32,
        achieved_rank: u32,
        function_evaluations: u64,
        relative_error: f64,
    },
    /// HSVD frame quality for a node (NaN recovery, condition).
    HtFrameQuality {
        node: u32,
        nan_recovery_used: bool,
        condition_number: f64,
    },
    /// Per-axis rank growth during advection.
    HtPerAxisAdvection {
        axis: u8,
        direction: AdvectDirection,
        pre_rank: u32,
        post_rank: u32,
        wall_us: u64,
    },
    /// Materialization attempt status.
    HtMaterializationStatus {
        element_count: u64,
        memory_bytes: usize,
        fits_in_memory: bool,
        materialized: bool,
    },
    /// Fiber sampling found negative values (positivity concern).
    HtFiberSampling {
        negative_values: u64,
        total_sampled: u64,
    },

    // ═══ TT (Tensor Train) Observability ═════════════════════════════════════

    /// TT-SVD recompression changed ranks.
    TtRecompression {
        old_ranks: Vec<u32>,
        new_ranks: Vec<u32>,
    },

    // ═══ Conservation (LoMaC) ════════════════════════════════════════════════

    /// LoMaC conservative projection completed.
    LoMaCComplete {
        correction_norm: f64,
        casimir_pre: f64,
        casimir_post: f64,
        wall_ms: f64,
    },
    /// LoMaC projection skipped.
    LoMaCSkipped { reason: String },

    // ═══ Exit Conditions ═════════════════════════════════════════════════════

    /// Per-condition status: how close to triggering (emitted each step).
    ExitConditionStatus {
        condition: ExitConditionKind,
        current_value: f64,
        threshold: f64,
        /// 0.0 = far from threshold, 1.0 = at threshold.
        fraction_to_threshold: f64,
    },
    /// An exit condition fired.
    ExitTriggered { reason: ExitReason },

    // ═══ Diagnostics ═════════════════════════════════════════════════════════

    /// Full diagnostics snapshot (boxed to keep SimEvent small).
    DiagnosticsComputed(Box<GlobalDiagnostics>),
    /// Per-quantity conservation drift.
    ConservationDrift {
        quantity: ConservedQuantity,
        relative_drift: f64,
    },

    // ═══ Warnings ════════════════════════════════════════════════════════════

    /// Structured non-fatal warning.
    Warning(SimWarning),

    // ═══ Build Phases ════════════════════════════════════════════════════════

    /// Pre-simulation build phase started.
    BuildPhaseStarted { phase: BuildPhase },
    /// Pre-simulation build phase completed with timing.
    BuildPhaseComplete { phase: BuildPhase, wall_ms: f64 },

    // ═══ Performance Profiling ═══════════════════════════════════════════════

    /// Memory usage snapshot for a component.
    MemorySnapshot {
        component: ComponentKind,
        bytes: usize,
        peak_bytes: usize,
    },
    /// FFT plan creation or cache hit.
    FftPlanStatus {
        direction: FftDirection,
        shape: [usize; 3],
        reused: bool,
        plan_wall_us: u64,
    },
    /// Rayon thread pool status at step boundary.
    RayonPoolStatus {
        active_threads: usize,
        total_threads: usize,
    },
    /// Fine-grained operation wall-clock timing.
    OperationTiming { operation: String, wall_us: u64 },
    /// Scratch buffer allocated for temporary computation.
    ScratchBufferAllocated { purpose: String, bytes: usize },
}
