#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# dev.sh — Unified development script for caustic & phasma
#
# Usage: ./dev.sh <command> [flags]
#
# Commands: test, bench, profile, build, lint, clean, info, help
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
    RED=$'\e[0;31m'
    GREEN=$'\e[0;32m'
    YELLOW=$'\e[0;33m'
    BLUE=$'\e[0;34m'
    MAGENTA=$'\e[0;35m'
    CYAN=$'\e[0;36m'
    BOLD=$'\e[1m'
    DIM=$'\e[2m'
    RESET=$'\e[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' BOLD='' DIM='' RESET=''
fi

# ── Project Detection ────────────────────────────────────────────────────────

detect_project() {
    if [[ ! -f Cargo.toml ]]; then
        echo -e "${RED}error:${RESET} no Cargo.toml found in $(pwd)" >&2
        exit 1
    fi

    if grep -q '^name = "caustic"' Cargo.toml 2>/dev/null; then
        PROJECT="caustic"
        SIBLING="phasma"
        SIBLING_DIR="../phasma"
    elif grep -q '^name = "phasma"' Cargo.toml 2>/dev/null; then
        PROJECT="phasma"
        SIBLING="caustic"
        SIBLING_DIR="../caustic"
    else
        echo -e "${RED}error:${RESET} unrecognised project in Cargo.toml" >&2
        exit 1
    fi
}

detect_project

# ── Helpers ──────────────────────────────────────────────────────────────────

log()  { echo -e "${CYAN}▸${RESET} $*"; }
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; }
hdr()  { echo -e "\n${BOLD}${BLUE}── $* ──${RESET}\n"; }

has_cmd() { command -v "$1" &>/dev/null; }

require_cmd() {
    if ! has_cmd "$1"; then
        err "$1 is not installed"
        if [[ -n "${2:-}" ]]; then
            echo -e "  ${DIM}install: $2${RESET}" >&2
        fi
        exit 1
    fi
}

require_perf_paranoid() {
    local paranoid_file="/proc/sys/kernel/perf_event_paranoid"
    [[ -f "$paranoid_file" ]] || return 0
    local level
    level=$(< "$paranoid_file")
    if (( level > 1 )); then
        warn "perf_event_paranoid is $level (needs ≤ 1 for non-root profiling)"
        log "run: echo 1 | sudo tee $paranoid_file"
        read -rp "  fix now? [Y/n] " ans
        if [[ "${ans:-y}" =~ ^[Yy]$ ]]; then
            echo 1 | sudo tee "$paranoid_file" > /dev/null
            ok "perf_event_paranoid set to 1"
        else
            err "cannot profile without perf_event_paranoid ≤ 1"
            exit 1
        fi
    fi
}

timestamp() { date +%Y-%m-%d_%H-%M-%S; }

ensure_dir() { mkdir -p "$1"; }

run_in_sibling() {
    if [[ ! -d "$SIBLING_DIR" ]]; then
        err "$SIBLING project not found at $SIBLING_DIR"
        exit 1
    fi
    log "running in ${BOLD}$SIBLING${RESET} (${SIBLING_DIR})"
    (cd "$SIBLING_DIR" && "$@")
}

# ── Interactive Target Picker ────────────────────────────────────────────────

pick_profile_target() {
    local targets=()
    local labels=()

    if [[ "$PROJECT" == "phasma" ]]; then
        # List preset configs for phasma
        if [[ -d configs ]]; then
            for cfg in configs/*.toml; do
                [[ -f "$cfg" ]] || continue
                local name
                name=$(basename "$cfg" .toml)
                targets+=("./target/profiling/phasma --config $cfg --batch")
                labels+=("phasma --config $cfg --batch")
            done
        fi
        # Add TUI mode
        targets+=("./target/profiling/phasma --config configs/plummer.toml --run")
        labels+=("phasma --config configs/plummer.toml --run (TUI)")
    elif [[ "$PROJECT" == "caustic" ]]; then
        # Benchmark binary
        if [[ -f benches/solver_kernels.rs ]]; then
            targets+=("./target/profiling/solver_kernels")
            labels+=("solver_kernels (benchmark binary)")
        fi
    fi

    # Custom command option always last
    targets+=("CUSTOM")
    labels+=("Custom command...")

    echo -e "\n${BOLD}Select target to profile:${RESET}"
    for i in "${!labels[@]}"; do
        printf "  ${CYAN}%2d${RESET}) %s\n" $((i + 1)) "${labels[$i]}"
    done

    local choice
    while true; do
        echo -ne "\n${BOLD}choice [1-${#labels[@]}]:${RESET} "
        read -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#labels[@]} )); then
            break
        fi
        warn "invalid selection"
    done

    local idx=$((choice - 1))
    if [[ "${targets[$idx]}" == "CUSTOM" ]]; then
        echo -ne "${BOLD}enter command:${RESET} "
        read -r PROFILE_CMD
    else
        PROFILE_CMD="${targets[$idx]}"
    fi

    echo -e "${DIM}→ $PROFILE_CMD${RESET}\n"
}

# ══════════════════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════════════════

# ── test ─────────────────────────────────────────────────────────────────────

cmd_test() {
    local run_all=false
    local mode=""  # empty = project default
    local ignored=""
    local filter=""
    local threads=""
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)             run_all=true ;;
            --release)         mode="release" ;;
            --debug)           mode="debug" ;;
            --ignored)         ignored="--ignored" ;;
            --include-ignored) ignored="--include-ignored" ;;
            --filter)          shift; filter="$1" ;;
            --threads)         shift; threads="$1" ;;
            --help|-h)         cmd_test_help; return ;;
            *)                 extra_args+=("$1") ;;
        esac
        shift
    done

    # Project defaults
    if [[ -z "$mode" ]]; then
        if [[ "$PROJECT" == "caustic" ]]; then
            mode="release"
        else
            mode="debug"
        fi
    fi

    local cargo_args=()
    [[ "$mode" == "release" ]] && cargo_args+=(--release)

    local test_args=(--)
    if [[ -z "$threads" ]]; then
        [[ "$PROJECT" == "caustic" ]] && test_args+=(--test-threads=1)
    else
        test_args+=(--test-threads="$threads")
    fi
    [[ -n "$ignored" ]] && test_args+=("$ignored")
    [[ -n "$filter" ]]  && test_args+=("$filter")
    test_args+=("${extra_args[@]}")

    hdr "test $PROJECT ($mode)"
    log "cargo test ${cargo_args[*]} ${test_args[*]}"
    cargo test "${cargo_args[@]}" "${test_args[@]}"
    ok "tests passed"

    if $run_all; then
        hdr "test $SIBLING"
        run_in_sibling ./dev.sh test ${mode:+--$mode} ${ignored:+$ignored} ${filter:+--filter "$filter"} ${threads:+--threads "$threads"}
    fi
}

cmd_test_help() {
    cat <<'EOF'
Usage: ./dev.sh test [flags]

Flags:
  --all               run tests for both projects
  --release           force release mode
  --debug             force debug mode
  --ignored           run only #[ignore] tests
  --include-ignored   run normal + ignored tests
  --filter <pattern>  filter test names
  --threads <n>       override --test-threads

Defaults:
  caustic: --release -- --test-threads=1
  phasma:  debug, parallel
EOF
}

# ── bench ────────────────────────────────────────────────────────────────────

cmd_bench() {
    local filter=""
    local save=""
    local compare=""
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --filter)  shift; filter="$1" ;;
            --save)    shift; save="$1" ;;
            --compare) shift; compare="$1" ;;
            --all)     ;; # only caustic has benches
            --help|-h) cmd_bench_help; return ;;
            *)         extra_args+=("$1") ;;
        esac
        shift
    done

    # If in phasma, run caustic benchmarks
    if [[ "$PROJECT" == "phasma" ]]; then
        hdr "bench (via ../caustic)"
        run_in_sibling ./dev.sh bench ${filter:+--filter "$filter"} ${save:+--save "$save"} ${compare:+--compare "$compare"} "${extra_args[@]}"
        return
    fi

    hdr "bench $PROJECT"

    local bench_args=()
    [[ -n "$filter" ]]  && bench_args+=(--bench solver_kernels -- "$filter")
    [[ -n "$save" ]]    && bench_args+=(-- --save-baseline "$save")
    [[ -n "$compare" ]] && bench_args+=(-- --baseline "$compare")
    bench_args+=("${extra_args[@]}")

    log "cargo bench ${bench_args[*]}"
    cargo bench "${bench_args[@]}"
    ok "benchmarks complete"
}

cmd_bench_help() {
    cat <<'EOF'
Usage: ./dev.sh bench [flags]

Flags:
  --filter <pattern>  filter benchmark names
  --save <name>       save baseline (criterion --save-baseline)
  --compare <name>    compare to baseline (criterion --baseline)
  --all               run benchmarks (only caustic has them)

Note: Benchmarks exist only in caustic. Running from phasma delegates to ../caustic.
EOF
}

# ── profile ──────────────────────────────────────────────────────────────────

cmd_profile() {
    if [[ $# -lt 1 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        cmd_profile_help
        return
    fi

    local tool="$1"; shift
    local PROFILE_CMD=""
    local outdir="target/profiles/$tool/$(timestamp)"

    case "$tool" in
        perf)        profile_perf "$outdir" "$@" ;;
        flamegraph)  profile_flamegraph "$outdir" "$@" ;;
        samply)      profile_samply "$@" ;;
        dhat)        profile_dhat "$outdir" "$@" ;;
        tracy)       profile_tracy "$@" ;;
        valgrind)    profile_valgrind "$outdir" "$@" ;;
        cachegrind)  profile_cachegrind "$outdir" "$@" ;;
        heaptrack)   profile_heaptrack "$outdir" "$@" ;;
        massif)      profile_massif "$outdir" "$@" ;;
        *)           err "unknown profiling tool: $tool"; cmd_profile_help; exit 1 ;;
    esac
}

profile_build() {
    local profile="${1:-profiling}"
    local features="${2:-}"

    local build_args=(--profile "$profile")
    [[ -n "$features" ]] && build_args+=(--features "$features")

    log "building with profile=${BOLD}$profile${RESET} ${features:+features=$features}"
    cargo build "${build_args[@]}"
}

profile_perf() {
    local outdir="$1"; shift
    require_cmd perf "sudo pacman -S perf  (or linux-tools on Ubuntu)"
    require_perf_paranoid

    profile_build profiling
    pick_profile_target
    ensure_dir "$outdir"

    hdr "perf record"
    log "output: $outdir/perf.data"
    perf record -g --call-graph dwarf -o "$outdir/perf.data" -- $PROFILE_CMD "$@"
    ok "recorded"

    log "opening perf report..."
    perf report -i "$outdir/perf.data"
}

profile_flamegraph() {
    local outdir="$1"; shift
    require_cmd cargo-flamegraph "cargo install flamegraph"
    require_perf_paranoid

    profile_build profiling
    pick_profile_target
    ensure_dir "$outdir"

    hdr "flamegraph"
    local svg="$outdir/flamegraph.svg"
    log "output: $svg"
    flamegraph -o "$svg" -- $PROFILE_CMD "$@"
    ok "flamegraph written to $svg"

    if has_cmd xdg-open; then
        xdg-open "$svg" 2>/dev/null &
    elif has_cmd open; then
        open "$svg" 2>/dev/null &
    fi
}

profile_samply() {
    require_cmd samply "cargo install samply"
    require_perf_paranoid

    profile_build profiling
    pick_profile_target

    hdr "samply"
    log "opens browser with profiler UI"
    samply record -- $PROFILE_CMD "$@"
}

profile_dhat() {
    local outdir="$1"; shift

    local feature_name
    if [[ "$PROJECT" == "caustic" ]]; then
        feature_name="dhat-heap"
    else
        feature_name="dhat"
    fi

    profile_build release "$feature_name"
    pick_profile_target
    ensure_dir "$outdir"

    hdr "dhat heap profiling"
    log "output: $outdir/"
    # dhat writes dhat-heap.json to cwd
    (cd "$outdir" && eval "$SCRIPT_DIR/$PROFILE_CMD" "$@")

    local json
    json=$(find "$outdir" -name 'dhat-heap.json' -o -name 'dhat-heap.*.json' 2>/dev/null | head -1)
    if [[ -n "$json" ]]; then
        ok "heap profile: $json"
        echo -e "  ${DIM}view at: https://nnethercote.github.io/dh_view/dh_view.html${RESET}"
    else
        warn "no dhat-heap.json found — did the program run long enough?"
    fi
}

profile_tracy() {
    local feature_name
    if [[ "$PROJECT" == "caustic" ]]; then
        feature_name="tracy"
    else
        feature_name="tracy"
    fi

    profile_build release "$feature_name"
    pick_profile_target

    hdr "tracy profiler"
    log "connect Tracy profiler to capture trace data"
    if has_cmd tracy-capture; then
        log "tracy-capture found — run it in a separate terminal"
    fi
    eval "$PROFILE_CMD" "$@"
}

profile_valgrind() {
    local outdir="$1"; shift
    require_cmd valgrind "sudo pacman -S valgrind  (or apt install valgrind)"

    profile_build profiling
    pick_profile_target
    ensure_dir "$outdir"

    hdr "callgrind"
    local outfile="$outdir/callgrind.out"
    log "output: $outfile"
    valgrind --tool=callgrind --callgrind-out-file="$outfile" $PROFILE_CMD "$@"
    ok "callgrind complete"

    if has_cmd callgrind_annotate; then
        callgrind_annotate "$outfile" | head -80
    fi
    if has_cmd kcachegrind; then
        log "opening kcachegrind..."
        kcachegrind "$outfile" 2>/dev/null &
    fi
}

profile_cachegrind() {
    local outdir="$1"; shift
    require_cmd valgrind "sudo pacman -S valgrind  (or apt install valgrind)"

    profile_build profiling
    pick_profile_target
    ensure_dir "$outdir"

    hdr "cachegrind"
    local outfile="$outdir/cachegrind.out"
    log "output: $outfile"
    valgrind --tool=cachegrind --cachegrind-out-file="$outfile" $PROFILE_CMD "$@"
    ok "cachegrind complete"

    if has_cmd cg_annotate; then
        cg_annotate "$outfile" | head -60
    fi
}

profile_heaptrack() {
    local outdir="$1"; shift
    require_cmd heaptrack "sudo pacman -S heaptrack  (or apt install heaptrack)"

    profile_build profiling
    pick_profile_target
    ensure_dir "$outdir"

    hdr "heaptrack"
    log "output: $outdir/"
    heaptrack -o "$outdir/heaptrack" $PROFILE_CMD "$@"
    ok "heaptrack complete"

    local latest
    latest=$(ls -t "$outdir"/heaptrack.*.zst 2>/dev/null | head -1)
    if [[ -n "$latest" ]] && has_cmd heaptrack_gui; then
        log "opening heaptrack_gui..."
        heaptrack_gui "$latest" 2>/dev/null &
    elif [[ -n "$latest" ]] && has_cmd heaptrack_print; then
        heaptrack_print "$latest" | head -40
    fi
}

profile_massif() {
    local outdir="$1"; shift
    require_cmd valgrind "sudo pacman -S valgrind  (or apt install valgrind)"

    profile_build profiling
    pick_profile_target
    ensure_dir "$outdir"

    hdr "massif"
    local outfile="$outdir/massif.out"
    log "output: $outfile"
    valgrind --tool=massif --massif-out-file="$outfile" $PROFILE_CMD "$@"
    ok "massif complete"

    if has_cmd ms_print; then
        ms_print "$outfile" | head -60
    fi
    if has_cmd massif-visualizer; then
        log "opening massif-visualizer..."
        massif-visualizer "$outfile" 2>/dev/null &
    fi
}

cmd_profile_help() {
    cat <<'EOF'
Usage: ./dev.sh profile <tool>

Tools:
  perf         perf record + perf report (needs linux perf)
  flamegraph   cargo-flamegraph → SVG (needs cargo install flamegraph)
  samply       samply record → browser (needs cargo install samply)
  dhat         heap profiling via dhat (uses --features dhat-heap/dhat)
  tracy        Tracy profiler (uses --features tracy)
  valgrind     callgrind analysis (needs valgrind)
  cachegrind   cache/branch miss analysis (needs valgrind)
  heaptrack    heap tracking (needs heaptrack)
  massif       valgrind heap profiler (needs valgrind)

Each tool presents an interactive target picker before running.
Output goes to target/profiles/<tool>/<timestamp>/
EOF
}

# ── build ────────────────────────────────────────────────────────────────────

cmd_build() {
    local profile=""
    local features=""
    local run_all=false
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --release)    profile="release" ;;
            --fast)
                if [[ "$PROJECT" == "phasma" ]]; then
                    profile="fast-release"
                else
                    profile="release"
                fi
                ;;
            --profiling)  profile="profiling" ;;
            --features)   shift; features="$1" ;;
            --all)        run_all=true ;;
            --help|-h)    cmd_build_help; return ;;
            *)            extra_args+=("$1") ;;
        esac
        shift
    done

    local build_args=()
    [[ -n "$profile" ]]  && build_args+=(--profile "$profile")
    [[ -n "$features" ]] && build_args+=(--features "$features")
    build_args+=("${extra_args[@]}")

    hdr "build $PROJECT ${profile:+($profile)}"
    log "cargo build ${build_args[*]}"
    cargo build "${build_args[@]}"
    ok "build complete"

    if $run_all; then
        hdr "build $SIBLING"
        run_in_sibling ./dev.sh build ${profile:+--$profile} ${features:+--features "$features"}
    fi
}

cmd_build_help() {
    cat <<'EOF'
Usage: ./dev.sh build [flags]

Flags:
  --release       release profile (fat LTO, codegen-units=1)
  --fast          fast-release (phasma: thin LTO; caustic: release)
  --profiling     release + debug symbols (for perf/samply)
  --features <f>  comma-separated features (e.g., jemalloc,tracy)
  --all           build both projects
EOF
}

# ── lint ─────────────────────────────────────────────────────────────────────

cmd_lint() {
    local fix=false
    local run_all=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fix)     fix=true ;;
            --all)     run_all=true ;;
            --help|-h) cmd_lint_help; return ;;
        esac
        shift
    done

    hdr "lint $PROJECT"

    if $fix; then
        log "cargo clippy --fix --allow-dirty"
        cargo clippy --fix --allow-dirty
        log "cargo fmt"
        cargo fmt
    else
        log "cargo clippy"
        cargo clippy
        log "cargo fmt --check"
        cargo fmt --check
    fi
    ok "lint passed"

    if $run_all; then
        hdr "lint $SIBLING"
        run_in_sibling ./dev.sh lint $($fix && echo "--fix")
    fi
}

cmd_lint_help() {
    cat <<'EOF'
Usage: ./dev.sh lint [flags]

Flags:
  --fix   apply clippy fixes and format code
  --all   lint both projects
EOF
}

# ── clean ────────────────────────────────────────────────────────────────────

cmd_clean() {
    local run_all=false
    local profiles_only=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)      run_all=true ;;
            --profiles) profiles_only=true ;;
            --help|-h)  cmd_clean_help; return ;;
        esac
        shift
    done

    hdr "clean $PROJECT"

    if $profiles_only; then
        log "removing target/profiles/"
        rm -rf target/profiles
    else
        log "cargo clean"
        cargo clean
    fi
    ok "clean complete"

    if $run_all; then
        hdr "clean $SIBLING"
        run_in_sibling ./dev.sh clean $($profiles_only && echo "--profiles")
    fi
}

cmd_clean_help() {
    cat <<'EOF'
Usage: ./dev.sh clean [flags]

Flags:
  --all        clean both projects
  --profiles   remove only target/profiles/ (keep build artifacts)
EOF
}

# ── doctor ───────────────────────────────────────────────────────────────────

# Tool registry: name:check_cmd:install_cmd:category
# Categories: required, recommended, optional
TOOL_REGISTRY=(
    "rustc:rustc:curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh:required"
    "cargo:cargo:(installed with rustc):required"
    "perf:perf:sudo pacman -S perf  # or: sudo apt install linux-tools-\$(uname -r):recommended"
    "cargo-flamegraph:cargo-flamegraph:cargo install flamegraph:recommended"
    "samply:samply:cargo install samply:recommended"
    "valgrind:valgrind:sudo pacman -S valgrind  # or: sudo apt install valgrind:recommended"
    "heaptrack:heaptrack:sudo pacman -S heaptrack  # or: sudo apt install heaptrack:optional"
    "tracy-capture:tracy-capture:yay -S tracy  # (AUR) or build from source:optional"
    "kcachegrind:kcachegrind:sudo pacman -S kcachegrind  # or: sudo apt install kcachegrind:optional"
    "massif-visualizer:massif-visualizer:sudo pacman -S massif-visualizer:optional"
    "heaptrack_gui:heaptrack_gui:(included with heaptrack):optional"
)

check_tool() {
    local name="$1" cmd="$2" install="$3" category="$4"
    if has_cmd "$cmd"; then
        echo -e "  ${GREEN}✓${RESET} $name"
        return 0
    else
        local tag
        case "$category" in
            required)    tag="${RED}REQUIRED${RESET}" ;;
            recommended) tag="${YELLOW}recommended${RESET}" ;;
            optional)    tag="${DIM}optional${RESET}" ;;
        esac
        echo -e "  ${RED}✗${RESET} $name  [$tag]"
        echo -e "    ${DIM}→ $install${RESET}"
        return 1
    fi
}

cmd_doctor() {
    hdr "doctor — checking prerequisites for $PROJECT"

    local missing_required=0
    local missing_recommended=0
    local missing_optional=0
    local cargo_installs=()
    local pacman_installs=()

    for entry in "${TOOL_REGISTRY[@]}"; do
        IFS=':' read -r name cmd install category <<< "$entry"
        if ! check_tool "$name" "$cmd" "$install" "$category"; then
            case "$category" in
                required)    missing_required=$((missing_required + 1)) ;;
                recommended) missing_recommended=$((missing_recommended + 1)) ;;
                optional)    missing_optional=$((missing_optional + 1)) ;;
            esac
            # Collect install commands for the summary
            if [[ "$install" == cargo\ install* ]]; then
                local pkg="${install#cargo install }"
                cargo_installs+=("$pkg")
            elif [[ "$install" == sudo\ pacman* ]]; then
                local pkg
                pkg=$(echo "$install" | sed 's/sudo pacman -S \([^ ]*\).*/\1/')
                pacman_installs+=("$pkg")
            fi
        fi
    done

    local total_missing=$((missing_required + missing_recommended + missing_optional))

    echo ""
    if [[ $total_missing -eq 0 ]]; then
        ok "all tools installed"
    else
        if [[ $missing_required -gt 0 ]]; then
            err "$missing_required required tool(s) missing"
        fi
        if [[ $missing_recommended -gt 0 ]]; then
            warn "$missing_recommended recommended tool(s) missing"
        fi
        if [[ $missing_optional -gt 0 ]]; then
            log "$missing_optional optional tool(s) missing"
        fi

        # Print a copy-paste install block
        hdr "install missing tools"

        if [[ ${#cargo_installs[@]} -gt 0 ]]; then
            echo -e "  ${BOLD}# Cargo tools${RESET}"
            echo -e "  cargo install ${cargo_installs[*]}"
            echo ""
        fi

        if [[ ${#pacman_installs[@]} -gt 0 ]]; then
            echo -e "  ${BOLD}# System packages (Arch Linux)${RESET}"
            echo -e "  sudo pacman -S --needed ${pacman_installs[*]}"
            echo ""
            echo -e "  ${BOLD}# System packages (Ubuntu/Debian)${RESET}"
            echo -e "  ${DIM}sudo apt install ${pacman_installs[*]}${RESET}"
            echo ""
        fi
    fi

    if [[ $missing_required -gt 0 ]]; then
        exit 1
    fi
}

# ── info ─────────────────────────────────────────────────────────────────────

cmd_info() {
    hdr "system info"

    echo -e "  ${BOLD}project${RESET}    $PROJECT"
    echo -e "  ${BOLD}directory${RESET}  $(pwd)"
    echo -e "  ${BOLD}sibling${RESET}    $SIBLING_DIR"

    echo ""
    if has_cmd rustc; then
        echo -e "  ${BOLD}rustc${RESET}      $(rustc --version)"
    fi
    if has_cmd cargo; then
        echo -e "  ${BOLD}cargo${RESET}      $(cargo --version)"
    fi

    hdr "cargo features"
    if [[ "$PROJECT" == "caustic" ]]; then
        echo "  jemalloc       tikv-jemallocator global allocator"
        echo "  mimalloc-alloc mimalloc global allocator"
        echo "  dhat-heap      dhat heap profiler"
        echo "  tracy          Tracy profiler integration"
        echo "  hdf5           HDF5 I/O"
        echo "  mpi            MPI domain decomposition"
    else
        echo "  jemalloc       → caustic/jemalloc"
        echo "  mimalloc       → caustic/mimalloc-alloc"
        echo "  dhat           → caustic/dhat-heap"
        echo "  tracy          → caustic/tracy"
        echo "  notifications  desktop notifications"
        echo "  mpi            → caustic/mpi"
    fi

    hdr "cargo profiles"
    echo "  dev            debug (default)"
    echo "  release        fat LTO, codegen-units=1"
    if [[ "$PROJECT" == "phasma" ]]; then
        echo "  fast-release   thin LTO, codegen-units=16"
    fi
    echo "  profiling      release + debug symbols"

    hdr "profiling tools"
    echo -e "  ${DIM}(run ./dev.sh doctor for full check with install commands)${RESET}"
    echo ""

    for entry in "${TOOL_REGISTRY[@]}"; do
        IFS=':' read -r name cmd install category <<< "$entry"
        if [[ "$category" == "required" ]]; then continue; fi
        if has_cmd "$cmd"; then
            echo -e "  ${GREEN}✓${RESET} $name"
        else
            echo -e "  ${RED}✗${RESET} $name"
        fi
    done
}

# ── help ─────────────────────────────────────────────────────────────────────

cmd_help() {
    cat <<EOF
${BOLD}dev.sh${RESET} — development script for ${BOLD}caustic${RESET} & ${BOLD}phasma${RESET}

${BOLD}Usage:${RESET} ./dev.sh <command> [flags]

${BOLD}Commands:${RESET}
  ${CYAN}doctor${RESET}     check prerequisites, show install commands for missing tools

  ${CYAN}test${RESET}       run tests
               --all  --release  --debug  --ignored  --include-ignored
               --filter <pattern>  --threads <n>

  ${CYAN}bench${RESET}      run criterion benchmarks (caustic)
               --filter <pattern>  --save <name>  --compare <name>

  ${CYAN}profile${RESET}    profiling with interactive target picker
               perf  flamegraph  samply  dhat  tracy
               valgrind  cachegrind  heaptrack  massif

  ${CYAN}build${RESET}      build project
               --release  --fast  --profiling  --features <f>  --all

  ${CYAN}lint${RESET}       clippy + fmt
               --fix  --all

  ${CYAN}clean${RESET}      clean build artifacts
               --all  --profiles

  ${CYAN}info${RESET}       show system info, features, profiles

  ${CYAN}help${RESET}       this message

${BOLD}Examples:${RESET}
  ./dev.sh doctor                      # check all prerequisites
  ./dev.sh test                        # default tests for this project
  ./dev.sh test --all --ignored        # all tests including #[ignore], both projects
  ./dev.sh bench --save v1             # save criterion baseline
  ./dev.sh profile flamegraph          # generate flamegraph SVG
  ./dev.sh profile dhat                # heap profiling
  ./dev.sh build --fast --features jemalloc
  ./dev.sh info                        # project info, features, profiles

${DIM}Detected project: $PROJECT (sibling: $SIBLING at $SIBLING_DIR)${RESET}
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

if [[ $# -lt 1 ]]; then
    cmd_help
    exit 0
fi

command="$1"; shift

case "$command" in
    doctor)  cmd_doctor ;;
    test)    cmd_test "$@" ;;
    bench)   cmd_bench "$@" ;;
    profile) cmd_profile "$@" ;;
    build)   cmd_build "$@" ;;
    lint)    cmd_lint "$@" ;;
    clean)   cmd_clean "$@" ;;
    info)    cmd_info ;;
    help|-h|--help) cmd_help ;;
    *)
        err "unknown command: $command"
        cmd_help
        exit 1
        ;;
esac
