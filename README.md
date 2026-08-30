# Learned Kernel

**Learned Kernel** is a prototype architecture that elevates Linux kernel policies into first-class, introspectable, verifiable, and learnable objects.

The system is derived from [Self-Compiler](https://github.com/dwan-ith/self-compiler): rather than letting an AI freely modify the kernel, a structured representation of kernel state is exposed to a learned component that proposes *bounded* changes. The deterministic kernel remains the correctness authority; the learned system is never trusted with that role.

> **Core:** Can an operating system expose its governing policies the way a compiler exposes its optimisation passes?

---

## Architecture

```
Kernel subsystems (scheduler / VM / network)
              │
         eBPF tracepoints
              │
    ┌─────────▼──────────┐
    │  KIR  (JSON schema) │  ← structured observation
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   Policy (ABI)      │  ← learned or heuristic
    │  observation_schema │
    │  action_schema      │
    │  decide(KIR)→Action │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │ Deterministic       │
    │ Validator           │  ← safety authority
    │  per-field bounds   │
    │  joint constraints  │
    └──────┬──────┬───────┘
         REJECT  APPROVE
                  │
         ┌────────▼────────┐
         │  KernelActuator  │  ← sysctl translation
         └─────────────────┘
                  │
             Provenance log
             (JSONL + hash)
                  │
           Reward calculator
                  │
         Drift manager →
         Offline trainer →
         new Policy artifact
```

---

## Component Reference

### `learned_kernel/policy/schemas.py` — KIR and Action ABI

The **Kernel Intermediate Representation** (KIR) is a Pydantic schema covering three subsystems:

| Subsystem | Key fields |
|-----------|-----------|
| Scheduler | `cpus[id].{utilization, runnable_tasks, irq_time}`, `total_context_switches`, `avg_latency_ms` |
| Memory    | `pressure`, `reclaim_rate_mb_s`, `cache_hit_rate` |
| Network   | `rx_queue_len`, `tx_queue_len`, `tcp_retransmits` |

Action schemas (`SchedulerAction`, `VMAction`, `NetAction`) enumerate every tunable parameter and map directly to sysctl keys.

### `learned_kernel/policy/core.py` — Policy ABI

`PolicyBase` requires every concrete policy to declare:
- **`observation_schema`** — which KIR fields it reads (dot-notation list)
- **`action_schema`** — which kernel parameters it may write
- **`decide(kstate)`** — the KIR → bounded PolicyAction mapping

This makes every policy introspectable without executing it, analogous to a GCC pass exposing its gate condition.

Concrete policies shipped:

| Class | Description |
|-------|-------------|
| `LinuxDefaultPolicy` | No-op baseline (standard CFS) |
| `HeuristicLatencyPolicy` | Threshold heuristic on mean CPU utilisation |
| `JointHeuristicPolicy` | Simultaneous scheduler + VM + network control |
| `LearnedLinearPolicy` | Offline-trained 6-feature linear map |

### `learned_kernel/validator/core.py` — Deterministic Validator

The validator enforces two tiers of constraint:

1. **Per-field bounds** — e.g. `target_latency_us ∈ [1000, 24000]`, `swappiness ∈ [0, 100]`, `dirty_background_ratio ≤ dirty_ratio`
2. **Joint cross-domain constraints** — e.g. maximum network queue depth is blocked when `memory.pressure > 0.8` (OOM-panic prevention)

The validator is called with the full KIR state, so constraints can be stateful with respect to observed system conditions.

### `learned_kernel/trainer/reward.py` — Normalised Reward Model

```
R = β·T̄ - α·L̄ - γ·Ē
```

All three terms are dimensionless and in [0, 1], so α, β, γ are directly interpretable as relative weights without dimensional confusion:

| Term | Normalisation |
|------|-------------|
| T̄ — throughput | context-switch delta / 10 000 s⁻¹ |
| L̄ — latency | avg_latency_ms / 20 ms |
| Ē — energy | mean CPU utilisation (already in [0, 1]) |

### `learned_kernel/trainer/rl_trainer.py` — Offline Trainer

`OfflineTrainer` performs **coordinate-wise numerical gradient ascent** over the 6-dimensional weight space of `LearnedLinearPolicy`:

```
for each round:
  for each weight dimension d:
    estimate ∂R/∂w[d] via (R(w[d]+ε) − R(w[d]−ε)) / 2ε
    w[d] += lr · grad
  hill-climb bias term
```

Key design properties:
- Runs entirely **outside the kernel hot-path**
- Invalid actions are **penalised (−50)**, not skipped, so the optimiser learns to stay within the validator's safety boundary
- No external ML library required

`LearnedLinearPolicy` reads a **6-feature vector** extracted from the full KIR:

```
f = [mean_util, runnable_norm, lag_norm, mem_pressure, net_queue_norm, irq_time]
lat_us = clip(w · f + bias,  1 000,  24 000)
```

### `learned_kernel/runtime/adaptation.py` — Drift Detection

`WorkloadDriftManager` maintains two rolling windows of equal size:
- **reference window** (earlier half)
- **current window** (recent half)

Drift is declared by a **Welch t-test** (`t < −t_threshold`), which requires degradation to be statistically significant relative to local variance. A plain scalar mean cannot distinguish a noisy-but-stable signal from genuine regime change.

### `learned_kernel/telemetry/provenance.py` — Audit Trail

Every decision is logged to `learned_kernel_audit.jsonl`:

```jsonc
{
  "timestamp": 1724071658.3,
  "kernel_state_hash": "a3f1b2c9...",   // SHA-256 of raw KIR JSON
  "policy_id": "learned-linear-v2",
  "policy_version": "2.0",
  "action_proposed": { "scheduler": { "target_latency_us": 9500 } },
  "validator_approved": true,
  "observed_reward": 0.312,
  "hardware": "generic-x86_64-sim"
}
```

---

## Running the Prototype

```bash
pip install pydantic
# eBPF telemetry additionally requires BCC/libbpf on a real Linux host.
# On Windows/macOS the runtime falls back to simulation mode automatically.

# Single control loop (3 cycles)
python -m learned_kernel.runtime.loop

# Offline RL training pipeline
python simulate_and_train.py

# Joint multi-subsystem control + cross-domain validator demo
python multi_subsystem_demo.py

# Online drift detection + policy hot-swap + provenance audit
python online_demo.py
```

---

## Limitations and Known Gaps

| Limitation | Status |
|------------|--------|
| eBPF metrics are mock-filled in `monitors.py` | Requires a real Linux host with BCC root access |
| `LearnedLinearPolicy` is a linear model with a 1-D action space | Sufficient for a prototype; a neural policy or decision-tree is the next step |
| Reward jitter and fairness (J, F) terms are architectural stubs | Need per-task latency histograms not yet collected by the monitor |
| `offline_trainer.train()` is synchronous in the demo | Production use would run it in a separate process/thread |
| No held-out workload evaluation | Critical for claiming generalisation; currently the trainer and evaluator see the same trajectories |

---

## Relation to Prior Work

Learned Kernel is intentionally distinct from existing ML-for-systems approaches:

| Project | Approach | Learned Kernel difference |
|---------|----------|----------------------|
| AutoOS | LLM explores Linux config space | Policy is a learnable object with explicit ABI contracts, not a config string |
| OS-R1 | RL agent tunes kernel parameters | Joint cross-domain constraints; validator owns correctness, not the policy |
| KernelX | Safe runtime tuning of performance constants | Adds a structured state representation (KIR), offline learning, and drift detection |
| LDOS | Learned policies as first-class OS mechanisms | Closest in spirit; Learned Kernel adds explicit observation/action schemas and cryptographic provenance |

---

> **Can an operating system expose its policies the way a compiler exposes its optimisation passes — with explicit state contracts, bounded actions, deterministic acceptance authority, and measurable feedback?**

Learned Kernel is a prototype existence proof that the answer is *yes* for a scheduler vertical slice, and a blueprint for extending that to VM, network, and power subsystems.
