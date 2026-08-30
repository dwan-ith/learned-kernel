import io
import contextlib

from learned_kernel.trainer.rl_trainer import OfflineTrainer, LearnedLinearPolicy, extract_features
from learned_kernel.tests.test_trainer import synthetic_history
from learned_kernel.policy.schemas import CPUMetrics, SchedulerState
from learned_kernel.policy.schemas import KernelIntermediateRepresentation, PolicyAction, SchedulerAction
from learned_kernel.simulator.env import KernelSimulator
from learned_kernel.trainer.reward import RewardCalculator

tr = OfflineTrainer(rounds=20, seed=5)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pol = tr.train([synthetic_history(80)], verbose=True)
print(pol)
print("report:", tr.last_report.summary())

def kir_at(util):
    return KernelIntermediateRepresentation(
        timestamp=100.0,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=util, runnable_tasks=int(util * 10), irq_time=0.01)},
            total_context_switches=0, avg_latency_ms=util * 12),
    )

for u in (0.2, 0.5, 0.9):
    a = pol.decide(kir_at(u))
    feats = extract_features(kir_at(u))
    raw = sum(w * f for w, f in zip(pol.weights, feats)) + pol.bias
    print(f"util={u}: raw={raw:.0f}us -> action T={a.scheduler.target_latency_us}us")

# What does each candidate achieve on a busy workload rollout?
rc = RewardCalculator()
for label, T in [("learned@u=0.9", pol.decide(kir_at(0.9)).scheduler.target_latency_us),
                 ("grid-best", 3000), ("default", 6000)]:
    e = KernelSimulator(seed=99)
    ks = e.current_kir()
    tot = 0.0
    for _ in range(60):
        prev = ks
        ks = e.step(PolicyAction(policy_id="g", scheduler=SchedulerAction(target_latency_us=T)))
        tot += rc.calculate_step_reward(prev, ks)
    print(f"{label:>14} T={T}us busy-rollout reward={tot:7.2f}")
