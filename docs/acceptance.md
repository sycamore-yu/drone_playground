# Acceptance evidence collector

本轮 PPO 稳定性任务按用户在 2026-10-15 的澄清执行：已有可复现的随机初始化
训练结果可以复用，不要求全部在当前源码版本重训；不同算法可以采用不同奖励，
任务目标与成功判据保持一致。用户已选定轻量筛查→一次完整验证→独立冻结测试，
每个算法仍须有三个独立训练种子；PPO Navigation 还须完成约定的续训能力保持检查。
跨算法 Actor/配置差异作为对照信息单独报告，不再将其他算法的缺项算作本算法
不收敛。历史三次连续达标和一次性候选冻结审计可复用，保留各自原始协议与成本；
不要求旧模型补做筛查或补训。当前为 14/18 格通过，缺 PPO Navigation 四格。

当前工作清单与复用结果见
[验收清单](../artifacts/verification/ppo-stability-20261015/acceptance-manifest.json) 和
[本轮矩阵](../artifacts/verification/ppo-stability-20261015/acceptance-protocol2.md)。
清单的 `reuse_reports` 将选定运行映射到已有完整核对报告；收集器复用其中的
逐回合、选模和模型完整性检查，并确认保存的配置、运行记录与冻结批次选择没有
变化。该模式不重复读取模型或逐回合 CSV，也不重新计算它们的哈希。只应指定
仍然有效、原始产物未修改的核对报告；旧格式报告可读，不恢复旧格式运行时加载。

下文的 2026-10-08 初始化验收记录不能作为随机初始化矩阵的收敛证明。
2026-10-10 起新项目只采用 [ADR-0017](adr/0017-eight-acceptance-gates.md)
定义的 S1–S4 与 L1–L4 **八项独立验收**。旧 C1–C6 的原始随机种子、
100/25 回合阈值与三阶段测试规则已全部纳入 **L3**；旧 S6 的
EGO/SUPER 原生运行范围已纳入 **S3**。本文原来的 20 项标签只属于历史报告。
**当前 v0.2 的选模标准改为 [ADR-0015](adr/0015-three-stage-learning-acceptance.md)：**
六个主场景各 8 回合轻量筛查（不能当成 C4 达标），通过后只做一次 25 回合／
主场景的完整 checkpoint validation，再对三个独立随机初始化种子分别运行
冻结 benchmark。C2/C3/C4 的任务质量门槛不变；旧初始化训练的 18/18 成绩不能
作为随机初始化证明。选模协议与初始化来源分别保留，不改写历史运行。

[Current 18-cell table](../results/acceptance/current.md) ·
[Full evidence JSON](../results/acceptance/current.json) ·
[Frozen CSV audit](../results/acceptance/final-audit.json) ·
[Collector](../tools/report_acceptance.py) · [Tests](../tests/test_acceptance_report.py)

All **18 cells passed** in the 2026-10-08 final collection: 36 explicitly selected
training runs and 5,400 independent frozen benchmark episodes. Each selected run
has its own three consecutive passing checkpoint evaluations and frozen evaluation.
The chosen LiDAR recipes are `navigation_lidar_initialized` for APG,
`navigation_lidar_finetune` for SHAC, and `navigation_lidar_interleaved` for PPO.
Earlier recipes and their failures remain recorded in [experiments](experiments.md),
outside this cohort. A mixed Navigation run contributes separate static and dynamic
cells; the CSV audit counts each episode only once. No diagnostic run substitutes
for a formal seed.

Tracking and Racing each record 900/900 successful frozen episodes; Depth records
1,800/1,800 across all eight scenes. LiDAR main scenes record 1,344/1,350, meeting
the per-scene, per-seed threshold in every case. Its S06 records 225/225 and D06
42/225, with individual D06 runs ranging from 0/25 to 15/25. These extension
outcomes are reported in full and do not enter the unchanged C4/C5 main-scene gate.

## Run the collector

The positional argument is a JSON manifest with an explicit `runs` list. `root` is
relative to the manifest's directory (or absolute); every run path and every relative
path recorded inside a run resolves against that root. `seeds` defaults to `[0,1,2]`
and must contain exactly three distinct nonnegative integers. Paths may be missing.
The collector retains them in `runs`, with unavailable identity and reasons; it does
not infer task or seed from a directory name. The corresponding matrix slots stay
missing until a readable `config.yaml` identifies a selected run.

```bash
pixi run --locked env JAX_PLATFORMS=cpu python tools/report_acceptance.py /path/to/manifest.json \
  --output-prefix results/acceptance/current
pixi run --locked env JAX_PLATFORMS=cpu pytest -q tests/test_acceptance_report.py
pixi run --locked env JAX_PLATFORMS=cpu ruff check tools/report_acceptance.py tests/test_acceptance_report.py
```

The collector writes `<prefix>.json` and `<prefix>.md`. Exit zero means the report
was generated, **not** that acceptance passed; consumers must read `all_passed` and
the individual cells. Duplicate paths, duplicate `(task,sensor,algorithm,seed)`
groups, and seeds outside the manifest cohort are errors. It reads the directories
named in the manifest, their checkpoint evaluation history, and explicitly recorded
initialization sources. It never searches other runs for a successful replacement.
It does not train, import project simulation code, query the GPU, or alter run records.

An optional `benchmark_runs` object maps a chosen training-run path to a separate
frozen-evaluation run, with both paths resolved against `root`. This permits a new
test cohort without rewriting the original training record. The collector requires
the selected checkpoint path and loaded payload SHA-256 recorded by the evaluation
CLI, matching task/actor/observation/physical inputs, a completed run with valid
config checksum, and the declared test seed base in the report and episode rows.
Older standalone evaluations without the recorded loaded-payload identity cannot
serve as verified replacements. The opt-in `benchmark.frozen_seed_base` is excluded
from three-seed training-recipe equality because it only reserves final-test cases;
the collector still verifies the actual evaluation cohort independently.
Checkpoint seed validation follows `benchmark.fixed_checkpoint_seeds`: fixed
cohorts keep their case seeds while evaluation indices still count consecutive
checks. Frozen benchmarks retain their separate declared test seeds.

Chosen manifest, embedded here to avoid adding another maintained tool file:

```json
{
  "root": ".",
  "seeds": [0, 1, 2],
  "runs": [
    "results/acceptance_tracking_ppo_s0",
    "results/acceptance_tracking_ppo_s1",
    "results/acceptance_tracking_ppo_s2",
    "results/acceptance_tracking_apg_s0",
    "results/acceptance_tracking_apg_s1",
    "results/acceptance_tracking_apg_s2",
    "results/acceptance_tracking_shac_s0",
    "results/acceptance_tracking_shac_s1",
    "results/acceptance_tracking_shac_s2",
    "results/acceptance_racing_initialized_ppo_s0",
    "results/acceptance_racing_initialized_ppo_s1",
    "results/acceptance_racing_initialized_ppo_s2",
    "results/acceptance_racing_initialized_apg_s0",
    "results/acceptance_racing_initialized_apg_s1",
    "results/acceptance_racing_initialized_apg_s2",
    "results/acceptance_racing_initialized_shac_s0",
    "results/acceptance_racing_initialized_shac_s1",
    "results/acceptance_racing_initialized_shac_s2",
    "results/acceptance_depth_initialized_ppo_s0",
    "results/acceptance_depth_initialized_ppo_s1",
    "results/acceptance_depth_initialized_ppo_s2",
    "results/acceptance_depth_initialized_apg_s0",
    "results/acceptance_depth_initialized_apg_s1",
    "results/acceptance_depth_initialized_apg_s2",
    "results/acceptance_depth_initialized_shac_s0",
    "results/acceptance_depth_initialized_shac_s1",
    "results/acceptance_depth_initialized_shac_s2",
    "results/acceptance_lidar_interleaved_ppo_s0",
    "results/acceptance_lidar_interleaved_ppo_s1",
    "results/lidar_interleaved_ppo_s2",
    "results/acceptance_lidar_initialized_apg_s0",
    "results/acceptance_lidar_initialized_apg_s1",
    "results/acceptance_lidar_initialized_apg_s2",
    "results/acceptance_lidar_finetune_shac_s0",
    "results/acceptance_lidar_finetune_shac_s1",
    "results/acceptance_lidar_finetune_shac_s2"
  ]
}
```

From the repository root, generate the current report directly from that explicit list:

```bash
pixi run --locked env JAX_PLATFORMS=cpu python - <<'PY'
import json
import subprocess
import sys
import tempfile
from pathlib import Path

text = Path("docs/acceptance.md").read_text()
manifest = json.loads(text.split("```json\n", 1)[1].split("```", 1)[0])
manifest["root"] = str(Path.cwd())
with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as stream:
    json.dump(manifest, stream)
    stream.flush()
    subprocess.run([sys.executable, "tools/report_acceptance.py", stream.name,
                    "--output-prefix", "results/acceptance/current"], check=True)
PY
```

The JSON report embeds the entire manifest and its checksum, so the temporary
manifest is not needed to audit the snapshot. To change the cohort, explicitly list
all chosen paths in the manifest, including failed or ongoing seeds.
The collector follows only the chosen paths and their recorded initialization
links. Regenerate as the selected runs complete; it does not discover additional runs.

## What is checked

The thresholds are the unchanged [spec C1–C6](spec.md#63-收敛标准-c1c6):

| Criterion | Evidence required |
|---|---|
| C1 | Three distinct training seeds with one chosen recipe per algorithm/task/sensor; separate selection and evaluation per seed. |
| C2 | Exactly 100 Tracking CSV episodes, at least 95 successes, mean successful-episode position RMSE at most 0.25 m. |
| C3 | Exactly 100 Racing CSV episodes, at least 90 successes; each success records five gates in order `1,2,3,4,2` and positive finite completion time. |
| C4 | Exactly 25 episodes per main Navigation scene, at least 23 successes in each; S01/S02/S03 and D01/D02/D03 graded separately, S06/D06 fully reported. |
| C5 | Recorded first full validation after screening and independent frozen benchmark. Historical consecutive triples and completed one-time frozen audits retain their original selection rule. Selected model update and disjoint case seeds must agree. |
| C6 | Actual logged updates, interaction counters, elapsed time, throughput, GPU memory and utilization samples, and initialization cost attribution. No training budget cap is introduced. |

CSV rows determine the denominator and outcome counts. Unknown/nonterminal outcomes,
wrong scenes, duplicate `(initialization_seed,world_index)` cases, missing/nonfinite
required metrics, and disagreements with report counts cannot pass. A smaller CSV
is incomplete even when `run.json` or `report.json` says accepted. Complete CSVs
below the quality threshold fail. Run status is preserved separately from verification.
An ongoing run is not declared complete from its status or a partial evaluation.
Runs whose recorded status is `running` remain `incomplete` while initialization
or evaluation evidence is still being written. Individual failed checks remain
visible; this status rule never makes a run pass or changes completed-run checks.

C5 reads every persisted `checkpoints/step-*/report.json` directory, including failed
evaluations and directories referenced by events. Missing evaluations break the
verifiable history. Checkpoint seeds must lie in `[1000000,2000000)`, benchmark
seeds in `[2000000,infinity)`, and actual CSV seeds must match the saved seed formula
and be disjoint across those partitions. With manifest `selection=first_full_validation_pass`,
full Navigation selection requires six main scenes; frozen evaluation still records all
eight. Static and dynamic benchmark results are graded separately. Historical complete
triples remain accepted without rewriting their selected checkpoint.

The collector reads v2 ZIP evidence and v3 Orbax model directories. Previously audited
runs use `reuse_reports`, without repeated model/CSV checks. For a newly collected v2
archive it verifies the payload checksum; for v3 it reads saved metadata and the completed
Actor/training save markers, without rehashing or restoring parameter arrays. This records
save completion, not a new byte-integrity or replay claim. Saved experiment and Actor kind
must match `config.yaml`. TensorBoard supplies v3 update/resource events using the existing
tensorboardX dependency. Archive SHA and payload SHA are distinct fields. Per-seed metrics and unavailable reasons,
per-scene outcomes and successful flight times, checkpoint history, configurations,
versions and source identities are retained in JSON.

The common Actor **configuration** check compares saved task/reference, sensor/input
settings, Actor kind and physical control settings across the three algorithms and
three seeds. It excludes batch size/device and algorithm-specific learning options;
full learning recipes are retained and must match within each three-seed group. Under
the single-full protocol, serialization, historical selection count and stopping policy
are disclosed but excluded from optimization-recipe equality. Rewards, observations,
physics, sampling, learning rates and other optimizer options remain in that comparison.
The expected architectures remain State MLP `[256,128]` LayerNorm/ELU, Depth
CNN `32/64/128` with GRU192, and LiDAR PointNet `64/128/1024` with GRU192.
Source hashes may differ across training sessions.
Configuration equality does not establish historical implementation equivalence;
the collector does not deserialize/run networks or independently replay physics.

Saved dynamics (`first_principles`), drone (`cf21B_500`), physics rate (500 Hz), and
task duration (Tracking 20 s, Racing 60 s, Navigation 300 s) are checked against the
chosen base contract. No physics or task thresholds are changed. The Navigation
goal radius remains specified as 0.5 m, body radius 0.07 m, nominal speed limit
20 m/s, with the world bounds/reset/event ordering in the spec. These hardcoded
historical constants are not established for records that omit them: independent
verification is `null` with a reason rather than a claim based on today's source
tree. A serialized `task_contract` in an evaluation report is retained verbatim
when present; its presence is not an independent verification of historical physics.

## Costs and evidence boundaries

`cost` gives recorded final counters, or explicitly labeled last-update lower bounds
for incomplete records. The number of logged events is not the number of updates;
logging is sparse. Elapsed training/session times include the work covered by those
existing timers, including compilation and in-loop evaluation; they are not isolated
kernel execution time. No missing values are replaced by zero estimates.
`cost_to_selection` separately records the selected checkpoint's counters and its
evaluation event's elapsed session time; resumed-session time is not silently treated
as cumulative time to target.

The following initialization lineage describes **historical v1 runs**, not archives loadable
by the current v2 runtime. Frozen results are retained as historical evidence.

Racing initialization is traced to
`results/apg_racing_diagnostic_s0/checkpoints/update-00000400.policy.zip`.
Its source training cost is reported **once** in `shared_initialization_costs`. The source run
uses Python **3.13** / JAX **0.9**, while the formal cohort uses Python **3.12** /
JAX **0.11**.

Depth initialization follows the full recorded chain:

1. Each chosen formal Depth run initializes from
   `results/apg_depth_mixed_diagnostic_s0/checkpoints/update-00000400.policy.zip`.
2. That mixed source initializes from
   `results/apg_depth_altitude_diagnostic_s0/checkpoints/update-00000100.policy.zip`.
3. The altitude source initializes from
   `results/reference_initialization/gain_0.167.policy.zip`, whose provenance records
   the historical weights archive, checksum, action scaling and transfer limitations.

The selected formal LiDAR runs follow explicit initialization links through
`results/apg_lidar_clearance_diagnostic_s0`, then
`results/apg_lidar_transfer_mixed_diagnostic_s0`, to
`results/reference_initialization/lidar.policy.zip`. That archive records the
historical weights/checksum and approximate transfer, including the omitted action
bias. These sources are collected because explicitly chosen runs reference them.

Each run's `initialization.shared_cost_id` identifies its direct source;
`initialization.cost_lineage` lists the entire ancestry in order. Each source entry
links to its own `initialization.shared_cost_id`, and `used_by` lists all formal
consumers, including indirect consumers. Recorded costs are deduplicated once per
source run across the cohort, irrespective of how many seeds or algorithms reuse
it. Diagnostic sources remain outside the formal seed cohort. Source costs cover
their full recorded runs, including work after the consumed checkpoint; no prefix
cost is invented. Terminal reference archives use their archive paths as identifiers,
retain archive and payload checksums and full provenance, and report historical
training cost as **unavailable**, with `null` metrics and reasons. A reference
archive's null `initial_checkpoint` does not establish training from scratch.

Formal initialized seed costs are additional training. They exclude source training
and unknown historical cost, so neither their sum nor the available source costs
establish a complete training-from-scratch cost. No historical cost is inferred as zero.

C6 memory/utilization are sampled for the **whole GPU with concurrent jobs**.
Their maxima are sample maxima, not measured process peaks. No isolated runtime,
GPU-seconds, energy, or speedup claim follows from these values. Those unavailable
isolated metrics are `null` with a reason. C6 here checks evidence availability;
it does not certify optimal throughput or utilization.

The selected cohort records 403,916,800 new training interactions. This excludes
shared initialization, unknown historical pretraining, and discarded trials.
The linked table reads actual files at collection time, preserving missing,
incomplete or failing seeds when present. Concurrent writes can yield an incomplete
snapshot; the final passing collection was generated after all 36 selected runs ended.

## Native S6 remains separate

The [native implementation and evidence workflow](ros_planner.md) already exists.
This small collector does not accept native paths or grade S6. S6 requires genuine
C++ gRPC/ROS1 EGO and SUPER decision/control trajectories, **each** with at least
10 static and 10 dynamic episodes covering the main scenes, and at least one safe
arrival in **each** of S01/S02/S03. Dynamic solving/control/collision/timeout outcomes
must be recorded; there is no dynamic success-rate threshold. Native results never
substitute for Learning C4's per-scene 23/25 requirement.
