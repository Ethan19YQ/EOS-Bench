# ALNS and optional weather-mask experiments

## Post-planning cloud evaluation

Cloud masking is **off by default**. When enabled, the shared loader records a fixed subset of observation windows as blocked and retains every window for scheduling. After a plan is constructed, blocked observations contribute no realised TP or TCR. The assignments and their resource consumption remain in the plan. TCR retains the original task count as its denominator.

```bash
# Original, unmasked input
python main_scheduler.py --scenario_dir docs --algos alns --weights 1 0 0 0 --workers_other 1

# Evaluate the completed plans with a shared 25% cloud mask
python main_scheduler.py --scenario_dir docs --algos alns profit_first --weights 1 0 0 0 --weather_mask --cloud_fraction 0.25 --weather_seed 42 --workers_other 1
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--weather_mask` | Enable cloud outcome evaluation after planning | Off |
| `--cloud_fraction` | Fraction of observation windows marked as blocked, from 0 to 1 | 0.2 |
| `--weather_seed` | Seed fixing the mask, independent of solver randomness | 0 |
| `--train_weather_mask` | Enable cloud losses in the PPO episode-end reward | Off |
| `--train_cloud_fraction` | Fraction of training observation windows to block | 0.2 |
| `--train_weather_seed` | Training mask seed, separate from evaluation and search seeds | 0 |
| `--seed_offset` | Offset class 1/3 search seeds for repeated runs | 0 |
| `--alns_iterations` | ALNS iteration limit | 200 |
| `--alns_time_limit` | Optional soft ALNS search-time cap in seconds | None |

The script's no-argument configuration block offers the same weather controls:

```python
weather_config = {"enabled": True, "blocked_fraction": 0.25, "seed": 42}
rl_training_weather_config = {"enabled": True, "blocked_fraction": 0.25, "seed": 7}
```

The first dictionary can also be passed to `run_single_scheduling()`, `run_benchmark()`, or `load_scheduling_problem_from_json()` as `weather_config=weather_config`. The second is passed to `run_benchmark()` as `rl_training_weather_config`, or to `PPORunConfig` as `training_weather_config`.

The number of blocked windows is the requested fraction multiplied by the number of valid loaded windows, rounded to the nearest integer (half upwards). The result JSON reports both requested and actual fractions, blocked window IDs, the source-window fingerprint and the mask digest. For small instances the actual fraction can differ due to rounding. The `weather_mask.outcomes` object records planned and realised TP/TCR and identifies failed scheduled observations.

A hash ranking based on the instance's window inventory and weather seed fixes the selection. The same instance and weather configuration produce the same mask across algorithms and search runs. Increasing the fraction while keeping the weather seed fixed produces nested masks. Changing the weather seed requests another mask, although small or empty instances can yield identical selections.

The fraction describes **optical observation windows**, rather than geographical cloud coverage. Each window has a fixed clear or blocked outcome. All solvers use the full opportunity set, and cloud outcomes are assessed after planning.

PPO evaluation uses the same full-window input as other solvers. Training has a separate switch, fraction and seed. Both phases attach an outcome mask when an existing scenario is read. The training scenario sampler and generator are unchanged, and cached episodes reuse the same mask. Feasibility masks and policy state features retain their usual definitions. At the end of a training episode, cloud losses are deducted from the profit and completion reward components. Allocated resources and the balance and timeliness components are retained. Episodes spanning several rollout buffers continue until this final settlement.

For training with a 25% mask and evaluating with a separately seeded 25% mask:

```bash
python main_scheduler.py --algos ppo --rl_train --train_weather_mask --train_cloud_fraction 0.25 --train_weather_seed 7 --weather_mask --cloud_fraction 0.25 --weather_seed 42
```

Training still reads the existing `output` scenario library. `--scenario_dir` selects evaluation inputs and does not change that training-library setting. Omitting `--train_weather_mask` retains unmasked training, regardless of evaluation settings. Omitting `--weather_mask` retains unmasked evaluation, regardless of training settings.

The batch entry places weather-trained checkpoints under a distinct model name. Automatic model signatures include the training-weather settings and outcome schema. Resuming a checkpoint requires matching settings and semantics. Checkpoints from the earlier window-filtering implementation therefore use a separate training run. Model metadata records the training configuration. After a separate training invocation, select the intended checkpoint in the existing `test_model_paths` configuration for evaluation.

Scenario descriptors continue to characterise the full opportunity set used by the scheduler.

## Evaluate several cloud levels without replanning

After saving the ALNS plan in the first example, run:

```bash
python evaluate_cloud_schedule.py --scenario docs/Scenario_S1_Sats20_M500_T0.5d_dist0.json --schedule output/schedules/scheduler_Scenario_S1_Sats20_M500_T0.5d_dist0_c3_alns_p1_c0_t0_b0.json --fractions 0 0.25 0.5 0.75 --weather_seed 42 --output output/cloud_results/alns_cloud.json
```

This command reads the original scenario and the stored assignments. It evaluates that same plan at each fraction and writes a separate JSON report. The report includes planned and realised TP/TCR, blocked window IDs and failed scheduled observations. The plan, allocated resources and source files remain unchanged. Any saved solver plan can be evaluated through this entry point.

New schedule exports include `sat_window_id`. Older exports can be read when each assignment has a unique matching window in the source scenario. The command checks scenario IDs, planning horizons and window matches. Use a plan constructed from the full opportunity set. An existing cloud-results file is replaced only when `--overwrite` is supplied, and the source scenario and schedule are protected from replacement.

## ALNS implementation and budget

`algorithms/alns.py` implements a baseline with random, marginal-loss and related removal, greedy, regret-2 and randomised-greedy repair, adaptive operator weights, simulated-annealing acceptance and best-solution retention. It uses the common feasibility and objective models. It is not a reproduction of a particular published implementation.

Defaults include a 0.05–0.20 removal fraction, a 32-task repair shortlist, a 128-candidate pool for marginal-loss removal, four sampled placements per task, a 20-iteration weight-update segment, reaction factor 0.2, initial temperature 0.02 and cooling factor 0.99. Removal count is capped by the repair shortlist. Regret-2 compares sampled insertion alternatives, including non-insertion with zero gain. Rewards are 5 for a new best, 2 for improving the current solution, 1 for another accepted solution and 0 for rejection.

These defaults are starting values, not parameters tuned for a reported comparison. Configure other ALNS parameters through `cfg_overrides` in the single-run API or `alns_cfg_overrides` in the batch API. The optional time cap is soft and excludes initial greedy construction. Reported runtime includes initialisation and search, but not JSON loading, mask preparation or plotting. Fixed iteration budgets without a time cap support seeded replay. Equal iteration counts across different algorithms do not imply equal computational effort.

## Repeated comparison protocol

Choose a configuration on separate development instances, freeze it, and use the same test instances, constraint settings and objective definitions for every compared solver. Record the hardware, worker count, search budgets, weather seed and search seeds. Use one worker for serial timing comparisons unless a different execution protocol is explicitly intended.

The existing batch entry supports adding ALNS to the other methods:

```bash
python main_scheduler.py --scenario_dir output --algos sa ga aco alns --weights 1 0 0 0 --workers_other 1
```

To repeat a fixed-weather comparison with ten different class 1/3 search seeds, run from PowerShell:

```powershell
foreach ($runSeed in 0..9) {
    python main_scheduler.py --scenario_dir output --schedule_dir output/revision_cloud25 --algos sa ga aco alns --weights 1 0 0 0 --weather_mask --cloud_fraction 0.25 --weather_seed 42 --seed_offset $runSeed --workers_other 1
}
```

Keep `--weather_seed` fixed when measuring search variability. To study cloud levels or seeds for a fixed plan, use `evaluate_cloud_schedule.py` on the stored result. Its zero-fraction result provides the unmasked reference. This keeps plan variability separate from the effect of the cloud mask.

Without `--weights`, the batch entry runs TP, TCR and equal four-metric weights separately. This creates multiple ALNS configurations; count them explicitly if they are all included in a manuscript. MIP skipping rules and PPO checkpoint requirements remain unchanged.

## Outputs and scope of validation

The common output contains TP, TCR, BD, TM and runtime. When cloud evaluation is enabled, TP and TCR are realised values. BD, TM and runtime describe the complete plan. `solver_metadata.alns` stores the algorithm configuration, resolved search seed, iterations and operator statistics. `weather_mask` stores the cloud configuration and outcomes.

Enabled batch runs use a `cloud_post_f<ratio>_s<seed>` output subdirectory, separate from the earlier `weather_f...` window-filtering results. Nonzero search offsets use an additional `seed_offset_<offset>` directory. Plan filenames also include a short mask digest. Reusing the same configuration, offset and output directory overwrites the corresponding plans; choose a new `--schedule_dir` for a new parameter study.

Keep the individual-run outputs before calculating means and sample standard deviations or drawing TP/TCR box plots. Do not turn functional smoke tests into reported comparative experiments. Tests can be run with:

```bash
python -m pytest tests/test_alns.py tests/test_weather_mask.py -q
```
