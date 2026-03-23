# 🚀 EOS-Bench
**A unified and extensible benchmark platform for Earth Observation (EO) satellite scheduling**

[Dataset on Hugging Face](https://huggingface.co/datasets/Ethan19YQ/EOS-Bench/tree/main)

EOS-Bench is a research-oriented benchmark platform for **Earth observation satellite scheduling**. It is designed to support **reproducible algorithm comparison**, **large-scale scenario generation**, and **end-to-end evaluation** across classical optimization methods and learning-based schedulers.

The platform covers the full workflow of EO scheduling experiments:

- realistic scenario generation based on **Orekit**
- unified constraint modeling for EO scheduling
- standardized multi-objective evaluation
- benchmarking across exact, heuristic, meta-heuristic, and RL-based methods
- dynamic **3D visualization** of scenarios and schedules through **CZML + Cesium**

EOS-Bench is particularly suitable for studies on:

- large-scale constellation scheduling
- agile / non-agile EO mission planning
- scalability analysis of scheduling algorithms
- comparison between RL and classical optimization methods
- reproducible EO scheduling benchmarks for academic research

---

## 📦 Dataset

The accompanying **EOS-Bench dataset** serves as the data companion of the repository.

In short:

- the **repository** provides the benchmark **engine**
- the **dataset** provides the benchmark **data**

Depending on the release version, the dataset may include:

- constellation configuration files
- city / target definition files
- benchmark scenario assets

This separation makes EOS-Bench easier to reproduce, extend, and share across different experimental settings.

---

## 🔥 Key Features

### 1. Scalable scenario generation
Generate EO scheduling scenarios for constellations ranging from **1 to 1000+ satellites** and planning horizons from **1 hour to 168 hours**.

### 2. High-fidelity constraint modeling
Supports both **Agile** and **Non-Agile** satellites, including:

- visibility constraints
- attitude transition time modeling
- per-orbit storage constraints
- per-orbit power constraints
- communication / downlink constraints

### 3. Unified benchmark interface
All algorithms are evaluated under a **shared constraint model** and a **standardized metric system**, enabling fair and reproducible comparison.

### 4. Multiple scheduling paradigms
EOS-Bench integrates representative methods from several algorithm families:

- **MIP** (exact optimization)
- **Heuristics**
- **Meta-heuristics**: SA / GA / ACO
- **Reinforcement Learning**: PPO

### 5. Multi-objective evaluation
The benchmark supports unified evaluation for:

- **TP**: Task Profit
- **TCR**: Task Completion Rate
- **TM**: Timeliness Metric
- **BD**: Balance Degree
- **RT**: Runtime
- **RV**: Robustness Variance (optional)

### 6. Interactive 3D visualization
Generated scenarios and schedules can be visualized through **CZML + Cesium**, enabling dynamic inspection of:

- orbital motion
- target locations
- observation execution
- schedule evolution over time

---

## 📂 Project Structure

```text
EOS-Bench/
│
├── algorithms/                  # Scheduling algorithms and factory
│   ├── mip.py                   # MIP-based scheduler
│   ├── heuristics.py            # Heuristic schedulers
│   ├── meta_sa.py               # Simulated annealing
│   ├── meta_ga.py               # Genetic algorithm
│   ├── meta_aco.py              # Ant colony optimization
│   ├── candidate_pool.py        # Candidate assignment generation
│   ├── objectives.py            # Multi-objective scoring
│   ├── random_utils.py          # Random seed utilities
│   └── ppo/                     # PPO policy, agent, and learning modules
│
├── core/                        # Static domain models and scenario generation
│   ├── models.py
│   └── scenario.py
│
├── schedulers/                  # Constraint model, engine, loader, metrics, RL env
│   ├── scenario_loader.py
│   ├── constraint_model.py
│   ├── engine.py
│   ├── evaluation_metrics.py
│   ├── balance_utils.py
│   ├── timeliness_utils.py
│   ├── rl_env.py
│   ├── rl_utils.py
│   └── rl_scenario_sampler.py
│
├── utils/                       # Orekit-based visibility computation
│   └── visibility.py
│
├── draw/                        # CZML and Cesium visualization utilities
│   ├── orekit_to_czml.py
│   ├── cesium_viewer.html
│   └── gantt_viewer.html
│
├── input/                       # Constellation definitions and target files
│   └── cities_data/
│
├── output/                      # Generated scenarios, schedules, models, and logs
│
├── main_generate.py             # Scenario generation entry
├── main_scheduler.py            # Scheduling / benchmarking / PPO train-test entry
└── main_draw.py                 # Visualization entry


## 📊 Evaluation Metrics

All algorithms are evaluated using the same metric definitions:

| Metric | Description                    |
| ------ | ------------------------------ |
| TP     | Task Profit                    |
| TCR    | Task Completion Rate           |
| TM     | Timeliness Metric              |
| BD     | Balance Degree                 |
| RT     | Runtime                        |
| RV     | Robustness Variance (optional) |

This unified evaluation layer makes EOS-Bench suitable for controlled and reproducible cross-method comparison.

---

## 🛠 Installation

### 1. Install Java

Orekit requires Java. OpenJDK 17 is recommended.

```bash
sudo apt update
sudo apt install -y openjdk-17-jdk
```

### 2. Install Python dependencies

Python **3.10+** is recommended.

```bash
pip install "orekit-jpype[jdk4py]" "jpype1==1.5.2"
pip install numpy pandas matplotlib scipy pulp
pip install torch
```

> `torch` is only required for PPO-based learning experiments.
> `pulp` is required for the MIP scheduler.

---

## 🚀 Quick Start

This section walks through a minimal end-to-end example:

* generate one EO scenario
* run one scheduling algorithm
* visualize the result in 3D

The example below uses:

* **20 satellites**
* **20 missions**
* **1-day horizon**
* **SA** as the scheduling algorithm
* **profit-only objective** (`TP`)

---

### Step 1. Generate a scenario

Open `main_generate.py` and set the configuration near the bottom as follows:

```python
if __name__ == "__main__":

    satellite_files = [
        "20_satellites",
    ]

    time_period_days_list = [1]

    missions_number = [
        (20,),
    ]

    ground_stations_dict = {}

    targets_file_name = None

    run_all_scenarios(
        satellite_files=satellite_files,
        time_period_days_list=time_period_days_list,
        missions_number=missions_number,
        ground_stations_dict=ground_stations_dict,
        targets_file_name=targets_file_name,
        max_workers=1,
    )
```

Then run:

```bash
python main_generate.py
```

This will generate a scenario JSON under:

```text
output/
```

A typical output filename is:

```text
output/Scenario_S1_Sats20_M20_T1.0d_dist1.json
```

It will also generate a summary file such as:

```text
output/scenario_summary_YYYYMMDD_HHMMSS.txt
```

#### Notes

* `targets_file_name = None` means **random mission generation**
* if you set `targets_file_name = "cities_01.json"` or a list such as `["cities_01", "cities_02"]`, targets will be loaded from `input/cities_data/`
* in **target-file mode**, the actual number of missions is determined by the target file, not by `missions_number`

---

### Step 2. Run a scheduling algorithm

Open `main_scheduler.py` and keep a simple benchmark configuration such as:

```python
def main():
    placement_mode = "earliest"
    downlink_duration_ratio = 1.0
    unassigned_penalty = 1000.0
    agility_profile = "Standard-Agility"
    non_agile_transition_s = 10.0

    rl_do_train = False
    rl_do_test = True

    algo_specs = [
        {"class_id": 3, "algo_name": "sa", "cfg_overrides": {}},
    ]

    objective_loop = [
        ObjectiveWeights(
            w_profit=1.0,
            w_completion=0.0,
            w_timeliness=0.0,
            w_balance=0.0,
        ),
    ]

    ...
```

Then run:

```bash
python main_scheduler.py
```

This will:

1. scan all `Scenario_*.json` files under `output/`
2. run the selected algorithm(s)
3. save schedule files and figures under:

```text
output/schedules/
```

A typical output filename is:

```text
output/schedules/scheduler_Scenario_S1_Sats20_M20_T1.0d_dist1_c3_sa_p1_c0_t0_b0.json
```

Additional outputs include:

* schedule JSON
* Gantt chart PNG
* batch run log, e.g. `runlog_batch_YYYYMMDD_HHMMSS.txt`

---

### Step 3. Visualize the result in 3D

Open `main_draw.py` and set:

```python
if __name__ == "__main__":
    scenario_file = "Scenario_S1_Sats20_M20_T1.0d_dist1"
    schedule_file = "scheduler_Scenario_S1_Sats20_M20_T1.0d_dist1_c3_sa_p1_c0_t0_b0"
    main_draw(scenario_file, schedule_file)
```

Then run:

```bash
python main_draw.py
```

This will:

1. generate `draw/orbit.czml`
2. start a local HTTP server
3. open the Cesium viewer in your browser

You can then inspect:

* satellite orbits
* target geometry
* observation execution
* schedule results over time

#### Visualization without a schedule file

The visualization entry also supports **scenario-only viewing**. If `schedule_file = None`, the viewer still opens normally and displays the orbital scene, but schedule-related panels remain empty.

Example:

```python
if __name__ == "__main__":
    scenario_file = "Scenario_S1_Sats20_M20_T1.0d_dist1"
    schedule_file = None
    main_draw(scenario_file, schedule_file)
```

---

## 🧪 Workflow Summary

A typical EOS-Bench experiment follows this pipeline:

```text
1. Configure main_generate.py
2. Run main_generate.py
3. Configure main_scheduler.py
4. Run main_scheduler.py
5. Configure main_draw.py
6. Run main_draw.py
```

---

## ⚙️ Scenario Generation Modes

EOS-Bench supports two main scenario generation modes.

### Random mode

Use:

```python
targets_file_name = None
```

This generates mission targets according to predefined random geographic distributions.

### Target-file mode

Use:

```python
targets_file_name = "cities_01.json"
```

or

```python
targets_file_name = ["cities_01", "cities_02"]
```

This loads targets from:

```text
input/cities_data/
```

This mode is useful for:

* fixed benchmark targets
* reproducible case studies
* city-specific scheduling experiments

---

## 🧠 Reinforcement Learning (PPO)

EOS-Bench includes a PPO-based scheduler for learning-based EO scheduling experiments.

The PPO module supports:

* train / test separation
* weighted multi-objective training
* checkpoint saving
* automatic resume
* scenario resampling every N episodes
* independent model files for different objective weights

Typical saved artifacts include:

* policy weights
* optimizer state
* episode index
* training return logs

### To enable PPO training

In `main_scheduler.py`, enable PPO in `algo_specs` and set:

```python
rl_do_train = True
```

For example:

```python
algo_specs = [
    {"class_id": 4, "algo_name": "ppo", "cfg_overrides": {}},
]
```

### To test a trained PPO model

Set:

```python
rl_do_test = True
test_model_paths = [
    MODELS_DIR / "ppo_model_p1_c0_t0_b0.pt"
]
```

---

## 🧩 Supported Algorithm Classes

EOS-Bench currently supports the following algorithm classes:

| Class ID | Type               | Examples                                                        |
| -------- | ------------------ | --------------------------------------------------------------- |
| 1        | Exact optimization | MIP                                                             |
| 2        | Heuristics         | completion-first, profit-first, balance-first, timeliness-first |
| 3        | Meta-heuristics    | SA, GA, ACO                                                     |
| 4        | Learning-based     | PPO                                                             |

This makes the platform suitable for both **classical OR benchmarking** and **modern AI scheduling studies**.

---

## 📁 Output Files

### Scenario generation outputs

Saved under:

```text
output/
```

Typical files:

* `Scenario_*.json`
* `scenario_summary_*.txt`

### Scheduling outputs

Saved under:

```text
output/schedules/
```

Typical files:

* `scheduler_*.json`
* `scheduler_*.png`
* `runlog_batch_*.txt`

### RL model outputs

Saved under:

```text
output/models/
```

Typical files:

* `ppo_model_*.pt`
* `*.meta.json`
* PPO training logs

### Visualization output

Saved under:

```text
draw/orbit.czml
```

---

## ➕ Adding New Algorithms

To add a new algorithm:

1. implement it in `algorithms/`
2. follow the existing unified scheduler interface
3. register it in the algorithm factory
4. add it to `algo_specs` in `main_scheduler.py`

Once registered, it can be integrated into the same benchmark workflow and evaluated with the same output pipeline.

---

## 📈 Designed For

EOS-Bench is designed for:

* algorithm comparison studies
* constellation scheduling scalability analysis
* EO benchmark construction
* reproducible scheduling experiments
* RL versus classical optimization studies
* engineering-oriented validation and visualization

---

## 📌 Notes

* visibility computation is powered by **Orekit**
* all algorithms share a **unified constraint and evaluation model**
* the current workflow is **script-config driven**
* users typically modify key variables in `main_generate.py`, `main_scheduler.py`, and `main_draw.py` before running experiments
* Cesium visualization depends on the local HTTP server started by `main_draw.py`

---

## 📚 Citation / Acknowledgement

If EOS-Bench helps your research, please consider citing the project repository and dataset in your experimental setup or benchmark section.

---

## 🌍 Related Resources

* **Codebase**: this repository
* **Dataset**: [EOS-Bench on Hugging Face](https://huggingface.co/datasets/Ethan19YQ/EOS-Bench/tree/main)

---

## ✅ Minimal Reproducible Example

For a first run, use the following minimal setup.

### `main_generate.py`

```python
satellite_files = ["20_satellites"]
time_period_days_list = [1]
missions_number = [(20,)]
targets_file_name = None
```

### `main_scheduler.py`

```python
algo_specs = [
    {"class_id": 3, "algo_name": "sa", "cfg_overrides": {}},
]

objective_loop = [
    ObjectiveWeights(1.0, 0.0, 0.0, 0.0),
]
```

### `main_draw.py`

```python
scenario_file = "Scenario_S1_Sats20_M20_T1.0d_dist1"
schedule_file = "scheduler_Scenario_S1_Sats20_M20_T1.0d_dist1_c3_sa_p1_c0_t0_b0"
```

Then run:

```bash
python main_generate.py
python main_scheduler.py
python main_draw.py
```

