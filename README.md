# 🚀  EOS-Bench
A unified and extensible benchmark platform for Earth Observation (EO) satellite scheduling.
---

This project provides:

* 📦 A scalable scenario generation framework (1–1000 satellites, 1h–168h horizon)
* ⚙️ A unified constraint & evaluation model
* 🧠 Multiple scheduling paradigms:

  * MIP (Exact Optimization)
  * Heuristics
  * Meta-heuristics (SA / GA / ACO)
  * PPO Reinforcement Learning
* 📊 Standardized performance metrics
* 🌍 3D dynamic visualization via Cesium

---

## 🔥 Key Features

* Unified scenario modeling and JSON format
* Fair cross-algorithm benchmarking
* Modular algorithm integration
* Multi-objective optimization support
* RL checkpoint saving & resume training
* Automated CZML generation and 3D visualization

---

## 📂 Project Structure

```
Satellite_Benchmark/
│
├── algorithms/          # Scheduling algorithms (MIP / Heuristic / Meta / PPO)
├── core/                # Scenario modeling & visibility computation
├── schedulers/          # Constraint model, scheduling engine, evaluation
├── draw/                # Cesium 3D visualization tools
├── input/               # Constellation & target data
├── output/              # Generated scenarios, schedules, RL models
│
├── main_generate.py     # Generate benchmark scenarios
├── main_scheduler.py    # Run benchmark & train/test RL
└── main_draw.py         # Launch 3D visualization
```

---

## 📊 Evaluation Metrics

All algorithms are evaluated using unified metrics:

| Metric | Description          |
| ------ | -------------------- |
| TP     | Task Profit          |
| TCR    | Task Completion Rate |
| BD     | Balance Degree       |
| TM     | Timeliness Metric    |
| RT     | Runtime              |

---

## 🛠 Installation

### 1️⃣ Install Java (Required for Orekit)

```bash
sudo apt update
sudo apt install -y openjdk-17-jdk
```

### 2️⃣ Install Python Dependencies

Python 3.10 recommended.

```bash
pip install "orekit-jpype[jdk4py]" "jpype1==1.5.2"
pip install numpy pandas matplotlib scipy
pip install torch
```

---

## 🚀 Quick Start

### Step 1 — Generate Benchmark Scenarios

```bash
python main_generate.py
```

Scenarios are saved in:

```
output/Scenario_*.json
```

---

### Step 2 — Run Benchmark / Train RL

```bash
python main_scheduler.py
```

Outputs:

* Schedule results → `output/schedules/`
* RL models → `output/models/`
* Performance logs → `runlog_*.txt`

---

### Step 3 — 3D Visualization

Edit scenario & schedule names in `main_draw.py`, then run:

```bash
python main_draw.py
```

A browser window will open with dynamic 3D visualization.

---

## 🧠 Reinforcement Learning (PPO)

The PPO module supports:

* Multi-objective weighted training
* Periodic checkpoint saving
* Automatic resume training
* Independent train/test mode
* Scenario resampling control (every N episodes)

Saved models include:

* Policy weights
* Optimizer state
* Episode index
* RNG states (for reproducibility)

---

## ➕ Adding New Algorithms

1. Implement your algorithm in `algorithms/`
2. Follow the unified interface
3. Register in `factory.py`
4. Add to `algo_specs` in `main_scheduler.py`

It will automatically integrate into batch benchmarking.

---

## 📈 Designed For

* Algorithm comparison studies
* Scalability analysis
* RL vs classical optimization research
* Reproducible scheduling experiments
* Engineering validation

---

## 📌 Notes

* Visibility computation is powered by **Orekit**
* All algorithms share a unified constraint model
* Designed for large-scale EO constellation scheduling research

---


