# 🚀  EOS-Bench
**A unified and extensible benchmark platform for **Earth Observation (EO) satellite scheduling**.**

[Dataset](https://huggingface.co/datasets/Ethan19YQ/EOS-Bench/tree/main)

Designed to tackle the complexities of large-scale constellation management, this project bridges the gap between traditional operations research and modern AI scheduling techniques. It features a complete pipeline from scenario generation (powered by Orekit) and diverse algorithm benchmarking, to interactive 3D visualization.

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

* **Scalable Scenario Generation**: Generate realistic tasks and orbital mechanics for constellations ranging from 1 to 1000+ satellites over 1h–168h horizons. Supports randomized mission distributions or precise JSON-based city targets.
* **High-Fidelity Constraint Modeling**: Natively supports **Agile and Non-Agile** satellites with piecewise attitude transition time models (Δroll, Δpitch, Δyaw), power consumption, and data storage limits.
* **High-Performance Candidate Pool**: Employs truncated, diversified candidate sampling (Earliest, Center, Latest, Random) coupled with Early Accept and dynamic sampling techniques to keep complexity at O(T · K) for massive scales.
* **Unified Multi-Objective Optimization**: Standardized 0~1 normalized evaluation for Profit, Completion Rate, Timeliness, and Balance Degree.
* **Automated 3D Visualization**: One-click generation of CZML files for dynamic orbit and scheduling visualization via Cesium.

---

## 📂 Project Structure

```
Satellite_Benchmark/
│
├── algorithms/          # Algorithm implementations & Factory
│   ├── mip.py           # MILP formulation (PuLP)
│   ├── heuristics.py    # TP, TCR, TM, BD heuristic schedulers
│   ├── meta_*.py        # SA, GA, ACO implementations
│   ├── ppo/             # PPO Policy, Agent, and RL training loops
│   ├── candidate_pool.py# Diverse candidate window generation
│   └── objectives.py    # Multi-objective scoring models
│
├── core/                # Static domain models & Scenario generation
├── schedulers/          # Constraint models, transition utils, RL env
├── draw/                # Cesium 3D visualization tools
├── input/               # Constellation setups & target data (e.g., cities.json)
├── output/              # Generated scenarios, schedules, RL models, logs
│
├── main_generate.py     # Script to generate benchmark scenarios
├── main_scheduler.py    # Main entry to run benchmark algorithms & train RL
└── main_draw.py         # Launch local server for 3D visualization
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


