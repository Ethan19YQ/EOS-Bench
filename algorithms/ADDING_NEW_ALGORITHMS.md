# ➕ Adding a New Scheduling Algorithm to EOS-Bench

This guide explains how to integrate a new scheduling algorithm into **EOS-Bench**.

EOS-Bench separates **algorithm-specific optimisation logic** from the shared **satellite scheduling environment, constraint model, evaluation metrics, logging, and visualisation pipeline**. Therefore, a new algorithm can be added without reimplementing the underlying scheduling constraints or evaluation framework.

The integration process consists of four main steps:

1. Implement the scheduler.
2. Register the scheduler in the algorithm factory.
3. Add the algorithm to the benchmark configuration.
4. Run the algorithm through the standard EOS-Bench workflow.

This document uses an **Adaptive Large Neighbourhood Search (ALNS)** scheduler as an example.

---

## 1. Unified Scheduler Interface

All scheduling algorithms in EOS-Bench follow the same interface. A custom scheduler should inherit from `BaseSchedulerAlgorithm` and implement the following `search()` method:

```python
def search(
    self,
    problem: SchedulingProblem,
    constraint_model: ConstraintModel,
    initial_schedule: Schedule
) -> Schedule:
```

The three inputs have distinct responsibilities:

- **`SchedulingProblem`**: contains the static scheduling data, including satellites, ground stations, observation tasks, task priorities and durations, and pre-computed visible windows.
- **`ConstraintModel`**: provides the shared feasibility-checking environment. Custom algorithms should use this model rather than independently reimplementing EOS scheduling constraints.
- **`Schedule`**: stores the assignments produced by the scheduler. The final `Schedule` returned by the algorithm is processed by the common EOS-Bench evaluation pipeline.

A typical feasibility check is performed through:

```python
constraint_model.is_feasible_assignment(candidate, current_schedule)
```

This design ensures that different algorithms are evaluated under the same scheduling rules and constraint model.

---

## 2. Step 1 — Implement the Scheduler

Create a new Python file under the `algorithms/` directory. For example:

```text
algorithms/alns.py
```

The scheduler class should inherit from `BaseSchedulerAlgorithm` and implement `search()`.

```python
# algorithms/alns.py

from schedulers.engine import BaseSchedulerAlgorithm
from schedulers.constraint_model import ConstraintModel, Schedule
from schedulers.scenario_loader import SchedulingProblem


class ALNSScheduler(BaseSchedulerAlgorithm):
    def __init__(self, cfg=None):
        # Store algorithm-specific configuration here
        self.cfg = cfg

    def search(
        self,
        problem: SchedulingProblem,
        constraint_model: ConstraintModel,
        initial_schedule: Schedule
    ) -> Schedule:
        # 1. Initialise the working schedule
        current_schedule = initial_schedule

        # 2. Implement the optimisation procedure here
        #
        # Example workflow:
        #   - generate candidate assignments
        #   - evaluate candidate solutions
        #   - check feasibility through ConstraintModel
        #   - accept/reject or update solutions according to
        #     the logic of your algorithm
        #
        # Example feasibility check:
        # if constraint_model.is_feasible_assignment(candidate, current_schedule):
        #     current_schedule.assignments.append(candidate)

        # 3. Return the final schedule
        return current_schedule
```

### Important integration principle

The custom algorithm should focus on **search and optimisation logic**. Whenever possible, use the shared EOS-Bench components for candidate generation, feasibility checking, objective evaluation, and reproducibility instead of creating separate implementations of the same functionality.

---

## 3. Built-in Utilities Available to Custom Algorithms

EOS-Bench provides several utilities that can be reused when implementing a new scheduler.

### 3.1 Candidate generation

`algorithms/candidate_pool.py` provides candidate assignment generation utilities.

```python
enumerate_task_candidates(...)
```

This utility can generate diversified candidate observation placements, including:

- earliest placements;
- centre placements;
- latest placements; and
- random placements.

Using the shared candidate-generation mechanism helps keep different algorithms consistent with the benchmark environment.

### 3.2 Objective evaluation

`algorithms/objectives.py` provides the shared objective model.

```python
ObjectiveModel(problem, weights).score(schedule)
```

This can be used to evaluate candidate schedules during iterative optimisation procedures such as simulated annealing, genetic algorithms, or other meta-heuristics.

### 3.3 Reproducible randomisation

`algorithms/random_utils.py` provides isolated random-number generation:

```python
make_rng(seed)
```

Use this utility when the algorithm contains stochastic operations so that experiments remain reproducible across runs.

---

## 4. Step 2 — Register the Algorithm in the Factory

After implementing the scheduler, register it in:

```text
algorithms/factory.py
```

First, import the scheduler class:

```python
from algorithms.alns import ALNSScheduler
```

Then add a corresponding branch to `create_algorithm()`:

```python
# algorithms/factory.py

from algorithms.alns import ALNSScheduler


def create_algorithm(algo_name: str, objective_weights=None, cfg_overrides=None):
    name = (algo_name or "").lower().strip()

    # ... existing algorithms ...

    if name in ("alns", "adaptive_large_ns"):
        return ALNSScheduler(cfg=cfg_overrides)
```

The registered name, such as `alns`, becomes the identifier used when selecting the algorithm from the benchmark configuration or command line.

---

## 5. Step 3 — Add the Algorithm to `main_scheduler.py`

Open:

```text
main_scheduler.py
```

Inside `run_benchmark()`, locate the `all_algo_specs` configuration list and add an entry for the new algorithm.

```python
all_algo_specs = [
    # ... existing algorithms ...

    {
        "class_id": 3,
        "algo_name": "alns",
        "cfg_overrides": {},
    },
]
```

EOS-Bench uses the following algorithm class identifiers:

| `class_id` | Algorithm class |
| :---: | :--- |
| `1` | Exact optimisation |
| `2` | Heuristics |
| `3` | Meta-heuristics |
| `4` | Learning-based methods |

For ALNS, `class_id=3` is appropriate because it is a meta-heuristic.

Make sure that the value of `algo_name` matches one of the names registered in `algorithms/factory.py`.

---

## 6. Step 4 — Run the New Algorithm

Once the implementation and registration are complete, the new algorithm can be executed through the standard EOS-Bench scheduler entry point.

For the ALNS example:

```bash
python main_scheduler.py --algos alns --workers_other 4
```

After registration, the custom scheduler uses the same EOS-Bench workflow as the built-in algorithms, including the common scheduling problem representation, constraint model, evaluation pipeline, logging, and downstream visualisation workflow.

---

## 7. Recommended Algorithm Structure

A custom scheduler will typically follow the structure below:

```text
SchedulingProblem
      │
      ▼
Generate / select candidate assignments
      │
      ▼
Algorithm-specific search or optimisation
      │
      ├── Candidate generation
      │     └── candidate_pool.py
      │
      ├── Feasibility checking
      │     └── ConstraintModel
      │
      ├── Objective evaluation
      │     └── objectives.py
      │
      └── Random operations, if required
            └── random_utils.py
      │
      ▼
Final Schedule
      │
      ▼
EOS-Bench evaluation and output pipeline
```

The main principle is that the **algorithm decides how to search**, while EOS-Bench provides the **problem data, feasibility environment, common evaluation framework, and standard output pipeline**.

---

## 8. Minimal Integration Checklist

Before running a newly added algorithm, check the following items:

- [ ] A new scheduler file has been created under `algorithms/`.
- [ ] The scheduler inherits from `BaseSchedulerAlgorithm`.
- [ ] The scheduler implements `search(problem, constraint_model, initial_schedule)`.
- [ ] The scheduler returns a valid `Schedule` object.
- [ ] Feasibility checks use the shared `ConstraintModel` where appropriate.
- [ ] The scheduler has been imported and registered in `algorithms/factory.py`.
- [ ] The algorithm name in `factory.py` matches the name used in `main_scheduler.py`.
- [ ] An entry has been added to `all_algo_specs` in `main_scheduler.py`.
- [ ] The correct `class_id` has been assigned.
- [ ] The algorithm can be selected through `--algos`.
- [ ] Stochastic algorithms use a reproducible random-number generator where appropriate.

---

## 9. Example File Locations

A typical integration may involve the following files:

```text
EOS-Bench/
│
├── algorithms/
│   ├── alns.py                 # Your new scheduling algorithm
│   ├── factory.py              # Register the algorithm here
│   ├── candidate_pool.py       # Optional shared candidate-generation utilities
│   ├── objectives.py           # Optional shared objective evaluation
│   └── random_utils.py         # Optional reproducible random utilities
│
├── schedulers/
│   ├── engine.py               # BaseSchedulerAlgorithm
│   ├── constraint_model.py     # ConstraintModel and Schedule
│   └── scenario_loader.py      # SchedulingProblem
│
└── main_scheduler.py           # Add the algorithm to the benchmark configuration
```

---

## 10. Summary

Adding a new scheduling method to EOS-Bench requires only four core actions:

1. **Implement** the optimisation logic as a `BaseSchedulerAlgorithm` subclass.
2. **Register** the scheduler in `algorithms/factory.py`.
3. **Configure** it in `main_scheduler.py`.
4. **Run** it through the standard benchmark entry point.

By following the unified scheduler interface, third-party algorithms can be compared with the built-in EOS-Bench methods under the same constraint definitions, evaluation metrics, and experimental workflow.
