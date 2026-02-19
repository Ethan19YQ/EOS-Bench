# algorithms/meta_ga.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from schedulers.scenario_loader import SchedulingProblem
from schedulers.engine import BaseSchedulerAlgorithm
from schedulers.constraint_model import ConstraintModel, Schedule, Assignment

from algorithms.candidate_pool import enumerate_task_candidates
from algorithms.objectives import ObjectiveWeights, ObjectiveModel
from algorithms.random_utils import make_rng, spawn_seeds


@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.08
    tournament_k: int = 3
    elitism: int = 2
    max_candidates_per_task: int = 128
    random_samples_per_window: int = 1
    restarts: int = 1
    seed: Optional[int] = None
    weights: ObjectiveWeights = field(default_factory=lambda: ObjectiveWeights(1.0, 0.0, 0.0, 0.0))
class GeneticAlgorithmScheduler(BaseSchedulerAlgorithm):
    def __init__(self, cfg: Optional[GAConfig] = None) -> None:
        self.cfg = cfg or GAConfig()

    def _build_candidates(self, problem: SchedulingProblem, cm: ConstraintModel, seed: Optional[int]):
        task_ids = list(problem.tasks.keys())
        # 用算法 seed 派生每个任务的子 seed，避免依赖 PYTHONHASHSEED/hash(tid)
        child_seeds = spawn_seeds(seed, len(task_ids))
        cand_map: Dict[str, List[Assignment]] = {}
        for i, tid in enumerate(task_ids):
            cand_map[tid] = enumerate_task_candidates(
                problem=problem,
                task=problem.tasks[tid],
                placement_mode=cm.placement_mode,
                downlink_duration_ratio=cm.downlink_duration_ratio,
                max_candidates=self.cfg.max_candidates_per_task,
                random_samples_per_window=self.cfg.random_samples_per_window,
                seed=(None if seed is None else int(child_seeds[i])),
                prefer_must_first=False,
            )
        return task_ids, cand_map

    def _decode(self, genes: List[int], task_ids: List[str], cand_map: Dict[str, List[Assignment]], cm: ConstraintModel) -> Schedule:
        sch = Schedule()
        # 解码顺序影响可行性：用“候选最少”优先（更稳）
        order = sorted(range(len(task_ids)), key=lambda i: len(cand_map.get(task_ids[i], [])))
        for i in order:
            tid = task_ids[i]
            g = genes[i]
            if g < 0:
                continue
            cands = cand_map.get(tid, [])
            if not cands or g >= len(cands):
                continue
            a = cands[g]
            if cm.is_feasible_assignment(a, sch):
                sch.assignments.append(a)
        return sch

    def _fitness(self, genes, task_ids, cand_map, cm, obj):
        return obj.score(self._decode(genes, task_ids, cand_map, cm))

    def _rand_individual(self, task_ids, cand_map, rng):
        genes = []
        for tid in task_ids:
            cands = cand_map.get(tid, [])
            if not cands:
                genes.append(-1)
            else:
                genes.append(rng.randrange(len(cands)) if rng.random() < 0.75 else -1)
        return genes

    def _tournament(self, pop, fits, rng):
        k = max(2, int(self.cfg.tournament_k))
        best_i, best_f = 0, float("-inf")
        for _ in range(k):
            i = rng.randrange(len(pop))
            if fits[i] > best_f:
                best_i, best_f = i, fits[i]
        return pop[best_i][:]

    def _crossover(self, a, b, rng):
        if rng.random() > self.cfg.crossover_rate or len(a) <= 1:
            return a[:], b[:]
        p = rng.randrange(1, len(a))
        return a[:p] + b[p:], b[:p] + a[p:]

    def _mutate(self, genes, task_ids, cand_map, rng):
        for i, tid in enumerate(task_ids):
            if rng.random() > self.cfg.mutation_rate:
                continue
            cands = cand_map.get(tid, [])
            if not cands:
                genes[i] = -1
            else:
                genes[i] = -1 if rng.random() < 0.25 else rng.randrange(len(cands))

    def _run_once(self, problem, cm, seed) -> Schedule:
        rng = make_rng(seed)
        obj = ObjectiveModel(problem, self.cfg.weights)

        task_ids, cand_map = self._build_candidates(problem, cm, seed)

        pop = [self._rand_individual(task_ids, cand_map, rng) for _ in range(self.cfg.population_size)]
        best_genes, best_fit = None, float("-inf")

        for _ in range(self.cfg.generations):
            fits = [self._fitness(ind, task_ids, cand_map, cm, obj) for ind in pop]

            for ind, f in zip(pop, fits):
                if f > best_fit:
                    best_fit = f
                    best_genes = ind[:]

            elite_n = max(0, int(self.cfg.elitism))
            elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:elite_n]
            new_pop = [pop[i][:] for i in elite_idx]

            while len(new_pop) < self.cfg.population_size:
                p1 = self._tournament(pop, fits, rng)
                p2 = self._tournament(pop, fits, rng)
                c1, c2 = self._crossover(p1, p2, rng)
                self._mutate(c1, task_ids, cand_map, rng)
                self._mutate(c2, task_ids, cand_map, rng)
                new_pop.append(c1)
                if len(new_pop) < self.cfg.population_size:
                    new_pop.append(c2)

            pop = new_pop

        if best_genes is None:
            return Schedule()
        return self._decode(best_genes, task_ids, cand_map, cm)

    def search(self, problem: SchedulingProblem, constraint_model: ConstraintModel, initial_schedule: Schedule) -> Schedule:
        best = deepcopy(initial_schedule)
        obj = ObjectiveModel(problem, self.cfg.weights)
        best_score = obj.score(best)

        base = self.cfg.seed
        for r in range(max(1, int(self.cfg.restarts))):
            s = (None if base is None else base + r * 10007)
            cand = self._run_once(problem, constraint_model, s)
            sc = obj.score(cand)
            if sc > best_score:
                best, best_score = cand, sc

        return best
