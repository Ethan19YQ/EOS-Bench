# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 2026

@author: QianY

analyze_combined_conflict.py

Main features:
1. Merge and schedule the functionalities of analyze_conflict_degree.py and analyze_resource_conflict.py.
2. Extract the specified 12 underlying metrics and calculate the final 10 English core metrics (including Λ_ed) for output.
3. Support extracting features by "Relative Path" and grouping them to calculate averages (stripping random seeds like _p, _cities_, _seed, _sf, _S).
4. Output two worksheets in the same Excel file:
   - CombinedSummary: Row-by-row scenario data.
   - AveragedSummary: Averages calculated for metric fields after merging by the relative path template.
5. Retain the detailed TXT report output.

Usage:
Modify the parameters directly at the bottom of the file, then run:
python analyze_combined_conflict.py
"""

import gc
import importlib.util
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# 0) Dynamically Load Dependent Scripts
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CD_PATH = BASE_DIR / "analyze_conflict_degree.py"
RC_PATH = BASE_DIR / "analyze_resource_conflict.py"


def _load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Dependent script not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


acd = _load_module("acd_module", CD_PATH)
arc = _load_module("arc_module", RC_PATH)


# =============================================================================
# 1) Base & Logging Utilities
# =============================================================================

def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def normalize_path_like_value(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    s = s.replace('\\', '/')
    s = re.sub(r'/+', '/', s)
    return s


def canonicalize_relative_path_for_group_display(path_value: Any) -> str:
    """Generate the merged relative path (stripping various seeds and numbers)."""
    s = normalize_path_like_value(path_value)
    if not s:
        return ""
    s = re.sub(r'_p\d+_', '_p_', s, flags=re.IGNORECASE)
    s = re.sub(r'_p\d+(?=\.|$)', '_p', s, flags=re.IGNORECASE)
    s = re.sub(r'_cities_\d+_', '_cities_', s, flags=re.IGNORECASE)
    s = re.sub(r'_cities_\d+(?=\.|$)', '_cities_', s, flags=re.IGNORECASE)
    s = re.sub(r'_seed\d+_', '_seed_', s, flags=re.IGNORECASE)
    s = re.sub(r'_seed\d+(?=\.|$)', '_seed', s, flags=re.IGNORECASE)
    s = re.sub(r'_sf\d+_', '_sf_', s, flags=re.IGNORECASE)
    s = re.sub(r'_sf\d+(?=\.|$)', '_sf', s, flags=re.IGNORECASE)
    s = re.sub(r'_S\d+_', '_S_', s)
    s = re.sub(r'_S\d+(?=\.|$)', '_S', s)
    return s


# =============================================================================
# 2) Extract & Calculate Core Metrics, Average by Group
# =============================================================================

def extract_and_calculate_metrics(
    cd_result: "acd.ConflictAnalysisResult",
    rc_result: "arc.ScenarioResourceConflictResult",
) -> Dict[str, Any]:
    
    # Calculate Excess Demand Ratio (Λ_ed)
    weighted_excess_area = rc_result.weighted_excess_demand_area
    total_conflict_duration = rc_result.total_conflict_duration_s
    peak_conflict_tasks = rc_result.peak_conflict_task_count
    
    denominator = total_conflict_duration * (peak_conflict_tasks - 1)
    if denominator > 0:
        excess_demand_ratio = weighted_excess_area / denominator
    else:
        excess_demand_ratio = 0.0

    return {
        "Relative Path": cd_result.rel_path,
        "Average Available Opportunities (Γ_ao)": cd_result.avg_candidates_per_task,
        "Opportunity Constrained Task Ratio (Γ_oc)": cd_result.hard_task_ratio_k2,
        "Task Interference Ratio (Γ_ti)": cd_result.task_conflict_density,
        "Average Task Pair Conflict Ratio (Γ_at)": cd_result.mean_pair_conflict_ratio,
        "Task Elasticity Ratio (Γ_te)": rc_result.avg_elasticity_score,
        "Observation Contention Ratio (Λ_oc)": cd_result.observation_contention_index,
        "Conflict Satellite Ratio (Λ_cs)": rc_result.satellite_conflict_ratio,
        "Timeline Overload Ratio (Λ_to)": rc_result.conflict_coverage_ratio,
        "Average Conflict tasks (Λ_ac)": rc_result.mean_conflict_task_count_on_conflict_steps,
        "Excess Demand Ratio (Λ_ed)": excess_demand_ratio
    }


def build_averaged_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge and average row-by-row scenario data based on canonicalize_relative_path."""
    grouped: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    display_paths: Dict[str, str] = {}

    for row in records:
        raw_path = row.get("Relative Path", "")
        group_display_path = canonicalize_relative_path_for_group_display(raw_path)
        group_key = group_display_path.lower()

        if not group_key:
            fallback_key = f"__row__{len(grouped)}"
            grouped[fallback_key] = [row]
            display_paths[fallback_key] = group_display_path
            continue

        if group_key not in grouped:
            grouped[group_key] = []
            display_paths[group_key] = group_display_path
        
        grouped[group_key].append(row)

    averaged_rows: List[Dict[str, Any]] = []

    for group_key, group_records in grouped.items():
        out_row: Dict[str, Any] = {"Relative Path": display_paths.get(group_key, "")}
        
        # Calculate averages for all metrics except "Relative Path"
        metric_keys = [k for k in group_records[0].keys() if k != "Relative Path"]
        
        for key in metric_keys:
            numeric_values = [r.get(key) for r in group_records if isinstance(r.get(key), (int, float))]
            if numeric_values:
                out_row[key] = sum(numeric_values) / len(numeric_values)
            else:
                out_row[key] = None
                
        averaged_rows.append(out_row)

    return averaged_rows


# =============================================================================
# 3) Detailed Output (TXT Generation Feature)
# =============================================================================

def write_combined_detail_txt(
    detail_txt: Path,
    cd_result: "acd.ConflictAnalysisResult",
    rc_result: "arc.ScenarioResourceConflictResult",
    sat_results: List["arc.SatelliteConflictResult"],
) -> None:
    with open(detail_txt, "w", encoding="utf-8") as f:
        f.write("Combined Conflict Analysis Detailed Results\n")
        f.write(f"Scenario ID: {cd_result.scenario_id}\n")
        f.write(f"Relative Path: {cd_result.rel_path}\n")
        
        f.write("\n[Task Conflict Degree Metrics - analyze_conflict_degree]\n")
        f.write(f"Average Candidates: {cd_result.avg_candidates_per_task:.6f}\n")
        f.write(f"Hard Task Ratio (k<=2): {cd_result.hard_task_ratio_k2:.6f}\n")
        f.write(f"Task Conflict Density: {cd_result.task_conflict_density:.6f}\n")
        f.write(f"Average Task Pair Conflict Ratio: {cd_result.mean_pair_conflict_ratio:.6f}\n")
        f.write(f"Observation Contention Index: {cd_result.observation_contention_index:.6f}\n")
        
        f.write("\n[Resource Conflict Metrics - analyze_resource_conflict]\n")
        f.write(f"Conflict Satellite Ratio: {rc_result.satellite_conflict_ratio:.6f}\n")
        f.write(f"Conflict Coverage Ratio: {rc_result.conflict_coverage_ratio:.6f}\n")
        f.write(f"Average Conflict Tasks: {rc_result.mean_conflict_task_count_on_conflict_steps:.6f}\n")
        f.write(f"Average Elasticity Index: {rc_result.avg_elasticity_score:.6f}\n")
        f.write(f"Total Conflict Duration (s): {rc_result.total_conflict_duration_s:.6f}\n")
        f.write(f"Peak Conflict Task Count: {rc_result.peak_conflict_task_count}\n")
        f.write(f"Weighted Excess Demand Area: {rc_result.weighted_excess_demand_area:.6f}\n")
        f.write("\n")

        for sat_res in sat_results:
            f.write("=" * 110 + "\n")
            f.write(f"Satellite: {sat_res.satellite_id}\n")
            f.write(f"Conflict Step Count: {sat_res.conflict_step_count}\n")
            f.write(f"Total Conflict Duration (s): {sat_res.conflict_duration_s:.4f}\n")
            f.write(f"Peak Conflict Task Count: {sat_res.peak_conflict_task_count}\n")
            
            if sat_res.conflict_step_count <= 0:
                f.write("This satellite has no conflict time.\n\n")
                continue

            merged_segments = arc.merge_conflict_segments_for_output(sat_res.conflict_segments)
            f.write("\n[Merged Continuous Conflict Windows]\n")
            f.write("Start Time | End Time | Duration (s) | Conflict Task Count | Conflict Source | Task ID List\n")
            f.write("-" * 120 + "\n")
            for seg in merged_segments:
                task_str = ", ".join(seg.task_ids)
                f.write(
                    f"{seg.start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{seg.end_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{seg.duration_s:>8.2f} | "
                    f"{seg.conflict_task_count:>10d} | "
                    f"{seg.conflict_source_label:^6s} | "
                    f"{task_str}\n"
                )
            f.write("\n")


# =============================================================================
# 4) Single File Analysis
# =============================================================================

def analyze_one_file(
    scenario_path: Path,
    output_dir: Path,
    detail_dir: Path,
    workers: int,
    agility_profile: str,
    non_agile_transition_s: float,
    downlink_duration_ratio: float,
    pair_chunk_size: int,
    observation_step_multiplier: int,
    max_candidates_per_task: Optional[int],
    max_parallel_total_candidates: int,
    analysis_time_step_s: Optional[float],
    default_transition_when_angle_missing_s: float,
    transition_search_horizon_s: Optional[float],
    long_conflict_thresholds_s: List[float],
    critical_feasible_duration_s: float,
    low_elasticity_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    
    file_name = scenario_path.name
    try:
        rel_path = str(scenario_path.resolve().relative_to(output_dir.resolve()))
    except Exception:
        rel_path = str(scenario_path.resolve())

    log(f"Reading scenario file: {scenario_path}")
    try:
        problem_cd = acd.load_scheduling_problem_from_json(scenario_path)
        cd_result = acd.analyze_conflict_degree(
            problem=problem_cd, file_name=file_name, rel_path=rel_path, workers=workers,
            agility_profile=agility_profile, non_agile_transition_s=non_agile_transition_s,
            downlink_duration_ratio=downlink_duration_ratio, pair_chunk_size=pair_chunk_size,
            observation_step_multiplier=observation_step_multiplier, max_candidates_per_task=max_candidates_per_task,
            max_parallel_total_candidates=max_parallel_total_candidates,
        )

        problem_rc = arc.load_scheduling_problem_from_json(scenario_path)
        rc_result, sat_results = arc.analyze_resource_conflict_scenario(
            problem=problem_rc, file_name=file_name, rel_path=rel_path, workers=workers,
            analysis_time_step_s=analysis_time_step_s, agility_profile=agility_profile,
            non_agile_transition_s=non_agile_transition_s, default_transition_when_angle_missing_s=default_transition_when_angle_missing_s,
            transition_search_horizon_s=transition_search_horizon_s, long_conflict_thresholds_s=long_conflict_thresholds_s,
            critical_feasible_duration_s=critical_feasible_duration_s, low_elasticity_threshold=low_elasticity_threshold,
        )

        detail_name = f"combined_conflict_detail_{arc.sanitize_filename(cd_result.scenario_id)}.txt"
        detail_txt = detail_dir / detail_name
        write_combined_detail_txt(detail_txt, cd_result, rc_result, sat_results)

        row = extract_and_calculate_metrics(cd_result, rc_result)
        
        del problem_cd, problem_rc, cd_result, rc_result, sat_results
        gc.collect()
        
        return row, detail_txt
        
    except Exception as e:
        log(f"ERROR: Failed to analyse file {file_name}: {e}")
        gc.collect()
        return None, None


# =============================================================================
# 5) Main Flow
# =============================================================================

def main_analyze(
    scenario_file: Optional[str],
    output_dir: str = "output",
    workers: int = 1,
    agility_profile: str = "Standard-Agility",
    non_agile_transition_s: float = 10.0,
    downlink_duration_ratio: float = 1.0,
    pair_chunk_size: int = 20,
    observation_step_multiplier: int = 1,
    max_candidates_per_task: Optional[int] = None,
    max_parallel_total_candidates: int = 200000,
    analysis_time_step_s: Optional[float] = None,
    default_transition_when_angle_missing_s: float = 11.66,
    transition_search_horizon_s: Optional[float] = None,
    long_conflict_thresholds_s: Optional[List[float]] = None,
    critical_feasible_duration_s: float = 120.0,
    low_elasticity_threshold: float = 2.0,
) -> None:
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if long_conflict_thresholds_s is None:
        long_conflict_thresholds_s = [60.0, 120.0, 300.0]

    detail_dir = output_dir_path / f"combined_conflict_details_{time.strftime('%Y%m%d_%H%M%S')}"
    detail_dir.mkdir(parents=True, exist_ok=True)
    
    excel_output_path = output_dir_path / f"conflict_analysis_metrics_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    results_list = []

    # Determine whether to process a single file or scan a folder
    files_to_process = []
    if scenario_file is not None and str(scenario_file).strip() != "":
        files_to_process.append(arc.resolve_single_scenario_path(str(scenario_file).strip(), output_dir_path))
    else:
        for p in arc.iter_scenario_jsons(output_dir_path):
            files_to_process.append(p)

    total_files = len(files_to_process)
    if total_files <= 0:
        log(f"No scenario JSONs found in {output_dir_path} for analysis.")
        return

    log(f"Found {total_files} scenarios in total. Starting analysis and extracting Excel metrics...")
    
    for idx, scenario_path in enumerate(files_to_process, start=1):
        log(f"File progress: {idx}/{total_files}")
        row, _ = analyze_one_file(
            scenario_path=scenario_path, output_dir=output_dir_path, detail_dir=detail_dir,
            workers=max(1, int(workers)), agility_profile=agility_profile, non_agile_transition_s=float(non_agile_transition_s),
            downlink_duration_ratio=float(downlink_duration_ratio), pair_chunk_size=max(1, int(pair_chunk_size)),
            observation_step_multiplier=max(1, int(observation_step_multiplier)), max_candidates_per_task=max_candidates_per_task,
            max_parallel_total_candidates=max(1, int(max_parallel_total_candidates)), analysis_time_step_s=analysis_time_step_s,
            default_transition_when_angle_missing_s=float(default_transition_when_angle_missing_s), transition_search_horizon_s=transition_search_horizon_s,
            long_conflict_thresholds_s=list(long_conflict_thresholds_s), critical_feasible_duration_s=float(critical_feasible_duration_s), low_elasticity_threshold=float(low_elasticity_threshold),
        )
        if row:
            results_list.append(row)

    # Generate the Excel file with two worksheets
    if results_list:
        log("Analysis completed. Generating Excel file with CombinedSummary and AveragedSummary worksheets...")
        
        # Build raw row-by-row data
        df_combined = pd.DataFrame(results_list)
        
        # Calculate averages by merging and deduplicating based on the path
        averaged_list = build_averaged_summary(results_list)
        df_averaged = pd.DataFrame(averaged_list)
        
        # Write to dual sheets
        with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
            df_combined.to_excel(writer, sheet_name="CombinedSummary", index=False)
            df_averaged.to_excel(writer, sheet_name="AveragedSummary", index=False)
            
        log(f"Data successfully written to Excel: {excel_output_path}")
    else:
        log("Failed to generate valid analysis data. Skipping Excel export.")


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.freeze_support()
    except Exception:
        pass

    from pathlib import Path

    scenario_file = None
    
    # 【修改这里】定位到当前脚本的上一级目录，再找 output 文件夹
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    workers = 10

    agility_profile = "Standard-Agility"
    non_agile_transition_s = 10.0
    downlink_duration_ratio = 1.0
    pair_chunk_size = 10
    observation_step_multiplier = 1
    max_candidates_per_task = None
    max_parallel_total_candidates = 200000

    analysis_time_step_s = None
    default_transition_when_angle_missing_s = 11.66
    transition_search_horizon_s = None
    long_conflict_thresholds_s = [60.0, 120.0, 300.0]
    critical_feasible_duration_s = 120.0
    low_elasticity_threshold = 2.0

    main_analyze(
        scenario_file=scenario_file,
        output_dir=output_dir,
        workers=workers,
        agility_profile=agility_profile,
        non_agile_transition_s=non_agile_transition_s,
        downlink_duration_ratio=downlink_duration_ratio,
        pair_chunk_size=pair_chunk_size,
        observation_step_multiplier=observation_step_multiplier,
        max_candidates_per_task=max_candidates_per_task,
        max_parallel_total_candidates=max_parallel_total_candidates,
        analysis_time_step_s=analysis_time_step_s,
        default_transition_when_angle_missing_s=default_transition_when_angle_missing_s,
        transition_search_horizon_s=transition_search_horizon_s,
        long_conflict_thresholds_s=long_conflict_thresholds_s,
        critical_feasible_duration_s=critical_feasible_duration_s,
        low_elasticity_threshold=low_elasticity_threshold,
    )