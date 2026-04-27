# -*- coding: utf-8 -*-
"""
Main functionality:
This script generates a CZML orbit file from a scenario JSON using Orekit,
starts a local HTTP server, and opens a Cesium-based visualization page.
If a schedule JSON is provided, it will also be loaded into the viewer.
"""

import argparse
import os
import sys
import subprocess
import time
import webbrowser
from typing import Optional


def generate_czml(project_root: str, scenario_rel_path: str) -> None:
    """
    Call draw/orekit_to_czml.demo(...) to generate draw/orbit.czml.

    project_root: Absolute path to the project root directory
    scenario_rel_path: Scenario JSON path relative to project_root
                       Example: 'output/Scenario_S1_Sats3_M10_T0.5d_dist1.json'
    """
    draw_dir = os.path.join(project_root, "draw")
    scenario_abs = os.path.abspath(os.path.join(project_root, scenario_rel_path))

    if not os.path.exists(scenario_abs):
        raise FileNotFoundError(f"Scenario file not found: {scenario_abs}")

    orekit_script = os.path.join(draw_dir, "orekit_to_czml.py")
    if not os.path.exists(orekit_script):
        raise FileNotFoundError(f"orekit_to_czml.py not found in {draw_dir}")

    print("=== Step 1: Generating CZML ===")
    print(f"Project root directory: {project_root}")
    print(f"Draw directory        : {draw_dir}")
    print(f"Scenario file         : {scenario_abs}")

    # Switch to the draw directory so that relative paths
    # such as orekit-data.zip in orekit_to_czml work correctly
    old_cwd = os.getcwd()
    try:
        os.chdir(draw_dir)

        # Add the draw directory to sys.path to make importing orekit_to_czml easier
        if draw_dir not in sys.path:
            sys.path.insert(0, draw_dir)

        import orekit_to_czml  # type: ignore

        # The output file is placed under draw with the fixed name orbit.czml
        output_czml = "orbit.czml"

        # Pass the absolute path of the scenario file to demo
        # so it does not depend on the current working directory
        orekit_to_czml.demo(
            scenario_path=scenario_abs,
            output_czml=output_czml,
        )

    finally:
        os.chdir(old_cwd)

    print("=== CZML Generation Completed: draw/orbit.czml ===\n")


def start_http_server_and_open(
    project_root: str,
    port: int,
    schedule_rel_path: Optional[str] = None,
) -> None:
    """
    Start http.server under project_root and open cesium_viewer.html.

    If schedule_rel_path is provided, append it as a URL parameter.
    Otherwise, open the viewer without schedule data.
    """
    print("=== Step 2: Starting Local HTTP Server ===")
    print(f"Working directory : {project_root}")
    print(f"Command           : python -m http.server {port}")

    # Start http.server in a separate process
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait briefly to give the server some time to start
    time.sleep(1.5)

    base_url = f"http://localhost:{port}/draw/cesium_viewer.html"

    if schedule_rel_path and str(schedule_rel_path).strip():
        schedule_http_path = "/" + schedule_rel_path.replace(os.sep, "/").lstrip("/")
        url = f"{base_url}?schedule={schedule_http_path}"
    else:
        url = base_url

    print("=== Step 3: Opening Browser ===")
    print(f"URL: {url}\n")

    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Failed to open the browser automatically. Please open it manually: {url}")
        print(f"Reason: {e}")

    print("Server started. Press Ctrl+C to stop.")

    try:
        # Block and wait until the user presses Ctrl+C
        proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down the server...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        print("Server has exited.")


def main_draw(scenario_path: str, schedule_path: Optional[str] = None, port: int = 8000) -> None:
    # Project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 1) Generate orbit.czml under draw/orbit.czml
    generate_czml(project_root, scenario_path)

    # 2) Start http.server and open the browser
    start_http_server_and_open(
        project_root,
        port=port,
        schedule_rel_path=schedule_path,
    )


# ----------------------------------------------------------------------------
# Entry point (Supports both CLI args and hardcoded execution)
# python main_draw.py --scenario output/Scenario_S1_Sats20_M20_T1.0d_dist1.json --port 8080
# python main_draw.py --scenario output/Scenario_S1_Sats20_M20_T1.0d_dist1.json --schedule output/schedules/scheduler_Scenario_S1_Sats20_M20_T1.0d_dist1_c1_mip_p0.25_c0.25_t0.25_b0.25.json
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Visualize the scheduling solution with Orekit and Cesium")
        
        # 
        parser.add_argument(
            "--scenario",
            type=str,
            required=True,
            help="Scenario JSON path relative to the project root directory (e.g., output/Scenario_S1_Sats20.json)"
        )
        parser.add_argument(
            "--schedule",
            type=str,
            default=None,
            help="Optional schedule JSON path relative to the project root directory (e.g., output/schedules/scheduler_Scenario_S1.json)"
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="HTTP server port (Default: 8000)"
        )

        args = parser.parse_args()

        main_draw(
            scenario_path=args.scenario,
            schedule_path=args.schedule,
            port=args.port
        )

    # 
    else:
        scenario_file = "Scenario_S1_Sats20_M20_T1.0d_dist1"  # Scenario filename
        schedule_file = None  #"scheduler_Scenario_S1_Sats20_M20_T1.0d_dist1_c1_mip_p0.25_c0.25_t0.25_b0.25"#None  # Optional schedule filename, set to None if not needed
        
        # 
        scenario_path = "output/" + scenario_file + ".json"
        
        if schedule_file and str(schedule_file).strip():
            schedule_path = "output/schedules/" + schedule_file + ".json"
        else:
            schedule_path = None

        main_draw(
            scenario_path=scenario_path, 
            schedule_path=schedule_path, 
            port=8000
        )