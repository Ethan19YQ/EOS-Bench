# -*- coding: utf-8 -*-
"""
"""

import argparse
import os
import sys
import subprocess
import time
import webbrowser


def generate_czml(project_root: str, scenario_rel_path: str) -> None:
    """
    调用 draw/orekit_to_czml.demo(...) 生成 draw/orbit.czml
    / Call draw/orekit_to_czml.demo(...) to generate draw/orbit.czml

    project_root: 0new 根目录的绝对路径 / Absolute path to the 0new root directory
    scenario_rel_path: 相对于 project_root 的场景 json 路径 / Scenario JSON path relative to project_root
                       例如 / e.g.: 'output/schedules/Scenario_S1_Sats3_M10_T0.5d_dist1.json'
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

    # 切到 draw 目录，让 orekit_to_czml 里的相对路径（orekit-data.zip 等）正常生效
    # / Switch to the draw directory so that relative paths (like orekit-data.zip) in orekit_to_czml work correctly
    old_cwd = os.getcwd()
    try:
        os.chdir(draw_dir)

        # 把 draw 目录加到 sys.path，方便 import orekit_to_czml
        # / Add draw directory to sys.path to easily import orekit_to_czml
        if draw_dir not in sys.path:
            sys.path.insert(0, draw_dir)

        import orekit_to_czml  # type: ignore

        # 输出文件就放在 draw 下，名字固定 orbit.czml
        # / The output file is placed under draw with a fixed name orbit.czml
        output_czml = "orbit.czml"

        # 这里把场景文件绝对路径传给 demo（不依赖当前工作目录）
        # / Here we pass the absolute path of the scenario file to demo (independent of the current working directory)
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
    schedule_rel_path: str,
) -> None:
    """
    在 project_root 下启动 http.server，并打开带 schedule 参数的 cesium_viewer.html
    / Start http.server under project_root and open cesium_viewer.html with the schedule parameter

    schedule_rel_path: 相对于 project_root 的调度 JSON 路径 / Schedule JSON path relative to project_root
                       例如 / e.g.: 'output/schedules/Scenario_S1_schedule.json'
    """
    print("=== Step 2: Starting Local HTTP Server ===")
    print(f"Working directory : {project_root}")
    print(f"Command           : python -m http.server {port}")

    # 在单独的进程中启动 http.server
    # / Start http.server in a separate process
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # 简单等一下，给服务器一点启动时间
    # / Wait briefly to give the server some time to start
    time.sleep(1.5)

    # schedule 的 HTTP 路径，比如 'output/...' -> '/output/...'
    # / HTTP path for the schedule, e.g., 'output/...' -> '/output/...'
    schedule_http_path = "/" + schedule_rel_path.replace(os.sep, "/").lstrip("/")

    url = (
        f"http://localhost:{port}/draw/cesium_viewer.html"
        f"?schedule={schedule_http_path}"
    )

    print("=== Step 3: Opening Browser ===")
    print(f"URL: {url}\n")

    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Failed to open browser automatically, please visit manually: {url}")
        print(f"Reason: {e}")

    print("Server started, press Ctrl+C to stop.")

    try:
        # 阻塞等待，直到用户 Ctrl+C
        # / Block and wait until the user presses Ctrl+C
        proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C detected, shutting down the server ...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        print("Server has exited.")


def main_draw(scenario_file, schedule_file):
    parser = argparse.ArgumentParser(description="Visualize scheduling scheme (Orekit + Cesium)")

    # 场景 JSON（包含卫星、任务、时间等）
    # / Scenario JSON (including satellites, missions, time, etc.)
    parser.add_argument(
        "--scenario",
        type=str,
        default="output/" + scenario_file + ".json",
        help=(
            "Scenario JSON path relative to the project root directory "
            "(Default: output/Scenario_S1_Sats3_M10_T0.5d_dist1.json)"
        ),
    )

    # 调度方案 JSON（包含 assignments + metrics）
    # / Schedule JSON (including assignments + metrics)
    parser.add_argument(
        "--schedule",
        type=str,
        default="output/schedules/" + schedule_file + ".json",
        help=(
            "Schedule JSON path relative to the project root directory "
            "(Default: output/schedules/Scenario_S1_schedule.json)"
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port (Default: 8000)",
    )

    args = parser.parse_args()

    # 0new 根目录
    # / 0new root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 1) 生成 orbit.czml（draw/orbit.czml）
    # / 1) Generate orbit.czml (draw/orbit.czml)
    generate_czml(project_root, args.scenario)

    # 2) 启动 http.server 并打开浏览器，带上 schedule 路径
    # / 2) Start http.server and open the browser with the schedule path
    start_http_server_and_open(
        project_root,
        port=args.port,
        schedule_rel_path=args.schedule,
    )


if __name__ == "__main__":
    scenario_file = "Scenario_S1_Sats1_M10_T1.0d_dist1" # 场景文件名 / Scenario filename
    schedule_file = "scheduler_Scenario_S1_Sats1_M10_T1.0d_dist1_c1_mip_p1_c0_t0_b0"  # 调度方案文件名 / Schedule filename
    main_draw(scenario_file, schedule_file)