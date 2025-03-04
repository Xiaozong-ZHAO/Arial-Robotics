#!/usr/bin/env python3

import argparse
import heapq
import math
import numpy as np
import yaml
import os
import json
import networkx as nx

from itertools import permutations
from time import sleep
import time
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from PRM3D import PRM3DPlanner
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
# ------------------------- AeroStack2 Python API -------------------------
import rclpy
from as2_python_api.drone_interface import DroneInterface
# ------------------------- TSP 缓存结果文件 -------------------------
RESULTS_FILE = "tsp_results.json"


def solve_tsp_dp(waypoints_3d):
    """
    waypoints_3d: [(x, y, z), ...]
    返回 (best_order, best_dist)
    """
    n = len(waypoints_3d)
    # 构造距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                x1, y1, z1 = waypoints_3d[i]
                x2, y2, z2 = waypoints_3d[j]
                dist_matrix[i, j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    # 使用 python_tsp 提供的动态规划方法求解
    best_order, best_dist = solve_tsp_dynamic_programming(dist_matrix)
    
    return tuple(best_order), best_dist

def solve_tsp_ls(waypoints_3d):
    """
    waypoints_3d: [(x, y, z), ...]
    返回 (best_order, best_dist)
    """
    coords = np.array(waypoints_3d)
    n = len(coords)

    # 利用广播机制生成距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # 启发式 / 局部搜索求解
    best_order, best_dist = solve_tsp_local_search(dist_matrix)

    return tuple(best_order), best_dist

def generate_path_ros2(path_3d):
    """
    将 A* 生成的 3D 路径转换为 ROS2 Path 消息格式
    :param path_3d: [(x, y, z), ...] A* 生成的路径点
    :return: Path 消息
    """
    path_msg = Path()
    
    # 设置 header
    path_msg.header = Header()
    path_msg.header.stamp = rclpy.time.Time().to_msg()  # 设置时间戳
    path_msg.header.frame_id = "map"  # 设定路径的参考坐标系
    
    # 遍历 A* 生成的路径点
    for (px, py, pz) in path_3d:
        pose = PoseStamped()
        pose.header = path_msg.header  # 统一时间戳和坐标系
        pose.pose.position.x = px
        pose.pose.position.y = py
        pose.pose.position.z = pz
        path_msg.poses.append(pose)
    
    return path_msg

# ------------------------- 读取场景文件 -------------------------
def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# ------------------------- 基础无人机任务 start/run/end -------------------------
TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED  = 2.0
SLEEP_TIME      = 0.5
SPEED           = 2.0
LAND_SPEED      = 0.5

def drone_start(drone_interface: DroneInterface) -> bool:
    print("Start mission")
    print("Arm")
    success = drone_interface.arm()
    print(f"Arm success: {success}")

    print("Offboard")
    success = drone_interface.offboard()
    print(f"Offboard success: {success}")

    print("Take Off")
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f"Take Off success: {success}")
    return success
#----------------------------new drone_run--------------------------------
def drone_run2(drone_interface: DroneInterface, planner: PRM3DPlanner) -> bool:
    """
    将 3D TSP + 3D A* 与无人机飞行控制结合，支持 JSON 缓存TSP结果
    """
    print("Run mission")
    # load the viewpoint list
    viewpoint_list = planner.viewpoints
    # calculate the TSP
    print("Calculating TSP...")
    # order, min_dist = solve_tsp_bruteforce_3d(viewpoint_list)
    order, min_dist = solve_tsp_ls(viewpoint_list)
    # order = (6, 4, 2, 7, 3, 9, 5, 1, 0, 8)
    # min_dist = 53.451108678324196
    print(f"TSP order={order}, dist={min_dist:.2f}")
    current_pose = (0.0, 0.0, TAKE_OFF_HEIGHT)
    # path planning for each viewpoint
    for idx in order:
        target_pose = viewpoint_list[idx]
        path_3d = planner.plan_path(current_pose, target_pose)
        if path_3d is None:
            print("A* 3D path not found, mission aborted.")
            return False
        # convert the path into ros2 path message
        path_ros2 = generate_path_ros2(path_3d)
        final_angle = planner.orientations[idx]
        success = drone_interface.follow_path.follow_path_with_yaw(
            path=path_ros2,
            speed=SPEED,
            angle=final_angle,
            frame_id='earth'
        )
        if not success:
            print("follow_path failed!")
            return False
        current_pose = target_pose
        # print(f"Reached viewpoint {target_pose[idx]}")
    return True

def drone_end(drone_interface: DroneInterface) -> bool:
    print("End mission")
    print("Land")
    success = drone_interface.land(speed=LAND_SPEED)
    print(f"Land success: {success}")
    if not success:
        return success
    print("Manual")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success

# ------------------------- main -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single drone mission with 3D path planning + TSP + JSON cache')
    parser.add_argument('scenario', type=str, help="Scenario YAML file to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True, help='Use simulation time')
    args = parser.parse_args()
    # 读取场景
    scenario_file = args.scenario
    scenario = read_scenario(scenario_file)
    # 把文件路径存入 scenario 以便做 scenario_key
    scenario["file_path"] = scenario_file  
    print(f"Running 3D mission for drone {args.namespace}")
    rclpy.init()

    viewpoint_list = []
    orientation_list = []
    viewpoint_list.append((0.0, 0.0, TAKE_OFF_HEIGHT))
    orientation_list.append(0.0)
    if "viewpoint_poses" in scenario:
        vp_keys = sorted(list(scenario["viewpoint_poses"].keys()), key=int)
        for k in vp_keys:
            vp = scenario["viewpoint_poses"][k]
            viewpoint_list.append((vp["x"], vp["y"], vp["z"]))
            orientation_list.append(vp["w"])
    else:
        viewpoint_list = []

    bounding_box = ((-12, -12, 0), (12, 12, 6))
    obstacles = scenario.get("obstacles", {})

    planner = PRM3DPlanner(
        viewpoints=viewpoint_list,
        orientations=orientation_list,
        num_samples=200,
        obstacles=obstacles,
        bounding_box=bounding_box,
        k=10
    )
    print("PRM Graph built with nodes:", len(planner.G.nodes))
    print("PRM Graph built with edges:", len(planner.G.edges))

    # 启动无人机接口
    uav = DroneInterface(
        drone_id=args.namespace,
        use_sim_time=args.use_sim_time,
        verbose=args.verbose
    )

    # ========== 开始飞行任务 ==========
    success = drone_start(uav)
    try:
        start_time = time.time()
        if success:
            success = drone_run2(uav, planner)
        duration = time.time() - start_time
        print("---------------------------------")
        print(f"Mission for {scenario_file} took {duration:.2f} seconds")
        print("---------------------------------")
    except KeyboardInterrupt:
        pass

    # ========== 结束任务 ==========
    success = drone_end(uav)
    uav.shutdown()
    rclpy.shutdown()
    print("Clean exit")
