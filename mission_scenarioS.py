#!/usr/bin/env python3

import argparse
import math
import time
import yaml
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

# AeroStack2 Python API
import rclpy
from as2_python_api.drone_interface import DroneInterface

# python_tsp
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search

# 自定义模块: PRM3D
from PRM3D import PRM3DPlanner

# ================== 常量 & 配置 ==================

TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED  = 2.0
SPEED           = 2.0
LAND_SPEED      = 0.5
SLEEP_TIME      = 0.5


# ================== TSP 求解函数 ==================
def solve_tsp_dp(waypoints_3d):
    """
    使用 python-tsp exact动态规划 算法求解
    waypoints_3d: [(x, y, z), ...]
    返回: (best_order, best_dist)
    """
    n = len(waypoints_3d)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                x1, y1, z1 = waypoints_3d[i]
                x2, y2, z2 = waypoints_3d[j]
                dist_matrix[i,j] = math.dist((x1,y1,z1), (x2,y2,z2))

    best_order, best_dist = solve_tsp_dynamic_programming(dist_matrix)
    return tuple(best_order), best_dist

def solve_tsp_ls(waypoints_3d):
    """
    使用 python-tsp heuristics 局部搜索 算法求解
    waypoints_3d: [(x, y, z), ...]
    返回: (best_order, best_dist)
    """
    coords = np.array(waypoints_3d)
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    best_order, best_dist = solve_tsp_local_search(dist_matrix)
    return tuple(best_order), best_dist


# ================== 其他工具函数 ==================
def read_scenario(file_path):
    """从YAML文件读取场景数据"""
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario


def generate_path_ros2(path_3d):
    """
    将离散的 (x,y,z) 列表转为 ROS2 Path 消息
    """
    path_msg = Path()
    path_msg.header = Header()
    path_msg.header.stamp = rclpy.time.Time().to_msg()
    path_msg.header.frame_id = "map"
    for (px, py, pz) in path_3d:
        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = px
        pose.pose.position.y = py
        pose.pose.position.z = pz
        path_msg.poses.append(pose)
    return path_msg

# ================== 无人机控制流程 ==================
def drone_start(drone_interface: DroneInterface) -> bool:
    print("=== Start mission ===")
    # Arm
    print("Arming...")
    success = drone_interface.arm()
    print(f"Arm success: {success}")
    if not success: return False

    # Offboard
    print("Setting Offboard...")
    success = drone_interface.offboard()
    print(f"Offboard success: {success}")
    if not success: return False

    # Take Off
    print("Taking off...")
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f"Takeoff success: {success}")
    return success


def perform_mission(drone_interface: DroneInterface, planner: PRM3DPlanner) -> bool:
    """
    1) 对 viewpoint_poses 进行 TSP
    2) 依序在 PRM3D 中规划路径
    3) 调用 follow_path + yaw 控制无人机飞行
    """
    print("=== Perform mission ===")

    # 取出必经点 + 朝向
    viewpoint_list = planner.viewpoints
    orientation_list = planner.orientations
    # 移除起点(0,0,TAKE_OFF_HEIGHT)那一项, 如果你只想对后续点TSP
    # 不移除也可以(看需求)
    # [0] 是起点的话
    # viewpoint_list = viewpoint_list[1:]
    # orientation_list = orientation_list[1:]

    # 进行TSP(此处以局部搜索为例)
    print("Calculating TSP with local_search...")
    order, min_dist = solve_tsp_ls(viewpoint_list)
    # 也可改成 order, min_dist = solve_tsp_dp(viewpoint_list)

    print(f"TSP order={order}, TSP dist={min_dist:.2f}")

    # 从(0,0,TAKE_OFF_HEIGHT)开始
    # 如果[0]就是 (0,0,TAKE_OFF_HEIGHT), 那TSP结果也包含它
    current_pose = viewpoint_list[0]

    for idx in order[1:]:  
        target_pose = viewpoint_list[idx]
        # PRM规划
        path_3d = planner.plan_path(current_pose, target_pose)
        if path_3d is None:
            print("No path found in PRM. Aborting.")
            return False
        
        # 转ROS2 Path
        path_ros2 = generate_path_ros2(path_3d)

        # 期望朝向
        yaw_target = orientation_list[idx]

        print(f"Flying from {current_pose} -> {target_pose}, yaw={yaw_target:.2f}...")
        success = drone_interface.follow_path.follow_path_with_yaw(
            path=path_ros2,
            speed=SPEED,
            angle=yaw_target,   # final yaw
            frame_id='earth'    # or "map" etc.
        )
        if not success:
            print("follow_path failed.")
            return False

        # 更新当前位置
        current_pose = target_pose
    return True


def drone_end(drone_interface: DroneInterface) -> bool:
    print("=== End mission ===")
    print("Landing...")
    success = drone_interface.land(speed=LAND_SPEED)
    print(f"Land success: {success}")
    if not success:
        return False

    print("Switching to Manual...")
    success = drone_interface.manual()
    print(f"Manual success: {success}")
    return success


# ================== main ==================
def main():
    parser = argparse.ArgumentParser(description='Single drone mission with PRM + TSP')
    parser.add_argument('scenario', type=str, help="Scenario YAML file to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True, help='Use simulation time')
    args = parser.parse_args()

    # 读取场景
    scenario_file = args.scenario
    scenario = read_scenario(scenario_file)
    scenario["file_path"] = scenario_file

    # 初始化
    rclpy.init()
    drone_ns = args.namespace
    print(f"Running 3D mission for drone {drone_ns}")

    # 提取 viewpoint + orientation
    viewpoint_list = []
    orientation_list = []
    # 可以先把起点 (0,0,TAKE_OFF_HEIGHT) 加入 viewpoints[0], yaw=0
    viewpoint_list.append((0.0, 0.0, TAKE_OFF_HEIGHT))
    orientation_list.append(0.0)

    if "viewpoint_poses" in scenario:
        vp_keys = sorted(list(scenario["viewpoint_poses"].keys()), key=int)
        for k in vp_keys:
            vp = scenario["viewpoint_poses"][k]
            viewpoint_list.append((vp["x"], vp["y"], vp["z"]))
            orientation_list.append(vp["w"])

    obstacles = scenario.get("obstacles", {})
    bounding_box = ((-12, -12, 0), (12, 12, 6))

    # 构建 PRM
    print("Building PRM3D...")
    planner = PRM3DPlanner(
        viewpoints=viewpoint_list,
        orientations=orientation_list,
        num_samples=200,
        obstacles=obstacles,
        bounding_box=bounding_box,
        k=10
    )
    print(f"PRM Graph has {len(planner.G.nodes)} nodes, {len(planner.G.edges)} edges.")

    # 创建无人机接口
    uav = DroneInterface(
        drone_id=drone_ns,
        use_sim_time=args.use_sim_time,
        verbose=args.verbose
    )

    # 任务流程
    success = drone_start(uav)
    try:
        start_t = time.time()
        if success:
            success = perform_mission(uav, planner)
        total_dur = time.time() - start_t
        print(f"Mission took {total_dur:.2f} seconds")
    except KeyboardInterrupt:
        success = False

    success2 = drone_end(uav)
    uav.shutdown()
    rclpy.shutdown()
    print("Clean exit")


if __name__ == "__main__":
    main()
