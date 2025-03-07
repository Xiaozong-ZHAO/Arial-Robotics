#!/usr/bin/env python3
import math
import time
import argparse
import threading

import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from rclpy.time import Time

# AeroStack2 Python API
from as2_python_api.drone_interface import DroneInterface

# python-tsp
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search

# 自定义模块: PRM3D
from PRM3D import PRM3DPlanner

# ================== 常量 & 配置 ==================
TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED  = 3.0
SPEED           = 2.0
LAND_SPEED      = 3.0

# ================== 距离记录节点 ==================
class MetricLogger(Node):
    """
    订阅 /drone0/ground_truth/pose (PoseStamped)
    连续计算无人机飞行距离
    """
    def __init__(self, drone_namespace="drone0"):
        super().__init__("distance_logger")
        self.prev_pos = None
        self.total_distance = 0.0

        self._flight_start_time = None
        self._flight_end_time = None

        topic_name = f"/{drone_namespace}/ground_truth/pose"

        # 配合发布端BEST_EFFORT QoS（可据实际情况更改）
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            PoseStamped,
            topic_name,
            self.pose_callback,
            qos_profile
        )
        self.get_logger().info(f"DistanceLogger subscribed to: {topic_name}")

    def pose_callback(self, msg):
        
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        if self.prev_pos is not None:
            dist = math.dist((x, y, z), self.prev_pos)
            self.total_distance += dist

        self.prev_pos = (x, y, z)

    def get_distance(self) -> float:
        """返回当前累计飞行距离"""
        return self.total_distance
    
    def start_flight_timer(self):
        self._flight_start_time = self.get_clock().now()
    
    def end_flight_timer(self):
        self._flight_end_time = self.get_clock().now()
    
    def get_flight_duration(self) -> float:
        if self._flight_start_time is None or self._flight_end_time is None:
            return 0.0
        dt = self._flight_end_time - self._flight_start_time
        return dt.nanoseconds * 1e-9


# ================== TSP 求解函数 ==================
def solve_tsp_dp(waypoints_3d):
    n = len(waypoints_3d)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                x1, y1, z1 = waypoints_3d[i]
                x2, y2, z2 = waypoints_3d[j]
                dist_matrix[i, j] = math.dist((x1, y1, z1), (x2, y2, z2))
    best_order, best_dist = solve_tsp_dynamic_programming(dist_matrix)
    return tuple(best_order), best_dist

def solve_tsp_ls(waypoints_3d):
    coords = np.array(waypoints_3d)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
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
    if not success:
        return False

    # Offboard
    print("Setting Offboard...")
    success = drone_interface.offboard()
    print(f"Offboard success: {success}")
    if not success:
        return False

    # Take Off
    print("Taking off...")
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f"Takeoff success: {success}")
    return success


def perform_mission(drone_interface: DroneInterface, 
                    planner: PRM3DPlanner,
                    metric_logger: MetricLogger) -> bool:
    print("=== Perform mission ===")

    viewpoint_list = planner.viewpoints
    orientation_list = planner.orientations

    # 进行TSP(此处以局部搜索为例)
    # if viewpoint_list larger than 10, use local search
    if len(viewpoint_list) <= 10:
        print("Calculating TSP with dynamic programming...")
        start_time = time.time()
        order, min_dist = solve_tsp_dp(viewpoint_list)
        duration = time.time() - start_time
        print(f"TSP duration={duration:.2f}, TSP dist={min_dist:.2f}")
    else:
        print("Calculating TSP with local search...")
        start_time = time.time()
        order, min_dist = solve_tsp_ls(viewpoint_list)
        duration = time.time() - start_time
        print(f"TSP duration={duration:.2f}, TSP dist={min_dist:.2f}")
    metric_logger.start_flight_timer()

    # 逐段飞到TSP路径里的下一个航点
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
        yaw_target = orientation_list[idx]

        print(f"Flying to viewpoint {idx}, yaw={yaw_target:.2f}...")
        success = drone_interface.follow_path.follow_path_with_yaw(
            path=path_ros2,
            speed=SPEED,
            angle=yaw_target,
            frame_id='earth'
        )
        if not success:
            print("follow_path failed.")
            return False

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
    parser = argparse.ArgumentParser(description='Single drone mission with PRM + TSP + distance logger')
    parser.add_argument('scenario', type=str, help="Scenario YAML file to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='Drone namespace')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True, help='Use simulation time')
    args = parser.parse_args()

    # 初始化
    rclpy.init()
    drone_ns = args.namespace
    print(f"Running 3D mission for drone {drone_ns}")

    # 读取场景
    scenario_file = args.scenario
    scenario = read_scenario(scenario_file)
    scenario["file_path"] = scenario_file

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
    drone_interface = DroneInterface(
        drone_id=drone_ns,
        use_sim_time=args.use_sim_time,
        verbose=args.verbose
    )

    # 创建距离记录节点 (改用 ground_truth/pose + BEST_EFFORT QoS)
    metric_logger = MetricLogger(drone_namespace=drone_ns)

    # MultiThreadedExecutor，执行 spin
    executor = MultiThreadedExecutor()
    executor.add_node(drone_interface)
    executor.add_node(metric_logger)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 任务流程
    # start_time = time.time()
    success = drone_start(drone_interface)
    if success:
        try:
            success = perform_mission(drone_interface, planner, metric_logger)
        except KeyboardInterrupt:
            success = False
    # total_dur = time.time() - start_time
    # print(f"Mission took {total_dur:.2f} seconds")

    # 结束并打印距离
    success2 = drone_end(drone_interface)
    # if successfully landed, get the stop_flight_time
    if success2:
        metric_logger.end_flight_timer()
    
    final_distance = metric_logger.get_distance()
    final_duration = metric_logger.get_flight_duration()
    print(f"===> Final flight distance: {final_distance:.2f} meters")
    print(f"===> Final flight duration: {final_duration:.2f} seconds")
    # 关闭节点
    drone_interface.shutdown()
    executor.shutdown()
    spin_thread.join()
    metric_logger.destroy_node()
    rclpy.shutdown()
    print("Clean exit")

if __name__ == "__main__":
    main()
