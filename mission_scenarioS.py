#!/usr/bin/env python3

import argparse
import heapq
import math
import numpy as np
import yaml
import os
import json
from itertools import permutations
from time import sleep
import time

# ------------------------- AeroStack2 Python API -------------------------
import rclpy
from as2_python_api.drone_interface import DroneInterface

# ------------------------- TSP 缓存结果文件 -------------------------
RESULTS_FILE = "tsp_results.json"

def load_tsp_results():
    """加载 TSP 计算结果，若文件不存在或为空，则返回空字典"""
    if not os.path.exists(RESULTS_FILE) or os.stat(RESULTS_FILE).st_size == 0:
        print(f"⚠️ {RESULTS_FILE} 文件不存在或为空，初始化计算...")
        return {}
    try:
        with open(RESULTS_FILE, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"⚠️ {RESULTS_FILE} 解析失败，重置为空字典。")
        return {}

def save_tsp_results(results):
    """保存计算结果到 JSON 文件"""
    with open(RESULTS_FILE, 'w') as file:
        json.dump(results, file, indent=4)

# ------------------------- 3D 占据栅格 -------------------------
class OccupancyGrid3D:
    def __init__(self, x_size=10.0, y_size=10.0, z_size=10.0, resolution=0.1,
                 origin_x=0.0, origin_y=0.0, origin_z=0.0):
        """
        3D 占据栅格地图，0 表示空闲，1 表示障碍。
        grid[z, y, x] = 0/1
        """
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.width  = int(x_size / resolution)
        self.height = int(y_size / resolution)
        self.depth  = int(z_size / resolution)
        self.grid = np.zeros((self.depth, self.height, self.width), dtype=np.uint8)

    def in_bounds(self, ix, iy, iz):
        return (0 <= ix < self.width and 
                0 <= iy < self.height and
                0 <= iz < self.depth)

    def world_to_map(self, x, y, z):
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        iz = int((z - self.origin_z) / self.resolution)
        return ix, iy, iz

    def map_to_world(self, ix, iy, iz):
        x = ix * self.resolution + self.origin_x
        y = iy * self.resolution + self.origin_y
        z = iz * self.resolution + self.origin_z
        return x, y, z

    def set_obstacle(self, x_min, x_max, y_min, y_max, z_min, z_max):
        """标记立方体范围内为障碍"""
        ix_min, iy_min, iz_min = self.world_to_map(x_min, y_min, z_min)
        ix_max, iy_max, iz_max = self.world_to_map(x_max, y_max, z_max)
        if ix_min > ix_max: ix_min, ix_max = ix_max, ix_min
        if iy_min > iy_max: iy_min, iy_max = iy_max, iy_min
        if iz_min > iz_max: iz_min, iz_max = iz_max, iz_min
        ix_min, ix_max = max(0, ix_min), min(self.width - 1, ix_max)
        iy_min, iy_max = max(0, iy_min), min(self.height - 1, iy_max)
        iz_min, iz_max = max(0, iz_min), min(self.depth - 1, iz_max)
        self.grid[iz_min:iz_max+1, iy_min:iy_max+1, ix_min:ix_max+1] = 1

# ------------------------- 3D A* 搜索 -------------------------
def a_star_search_3d(grid_map: OccupancyGrid3D, start_xyz, goal_xyz):
    sx, sy, sz = start_xyz
    gx, gy, gz = goal_xyz
    start_ix, start_iy, start_iz = grid_map.world_to_map(sx, sy, sz)
    goal_ix,  goal_iy,  goal_iz  = grid_map.world_to_map(gx, gy, gz)

    # 基础检查
    if not grid_map.in_bounds(start_ix, start_iy, start_iz):
        print("Start out of map bounds.")
        return None
    if not grid_map.in_bounds(goal_ix, goal_iy, goal_iz):
        print("Goal out of map bounds.")
        return None
    if grid_map.grid[start_iz, start_iy, start_ix] == 1 or grid_map.grid[goal_iz, goal_iy, goal_ix] == 1:
        print("Start or Goal in obstacle!")
        return None

    open_list = []
    heapq.heappush(open_list, (0, (start_ix, start_iy, start_iz)))
    g_cost = {(start_ix, start_iy, start_iz): 0.0}
    parent = {(start_ix, start_iy, start_iz): None}

    # 6邻接(上下前后左右)，也可扩展为26邻接
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    while open_list:
        f, current = heapq.heappop(open_list)
        if current == (goal_ix, goal_iy, goal_iz):
            return reconstruct_path_3d(parent, current, grid_map)

        cx, cy, cz = current
        for dx, dy, dz in directions:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if not grid_map.in_bounds(nx, ny, nz):
                continue
            if grid_map.grid[nz, ny, nx] == 1:
                continue

            cost_new = g_cost[current] + 1.0
            if (nx, ny, nz) not in g_cost or cost_new < g_cost[(nx, ny, nz)]:
                g_cost[(nx, ny, nz)] = cost_new
                # 欧几里得距离做启发
                h = math.sqrt((goal_ix - nx)**2 + (goal_iy - ny)**2 + (goal_iz - nz)**2)
                f_new = cost_new + h
                parent[(nx, ny, nz)] = current
                heapq.heappush(open_list, (f_new, (nx, ny, nz)))
    print("A* 3D failed: no path found!")
    return None

def reconstruct_path_3d(parent, current, grid_map: OccupancyGrid3D):
    path = []
    while current is not None:
        cx, cy, cz = current
        wx, wy, wz = grid_map.map_to_world(cx, cy, cz)
        path.append((wx, wy, wz))
        current = parent[current]
    return list(reversed(path))

# ------------------------- 3D TSP（穷举示例）-------------------------
def solve_tsp_bruteforce_3d(waypoints_3d):
    """
    waypoints_3d: [(x, y, z), ...]
    返回 (best_order, best_dist)
    """
    n = len(waypoints_3d)
    best_order, best_dist = None, float('inf')
    for perm in permutations(range(n)):
        dist_sum = 0.0
        for i in range(n - 1):
            x1, y1, z1 = waypoints_3d[perm[i]]
            x2, y2, z2 = waypoints_3d[perm[i+1]]
            dist_sum += math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if dist_sum < best_dist:
            best_dist = dist_sum
            best_order = perm
    return best_order, best_dist

# ------------------------- 读取场景文件 -------------------------
def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# ------------------------- 基础无人机任务 start/run/end -------------------------
TAKE_OFF_HEIGHT = 1.0
TAKE_OFF_SPEED  = 1.0
SLEEP_TIME      = 0.5
SPEED           = 1.0
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

def drone_run(drone_interface: DroneInterface, scenario: dict) -> bool:
    """
    将 3D TSP + 3D A* 与无人机飞行控制结合，支持 JSON 缓存TSP结果
    """
    print("Run mission")

    if "viewpoint_poses" not in scenario:
        print("No viewpoint_poses in scenario, nothing to do.")
        return False

    # 1) 收集目标点(3D)
    pose_keys = list(scenario["viewpoint_poses"].keys())
    targets_3d = []
    for key in pose_keys:
        vp = scenario["viewpoint_poses"][key]
        targets_3d.append((vp["x"], vp["y"], vp["z"]))

    # 2) 生成 scenario_key 并读取/写入 JSON 缓存
    #    这里举例用场景文件名做key，或者用 scenario["name"] 也行
    scenario_key = os.path.basename(scenario.get("file_path", "undefined.yaml"))
    # 若 scenario 里没有 "file_path" 字段，可直接用:
    # scenario_key = os.path.basename(args_scenario_file) # 需传进来

    # 如果 scenario 里存有 "name" 字段，可以改成:
    # scenario_key = scenario.get("name", "default_scenario")
    
    if scenario_key == "undefined.yaml":
        # 如果 scenario 里没有 file_path，就退化到 "default_scenario"
        scenario_key = scenario.get("name", "default_scenario")

    tsp_results = load_tsp_results()
    if scenario_key in tsp_results:
        print(f"读取缓存TSP结果: {scenario_key}")
        order = tsp_results[scenario_key]["order"]
        min_dist = tsp_results[scenario_key]["min_dist"]
    else:
        print(f"计算TSP(3D): {scenario_key}")
        order, min_dist = solve_tsp_bruteforce_3d(targets_3d)
        tsp_results[scenario_key] = {"order": order, "min_dist": min_dist}
        save_tsp_results(tsp_results)

    print(f"TSP order={order}, dist={min_dist:.2f}")

    # 3) 构建 3D 占据栅格 & 加载障碍
    #    根据需要设置地图大小、分辨率
    grid_map_3d = OccupancyGrid3D(
        x_size=24.0, y_size=24.0, z_size=6.0,
        resolution=0.5,
        origin_x=-12.0, origin_y=-12.0, origin_z=0.0
    )

    if "obstacles" in scenario:
        # 假设 obstacles 是 dict: obs_name -> {x, y, z, d, h, w}, 代表包围盒
        for _, obs in scenario["obstacles"].items():
            x_min = obs["x"] - 0.5 * obs["d"]
            x_max = obs["x"] + 0.5 * obs["d"]
            y_min = obs["y"] - 0.5 * obs["h"]
            y_max = obs["y"] + 0.5 * obs["h"]
            z_min = obs["z"] - 0.5 * obs["w"]
            z_max = obs["z"] + 0.5 * obs["w"]
            grid_map_3d.set_obstacle(x_min, x_max, y_min, y_max, z_min, z_max)

    print("3D Occupancy map built.")

    # 4) 从 (0,0,TAKE_OFF_HEIGHT) 出发，依次访问 TSP 顺序下的目标
    current_pose = (0.0, 0.0, TAKE_OFF_HEIGHT)
    for idx in order:
        goal_xyz = targets_3d[idx]
        print(f"Planning path from {current_pose} to {goal_xyz}")
        path_3d = a_star_search_3d(grid_map_3d, current_pose, goal_xyz)
        if path_3d is None:
            print("A* 3D path not found, mission aborted.")
            return False

        # 逐点移动到目标(或只移动到终点)
        for (px, py, pz) in path_3d:
            success = drone_interface.go_to.go_to_point_with_yaw(
                [px, py, pz], angle=0.0, speed=SPEED)
            if not success:
                print("go_to failed!")
                return False
            # sleep(SLEEP_TIME)

        # 若有需要，给出最终 yaw (场景中 viewpoint_poses[key]["w"])
        final_angle = scenario["viewpoint_poses"][pose_keys[idx]].get("w", 0.0)
        success = drone_interface.go_to.go_to_point_with_yaw(
            [goal_xyz[0], goal_xyz[1], goal_xyz[2]],
            angle=final_angle,
            speed=SPEED
        )
        if not success:
            return False
        # sleep(SLEEP_TIME)

        # 更新当前位置
        current_pose = goal_xyz
        print(f"Reached viewpoint {pose_keys[idx]}")

    print("All viewpoint_poses visited with 3D path planning!")
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
            success = drone_run(uav, scenario)
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
