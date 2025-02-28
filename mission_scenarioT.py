import argparse
import heapq
import math
import numpy as np
import yaml
from itertools import permutations
import json
import os

RESULTS_FILE = "tsp_results.json"

# ================= 2D 栅格地图 =================
class OccupancyGrid2D:
    def __init__(self, x_size=10.0, y_size=10.0, resolution=0.1, origin_x=0.0, origin_y=0.0):
        """
        2D 占据栅格地图，0 表示空闲，1 表示障碍。
        """
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = int(x_size / resolution)
        self.height = int(y_size / resolution)
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def world_to_map(self, x, y):
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        return ix, iy

    def map_to_world(self, ix, iy):
        x = ix * self.resolution + self.origin_x
        y = iy * self.resolution + self.origin_y
        return x, y

    def set_obstacle(self, x_min, x_max, y_min, y_max):
        ix_min, iy_min = self.world_to_map(x_min, y_min)
        ix_max, iy_max = self.world_to_map(x_max, y_max)
        ix_min, ix_max = max(0, ix_min), min(self.width - 1, ix_max)
        iy_min, iy_max = max(0, iy_min), min(self.height - 1, iy_max)
        self.grid[iy_min:iy_max+1, ix_min:ix_max+1] = 1

# ================= A* 搜索 =================
def a_star_search(grid_map: OccupancyGrid2D, start_xy, goal_xy):
    start_ix, start_iy = grid_map.world_to_map(*start_xy)
    goal_ix, goal_iy = grid_map.world_to_map(*goal_xy)
    
    if grid_map.grid[start_iy, start_ix] == 1 or grid_map.grid[goal_iy, goal_ix] == 1:
        return None
    
    open_list = []
    heapq.heappush(open_list, (0, (start_ix, start_iy)))
    g_cost = {(start_ix, start_iy): 0.0}
    parent = {(start_ix, start_iy): None}
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while open_list:
        f, current = heapq.heappop(open_list)
        if current == (goal_ix, goal_iy):
            return reconstruct_path(parent, current, grid_map)
        
        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_map.width and 0 <= ny < grid_map.height and grid_map.grid[ny, nx] == 0:
                new_g = g_cost[current] + 1.0
                if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = new_g
                    h = abs(goal_ix - nx) + abs(goal_iy - ny)
                    f_new = new_g + h
                    parent[(nx, ny)] = current
                    heapq.heappush(open_list, (f_new, (nx, ny)))
    return None

def reconstruct_path(parent, current, grid_map):
    path = []
    while current is not None:
        cx, cy = current
        wx, wy = grid_map.map_to_world(cx, cy)
        path.append((wx, wy))
        current = parent[current]
    return list(reversed(path))

# ================= TSP 求解（暴力搜索） =================
def solve_tsp_bruteforce(waypoints_2d):
    best_order, best_dist = None, float('inf')
    for perm in permutations(range(len(waypoints_2d))):
        dist_sum = sum(math.hypot(waypoints_2d[perm[i+1]][0] - waypoints_2d[perm[i]][0],
                                   waypoints_2d[perm[i+1]][1] - waypoints_2d[perm[i]][1])
                        for i in range(len(perm) - 1))
        if dist_sum < best_dist:
            best_dist, best_order = dist_sum, perm
    return best_order, best_dist

# ================= 读取 YAML 文件 =================
def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# ================= 读取 JSON 结果 =================
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

# ================= 存储 JSON 结果 =================
def save_tsp_results(results):
    """保存计算结果到 JSON 文件"""
    with open(RESULTS_FILE, 'w') as file:
        json.dump(results, file, indent=4)

# ================= 测试代码 =================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description='Single drone mission')

    parser.add_argument('scenario', type=str, help="scenario file to attempt to execute")
    parser.add_argument('-n', '--namespace',
                        type=str,
                        default='drone0',
                        help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time',
                        action='store_true',
                        default=True,
                        help='Use simulation time')

    args = parser.parse_args()
    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time
    scenario = read_scenario(args.scenario)

    pose_keys = list(scenario["viewpoint_poses"].keys())
    targets = []
    for key in pose_keys:
        vp = scenario["viewpoint_poses"][key]
        targets.append((vp["x"], vp["y"], vp["z"]))
        # 读取已有的TSP计算结果
    tsp_results = load_tsp_results()

    # 生成唯一的 scenario 键值
    scenario_key = os.path.basename(args.scenario)

    if scenario_key in tsp_results:
        print(f"读取缓存结果: {scenario_key}")
        order = tsp_results[scenario_key]["order"]
        min_dist = tsp_results[scenario_key]["min_dist"]
    else:
        print(f"计算TSP路径: {scenario_key}")
        order, min_dist = solve_tsp_bruteforce(targets)
        tsp_results[scenario_key] = {"order": order, "min_dist": min_dist}
        save_tsp_results(tsp_results)
    
    # 创建地图
    occ_map = OccupancyGrid2D(x_size=20.0, y_size=20.0, resolution=0.5,
                        origin_x=-10.0,  origin_y=-10.0)
    if "obstacles" in scenario:
        for _, obs in scenario["obstacles"].items():
            occ_map.set_obstacle(obs["x"]-0.5*obs["d"], obs["x"]+0.5*obs["d"],
                                 obs["y"]-0.5*obs["h"], obs["y"]+0.5*obs["h"])
    print("TSP Best Order:", order, "Min Distance:", min_dist)
    
    # 逐个执行 A*
    start = (0.0, 0.0)
    for idx in order:
        goal = targets[idx]
        goal_xy = goal[:2]  # 只取 x, y
        start_xy = start[:2]  # 只取 x, y
        path = a_star_search(occ_map, start_xy, goal_xy)
        print(f"A* Path to {goal}:", path)
        if path is not None:
            start = goal