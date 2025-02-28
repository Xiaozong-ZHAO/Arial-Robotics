import argparse
from time import sleep
import time
import yaml
import heapq

from as2_python_api.drone_interface import DroneInterface
import rclpy
import numpy as np
import math

TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0  # Max speed in m/s
SLEEP_TIME = 0.5  # Sleep time between behaviors in seconds
SPEED = 1.0  # Max speed in m/s
LAND_SPEED = 0.5  # Max speed in m/s

class OccupancyGrid2D:
    def __init__(self, x_size=10.0, y_size=10.0, resolution=0.1,
                 origin_x=0.0, origin_y=0.0):
        """
        简易2D栅格地图，记录障碍物。可根据需求改成3D或用现成库。
        """
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width  = int(x_size / resolution)
        self.height = int(y_size / resolution)
        # 0 表示空闲，1 表示障碍
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def world_to_map(self, x, y):
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        return (ix, iy)

    def map_to_world(self, ix, iy):
        x = ix * self.resolution + self.origin_x
        y = iy * self.resolution + self.origin_y
        return (x, y)

    def set_obstacle(self, x_min, x_max, y_min, y_max):
        """将指定范围[x_min, x_max] x [y_min, y_max]标记为障碍"""
        ix_min, iy_min = self.world_to_map(x_min, y_min)
        ix_max, iy_max = self.world_to_map(x_max, y_max)

        if ix_min > ix_max:
            ix_min, ix_max = ix_max, ix_min
        if iy_min > iy_max:
            iy_min, iy_max = iy_max, iy_min

        ix_min = max(0, ix_min)
        ix_max = min(self.width - 1, ix_max)
        iy_min = max(0, iy_min)
        iy_max = min(self.height - 1, iy_max)

        self.grid[iy_min:iy_max+1, ix_min:ix_max+1] = 1


# ============ A* 路径规划函数 ============
def a_star_search(grid_map: OccupancyGrid2D, start_xy, goal_xy):
    """
    在2D栅格地图上用A*搜索，start_xy, goal_xy: (x, y) in world coords
    返回：路径点列表[(x0, y0), (x1, y1), ...]，若失败返回None
    """
    start_ix, start_iy = grid_map.world_to_map(*start_xy)
    goal_ix,  goal_iy  = grid_map.world_to_map(*goal_xy)

    # 若起点或终点在障碍里，直接失败
    if grid_map.grid[start_iy, start_ix] == 1 or grid_map.grid[goal_iy, goal_ix] == 1:
        print("Start or goal in obstacle!")
        return None

    open_list = []
    heapq.heappush(open_list, (0, (start_ix, start_iy)))
    g_cost = {(start_ix, start_iy): 0.0}
    parent = {(start_ix, start_iy): None}

    # 4邻接可改成8邻接
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while open_list:
        f, current = heapq.heappop(open_list)
        # 到达终点
        if current == (goal_ix, goal_iy):
            print("A* succeeded!")
            return reconstruct_path(parent, current, grid_map)

        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            # 越界
            if nx < 0 or nx >= grid_map.width or ny < 0 or ny >= grid_map.height:
                continue
            # 障碍
            if grid_map.grid[ny, nx] == 1:
                continue
            # 计算新代价
            new_g = g_cost[current] + 1.0  # 简单相邻步代价=1
            # 若新代价更小，或未访问过，更新
            if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                # 更新代价
                g_cost[(nx, ny)] = new_g
                h = abs(goal_ix - nx) + abs(goal_iy - ny)  # 曼哈顿启发
                f_new = new_g + h
                parent[(nx, ny)] = current
                heapq.heappush(open_list, (f_new, (nx, ny)))
    print("A* failed!")
    return None

def reconstruct_path(parent, current, grid_map: OccupancyGrid2D):
    path = []
    while current is not None:
        cx, cy = current
        wx, wy = grid_map.map_to_world(cx, cy)
        path.append((wx, wy))
        current = parent[current]
    return list(reversed(path))


# ============ TSP 求解（示例：穷举）============
from itertools import permutations

def solve_tsp_bruteforce(waypoints_2d):
    """
    waypoints_2d: [(x, y), (x, y), ...]
    返回：最优访问顺序(perm)，以及最小距离。
    仅适合点数较少时演示用
    """
    n = len(waypoints_2d)
    indices = range(n)
    best_order = None
    best_dist = float('inf')

    for perm in permutations(indices):
        dist_sum = 0.0
        for i in range(n - 1):
            x1, y1 = waypoints_2d[perm[i]]
            x2, y2 = waypoints_2d[perm[i+1]]
            dist_sum += math.hypot(x2 - x1, y2 - y1)
        if dist_sum < best_dist:
            best_dist = dist_sum
            best_order = perm
    return best_order, best_dist


# ============ 任务流程：start, run, end ============

def drone_start(drone_interface: DroneInterface) -> bool:
    print('Start mission')

    # Arm
    print('Arm')
    success = drone_interface.arm()
    print(f'Arm success: {success}')

    # Offboard
    print('Offboard')
    success = drone_interface.offboard()
    print(f'Offboard success: {success}')

    # Take Off
    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'Take Off success: {success}')

    return success


def drone_run(drone_interface: DroneInterface, scenario: dict) -> bool:
    """
    本函数中加入“构建地图 -> 求解TSP -> A*避障飞行”的整合逻辑示例
    """
    print('Run mission')
    
    # ========== 0) 从scenario中载入信息，比如obstacles和viewpoint_poses ===========
    # 你也可以改成读取 'aruco_targets' 等字段
    if "viewpoint_poses" not in scenario:
        print("No viewpoint_poses in scenario, nothing to do.")
        return False
    
    # 载入目标点
    # viewpoint_poses 是一个 dict: { "id1": {"x":..., "y":..., "z":..., "w":...}, ... }
    # 下面收集成列表
    pose_keys = list(scenario["viewpoint_poses"].keys())
    targets = []
    for key in pose_keys:
        vp = scenario["viewpoint_poses"][key]
        targets.append((vp["x"], vp["y"], vp["z"]))
    
    # ========== 1) 构建地图（占据栅格示例） ===========
    occ_map = OccupancyGrid2D(x_size=20.0, y_size=20.0, resolution=0.5,
                              origin_x=-10.0,  origin_y=-10.0)
    # 如果scenario里有obstacles，就填进map
    if "obstacles" in scenario:
        for _, obs in scenario["obstacles"].items():
            print(obs)
            occ_map.set_obstacle(obs["x"]-0.5*obs["d"], obs["x"]+0.5*obs["d"],
                                 obs["y"]-0.5*obs["h"], obs["y"]+0.5*obs["h"])
    print("Occupancy map built.")
    
    # ========== 2) TSP：决定访问顺序（只用 x,y） ===========
    # 这里忽略z，仅用 (x,y) 做距离矩阵
    waypoints_2d = [(t[0], t[1]) for t in targets]
    order, min_dist = solve_tsp_bruteforce(waypoints_2d)
    
    print(f"TSP best order: {order}, total distance={min_dist:.2f}")
    
    # ========== 3) 获取无人机当前位置(简化假设) ===========
    #   也可用 /self_localization/pose 或 drone_interface.get_position() 之类获取
    #   这里先假设在(0,0,TAKE_OFF_HEIGHT)
    current_pose_3d = (0.0, 0.0, TAKE_OFF_HEIGHT)
    print(f"Start from {current_pose_3d}")
    # ========== 4) 按顺序逐个访问目标 ===========
    for idx in order:
        goal_xyz = targets[idx]
        # A* 仅做2D规划：start=(cx,cy), goal=(gx,gy)，z先忽略
        print(f"Going to viewpoint {pose_keys[idx]} at {goal_xyz}")
        plan_2d = a_star_search(occ_map, (current_pose_3d[0], current_pose_3d[1]),
                                           (goal_xyz[0], goal_xyz[1]))
        if plan_2d is None:
            print(f"A* failed from {current_pose_3d[:2]} to {goal_xyz[:2]}!")
            return False
        # ========== 逐段走完A*路径(或直接跳到终点) ==========
        # 为演示，这里逐段 go_to；也可以只用 go_to_point_with_yaw(目标终点)
        for (px, py) in plan_2d:
            waypoint_3d = [px, py, current_pose_3d[2]]  # 保持高度不变
            # 这里暂不做yaw的复杂计算，可直接设 angle=0 或者基于飞行方向
            success = drone_interface.go_to.go_to_point_with_yaw(
                waypoint_3d, angle=0.0, speed=SPEED)
            if not success:
                print("go_to failed!")
                return False
            sleep(SLEEP_TIME)

        # 最后修正朝向(若 scenario 要求对着vp["w"])
        # 例如:
        final_angle = scenario["viewpoint_poses"][pose_keys[idx]]["w"]
        success = drone_interface.go_to.go_to_point_with_yaw(
            [goal_xyz[0], goal_xyz[1], goal_xyz[2]],
            angle=final_angle, speed=SPEED)
        if not success:
            return False
        sleep(SLEEP_TIME)

        # 更新当前pose
        current_pose_3d = goal_xyz

        print(f"Reached viewpoint {pose_keys[idx]} at {goal_xyz}, done.")

    print("All viewpoint_poses visited with obstacle avoidance!")
    return True


def drone_end(drone_interface: DroneInterface) -> bool:
    print('End mission')

    # Land
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    if not success:
        return success

    # Manual
    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')

    return success

# ================== 读取场景文件 ==================
def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# ================== main ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single drone mission with path planning + TSP')
    parser.add_argument('scenario', type=str, help="scenario file to attempt to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0',
                        help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True,
                        help='Use simulation time')
    args = parser.parse_args()

    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time

    print(f'Running mission for drone {drone_namespace}')

    print(f"Reading scenario {args.scenario}")
    scenario = read_scenario(args.scenario)

    rclpy.init()

    uav = DroneInterface(
        drone_id=drone_namespace,
        use_sim_time=use_sim_time,
        verbose=verbosity)

    # ========== 开始飞行任务 ==========
    success = drone_start(uav)
    try:
        start_time = time.time()
        if success:
            success = drone_run(uav, scenario)
        duration = time.time() - start_time
        print("---------------------------------")
        print(f"Tour of {args.scenario} took {duration:.2f} seconds")
        print("---------------------------------")
    except KeyboardInterrupt:
        pass

    # ========== 任务结束 ==========
    success = drone_end(uav)
    uav.shutdown()
    rclpy.shutdown()
    print('Clean exit')
    exit(0)