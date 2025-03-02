import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

class PRM3DPlanner:
    def __init__(self, 
                 viewpoints,       # list of (x,y,z) 必经航点
                 num_samples,      # 随机采样数
                 obstacles, 
                 bounding_box, 
                 k=10):
        """
        初始化 3D PRM 规划器
        :param viewpoints: list of (x,y,z) - 必须纳入图的航点
        :param num_samples: 随机采样数量
        :param obstacles: 形如 scenario["obstacles"], dict: obs_id -> {x,y,z,d,h,w}
        :param bounding_box: ((xmin,ymin,zmin), (xmax,ymax,zmax)) 采样范围
        :param k: k近邻
        """
        self.viewpoints = viewpoints
        self.num_samples = num_samples
        self.obstacles = obstacles
        self.bounding_box = bounding_box
        self.k = k

        self.G = nx.Graph()      # NetworkX图
        self.idx_to_xyz = {}     # node_id -> (x,y,z)
        self.build_prm()         # 构建 PRM
    
    # ========== 一系列辅助函数（与之前类似） ==========
    def is_point_in_box(self, pt, box_min, box_max):
        return (box_min[0] <= pt[0] <= box_max[0] and
                box_min[1] <= pt[1] <= box_max[1] and
                box_min[2] <= pt[2] <= box_max[2])

    def is_in_collision(self, pt):
        """
        判断pt是否在任一障碍盒(AABB)内
        """
        for _, obs in self.obstacles.items():
            cx, cy, cz = obs["x"], obs["y"], obs["z"]
            dx, dy, dz = obs["d"], obs["h"], obs["w"]
            box_min = (cx - dx/2, cy - dy/2, cz - dz/2)
            box_max = (cx + dx/2, cy + dy/2, cz + dz/2)
            if self.is_point_in_box(pt, box_min, box_max):
                return True
        return False
    
    def check_segment_box_intersect(self, p1, p2, box_min, box_max):
        """
        判断线段 p1->p2 是否与盒子(AABB)相交
        """
        # 如果起点或终点在盒子内,就算相交
        if self.is_point_in_box(p1, box_min, box_max):
            return True
        if self.is_point_in_box(p2, box_min, box_max):
            return True
        
        # 用 parametric t in [0,1] 测试
        seg_dir = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
        tmin, tmax = 0.0, 1.0
        for i in range(3):
            if abs(seg_dir[i]) < 1e-9:
                if p1[i] < box_min[i] or p1[i] > box_max[i]:
                    return False
            else:
                inv_d = 1.0 / seg_dir[i]
                t1 = (box_min[i] - p1[i]) * inv_d
                t2 = (box_max[i] - p1[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmax < tmin:
                    return False
        return True
    
    def collision_free_edge(self, p1, p2):
        """
        线段p1->p2与任意障碍碰撞则返回False
        """
        for _, obs in self.obstacles.items():
            cx, cy, cz = obs["x"], obs["y"], obs["z"]
            dx, dy, dz = obs["d"], obs["h"], obs["w"]
            box_min = (cx - dx/2, cy - dy/2, cz - dz/2)
            box_max = (cx + dx/2, cy + dy/2, cz + dz/2)
            if self.check_segment_box_intersect(p1, p2, box_min, box_max):
                return False
        return True

    # ========== 核心：构建PRM ==========
    def build_prm(self):
        """
        步骤：
          1) 先将viewpoints强制加入图
          2) 在bounding_box内做num_samples次随机采样，如不在障碍内则加入图
          3) 对所有节点(含viewpoints + 采样点)，找k近邻，若线段无碰撞就加边
        """
        box_min, box_max = self.bounding_box

        node_id = 0

        # ---- (1) 把viewpoints强制做图节点 ----
        for vp in self.viewpoints:
            self.G.add_node(node_id)
            self.idx_to_xyz[node_id] = vp
            node_id += 1

        # ---- (2) 采样普通节点 ----
        sample_nodes = []
        for _ in range(self.num_samples):
            x = random.uniform(box_min[0], box_max[0])
            y = random.uniform(box_min[1], box_max[1])
            z = random.uniform(box_min[2], box_max[2])
            p = (x,y,z)
            if not self.is_in_collision(p):
                self.G.add_node(node_id)
                self.idx_to_xyz[node_id] = p
                sample_nodes.append(node_id)
                node_id += 1
        
        # ---- (3) 构造Edges: k近邻 + 碰撞检测 ----
        all_node_ids = list(self.G.nodes)
        for i in all_node_ids:
            pi = self.idx_to_xyz[i]
            # 计算到其他点的距离
            dist_list = []
            for j in all_node_ids:
                if j == i: 
                    continue
                pj = self.idx_to_xyz[j]
                dist_ij = math.dist(pi, pj)
                dist_list.append((dist_ij, j))
            dist_list.sort(key=lambda x: x[0])
            # 只连最近k个
            for _, j in dist_list[:self.k]:
                pj = self.idx_to_xyz[j]
                if self.collision_free_edge(pi, pj):
                    self.G.add_edge(i, j, weight=math.dist(pi,pj))

    def plan_path(self, start_xyz, goal_xyz):
        """
        在PRM图中加入临时start, goal, 然后A*搜索
        """
        if len(self.G.nodes) == 0:
            print("Graph is empty!")
            return None
        
        # 创建临时节点
        max_id = max(self.G.nodes) if len(self.G.nodes)>0 else 0
        sid, gid = max_id+1, max_id+2
        self.G.add_node(sid)
        self.G.add_node(gid)
        self.idx_to_xyz[sid] = start_xyz
        self.idx_to_xyz[gid]  = goal_xyz

        # 连接start, goal与其他节点(若collision_free)
        for n in list(self.G.nodes):
            if n in [sid, gid]:
                continue
            pn = self.idx_to_xyz[n]
            dist_s = math.dist(start_xyz, pn)
            dist_g = math.dist(goal_xyz,  pn)
            if self.collision_free_edge(start_xyz, pn):
                self.G.add_edge(sid, n, weight=dist_s)
            if self.collision_free_edge(goal_xyz, pn):
                self.G.add_edge(gid, n, weight=dist_g)

        # A*搜索
        def heuristic(u, v):
            return math.dist(self.idx_to_xyz[u], self.idx_to_xyz[v])

        try:
            node_path = nx.astar_path(self.G, sid, gid, heuristic=heuristic, weight='weight')
        except nx.NetworkXNoPath:
            # 清理并返回None
            self.G.remove_node(sid)
            self.G.remove_node(gid)
            return None

        # 提取轨迹
        path_3d = [self.idx_to_xyz[nid] for nid in node_path]

        # 移除临时节点
        self.G.remove_node(sid)
        self.G.remove_node(gid)
        return path_3d

    # ========== 可视化函数 ==========
    def plot_prm_3d(self, path_3d=None):
        """
        用matplotlib 3D绘制PRM节点、边、障碍物，以及可选的规划路径
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 1) 画节点
        node_xyz = [self.idx_to_xyz[nid] for nid in self.G.nodes]
        xs = [p[0] for p in node_xyz]
        ys = [p[1] for p in node_xyz]
        zs = [p[2] for p in node_xyz]
        ax.scatter(xs, ys, zs, marker='o', c='b', s=8, alpha=0.5, label='PRM Nodes')
        # if the coordinates equals to the viewpoint, mark it as red
        for vp in self.viewpoints:
            if vp in node_xyz:
                idx = node_xyz.index(vp)
                ax.scatter(xs[idx], ys[idx], zs[idx], marker='o', c='r', s=20)

        # 2) 画边
        for (u,v) in self.G.edges:
            p1 = self.idx_to_xyz[u]
            p2 = self.idx_to_xyz[v]
            ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], 'gray', linewidth=0.5)

        # 3) 画障碍
        for _, obs in self.obstacles.items():
            self.draw_aabb(ax, obs)
        
        # 4) 如果有path，则画路径
        if path_3d is not None and len(path_3d)>1:
            px = [p[0] for p in path_3d]
            py = [p[1] for p in path_3d]
            pz = [p[2] for p in path_3d]
            ax.plot(px, py, pz, c='r', linewidth=2, label='Solution Path')
            ax.scatter(px, py, pz, c='r', marker='^', s=20)

        # 5) 设置坐标范围
        (box_min, box_max) = self.bounding_box
        ax.set_xlim(box_min[0], box_max[0])
        ax.set_ylim(box_min[1], box_max[1])
        ax.set_zlim(box_min[2], box_max[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()

    def draw_aabb(self, ax, obs):
        """
        在Axes3D上画一个 axis-aligned box (中心(cx,cy,cz), 尺寸(dx,dy,dz)).
        """
        cx, cy, cz = obs['x'], obs['y'], obs['z']
        dx, dy, dz = obs['d'], obs['h'], obs['w']
        x1, x2 = cx - dx/2, cx + dx/2
        y1, y2 = cy - dy/2, cy + dy/2
        z1, z2 = cz - dz/2, cz + dz/2
        
        corners = [
            (x1,y1,z1), (x1,y1,z2), (x1,y2,z1), (x1,y2,z2),
            (x2,y1,z1), (x2,y1,z2), (x2,y2,z1), (x2,y2,z2)
        ]
        edges = [
            (0,1), (0,2), (0,4),
            (3,1), (3,2), (3,7),
            (5,1), (5,4), (5,7),
            (6,2), (6,4), (6,7)
        ]
        for (i,j) in edges:
            (x_i,y_i,z_i) = corners[i]
            (x_j,y_j,z_j) = corners[j]
            ax.plot([x_i,x_j], [y_i,y_j], [z_i,z_j], linestyle='--', color='g', linewidth=1)


# =========== 小demo: how to use =========== 
if __name__ == "__main__":
    scenario_file = "scenarios/scenario4.yaml"  # 你自定义
    with open(scenario_file,"r") as f:
        scenario = yaml.safe_load(f)
    
    # 假设 viewpoint_poses 是 dict: {id: {x,y,z,...}, ...}
    viewpoint_list = []
    if "viewpoint_poses" in scenario:
        vp_keys = sorted(list(scenario["viewpoint_poses"].keys()), key=int)
        for k in vp_keys:
            vp = scenario["viewpoint_poses"][k]
            viewpoint_list.append((vp["x"], vp["y"], vp["z"]))
    else:
        viewpoint_list = []

    bounding_box = ((-10, -10, 0), (10, 10, 6))
    obstacles = scenario.get("obstacles", {})

    # 初始化PRM, 采样200个随机节点
    planner = PRM3DPlanner(
        viewpoints=viewpoint_list,
        num_samples=200,
        obstacles=obstacles,
        bounding_box=bounding_box,
        k=10
    )
    print("PRM Graph built with nodes:", len(planner.G.nodes))
    print("PRM Graph built with edges:", len(planner.G.edges))

    # 举例：规划从(0,0,1) ->(8,8,3)
    start_xyz = (0,0,1)
    goal_xyz  = (8,8,3)
    path_3d = planner.plan_path(start_xyz, goal_xyz)

    if path_3d is None:
        print("No path found!")
        planner.plot_prm_3d(path_3d=None)
    else:
        length = sum(math.dist(path_3d[i], path_3d[i+1]) for i in range(len(path_3d)-1))
        print(f"Found path with {len(path_3d)} points, length={length:.2f}")
        # 可视化
        planner.plot_prm_3d(path_3d=path_3d)
