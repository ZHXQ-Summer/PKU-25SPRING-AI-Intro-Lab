import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 0.75
TARGET_THREHOLD = 0.25
MAX_ITER=50000
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###     
        walls_x = walls[:, 0]
        walls_y = walls[:, 1]
        self.x_min, self.x_max = np.min(walls_x), np.max(walls_x)
        self.y_min, self.y_max = np.min(walls_y), np.max(walls_y)
        self.x_low = self.x_min + 0.5  # 障碍物边界外扩0.5单位
        self.x_high = self.x_max - 0.5
        self.y_low = self.y_min + 0.5
        self.y_high = self.y_max - 0.5
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        self.cur_idx=[1,0]
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      

        ### 你的代码 ###
        # 如有必要，此行可删除
        if(self.map.checkoccupy(next_food)):
            return[]
        self.path = self.build_tree(current_position, next_food)
        self.cur_idx=[1,0]
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        if self.path: 
            if(self.cur_idx[1]>3 and self.cur_idx[0]<len(self.path)-1):
                self.cur_idx[0]+=1
                self.cur_idx[1]=0
            target_pose = self.path[self.cur_idx[0]]
            self.cur_idx[1]+=1  
        else:          # 路径为空时
            target_pose = current_position.copy()
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        ### 你的代码 ###
        for _ in range(MAX_ITER):
            # 随机采样（包含目标偏向）
            if np.random.rand() < 0.15:  # 15%概率采样目标点
                rand_point = goal
            else:
                # 在可行区域内生成随机点
                while True:
                    rand_x = np.random.uniform(self.x_low, self.x_high)
                    rand_y = np.random.uniform(self.y_low, self.y_high)
                    if not self.map.checkoccupy([rand_x, rand_y]):
                        rand_point = np.array([rand_x, rand_y])
                        break
            
            # 寻找最近节点
            nearest_idx, _ = self.find_nearest_point(rand_point, graph)
            nearest_node = graph[nearest_idx]
            success, new_point = self.connect_a_to_b(nearest_node.pos, (rand_point))
            if success:
                new_node = TreeNode(nearest_idx, new_point[0], new_point[1])
                graph.append(new_node)
                
                # 接近目标则终止
                check,_=self.map.checkline([new_point[0]-1e-6, new_point[1]-1e-6],goal.tolist())
                if np.linalg.norm(new_point - goal) < 1e-2 :
                    return self.extract_path(graph, len(graph)-1)
        
        return []  # 未找到路径
        
        ### 你的代码 ###
        return path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        min_dist = 10000000.
        ### 你的代码 ###
        for i, node in enumerate(graph):
            dist = np.linalg.norm(point - node.pos)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx, min_dist

    
    def connect_a_to_b(self, a, b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        new_point = np.zeros(2)
        ### 你的代码 ###
        direction = b - a
        distance = np.linalg.norm(direction)
        
        if distance < 1e-20:
            return False, a.copy()
        
        # 计算扩展步长
        step_size = min(STEP_DISTANCE, distance)
        new_point = a + (direction / distance) * step_size
        
        # 执行碰撞检测
        collision,_ = self.map.checkline([a[0], a[1]],[new_point[0], new_point[1]])
        check=self.map.checkoccupy([new_point[0], new_point[1]])
        return (not collision) and (not check), new_point
    def extract_path(self, graph, goal_idx):
        """路径回溯"""
        path = []
        current_idx = goal_idx
        while current_idx != -1:
            path.append(graph[current_idx].pos)
            current_idx = graph[current_idx].parent_idx
        return path[::-1]  # 反转路径顺序

