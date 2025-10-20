from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
k=0.28
pos_d=0.21
theta_d=0.21
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """

    all_particles: List[Particle] = []
    x_centers = walls[:, 0]
    y_centers = walls[:, 1]
    
    # 实际可行区域 = 整个地图范围向内缩进0.5单位
    x_min = np.min(x_centers) + 0.5
    x_max = np.max(x_centers) - 0.5
    y_min = np.min(y_centers) + 0.5
    y_max = np.max(y_centers) - 0.5
    # 预存储墙壁坐标加速计算
    walls_x = walls[:, 0]
    walls_y = walls[:, 1]
    # 简单拒绝采样
    while len(all_particles) < N:
        # 在可行区域内随机采样
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        # 向量化碰撞检测（比循环快100倍）
        dx = np.abs(x - walls_x)
        dy = np.abs(y - walls_y)
        if not np.any( (dx <= 0.5) & (dy <= 0.5) ):
            all_particles.append(Particle(x=x, y=y, theta=np.random.uniform(0, 2*np.pi), weight=1.0/N))
    
    return all_particles



def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight=np.exp(-k*np.linalg.norm((estimated-gt)))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles = []
    N = len(particles)
    
    # 初始化空粒子列表
    for _ in range(N):
        resampled_particles.append(Particle(0, 0, 0, 0))  # 初始权重设为0
    
    ### 改进代码 ###
    if N == 0:
        return resampled_particles
    
### 步骤1：计算有效区域边界 ###
    walls_x = walls[:, 0]
    walls_y = walls[:, 1]
    x_min, x_max = walls_x.min(), walls_x.max()
    y_min, y_max = walls_y.min(), walls_y.max()
    x_low, x_high = x_min + 0.5, x_max - 0.5
    y_low, y_high = y_min + 0.5, y_max - 0.5

    ### 步骤2：生成权重数组（越界或碰撞的权重为0） ###
    # 预计算墙壁坐标加速碰撞检测
    walls_centers = walls  # shape (M,2)
    
    weights = []
    for p in particles:
        x, y = p.position
        # 检查越界
        out_of_bound = (x < x_low) or (x > x_high) or (y < y_low) or (y > y_high)
        if out_of_bound:
            weights.append(0.0)
            continue
        
        # 检查墙壁碰撞（向量化检测）
        dx = np.abs(x - walls_centers[:, 0])
        dy = np.abs(y - walls_centers[:, 1])
        collision = np.any( (dx <= 0.5) & (dy <= 0.5) )
        weights.append(0.0 if collision else p.weight)
    
    weights = np.array(weights)
    
    # 归一化处理
    sum_weights = np.sum(weights)
    if sum_weights < 1e-10:  # 所有粒子越界的处理
        weights = np.ones(N) / N  # 退化为均匀分布
    else:
        weights = weights / sum_weights
    
    # 核心采样逻辑
    indices = np.random.choice(N, size=N, p=weights)
    parent_positions = np.array([particles[i].position for i in indices])
    parent_thetas = np.array([particles[i].theta for i in indices])
    

    
    for i in range(N):
        # 生成带噪声的新位置
        new_pos = parent_positions[i] + np.random.normal(0, pos_d, 2)
        new_theta = parent_thetas[i] + np.random.normal(0, theta_d)
        
        
        # 更新粒子属性
        resampled_particles[i].position = new_pos
        resampled_particles[i].theta = new_theta % (2*np.pi)
        resampled_particles[i].weight = 1/N
    
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta += dtheta
    delta_x = traveled_distance * np.cos(p.theta)
    delta_y = traveled_distance * np.sin(p.theta)
    p.position += np.array([delta_x, delta_y])
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ''' ### 你的代码 ###
        # 提取粒子属性（向量化操作加速）
    weights = np.array([p.weight for p in particles])
    positions = np.array([p.position for p in particles])  # shape (N,2)
    thetas = np.array([p.theta for p in particles])        # shape (N,)
    
    # 计算加权平均位置
    sum_weights = np.sum(weights)
    weighted_x = np.dot(positions[:, 0], weights) 
    weighted_y = np.dot(positions[:, 1], weights)
    mean_x = weighted_x / sum_weights
    mean_y = weighted_y / sum_weights
    
    # 计算加权平均朝向（向量平均）
    cos_theta = np.dot(np.cos(thetas), weights) / sum_weights
    sin_theta = np.dot(np.sin(thetas), weights) / sum_weights
    mean_theta = np.arctan2(sin_theta, cos_theta)
    
    # 构建结果对象
    final_result.position = np.array([mean_x, mean_y])          # 位置更新
    final_result.theta = (mean_theta + 2*np.pi) % (2*np.pi)     # 角度归一化到[0,2π)
    final_result.weight = 1.0  
    ### 你的代码 ###'''
    max_weight = -float('inf')
    best_particle = None
    for p in particles:
        if p.weight > max_weight:
            max_weight = p.weight
            best_particle = p
    
    # 复制最佳粒子状态（深拷贝位置数组）
    if best_particle is not None:
        final_result.position = np.copy(best_particle.position)
        final_result.theta = best_particle.theta
        final_result.weight = 1.0  # 最终结果权重设为1
    
    return final_result
    return final_result