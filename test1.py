import math
import random
import matplotlib.pyplot as plt


# -------------------------------
# 辅助向量函数
# -------------------------------
def vec_sub(a, b):
    """向量减法：a - b"""
    return (a[0] - b[0], a[1] - b[1])


def vec_norm(v):
    """计算向量长度"""
    return math.hypot(v[0], v[1])


def vec_unit(v):
    """单位向量（当向量为零时返回(0,0)）"""
    norm = vec_norm(v)
    if norm == 0:
        return (0.0, 0.0)
    return (v[0] / norm, v[1] / norm)


# -------------------------------
# 参数设置
# -------------------------------
# 模拟大数据集特征数量
NUM_FEATURES = 300
# 设置模拟帧数
NUM_FRAMES = 50
dt = 1.0  # 每帧时间步长

# 阈值与权重（参考论文中相应公式）
D_label_collision = 30.0  # 标签-标签排斥作用阈值
D_feature_collision = 20.0  # 标签-特征排斥作用阈值
D_pull = 15.0  # 拉力作用的最小距离阈值
c_friction = 0.7  # 摩擦力系数

# 各力权重（可调）
w_label_collision = 50.0
w_feature_collision = 50.0
w_pull = 25.0
w_friction = 6.0
w_space = 20.0
w_time = 15.0

# -------------------------------
# 数据生成：随机生成特征及对应标签
# -------------------------------
features = {}
labels = {}
constraint_pos = {}  # 空间约束目标位置（通常由全局优化给出，这里简单设定为：特征位置上方固定偏移）

# 为了方便观察，将特征分布在一个较大区域内
AREA_WIDTH = 1000
AREA_HEIGHT = 1000

for i in range(1, NUM_FEATURES + 1):
    # 随机特征位置，速度和半径（半径相对较小）
    pos = (random.uniform(0, AREA_WIDTH), random.uniform(0, AREA_HEIGHT))
    # 速度随机：方向随机，速度在 0~5 像素/帧之间
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(0, 5)
    vel = (speed * math.cos(angle), speed * math.sin(angle))
    features[i] = {"pos": pos, "vel": vel, "r": random.uniform(3, 6)}
    # 标签初始位置：在特征上方（比如加一个固定的偏移），标签尺寸较大
    offset = (0, random.uniform(15, 25))
    label_pos = (pos[0] + offset[0], pos[1] + offset[1])
    labels[i] = {"feature": i, "pos": label_pos, "vel": (0.0, 0.0), "s": random.uniform(40, 60)}
    # 空间约束目标位置：这里设定为特征上方一个理想位置（例如离特征固定距离）
    constraint_pos[i] = (pos[0], pos[1] + 30)


# -------------------------------
# 定义各力的计算函数
# -------------------------------

def label_label_collision_force(id_i, id_j):
    p_i, p_j = labels[id_i]["pos"], labels[id_j]["pos"]
    # 假设标签为矩形，s为宽度，这里简单用一半宽度作为半径
    r_i, r_j = labels[id_i]["s"] / 2.0, labels[id_j]["s"] / 2.0
    diff = vec_sub(p_i, p_j)
    dist = vec_norm(diff)
    d_gap = max(dist - (r_i + r_j), 0.0)
    if dist == 0:
        return (0.0, 0.0)
    factor = min(d_gap / D_label_collision - 1.0, 0.0)
    direction = vec_unit(diff)
    return (factor * direction[0], factor * direction[1])


def label_feature_collision_force(id_i, feature_j):
    p_i = labels[id_i]["pos"]
    l_j = features[feature_j]["pos"]
    r_label = labels[id_i]["s"] / 2.0
    r_feat = features[feature_j]["r"]
    diff = vec_sub(p_i, l_j)
    dist = vec_norm(diff)
    d_gap = max(dist - (r_label + r_feat), 0.0)
    if dist == 0:
        return (0.0, 0.0)
    factor = min(d_gap / D_feature_collision - 1.0, 0.0)
    direction = vec_unit(diff)
    return (factor * direction[0], factor * direction[1])


def feature_label_pull_force(id_i):
    p_i = labels[id_i]["pos"]
    feat_id = labels[id_i]["feature"]
    l_i = features[feat_id]["pos"]
    r_label = labels[id_i]["s"] / 2.0
    r_feat = features[feat_id]["r"]
    diff = vec_sub(p_i, l_i)
    dist = vec_norm(diff)
    if dist - (r_label + r_feat) <= D_pull:
        return (0.0, 0.0)
    mag = -math.log((dist - (r_label + r_feat)) / D_pull + 1.0)
    direction = vec_unit(diff)
    return (mag * direction[0], mag * direction[1])


def friction_force(id_i):
    feat_id = labels[id_i]["feature"]
    v_label = labels[id_i]["vel"]
    v_feat = features[feat_id]["vel"]
    return (-c_friction * (v_label[0] - v_feat[0]),
            -c_friction * (v_label[1] - v_feat[1]))


def space_constraint_force(id_i):
    if constraint_pos.get(id_i) is None:
        return (0.0, 0.0)
    p_i = labels[id_i]["pos"]
    target = constraint_pos[id_i]
    diff = vec_sub(p_i, target)
    dist = vec_norm(diff)
    if dist == 0:
        return (0.0, 0.0)
    mag = math.log(dist + 1.0)
    direction = vec_unit(diff)
    # 为了将标签拉向目标，这里实际可取负方向（根据具体需求调整）
    return (-mag * direction[0], -mag * direction[1])


def time_constraint_force(id_i, neighbor_feat_id, dt):
    # 对于标签 i，计算来自邻居特征 neighbor_feat_id 的时间约束力
    feat_i = labels[id_i]["feature"]
    v_i = features[feat_i]["vel"]
    v_j = features[neighbor_feat_id]["vel"]
    l_j_current = features[neighbor_feat_id]["pos"]
    # 预测未来位置：l'_j = l_j_current + (v_j - v_i) * dt
    rel_future_pos = (l_j_current[0] + (v_j[0] - v_i[0]) * dt,
                      l_j_current[1] + (v_j[1] - v_i[1]) * dt)
    speed_i = vec_norm(v_i)
    speed_j = vec_norm(v_j)
    if speed_i == 0 or speed_j == 0:
        vel_ratio_log = math.log(max(speed_i, speed_j) / 1e-5 + 1e-5)
    else:
        vel_ratio_log = math.log(max(speed_i, speed_j) / min(speed_i, speed_j))
    p_i = labels[id_i]["pos"]
    dist_now = vec_norm(vec_sub(p_i, l_j_current))
    dist_future = vec_norm(vec_sub(p_i, rel_future_pos))
    if dist_future == 0 or dist_now == 0:
        return (0.0, 0.0)
    dist_ratio_factor = min(dist_now / dist_future - 1.0, 0.0)
    direction = vec_unit(vec_sub(p_i, rel_future_pos))
    return (vel_ratio_log * dist_ratio_factor * direction[0],
            vel_ratio_log * dist_ratio_factor * direction[1])


# -------------------------------
# 模拟迭代：更新标签位置（大数据集）
# -------------------------------
# 为了记录运动轨迹，保存每帧标签的位置（仅记录部分标签或聚合信息）
trajectory = {i: [labels[i]["pos"]] for i in labels}

for frame in range(NUM_FRAMES):
    # 对每个标签，计算所有作用力，并更新状态
    for i in labels:
        total_fx = total_fy = 0.0

        # 1. 标签-标签碰撞力与来自邻居的时间约束力
        for j in labels:
            if i == j:
                continue
            fx_ll, fy_ll = label_label_collision_force(i, j)
            # 使用标签 j 所属的特征作为时间约束力的来源
            fx_time, fy_time = time_constraint_force(i, labels[j]["feature"], dt)
            total_fx += w_label_collision * fx_ll + w_time * fx_time
            total_fy += w_label_collision * fy_ll + w_time * fy_time

        # 2. 标签-特征碰撞力：对所有特征计算（包括自身）
        for fid in features:
            fx_lf, fy_lf = label_feature_collision_force(i, fid)
            total_fx += w_feature_collision * fx_lf
            total_fy += w_feature_collision * fy_lf

        # 3. 特征–标签拉力（使标签靠近对应特征）
        fx_pull, fy_pull = feature_label_pull_force(i)
        total_fx += w_pull * fx_pull
        total_fy += w_pull * fy_pull

        # 4. 摩擦力
        fx_fric, fy_fric = friction_force(i)
        total_fx += w_friction * fx_fric
        total_fy += w_friction * fy_fric

        # 5. 空间约束力（引导标签向全局优化给出的目标位置移动）
        fx_space, fy_space = space_constraint_force(i)
        total_fx += w_space * fx_space
        total_fy += w_space * fy_space

        # 6. 用合力计算加速度（质量 m=1），更新速度和位置
        ax, ay = total_fx, total_fy
        vx, vy = labels[i]["vel"]
        new_v = (vx + ax * dt, vy + ay * dt)
        labels[i]["vel"] = new_v
        px, py = labels[i]["pos"]
        new_p = (px + new_v[0] * dt, py + new_v[1] * dt)
        labels[i]["pos"] = new_p

        # 保存轨迹数据
        trajectory[i].append(new_p)

# -------------------------------
# 绘制结果：展示特征与标签最终位置（以及部分轨迹）
# -------------------------------
plt.figure(figsize=(10, 10))
# 绘制所有特征位置（蓝色点）
feat_x = [features[i]["pos"][0] for i in features]
feat_y = [features[i]["pos"][1] for i in features]
plt.scatter(feat_x, feat_y, c='blue', label='Features', s=20)
# 绘制所有标签最终位置（红色点）
label_x = [labels[i]["pos"][0] for i in labels]
label_y = [labels[i]["pos"][1] for i in labels]
plt.scatter(label_x, label_y, c='red', label='Labels', s=20)
# 绘制部分标签的轨迹（例如随机选取 10 个标签）
for i in random.sample(list(trajectory.keys()), 10):
    traj = trajectory[i]
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    plt.plot(xs, ys, '--', linewidth=1)
    # 用一条线连接特征与最终标签（用虚线表示引导线）
    feat = features[labels[i]["feature"]]["pos"]
    plt.plot([feat[0], labels[i]["pos"][0]], [feat[1], labels[i]["pos"][1]], 'k--', linewidth=0.5)

plt.title("Dynamic Label Layout on Large Dataset (After {} Frames)".format(NUM_FRAMES))
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
