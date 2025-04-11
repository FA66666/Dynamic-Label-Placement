from feature import Feature

def initialize_features_from_data(red_positions, green_positions, blue_positions, frame_interval=0.05):
    """
    使用轨迹数据初始化 Feature 对象列表
    
    :param red_positions: 红色点轨迹列表 [(x, y), ...]
    :param green_positions: 绿色点轨迹列表 [(x, y), ...]
    :param blue_positions: 蓝色点轨迹列表 [(x, y), ...]
    :param frame_interval: 帧之间的时间间隔（秒）
    :return: 包含三个 Feature 对象的列表
    """
    def compute_velocity(positions):
        """计算初始速度"""
        if len(positions) < 2:
            return (0.0, 0.0)
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        return (dx / frame_interval, dy / frame_interval)

    # 创建特征对象
    features = []
    colors = ['red', 'green', 'blue']
    positions = [red_positions, green_positions, blue_positions]
    
    for i, (color, pos_list) in enumerate(zip(colors, positions)):
        feature = Feature(
            id=i,
            color=color,
            position=pos_list[0],
            velocity=compute_velocity(pos_list),
            radius=5
        )
        feature.trajectory = pos_list
        features.append(feature)
    
    return features

# 示例调用（假设已通过 generate_features() 生成轨迹数据）
if __name__ == '__main__':
    # 生成轨迹数据（假设已通过 generate_features() 生成）
    from generate_data import generate_features  # 替换为你的生成函数路径
    features = generate_features()
    red_positions = features[0].trajectory
    green_positions = features[1].trajectory
    blue_positions = features[2].trajectory

    # 使用轨迹数据初始化 Feature 对象
    new_features = initialize_features_from_data(
        red_positions,
        green_positions,
        blue_positions,
        frame_interval=0.05  # 假设帧间隔为 50ms（0.05秒）
    )

    # 输出结果
    for feature in new_features:
        print(f"Feature ID: {feature.id}")
        print(f"  Color: {feature.color}")
        print(f"  Initial Position: {feature.position}")
        print(f"  Velocity: {feature.velocity}")
        print(f"  Trajectory_Length: {len(feature.trajectory)}")
        print(f"  Trajectory: {feature.trajectory[:3]} ...")  # 显示前3个点