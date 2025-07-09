from feature import Feature

def initialize_features_from_data(positions_list, frame_interval=0.05):
    """
    使用轨迹数据初始化 Feature 对象列表
    
    :param positions_list: 包含所有特征点轨迹的列表
    :param frame_interval: 帧之间的时间间隔（秒）
    :return: 包含所有 Feature 对象的列表
    """
    def compute_velocity(positions):
        """计算初始速度"""
        if len(positions) < 2:
            return (0.0, 0.0)
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        return (dx / frame_interval, dy / frame_interval)

    # 定义7个特征点的颜色
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']
    
    # 创建特征对象
    features = []
    for i, (color, pos_list) in enumerate(zip(colors, positions_list)):
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
