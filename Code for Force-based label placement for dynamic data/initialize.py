from feature import Feature

def initialize_features(positions_list,frame_interval = 0.05):
    """
    使用轨迹数据初始化 Feature 对象列表
    
    :param positions_list: 包含所有特征点轨迹的列表
    :param frame_interval: 帧之间的时间间隔（秒）
    :return: 包含所有 Feature 对象的列表
    """

    def compute_velocity(positions):
        if len(positions) < 2:
            return (0, 0)
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]

        return (dx/frame_interval, dy/frame_interval)
    
    features = []
    for i,pos_list in enumerate( positions_list ):
        feature = Feature(
            id = i,
            position = pos_list[0],
            velocity = compute_velocity(pos_list),
            radius = 1
        )
        feature.trajectory = pos_list
        features.append(feature)

    return features