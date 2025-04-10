from PIL import Image, ImageSequence
import numpy as np

from feature import Feature


def initialize_features_from_gif(gif_path, frame_interval=1):
    """
    从包含三个运动点（红、绿、蓝）的 GIF 动图中提取要素点数据，
    返回一个包含三个 Feature 对象的列表。

    :param gif_path: GIF 文件路径
    :param frame_interval: 帧之间的时间间隔（单位可为秒），用于计算速度
    :return: 包含三个 Feature 对象的列表，分别代表红、绿、蓝三个要素
    """
    # 打开 GIF 文件
    gif = Image.open(gif_path)

    # 用于存储每个颜色对应的点在各帧中的中心位置
    red_positions = []
    green_positions = []
    blue_positions = []

    # 遍历 GIF 的每一帧
    for frame in ImageSequence.Iterator(gif):
        # 转换为 RGB 格式
        frame = frame.convert("RGB")
        arr = np.array(frame)  # shape: (height, width, 3)

        # 根据颜色阈值检测红、绿、蓝点
        red_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
        # 修改绿色的颜色范围
        green_mask = (arr[:, :, 1] > 150) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100)
        blue_mask = (arr[:, :, 2] > 200) & (arr[:, :, 0] < 50) & (arr[:, :, 1] < 50)

        # 找到各个颜色点的像素位置（注意 np.argwhere 返回 (row, col)）
        red_coords = np.argwhere(red_mask)
        green_coords = np.argwhere(green_mask)
        blue_coords = np.argwhere(blue_mask)

        # 若检测到目标颜色的点，则计算其质心（centroid）
        if red_coords.shape[0] > 0:
            red_centroid = np.mean(red_coords, axis=0)
            # 转换为 float 类型
            red_positions.append((float(red_centroid[1]), float(red_centroid[0])))
        else:
            red_positions.append(None)

        if green_coords.shape[0] > 0:
            green_centroid = np.mean(green_coords, axis=0)
            # 转换为 float 类型
            green_positions.append((float(green_centroid[1]), float(green_centroid[0])))
        else:
            green_positions.append(None)

        if blue_coords.shape[0] > 0:
            blue_centroid = np.mean(blue_coords, axis=0)
            # 转换为 float 类型
            blue_positions.append((float(blue_centroid[1]), float(blue_centroid[0])))
        else:
            blue_positions.append(None)
            
        # 当前帧索引
        current_frame = len(red_positions) - 1
        
        # 获取当前帧的三个点的数据
        current_positions = [red_positions[current_frame], green_positions[current_frame], blue_positions[current_frame]]
        
        # 计算当前帧中非空点的数量
        non_none_count = sum(1 for pos in current_positions if pos is not None)
        
        # 如果有1或2个空点，则用非空点数据填充
        if 0 < non_none_count < 3:
            # 找到第一个非空点
            first_valid_position = next((pos for pos in current_positions if pos is not None), None)
            
            # 用找到的非空点数据填充空点
            if red_positions[current_frame] is None:
                red_positions[current_frame] = first_valid_position
            if green_positions[current_frame] is None:
                green_positions[current_frame] = first_valid_position
            if blue_positions[current_frame] is None:
                blue_positions[current_frame] = first_valid_position

    # 定义一个辅助函数计算初始速度（用前两帧的位移计算平均速度）
    def compute_velocity(positions):
        if len(positions) < 2 or positions[0] is None or positions[1] is None:
            return (0.0, 0.0)  # 返回 float 类型
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        return (dx / frame_interval, dy / frame_interval)

    red_velocity = compute_velocity(red_positions)
    green_velocity = compute_velocity(green_positions)
    blue_velocity = compute_velocity(blue_positions)

    # 对于点特征，可以设定一个常量半径，例如 5
    red_radius = 5
    green_radius = 5
    blue_radius = 5

    # 初始化 Feature 对象，取各自轨迹的第一个非 None 值作为初始位置
    red_initial = next((pos for pos in red_positions if pos is not None), (0.0, 0.0))
    green_initial = next((pos for pos in green_positions if pos is not None), (0.0, 0.0))
    blue_initial = next((pos for pos in blue_positions if pos is not None), (0.0, 0.0))

    # 将颜色名称作为color传入
    red_feature = Feature(id=0, color="red", position=red_initial, velocity=red_velocity, radius=red_radius)
    green_feature = Feature(id=1, color="green", position=green_initial, velocity=green_velocity, radius=green_radius)
    blue_feature = Feature(id=2, color="blue", position=blue_initial, velocity=blue_velocity, radius=blue_radius)

    # 将检测到的每帧位置记录为轨迹（过滤掉 None 值）
    red_feature.trajectory = [pos for pos in red_positions if pos is not None]
    green_feature.trajectory = [pos for pos in green_positions if pos is not None]
    blue_feature.trajectory = [pos for pos in blue_positions if pos is not None]

    return [red_feature, green_feature, blue_feature]



# 示例调用
if __name__ == '__main__':
    gif_path = 'input2.gif'  # 请将此路径修改为你的 GIF 文件路径
    features = initialize_features_from_gif(gif_path)

    # 输出每个要素的信息（ID、颜色、初始位置、速度、轨迹）
    for feature in features:
        print(f"Feature ID: {feature.id}")
        print(f"  Color: {feature.color}")
        print(f"  Initial Position: {feature.position}")
        print(f"  Velocity: {feature.velocity}")
        print(f"  Trajectory_Length: {len(feature.trajectory)}")
        print(f"  Trajectory: {feature.trajectory}")
