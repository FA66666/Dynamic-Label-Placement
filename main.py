from Global_spatiotemporal_joint_optimization import LabelOptimizer, params
from initialize import initialize_features_from_gif
from label import Label

if __name__ == '__main__':


    # 读取图片
    gif_path = 'input.gif'  # 替换为你的GIF文件路径
    features = initialize_features_from_gif(gif_path)

    # 创建标签对象（假设每个特征对应一个标签）
    labels = []
    for feature in features:
        # 初始化标签位置为特征初始位置的右侧
        initial_x = feature.position[0] + feature.radius + 20  # 初始位置在特征右侧
        initial_y = feature.position[1]
        label = Label(
            id=feature.id,
            feature=feature,
            position=(initial_x, initial_y),
        )
        labels.append(label)

    # 设置可视区域边界（假设为800x600）
    max_x = 1000
    max_y = 1000

    # print(labels)
    # 执行全局静态优化
    optimizer = LabelOptimizer(labels, features, params, max_x, max_y)
    optimized_labels = optimizer.optimize()

    # print(optimized_labels)


    # 输出每个要素的信息（ID、初始位置、速度、轨迹）
    for label in optimized_labels:
        print(f"label ID: {label.id}")
        print(f"  Initial Position: {label.position}")
        print(f"  Velocity: {label.velocity}")
        print(f"  Feature: {label.feature}")

