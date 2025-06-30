class Feature:
    def __init__(self, id, color,position, velocity, radius=5):
        """
        :param id: 特征的唯一标识符
        :param position: 特征的当前位置，使用二维坐标 (x, y)
        :param velocity: 特征的速度，使用二维速度向量 (vx, vy)
        :param radius: 可选，特征的大小（例如，半径），如果是点特征可以为空
        """
        self.id = id
        self.color = color  # 颜色名称
        self.position = position  # 当前特征位置 (x, y)
        self.velocity = velocity  # 当前特征速度 (vx, vy)
        self.radius = radius  # 特征半径，可选
        self.trajectory = [position]  # 初始化轨迹，存储过去的位置 (x, y)

    def move(self, time_delta):
        """
        根据时间增量更新特征的位置并记录其轨迹
        :param time_delta: 时间增量
        """
        # 计算新位置
        self.position = (self.position[0] + self.velocity[0] * time_delta,
                         self.position[1] + self.velocity[1] * time_delta)

        # 将新位置添加到轨迹中
        self.trajectory.append(self.position)

    def get_trajectory(self):
        """
        获取特征的运动轨迹
        :return: 轨迹数组，包含所有历史位置
        """
        return self.trajectory
