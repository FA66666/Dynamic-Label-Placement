class Force:
    def __init__(self, force_type, magnitude, direction=(0, 0)):
        """
        :param force_type: 力的类型（如 'label-label', 'label-feature' 等）
        :param magnitude: 力的大小
        :param direction: 力的方向，默认是(0, 0)表示作用力
        """
        self.force_type = force_type
        self.magnitude = magnitude
        self.direction = direction  # 可以用(x, y)表示方向

    def apply(self, label):
        """
        计算该力对标签的影响，更新标签的速度和位置
        :param label: 需要应用力的标签
        :return: 返回新的力影响的方向或加速度
        """

        def apply(self, label, feature=None):
            """
            计算该力对标签的影响，更新标签的速度和位置
            :param label: 需要应用力的标签
            :param feature: 需要应用力的特征（仅适用于标签-特征力）
            :return: 返回新的力影响的方向或加速度
            """
            if self.force_type == "label-label":
                return self.label_label_repulsion(label)
            elif self.force_type == "label-feature":
                return self.label_feature_attraction(label, feature)
            elif self.force_type == "label-pull":
                return self.label_pull(label, feature)
            elif self.force_type == "friction":
                return self.friction(label)
            elif self.force_type == "time":
                return self.time_constraint(label, feature)
            elif self.force_type == "space":
                return self.space_constraint(label, feature)
            # 其他类型的力可以在这里扩展
            return (0, 0)  # 如果没有应用力，返回零
