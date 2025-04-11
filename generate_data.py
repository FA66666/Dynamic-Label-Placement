import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_trajectories():
    """
    生成三个点的轨迹数据
    返回: (red_positions, green_positions, blue_positions)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)  # 修改y轴范围以匹配坐标系统

    # 定义两个外切圆的参数
    O1 = (300.0, 500.0)  # 修改y坐标以匹配新的坐标系统
    O2 = (700.0, 500.0)  # 修改y坐标以匹配新的坐标系统
    r1 = 200.0
    r2 = 200.0

    # 绿点始终位于圆心连线的中点
    green_position = (500.0, 500.0)

    red_dot, = ax.plot([], [], 'ro', markersize=10)
    blue_dot, = ax.plot([], [], 'bo', markersize=10)
    green_dot, = ax.plot([], [], 'go', markersize=10)

    red_positions = []
    blue_positions = []
    green_positions = []

    def init():
        red_dot.set_data([], [])
        blue_dot.set_data([], [])
        green_dot.set_data([], [])
        return red_dot, blue_dot, green_dot

    def animate(frame):
        theta = (2 * np.pi * frame) / 100  # 100帧完成一个周期

        # 红点轨迹：绕 O1 做圆周运动
        x_red = O1[0] + r1 * np.cos(theta)
        y_red = O1[1] + r1 * np.sin(theta)

        # 蓝点轨迹：绕 O2 做圆周运动，相位与红点相反
        x_blue = O2[0] + r2 * np.cos(theta + np.pi)
        y_blue = O2[1] + r2 * np.sin(theta + np.pi)

        # 绿点始终位于圆心连线的中点
        x_green = green_position[0]
        y_green = green_position[1]

        # 存储坐标（强制转换为 float）
        red_positions.append((float(x_red), float(y_red)))
        blue_positions.append((float(x_blue), float(y_blue)))
        green_positions.append((float(x_green), float(y_green)))

        # 更新点的位置
        red_dot.set_data([x_red], [y_red])
        blue_dot.set_data([x_blue], [y_blue])
        green_dot.set_data([x_green], [y_green])

        return red_dot, blue_dot, green_dot

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=100,
        interval=50,
        blit=True
    )

    anim.save('input.gif', 
              writer='pillow', 
              dpi=100,
              savefig_kwargs={'facecolor': 'white'})

    plt.close()
    return red_positions, green_positions, blue_positions

if __name__ == '__main__':
    red_positions, green_positions, blue_positions = generate_trajectories()
    print(f"轨迹数据生成完成，每个轨迹包含 {len(red_positions)} 个点")
