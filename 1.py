import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置图形尺寸为 10x10 英寸（1000×1000 像素，当 dpi=100 时）
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_axis_off()

# 设置坐标轴范围
ax.set_xlim(0, 2 * np.pi)  # x 轴范围 0 到 2π
ax.set_ylim(-1000, 1000)   # y 轴范围确保点可见

# 初始化三个点
red_dot, = ax.plot([], [], 'ro', markersize=5)
blue_dot, = ax.plot([], [], 'bo', markersize=5)
green_dot, = ax.plot([], [], 'go', markersize=5)

def init():
    red_dot.set_data([], [])
    blue_dot.set_data([], [])
    green_dot.set_data([], [])
    return red_dot, blue_dot, green_dot

def animate(frame):
    t = frame * (2 * np.pi) / 100
    x = t
    y_red = 800 * np.sin(x)
    y_blue = -800 * np.sin(x)
    y_green = 0.0
    
    red_dot.set_data([x], [y_red])
    blue_dot.set_data([x], [y_blue])
    green_dot.set_data([x], [y_green])
    return red_dot, blue_dot, green_dot

# 创建动画对象
anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=100,
    interval=50,  # 每帧间隔 50 毫秒（即 20 FPS）
    blit=True
)

# 保存为 GIF（分辨率为 1000×1000 像素）
anim.save('animation_1000x1000.gif', 
          writer='pillow', 
          dpi=100, 
          savefig_kwargs={'facecolor': 'white'})  # 设置背景为白色

# 关闭图形
plt.close()