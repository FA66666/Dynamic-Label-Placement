# config.py

# ----------------------------------------------------------------------
# 通用配置 (General Configuration)
# ----------------------------------------------------------------------
# 场景尺寸
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

# 文件路径
INPUT_FILE = 'sample_input.json'
OUTPUT_FILE = 'output_positions.json'

# ----------------------------------------------------------------------
# 核心算法配置 (Core Algorithm Configuration)
# ----------------------------------------------------------------------

radii_to_search = [25, 35, 45, 55, 65]
points_per_circle = 16

# 标签锚点周围的间隔半径
LABEL_RADIUS = 2

# 碰撞预测的未来帧数窗口
PREDICTION_WINDOW = 15

# 时间步长 (单位：帧)
DT = 1.0

# 位图算法中用于位压缩的整数位数 (通常为32或64)
N_VAL = 32

# ----------------------------------------------------------------------
# 卡尔曼滤波器调优参数 (Kalman Filter Tuning)
# ----------------------------------------------------------------------
# 过程噪声协方差 (描述模型预测的不确定性)
# 值越大，意味着越不相信模型的预测，更能适应突变
KALMAN_Q = 0.1

# 测量噪声协方差 (描述传感器观测的不确定性)
# 值越大，意味着越不相信观测值，滤波结果更平滑
KALMAN_R = 1.0

# 初始估计误差协方差
# 一个较大的初始值，表示对初始状态的不确定性很高
KALMAN_P_INITIAL = 100

# ----------------------------------------------------------------------
# 可视化输出配置 (Visualization Output Configuration)
# ----------------------------------------------------------------------
# GIF 每帧的持续时间 (毫秒)，50ms = 20 FPS
GIF_DURATION_MS = 50

# 背景颜色
BG_COLOR = 'white'

# 标签框颜色
BOX_OUTLINE_COLOR = 'black'
BOX_FILL_COLOR = '#f0f0f0'  # 浅灰色

# 文本颜色
TEXT_COLOR = 'black'

# 锚点颜色和大小
ANCHOR_COLOR = 'red'
ANCHOR_RADIUS = 4

# 连线颜色
LINE_COLOR = 'grey'

# 字体设置
FONT_NAME = "Arial.ttf"
FONT_SIZE = 12

# GIF 输出文件名
GIF_OUTPUT_FILENAME = 'output.gif'