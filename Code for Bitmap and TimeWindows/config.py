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

# 候选位置生成的搜索半径和每圈点数
radii_to_search = [25, 35, 45, 55, 65, 75]
points_per_circle = 24

# 标签锚点周围的间隔半径
LABEL_RADIUS = 2

# 碰撞预测的未来帧数窗口
PREDICTION_WINDOW = 5

# 时间步长 (单位：帧)
DT = 1.0

# 位图算法中用于位压缩的整数位数 (通常为32或64)
N_VAL = 32

# ----------------------------------------------------------------------
# 成本函数权重 (Cost Function Weights)
# ----------------------------------------------------------------------
# W_OVERLAP: 重叠面积成本的权重。必须设为一个非常大的值，以确保算法最优先解决重叠问题。
# W_MOVE: 移动距离成本的权重。
# W_LINE: 引导线长度成本的权重。
# W_ANGLE: 角度变化成本的权重。
W_OVERLAP = 1000000  
W_MOVE = 10
W_LINE = 70
W_ANGLE = 150

# ----------------------------------------------------------------------
# 卡尔曼滤波器调优参数 (Kalman Filter Tuning)
# ----------------------------------------------------------------------
# 过程噪声协方差
KALMAN_Q = 0.1

# 测量噪声协方差
KALMAN_R = 1.0

# 初始估计误差协方差
KALMAN_P_INITIAL = 100

# ----------------------------------------------------------------------
# 可视化输出配置 (Visualization Output Configuration)
# ----------------------------------------------------------------------
GIF_DURATION_MS = 50
BG_COLOR = 'white'
BOX_OUTLINE_COLOR = 'black'
BOX_FILL_COLOR = '#f0f0f0'
TEXT_COLOR = 'black'
ANCHOR_COLOR = 'red'
ANCHOR_RADIUS = 4
LINE_COLOR = 'grey'
FONT_NAME = "Arial.ttf"
FONT_SIZE = 12
GIF_OUTPUT_FILENAME = 'output.gif'