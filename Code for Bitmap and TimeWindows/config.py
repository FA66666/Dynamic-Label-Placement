
# ===================== 基本参数 =====================
# 场景宽高
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

# 输入输出文件名
INPUT_FILE = 'sample_input.json'
OUTPUT_FILE = 'output_positions.json'

# ===================== 算法参数 =====================
# 候选位置搜索半径列表
radii_to_search = list(range(15, 90, 5))
# 标签锚点间隔半径
LABEL_RADIUS = 2
# 预测未来帧数
PREDICTION_WINDOW = 15
# 时间步长（帧）
DT = 1.0
# 位图压缩位数
N_VAL = 32

# ===================== 方向预筛选 =====================
# 方向采样数
NUM_DIRECTIONS_TO_EVALUATE = 72
# 进入精算的方向数
TOP_N_DIRECTIONS_TO_SEARCH = 16
# 拥挤度计算空间窗口半径
SPATIAL_WINDOW_RADIUS = 250.0
# 拥挤度函数参数
CLUTTER_K_SENSITIVITY = 2.0  # 距离敏感度
CLUTTER_SIGMA_DEGREES = 135   # 角度影响范围
# 角度连续性惩罚权重
W_PRESCREEN_ANGLE = 2.5

# ===================== 成本函数权重 =====================
# 重叠、移动、引导线、角度变化权重
W_OVERLAP = 10000000
W_MOVE = 1.5
W_LINE = 70
W_ANGLE = 12

# ===================== 卡尔曼滤波参数 =====================
# 过程噪声、测量噪声、初始误差
KALMAN_Q = 0.1
KALMAN_R = 1.0
KALMAN_P_INITIAL = 100

# ===================== 可视化参数 =====================
# GIF帧间隔(ms)及样式
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