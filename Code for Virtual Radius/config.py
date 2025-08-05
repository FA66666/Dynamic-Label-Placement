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
# 方向采样数
NUM_DIRECTIONS_TO_EVALUATE = 72

# ===================== 成本函数权重 (已修改) =====================
# 重叠、引导线、移动距离权重
W_OVERLAP = 10000000
W_LINE = 70
W_MOVEMENT = 10 # <-- 新增：对“移动距离”的惩罚 (取代了旋转和平移)

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