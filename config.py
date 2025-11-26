"""
配置文件：存储路径、列名和全局参数
"""
import os

# 文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "法实证.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# 创建输出目录
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 列名定义（Excel列名已改为英文字母）
COLUMNS = {
    'A1': 'A1',
    'A2': 'A2',
    'A3': 'A3',
    'B1': 'B1',
    'B2': 'B2',
    'B3': 'B3',
    'C1': 'C1',
    'C2': 'C2',
    'Y': 'Y'
}

# 变量分组
A_CLASS = ['A1', 'A2', 'A3']  # 人格从属性要素
B_CLASS = ['B1', 'B2', 'B3']  # 经济从属性要素
C_CLASS = ['C1', 'C2']        # 组织从属性要素
ALL_FEATURES = A_CLASS + B_CLASS + C_CLASS
TARGET = 'Y'

# 变量描述（用于图表标注）- 使用英文字母
VAR_LABELS = {
    'A1': 'A1',
    'A2': 'A2',
    'A3': 'A3',
    'B1': 'B1',
    'B2': 'B2',
    'B3': 'B3',
    'C1': 'C1',
    'C2': 'C2',
    'Y': 'Y'
}

# 变量中文说明（仅供参考）
VAR_LABELS_CN = {
    'A1': '接受指令劳动',
    'A2': '受指令约束(1)',
    'A3': '受指令约束(2)',
    'B1': '生存依赖性',
    'B2': '为资方劳动',
    'B3': '薪资由企业决定',
    'C1': '受指令约束(组织)',
    'C2': '为资方劳动(组织)',
    'Y': '认定劳动关系'
}

# 取值说明
# 1: 明确提出该要素
# 0: 未提及该要素
# -1: 否认要素存在

# 图表样式
FIGURE_SIZE = (10, 6)
FONT_SIZE = 12

# 随机种子（保证结果可复现）
RANDOM_STATE = 42
