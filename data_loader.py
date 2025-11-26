"""
数据加载与预处理模块
"""
import pandas as pd
import numpy as np
from config import DATA_FILE, ALL_FEATURES, TARGET, COLUMNS

def load_data():
    """
    加载Excel数据，并将列名重命名为简短的内部名称
    
    Returns:
        pd.DataFrame: 加载的数据
    """
    print(f"正在加载数据: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"原始列名: {list(df.columns)}")
    
    # 将Excel列名重命名为简短的内部列名
    # COLUMNS 格式: {'A1': 'A1：接受指令劳动', ...}
    # 需要反向映射: {'A1：接受指令劳动': 'A1', ...}
    rename_map = {v: k for k, v in COLUMNS.items()}
    df = df.rename(columns=rename_map)
    print(f"重命名后列名: {list(df.columns)}")
    
    return df

def check_data_quality(df):
    """
    检查数据质量
    
    Args:
        df: 数据框
    """
    print("\n=== 数据质量检查 ===")
    print(f"总样本数: {len(df)}")
    print(f"\n各列缺失值统计:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("无缺失值")
    
    print(f"\n各列数据类型:")
    print(df.dtypes)
    
    # 检查取值范围（应该是-1, 0, 1）
    print(f"\n各列取值范围检查（应为 -1, 0, 1）:")
    for col in df.columns:
        unique_vals = sorted(df[col].dropna().unique())
        print(f"{col}: {unique_vals}")

def preprocess_data(df):
    """
    数据预处理
    
    Args:
        df: 原始数据框
        
    Returns:
        pd.DataFrame: 预处理后的数据
    """
    df_clean = df.copy()
    
    # 确保所有特征列的值在 {-1, 0, 1} 范围内
    all_vars = ALL_FEATURES + [TARGET]
    for col in all_vars:
        if col in df_clean.columns:
            # 将非法值设为 NaN
            df_clean.loc[~df_clean[col].isin([-1, 0, 1]), col] = np.nan
    
    # 删除目标变量为空的行
    if TARGET in df_clean.columns:
        df_clean = df_clean.dropna(subset=[TARGET])
    
    print(f"\n预处理后样本数: {len(df_clean)}")
    
    return df_clean

def get_clean_data():
    """
    获取清洗后的数据（主函数）
    
    Returns:
        pd.DataFrame: 清洗后的数据
    """
    df = load_data()
    check_data_quality(df)
    df_clean = preprocess_data(df)
    return df_clean

if __name__ == "__main__":
    # 测试数据加载
    df = get_clean_data()
    print("\n=== 数据预览 ===")
    print(df.head())
    print("\n=== 基本统计 ===")
    print(df.describe())
