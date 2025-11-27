"""
第一步：描述性统计分析
核心功能：各要素取值分布、条件概率分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, VAR_LABELS, FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_value_distribution(df):
    """
    绘制各要素取值分布图
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # 红、橙、绿
    
    for idx, col in enumerate(ALL_FEATURES + [TARGET]):
        ax = axes[idx]
        value_counts = df[col].value_counts().sort_index()
        
        # 补全缺失的类别以便颜色对应
        for val in [-1, 0, 1]:
            if val not in value_counts:
                value_counts[val] = 0
        value_counts = value_counts.sort_index()
        
        bars = ax.bar(value_counts.index, value_counts.values, color=colors)
        
        ax.set_title(VAR_LABELS.get(col, col), fontsize=12, fontweight='bold')
        ax.set_xlabel('取值', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['否认(-1)', '未提及(0)', '肯定(1)'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('各要素取值分布情况', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/1_要素取值分布.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/1_要素取值分布.png")
    plt.close()

def plot_conditional_probability(df):
    """
    绘制条件概率热力图
    """
    # 准备数据：每个要素在不同取值下的认定比例
    prob_matrix = []
    row_labels = []
    
    for col in ALL_FEATURES:
        row = []
        for val in [-1, 0, 1]:
            subset = df[df[col] == val]
            if len(subset) > 0:
                prob = (subset[TARGET] == 1).sum() / len(subset) * 100
            else:
                prob = np.nan
            row.append(prob)
        prob_matrix.append(row)
        row_labels.append(VAR_LABELS.get(col, col))
    
    prob_df = pd.DataFrame(prob_matrix, 
                          index=row_labels,
                          columns=['否认(-1)', '未提及(0)', '肯定(1)'])
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_df, annot=True, fmt='.1f', cmap='RdYlGn', 
               cbar_kws={'label': '认定劳动关系比例(%)'}, 
               vmin=0, vmax=100)
    plt.title('各要素取值下认定劳动关系的比例(%)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('要素取值', fontsize=12)
    plt.ylabel('要素', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/2_条件概率热力图.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/2_条件概率热力图.png")
    plt.close()

def save_descriptive_stats(df):
    """
    保存描述性统计表
    """
    results = []
    for col in ALL_FEATURES + [TARGET]:
        value_counts = df[col].value_counts().sort_index()
        total = len(df[col].dropna())
        
        row = {'变量': VAR_LABELS.get(col, col)}
        for val in [-1, 0, 1]:
            count = value_counts.get(val, 0)
            pct = count / total * 100 if total > 0 else 0
            row[f'{val}(次数)'] = count
            row[f'{val}(%)'] = f'{pct:.1f}%'
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{RESULTS_DIR}/1_描述性统计表.csv", index=False, encoding='utf-8-sig')
    print(f"表格已保存: {RESULTS_DIR}/1_描述性统计表.csv")

def main():
    print("=" * 60)
    print("步骤1: 描述性统计分析")
    print("=" * 60)
    
    df = get_clean_data()
    plot_value_distribution(df)
    plot_conditional_probability(df)
    save_descriptive_stats(df)
    
    print("\n描述性统计分析完成！")

if __name__ == "__main__":
    main()
