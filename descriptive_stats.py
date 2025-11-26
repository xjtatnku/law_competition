"""
描述性统计与可视化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, A_CLASS, B_CLASS, C_CLASS, 
                    VAR_LABELS, FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def value_distribution(df):
    """
    各要素取值分布统计
    """
    print("\n=== 各要素取值分布 ===")
    
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
    print(result_df.to_string(index=False))
    
    # 保存结果
    result_df.to_csv(f"{RESULTS_DIR}/取值分布统计.csv", index=False, encoding='utf-8-sig')
    
    return result_df

def plot_value_distribution(df):
    """
    绘制各要素取值分布图
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(ALL_FEATURES + [TARGET]):
        ax = axes[idx]
        value_counts = df[col].value_counts().sort_index()
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c']  # 红、橙、绿
        bars = ax.bar(value_counts.index, value_counts.values, color=colors)
        
        ax.set_title(VAR_LABELS.get(col, col), fontsize=12, fontweight='bold')
        ax.set_xlabel('取值', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['否认(-1)', '未提及(0)', '肯定(1)'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/1_要素取值分布.png", dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {FIGURES_DIR}/1_要素取值分布.png")
    plt.close()

def conditional_probability(df):
    """
    各要素取值下认定劳动关系的比例（条件概率）
    """
    print("\n=== 各要素对劳动关系认定的影响 ===")
    
    results = []
    for col in ALL_FEATURES:
        for val in [-1, 0, 1]:
            subset = df[df[col] == val]
            if len(subset) > 0:
                认定比例 = (subset[TARGET] == 1).sum() / len(subset) * 100
                results.append({
                    '要素': VAR_LABELS.get(col, col),
                    '取值': val,
                    '取值说明': {-1: '否认', 0: '未提及', 1: '肯定'}[val],
                    '样本数': len(subset),
                    '认定劳动关系比例(%)': f'{认定比例:.1f}%'
                })
    
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))
    
    # 保存结果
    result_df.to_csv(f"{RESULTS_DIR}/条件概率统计.csv", index=False, encoding='utf-8-sig')
    
    return result_df

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

def overall_statistics(df):
    """
    总体统计
    """
    print("\n=== 总体统计 ===")
    print(f"总样本数: {len(df)}")
    print(f"认定劳动关系: {(df[TARGET] == 1).sum()} ({(df[TARGET] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"未认定劳动关系: {(df[TARGET] == 0).sum()} ({(df[TARGET] == 0).sum() / len(df) * 100:.1f}%)")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("描述性统计分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    overall_statistics(df)
    value_distribution(df)
    plot_value_distribution(df)
    conditional_probability(df)
    plot_conditional_probability(df)
    
    print("\n" + "=" * 60)
    print("描述性统计分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
