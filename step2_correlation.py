"""
第二步：相关性分析
核心功能：四种独立性系数检验 (Pearson, Spearman, Kendall, Cramér's V)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from data_loader import get_clean_data
from config import (A_CLASS, B_CLASS, C_CLASS, VAR_LABELS, FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_cramers_v(x, y):
    """计算Cramér's V系数"""
    contingency = pd.crosstab(x, y)
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    return cramers_v, p_value

def calculate_all_coefficients(df, var1, var2):
    """计算四种独立性系数"""
    data = df[[var1, var2]].dropna()
    x = data[var1]
    y = data[var2]
    
    return {
        'pearson': pearsonr(x, y),
        'spearman': spearmanr(x, y),
        'kendall': kendalltau(x, y),
        'cramers_v': calculate_cramers_v(x, y)
    }

def plot_coefficient_comparison(results_df, title_prefix, filename_prefix):
    """绘制系数对比图"""
    coeffs_data = {
        'Pearson': [float(x) for x in results_df['Pearson']],
        'Spearman': [float(x) for x in results_df['Spearman']],
        'Kendall Tau': [float(x) for x in results_df['Kendall_Tau']],
        "Cramér's V": [float(x) for x in results_df['Cramers_V']]
    }
    
    vars_labels = [label.split(' vs ')[1] for label in results_df['配对']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, (coef_name, coef_values) in enumerate(coeffs_data.items()):
        ax = axes[idx]
        bars = ax.barh(vars_labels, coef_values, color=colors[idx], alpha=0.7)
        ax.set_xlabel('系数值', fontsize=11)
        ax.set_title(f'{coef_name}', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.1, 1.0)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, coef_values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10)
    
    plt.suptitle(f'{title_prefix} 四种独立性系数对比', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename_prefix}_系数对比.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/{filename_prefix}_系数对比.png")
    plt.close()

def analyze_coefficients(df, target_var, compare_vars, prefix):
    """分析指定变量组的相关性"""
    results = []
    for var in compare_vars:
        coeffs = calculate_all_coefficients(df, target_var, var)
        results.append({
            '配对': f'{target_var} vs {VAR_LABELS[var]}',
            'Pearson': f"{coeffs['pearson'][0]:.4f}",
            'Spearman': f"{coeffs['spearman'][0]:.4f}",
            'Kendall_Tau': f"{coeffs['kendall'][0]:.4f}",
            'Cramers_V': f"{coeffs['cramers_v'][0]:.4f}"
        })
    return pd.DataFrame(results)

def plot_heatmap_all(df):
    """绘制所有变量的热力图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    coef_types = ['pearson', 'spearman', 'kendall', 'cramers_v']
    coef_names = ['Pearson', 'Spearman', 'Kendall Tau', "Cramér's V"]
    
    all_vars = A_CLASS + B_CLASS + C_CLASS
    labels = [VAR_LABELS[v] for v in all_vars]
    n = len(all_vars)
    
    for idx, (coef_type, coef_name) in enumerate(zip(coef_types, coef_names)):
        ax = axes[idx // 2, idx % 2]
        matrix = np.zeros((n, n))
        
        for i, var1 in enumerate(all_vars):
            for j, var2 in enumerate(all_vars):
                if i == j: matrix[i, j] = 1.0
                elif i < j:
                    coeffs = calculate_all_coefficients(df, var1, var2)
                    val = coeffs[coef_type][0]
                    matrix[i, j] = val
                    matrix[j, i] = val
        
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=10)
        
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha="center", va="center", fontsize=8)
        
        ax.set_title(f'{coef_name} 相关系数矩阵', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/5_全部变量四种系数热力图.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/5_全部变量四种系数热力图.png")
    plt.close()

def main():
    print("=" * 60)
    print("步骤2: 相关性分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    # C1 vs A类
    res_c1 = analyze_coefficients(df, 'C1', A_CLASS, 'C1与A类')
    res_c1.to_csv(f"{RESULTS_DIR}/2_C1与A类相关性.csv", index=False, encoding='utf-8-sig')
    plot_coefficient_comparison(res_c1, 'C1(组织从属性-受指令约束) 与 A类', '3_C1与A类')
    
    # C2 vs B类
    res_c2 = analyze_coefficients(df, 'C2', B_CLASS, 'C2与B类')
    res_c2.to_csv(f"{RESULTS_DIR}/2_C2与B类相关性.csv", index=False, encoding='utf-8-sig')
    plot_coefficient_comparison(res_c2, 'C2(组织从属性-为资方劳动) 与 B类', '4_C2与B类')
    
    # 全变量热力图
    plot_heatmap_all(df)
    
    print("\n相关性分析完成！")

if __name__ == "__main__":
    main()
