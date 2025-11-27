"""
增强版相关性分析：使用四种独立性检验系数
包括：Pearson相关系数、Cramér's V、Kendall's Tau、Spearman相关系数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from scipy.stats.contingency import association
from data_loader import get_clean_data
from config import (A_CLASS, B_CLASS, C_CLASS, VAR_LABELS, 
                    FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_cramers_v(x, y):
    """
    计算Cramér's V系数
    """
    contingency = pd.crosstab(x, y)
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return cramers_v, p_value

def calculate_all_coefficients(df, var1, var2):
    """
    计算四种独立性系数
    """
    # 去除缺失值
    data = df[[var1, var2]].dropna()
    x = data[var1]
    y = data[var2]
    
    # 1. Pearson相关系数
    pearson_coef, pearson_p = pearsonr(x, y)
    
    # 2. Spearman相关系数
    spearman_coef, spearman_p = spearmanr(x, y)
    
    # 3. Kendall's Tau
    kendall_coef, kendall_p = kendalltau(x, y)
    
    # 4. Cramér's V
    cramers_v, cramers_p = calculate_cramers_v(x, y)
    
    return {
        'pearson': (pearson_coef, pearson_p),
        'spearman': (spearman_coef, spearman_p),
        'kendall': (kendall_coef, kendall_p),
        'cramers_v': (cramers_v, cramers_p)
    }

def analyze_c1_vs_a_comprehensive(df):
    """
    C1 vs A类要素的综合相关性分析
    """
    print("\n" + "="*60)
    print("C1 与 A类要素的四种独立性系数分析")
    print("="*60)
    
    results = []
    
    for a_var in A_CLASS:
        coeffs = calculate_all_coefficients(df, 'C1', a_var)
        
        result = {
            '配对': f'C1 vs {VAR_LABELS[a_var]}',
            'Pearson': f"{coeffs['pearson'][0]:.4f}",
            'Pearson_p': f"{coeffs['pearson'][1]:.4f}",
            'Spearman': f"{coeffs['spearman'][0]:.4f}",
            'Spearman_p': f"{coeffs['spearman'][1]:.4f}",
            'Kendall_Tau': f"{coeffs['kendall'][0]:.4f}",
            'Kendall_p': f"{coeffs['kendall'][1]:.4f}",
            'Cramers_V': f"{coeffs['cramers_v'][0]:.4f}",
            'Cramers_p': f"{coeffs['cramers_v'][1]:.4f}"
        }
        results.append(result)
        
        print(f"\n{VAR_LABELS[a_var]}:")
        print(f"  Pearson:      {coeffs['pearson'][0]:.4f} (p={coeffs['pearson'][1]:.4f})")
        print(f"  Spearman:     {coeffs['spearman'][0]:.4f} (p={coeffs['spearman'][1]:.4f})")
        print(f"  Kendall Tau:  {coeffs['kendall'][0]:.4f} (p={coeffs['kendall'][1]:.4f})")
        print(f"  Cramér's V:   {coeffs['cramers_v'][0]:.4f} (p={coeffs['cramers_v'][1]:.4f})")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/C1与A类_四种独立性系数.csv", 
                      index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {RESULTS_DIR}/C1与A类_四种独立性系数.csv")
    
    return results_df

def analyze_c2_vs_b_comprehensive(df):
    """
    C2 vs B类要素的综合相关性分析
    """
    print("\n" + "="*60)
    print("C2 与 B类要素的四种独立性系数分析")
    print("="*60)
    
    results = []
    
    for b_var in B_CLASS:
        coeffs = calculate_all_coefficients(df, 'C2', b_var)
        
        result = {
            '配对': f'C2 vs {VAR_LABELS[b_var]}',
            'Pearson': f"{coeffs['pearson'][0]:.4f}",
            'Pearson_p': f"{coeffs['pearson'][1]:.4f}",
            'Spearman': f"{coeffs['spearman'][0]:.4f}",
            'Spearman_p': f"{coeffs['spearman'][1]:.4f}",
            'Kendall_Tau': f"{coeffs['kendall'][0]:.4f}",
            'Kendall_p': f"{coeffs['kendall'][1]:.4f}",
            'Cramers_V': f"{coeffs['cramers_v'][0]:.4f}",
            'Cramers_p': f"{coeffs['cramers_v'][1]:.4f}"
        }
        results.append(result)
        
        print(f"\n{VAR_LABELS[b_var]}:")
        print(f"  Pearson:      {coeffs['pearson'][0]:.4f} (p={coeffs['pearson'][1]:.4f})")
        print(f"  Spearman:     {coeffs['spearman'][0]:.4f} (p={coeffs['spearman'][1]:.4f})")
        print(f"  Kendall Tau:  {coeffs['kendall'][0]:.4f} (p={coeffs['kendall'][1]:.4f})")
        print(f"  Cramér's V:   {coeffs['cramers_v'][0]:.4f} (p={coeffs['cramers_v'][1]:.4f})")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/C2与B类_四种独立性系数.csv", 
                      index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {RESULTS_DIR}/C2与B类_四种独立性系数.csv")
    
    return results_df

def plot_coefficient_comparison_c1_a(results_df):
    """
    绘制C1与A类的四种系数对比图
    """
    # 提取数值
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
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, coef_values)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10)
    
    plt.suptitle('C1(组织从属性-受指令约束) 与 A类要素的四种独立性系数对比', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_C1与A类_四种系数对比.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_C1与A类_四种系数对比.png")
    plt.close()

def plot_coefficient_comparison_c2_b(results_df):
    """
    绘制C2与B类的四种系数对比图
    """
    # 提取数值
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
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, coef_values)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10)
    
    plt.suptitle('C2(组织从属性-为资方劳动) 与 B类要素的四种独立性系数对比', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_C2与B类_四种系数对比.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_C2与B类_四种系数对比.png")
    plt.close()

def plot_heatmap_all_coefficients(df):
    """
    绘制所有系数的热力图矩阵
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    coef_types = ['pearson', 'spearman', 'kendall', 'cramers_v']
    coef_names = ['Pearson', 'Spearman', 'Kendall Tau', "Cramér's V"]
    
    for idx, (coef_type, coef_name) in enumerate(zip(coef_types, coef_names)):
        ax = axes[idx // 2, idx % 2]
        
        # 构建系数矩阵
        all_vars = A_CLASS + B_CLASS + C_CLASS
        n = len(all_vars)
        matrix = np.zeros((n, n))
        
        for i, var1 in enumerate(all_vars):
            for j, var2 in enumerate(all_vars):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    coeffs = calculate_all_coefficients(df, var1, var2)
                    matrix[i, j] = coeffs[coef_type][0]
                    matrix[j, i] = coeffs[coef_type][0]
        
        # 绘制热力图
        labels = [VAR_LABELS[v] for v in all_vars]
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'{coef_name} 相关系数矩阵', fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('系数值', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_全部变量四种系数热力图.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_全部变量四种系数热力图.png")
    plt.close()

def main():
    """
    主函数
    """
    print("=" * 60)
    print("增强版相关性分析 - 四种独立性系数检验")
    print("=" * 60)
    
    df = get_clean_data()
    
    # C1 vs A类分析
    c1_a_results = analyze_c1_vs_a_comprehensive(df)
    plot_coefficient_comparison_c1_a(c1_a_results)
    
    # C2 vs B类分析
    c2_b_results = analyze_c2_vs_b_comprehensive(df)
    plot_coefficient_comparison_c2_b(c2_b_results)
    
    # 全部变量的系数热力图
    plot_heatmap_all_coefficients(df)
    
    print("\n" + "=" * 60)
    print("增强版相关性分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
