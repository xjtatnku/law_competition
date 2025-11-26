"""
相关性分析：检验C1与A类、C2与B类的相关性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency
from data_loader import get_clean_data
from config import (A_CLASS, B_CLASS, C_CLASS, VAR_LABELS, 
                    FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_correlation_matrix(df, vars_list):
    """
    计算相关系数矩阵（Spearman）
    """
    corr_matrix = df[vars_list].corr(method='spearman')
    return corr_matrix

def plot_correlation_heatmap(corr_matrix, title, filename):
    """
    绘制相关系数热力图
    """
    # 重命名索引和列
    corr_matrix_labeled = corr_matrix.copy()
    corr_matrix_labeled.index = [VAR_LABELS.get(c, c) for c in corr_matrix.index]
    corr_matrix_labeled.columns = [VAR_LABELS.get(c, c) for c in corr_matrix.columns]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_labeled, annot=True, fmt='.3f', 
               cmap='coolwarm', center=0, vmin=-1, vmax=1,
               square=True, linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/{filename}")
    plt.close()

def c1_vs_a_analysis(df):
    """
    C1 vs A类要素相关性分析
    """
    print("\n=== C1 与 A类要素相关性分析 ===")
    
    vars_to_analyze = A_CLASS + ['C1']
    corr_matrix = calculate_correlation_matrix(df, vars_to_analyze)
    
    print("\nSpearman 相关系数矩阵:")
    print(corr_matrix)
    
    # 保存相关系数矩阵
    corr_matrix_labeled = corr_matrix.copy()
    corr_matrix_labeled.index = [VAR_LABELS.get(c, c) for c in corr_matrix.index]
    corr_matrix_labeled.columns = [VAR_LABELS.get(c, c) for c in corr_matrix.columns]
    corr_matrix_labeled.to_csv(f"{RESULTS_DIR}/C1与A类相关系数.csv", encoding='utf-8-sig')
    
    # 绘制热力图
    plot_correlation_heatmap(corr_matrix, 
                            'C1与A类要素(人格从属性)的相关性',
                            '3_C1与A类相关性热力图.png')
    
    # 提取C1与A类的相关系数
    c1_corr = corr_matrix.loc['C1', A_CLASS]
    print(f"\nC1 与各 A类要素的相关系数:")
    for var in A_CLASS:
        print(f"  C1 vs {VAR_LABELS[var]}: {c1_corr[var]:.3f}")
    
    # 卡方检验（列联表）
    print("\n卡方检验结果:")
    chi_results = []
    for var in A_CLASS:
        # 构建列联表
        contingency = pd.crosstab(df['C1'], df[var])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        chi_results.append({
            '配对': f'C1 vs {VAR_LABELS[var]}',
            'Chi2统计量': f'{chi2:.3f}',
            'p值': f'{p_value:.4f}',
            '显著性': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
        print(f"  C1 vs {VAR_LABELS[var]}: Chi2={chi2:.3f}, p={p_value:.4f}")
    
    chi_df = pd.DataFrame(chi_results)
    chi_df.to_csv(f"{RESULTS_DIR}/C1与A类卡方检验.csv", index=False, encoding='utf-8-sig')
    
    return corr_matrix

def c2_vs_b_analysis(df):
    """
    C2 vs B类要素相关性分析
    """
    print("\n=== C2 与 B类要素相关性分析 ===")
    
    vars_to_analyze = B_CLASS + ['C2']
    corr_matrix = calculate_correlation_matrix(df, vars_to_analyze)
    
    print("\nSpearman 相关系数矩阵:")
    print(corr_matrix)
    
    # 保存相关系数矩阵
    corr_matrix_labeled = corr_matrix.copy()
    corr_matrix_labeled.index = [VAR_LABELS.get(c, c) for c in corr_matrix.index]
    corr_matrix_labeled.columns = [VAR_LABELS.get(c, c) for c in corr_matrix.columns]
    corr_matrix_labeled.to_csv(f"{RESULTS_DIR}/C2与B类相关系数.csv", encoding='utf-8-sig')
    
    # 绘制热力图
    plot_correlation_heatmap(corr_matrix, 
                            'C2与B类要素(经济从属性)的相关性',
                            '4_C2与B类相关性热力图.png')
    
    # 提取C2与B类的相关系数
    c2_corr = corr_matrix.loc['C2', B_CLASS]
    print(f"\nC2 与各 B类要素的相关系数:")
    for var in B_CLASS:
        print(f"  C2 vs {VAR_LABELS[var]}: {c2_corr[var]:.3f}")
    
    # 卡方检验
    print("\n卡方检验结果:")
    chi_results = []
    for var in B_CLASS:
        contingency = pd.crosstab(df['C2'], df[var])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        chi_results.append({
            '配对': f'C2 vs {VAR_LABELS[var]}',
            'Chi2统计量': f'{chi2:.3f}',
            'p值': f'{p_value:.4f}',
            '显著性': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
        print(f"  C2 vs {VAR_LABELS[var]}: Chi2={chi2:.3f}, p={p_value:.4f}")
    
    chi_df = pd.DataFrame(chi_results)
    chi_df.to_csv(f"{RESULTS_DIR}/C2与B类卡方检验.csv", index=False, encoding='utf-8-sig')
    
    return corr_matrix

def plot_correlation_comparison(df):
    """
    绘制C1/C2与其他要素的相关系数对比图
    """
    # 计算C1与A类、C2与B类的相关系数
    c1_corr = [df[['C1', var]].corr(method='spearman').iloc[0, 1] for var in A_CLASS]
    c2_corr = [df[['C2', var]].corr(method='spearman').iloc[0, 1] for var in B_CLASS]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # C1与A类
    ax1.barh([VAR_LABELS[v] for v in A_CLASS], c1_corr, color='steelblue')
    ax1.set_xlabel('Spearman相关系数', fontsize=12)
    ax1.set_title('C1(组织从属性-受指令约束)\n与A类要素的相关性', 
                 fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_xlim(-0.1, 1)
    for i, v in enumerate(c1_corr):
        ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
    
    # C2与B类
    ax2.barh([VAR_LABELS[v] for v in B_CLASS], c2_corr, color='coral')
    ax2.set_xlabel('Spearman相关系数', fontsize=12)
    ax2.set_title('C2(组织从属性-为资方劳动)\n与B类要素的相关性', 
                 fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlim(-0.1, 1)
    for i, v in enumerate(c2_corr):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/5_C1C2相关性对比图.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/5_C1C2相关性对比图.png")
    plt.close()

def main():
    """
    主函数
    """
    print("=" * 60)
    print("相关性分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    c1_vs_a_analysis(df)
    c2_vs_b_analysis(df)
    plot_correlation_comparison(df)
    
    print("\n" + "=" * 60)
    print("相关性分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
