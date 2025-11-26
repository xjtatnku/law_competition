"""
模型比较：嵌套模型检验C1/C2的增量贡献
使用似然比检验(Likelihood Ratio Test)比较嵌套模型
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, A_CLASS, B_CLASS, C_CLASS,
                    VAR_LABELS, FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def fit_logit_model(df, features):
    """
    拟合逻辑回归模型并返回结果
    """
    data = df[features + [TARGET]].dropna()
    X = data[features]
    y = data[TARGET]
    
    X_sm = sm.add_constant(X)
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=False)
    
    return result, len(data)

def likelihood_ratio_test(model_restricted, model_full, df_diff):
    """
    似然比检验
    
    Args:
        model_restricted: 受限模型（较少参数）
        model_full: 完整模型（较多参数）
        df_diff: 自由度差异（参数个数差异）
    
    Returns:
        dict: 检验结果
    """
    # 计算似然比统计量
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    
    # 计算p值
    p_value = stats.chi2.sf(lr_stat, df_diff)
    
    # AIC和BIC比较
    aic_diff = model_restricted.aic - model_full.aic
    bic_diff = model_restricted.bic - model_full.bic
    
    return {
        'LR统计量': lr_stat,
        'p值': p_value,
        '自由度差': df_diff,
        'AIC差异': aic_diff,
        'BIC差异': bic_diff,
        '显著性': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    }

def test_c1_incremental_contribution(df):
    """
    检验C1的增量贡献
    """
    print("\n" + "="*60)
    print("检验C1的增量贡献")
    print("="*60)
    
    # 基准模型：A类 + B类
    print("\n拟合基准模型 (A类 + B类)...")
    model_base, n_base = fit_logit_model(df, A_CLASS + B_CLASS)
    
    # 完整模型：A类 + B类 + C1
    print("拟合完整模型 (A类 + B类 + C1)...")
    model_full, n_full = fit_logit_model(df, A_CLASS + B_CLASS + ['C1'])
    
    # 确保样本数相同
    if n_base != n_full:
        print(f"警告: 样本数不一致 (基准={n_base}, 完整={n_full})")
    
    # 似然比检验
    lr_result = likelihood_ratio_test(model_base, model_full, df_diff=1)
    
    print(f"\n似然比检验结果:")
    print(f"  LR统计量 = {lr_result['LR统计量']:.4f}")
    print(f"  p值 = {lr_result['p值']:.4f} {lr_result['显著性']}")
    print(f"  自由度差 = {lr_result['自由度差']}")
    
    print(f"\n信息准则比较:")
    print(f"  基准模型 AIC = {model_base.aic:.4f}")
    print(f"  完整模型 AIC = {model_full.aic:.4f}")
    print(f"  AIC差异 = {lr_result['AIC差异']:.4f} {'(完整模型更优)' if lr_result['AIC差异'] > 0 else '(基准模型更优)'}")
    print(f"  基准模型 BIC = {model_base.bic:.4f}")
    print(f"  完整模型 BIC = {model_full.bic:.4f}")
    print(f"  BIC差异 = {lr_result['BIC差异']:.4f} {'(完整模型更优)' if lr_result['BIC差异'] > 0 else '(基准模型更优)'}")
    
    print(f"\n伪R²比较:")
    print(f"  基准模型 = {model_base.prsquared:.4f}")
    print(f"  完整模型 = {model_full.prsquared:.4f}")
    print(f"  差异 = {model_full.prsquared - model_base.prsquared:.4f}")
    
    # C1的系数
    c1_coef = model_full.params['C1']
    c1_pvalue = model_full.pvalues['C1']
    c1_or = np.exp(c1_coef)
    
    print(f"\nC1的回归系数:")
    print(f"  系数 = {c1_coef:.4f}")
    print(f"  p值 = {c1_pvalue:.4f} {'***' if c1_pvalue < 0.001 else '**' if c1_pvalue < 0.01 else '*' if c1_pvalue < 0.05 else 'ns'}")
    print(f"  Odds Ratio = {c1_or:.4f}")
    
    return {
        'test_name': 'C1增量贡献检验',
        'base_model': 'A类+B类',
        'full_model': 'A类+B类+C1',
        **lr_result,
        'base_aic': model_base.aic,
        'full_aic': model_full.aic,
        'base_bic': model_base.bic,
        'full_bic': model_full.bic,
        'base_pseudo_r2': model_base.prsquared,
        'full_pseudo_r2': model_full.prsquared,
        'c_coef': c1_coef,
        'c_pvalue': c1_pvalue,
        'c_or': c1_or
    }

def test_c2_incremental_contribution(df):
    """
    检验C2的增量贡献
    """
    print("\n" + "="*60)
    print("检验C2的增量贡献")
    print("="*60)
    
    # 基准模型：A类 + B类
    print("\n拟合基准模型 (A类 + B类)...")
    model_base, n_base = fit_logit_model(df, A_CLASS + B_CLASS)
    
    # 完整模型：A类 + B类 + C2
    print("拟合完整模型 (A类 + B类 + C2)...")
    model_full, n_full = fit_logit_model(df, A_CLASS + B_CLASS + ['C2'])
    
    if n_base != n_full:
        print(f"警告: 样本数不一致 (基准={n_base}, 完整={n_full})")
    
    # 似然比检验
    lr_result = likelihood_ratio_test(model_base, model_full, df_diff=1)
    
    print(f"\n似然比检验结果:")
    print(f"  LR统计量 = {lr_result['LR统计量']:.4f}")
    print(f"  p值 = {lr_result['p值']:.4f} {lr_result['显著性']}")
    print(f"  自由度差 = {lr_result['自由度差']}")
    
    print(f"\n信息准则比较:")
    print(f"  基准模型 AIC = {model_base.aic:.4f}")
    print(f"  完整模型 AIC = {model_full.aic:.4f}")
    print(f"  AIC差异 = {lr_result['AIC差异']:.4f} {'(完整模型更优)' if lr_result['AIC差异'] > 0 else '(基准模型更优)'}")
    print(f"  基准模型 BIC = {model_base.bic:.4f}")
    print(f"  完整模型 BIC = {model_full.bic:.4f}")
    print(f"  BIC差异 = {lr_result['BIC差异']:.4f} {'(完整模型更优)' if lr_result['BIC差异'] > 0 else '(基准模型更优)'}")
    
    print(f"\n伪R²比较:")
    print(f"  基准模型 = {model_base.prsquared:.4f}")
    print(f"  完整模型 = {model_full.prsquared:.4f}")
    print(f"  差异 = {model_full.prsquared - model_base.prsquared:.4f}")
    
    # C2的系数
    c2_coef = model_full.params['C2']
    c2_pvalue = model_full.pvalues['C2']
    c2_or = np.exp(c2_coef)
    
    print(f"\nC2的回归系数:")
    print(f"  系数 = {c2_coef:.4f}")
    print(f"  p值 = {c2_pvalue:.4f} {'***' if c2_pvalue < 0.001 else '**' if c2_pvalue < 0.01 else '*' if c2_pvalue < 0.05 else 'ns'}")
    print(f"  Odds Ratio = {c2_or:.4f}")
    
    return {
        'test_name': 'C2增量贡献检验',
        'base_model': 'A类+B类',
        'full_model': 'A类+B类+C2',
        **lr_result,
        'base_aic': model_base.aic,
        'full_aic': model_full.aic,
        'base_bic': model_base.bic,
        'full_bic': model_full.bic,
        'base_pseudo_r2': model_base.prsquared,
        'full_pseudo_r2': model_full.prsquared,
        'c_coef': c2_coef,
        'c_pvalue': c2_pvalue,
        'c_or': c2_or
    }

def test_c1_c2_joint_contribution(df):
    """
    检验C1和C2的联合增量贡献
    """
    print("\n" + "="*60)
    print("检验C1和C2的联合增量贡献")
    print("="*60)
    
    # 基准模型：A类 + B类
    print("\n拟合基准模型 (A类 + B类)...")
    model_base, n_base = fit_logit_model(df, A_CLASS + B_CLASS)
    
    # 完整模型：A类 + B类 + C1 + C2
    print("拟合完整模型 (A类 + B类 + C1 + C2)...")
    model_full, n_full = fit_logit_model(df, ALL_FEATURES)
    
    if n_base != n_full:
        print(f"警告: 样本数不一致 (基准={n_base}, 完整={n_full})")
    
    # 似然比检验
    lr_result = likelihood_ratio_test(model_base, model_full, df_diff=2)
    
    print(f"\n似然比检验结果:")
    print(f"  LR统计量 = {lr_result['LR统计量']:.4f}")
    print(f"  p值 = {lr_result['p值']:.4f} {lr_result['显著性']}")
    print(f"  自由度差 = {lr_result['自由度差']}")
    
    print(f"\n信息准则比较:")
    print(f"  AIC差异 = {lr_result['AIC差异']:.4f} {'(完整模型更优)' if lr_result['AIC差异'] > 0 else '(基准模型更优)'}")
    print(f"  BIC差异 = {lr_result['BIC差异']:.4f} {'(完整模型更优)' if lr_result['BIC差异'] > 0 else '(基准模型更优)'}")
    
    print(f"\n伪R²比较:")
    print(f"  基准模型 = {model_base.prsquared:.4f}")
    print(f"  完整模型 = {model_full.prsquared:.4f}")
    print(f"  差异 = {model_full.prsquared - model_base.prsquared:.4f}")
    
    return {
        'test_name': 'C1+C2联合增量贡献检验',
        'base_model': 'A类+B类',
        'full_model': 'A类+B类+C1+C2',
        **lr_result,
        'base_aic': model_base.aic,
        'full_aic': model_full.aic,
        'base_bic': model_base.bic,
        'full_bic': model_full.bic,
        'base_pseudo_r2': model_base.prsquared,
        'full_pseudo_r2': model_full.prsquared
    }

def plot_model_comparison(results_list):
    """
    绘制模型比较可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取数据
    test_names = [r['test_name'] for r in results_list]
    lr_stats = [r['LR统计量'] for r in results_list]
    p_values = [r['p值'] for r in results_list]
    aic_diffs = [r['AIC差异'] for r in results_list]
    pseudo_r2_diffs = [r['full_pseudo_r2'] - r['base_pseudo_r2'] for r in results_list]
    
    # 1. LR统计量
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(test_names)), lr_stats, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels([f"C1\n增量", f"C2\n增量", f"C1+C2\n联合"], fontsize=10)
    ax1.set_ylabel('LR统计量', fontsize=12)
    ax1.set_title('似然比统计量对比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars1, lr_stats)):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. p值（负对数）
    ax2 = axes[0, 1]
    neg_log_p = [-np.log10(p) for p in p_values]
    bars2 = ax2.bar(range(len(test_names)), neg_log_p, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels([f"C1\n增量", f"C2\n增量", f"C1+C2\n联合"], fontsize=10)
    ax2.set_ylabel('-log10(p值)', fontsize=12)
    ax2.set_title('显著性水平对比', fontsize=12, fontweight='bold')
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', 
               linewidth=1, label='p=0.05')
    ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', 
               linewidth=1, label='p=0.01')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, val, p) in enumerate(zip(bars2, neg_log_p, p_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'p={p:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. AIC差异
    ax3 = axes[1, 0]
    colors3 = ['green' if x > 0 else 'red' for x in aic_diffs]
    bars3 = ax3.bar(range(len(test_names)), aic_diffs, color=colors3, alpha=0.7)
    ax3.set_xticks(range(len(test_names)))
    ax3.set_xticklabels([f"C1\n增量", f"C2\n增量", f"C1+C2\n联合"], fontsize=10)
    ax3.set_ylabel('AIC差异 (基准-完整)', fontsize=12)
    ax3.set_title('AIC改进 (>0为完整模型更优)', fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars3, aic_diffs)):
        ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
    # 4. 伪R²提升
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(test_names)), pseudo_r2_diffs, color='purple', alpha=0.7)
    ax4.set_xticks(range(len(test_names)))
    ax4.set_xticklabels([f"C1\n增量", f"C2\n增量", f"C1+C2\n联合"], fontsize=10)
    ax4.set_ylabel('伪R2提升', fontsize=12)
    ax4.set_title('伪R2改进', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars4, pseudo_r2_diffs)):
        ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/16_模型比较综合分析.png", dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {FIGURES_DIR}/16_模型比较综合分析.png")
    plt.close()

def main():
    """
    主函数
    """
    print("=" * 60)
    print("模型比较与嵌套检验")
    print("=" * 60)
    
    df = get_clean_data()
    
    # 执行三个检验
    result_c1 = test_c1_incremental_contribution(df)
    result_c2 = test_c2_incremental_contribution(df)
    result_joint = test_c1_c2_joint_contribution(df)
    
    results_list = [result_c1, result_c2, result_joint]
    
    # 汇总表格
    summary = pd.DataFrame([{
        '检验': r['test_name'],
        '基准模型': r['base_model'],
        '完整模型': r['full_model'],
        'LR统计量': f"{r['LR统计量']:.4f}",
        'p值': f"{r['p值']:.4f}",
        '显著性': r['显著性'],
        'AIC差异': f"{r['AIC差异']:.4f}",
        'BIC差异': f"{r['BIC差异']:.4f}",
        '伪R²提升': f"{r['full_pseudo_r2'] - r['base_pseudo_r2']:.4f}"
    } for r in results_list])
    
    print("\n" + "="*60)
    print("模型比较汇总:")
    print("="*60)
    print(summary.to_string(index=False))
    
    summary.to_csv(f"{RESULTS_DIR}/模型比较汇总.csv", index=False, encoding='utf-8-sig')
    
    # 绘图
    plot_model_comparison(results_list)
    
    print("\n" + "=" * 60)
    print("模型比较分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("\n解读提示:")
    print("- LR检验p值<0.05: 加入C1/C2后模型显著改进")
    print("- AIC/BIC差异>0: 完整模型更优")
    print("- 伪R²提升: 解释力的增加")
    print("=" * 60)

if __name__ == "__main__":
    main()
