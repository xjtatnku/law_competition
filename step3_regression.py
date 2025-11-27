"""
第三步：回归分析 (线性回归 + 逻辑回归)
核心功能：线性回归(OLS)系数、逻辑回归系数、Odds Ratios、ROC曲线
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, VAR_LABELS, FIGURES_DIR, RESULTS_DIR)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def fit_ols_model(df, features, model_name):
    """拟合OLS线性回归模型 (线性概率模型)"""
    data = df[features + [TARGET]].dropna()
    X = sm.add_constant(data[features])
    y = data[TARGET]
    
    model = sm.OLS(y, X).fit()
    
    results = []
    for var in features:
        results.append({
            '变量': VAR_LABELS.get(var, var),
            '系数': model.params[var],
            'p值': model.pvalues[var],
            '显著性': '***' if model.pvalues[var] < 0.001 else '**' if model.pvalues[var] < 0.01 else '*' if model.pvalues[var] < 0.05 else 'ns'
        })
    return pd.DataFrame(results), model

def fit_logit_model(df, features, model_name):
    """拟合逻辑回归模型"""
    data = df[features + [TARGET]].dropna()
    X = sm.add_constant(data[features])
    y = data[TARGET]
    
    model = sm.Logit(y, X).fit(disp=False)
    
    results = []
    conf = model.conf_int()
    for var in features:
        results.append({
            '变量': VAR_LABELS.get(var, var),
            '系数': model.params[var],
            'p值': model.pvalues[var],
            'Odds Ratio': np.exp(model.params[var]),
            'CI_Lower': np.exp(conf.loc[var][0]),
            'CI_Upper': np.exp(conf.loc[var][1]),
            '显著性': '***' if model.pvalues[var] < 0.001 else '**' if model.pvalues[var] < 0.01 else '*' if model.pvalues[var] < 0.05 else 'ns'
        })
    return pd.DataFrame(results), model, y, model.predict(X)

def plot_coefficients(results_df, title, filename, color_pos='green', color_neg='red'):
    """绘制系数图"""
    plt.figure(figsize=(10, 6))
    vars_list = results_df['变量'].tolist()
    coefs = results_df['系数'].tolist()
    colors = [color_neg if x < 0 else color_pos for x in coefs]
    
    plt.barh(vars_list, coefs, color=colors, alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('系数估计值', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    for i, (coef, sig) in enumerate(zip(coefs, results_df['显著性'])):
        plt.text(coef + (0.02 if coef > 0 else -0.02), i, sig, 
                va='center', ha='left' if coef > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/{filename}")
    plt.close()

def plot_odds_ratios(results_df, filename):
    """绘制Odds Ratios森林图"""
    plt.figure(figsize=(10, 6))
    vars_list = results_df['变量'].tolist()
    ors = results_df['Odds Ratio'].tolist()
    ci_low = results_df['CI_Lower'].tolist()
    ci_high = results_df['CI_Upper'].tolist()
    
    y_pos = np.arange(len(vars_list))
    plt.errorbar(ors, y_pos, xerr=[np.array(ors)-np.array(ci_low), np.array(ci_high)-np.array(ors)], 
                fmt='o', markersize=8, capsize=5)
    
    plt.axvline(x=1, color='red', linestyle='--', label='OR=1 (无影响)')
    plt.yticks(y_pos, vars_list)
    plt.xlabel('Odds Ratio (95% CI)', fontsize=12)
    plt.title('逻辑回归 - Odds Ratios (风险比)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/{filename}")
    plt.close()

def main():
    print("=" * 60)
    print("步骤3: 回归分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    # 1. 线性回归 (OLS) - 全部要素
    print("\n拟合线性回归模型...")
    ols_res, ols_model = fit_ols_model(df, ALL_FEATURES, "线性回归")
    ols_res.to_csv(f"{RESULTS_DIR}/3_线性回归结果.csv", index=False, encoding='utf-8-sig')
    plot_coefficients(ols_res, '线性回归 (OLS) 系数', '6_线性回归系数.png', '#1f77b4', '#ff7f0e')
    
    # 2. 逻辑回归 - 全部要素
    print("拟合逻辑回归模型...")
    logit_res, logit_model, y_true, y_pred = fit_logit_model(df, ALL_FEATURES, "逻辑回归")
    logit_res.to_csv(f"{RESULTS_DIR}/3_逻辑回归结果.csv", index=False, encoding='utf-8-sig')
    plot_coefficients(logit_res, '逻辑回归系数', '7_逻辑回归系数.png', 'green', 'red')
    plot_odds_ratios(logit_res, '8_Odds_Ratios森林图.png')
    
    print("\n回归分析完成！")

if __name__ == "__main__":
    main()
