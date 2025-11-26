"""
逻辑回归分析：量化各要素对劳动关系认定的边际影响
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import statsmodels.api as sm
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, A_CLASS, B_CLASS, C_CLASS,
                    VAR_LABELS, FIGURES_DIR, RESULTS_DIR, RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def fit_logistic_model(df, features, model_name):
    """
    拟合逻辑回归模型
    
    Args:
        df: 数据框
        features: 特征列表
        model_name: 模型名称
    
    Returns:
        dict: 模型结果
    """
    print(f"\n{'='*50}")
    print(f"模型: {model_name}")
    print(f"{'='*50}")
    
    # 准备数据
    data = df[features + [TARGET]].dropna()
    X = data[features]
    y = data[TARGET]
    
    print(f"样本数: {len(data)}")
    print(f"认定劳动关系: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # 使用statsmodels进行详细的统计检验
    X_sm = sm.add_constant(X)
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=False)
    
    print("\n模型摘要:")
    print(result.summary())
    
    # 提取关键统计量
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()
    
    # 计算odds ratio
    odds_ratios = np.exp(params)
    odds_conf_int = np.exp(conf_int)
    
    # 构建结果表
    results_table = []
    for var in features:
        results_table.append({
            '变量': VAR_LABELS.get(var, var),
            '系数': f'{params[var]:.4f}',
            'p值': f'{pvalues[var]:.4f}',
            '显著性': '***' if pvalues[var] < 0.001 else '**' if pvalues[var] < 0.01 else '*' if pvalues[var] < 0.05 else 'ns',
            'Odds Ratio': f'{odds_ratios[var]:.4f}',
            'OR 95% CI': f'[{odds_conf_int.loc[var, 0]:.4f}, {odds_conf_int.loc[var, 1]:.4f}]'
        })
    
    results_df = pd.DataFrame(results_table)
    print("\n回归系数与Odds Ratio:")
    print(results_df.to_string(index=False))
    
    # 预测与评估
    y_pred_proba = result.predict(X_sm)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = (y_pred == y).sum() / len(y)
    auc = roc_auc_score(y, y_pred_proba)
    
    print(f"\n模型性能:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  伪R²: {result.prsquared:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    print(f"\n混淆矩阵:")
    cm_df = pd.DataFrame(cm, 
                        index=['实际=0', '实际=1'],
                        columns=['预测=0', '预测=1'])
    print(cm_df)
    
    return {
        'model_name': model_name,
        'features': features,
        'result': result,
        'results_df': results_df,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'auc': auc,
        'pseudo_r2': result.prsquared,
        'confusion_matrix': cm
    }

def plot_coefficients(model_results_list):
    """
    绘制不同模型的系数对比图
    """
    fig, axes = plt.subplots(len(model_results_list), 1, 
                            figsize=(12, 4*len(model_results_list)))
    
    if len(model_results_list) == 1:
        axes = [axes]
    
    for idx, model_result in enumerate(model_results_list):
        ax = axes[idx]
        results_df = model_result['results_df']
        
        # 提取系数
        vars_list = results_df['变量'].tolist()
        coefs = [float(x) for x in results_df['系数'].tolist()]
        colors = ['red' if x < 0 else 'green' for x in coefs]
        
        # 绘制条形图
        bars = ax.barh(vars_list, coefs, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('回归系数', fontsize=12)
        ax.set_title(f"{model_result['model_name']} - 回归系数", 
                    fontsize=12, fontweight='bold')
        
        # 添加显著性标记
        for i, (var, coef, sig) in enumerate(zip(vars_list, coefs, results_df['显著性'])):
            ax.text(coef + (0.05 if coef > 0 else -0.05), i, sig, 
                   va='center', ha='left' if coef > 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/8_逻辑回归系数对比.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/8_逻辑回归系数对比.png")
    plt.close()

def plot_odds_ratios(model_results_list):
    """
    绘制Odds Ratios森林图
    """
    fig, axes = plt.subplots(len(model_results_list), 1, 
                            figsize=(12, 4*len(model_results_list)))
    
    if len(model_results_list) == 1:
        axes = [axes]
    
    for idx, model_result in enumerate(model_results_list):
        ax = axes[idx]
        results_df = model_result['results_df']
        
        vars_list = results_df['变量'].tolist()
        odds_ratios = [float(x) for x in results_df['Odds Ratio'].tolist()]
        
        # 解析置信区间
        ci_low = []
        ci_high = []
        for ci_str in results_df['OR 95% CI']:
            ci_str = ci_str.strip('[]')
            low, high = ci_str.split(', ')
            ci_low.append(float(low))
            ci_high.append(float(high))
        
        # 绘制森林图
        y_pos = np.arange(len(vars_list))
        ax.errorbar(odds_ratios, y_pos, 
                   xerr=[np.array(odds_ratios) - np.array(ci_low), 
                         np.array(ci_high) - np.array(odds_ratios)],
                   fmt='o', markersize=8, capsize=5, capthick=2)
        
        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='OR=1 (无影响)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vars_list)
        ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
        ax.set_title(f"{model_result['model_name']} - Odds Ratios", 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/9_Odds_Ratios森林图.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/9_Odds_Ratios森林图.png")
    plt.close()

def plot_roc_curves(model_results_list):
    """
    绘制ROC曲线
    """
    plt.figure(figsize=(10, 8))
    
    for model_result in model_results_list:
        fpr, tpr, _ = roc_curve(model_result['y_true'], model_result['y_pred_proba'])
        auc = model_result['auc']
        plt.plot(fpr, tpr, linewidth=2, 
                label=f"{model_result['model_name']} (AUC={auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/10_ROC曲线对比.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/10_ROC曲线对比.png")
    plt.close()

def main():
    """
    主函数
    """
    print("=" * 60)
    print("逻辑回归分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    # 建立多个模型进行对比
    models = []
    
    # 模型1: 仅A类+B类
    model1 = fit_logistic_model(df, A_CLASS + B_CLASS, "模型1: A类+B类")
    models.append(model1)
    model1['results_df'].to_csv(f"{RESULTS_DIR}/逻辑回归_模型1_AB类.csv", 
                                index=False, encoding='utf-8-sig')
    
    # 模型2: A类+B类+C1
    model2 = fit_logistic_model(df, A_CLASS + B_CLASS + ['C1'], "模型2: A类+B类+C1")
    models.append(model2)
    model2['results_df'].to_csv(f"{RESULTS_DIR}/逻辑回归_模型2_AB类+C1.csv", 
                                index=False, encoding='utf-8-sig')
    
    # 模型3: A类+B类+C2
    model3 = fit_logistic_model(df, A_CLASS + B_CLASS + ['C2'], "模型3: A类+B类+C2")
    models.append(model3)
    model3['results_df'].to_csv(f"{RESULTS_DIR}/逻辑回归_模型3_AB类+C2.csv", 
                                index=False, encoding='utf-8-sig')
    
    # 模型4: 全部要素
    model4 = fit_logistic_model(df, ALL_FEATURES, "模型4: 全部要素")
    models.append(model4)
    model4['results_df'].to_csv(f"{RESULTS_DIR}/逻辑回归_模型4_全部要素.csv", 
                                index=False, encoding='utf-8-sig')
    
    # 绘制对比图
    plot_coefficients(models)
    plot_odds_ratios(models)
    plot_roc_curves(models)
    
    # 模型性能对比表
    comparison = pd.DataFrame([{
        '模型': m['model_name'],
        '特征数': len(m['features']),
        '准确率': f"{m['accuracy']:.4f}",
        'AUC': f"{m['auc']:.4f}",
        '伪R²': f"{m['pseudo_r2']:.4f}"
    } for m in models])
    
    print("\n" + "=" * 60)
    print("模型性能对比:")
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{RESULTS_DIR}/逻辑回归_模型对比.csv", 
                     index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("逻辑回归分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
