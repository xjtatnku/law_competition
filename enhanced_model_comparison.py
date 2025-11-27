"""
增强版模型比较：逻辑回归和决策树的详细性能评估
包括：似然比检验(LRT)、AIC、BIC、混淆矩阵、AUC、ROC曲线等
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, accuracy_score,
                            precision_score, recall_score, f1_score)
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, A_CLASS, B_CLASS, C_CLASS,
                    VAR_LABELS, FIGURES_DIR, RESULTS_DIR, RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 逻辑回归模型部分 ========================

def fit_logit_model_detailed(df, features, model_name):
    """
    拟合逻辑回归模型并返回详细结果
    """
    print(f"\n{'='*50}")
    print(f"逻辑回归 - {model_name}")
    print(f"{'='*50}")
    
    # 准备数据
    data = df[features + [TARGET]].dropna()
    X = data[features]
    y = data[TARGET]
    
    print(f"样本数: {len(data)}")
    
    # 使用statsmodels进行详细的统计检验
    X_sm = sm.add_constant(X)
    logit_model = sm.Logit(y, X_sm)
    result = logit_model.fit(disp=False)
    
    # 预测
    y_pred_proba = result.predict(X_sm)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 性能指标
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"对数似然: {result.llf:.4f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")
    print(f"伪R²: {result.prsquared:.4f}")
    
    return {
        'model_name': model_name,
        'features': features,
        'result': result,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'llf': result.llf,
        'aic': result.aic,
        'bic': result.bic,
        'pseudo_r2': result.prsquared,
        'n_params': len(result.params)
    }

def likelihood_ratio_test_lr(model_base, model_full):
    """
    似然比检验（逻辑回归）
    """
    lr_stat = 2 * (model_full['llf'] - model_base['llf'])
    df_diff = model_full['n_params'] - model_base['n_params']
    p_value = stats.chi2.sf(lr_stat, df_diff)
    
    return {
        'LR统计量': lr_stat,
        'p值': p_value,
        '自由度差': df_diff,
        '显著性': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    }

def analyze_logistic_regression_models(df):
    """
    逻辑回归模型全面分析
    """
    print("\n" + "="*60)
    print("逻辑回归模型详细分析")
    print("="*60)
    
    # 拟合四个模型
    model1_lr = fit_logit_model_detailed(df, A_CLASS + B_CLASS, "模型1: A+B")
    model2_lr = fit_logit_model_detailed(df, A_CLASS + B_CLASS + ['C1'], "模型2: A+B+C1")
    model3_lr = fit_logit_model_detailed(df, A_CLASS + B_CLASS + ['C2'], "模型3: A+B+C2")
    model4_lr = fit_logit_model_detailed(df, ALL_FEATURES, "模型4: A+B+C1+C2")
    
    models_lr = [model1_lr, model2_lr, model3_lr, model4_lr]
    
    # 似然比检验
    print("\n" + "="*60)
    print("逻辑回归 - 似然比检验")
    print("="*60)
    
    lrt_results = []
    
    # 模型1 vs 模型2 (C1的贡献)
    lrt_1_2 = likelihood_ratio_test_lr(model1_lr, model2_lr)
    lrt_results.append({
        '对比': '模型1 vs 模型2 (C1)',
        **lrt_1_2
    })
    print(f"\n模型1 vs 模型2 (C1的增量贡献):")
    print(f"  LR统计量 = {lrt_1_2['LR统计量']:.4f}")
    print(f"  p值 = {lrt_1_2['p值']:.4f} {lrt_1_2['显著性']}")
    
    # 模型1 vs 模型3 (C2的贡献)
    lrt_1_3 = likelihood_ratio_test_lr(model1_lr, model3_lr)
    lrt_results.append({
        '对比': '模型1 vs 模型3 (C2)',
        **lrt_1_3
    })
    print(f"\n模型1 vs 模型3 (C2的增量贡献):")
    print(f"  LR统计量 = {lrt_1_3['LR统计量']:.4f}")
    print(f"  p值 = {lrt_1_3['p值']:.4f} {lrt_1_3['显著性']}")
    
    # 模型1 vs 模型4 (C1+C2的联合贡献)
    lrt_1_4 = likelihood_ratio_test_lr(model1_lr, model4_lr)
    lrt_results.append({
        '对比': '模型1 vs 模型4 (C1+C2)',
        **lrt_1_4
    })
    print(f"\n模型1 vs 模型4 (C1+C2的联合增量贡献):")
    print(f"  LR统计量 = {lrt_1_4['LR统计量']:.4f}")
    print(f"  p值 = {lrt_1_4['p值']:.4f} {lrt_1_4['显著性']}")
    
    lrt_df = pd.DataFrame(lrt_results)
    lrt_df.to_csv(f"{RESULTS_DIR}/逻辑回归_似然比检验.csv", 
                  index=False, encoding='utf-8-sig')
    
    # 模型性能对比表
    comparison_lr = pd.DataFrame([{
        '模型': m['model_name'],
        '特征数': len(m['features']),
        '参数数': m['n_params'],
        '对数似然': f"{m['llf']:.4f}",
        'AIC': f"{m['aic']:.4f}",
        'BIC': f"{m['bic']:.4f}",
        '伪R²': f"{m['pseudo_r2']:.4f}",
        '准确率': f"{m['accuracy']:.4f}",
        'AUC': f"{m['auc']:.4f}",
        '精确率': f"{m['precision']:.4f}",
        '召回率': f"{m['recall']:.4f}",
        'F1': f"{m['f1']:.4f}"
    } for m in models_lr])
    
    print("\n" + "="*60)
    print("逻辑回归模型性能对比:")
    print("="*60)
    print(comparison_lr.to_string(index=False))
    comparison_lr.to_csv(f"{RESULTS_DIR}/逻辑回归_模型对比详细.csv", 
                        index=False, encoding='utf-8-sig')
    
    return models_lr, lrt_df, comparison_lr

# ======================== 决策树模型部分 ========================

def fit_decision_tree_detailed(df, features, model_name):
    """
    拟合决策树模型并返回详细结果
    """
    print(f"\n{'='*50}")
    print(f"决策树 - {model_name}")
    print(f"{'='*50}")
    
    # 准备数据
    data = df[features + [TARGET]].dropna()
    X = data[features]
    y = data[TARGET]
    
    print(f"样本数: {len(data)}")
    
    # 训练决策树 - 优化参数
    dt = DecisionTreeClassifier(max_depth=5, 
                                random_state=RANDOM_STATE,
                                min_samples_split=5,
                                min_samples_leaf=3)
    dt.fit(X, y)
    
    # 预测
    y_pred = dt.predict(X)
    y_pred_proba = dt.predict_proba(X)[:, 1]
    
    # 性能指标
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"树深度: {dt.get_depth()}")
    print(f"叶子节点数: {dt.get_n_leaves()}")
    
    return {
        'model_name': model_name,
        'features': features,
        'model': dt,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'tree_depth': dt.get_depth(),
        'n_leaves': dt.get_n_leaves()
    }

def analyze_decision_tree_models(df):
    """
    决策树模型全面分析
    """
    print("\n" + "="*60)
    print("决策树模型详细分析")
    print("="*60)
    
    # 拟合四个模型
    model1_dt = fit_decision_tree_detailed(df, A_CLASS + B_CLASS, "模型1: A+B")
    model2_dt = fit_decision_tree_detailed(df, A_CLASS + B_CLASS + ['C1'], "模型2: A+B+C1")
    model3_dt = fit_decision_tree_detailed(df, A_CLASS + B_CLASS + ['C2'], "模型3: A+B+C2")
    model4_dt = fit_decision_tree_detailed(df, ALL_FEATURES, "模型4: A+B+C1+C2")
    
    models_dt = [model1_dt, model2_dt, model3_dt, model4_dt]
    
    # 模型性能对比表
    comparison_dt = pd.DataFrame([{
        '模型': m['model_name'],
        '特征数': len(m['features']),
        '树深度': m['tree_depth'],
        '叶子节点数': m['n_leaves'],
        '准确率': f"{m['accuracy']:.4f}",
        'AUC': f"{m['auc']:.4f}",
        '精确率': f"{m['precision']:.4f}",
        '召回率': f"{m['recall']:.4f}",
        'F1': f"{m['f1']:.4f}"
    } for m in models_dt])
    
    print("\n" + "="*60)
    print("决策树模型性能对比:")
    print("="*60)
    print(comparison_dt.to_string(index=False))
    comparison_dt.to_csv(f"{RESULTS_DIR}/决策树_模型对比详细.csv", 
                        index=False, encoding='utf-8-sig')
    
    return models_dt, comparison_dt

# ======================== 可视化部分 ========================

def plot_confusion_matrices_lr(models_lr):
    """
    绘制逻辑回归的混淆矩阵
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model in enumerate(models_lr):
        ax = axes[idx]
        cm = model['confusion_matrix']
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': '样本数'},
                   xticklabels=['预测:否', '预测:是'],
                   yticklabels=['实际:否', '实际:是'])
        
        ax.set_title(f"{model['model_name']}\n准确率={model['accuracy']:.3f}, AUC={model['auc']:.3f}",
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('实际值', fontsize=10)
        ax.set_xlabel('预测值', fontsize=10)
    
    plt.suptitle('逻辑回归 - 混淆矩阵对比', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_逻辑回归_混淆矩阵.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_逻辑回归_混淆矩阵.png")
    plt.close()

def plot_confusion_matrices_dt(models_dt):
    """
    绘制决策树的混淆矩阵
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model in enumerate(models_dt):
        ax = axes[idx]
        cm = model['confusion_matrix']
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                   cbar_kws={'label': '样本数'},
                   xticklabels=['预测:否', '预测:是'],
                   yticklabels=['实际:否', '实际:是'])
        
        ax.set_title(f"{model['model_name']}\n准确率={model['accuracy']:.3f}, AUC={model['auc']:.3f}",
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('实际值', fontsize=10)
        ax.set_xlabel('预测值', fontsize=10)
    
    plt.suptitle('决策树 - 混淆矩阵对比', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_决策树_混淆矩阵.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_决策树_混淆矩阵.png")
    plt.close()

def plot_roc_curves_lr(models_lr):
    """
    绘制逻辑回归的ROC曲线
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, model in enumerate(models_lr):
        fpr, tpr, _ = roc_curve(model['y_true'], model['y_pred_proba'])
        auc = model['auc']
        plt.plot(fpr, tpr, linewidth=2.5, color=colors[idx],
                label=f"{model['model_name']} (AUC={auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机猜测 (AUC=0.5)')
    plt.xlabel('假阳性率 (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('真阳性率 (TPR)', fontsize=12, fontweight='bold')
    plt.title('逻辑回归 - ROC曲线对比', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_逻辑回归_ROC曲线.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_逻辑回归_ROC曲线.png")
    plt.close()

def plot_roc_curves_dt(models_dt):
    """
    绘制决策树的ROC曲线
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, model in enumerate(models_dt):
        fpr, tpr, _ = roc_curve(model['y_true'], model['y_pred_proba'])
        auc = model['auc']
        plt.plot(fpr, tpr, linewidth=2.5, color=colors[idx],
                label=f"{model['model_name']} (AUC={auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机猜测 (AUC=0.5)')
    plt.xlabel('假阳性率 (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('真阳性率 (TPR)', fontsize=12, fontweight='bold')
    plt.title('决策树 - ROC曲线对比', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_决策树_ROC曲线.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_决策树_ROC曲线.png")
    plt.close()

def plot_aic_bic_comparison_lr(models_lr):
    """
    绘制逻辑回归的AIC和BIC对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = [m['model_name'] for m in models_lr]
    aic_values = [m['aic'] for m in models_lr]
    bic_values = [m['bic'] for m in models_lr]
    
    x = np.arange(len(model_names))
    
    # AIC
    bars1 = ax1.bar(x, aic_values, color='steelblue', alpha=0.7, width=0.6)
    ax1.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AIC值', fontsize=12, fontweight='bold')
    ax1.set_title('AIC对比 (值越小越好)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('模型', 'M') for m in model_names], fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, aic_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # BIC
    bars2 = ax2.bar(x, bic_values, color='coral', alpha=0.7, width=0.6)
    ax2.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BIC值', fontsize=12, fontweight='bold')
    ax2.set_title('BIC对比 (值越小越好)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('模型', 'M') for m in model_names], fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, bic_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('逻辑回归 - AIC与BIC信息准则对比', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_逻辑回归_AIC_BIC对比.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_逻辑回归_AIC_BIC对比.png")
    plt.close()

def plot_performance_metrics_comparison(models_lr, models_dt):
    """
    绘制逻辑回归与决策树的性能指标对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['accuracy', 'auc', 'precision', 'recall']
    metric_names = ['准确率', 'AUC', '精确率', '召回率']
    
    model_names_short = ['M1:A+B', 'M2:A+B+C1', 'M3:A+B+C2', 'M4:全部']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        lr_values = [m[metric] for m in models_lr]
        dt_values = [m[metric] for m in models_dt]
        
        x = np.arange(len(model_names_short))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lr_values, width, label='逻辑回归', 
                      color='#4C72B0', alpha=0.8)
        bars2 = ax.bar(x + width/2, dt_values, width, label='决策树', 
                      color='#55A868', alpha=0.8)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name}对比', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names_short, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('逻辑回归 vs 决策树 - 性能指标对比', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_LR_vs_DT_性能对比.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_LR_vs_DT_性能对比.png")
    plt.close()

def plot_lrt_results(lrt_df):
    """
    绘制似然比检验结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    comparisons = lrt_df['对比'].tolist()
    lr_stats = lrt_df['LR统计量'].tolist()
    p_values = lrt_df['p值'].tolist()
    
    # LR统计量
    bars1 = ax1.barh(comparisons, lr_stats, color='steelblue', alpha=0.7)
    ax1.set_xlabel('LR统计量', fontsize=12, fontweight='bold')
    ax1.set_title('似然比统计量', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars1, lr_stats):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    # p值（负对数）
    neg_log_p = [-np.log10(p) for p in p_values]
    bars2 = ax2.barh(comparisons, neg_log_p, color='coral', alpha=0.7)
    ax2.set_xlabel('-log10(p值)', fontsize=12, fontweight='bold')
    ax2.set_title('显著性水平', fontsize=13, fontweight='bold')
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
               linewidth=2, label='p=0.05')
    ax2.axvline(x=-np.log10(0.01), color='orange', linestyle='--', 
               linewidth=2, label='p=0.01')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val, p in zip(bars2, neg_log_p, p_values):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f'p={p:.4f}',
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.suptitle('逻辑回归 - 似然比检验结果', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/Enhanced_逻辑回归_LRT结果.png", 
                dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/Enhanced_逻辑回归_LRT结果.png")
    plt.close()

def main():
    """
    主函数
    """
    print("=" * 60)
    print("增强版模型比较 - 详细性能评估")
    print("=" * 60)
    
    df = get_clean_data()
    
    # 逻辑回归分析
    models_lr, lrt_df, comparison_lr = analyze_logistic_regression_models(df)
    
    # 决策树分析
    models_dt, comparison_dt = analyze_decision_tree_models(df)
    
    # 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    plot_confusion_matrices_lr(models_lr)
    plot_confusion_matrices_dt(models_dt)
    plot_roc_curves_lr(models_lr)
    plot_roc_curves_dt(models_dt)
    plot_aic_bic_comparison_lr(models_lr)
    plot_lrt_results(lrt_df)
    plot_performance_metrics_comparison(models_lr, models_dt)
    
    print("\n" + "=" * 60)
    print("增强版模型比较分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
