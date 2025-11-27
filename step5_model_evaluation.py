"""
第五步：模型评估与综合比较
核心功能：ROC曲线、混淆矩阵、似然比检验(LRT)、AIC/BIC比较、综合性能指标
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, A_CLASS, B_CLASS, VAR_LABELS, FIGURES_DIR, RESULTS_DIR, RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def fit_models(df):
    """拟合用于比较的模型"""
    # 逻辑回归模型
    data_lr = df[ALL_FEATURES + [TARGET]].dropna()
    X_lr = sm.add_constant(data_lr[ALL_FEATURES])
    y_lr = data_lr[TARGET]
    lr_model = sm.Logit(y_lr, X_lr).fit(disp=False)
    
    # 决策树模型
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=3, random_state=RANDOM_STATE)
    dt_model.fit(data_lr[ALL_FEATURES], y_lr)
    
    return {
        'LR': {'model': lr_model, 'X': X_lr, 'y': y_lr, 'name': '逻辑回归'},
        'DT': {'model': dt_model, 'X': data_lr[ALL_FEATURES], 'y': y_lr, 'name': '决策树'}
    }

def plot_roc_comparison(models, filename):
    """绘制ROC曲线对比"""
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#2ca02c']
    
    for idx, (key, m) in enumerate(models.items()):
        if key == 'LR':
            y_prob = m['model'].predict(m['X'])
        else:
            y_prob = m['model'].predict_proba(m['X'])[:, 1]
            
        fpr, tpr, _ = roc_curve(m['y'], y_prob)
        auc = roc_auc_score(m['y'], y_prob)
        plt.plot(fpr, tpr, linewidth=2.5, color=colors[idx], label=f"{m['name']} (AUC={auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('逻辑回归 vs 决策树 - ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/{filename}")
    plt.close()

def plot_confusion_matrices(models):
    """绘制混淆矩阵对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (key, m) in enumerate(models.items()):
        ax = axes[idx]
        if key == 'LR':
            y_pred = (m['model'].predict(m['X']) >= 0.5).astype(int)
        else:
            y_pred = m['model'].predict(m['X'])
            
        cm = confusion_matrix(m['y'], y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{m['name']} - 混淆矩阵", fontsize=12, fontweight='bold')
        ax.set_xlabel('预测值')
        ax.set_ylabel('实际值')
        
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/11_混淆矩阵对比.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/11_混淆矩阵对比.png")
    plt.close()

def calculate_metrics(models):
    """计算性能指标"""
    results = []
    for key, m in models.items():
        if key == 'LR':
            y_pred = (m['model'].predict(m['X']) >= 0.5).astype(int)
            y_prob = m['model'].predict(m['X'])
        else:
            y_pred = m['model'].predict(m['X'])
            y_prob = m['model'].predict_proba(m['X'])[:, 1]
            
        results.append({
            '模型': m['name'],
            '准确率': accuracy_score(m['y'], y_pred),
            '精确率': precision_score(m['y'], y_pred),
            '召回率': recall_score(m['y'], y_pred),
            'F1分数': f1_score(m['y'], y_pred),
            'AUC': roc_auc_score(m['y'], y_prob)
        })
    return pd.DataFrame(results)

def lrt_and_aic_analysis(df):
    """似然比检验与AIC分析 (仅针对逻辑回归嵌套模型)"""
    # 基准模型: A+B
    data = df[ALL_FEATURES + [TARGET]].dropna()
    X_base = sm.add_constant(data[A_CLASS + B_CLASS])
    X_full = sm.add_constant(data[ALL_FEATURES])
    y = data[TARGET]
    
    model_base = sm.Logit(y, X_base).fit(disp=False)
    model_full = sm.Logit(y, X_full).fit(disp=False)
    
    # LRT
    lr_stat = 2 * (model_full.llf - model_base.llf)
    df_diff = model_full.df_model - model_base.df_model
    p_val = stats.chi2.sf(lr_stat, df_diff)
    
    # 绘图 AIC
    plt.figure(figsize=(8, 6))
    plt.bar(['基准模型(A+B)', '完整模型(全部)'], [model_base.aic, model_full.aic], 
           color=['#ff7f0e', '#1f77b4'], alpha=0.7, width=0.5)
    plt.title('AIC信息准则对比 (越小越好)', fontsize=14, fontweight='bold')
    plt.ylabel('AIC值')
    
    # 添加数值
    plt.text(0, model_base.aic, f'{model_base.aic:.1f}', ha='center', va='bottom')
    plt.text(1, model_full.aic, f'{model_full.aic:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/13_AIC对比.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/13_AIC对比.png")
    plt.close()
    
    return {
        'LR统计量': lr_stat,
        'p值': p_val,
        'AIC差异': model_base.aic - model_full.aic
    }

def main():
    print("=" * 60)
    print("步骤5: 模型评估与综合比较")
    print("=" * 60)
    
    df = get_clean_data()
    models = fit_models(df)
    
    # 1. ROC曲线
    plot_roc_comparison(models, '12_ROC曲线对比.png')
    
    # 2. 混淆矩阵
    plot_confusion_matrices(models)
    
    # 3. 性能指标表
    metrics = calculate_metrics(models)
    metrics.to_csv(f"{RESULTS_DIR}/4_模型综合性能对比.csv", index=False, encoding='utf-8-sig')
    print(f"\n模型性能指标:\n{metrics}")
    
    # 4. 似然比检验与AIC
    lrt_res = lrt_and_aic_analysis(df)
    print(f"\n似然比检验(LRT)结果: LR={lrt_res['LR统计量']:.4f}, p={lrt_res['p值']:.4f}")
    
    print("\n模型评估完成！")

if __name__ == "__main__":
    main()
