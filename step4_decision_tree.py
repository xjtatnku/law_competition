"""
第四步：决策树分析
核心功能：决策树模型构建、结构可视化、特征重要性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, VAR_LABELS, FIGURES_DIR, RESULTS_DIR, RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def train_and_plot_tree(df, features, model_name, filename_suffix):
    """训练决策树并绘图"""
    data = df[features + [TARGET]].dropna()
    X = data[features]
    y = data[TARGET]
    
    dt = DecisionTreeClassifier(max_depth=5, 
                                min_samples_split=5,
                                min_samples_leaf=3,
                                random_state=RANDOM_STATE)
    dt.fit(X, y)
    
    # 绘制树结构
    feature_names = [VAR_LABELS.get(f, f) for f in features]
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_names, class_names=['未认定', '认定'], 
             filled=True, rounded=True, fontsize=10)
    plt.title(f'{model_name} - 决策树结构 (深度=5)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/9_{filename_suffix}_结构图.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/9_{filename_suffix}_结构图.png")
    plt.close()
    
    return dt, feature_names

def plot_feature_importance(models_info):
    """绘制特征重要性对比"""
    fig, axes = plt.subplots(len(models_info), 1, figsize=(12, 5*len(models_info)))
    if len(models_info) == 1: axes = [axes]
    
    for idx, info in enumerate(models_info):
        ax = axes[idx]
        dt = info['model']
        feats = info['features']
        
        importances = dt.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 过滤掉重要性为0的特征
        non_zero = importances[indices] > 0
        sorted_feats = np.array(feats)[indices][non_zero]
        sorted_imps = importances[indices][non_zero]
        
        bars = ax.barh(sorted_feats, sorted_imps, color='steelblue', alpha=0.7)
        ax.set_title(f"{info['name']} - 特征重要性", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, val in zip(bars, sorted_imps):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                   va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/10_决策树特征重要性.png", dpi=300, bbox_inches='tight')
    print(f"图表已保存: {FIGURES_DIR}/10_决策树特征重要性.png")
    plt.close()

def main():
    print("=" * 60)
    print("步骤4: 决策树分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    models_info = []
    
    # 训练并在列表中保存模型信息
    dt_all, feats_all = train_and_plot_tree(df, ALL_FEATURES, "全部要素", "全部要素")
    models_info.append({'name': '全部要素', 'model': dt_all, 'features': feats_all})
    
    # 绘制特征重要性
    plot_feature_importance(models_info)
    
    print("\n决策树分析完成！")

if __name__ == "__main__":
    main()
