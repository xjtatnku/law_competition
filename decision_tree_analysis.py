"""
决策树分析：规则提取与要素重要性排序
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import get_clean_data
from config import (ALL_FEATURES, TARGET, A_CLASS, B_CLASS, C_CLASS,
                    VAR_LABELS, FIGURES_DIR, RESULTS_DIR, RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def train_decision_tree(df, features, model_name, max_depth=4):
    """
    训练决策树模型
    
    Args:
        df: 数据框
        features: 特征列表
        model_name: 模型名称
        max_depth: 最大深度
    
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
    print(f"特征数: {len(features)}")
    
    # 训练决策树
    dt = DecisionTreeClassifier(max_depth=max_depth, 
                                random_state=RANDOM_STATE,
                                min_samples_split=10,
                                min_samples_leaf=5)
    dt.fit(X, y)
    
    # 预测
    y_pred = dt.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\n模型性能:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  树的深度: {dt.get_depth()}")
    print(f"  叶子节点数: {dt.get_n_leaves()}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        '特征': [VAR_LABELS.get(f, f) for f in features],
        '重要性': dt.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print(f"\n特征重要性排序:")
    print(feature_importance.to_string(index=False))
    
    # 混淆矩阵
    cm = confusion_matrix(y, y_pred)
    print(f"\n混淆矩阵:")
    cm_df = pd.DataFrame(cm, 
                        index=['实际=0', '实际=1'],
                        columns=['预测=0', '预测=1'])
    print(cm_df)
    
    # 分类报告
    print(f"\n分类报告:")
    print(classification_report(y, y_pred, 
                               target_names=['未认定劳动关系', '认定劳动关系']))
    
    return {
        'model_name': model_name,
        'features': features,
        'model': dt,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'confusion_matrix': cm
    }

def plot_tree_structure(model_result, filename):
    """
    绘制决策树结构图
    """
    dt = model_result['model']
    features = model_result['features']
    feature_names = [VAR_LABELS.get(f, f) for f in features]
    
    plt.figure(figsize=(20, 12))
    plot_tree(dt, 
             feature_names=feature_names,
             class_names=['未认定', '认定'],
             filled=True,
             rounded=True,
             fontsize=10)
    plt.title(f"{model_result['model_name']} - 决策树结构", 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"决策树结构图已保存: {FIGURES_DIR}/{filename}")
    plt.close()

def plot_feature_importance(model_results_list):
    """
    绘制特征重要性对比图
    """
    fig, axes = plt.subplots(len(model_results_list), 1, 
                            figsize=(12, 5*len(model_results_list)))
    
    if len(model_results_list) == 1:
        axes = [axes]
    
    for idx, model_result in enumerate(model_results_list):
        ax = axes[idx]
        fi_df = model_result['feature_importance']
        
        # 只显示重要性>0的特征
        fi_df_nonzero = fi_df[fi_df['重要性'] > 0]
        
        if len(fi_df_nonzero) > 0:
            bars = ax.barh(fi_df_nonzero['特征'], fi_df_nonzero['重要性'], 
                          color='steelblue', alpha=0.7)
            ax.set_xlabel('重要性', fontsize=12)
            ax.set_title(f"{model_result['model_name']} - 特征重要性", 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}',
                       ha='left', va='center', fontsize=10)
        else:
            ax.text(0.5, 0.5, '所有特征重要性为0', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/11_决策树特征重要性.png", dpi=300, bbox_inches='tight')
    print(f"特征重要性图已保存: {FIGURES_DIR}/11_决策树特征重要性.png")
    plt.close()

def export_tree_rules(model_result):
    """
    导出决策树规则（文本格式）
    """
    dt = model_result['model']
    features = model_result['features']
    feature_names = [VAR_LABELS.get(f, f) for f in features]
    
    tree_rules = export_text(dt, feature_names=feature_names)
    
    print(f"\n{model_result['model_name']} - 决策规则:")
    print(tree_rules)
    
    # 保存到文件
    filename = f"{RESULTS_DIR}/决策树规则_{model_result['model_name'].replace(':', '_').replace(' ', '_')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{model_result['model_name']} - 决策规则\n")
        f.write("=" * 60 + "\n\n")
        f.write(tree_rules)
    
    print(f"决策规则已保存: {filename}")
    
    return tree_rules

def compare_with_without_c(df):
    """
    对比有无C1/C2时决策树的差异
    """
    print("\n" + "="*60)
    print("对比分析：C1/C2对决策树的影响")
    print("="*60)
    
    models = []
    
    # 模型1: 仅A+B类
    model1 = train_decision_tree(df, A_CLASS + B_CLASS, "决策树1: A类+B类", max_depth=4)
    models.append(model1)
    model1['feature_importance'].to_csv(
        f"{RESULTS_DIR}/决策树_模型1_特征重要性.csv", 
        index=False, encoding='utf-8-sig')
    plot_tree_structure(model1, "12_决策树1_AB类.png")
    export_tree_rules(model1)
    
    # 模型2: A+B+C1
    model2 = train_decision_tree(df, A_CLASS + B_CLASS + ['C1'], 
                                "决策树2: A类+B类+C1", max_depth=4)
    models.append(model2)
    model2['feature_importance'].to_csv(
        f"{RESULTS_DIR}/决策树_模型2_特征重要性.csv", 
        index=False, encoding='utf-8-sig')
    plot_tree_structure(model2, "13_决策树2_AB类+C1.png")
    export_tree_rules(model2)
    
    # 模型3: A+B+C2
    model3 = train_decision_tree(df, A_CLASS + B_CLASS + ['C2'], 
                                "决策树3: A类+B类+C2", max_depth=4)
    models.append(model3)
    model3['feature_importance'].to_csv(
        f"{RESULTS_DIR}/决策树_模型3_特征重要性.csv", 
        index=False, encoding='utf-8-sig')
    plot_tree_structure(model3, "14_决策树3_AB类+C2.png")
    export_tree_rules(model3)
    
    # 模型4: 全部要素
    model4 = train_decision_tree(df, ALL_FEATURES, 
                                "决策树4: 全部要素", max_depth=4)
    models.append(model4)
    model4['feature_importance'].to_csv(
        f"{RESULTS_DIR}/决策树_模型4_特征重要性.csv", 
        index=False, encoding='utf-8-sig')
    plot_tree_structure(model4, "15_决策树4_全部要素.png")
    export_tree_rules(model4)
    
    # 绘制特征重要性对比
    plot_feature_importance(models)
    
    # 模型性能对比
    comparison = pd.DataFrame([{
        '模型': m['model_name'],
        '特征数': len(m['features']),
        '准确率': f"{m['accuracy']:.4f}",
        '树深度': m['model'].get_depth(),
        '叶子节点数': m['model'].get_n_leaves()
    } for m in models])
    
    print("\n" + "="*60)
    print("决策树模型性能对比:")
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{RESULTS_DIR}/决策树_模型对比.csv", 
                     index=False, encoding='utf-8-sig')
    
    # 分析C1和C2在决策树中的作用
    print("\n" + "="*60)
    print("C1和C2在决策树中的重要性分析:")
    print("="*60)
    
    # 在包含C1的模型中查看C1的重要性
    c1_importance_in_model2 = model2['feature_importance']
    c1_row = c1_importance_in_model2[c1_importance_in_model2['特征'] == VAR_LABELS['C1']]
    if not c1_row.empty:
        c1_imp = c1_row['重要性'].values[0]
        print(f"模型2中C1的重要性: {c1_imp:.4f}")
        c1_rank = (c1_importance_in_model2['重要性'] > c1_imp).sum() + 1
        print(f"C1在模型2中的重要性排名: 第{c1_rank}位/{len(model2['features'])}个特征")
    
    # 在包含C2的模型中查看C2的重要性
    c2_importance_in_model3 = model3['feature_importance']
    c2_row = c2_importance_in_model3[c2_importance_in_model3['特征'] == VAR_LABELS['C2']]
    if not c2_row.empty:
        c2_imp = c2_row['重要性'].values[0]
        print(f"模型3中C2的重要性: {c2_imp:.4f}")
        c2_rank = (c2_importance_in_model3['重要性'] > c2_imp).sum() + 1
        print(f"C2在模型3中的重要性排名: 第{c2_rank}位/{len(model3['features'])}个特征")
    
    # 在全部要素模型中查看C1和C2
    c1_row_full = model4['feature_importance'][model4['feature_importance']['特征'] == VAR_LABELS['C1']]
    c2_row_full = model4['feature_importance'][model4['feature_importance']['特征'] == VAR_LABELS['C2']]
    
    if not c1_row_full.empty:
        print(f"模型4中C1的重要性: {c1_row_full['重要性'].values[0]:.4f}")
    if not c2_row_full.empty:
        print(f"模型4中C2的重要性: {c2_row_full['重要性'].values[0]:.4f}")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("决策树分析")
    print("=" * 60)
    
    df = get_clean_data()
    
    compare_with_without_c(df)
    
    print("\n" + "=" * 60)
    print("决策树分析完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
