"""
冗余性检验：用A类预测C1，用B类预测C2
检验C1/C2是否能被其他要素完全解释
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from data_loader import get_clean_data
from config import (A_CLASS, B_CLASS, VAR_LABELS, 
                    FIGURES_DIR, RESULTS_DIR, RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def predict_c1_from_a(df):
    """
    用A类要素预测C1
    """
    print("\n=== 用A类要素预测C1 ===")
    
    # 准备数据（去除C1缺失的行）
    data = df[A_CLASS + ['C1']].dropna()
    X = data[A_CLASS]
    y = data['C1']
    
    print(f"有效样本数: {len(data)}")
    print(f"C1取值分布: {y.value_counts().sort_index().to_dict()}")
    
    # 线性回归（把-1,0,1当作连续变量）
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    r2 = r2_score(y, y_pred_lr)
    rmse = np.sqrt(mean_squared_error(y, y_pred_lr))
    
    print(f"\n线性回归结果:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  回归系数:")
    for i, var in enumerate(A_CLASS):
        print(f"    {VAR_LABELS[var]}: {lr.coef_[i]:.4f}")
    print(f"  截距: {lr.intercept_:.4f}")
    
    # 有序逻辑回归的近似：多分类逻辑回归
    # 将-1,0,1映射为0,1,2
    y_encoded = y + 1  # -1->0, 0->1, 1->2
    
    mlr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=RANDOM_STATE)
    mlr.fit(X, y_encoded)
    y_pred_mlr = mlr.predict(X)
    
    accuracy = accuracy_score(y_encoded, y_pred_mlr)
    
    print(f"\n多分类逻辑回归结果:")
    print(f"  准确率 = {accuracy:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_encoded, y_pred_mlr)
    print(f"  混淆矩阵:")
    cm_df = pd.DataFrame(cm, 
                        index=['实际=-1', '实际=0', '实际=1'],
                        columns=['预测=-1', '预测=0', '预测=1'])
    print(cm_df)
    
    # 保存结果
    results = {
        '预测目标': 'C1',
        '预测变量': ', '.join([VAR_LABELS[v] for v in A_CLASS]),
        '样本数': len(data),
        '线性回归R²': f'{r2:.4f}',
        '线性回归RMSE': f'{rmse:.4f}',
        '逻辑回归准确率': f'{accuracy:.4f}'
    }
    
    # 绘图：实际值 vs 预测值
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 线性回归散点图
    axes[0].scatter(y, y_pred_lr, alpha=0.5, s=30)
    axes[0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='完美预测线')
    axes[0].set_xlabel('C1实际值', fontsize=12)
    axes[0].set_ylabel('C1预测值(线性回归)', fontsize=12)
    axes[0].set_title(f'A类预测C1 - 线性回归\nR2={r2:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 混淆矩阵热力图
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
               xticklabels=['预测=-1', '预测=0', '预测=1'],
               yticklabels=['实际=-1', '实际=0', '实际=1'])
    axes[1].set_title(f'A类预测C1 - 逻辑回归混淆矩阵\n准确率={accuracy:.4f}', 
                     fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/6_A类预测C1.png", dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {FIGURES_DIR}/6_A类预测C1.png")
    plt.close()
    
    return results

def predict_c2_from_b(df):
    """
    用B类要素预测C2
    """
    print("\n=== 用B类要素预测C2 ===")
    
    # 准备数据
    data = df[B_CLASS + ['C2']].dropna()
    X = data[B_CLASS]
    y = data['C2']
    
    print(f"有效样本数: {len(data)}")
    print(f"C2取值分布: {y.value_counts().sort_index().to_dict()}")
    
    # 线性回归
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    r2 = r2_score(y, y_pred_lr)
    rmse = np.sqrt(mean_squared_error(y, y_pred_lr))
    
    print(f"\n线性回归结果:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  回归系数:")
    for i, var in enumerate(B_CLASS):
        print(f"    {VAR_LABELS[var]}: {lr.coef_[i]:.4f}")
    print(f"  截距: {lr.intercept_:.4f}")
    
    # 多分类逻辑回归
    y_encoded = y + 1
    
    mlr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=RANDOM_STATE)
    mlr.fit(X, y_encoded)
    y_pred_mlr = mlr.predict(X)
    
    accuracy = accuracy_score(y_encoded, y_pred_mlr)
    
    print(f"\n多分类逻辑回归结果:")
    print(f"  准确率 = {accuracy:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_encoded, y_pred_mlr)
    print(f"  混淆矩阵:")
    cm_df = pd.DataFrame(cm, 
                        index=['实际=-1', '实际=0', '实际=1'],
                        columns=['预测=-1', '预测=0', '预测=1'])
    print(cm_df)
    
    # 保存结果
    results = {
        '预测目标': 'C2',
        '预测变量': ', '.join([VAR_LABELS[v] for v in B_CLASS]),
        '样本数': len(data),
        '线性回归R²': f'{r2:.4f}',
        '线性回归RMSE': f'{rmse:.4f}',
        '逻辑回归准确率': f'{accuracy:.4f}'
    }
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 线性回归散点图
    axes[0].scatter(y, y_pred_lr, alpha=0.5, s=30)
    axes[0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='完美预测线')
    axes[0].set_xlabel('C2实际值', fontsize=12)
    axes[0].set_ylabel('C2预测值(线性回归)', fontsize=12)
    axes[0].set_title(f'B类预测C2 - 线性回归\nR2={r2:.4f}', 
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 混淆矩阵热力图
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
               xticklabels=['预测=-1', '预测=0', '预测=1'],
               yticklabels=['实际=-1', '实际=0', '实际=1'])
    axes[1].set_title(f'B类预测C2 - 逻辑回归混淆矩阵\n准确率={accuracy:.4f}', 
                     fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/7_B类预测C2.png", dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {FIGURES_DIR}/7_B类预测C2.png")
    plt.close()
    
    return results

def main():
    """
    主函数
    """
    print("=" * 60)
    print("冗余性检验")
    print("=" * 60)
    
    df = get_clean_data()
    
    results_c1 = predict_c1_from_a(df)
    results_c2 = predict_c2_from_b(df)
    
    # 汇总结果
    summary = pd.DataFrame([results_c1, results_c2])
    print("\n=== 冗余性检验汇总 ===")
    print(summary.to_string(index=False))
    
    summary.to_csv(f"{RESULTS_DIR}/冗余性检验汇总.csv", index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("冗余性检验完成！")
    print(f"结果已保存至: {RESULTS_DIR}")
    print(f"图表已保存至: {FIGURES_DIR}")
    print("\n解读提示:")
    print("- R²越接近1，说明C1/C2越能被A/B类要素线性解释，冗余性越高")
    print("- 准确率越高，说明C1/C2的取值越能被A/B类准确预测")
    print("=" * 60)

if __name__ == "__main__":
    main()
