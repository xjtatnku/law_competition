"""
法律实证分析报告生成器
将统计结果转化为法律语言的论述
"""
import pandas as pd
import numpy as np
from datetime import datetime
from config import RESULTS_DIR, VAR_LABELS
from legal_interpretation import *

def load_results():
    """加载所有结果文件"""
    results = {}
    results['取值分布'] = pd.read_csv(f"{RESULTS_DIR}/取值分布统计.csv", encoding='utf-8-sig')
    results['条件概率'] = pd.read_csv(f"{RESULTS_DIR}/条件概率统计.csv", encoding='utf-8-sig')
    results['C1与A类相关'] = pd.read_csv(f"{RESULTS_DIR}/C1与A类相关系数.csv", encoding='utf-8-sig', index_col=0)
    results['C2与B类相关'] = pd.read_csv(f"{RESULTS_DIR}/C2与B类相关系数.csv", encoding='utf-8-sig', index_col=0)
    results['冗余性检验'] = pd.read_csv(f"{RESULTS_DIR}/冗余性检验汇总.csv", encoding='utf-8-sig')
    results['逻辑回归_模型4'] = pd.read_csv(f"{RESULTS_DIR}/逻辑回归_模型4_全部要素.csv", encoding='utf-8-sig')
    results['决策树_模型4_特征'] = pd.read_csv(f"{RESULTS_DIR}/决策树_模型4_特征重要性.csv", encoding='utf-8-sig')
    results['决策树对比'] = pd.read_csv(f"{RESULTS_DIR}/决策树_模型对比.csv", encoding='utf-8-sig')
    results['模型比较'] = pd.read_csv(f"{RESULTS_DIR}/模型比较汇总.csv", encoding='utf-8-sig')
    return results

def main():
    """生成完整的法律实证分析报告"""
    print("正在生成法律实证分析报告...")
    
    # 加载结果
    results = load_results()
    
    # 生成报告各部分
    report = []
    report.append(generate_header())
    report.append(generate_background())
    report.append(generate_descriptive_analysis(results))
    report.append(generate_redundancy_analysis(results))
    report.append(generate_regression_analysis(results))
    report.append(generate_tree_analysis(results))
    report.append(generate_model_comparison(results))
    report.append(generate_conclusion(results))
    report.append(generate_recommendations())
    
    # 保存报告
    full_report = "\n".join(report)
    output_file = f"{RESULTS_DIR}/法律实证分析报告.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"\n报告已生成: {output_file}")
    print(f"报告总字数: {len(full_report)} 字符")
    
    # 也输出到控制台
    print("\n" + "="*80)
    print("报告预览（前2000字符）：")
    print("="*80)
    print(full_report[:2000])
    print("\n... [完整报告请查看文件] ...")

if __name__ == "__main__":
    main()
