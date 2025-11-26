"""
主运行脚本：依次执行所有分析模块
"""
import sys
import time

def run_module(module_name, description):
    """
    运行单个分析模块
    """
    print("\n" + "="*80)
    print(f"开始执行: {description}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        # 动态导入并运行
        module = __import__(module_name)
        module.main()
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} 完成 (耗时: {elapsed:.2f}秒)")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} 出错 (耗时: {elapsed:.2f}秒)")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数：执行完整的分析流程
    """
    print("="*80)
    print(" " * 20 + "法学实证分析 - 完整流程")
    print("="*80)
    print("\n本程序将依次执行以下分析:")
    print("  1. 描述性统计分析")
    print("  2. 相关性分析 (C1 vs A类, C2 vs B类)")
    print("  3. 冗余性检验 (A类能否预测C1, B类能否预测C2)")
    print("  4. 逻辑回归分析 (量化边际影响)")
    print("  5. 决策树分析 (规则提取与要素重要性)")
    print("  6. 模型比较 (嵌套模型检验C1/C2的增量贡献)")
    print("\n" + "="*80)
    
    input("\n按Enter键开始分析...")
    
    overall_start = time.time()
    results = {}
    
    # 定义分析模块
    modules = [
        ("descriptive_stats", "1. 描述性统计分析"),
        ("correlation_analysis", "2. 相关性分析"),
        ("redundancy_test", "3. 冗余性检验"),
        ("logistic_regression", "4. 逻辑回归分析"),
        ("decision_tree_analysis", "5. 决策树分析"),
        ("model_comparison", "6. 模型比较")
    ]
    
    # 依次执行
    for module_name, description in modules:
        success = run_module(module_name, description)
        results[description] = success
        
        if not success:
            print(f"\n警告: {description} 执行失败，但继续执行后续分析...")
    
    # 总结
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*80)
    print(" " * 30 + "分析完成汇总")
    print("="*80)
    print(f"\n总耗时: {overall_elapsed:.2f}秒\n")
    
    print("各模块执行状态:")
    for desc, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {status}  {desc}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 所有分析已成功完成！")
        print(f"\n结果文件位置:")
        print(f"  - 统计结果: outputs/results/")
        print(f"  - 图表: outputs/figures/")
    else:
        print("\n⚠ 部分分析执行失败，请检查错误信息。")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
