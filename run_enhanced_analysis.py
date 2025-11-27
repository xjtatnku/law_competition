"""
运行所有增强版分析脚本
"""
import os
import sys
import subprocess
import time

def run_script(script_name):
    """
    运行单个脚本
    """
    print(f"\n{'='*60}")
    print(f"正在运行: {script_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # 使用当前Python解释器运行脚本
        result = subprocess.run([sys.executable, script_name], check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'-'*60}")
        print(f"{script_name} 运行成功! (耗时: {duration:.2f}秒)")
        print(f"{'-'*60}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'!'*60}")
        print(f"{script_name} 运行失败!")
        print(f"错误信息: {e}")
        print(f"{'!'*60}")
        return False

def main():
    """
    主函数
    """
    print("开始运行增强版实证分析流程...")
    print(f"工作目录: {os.getcwd()}")
    
    scripts = [
        "check_data.py",                    # 1. 数据检查（原有）
        "enhanced_correlation_analysis.py", # 2. 增强版相关性分析（四种系数）
        "decision_tree_analysis.py",        # 3. 决策树分析（已优化参数）
        "enhanced_model_comparison.py"      # 4. 增强版模型比较（详细指标）
    ]
    
    success_count = 0
    
    for script in scripts:
        if os.path.exists(script):
            if run_script(script):
                success_count += 1
        else:
            print(f"错误: 找不到脚本 {script}")
    
    print(f"\n{'='*60}")
    print(f"所有分析完成! 成功: {success_count}/{len(scripts)}")
    print(f"结果保存在: outputs/results/")
    print(f"图表保存在: outputs/figures/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
