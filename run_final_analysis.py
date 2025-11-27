"""
运行最终版完整分析流程
"""
import os
import sys
import subprocess
import time

def run_script(script_name):
    """运行单个脚本"""
    print(f"\n{'='*60}")
    print(f"正在运行: {script_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n[成功] {script_name} (耗时: {time.time() - start_time:.2f}秒)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[失败] {script_name} - {e}")
        return False

def main():
    print("开始运行最终版实证分析流程...")
    print(f"工作目录: {os.getcwd()}")
    
    # 清理旧图片? (可选，这里暂不删除以免误删)
    
    scripts = [
        "step1_descriptive.py",      # 1. 描述性统计
        "step2_correlation.py",      # 2. 相关性分析
        "step3_regression.py",       # 3. 回归分析 (线性+逻辑)
        "step4_decision_tree.py",    # 4. 决策树
        "step5_model_evaluation.py"  # 5. 模型评估
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
