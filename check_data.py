"""
数据检查脚本：帮助用户确认Excel文件的列名和数据格式
运行此脚本以检查数据是否符合要求
"""
import pandas as pd
import os
from config import DATA_FILE, ALL_FEATURES, TARGET, COLUMNS

def check_excel_structure():
    """
    检查Excel文件结构
    """
    print("="*60)
    print("数据检查工具")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(DATA_FILE):
        print(f"\n[X] 错误: 找不到数据文件")
        print(f"   期望路径: {DATA_FILE}")
        print(f"\n请确保Excel文件存在，并且路径正确。")
        return False
    
    print(f"\n[OK] 数据文件存在: {DATA_FILE}")
    
    # 读取Excel
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"[OK] 成功读取Excel文件")
        print(f"  - 总行数: {len(df)}")
        print(f"  - 总列数: {len(df.columns)}")
    except Exception as e:
        print(f"\n[X] 读取Excel文件失败")
        print(f"   错误: {str(e)}")
        return False
    
    # 显示实际列名
    print(f"\n实际Excel列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # 检查期望的列名（Excel中的实际列名）
    expected_excel_cols = [COLUMNS[key] for key in (ALL_FEATURES + [TARGET])]
    print(f"\n期望的列名（在config.py中配置的Excel列名）:")
    for i, col in enumerate(expected_excel_cols, 1):
        print(f"  {i}. {col}")
    
    # 检查匹配情况
    print(f"\n列名匹配检查:")
    missing_cols = []
    for excel_col in expected_excel_cols:
        if excel_col in df.columns:
            print(f"  [OK] {excel_col}")
        else:
            print(f"  [X] {excel_col} (缺失)")
            missing_cols.append(excel_col)
    
    if missing_cols:
        print(f"\n[!] 警告: 有 {len(missing_cols)} 个期望列名在Excel中找不到")
        print(f"   缺失的列: {', '.join(missing_cols)}")
        print(f"\n建议操作:")
        print(f"  1. 检查Excel文件的列名是否正确")
        print(f"  2. 或修改 config.py 中的 COLUMNS 配置，使其与实际列名匹配")
        print(f"\n示例修改（在config.py中）:")
        print(f"  COLUMNS = {{")
        for col in df.columns:
            print(f"      '{col}': '{col}',")
        print(f"  }}")
        return False
    
    print(f"\n[OK] 所有期望列名都存在")
    
    # 检查数据取值
    print(f"\n数据取值检查（应为 -1, 0, 1）:")
    all_valid = True
    for excel_col in expected_excel_cols:
        unique_vals = sorted(df[excel_col].dropna().unique())
        invalid_vals = [v for v in unique_vals if v not in [-1, 0, 1]]
        
        if invalid_vals:
            print(f"  [!] {excel_col}: {unique_vals} (包含非法值: {invalid_vals})")
            all_valid = False
        else:
            print(f"  [OK] {excel_col}: {unique_vals}")
    
    if not all_valid:
        print(f"\n[!] 警告: 某些列包含非 {{-1, 0, 1}} 的值")
        print(f"   这些值在分析时会被当作缺失值处理")
    
    # 检查缺失值
    print(f"\n缺失值统计:")
    missing_counts = df[expected_excel_cols].isnull().sum()
    if missing_counts.sum() == 0:
        print(f"  [OK] 无缺失值")
    else:
        print(f"  缺失值数量:")
        for col, count in missing_counts.items():
            if count > 0:
                pct = count / len(df) * 100
                print(f"    {col}: {count} ({pct:.1f}%)")
    
    # 目标变量分布（使用Excel列名）
    target_excel_name = COLUMNS[TARGET]
    print(f"\n目标变量({target_excel_name})分布:")
    target_counts = df[target_excel_name].value_counts().sort_index()
    for val, count in target_counts.items():
        pct = count / len(df[target_excel_name].dropna()) * 100
        label = "认定劳动关系" if val == 1 else "未认定/无法认定"
        print(f"  {val} ({label}): {count} ({pct:.1f}%)")
    
    # 数据预览
    print(f"\n数据预览（前5行）:")
    print(df[expected_excel_cols].head())
    
    # 总结
    print("\n" + "="*60)
    if not missing_cols and all_valid:
        print("[OK] 数据检查通过！可以开始分析。")
        print("\n运行分析:")
        print("  python run_all_analysis.py")
    else:
        print("[!] 数据检查发现问题，请根据上述提示修正。")
    print("="*60)
    
    return not missing_cols and all_valid

if __name__ == "__main__":
    try:
        check_excel_structure()
    except Exception as e:
        print(f"\n[X] 检查过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
