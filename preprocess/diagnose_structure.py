# diagnose_structure.py
"""诊断 S&P 500 CSV 文件的结构"""

import pandas as pd
import numpy as np

print("=" * 70)
print("S&P 500 数据集结构诊断")
print("=" * 70)

# 读取前10行用于分析
df_sample = pd.read_csv('dataset/SnP_daily_update.csv', header=None, nrows=10)
print(f"\n[1] 文件基本信息")
print(f"    总列数: {df_sample.shape[1]}")
print(f"    前10行形状: {df_sample.shape}")

print(f"\n[2] 前5行每行前10个单元格的内容")
for row_idx in range(min(5, len(df_sample))):
    row_data = df_sample.iloc[row_idx, :10].tolist()
    print(f"    行{row_idx}: {row_data}")

print(f"\n[3] 第0行（第一行）的内容特征")
row0 = df_sample.iloc[0, :]
print(f"    数据类型: {row0.dtype}")
print(f"    前5个值: {row0.iloc[:5].tolist()}")
print(f"    第0个值（第一列）: '{row0.iloc[0]}'")
print(f"    第1个值: '{row0.iloc[1]}'")
print(f"    第2个值: '{row0.iloc[2]}'")

print(f"\n[4] 第1行（第二行）的内容特征")
row1 = df_sample.iloc[1, :]
print(f"    数据类型: {row1.dtype}")
print(f"    前5个值: {row1.iloc[:5].tolist()}")
print(f"    第0个值: '{row1.iloc[0]}'")
print(f"    第1个值: '{row1.iloc[1]}'")
print(f"    第2个值: '{row1.iloc[2]}'")

print(f"\n[5] 第2行（第三行）的内容特征")
row2 = df_sample.iloc[2, :]
print(f"    数据类型: {row2.dtype}")
print(f"    前5个值: {row2.iloc[:5].tolist()}")
print(f"    第0个值: '{row2.iloc[0]}'")
print(f"    第1个值: '{row2.iloc[1]}'")

print(f"\n[6] 检查特定列的范围")
# 检查是否有列名包含 'Close'
close_like_cols = []
for col_idx in range(min(20, df_sample.shape[1])):
    val = str(df_sample.iloc[1, col_idx])
    if 'close' in val.lower():
        close_like_cols.append((col_idx, val))
print(f"    前20列中类似'Close'的列: {close_like_cols}")

print(f"\n[7] 检查是否有数值列")
# 检查第2行开始是否有数值
numeric_check = []
for col_idx in range(min(10, df_sample.shape[1])):
    val = df_sample.iloc[2, col_idx]
    try:
        float(val)
        numeric_check.append((col_idx, True, val))
    except:
        numeric_check.append((col_idx, False, str(val)[:20]))
print(f"    第2行前10列是否为数值: {numeric_check}")

print(f"\n[8] 数据可能的组织方式推断")
print("    根据观察：")
if row0.iloc[0] == 'Price' and row1.iloc[0] == 'Ticker':
    print("    ✓ 第0行第0列是 'Price'，第1行第0列是 'Ticker'")
    print("    → 数据可能是：第一列是指标类型，第一行是价格类型，第二行是股票代码")
elif row0.iloc[0] == 'Ticker' and row1.iloc[0] == 'Date':
    print("    ✓ 第0行第0列是 'Ticker'，第1行第0列是 'Date'")
    print("    → 数据可能是：第一列是行标签，后面是数据")
else:
    print(f"    ? 未识别出标准格式")

print(f"\n[9] 建议")
print("    请将以上输出复制给我，我将根据实际结构编写正确的预处理脚本")