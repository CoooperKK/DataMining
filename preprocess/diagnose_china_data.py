# diagnose_china_data.py
"""诊断国内股票数据结构"""

import pandas as pd
import numpy as np

print("=" * 60)
print("国内数据结构诊断")
print("=" * 60)

# 读取数据
df = pd.read_csv('./dataset/stock_data.csv')

print(f"\n[1] 基本信息")
print(f"形状: {df.shape}")
print(f"列数: {len(df.columns)}")

print(f"\n[2] 前3行前5列")
print(df.iloc[:3, :5])

print(f"\n[3] 第0行内容（行业信息）")
print(df.iloc[0, :5].tolist())

print(f"\n[4] 第1行内容（可能是日期或数据）")
print(df.iloc[1, :5].tolist())

print(f"\n[5] 第2行内容")
print(df.iloc[2, :5].tolist())

print(f"\n[6] 最后5列列名")
print(df.columns[-5:].tolist())

print(f"\n[7] 数据类型统计")
print(df.dtypes.value_counts())

print(f"\n[8] 检查是否有日期列")
print(f"第一列列名: '{df.columns[0]}'")
print(f"第一列前3个值: {df.iloc[:3, 0].tolist()}")