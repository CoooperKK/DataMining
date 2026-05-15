# prepare_sp500_paper.py
"""
完全按照论文方法预处理 S&P 500 数据集
论文参数：
- 时间范围：2013-02-08 到 2018-02-08（5年）
- 数据：Close 价格
- 价格指数：起点 = 100
- 平滑：SMA-10
- 删除：起始日期不一致的公司（35家）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("论文复现：S&P 500 数据预处理")
print("(2013-02-08 到 2018-02-08, 价格指数, SMA-10)")
print("=" * 70)

# ==========================================
# 1. 加载原始数据
# ==========================================
print("\n[1] 加载原始数据...")

# 数据文件结构：
# 行0: 指标类型 (Price, Close, Volume...)
# 行1: 股票代码 (A, AAPL, ABBV...)
# 行2: 行标签 (Date, NaN, NaN...)
# 行3+: 实际数据

# 读取所有数据
df_full = pd.read_csv('dataset/SnP_daily_update.csv', header=None)

# 解析结构
price_types = df_full.iloc[0, :].tolist()     # 第一行：指标类型
tickers = df_full.iloc[1, :].tolist()         # 第二行：股票代码
row_labels = df_full.iloc[2, :].tolist()      # 第三行：行标签

print(f"原始数据形状: {df_full.shape}")
print(f"股票代码数量: {len([t for t in tickers if str(t) != 'Ticker'])}")

# ==========================================
# 2. 提取 Close 价格数据
# ==========================================
print("\n[2] 提取 Close 价格数据...")

# 找到所有 Close 价格的列（论文只使用 Close 价格）
close_indices = []
close_tickers = []

for i, pt in enumerate(price_types):
    if pd.notna(pt) and str(pt).strip() == 'Close':
        close_indices.append(i)
        close_tickers.append(tickers[i])

print(f"找到 Close 价格列: {len(close_indices)} 个")

# 提取 Close 价格数据（从第4行开始，索引3）
close_data_raw = df_full.iloc[3:, close_indices].copy()
close_data_raw = close_data_raw.apply(pd.to_numeric, errors='coerce')
close_data_raw.columns = close_tickers

# 提取日期
date_col = df_full.iloc[3:, 0].copy()
dates = pd.to_datetime(date_col, errors='coerce')

# 设置索引
close_data_raw.index = dates
close_data_raw.index.name = 'Date'

print(f"Close 价格数据形状: {close_data_raw.shape}")
print(f"日期范围: {close_data_raw.index.min()} 到 {close_data_raw.index.max()}")

# ==========================================
# 3. 筛选时间范围：2013-02-08 到 2018-02-08
# ==========================================
print("\n[3] 筛选时间范围 (2013-02-08 到 2018-02-08)...")

start_date = pd.Timestamp('2013-02-08')
end_date = pd.Timestamp('2018-02-08')

price_filtered = close_data_raw[(close_data_raw.index >= start_date) & 
                                 (close_data_raw.index <= end_date)]

print(f"筛选后数据形状: {price_filtered.shape}")
print(f"实际日期范围: {price_filtered.index.min()} 到 {price_filtered.index.max()}")
print(f"交易日数量: {len(price_filtered)}")

# ==========================================
# 4. 删除起始日期不一致的公司（论文中删除了35家）
# ==========================================
print("\n[4] 删除起始日期不一致的公司...")

# 找出每只股票第一个非NaN的日期
first_valid_date = {}
for col in price_filtered.columns:
    first_valid = price_filtered[col].first_valid_index()
    first_valid_date[col] = first_valid

# 找出最常见的起始日期
most_common_date = pd.Series(first_valid_date).mode()[0]
print(f"最常见的起始日期: {most_common_date}")

# 保留起始日期为 most_common_date 的股票
valid_stocks = [col for col, fd in first_valid_date.items() if fd == most_common_date]
price_clean = price_filtered[valid_stocks]

print(f"删除起始日期不同的股票后: {len(valid_stocks)} 只")
print(f"被删除的股票数: {len(price_filtered.columns) - len(valid_stocks)}")

# ==========================================
# 5. 转换为价格指数（起点 = 100）
# ==========================================
print("\n[5] 转换为价格指数 (起点 = 100)...")

# 论文方法：将第一天的价格调整为 100
price_index = pd.DataFrame(index=price_clean.index, columns=price_clean.columns)

for col in price_clean.columns:
    first_price = price_clean[col].dropna().iloc[0]
    price_index[col] = price_clean[col] / first_price * 100

print(f"价格指数范围: {price_index.min().min():.2f} - {price_index.max().max():.2f}")

# ==========================================
# 6. SMA-10 平滑处理
# ==========================================
print("\n[6] SMA-10 平滑处理...")

window_size = 10  # 论文确定的最优窗口
price_smoothed = price_index.rolling(window=window_size, min_periods=1).mean()
price_smoothed = price_smoothed.bfill().ffill()  # 填充边缘缺失

print(f"平滑后数据形状: {price_smoothed.shape}")

# ==========================================
# 7. 处理缺失值
# ==========================================
print("\n[7] 处理缺失值...")

# 删除缺失过多的股票（>10%）
missing_ratio = price_smoothed.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.1].index
price_final = price_smoothed[valid_stocks]

print(f"删除缺失>10%的股票后: {len(valid_stocks)} 只")

# 填充剩余缺失值（前向+后向填充）
price_final = price_final.ffill().bfill()

# 检查是否还有缺失
if price_final.isna().any().any():
    price_final = price_final.fillna(100)  # 用起点值填充

print(f"最终数据: {price_final.shape[0]} 天 × {price_final.shape[1]} 只股票")

# ==========================================
# 8. 限制股票数量（可选，论文有约470只）
# ==========================================
print("\n[8] 最终数据准备...")

# 论文实际使用了约470只股票，这里可以全部保留
# 如果需要限制数量以加快速度，取消下面的注释
# MAX_STOCKS = 200
# if price_final.shape[1] > MAX_STOCKS:
#     price_final = price_final.iloc[:, :MAX_STOCKS]
#     print(f"限制为前 {MAX_STOCKS} 只股票")

print(f"最终: {price_final.shape[1]} 只股票 × {price_final.shape[0]} 天")

# ==========================================
# 9. 保存为原始代码兼容的格式
# ==========================================
print("\n[9] 保存数据...")

# 转置：行=股票，列=交易日
price_T = price_final.T

# 构建输出（需要包含行业信息行）
output_data = []

# 第0行：行业信息（论文有行业数据，这里用 'stock' 占位）
industry_row = {stock: 'stock' for stock in price_T.index}
output_data.append(industry_row)

# 后续行：价格指数数据
for date in price_final.index:
    row = price_final.loc[date].to_dict()
    output_data.append(row)

# 转换为DataFrame
final_df = pd.DataFrame(output_data)

# 添加日期列
date_strs = price_final.index.strftime('%Y-%m-%d').tolist()
final_df.insert(0, 'date', ['industry'] + date_strs)
final_df.insert(0, 'index', range(len(final_df)))

# 保存
output_path = 'dataset/stock_data_sp500_paper.csv'
final_df.to_csv(output_path, index=False)
print(f"\n✅ 数据已保存: {output_path}")

# ==========================================
# 10. 验证
# ==========================================
print("\n[10] 验证...")
print(f"文件形状: {final_df.shape[0]} 行 × {final_df.shape[1]} 列")
print(f"股票数量: {price_final.shape[1]}")
print(f"交易日数量: {price_final.shape[0]}")
print(f"日期范围: {price_final.index.min().date()} 到 {price_final.index.max().date()}")

# 显示前2行前5列
print("\n前2行前5列:")
print(final_df.iloc[:2, :5])

# 保存股票列表
with open('dataset/sp500_tickers_paper.txt', 'w') as f:
    for stock in price_final.columns:
        f.write(f"{stock}\n")

print("\n✅ 预处理完成！")
print("下一步: 使用 dataset/stock_data_sp500_paper.csv 运行聚类分析")