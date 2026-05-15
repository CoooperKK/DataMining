# prepare_sp500_simple.py
"""
简化版 S&P 500 数据预处理
直接输出原始代码兼容的格式
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("预处理 S&P 500 数据集（简化版）")
print("=" * 60)

# ==========================================
# 1. 加载数据
# ==========================================
print("\n[1] 加载数据...")

# 跳过前3行，第3行开始是数据
df = pd.read_csv('dataset/SnP_daily_update.csv', header=None, skiprows=3)

# 读取股票代码（第1行，跳过第0列）
tickers_row = pd.read_csv('dataset/SnP_daily_update.csv', header=None, nrows=1, skiprows=1)
tickers = tickers_row.iloc[0, 1:].tolist()

print(f"股票数量: {len(tickers)}")
print(f"数据形状: {df.shape}")

# ==========================================
# 2. 提取日期和价格数据
# ==========================================
print("\n[2] 提取数据...")

# 日期列
dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')

# 价格数据（第1列开始）
price_data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
price_data.columns = tickers

print(f"价格数据: {price_data.shape}")

# ==========================================
# 3. 计算收益率并筛选最近一年
# ==========================================
print("\n[3] 计算收益率...")

returns = price_data.pct_change()
returns = returns.iloc[1:]  # 删除第一行NaN
returns.index = dates.iloc[1:]

# 筛选最近一年
end_date = returns.index.max()
start_date = end_date - pd.DateOffset(years=1)
returns_year = returns[(returns.index >= start_date) & (returns.index <= end_date)]

print(f"最近一年: {len(returns_year)} 个交易日")

# ==========================================
# 4. 清洗数据
# ==========================================
print("\n[4] 清洗数据...")

# 删除缺失过多的股票
missing_ratio = returns_year.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.2].index
returns_clean = returns_year[valid_stocks]

# 填充缺失值
returns_clean = returns_clean.fillna(0)

# 限制股票数量（避免太慢）
MAX_STOCKS = 300
if len(returns_clean.columns) > MAX_STOCKS:
    returns_clean = returns_clean.iloc[:, :MAX_STOCKS]

print(f"最终: {len(returns_clean.columns)} 只股票 × {len(returns_clean)} 天")

# ==========================================
# 5. 保存为原始代码兼容格式
# ==========================================
print("\n[5] 保存数据...")

# 转置并添加行业行
returns_T = returns_clean.T

# 构建输出
output_data = []

# 第0行：行业信息（全部填 'stock'）
industry_row = {col: 'stock' for col in returns_T.index}
output_data.append(industry_row)

# 后续行：收益率数据
for date in returns_clean.index:
    row = returns_clean.loc[date].to_dict()
    output_data.append(row)

# 转换为DataFrame
final_df = pd.DataFrame(output_data)

# 添加日期列（第一行是行业，所以日期列第一行填 'industry'）
final_df.insert(0, 'date', ['industry'] + returns_clean.index.strftime('%Y-%m-%d').tolist())
final_df.insert(0, 'index', range(len(final_df)))

# 保存
final_df.to_csv('./dataset/stock_data_sp500.csv', index=False)
print(f"\n✅ 已保存: stock_data_sp500.csv")
print(f"   形状: {final_df.shape[0]} 行 × {final_df.shape[1]} 列")

# 验证
print("\n[6] 验证...")
print(f"前2行前4列:")
print(final_df.iloc[:2, :4])