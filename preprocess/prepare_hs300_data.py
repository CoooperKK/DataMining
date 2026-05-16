# preprocess_returns.py
"""
将原始数据预处理为收益率格式
输出：直接可用于聚类的收益率数据（保持原始涨跌幅，不做价格指数转换）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("数据预处理（收益率格式）")
print("=" * 70)

# ==========================================
# 1. 加载国内数据
# ==========================================
print("\n[1] 加载国内数据...")
df = pd.read_csv('./dataset/stock_data.csv')

print(f"原始数据形状: {df.shape}")

# ==========================================
# 2. 解析数据结构
# ==========================================
print("\n[2] 解析数据结构...")

# 第0行：行业信息（第2列开始）
industry_row = df.iloc[0, 2:].to_dict()
print(f"行业信息示例: {list(industry_row.items())[:3]}...")

# 股票名称（列名，从第2列开始）
stock_names = df.columns[2:].tolist()
print(f"股票数量: {len(stock_names)}")

# 提取日期（第1行开始，第1列）
date_col = df.iloc[1:, 1].tolist()
print(f"日期范围: {date_col[0]} 到 {date_col[-1]}")
print(f"交易日数量: {len(date_col)}")

# 收益率数据（第1行开始，第2列开始）
returns_data = df.iloc[1:, 2:].astype(float)
print(f"收益率数据形状: {returns_data.shape}")

# ==========================================
# 3. 数据清洗
# ==========================================
print("\n[3] 数据清洗...")

# 检查缺失值
missing_ratio = returns_data.isna().mean()
print(f"缺失值比例: 最高={missing_ratio.max():.4f}, 平均={missing_ratio.mean():.4f}")

# 填充缺失值
returns_data = returns_data.ffill().bfill().fillna(0)

# 检查异常值
min_return = returns_data.min().min()
max_return = returns_data.max().max()
print(f"收益率范围: {min_return:.4f} 到 {max_return:.4f}")

# ==========================================
# 4. 筛选时间范围（可选）
# ==========================================
print("\n[4] 筛选时间范围...")

USE_RECENT_YEAR = False  # 设为 True 使用最近一年

if USE_RECENT_YEAR:
    time_window = 250
    returns_data = returns_data.iloc[-time_window:, :]
    print(f"使用最近 {time_window} 个交易日")
    date_col = date_col[-time_window:]
else:
    print(f"使用全部 {returns_data.shape[0]} 个交易日")

# ==========================================
# 5. 清洗数据（删除波动过小的股票）
# ==========================================
print("\n[5] 清洗数据...")

# 计算每只股票的方差（波动性）
variances = returns_data.var()
print(f"方差范围: {variances.min():.6f} 到 {variances.max():.6f}")

# 删除方差过小的股票（波动率 < 0.0001，即几乎不动的股票）
min_variance = 0.0001
valid_stocks = variances[variances >= min_variance].index
returns_clean = returns_data[valid_stocks]

print(f"删除低波动股票后: {len(valid_stocks)} 只")

# 可选：限制股票数量
MAX_STOCKS = 200
if MAX_STOCKS and returns_clean.shape[1] > MAX_STOCKS:
    returns_clean = returns_clean.iloc[:, :MAX_STOCKS]
    print(f"限制为前 {MAX_STOCKS} 只股票")

print(f"最终数据: {returns_clean.shape[0]} 天 × {returns_clean.shape[1]} 只股票")

# ==========================================
# 6. 标准化
# ==========================================
print("\n[6] 标准化...")

from sklearn.preprocessing import StandardScaler

# 转置为 (股票数, 天数) 格式
X = returns_clean.T.values

# Z-score 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化后数据形状: {X_scaled.shape[0]} 只股票 × {X_scaled.shape[1]} 个特征")

# ==========================================
# 7. 保存为聚类脚本兼容的格式
# ==========================================
print("\n[7] 保存数据...")

# 构建输出（需要包含行业信息行）
output_data = []

# 第0行：行业信息
industry_row_output = {}
for stock in returns_clean.columns:
    industry_row_output[stock] = industry_row.get(stock, '未知')
output_data.append(industry_row_output)

# 后续行：收益率数据
for i in range(len(returns_clean)):
    row = returns_clean.iloc[i].to_dict()
    output_data.append(row)

# 转换为DataFrame
final_df = pd.DataFrame(output_data)

# 添加日期列
final_df.insert(0, 'date', ['industry'] + date_col)
final_df.insert(0, 'index', range(len(final_df)))

# 保存
output_path = './dataset/stock_data_returns.csv'
final_df.to_csv(output_path, index=False)
print(f"\n✅ 数据已保存: {output_path}")

# ==========================================
# 8. 验证
# ==========================================
print("\n[8] 验证...")
print(f"文件形状: {final_df.shape[0]} 行 × {final_df.shape[1]} 列")
print(f"股票数量: {returns_clean.shape[1]}")
print(f"交易日数量: {returns_clean.shape[0]}")
print(f"收益率范围: {returns_clean.min().min():.4f} - {returns_clean.max().max():.4f}")

print("\n前3行前5列:")
print(final_df.iloc[:3, :5])

print("\n✅ 预处理完成！")
print("下一步: 修改 similarity_cmp_returns.py 中的数据路径")
print("   raw_df = pd.read_csv('./dataset/stock_data_returns.csv')")