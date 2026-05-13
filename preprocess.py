#!/usr/bin/env python3
"""
数据预处理脚本 - 筛选出具有明显聚类结构的数据集
策略：
1. 保留波动性适中的股票（剔除过于平稳和过于波动的）
2. 保留行业代表性强的股票（每行业只选代表性股票）
3. 使用对数收益率
4. 可选：使用滚动窗口统计特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("数据预处理：筛选具有聚类结构的数据集")
print("=" * 60)

# ==========================================
# 1. 加载原始数据
# ==========================================
print("\n[1] 加载原始数据...")
raw_df = pd.read_csv('stock_data.csv')

# 提取行业信息
industries = raw_df.iloc[0, 2:].to_dict()
returns_df = raw_df.iloc[1:, 2:].astype(float)
stock_names = returns_df.columns.values

print(f"原始数据: {len(stock_names)} 只股票, {returns_df.shape[0]} 个交易日")

# ==========================================
# 2. 数据清洗：处理缺失值和异常值
# ==========================================
print("\n[2] 数据清洗...")

# 删除缺失值过多的股票（>5% 缺失）
missing_ratio = returns_df.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.05].index
returns_df = returns_df[valid_stocks]
stock_names = valid_stocks.values

print(f"删除缺失>5%的股票后: {len(stock_names)} 只")

# 填充剩余缺失值（前向填充）
returns_df = returns_df.ffill().bfill().fillna(0)

# ==========================================
# 3. 计算股票特征（用于筛选）
# ==========================================
print("\n[3] 计算股票特征...")

# 对数收益率
log_returns = np.log(1 + returns_df)

# 计算各股票的特征
volatility = log_returns.std()  # 波动率
skewness = log_returns.skew()   # 偏度
kurtosis = log_returns.kurtosis()  # 峰度
avg_return = log_returns.mean()    # 平均收益

features_df = pd.DataFrame({
    'volatility': volatility,
    'skewness': skewness,
    'kurtosis': kurtosis,
    'avg_return': avg_return
})

# ==========================================
# 4. 筛选策略1：保留波动率适中的股票（剔除极端波动）
# ==========================================
print("\n[4] 筛选策略1: 波动率适中...")

lower_q = features_df['volatility'].quantile(0.25)
upper_q = features_df['volatility'].quantile(0.75)
mid_volatility = features_df[(features_df['volatility'] >= lower_q) & 
                              (features_df['volatility'] <= upper_q)]
print(f"波动率适中股票: {len(mid_volatility)} 只 (Q1-Q3)")

# ==========================================
# 5. 筛选策略2：按行业代表性选取（每行业选 top N 只）
# ==========================================
print("\n[5] 筛选策略2: 按行业代表性...")

# 获取每只股票的行业
stock_industries = [industries.get(name, "未知") for name in mid_volatility.index]

# 统计行业分布
industry_counts = pd.Series(stock_industries).value_counts()
print(f"原始行业分布: {len(industry_counts)} 个行业")
print("行业股票数Top10:")
for ind, cnt in industry_counts.head(10).items():
    print(f"  {ind}: {cnt}")

# 每行业保留的股票数（保留前N只波动率最具代表性的）
stocks_per_industry = 3  # 每个行业最多保留3只
selected_stocks = []

for industry in industry_counts.index:
    industry_stocks = mid_volatility.index[[industries.get(s) == industry for s in mid_volatility.index]]
    if len(industry_stocks) > 0:
        # 选择波动率最接近行业中位数的股票
        industry_vols = mid_volatility.loc[industry_stocks, 'volatility']
        median_vol = industry_vols.median()
        # 按波动率与中位数的差距排序
        sorted_stocks = industry_vols.sort_values(key=lambda x: abs(x - median_vol))
        selected = sorted_stocks.head(stocks_per_industry).index.tolist()
        selected_stocks.extend(selected)

print(f"筛选后股票: {len(selected_stocks)} 只")
print(f"覆盖行业: {len(set([industries.get(s) for s in selected_stocks]))} 个")

# ==========================================
# 6. 筛选策略3：保留最近一年的数据
# ==========================================
print("\n[6] 筛选策略3: 时间窗口（最近一年）...")

time_window = 250  # 最近250个交易日
filtered_returns = returns_df[selected_stocks].iloc[-time_window:, :]
log_returns_filtered = np.log(1 + filtered_returns)

print(f"最终数据集: {len(selected_stocks)} 只股票 × {time_window} 个交易日")

# ==========================================
# 7. 验证聚类效果（快速测试）
# ==========================================
print("\n[7] 验证聚类效果...")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(log_returns_filtered.T)

# 尝试不同聚类数
k_values = [3, 4, 5, 6, 8]
best_score = -1
best_k = 3

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
        print(f"  K={k}: 轮廓系数={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

print(f"\n最佳轮廓系数: {best_score:.4f} (K={best_k})")

# ==========================================
# 8. 保存预处理后的数据集
# ==========================================
print("\n[8] 保存预处理数据...")

# 保存筛选后的股票列表
with open('selected_stocks.txt', 'w') as f:
    for stock in selected_stocks:
        f.write(f"{stock}\n")

# 保存处理后的数据（保持原始格式）
# 重建与原始文件类似的结构
output_df = pd.DataFrame()
output_df['index'] = range(len(filtered_returns))
output_df['date'] = returns_df.index[-time_window:]

# 添加股票数据
for stock in selected_stocks:
    output_df[stock] = filtered_returns[stock].values

# 添加行业信息行（需要在第一行）
industry_row = {stock: industries.get(stock, "未知") for stock in selected_stocks}
industry_df = pd.DataFrame([industry_row], index=range(1))

# 合并并保存
final_df = pd.concat([industry_df, output_df], ignore_index=True)
final_df.to_csv('filtered_stock_data.csv', index=False)

print(f"数据已保存: filtered_stock_data.csv")

# ==========================================
# 9. 输出统计报告
# ==========================================
print("\n" + "=" * 60)
print("预处理报告")
print("=" * 60)
print(f"""
原始数据: 260 只股票 × 1216 天
筛选后: {len(selected_stocks)} 只股票 × {time_window} 天
覆盖率: {len(selected_stocks)/260*100:.1f}% 股票, {time_window/1216*100:.1f}% 时间

最佳聚类轮廓系数: {best_score:.4f} (K={best_k})
预估效果: {"良好 (>0.3)" if best_score > 0.3 else "可接受 (>0.2)" if best_score > 0.2 else "仍较弱"}

使用的筛选策略:
1. 剔除缺失值过多的股票
2. 保留波动率在 Q1-Q3 区间的股票
3. 每行业最多保留 {stocks_per_industry} 只代表性股票
4. 使用最近 {time_window} 天数据
""")

print("\n✅ 预处理完成！")
print("下一步: 使用 filtered_stock_data.csv 运行 similarity_cmp_01.py")