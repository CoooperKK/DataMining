#!/usr/bin/env python3
"""
数据预处理脚本 V2 - 筛选具有明显涨跌波动的股票
核心策略：
1. 只保留波动率高的股票（剔除低波动/死股票）
2. 每行业保留波动率最大的代表股票
3. 时间窗口可调（默认最近250天）
4. 可选：按不同时间段评估稳定性
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("数据预处理 V2：筛选高波动、高辨识度股票")
print("=" * 60)

# ==========================================
# 可配置参数
# ==========================================
TIME_WINDOW = 250          # 时间窗口（天），可选: 125(半年), 250(一年), 500(两年)
VOLATILITY_QUANTILE = 0.85  # 保留波动率前(1-QUANTILE)%的股票，默认65%表示保留前35%
STOCKS_PER_INDUSTRY = 3    # 每个行业最多保留几只股票
TEST_TIME_WINDOWS = [125, 250, 500]  # 测试不同时间窗口的稳定性

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
# 2. 数据清洗
# ==========================================
print("\n[2] 数据清洗...")

# 删除缺失值过多的股票（>5% 缺失）
missing_ratio = returns_df.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.05].index
returns_df = returns_df[valid_stocks]
stock_names = valid_stocks.values

print(f"删除缺失>5%的股票后: {len(stock_names)} 只")

# 填充剩余缺失值
returns_df = returns_df.ffill().bfill().fillna(0)

# ==========================================
# 3. 计算股票特征（使用全部数据）
# ==========================================
print("\n[3] 计算股票特征...")

# 对数收益率
log_returns = np.log(1 + returns_df)

# 核心特征：波动率（标准差）—— 越大表示涨跌越明显
volatility = log_returns.std()
# 辅助特征：平均绝对收益 —— 同样反映活跃度
mean_abs_return = log_returns.abs().mean()
# 最大单日波动
max_daily_move = log_returns.abs().max()

features_df = pd.DataFrame({
    'volatility': volatility,
    'mean_abs_return': mean_abs_return,
    'max_daily_move': max_daily_move,
    'skewness': log_returns.skew(),
    'kurtosis': log_returns.kurtosis()
})

# 综合活跃度得分（波动率权重最高）
features_df['activity_score'] = (
    features_df['volatility'] * 0.5 + 
    features_df['mean_abs_return'] * 0.3 + 
    features_df['max_daily_move'] * 0.2
)

# ==========================================
# 4. 筛选策略1：只保留高波动股票（剔除死股票）
# ==========================================
print(f"\n[4] 筛选策略1: 高波动股票 (波动率前 {int((1-VOLATILITY_QUANTILE)*100)}%)...")

vol_threshold = features_df['volatility'].quantile(VOLATILITY_QUANTILE)
high_volatility = features_df[features_df['volatility'] >= vol_threshold]
print(f"高波动股票: {len(high_volatility)} 只")
print(f"波动率阈值: {vol_threshold:.6f}")

# 显示波动率最低的几只（被剔除的）
print(f"被剔除的低波动股票示例（波动率最小5只）:")
low_vol_example = features_df.nsmallest(5, 'volatility')
for stock, vol in low_vol_example['volatility'].items():
    print(f"  {stock}: 波动率={vol:.6f}")

# ==========================================
# 5. 筛选策略2：每行业保留活跃度最高的股票
# ==========================================
print(f"\n[5] 筛选策略2: 每行业保留 {STOCKS_PER_INDUSTRY} 只最活跃股票...")

# 获取每只股票的行业
stock_industries = [industries.get(name, "未知") for name in high_volatility.index]
high_volatility['industry'] = stock_industries

# 按行业分组，每行业选择活跃度最高的股票
selected_stocks = []
industry_stats = {}

for industry, group in high_volatility.groupby('industry'):
    # 按活跃度排序
    sorted_group = group.sort_values('activity_score', ascending=False)
    n_select = min(STOCKS_PER_INDUSTRY, len(group))
    selected = sorted_group.head(n_select).index.tolist()
    selected_stocks.extend(selected)
    industry_stats[industry] = {
        'total': len(group),
        'selected': n_select,
        'avg_volatility': group['volatility'].mean()
    }

print(f"筛选后股票: {len(selected_stocks)} 只")
print(f"覆盖行业: {len(industry_stats)} 个")

# 显示选中的行业和股票数
print("\n各行业选中情况（前10）:")
sorted_industries = sorted(industry_stats.items(), key=lambda x: x[1]['selected'], reverse=True)
for ind, stats in sorted_industries[:10]:
    print(f"  {ind}: {stats['selected']}/{stats['total']} 只, 平均波动率={stats['avg_volatility']:.5f}")

# ==========================================
# 6. 评估不同时间窗口的稳定性
# ==========================================
print("\n[6] 评估不同时间窗口的聚类稳定性...")

window_results = {}
for window in TEST_TIME_WINDOWS:
    if window <= returns_df.shape[0]:
        window_returns = returns_df[selected_stocks].iloc[-window:, :]
        log_returns_window = np.log(1 + window_returns)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(log_returns_window.T)
        
        # K=3 快速测试
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        window_results[window] = score
        
        # 显示各窗口的波动率
        vol_mean = log_returns_window.std().mean()
        print(f"  {window}天窗口: 轮廓系数={score:.4f}, 平均波动率={vol_mean:.5f}")

# 选出最佳时间窗口
best_window = max(window_results, key=window_results.get)
print(f"\n最佳时间窗口: {best_window}天 (轮廓系数={window_results[best_window]:.4f})")

# ==========================================
# 7. 使用最佳窗口保存数据
# ==========================================
print(f"\n[7] 使用最佳时间窗口 ({best_window}天) 保存数据...")

final_window = best_window
filtered_returns = returns_df[selected_stocks].iloc[-final_window:, :]
log_returns_filtered = np.log(1 + filtered_returns)

print(f"最终数据集: {len(selected_stocks)} 只股票 × {final_window} 个交易日")

# 验证高波动性
final_volatility = log_returns_filtered.std()
low_vol_count = (final_volatility < 0.005).sum()
print(f"低波动股票 (<0.005): {low_vol_count}/{len(selected_stocks)} 只 (目标: 尽量少)")

# ==========================================
# 8. 验证聚类效果
# ==========================================
print("\n[8] 验证聚类效果...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(log_returns_filtered.T)

k_values = [3, 4, 5, 6, 8]
best_score = -1
best_k = 3

print("聚类测试:")
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
# 9. 保存预处理后的数据集
# ==========================================
print("\n[9] 保存预处理数据...")

# 保存筛选后的股票列表
with open('selected_stocks_high_vol.txt', 'w') as f:
    for stock in selected_stocks:
        f.write(f"{stock}\n")

# 保存处理后的数据
output_df = pd.DataFrame()
output_df['index'] = range(len(filtered_returns))
output_df['date'] = returns_df.index[-final_window:]

for stock in selected_stocks:
    output_df[stock] = filtered_returns[stock].values

# 添加行业信息行
industry_row = {stock: industries.get(stock, "未知") for stock in selected_stocks}
industry_df = pd.DataFrame([industry_row], index=range(1))

final_df = pd.concat([industry_df, output_df], ignore_index=True)
final_df.to_csv('filtered_stock_data_high_vol.csv', index=False)

print(f"数据已保存: filtered_stock_data_high_vol.csv")

# ==========================================
# 10. 输出统计报告
# ==========================================
print("\n" + "=" * 70)
print("预处理报告 V2")
print("=" * 70)
print(f"""
原始数据: 260 只股票 × {returns_df.shape[0]} 天
筛选后: {len(selected_stocks)} 只股票 × {final_window} 天
筛选比例: {len(selected_stocks)/260*100:.1f}% 股票, {final_window/returns_df.shape[0]*100:.1f}% 时间

核心指标:
- 高波动标准: 波动率前 {int((1-VOLATILITY_QUANTILE)*100)}% (阈值={vol_threshold:.5f})
- 每行业最多 {STOCKS_PER_INDUSTRY} 只股票
- 低波动股票残留: {low_vol_count}/{len(selected_stocks)} 只

聚类评估:
- 最佳轮廓系数: {best_score:.4f} (K={best_k})
- 最佳时间窗口: {best_window} 天

对比之前的糟糕数据:
- 旧筛选: 101/108 只股票波动率 ≤ 0.001 (93.5% 死股票)
- 新筛选: 低波动股票大幅减少
""")

print("\n✅ 预处理完成！")
print("下一步: 使用 filtered_stock_data_high_vol.csv 运行 similarity_cmp_01.py")