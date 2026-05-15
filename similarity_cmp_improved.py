# similarity_cmp_improved.py
"""
参考论文改进的 DTW 聚类
主要改进：
1. 使用价格指数（统一起点100）代替收益率
2. 使用 SMA 平滑
3. 标准 DTW（不过度降维）
4. 添加 Elbow + Silhouette 确定最佳 K
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("改进版时间序列聚类分析（参考论文方法）")
print("=" * 60)

# ==========================================
# 改进1：数据预处理 - 转换为价格指数
# ==========================================
print("\n[1] 数据预处理...")

# 读取原始数据
raw_df = pd.read_csv('./dataset/stock_data.csv')

# 提取收益率数据（原始数据已经收益率）
returns_df = raw_df.iloc[1:, 2:].astype(float)
stock_names = returns_df.columns.values
dates = raw_df.iloc[1:, 1].tolist()

print(f"原始收益率数据: {returns_df.shape}")

# ==========================================
# 改进2：转换为价格指数（统一起点 = 100）
# ==========================================
print("\n[2] 转换为价格指数...")

# 添加：限制收益率范围，避免价格变为负数
# 将收益率限制在 -0.99 到 3.0 之间（避免 log(0) 或负价格）
returns_clipped = returns_df.clip(lower=-0.99, upper=3.0)
print(f"限制收益率范围后: 最小={returns_clipped.min().min():.4f}, 最大={returns_clipped.max().max():.4f}")

# 从收益率计算价格指数：P_t = P_{t-1} * (1 + r_t)，起点 = 100
price_index = pd.DataFrame(index=returns_clipped.index, columns=returns_clipped.columns)
price_index.iloc[0] = 100  # 起点统一为100

for i in range(1, len(returns_clipped)):
    price_index.iloc[i] = price_index.iloc[i-1] * (1 + returns_clipped.iloc[i])

print(f"价格指数形状: {price_index.shape}")
print(f"价格范围: {price_index.min().min():.2f} - {price_index.max().max():.2f}")

# 检查是否有负数或NaN
if (price_index < 0).any().any():
    print("⚠️ 警告：存在负数价格，需要进一步处理")
    price_index = price_index.clip(lower=0.01)
# ==========================================
# 改进3：SMA 平滑处理（参考论文 SMA-10）
# ==========================================
print("\n[3] SMA 平滑处理...")

window_size = 10  # 论文中的最优窗口
price_index_smoothed = price_index.rolling(window=window_size, min_periods=1).mean()
price_index_smoothed = price_index_smoothed.bfill()

print(f"平滑后数据形状: {price_index_smoothed.shape}")

# ==========================================
# 改进4：筛选最近一年数据（250天）
# ==========================================
print("\n[4] 筛选最近一年...")

time_window = 250
price_recent = price_index_smoothed.iloc[-time_window:, :]
print(f"使用最近 {time_window} 个交易日")

# 删除缺失值过多的股票
missing_ratio = price_recent.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.2].index
price_clean = price_recent[valid_stocks]
price_clean = price_clean.ffill().bfill()

print(f"清洗后: {price_clean.shape[1]} 只股票 × {price_clean.shape[0]} 天")

# 限制股票数量（可选）
MAX_STOCKS = 100
if price_clean.shape[1] > MAX_STOCKS:
    price_clean = price_clean.iloc[:, :MAX_STOCKS]
    print(f"限制为前 {MAX_STOCKS} 只股票")

# ==========================================
# 改进5：标准化（对价格指数进行标准化）
# ==========================================
print("\n[5] 标准化...")

# 使用 tslearn 的标准化（适合时间序列）
X_scaled_3d = TimeSeriesScalerMeanVariance().fit_transform(price_clean.T.values)
print(f"3D数据形状: {X_scaled_3d.shape}")

# 转换为 2D 数组（KMeans 需要）
# 方法：将时间步长展平为特征
n_samples, n_timestamps, n_features = X_scaled_3d.shape
X_scaled = X_scaled_3d.reshape(n_samples, n_timestamps * n_features)
print(f"2D数据形状: {X_scaled.shape[0]} 只股票 × {X_scaled.shape[1]} 个特征")


# ==========================================
# 改进6：确定最佳 K 值（Elbow + Silhouette）
# ==========================================
print("\n[6] 确定最佳聚类数...")

k_range = range(2, 11)
silhouette_scores = []
wcss = []

for k in k_range:
    # 使用 Euclidean 快速测试
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    wcss.append(kmeans.inertia_)
    print(f"  K={k}: Silhouette={sil_score:.4f}, WCSS={kmeans.inertia_:.0f}")

# 选择最佳 K（轮廓系数最高）
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n最佳聚类数 K = {best_k} (基于轮廓系数)")

n_clusters = 2

# ==========================================
# 改进7：DTW 聚类（标准 DTW，不过度降维）
# ==========================================
print("\n[7] 运行 DTW 聚类...")
results = {}

# --- Euclidean ---
print("  -> Euclidean K-Means...")
km_ed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_ed = km_ed.fit_predict(X_scaled)
results['Euclidean'] = {
    'labels': labels_ed,
    'silhouette': silhouette_score(X_scaled, labels_ed),
    'db_index': davies_bouldin_score(X_scaled, labels_ed),
    'ch_index': calinski_harabasz_score(X_scaled, labels_ed)
}

# --- DTW（标准 DTW，不减维）---
print("  -> DTW K-Means...")
t0 = time.time()

# 注意：这里不使用 PAA 降维，直接使用原始数据
km_dtw = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    max_iter=10,
    metric_params={
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": 5  # 允许 30 天的滞后（约一个月）
    },
    n_jobs=1,
    random_state=42,
    verbose=False
)

labels_dtw = km_dtw.fit_predict(X_scaled)
dtw_time = time.time() - t0

results['DTW'] = {
    'labels': labels_dtw,
    'silhouette': silhouette_score(X_scaled, labels_dtw),
    'db_index': davies_bouldin_score(X_scaled, labels_dtw),
    'ch_index': calinski_harabasz_score(X_scaled, labels_dtw)
}

print(f"  DTW 耗时: {dtw_time:.2f} 秒")

# ==========================================
# 结果输出
# ==========================================
print("\n" + "=" * 60)
print("聚类评估结果")
print("=" * 60)
print(f"{'方法':<15} {'轮廓系数':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<15}")
print("-" * 60)

for method, scores in results.items():
    print(f"{method:<15} {scores['silhouette']:.4f}       {scores['db_index']:.4f}          {scores['ch_index']:.2f}")

print("=" * 60)

# 判断哪个方法更好
if results['DTW']['silhouette'] > results['Euclidean']['silhouette']:
    print(f"\n✅ DTW 效果更好 (轮廓系数: {results['DTW']['silhouette']:.4f} > {results['Euclidean']['silhouette']:.4f})")
else:
    print(f"\n⚠️ Euclidean 效果更好 (轮廓系数: {results['Euclidean']['silhouette']:.4f} > {results['DTW']['silhouette']:.4f})")

# ==========================================
# 可视化对比
# ==========================================
print("\n[8] 生成可视化...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. 轮廓系数对比
methods = list(results.keys())
sil_scores = [results[m]['silhouette'] for m in methods]
axes[0].bar(methods, sil_scores, color=['steelblue', 'coral'])
axes[0].axhline(y=0, color='gray', linestyle='--')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('轮廓系数对比')
axes[0].set_ylim(-0.2, 0.5)

# 2. Davies-Bouldin Index 对比
db_scores = [results[m]['db_index'] for m in methods]
axes[1].bar(methods, db_scores, color=['steelblue', 'coral'])
axes[1].set_ylabel('Davies-Bouldin Index')
axes[1].set_title('DBI对比（越小越好）')

# 3. 聚类分布（PCA可视化）
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

axes[2].scatter(X_pca[results['DTW']['labels'] == 0, 0], X_pca[results['DTW']['labels'] == 0, 1], 
                label='Cluster 0', alpha=0.6, s=30)
axes[2].scatter(X_pca[results['DTW']['labels'] == 1, 0], X_pca[results['DTW']['labels'] == 1, 1], 
                label='Cluster 1', alpha=0.6, s=30)
if n_clusters > 2:
    axes[2].scatter(X_pca[results['DTW']['labels'] == 2, 0], X_pca[results['DTW']['labels'] == 2, 1], 
                    label='Cluster 2', alpha=0.6, s=30)
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].set_title('DTW 聚类结果 PCA 可视化')
axes[2].legend()

plt.tight_layout()
plt.savefig('dtw_improved_results.png', dpi=150)
print("可视化已保存: dtw_improved_results.png")

print("\n✅ 程序运行完成！")