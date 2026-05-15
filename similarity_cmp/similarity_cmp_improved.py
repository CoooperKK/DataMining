# similarity_cmp_final.py
"""
参考论文改进的 DTW 聚类
直接使用价格指数数据（已预处理为起点100 + SMA-10平滑）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.decomposition import PCA

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
# 1. 加载价格指数数据（已预处理）
# ==========================================
print("\n[1] 加载价格指数数据...")

# 读取价格指数数据（已经是起点100 + SMA-10平滑）
raw_df = pd.read_csv('./dataset/stock_data_sp500_paper.csv')

# 提取价格指数数据（跳过第一行行业信息和第一列索引）
price_data = raw_df.iloc[1:, 2:].astype(float)

price_data = price_data.iloc[:, :50]
print(f"限制为前50只股票: {price_data.shape}")

stock_names = price_data.columns.values
dates = raw_df.iloc[1:, 1].tolist()

print(f"价格指数数据形状: {price_data.shape}")
print(f"股票数量: {len(stock_names)}")
print(f"交易日数量: {len(dates)}")
print(f"价格范围: {price_data.min().min():.2f} - {price_data.max().max():.2f}")

# ==========================================
# 2. 筛选时间窗口（论文使用完整5年，可选最近一年）
# ==========================================
print("\n[2] 筛选时间窗口...")

USE_RECENT_YEAR = False  # 设为 True 使用最近一年，False 使用全部数据

if USE_RECENT_YEAR:
    time_window = 250
    price_data = price_data.iloc[-time_window:, :]
    print(f"使用最近 {time_window} 个交易日")
else:
    print(f"使用全部 {price_data.shape[0]} 个交易日（论文方法）")

# ==========================================
# 3. 清洗数据（删除缺失过多的股票）
# ==========================================
print("\n[3] 清洗数据...")

# 删除缺失过多的股票
missing_ratio = price_data.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.1].index
price_clean = price_data[valid_stocks]

print(f"删除缺失>10%的股票后: {len(valid_stocks)} 只")

# 填充剩余缺失值
price_clean = price_clean.ffill().bfill().fillna(100)

# ==========================================
# 4. 标准化
# ==========================================
print("\n[4] 标准化...")

# 转置为 (股票数, 天数) 格式
X = price_clean.T.values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化后数据形状: {X_scaled.shape[0]} 只股票 × {X_scaled.shape[1]} 个特征")

# ==========================================
# 5. 确定最佳 K 值（Elbow + Silhouette）
# ==========================================
print("\n[5] 确定最佳聚类数...")

k_range = range(2, 11)
silhouette_scores = []
wcss = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    wcss.append(kmeans.inertia_)
    print(f"  K={k}: Silhouette={sil_score:.4f}")

# 选择最佳 K（论文选择 K=3 或 4）
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n轮廓系数建议 K = {best_k}")
print("论文建议 K = 3 或 4")

# 手动设置 K（论文使用 K=3）
n_clusters = 3
print(f"使用 K = {n_clusters}（论文设置）")

# ==========================================
# 6. Euclidean 聚类
# ==========================================
print("\n[6] 运行 Euclidean K-Means...")
km_ed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_ed = km_ed.fit_predict(X_scaled)

sil_ed = silhouette_score(X_scaled, labels_ed)
db_ed = davies_bouldin_score(X_scaled, labels_ed)
ch_ed = calinski_harabasz_score(X_scaled, labels_ed)

print(f"  Silhouette: {sil_ed:.4f}")
print(f"  Davies-Bouldin: {db_ed:.4f}")
print(f"  Calinski-Harabasz: {ch_ed:.2f}")

# ==========================================
# 7. DTW 聚类
# ==========================================
print("\n[7] 运行 DTW K-Means...")
t0 = time.time()

# 使用 DTW 距离
km_dtw = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    max_iter=10,
    metric_params={
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": 10
    },
    n_jobs=1,
    random_state=42,
    verbose=False
)

labels_dtw = km_dtw.fit_predict(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1))
dtw_time = time.time() - t0

sil_dtw = silhouette_score(X_scaled, labels_dtw)
db_dtw = davies_bouldin_score(X_scaled, labels_dtw)
ch_dtw = calinski_harabasz_score(X_scaled, labels_dtw)

print(f"  Silhouette: {sil_dtw:.4f}")
print(f"  Davies-Bouldin: {db_dtw:.4f}")
print(f"  Calinski-Harabasz: {ch_dtw:.2f}")
print(f"  耗时: {dtw_time:.2f} 秒")

# ==========================================
# 8. 结果对比
# ==========================================
print("\n" + "=" * 60)
print("聚类评估结果对比")
print("=" * 60)
print(f"{'方法':<15} {'轮廓系数':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<15}")
print("-" * 60)
print(f"{'Euclidean':<15} {sil_ed:.4f}       {db_ed:.4f}          {ch_ed:.2f}")
print(f"{'DTW':<15} {sil_dtw:.4f}       {db_dtw:.4f}          {ch_dtw:.2f}")
print("=" * 60)

# 判断哪个方法更好
if sil_dtw > sil_ed:
    print(f"\n✅ DTW 效果更好 (轮廓系数: {sil_dtw:.4f} > {sil_ed:.4f})")
else:
    print(f"\n⚠️ Euclidean 效果更好 (轮廓系数: {sil_ed:.4f} > {sil_dtw:.4f})")

# ==========================================
# 9. 可视化
# ==========================================
print("\n[8] 生成可视化...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 轮廓系数对比
axes[0].bar(['Euclidean', 'DTW'], [sil_ed, sil_dtw], color=['steelblue', 'coral'])
axes[0].axhline(y=0, color='gray', linestyle='--')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('轮廓系数对比')
axes[0].set_ylim(-0.2, 0.5)

# DBI 对比
axes[1].bar(['Euclidean', 'DTW'], [db_ed, db_dtw], color=['steelblue', 'coral'])
axes[1].set_ylabel('Davies-Bouldin Index')
axes[1].set_title('DBI对比（越小越好）')

# PCA 可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

axes[2].scatter(X_pca[labels_dtw == 0, 0], X_pca[labels_dtw == 0, 1], 
                label='Cluster 0', alpha=0.6, s=30)
axes[2].scatter(X_pca[labels_dtw == 1, 0], X_pca[labels_dtw == 1, 1], 
                label='Cluster 1', alpha=0.6, s=30)
if n_clusters > 2:
    axes[2].scatter(X_pca[labels_dtw == 2, 0], X_pca[labels_dtw == 2, 1], 
                    label='Cluster 2', alpha=0.6, s=30)

axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].set_title(f'DTW 聚类结果 (K={n_clusters})')
axes[2].legend()

plt.tight_layout()
plt.savefig('dtw_paper_results.png', dpi=150)
print("可视化已保存: dtw_paper_results.png")

print("\n✅ 程序运行完成！")