# similarity_cmp_returns.py
"""
基于收益率的聚类分析
直接使用收益率数据（已标准化）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomTreesEmbedding
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("基于收益率的聚类分析")
print("对比 Euclidean、Pearson、DTW、Isolation Kernel")
print("=" * 60)

# ==========================================
# 1. 加载收益率数据
# ==========================================
print("\n[1] 加载收益率数据...")

# 读取收益率数据
raw_df = pd.read_csv('./dataset/stock_data_returns.csv')

# 提取收益率数据（跳过第一行行业信息和第一列索引）
returns_data = raw_df.iloc[1:, 2:].astype(float)
stock_names = returns_data.columns.values
dates = raw_df.iloc[1:, 1].tolist()

print(f"收益率数据形状: {returns_data.shape}")
print(f"股票数量: {len(stock_names)}")
print(f"交易日数量: {len(dates)}")
print(f"收益率范围: {returns_data.min().min():.4f} - {returns_data.max().max():.4f}")

# ==========================================
# 2. 标准化
# ==========================================
print("\n[2] 标准化...")

# 转置为 (股票数, 天数) 格式
X = returns_data.T.values

# Z-score 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化后数据形状: {X_scaled.shape[0]} 只股票 × {X_scaled.shape[1]} 个特征")

# ==========================================
# 3. 确定最佳 K 值
# ==========================================
print("\n[3] 确定最佳聚类数...")

k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"  K={k}: Silhouette={sil_score:.4f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n轮廓系数建议 K = {best_k}")

# 使用 K=3（保持与论文一致）
n_clusters = 3
print(f"使用 K = {n_clusters}")

# ==========================================
# 存储所有结果
# ==========================================
results = {}

# ==========================================
# 4. Euclidean 聚类
# ==========================================
print("\n[4] 运行 Euclidean K-Means...")
t0 = time.time()
km_ed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_ed = km_ed.fit_predict(X_scaled)
euclidean_time = time.time() - t0

results['Euclidean'] = {
    'labels': labels_ed,
    'silhouette': silhouette_score(X_scaled, labels_ed),
    'db_index': davies_bouldin_score(X_scaled, labels_ed),
    'ch_index': calinski_harabasz_score(X_scaled, labels_ed),
    'time': euclidean_time
}
print(f"  Silhouette: {results['Euclidean']['silhouette']:.4f}")
print(f"  Davies-Bouldin: {results['Euclidean']['db_index']:.4f}")
print(f"  Calinski-Harabasz: {results['Euclidean']['ch_index']:.2f}")
print(f"  耗时: {results['Euclidean']['time']:.2f} 秒")

# ==========================================
# 5. Pearson Correlation 聚类
# ==========================================
print("\n[5] 运行 Pearson Correlation 聚类...")
t0 = time.time()

# 计算相关系数矩阵
corr_matrix = np.corrcoef(X_scaled)
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
dist_pearson = 1 - np.abs(corr_matrix)
dist_pearson = (dist_pearson + dist_pearson.T) / 2
np.fill_diagonal(dist_pearson, 0)

# MDS 将距离矩阵转换为坐标
mds = MDS(n_components=50, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
X_pearson = mds.fit_transform(dist_pearson)

# K-Means 聚类
km_pearson = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_pearson = km_pearson.fit_predict(X_pearson)

pearson_time = time.time() - t0

results['Pearson'] = {
    'labels': labels_pearson,
    'silhouette': silhouette_score(X_pearson, labels_pearson),
    'db_index': davies_bouldin_score(X_scaled, labels_pearson),
    'ch_index': calinski_harabasz_score(X_scaled, labels_pearson),
    'time': pearson_time
}
print(f"  Silhouette: {results['Pearson']['silhouette']:.4f}")
print(f"  Davies-Bouldin: {results['Pearson']['db_index']:.4f}")
print(f"  Calinski-Harabasz: {results['Pearson']['ch_index']:.2f}")
print(f"  耗时: {results['Pearson']['time']:.2f} 秒")

# ==========================================
# 6. DTW 聚类
# ==========================================
print("\n[6] 运行 DTW K-Means...")
t0 = time.time()

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

# 需要将数据 reshape 为 (样本数, 时间步长, 特征数)
labels_dtw = km_dtw.fit_predict(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1))
dtw_time = time.time() - t0

results['DTW'] = {
    'labels': labels_dtw,
    'silhouette': silhouette_score(X_scaled, labels_dtw),
    'db_index': davies_bouldin_score(X_scaled, labels_dtw),
    'ch_index': calinski_harabasz_score(X_scaled, labels_dtw),
    'time': dtw_time
}
print(f"  Silhouette: {results['DTW']['silhouette']:.4f}")
print(f"  Davies-Bouldin: {results['DTW']['db_index']:.4f}")
print(f"  Calinski-Harabasz: {results['DTW']['ch_index']:.2f}")
print(f"  耗时: {results['DTW']['time']:.2f} 秒")

# ==========================================
# 7. Isolation Kernel 聚类
# ==========================================
print("\n[7] 运行 Isolation Kernel 聚类...")
t0 = time.time()

# 使用随机森林构建核矩阵
rte = RandomTreesEmbedding(
    n_estimators=200,
    random_state=42,
    max_depth=6,
    min_samples_split=2
)
leaf_indices = rte.fit_transform(X_scaled)

# 计算相似度矩阵
sim_ik = (leaf_indices @ leaf_indices.T).toarray() / 200.0
dist_ik = 1 - sim_ik
dist_ik = (dist_ik + dist_ik.T) / 2
np.fill_diagonal(dist_ik, 0)

# 谱聚类
labels_ik = SpectralClustering(
    n_clusters=n_clusters,
    affinity='precomputed',
    random_state=42,
    assign_labels='kmeans'
).fit_predict(sim_ik)

ik_time = time.time() - t0

results['Isolation Kernel'] = {
    'labels': labels_ik,
    'silhouette': silhouette_score(dist_ik, labels_ik, metric='precomputed'),
    'db_index': davies_bouldin_score(X_scaled, labels_ik),
    'ch_index': calinski_harabasz_score(X_scaled, labels_ik),
    'time': ik_time
}
print(f"  Silhouette: {results['Isolation Kernel']['silhouette']:.4f}")
print(f"  Davies-Bouldin: {results['Isolation Kernel']['db_index']:.4f}")
print(f"  Calinski-Harabasz: {results['Isolation Kernel']['ch_index']:.2f}")
print(f"  耗时: {results['Isolation Kernel']['time']:.2f} 秒")

# ==========================================
# 8. 结果对比
# ==========================================
print("\n" + "=" * 70)
print("聚类评估结果对比（4种方法）")
print("=" * 70)
print(f"{'方法':<18} {'轮廓系数':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<12} {'耗时(秒)':<10}")
print("-" * 70)

for method, scores in results.items():
    print(f"{method:<18} {scores['silhouette']:.4f}       {scores['db_index']:.4f}          {scores['ch_index']:.2f}        {scores['time']:.2f}")

print("=" * 70)

best_method = max(results, key=lambda x: results[x]['silhouette'])
print(f"\n🏆 最佳方法: {best_method} (轮廓系数 = {results[best_method]['silhouette']:.4f})")

# ==========================================
# 9. 可视化
# ==========================================
print("\n[8] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = list(results.keys())
sil_scores = [results[m]['silhouette'] for m in methods]
db_scores = [results[m]['db_index'] for m in methods]
ch_scores = [results[m]['ch_index'] for m in methods]
times = [results[m]['time'] for m in methods]

axes[0, 0].bar(methods, sil_scores, color=['steelblue', 'coral', 'green', 'purple'])
axes[0, 0].axhline(y=0, color='gray', linestyle='--')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('轮廓系数对比（越高越好）')
axes[0, 0].set_ylim(-0.2, 0.5)

axes[0, 1].bar(methods, db_scores, color=['steelblue', 'coral', 'green', 'purple'])
axes[0, 1].set_ylabel('Davies-Bouldin Index')
axes[0, 1].set_title('DBI对比（越小越好）')

axes[1, 0].bar(methods, ch_scores, color=['steelblue', 'coral', 'green', 'purple'])
axes[1, 0].set_ylabel('Calinski-Harabasz Index')
axes[1, 0].set_title('CH指数对比（越高越好）')

axes[1, 1].bar(methods, times, color=['steelblue', 'coral', 'green', 'purple'])
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].set_title('计算耗时对比（越低越好）')

plt.tight_layout()
plt.savefig('returns_comparison.png', dpi=150)
print("可视化已保存: returns_comparison.png")

# PCA 可视化
print("\n[9] 生成PCA聚类分布图...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
methods_to_plot = ['Euclidean', 'Pearson', 'DTW', 'Isolation Kernel']

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes2[row, col]
    
    labels = results[method]['labels']
    
    for i in range(n_clusters):
        ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1],
                   label=f'Cluster {i}', alpha=0.6, s=30, 
                   color=colors[i % len(colors)])
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'{method} (K={n_clusters}, Sil={results[method]["silhouette"]:.4f})')
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('returns_clustering.png', dpi=150)
print("可视化已保存: returns_clustering.png")

print("\n✅ 程序运行完成！")