# similarity_cmp_final.py
"""
参考论文改进的 DTW 聚类
直接使用价格指数数据（已预处理为起点100 + SMA-10平滑）
添加 Pearson Correlation 和 Isolation Kernel 对比
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

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("改进版时间序列聚类分析（参考论文方法）")
print("添加 Pearson Correlation 和 Isolation Kernel 对比")
print("=" * 60)

# ==========================================
# 1. 加载价格指数数据（已预处理）
# ==========================================
print("\n[1] 加载价格指数数据...")

# 读取价格指数数据（已经是起点100 + SMA-10平滑）
raw_df = pd.read_csv('./dataset/stock_data_sp500_paper.csv')

# 提取价格指数数据（跳过第一行行业信息和第一列索引）
price_data = raw_df.iloc[1:, 2:].astype(float)

# 限制股票数量（可选，加速计算）
price_data = price_data.iloc[:, :200]
print(f"限制为前200只股票: {price_data.shape}")

stock_names = price_data.columns.values
dates = raw_df.iloc[1:, 1].tolist()

print(f"价格指数数据形状: {price_data.shape}")
print(f"股票数量: {len(stock_names)}")
print(f"交易日数量: {len(dates)}")
print(f"价格范围: {price_data.min().min():.2f} - {price_data.max().max():.2f}")

# ==========================================
# 2. 筛选时间窗口
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
# 3. 清洗数据
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
# 5. 确定最佳 K 值
# ==========================================
print("\n[5] 确定最佳聚类数...")

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

# 使用 K=3（论文设置）
n_clusters = 3
print(f"使用 K = {n_clusters}（论文设置）")

# ==========================================
# 存储所有结果
# ==========================================
results = {}

# ==========================================
# 6. Euclidean 聚类
# ==========================================
print("\n[6] 运行 Euclidean K-Means...")
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
# 7. Pearson Correlation 聚类
# ==========================================
print("\n[7] 运行 Pearson Correlation 聚类...")
t0 = time.time()

# 计算相关系数矩阵
corr_matrix = np.corrcoef(X_scaled)
# 处理可能的 NaN
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
# 转换为距离：距离 = 1 - |相关系数|
dist_pearson = 1 - np.abs(corr_matrix)
# 确保距离矩阵对称
dist_pearson = (dist_pearson + dist_pearson.T) / 2
np.fill_diagonal(dist_pearson, 0)

# 使用层次聚类
labels_pearson = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='precomputed',
    linkage='average'
).fit_predict(dist_pearson)

pearson_time = time.time() - t0

# 注意：轮廓系数使用预计算的距离矩阵
results['Pearson'] = {
    'labels': labels_pearson,
    'silhouette': silhouette_score(dist_pearson, labels_pearson, metric='precomputed'),
    'db_index': davies_bouldin_score(X_scaled, labels_pearson),
    'ch_index': calinski_harabasz_score(X_scaled, labels_pearson),
    'time': pearson_time
}
print(f"  Silhouette: {results['Pearson']['silhouette']:.4f}")
print(f"  Davies-Bouldin: {results['Pearson']['db_index']:.4f}")
print(f"  Calinski-Harabasz: {results['Pearson']['ch_index']:.2f}")
print(f"  耗时: {results['Pearson']['time']:.2f} 秒")

# ==========================================
# 8. DTW 聚类
# ==========================================
print("\n[8] 运行 DTW K-Means...")
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
# 9. Isolation Kernel 聚类
# ==========================================
print("\n[9] 运行 Isolation Kernel 聚类...")
t0 = time.time()

# 使用随机森林构建核矩阵
rte = RandomTreesEmbedding(
    n_estimators=200,
    random_state=42,
    max_depth=6,
    min_samples_split=2
)
leaf_indices = rte.fit_transform(X_scaled)

# 计算相似度矩阵（共同落入同一叶子的比例）
sim_ik = (leaf_indices @ leaf_indices.T).toarray() / 200.0

# 将相似度转换为距离
dist_ik = 1 - sim_ik
dist_ik = (dist_ik + dist_ik.T) / 2
np.fill_diagonal(dist_ik, 0)

# 使用谱聚类
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
# 10. 结果对比
# ==========================================
print("\n" + "=" * 70)
print("聚类评估结果对比（4种方法）")
print("=" * 70)
print(f"{'方法':<18} {'轮廓系数':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<12} {'耗时(秒)':<10}")
print("-" * 70)

for method, scores in results.items():
    print(f"{method:<18} {scores['silhouette']:.4f}       {scores['db_index']:.4f}          {scores['ch_index']:.2f}        {scores['time']:.2f}")

print("=" * 70)

# 找出最佳方法
best_method = max(results, key=lambda x: results[x]['silhouette'])
print(f"\n🏆 最佳方法: {best_method} (轮廓系数 = {results[best_method]['silhouette']:.4f})")

# ==========================================
# 11. 可视化
# ==========================================
print("\n[10] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = list(results.keys())
sil_scores = [results[m]['silhouette'] for m in methods]
db_scores = [results[m]['db_index'] for m in methods]
ch_scores = [results[m]['ch_index'] for m in methods]
times = [results[m]['time'] for m in methods]

# 轮廓系数对比
axes[0, 0].bar(methods, sil_scores, color=['steelblue', 'coral', 'green', 'purple'])
axes[0, 0].axhline(y=0, color='gray', linestyle='--')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('轮廓系数对比（越高越好）')
axes[0, 0].set_ylim(-0.2, 0.5)

# DBI 对比
axes[0, 1].bar(methods, db_scores, color=['steelblue', 'coral', 'green', 'purple'])
axes[0, 1].set_ylabel('Davies-Bouldin Index')
axes[0, 1].set_title('DBI对比（越小越好）')

# CH 指数对比
axes[1, 0].bar(methods, ch_scores, color=['steelblue', 'coral', 'green', 'purple'])
axes[1, 0].set_ylabel('Calinski-Harabasz Index')
axes[1, 0].set_title('CH指数对比（越高越好）')

# 耗时对比
axes[1, 1].bar(methods, times, color=['steelblue', 'coral', 'green', 'purple'])
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].set_title('计算耗时对比（越低越好）')

plt.tight_layout()
plt.savefig('resuslt/sp500/top200/four_methods_comparison.png', dpi=150)
print("可视化已保存: four_methods_comparison.png")

# ==========================================
# 12. 所有方法的 PCA 可视化
# ==========================================
print("\n[11] 生成所有方法的聚类分布图...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

# 方法列表和对应的标签
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
    ax.set_title(f'{method} 聚类结果 (K={n_clusters}, Silhouette={results[method]["silhouette"]:.4f})')
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig('resuslt/sp500/top200/all_methods_clustering.png', dpi=150)
print("可视化已保存: all_methods_clustering.png")

# ==========================================
# 13. 相关性热力图对比（按每种方法的聚类排序）
# ==========================================
print("\n[12] 生成相关性热力图对比...")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))

corr_matrix_plot = np.corrcoef(X_scaled)

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes3[row, col]
    
    labels = results[method]['labels']
    sorted_idx = np.argsort(labels)
    corr_sorted = corr_matrix_plot[sorted_idx][:, sorted_idx]
    
    im = ax.imshow(corr_sorted, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_title(f'{method} 聚类排序的热力图\n(轮廓系数={results[method]["silhouette"]:.4f})')
    ax.set_xlabel('股票 (按聚类排序)')
    ax.set_ylabel('股票 (按聚类排序)')
    
    # 添加聚类分隔线
    # 找出每个聚类的边界
    unique_labels = np.unique(labels)
    boundary_positions = []
    current_pos = 0
    for label in sorted(unique_labels):
        count = np.sum(labels[sorted_idx] == label)
        current_pos += count
        boundary_positions.append(current_pos)
    
    for pos in boundary_positions[:-1]:
        ax.axhline(y=pos, color='white', linewidth=1, linestyle='-')
        ax.axvline(x=pos, color='white', linewidth=1, linestyle='-')

plt.tight_layout()
plt.colorbar(im, ax=axes3, label='相关系数')
plt.savefig('resuslt/sp500/top200/all_methods_heatmap.png', dpi=150)
print("可视化已保存: all_methods_heatmap.png")

# ==========================================
# 14. 聚类分布饼图（展示各簇大小）
# ==========================================
print("\n[13] 生成聚类分布饼图...")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes4[row, col]
    
    labels = results[method]['labels']
    unique, counts = np.unique(labels, return_counts=True)
    
    ax.pie(counts, labels=[f'Cluster {i}' for i in unique], 
           autopct='%1.1f%%', colors=colors[:len(unique)])
    ax.set_title(f'{method} 聚类分布\n(轮廓系数={results[method]["silhouette"]:.4f})')

plt.tight_layout()
plt.savefig('resuslt/sp500/top200/all_methods_piecharts.png', dpi=150)
print("可视化已保存: all_methods_piecharts.png")

print("\n✅ 所有可视化生成完成！")
print("生成的文件:")
print("  - all_methods_clustering.png (PCA聚类分布图)")
print("  - all_methods_heatmap.png (相关性热力图)")
print("  - all_methods_piecharts.png (聚类分布饼图)")