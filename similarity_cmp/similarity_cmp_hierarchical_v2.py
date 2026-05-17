# similarity_cmp_hierarchical.py
"""
参考论文改进的聚类分析
统一使用层次聚类（AgglomerativeClustering）
添加 CPCC 指标评估保真率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomTreesEmbedding
from tslearn.metrics import dtw_path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("基于层次聚类的相似度度量对比")
print("统一使用层次聚类（AgglomerativeClustering）")
print("=" * 60)

# ==========================================
# 1. 加载价格指数数据（已预处理）
# ==========================================
print("\n[1] 加载价格指数数据...")

raw_df = pd.read_csv('./dataset/stock_data_sp500_paper.csv')
price_data = raw_df.iloc[1:, 2:].astype(float)

# 限制股票数量
stock_num = 300
price_data = price_data.iloc[:, :stock_num]
print(f"限制为前 {stock_num} 只股票: {price_data.shape}")

stock_names = price_data.columns.values
dates = raw_df.iloc[1:, 1].tolist()

# ==========================================
# 2. 筛选时间窗口
# ==========================================
print("\n[2] 筛选时间窗口...")
USE_RECENT_YEAR = False

if USE_RECENT_YEAR:
    time_window = 250
    price_data = price_data.iloc[-time_window:, :]
    print(f"使用最近 {time_window} 个交易日")
else:
    print(f"使用全部 {price_data.shape[0]} 个交易日")

# ==========================================
# 3. 清洗数据
# ==========================================
print("\n[3] 清洗数据...")

missing_ratio = price_data.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.1].index
price_clean = price_data[valid_stocks]
price_clean = price_clean.ffill().bfill().fillna(100)

# ==========================================
# 4. 标准化
# ==========================================
print("\n[4] 标准化...")

# X = price_clean.T.values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print(f"标准化后数据形状: {X_scaled.shape[0]} 只股票 × {X_scaled.shape[1]} 个特征")

X_scaled = price_clean.T.values  # 不标准化

# ==========================================
# 5. 确定最佳 K 值
# ==========================================
print("\n[5] 确定最佳聚类数...")

k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    clusterer = AgglomerativeClustering(n_clusters=k, linkage='average')
    labels = clusterer.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"  K={k}: Silhouette={sil_score:.4f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n轮廓系数建议 K = {best_k}")

n_clusters = 3
print(f"使用 K = {n_clusters}（论文设置）")

# ==========================================
# 存储所有结果
# ==========================================
results = {}

# ==========================================
# 辅助函数：计算 CPCC（Cophentic Correlation Coefficient）
# ==========================================
def calculate_cpcc(dist_matrix, linkage_matrix):
    """
    计算 Cophentic Correlation Coefficient（保真率）
    衡量原始距离与树状图距离的相关性
    """
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform
    
    # 将距离矩阵转换为压缩格式
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # 计算 cophenetic 相关系数
    cpcc, _ = cophenet(linkage_matrix, condensed_dist)
    return cpcc

# ==========================================
# 6. Euclidean 距离 + 层次聚类
# ==========================================
print("\n[6] 运行 Euclidean + 层次聚类...")
t0 = time.time()

# 计算欧氏距离矩阵
from sklearn.metrics.pairwise import euclidean_distances
dist_euclidean = euclidean_distances(X_scaled)
dist_euclidean = (dist_euclidean + dist_euclidean.T) / 2
np.fill_diagonal(dist_euclidean, 0)

# 层次聚类
from scipy.cluster.hierarchy import linkage
linkage_euclidean = linkage(squareform(dist_euclidean), method='average')

# 计算 CPCC
cpcc_euclidean = calculate_cpcc(dist_euclidean, linkage_euclidean)

# 获取聚类标签
from scipy.cluster.hierarchy import fcluster
labels_euclidean = fcluster(linkage_euclidean, t=n_clusters, criterion='maxclust') - 1

euclidean_time = time.time() - t0

results['Euclidean'] = {
    'labels': labels_euclidean,
    'linkage': linkage_euclidean,
    'silhouette': silhouette_score(X_scaled, labels_euclidean),
    'db_index': davies_bouldin_score(X_scaled, labels_euclidean),
    'ch_index': calinski_harabasz_score(X_scaled, labels_euclidean),
    'cpcc': cpcc_euclidean,
    'time': euclidean_time
}
print(f"  Silhouette: {results['Euclidean']['silhouette']:.4f}")
print(f"  CPCC: {results['Euclidean']['cpcc']:.4f}")
print(f"  Davies-Bouldin: {results['Euclidean']['db_index']:.4f}")
print(f"  耗时: {results['Euclidean']['time']:.2f} 秒")

# ==========================================
# 7. Pearson 距离 + 层次聚类
# ==========================================
print("\n[7] 运行 Pearson + 层次聚类...")
t0 = time.time()

# 计算 Pearson 距离矩阵
corr_matrix = np.corrcoef(X_scaled)
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
dist_pearson = 1 - np.abs(corr_matrix)
dist_pearson = (dist_pearson + dist_pearson.T) / 2
np.fill_diagonal(dist_pearson, 0)

# 层次聚类
linkage_pearson = linkage(squareform(dist_pearson), method='average')
cpcc_pearson = calculate_cpcc(dist_pearson, linkage_pearson)

labels_pearson = fcluster(linkage_pearson, t=n_clusters, criterion='maxclust') - 1
pearson_time = time.time() - t0

results['Pearson'] = {
    'labels': labels_pearson,
    'linkage': linkage_pearson,
    'silhouette': silhouette_score(dist_pearson, labels_pearson, metric='precomputed'),
    'db_index': davies_bouldin_score(X_scaled, labels_pearson),
    'ch_index': calinski_harabasz_score(X_scaled, labels_pearson),
    'cpcc': cpcc_pearson,
    'time': pearson_time
}
print(f"  Silhouette: {results['Pearson']['silhouette']:.4f}")
print(f"  CPCC: {results['Pearson']['cpcc']:.4f}")
print(f"  Davies-Bouldin: {results['Pearson']['db_index']:.4f}")
print(f"  耗时: {results['Pearson']['time']:.2f} 秒")

# ==========================================
# 8. DTW 距离 + 层次聚类
# ==========================================
print("\n[8] 运行 DTW + 层次聚类...")
t0 = time.time()

# 计算 DTW 距离矩阵（简化版，加速计算）
n_stocks = X_scaled.shape[0]
dist_dtw = np.zeros((n_stocks, n_stocks))

# 限制计算量，如果股票太多则采样
MAX_DTW_STOCKS = 100
if n_stocks > MAX_DTW_STOCKS:
    print(f"  股票数量较多({n_stocks})，DTW计算可能较慢...")

for i in range(n_stocks):
    for j in range(i+1, n_stocks):
        # 使用快速 DTW
        from tslearn.metrics import dtw
        dist = dtw(X_scaled[i], X_scaled[j], global_constraint='sakoe_chiba', sakoe_chiba_radius=10)
        dist_dtw[i, j] = dist
        dist_dtw[j, i] = dist
    if (i+1) % 20 == 0:
        print(f"  已计算 {i+1}/{n_stocks} 只股票")

dist_dtw = (dist_dtw + dist_dtw.T) / 2
np.fill_diagonal(dist_dtw, 0)

# 层次聚类
linkage_dtw = linkage(squareform(dist_dtw), method='average')
cpcc_dtw = calculate_cpcc(dist_dtw, linkage_dtw)

labels_dtw = fcluster(linkage_dtw, t=n_clusters, criterion='maxclust') - 1
dtw_time = time.time() - t0

results['DTW'] = {
    'labels': labels_dtw,
    'linkage': linkage_dtw,
    'silhouette': silhouette_score(dist_dtw, labels_dtw, metric='precomputed'),
    'db_index': davies_bouldin_score(X_scaled, labels_dtw),
    'ch_index': calinski_harabasz_score(X_scaled, labels_dtw),
    'cpcc': cpcc_dtw,
    'time': dtw_time
}
print(f"  Silhouette: {results['DTW']['silhouette']:.4f}")
print(f"  CPCC: {results['DTW']['cpcc']:.4f}")
print(f"  Davies-Bouldin: {results['DTW']['db_index']:.4f}")
print(f"  耗时: {results['DTW']['time']:.2f} 秒")

# ==========================================
# 9. Isolation Kernel 距离 + 层次聚类
# ==========================================
print("\n[9] 运行 Isolation Kernel + 层次聚类...")
t0 = time.time()

# 随机森林构建核矩阵
# 产生更高相似度的参数
n_estimators = 200

rte = RandomTreesEmbedding(
    n_estimators=n_estimators,
    max_depth=5,            # 更浅的树
    min_samples_split=3,
    max_leaf_nodes=200,     # 更多的叶子
    random_state=42
)
leaf_indices = rte.fit_transform(X_scaled)

# 计算相似度矩阵
sim_ik = (leaf_indices @ leaf_indices.T).toarray() / n_estimators

# 转换为距离
dist_ik = 1 - sim_ik
dist_ik = (dist_ik + dist_ik.T) / 2
np.fill_diagonal(dist_ik, 0)


# 层次聚类
linkage_ik = linkage(squareform(dist_ik), method='average')
cpcc_ik = calculate_cpcc(dist_ik, linkage_ik)

labels_ik = fcluster(linkage_ik, t=n_clusters, criterion='maxclust') - 1
ik_time = time.time() - t0

# 聚类分布
unique, counts = np.unique(labels_ik, return_counts=True)
print(f"  聚类分布: {dict(zip(unique, counts))}")

results['Isolation Kernel'] = {
    'labels': labels_ik,
    'linkage': linkage_ik,
    'silhouette': silhouette_score(dist_ik, labels_ik, metric='precomputed'),
    'db_index': davies_bouldin_score(X_scaled, labels_ik),
    'ch_index': calinski_harabasz_score(X_scaled, labels_ik),
    'cpcc': cpcc_ik,
    'time': ik_time
}
print(f"  Silhouette: {results['Isolation Kernel']['silhouette']:.4f}")
print(f"  CPCC: {results['Isolation Kernel']['cpcc']:.4f}")
print(f"  Davies-Bouldin: {results['Isolation Kernel']['db_index']:.4f}")
print(f"  耗时: {results['Isolation Kernel']['time']:.2f} 秒")
# ==========================================
# 10. 结果对比表格
# ==========================================
print("\n" + "=" * 80)
print("Clustering Evaluation Results (4 Similarity Measures)")
print("=" * 80)
print(f"{'Method':<18} {'Silhouette':<12} {'CPCC':<10} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<12} {'Time(s)':<10}")
print("-" * 80)

for method, scores in results.items():
    print(f"{method:<18} {scores['silhouette']:.4f}       {scores['cpcc']:.4f}     {scores['db_index']:.4f}          {scores['ch_index']:.2f}        {scores['time']:.2f}")

print("=" * 80)

best_method = max(results, key=lambda x: results[x]['silhouette'])
print(f"\n🏆 Best Method (by Silhouette): {best_method} (Silhouette = {results[best_method]['silhouette']:.4f})")

best_cpcc = max(results, key=lambda x: results[x]['cpcc'])
print(f"🏆 Best Method (by CPCC): {best_cpcc} (CPCC = {results[best_cpcc]['cpcc']:.4f})")

# # ==========================================
# # 11. 折线图展示聚类效果（学术风格）- 所有股票曲线
# # ==========================================
# print("\n[10] Generating cluster assignment plots...")

# # 设置学术风格
# plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# plt.rcParams['font.size'] = 11
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['legend.fontsize'] = 9
# plt.rcParams['figure.dpi'] = 150
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['savefig.bbox'] = 'tight'

# # 学术颜色方案
# cluster_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3', '#F781BF', '#A65628', '#FDC086']

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('Stock Price Index Curves Colored by Cluster Assignment', fontsize=14, fontweight='bold', y=0.98)

# methods_to_plot = ['Euclidean', 'Pearson', 'DTW', 'Isolation Kernel']

# # 计算时间轴
# if USE_RECENT_YEAR:
#     n_days = time_window if isinstance(time_window, int) else 250
#     time_points = np.arange(1, n_days + 1)
# else:
#     n_days = X_scaled.shape[1]
#     time_points = np.arange(1, n_days + 1)

# # 采样参数：最多绘制120条曲线
# MAX_LINES_TOTAL = 100
# np.random.seed(42)

# for idx, method in enumerate(methods_to_plot):
#     row = idx // 2
#     col = idx % 2
#     ax = axes[row, col]
    
#     labels = results[method]['labels']
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     largest_cluster = unique_labels[np.argmax(counts)]
    
#     # 为每个簇分配颜色（最大簇为橙黄色）
#     other_colors = [c for c in cluster_colors if c != '#FF8C00']
#     label_to_color = {}
#     other_idx = 0
#     for label in unique_labels:
#         if label == largest_cluster:
#             label_to_color[label] = '#FF8C00'  # 橙黄色
#         else:
#             label_to_color[label] = other_colors[other_idx % len(other_colors)]
#             other_idx += 1
    
#     # 按簇分组采样，总数不超过 MAX_LINES_TOTAL
#     cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
#     sample_indices = []
    
#     for label in unique_labels:
#         cluster_size = len(cluster_indices[label])
#         if label == largest_cluster:
#             n_sample = min(cluster_size, int(MAX_LINES_TOTAL * 0.6))
#         else:
#             other_total = sum(len(cluster_indices[l]) for l in unique_labels if l != largest_cluster)
#             if other_total > 0:
#                 n_sample = min(cluster_size, max(1, int(MAX_LINES_TOTAL * 0.4 * cluster_size / other_total)))
#             else:
#                 n_sample = min(cluster_size, 3)
#         sampled = np.random.choice(cluster_indices[label], n_sample, replace=False)
#         sample_indices.extend(sampled)
    
#     if len(sample_indices) > MAX_LINES_TOTAL:
#         sample_indices = np.random.choice(sample_indices, MAX_LINES_TOTAL, replace=False)
    
#     # 绘制曲线：最大簇更突出
#     for i in sample_indices:
#         color = label_to_color[labels[i]]
#         if labels[i] == largest_cluster:
#             ax.plot(time_points, X_scaled[i, :], color=color, linewidth=1.2, alpha=0.5)
#         else:
#             ax.plot(time_points, X_scaled[i, :], color=color, linewidth=0.6, alpha=0.5)
    
#     # 图例
#     legend_handles = []
#     for label in unique_labels:
#         count = np.sum(labels == label)
#         color = label_to_color[label]
#         if label == largest_cluster:
#             label_name = f'Major Cluster (n={count})'
#         else:
#             label_name = f'Cluster {label+1} (n={count})'
#         legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label_name))
#     ax.legend(handles=legend_handles, loc='best', frameon=True, fancybox=True, fontsize=8)
    
#     ax.set_title(f'{method}', fontweight='bold')
#     ax.set_xlabel('Trading Day', fontsize=10)
#     ax.set_ylabel('Standardized Price Index', fontsize=10)
#     ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
#     ax.set_xlim(1, n_days)

# plt.tight_layout()
# plt.savefig('result/hierarchical/cluster_curves_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('result/hierarchical/cluster_curves_comparison.pdf', format='pdf', bbox_inches='tight')
# print("Visualization saved: result/hierarchical/cluster_curves_comparison.png/pdf")


# ==========================================
# 11. 折线图展示聚类效果（学术风格）- 所有股票曲线
# ==========================================
print("\n[10] Generating cluster assignment plots...")

# 设置学术风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 学术颜色方案
cluster_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3', '#F781BF', '#A65628', '#FDC086']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stock Price Index Curves Colored by Cluster Assignment (Excluding Major Cluster)', fontsize=14, fontweight='bold', y=0.98)

methods_to_plot = ['Euclidean', 'Pearson', 'DTW', 'Isolation Kernel']

# 计算时间轴
if USE_RECENT_YEAR:
    n_days = time_window if isinstance(time_window, int) else 250
    time_points = np.arange(1, n_days + 1)
else:
    n_days = X_scaled.shape[1]
    time_points = np.arange(1, n_days + 1)

# 采样参数：最多绘制100条曲线
MAX_LINES_TOTAL = 100
np.random.seed(42)

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    labels = results[method]['labels']
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    
    # 为每个簇分配颜色
    label_to_color = {}
    other_labels = [l for l in unique_labels if l != largest_cluster]
    for i, label in enumerate(other_labels):
        label_to_color[label] = cluster_colors[i % len(cluster_colors)]
    
    # 按簇分组采样，只画非最大簇
    cluster_indices = {label: np.where(labels == label)[0] for label in other_labels}
    sample_indices = []
    other_total = sum(len(cluster_indices[l]) for l in other_labels)
    
    for label in other_labels:
        cluster_size = len(cluster_indices[label])
        if other_total > 0:
            n_sample = min(cluster_size, max(1, int(MAX_LINES_TOTAL * cluster_size / other_total)))
        else:
            n_sample = min(cluster_size, 3)
        sampled = np.random.choice(cluster_indices[label], n_sample, replace=False)
        sample_indices.extend(sampled)
    
    if len(sample_indices) > MAX_LINES_TOTAL:
        sample_indices = np.random.choice(sample_indices, MAX_LINES_TOTAL, replace=False)
    
    # 绘制曲线
    for i in sample_indices:
        color = label_to_color[labels[i]]
        ax.plot(time_points, X_scaled[i, :], color=color, linewidth=0.8, alpha=0.6)
    
    # 图例（只显示非最大簇）
    legend_handles = []
    for label in other_labels:
        count = np.sum(labels == label)
        color = label_to_color[label]
        label_name = f'Cluster {label+1} (n={count})'
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label_name))
    ax.legend(handles=legend_handles, loc='best', frameon=True, fancybox=True, fontsize=8)
    
    # 在标题中注明最大簇被排除
    major_count = np.sum(labels == largest_cluster)
    ax.set_title(f'{method} (Major Cluster n={major_count} excluded)', fontweight='bold')
    ax.set_xlabel('Trading Day', fontsize=10)
    ax.set_ylabel('Standardized Price Index', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_xlim(1, n_days)

plt.tight_layout()
plt.savefig('result/hierarchical/cluster_curves_comparison_no_major.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/cluster_curves_comparison_no_major.pdf', format='pdf', bbox_inches='tight')
print("Visualization saved: result/hierarchical/cluster_curves_comparison_no_major.png/pdf")



# ==========================================
# 12. 树状图展示（学术风格）
# ==========================================
print("\n[11] Generating dendrograms...")

from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

# 设置树状图颜色
set_link_color_palette(['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3'])

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Hierarchical Clustering Dendrograms', fontsize=14, fontweight='bold', y=0.98)

n_display = min(40, len(stock_names))

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes2[row, col]
    
    linkage_matrix = results[method]['linkage']
    
    # 创建树状图
    dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=n_display,
               leaf_rotation=45, leaf_font_size=7,
               color_threshold=0.7 * max(linkage_matrix[:, 2]),
               above_threshold_color='#888888')
    
    ax.set_title(f'{method}', fontweight='bold')
    ax.set_xlabel('Stock Index', fontsize=10)
    ax.set_ylabel('Distance', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # 添加CPCC值
    ax.text(0.98, 0.02, f'CPCC = {results[method]["cpcc"]:.4f}', 
            transform=ax.transAxes, fontsize=9,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('result/hierarchical/dendrograms_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/dendrograms_comparison.pdf', format='pdf', bbox_inches='tight')
print("Visualization saved: result/hierarchical/dendrograms_comparison.png/pdf")

# ==========================================
# 13. 指标对比柱状图（学术风格）
# ==========================================
print("\n[12] Generating metrics comparison bar charts...")

# 使用学术风格的配色
academic_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle('Clustering Performance Metrics', fontsize=14, fontweight='bold', y=0.98)

methods = list(results.keys())
sil_scores = [results[m]['silhouette'] for m in methods]
cpcc_scores = [results[m]['cpcc'] for m in methods]
db_scores = [results[m]['db_index'] for m in methods]

# 轮廓系数
bars1 = axes3[0].bar(methods, sil_scores, color=academic_colors, edgecolor='black', linewidth=0.8, alpha=0.8)
axes3[0].axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
axes3[0].set_ylabel('Silhouette Score', fontsize=11)
axes3[0].set_title('(a) Silhouette Score (Higher is Better)', fontsize=10)
axes3[0].set_ylim(-0.1, 0.9)
axes3[0].grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
# 添加数值标签
for bar, score in zip(bars1, sil_scores):
    axes3[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9)

# CPCC
bars2 = axes3[1].bar(methods, cpcc_scores, color=academic_colors, edgecolor='black', linewidth=0.8, alpha=0.8)
axes3[1].set_ylabel('CPCC (Cophenetic Correlation)', fontsize=11)
axes3[1].set_title('(b) CPCC (Higher is Better)', fontsize=10)
axes3[1].set_ylim(0, 1)
axes3[1].grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
for bar, score in zip(bars2, cpcc_scores):
    axes3[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)

# DBI
bars3 = axes3[2].bar(methods, db_scores, color=academic_colors, edgecolor='black', linewidth=0.8, alpha=0.8)
axes3[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes3[2].set_title('(c) Davies-Bouldin Index (Lower is Better)', fontsize=10)
axes3[2].grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
axes3[2].set_ylim(0, max(db_scores) * 1.2)
for bar, score in zip(bars3, db_scores):
    axes3[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + (max(db_scores) * 0.03),
             f'{score:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('result/hierarchical/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/metrics_comparison.pdf', format='pdf', bbox_inches='tight')
print("Visualization saved: result/hierarchical/metrics_comparison.png/pdf")

# ==========================================
# 14. 附加：算法效率对比图
# ==========================================
print("\n[13] Generating computational efficiency comparison...")

fig4, ax4 = plt.subplots(figsize=(8, 5))
times = [results[m]['time'] for m in methods]

bars = ax4.bar(methods, times, color=academic_colors, edgecolor='black', linewidth=0.8, alpha=0.8)
ax4.set_ylabel('Computational Time (seconds)', fontsize=12)
ax4.set_title('Computational Efficiency Comparison', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)

for bar, t in zip(bars, times):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{t:.2f}s', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('result/hierarchical/time_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/time_comparison.pdf', format='pdf', bbox_inches='tight')
print("Visualization saved: result/hierarchical/time_comparison.png/pdf")

print("\n" + "=" * 60)
print("✅ All visualizations generated successfully!")
print("=" * 60)
print("\nGenerated files (PNG + PDF):")
print("  - cluster_centroids_comparison.png/pdf (Cluster centroids line plots)")
print("  - dendrograms_comparison.png/pdf (Dendrograms)")
print("  - metrics_comparison.png/pdf (Performance metrics bar charts)")
print("  - time_comparison.png/pdf (Computational time comparison)")

# ==========================================
# 15. 金融视角的聚类评估指标（修正版）
# ==========================================
print("\n[14] Computing financial performance metrics...")

# 从价格指数计算日收益率（用于投资组合回测）
returns_clean = price_clean.pct_change().dropna()   # shape: (天数, 股票数)
print(f"收益率数据形状: {returns_clean.shape}")

# 基准组合：等权重所有股票
benchmark_returns = returns_clean.mean(axis=1)

def build_cluster_portfolio(labels, returns_df, n_clusters):
    """
    从每个簇中选择波动率最低的股票作为代表，构建等权重组合。
    返回: (组合收益率Series, 所选股票列表)
    """
    selected_stocks = []
    for c in range(n_clusters):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cluster_returns = returns_df.loc[:, mask]
        # 选出簇内波动率最小的股票
        vol = cluster_returns.std()
        rep_stock = vol.idxmin()
        selected_stocks.append(rep_stock)
    # 构建等权重组合收益率
    port_returns = returns_df[selected_stocks].mean(axis=1)
    return port_returns, selected_stocks

# 存储金融指标
financial_results = {}

for method, scores in results.items():
    labels = scores['labels']
    unique_clusters = np.unique(labels)
    actual_n_clusters = len(unique_clusters)
    if actual_n_clusters < 2:
        print(f"  {method}: 只有 {actual_n_clusters} 个簇，跳过金融评估")
        continue

    # 构建组合
    port_returns, stocks_used = build_cluster_portfolio(labels, returns_clean, actual_n_clusters)

    # 夏普比率（年化，无风险利率=0）
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(252)

    # 多元化比率 = (加权平均波动率) / (组合波动率)
    # 权重为等权重 1/k
    k = len(stocks_used)
    weights = np.ones(k) / k
    # 所选股票各自的波动率
    indiv_vol = returns_clean[stocks_used].std()
    weighted_avg_vol = np.sum(weights * indiv_vol)
    port_vol = port_returns.std()
    div_ratio = weighted_avg_vol / port_vol

    financial_results[method] = {
        'Sharpe Ratio': sharpe,
        'Diversification Ratio': div_ratio,
        'Num Clusters Used': actual_n_clusters
    }
    print(f"  {method}: Sharpe={sharpe:.4f}, DivRatio={div_ratio:.4f} (clusters={actual_n_clusters})")

# 基准指标
bench_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252)
bench_indiv_vol = returns_clean.std().mean()      # 所有股票的平均波动率
bench_port_vol = benchmark_returns.std()
bench_div_ratio = bench_indiv_vol / bench_port_vol
print(f"\n  Benchmark (Equal-Weight): Sharpe={bench_sharpe:.4f}, DivRatio={bench_div_ratio:.4f}")

# 输出对比表格
print("\n" + "=" * 70)
print("Financial Performance Comparison (vs Equal-Weight Benchmark)")
print("=" * 70)
print(f"{'Method':<20} {'Sharpe Ratio':<15} {'Diversification Ratio':<22} {'Clusters Used':<15}")
print("-" * 70)
for method, metrics in financial_results.items():
    print(f"{method:<20} {metrics['Sharpe Ratio']:.4f}          {metrics['Diversification Ratio']:.4f}               {metrics['Num Clusters Used']}")
print(f"{'Benchmark (EW)':<20} {bench_sharpe:.4f}          {bench_div_ratio:.4f}               -")
print("=" * 70)

# 可视化金融指标对比
fig5, ax5 = plt.subplots(figsize=(10, 5))
methods_fin = list(financial_results.keys())
sharpe_vals = [financial_results[m]['Sharpe Ratio'] for m in methods_fin]
div_vals = [financial_results[m]['Diversification Ratio'] for m in methods_fin]

x = np.arange(len(methods_fin))
width = 0.35

ax5.bar(x - width/2, sharpe_vals, width, label='Sharpe Ratio', color='#4C72B0')
ax5.bar(x + width/2, div_vals, width, label='Diversification Ratio', color='#DD8452')
ax5.axhline(y=bench_sharpe, color='#4C72B0', linestyle='--', alpha=0.7,
            label=f'Benchmark Sharpe = {bench_sharpe:.3f}')
ax5.axhline(y=bench_div_ratio, color='#DD8452', linestyle='--', alpha=0.7,
            label=f'Benchmark DivRatio = {bench_div_ratio:.3f}')
ax5.set_xticks(x)
ax5.set_xticklabels(methods_fin, rotation=45, ha='right')
ax5.set_ylabel('Score')
ax5.set_title('Financial Metrics Comparison (Higher is Better)')
# 图例放到右上角外部，避免与基准线重叠
ax5.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True, fancybox=True)
plt.tight_layout()
plt.savefig('result/hierarchical/financial_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/financial_metrics_comparison.pdf', format='pdf', bbox_inches='tight')
print("Financial metrics chart saved: result/hierarchical/financial_metrics_comparison.png/pdf")