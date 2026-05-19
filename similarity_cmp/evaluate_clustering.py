# evaluate_clustering.py
"""
读取保存的聚类结果，计算各项指标
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform

print("=" * 60)
print("Clustering Evaluation (Loading Saved Results)")
print("=" * 60)

# ==========================================
# 1. 加载保存的结果
# ==========================================
print("\n[1] Loading saved clustering results...")
with open('clustering_results.pkl', 'rb') as f:
    saved = pickle.load(f)

results = saved['results']
stock_names = saved['stock_names']
X_scaled = saved['X_scaled']
price_clean = saved['price_clean']
dates = saved['dates']

print(f"Loaded {len(results)} methods")
print(f"Data shape: {X_scaled.shape}")

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

methods_to_plot = ['Euclidean', 'Pearson', 'DTW', 'Isolation Kernel']

# 计算时间轴
if USE_RECENT_YEAR:
    n_days = time_window if isinstance(time_window, int) else 250
    time_points = np.arange(1, n_days + 1)
else:
    n_days = X_scaled.shape[1]
    time_points = np.arange(1, n_days + 1)

# 采样参数
MAX_LINES_TOTAL = 100
np.random.seed(42)

# ==========================================
# 情况1：包含所有簇（最大簇用橙黄色突出）
# ==========================================
print("  Generating plots WITH major cluster...")

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Stock Price Index Curves Colored by Cluster Assignment', fontsize=14, fontweight='bold', y=0.98)

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes1[row, col]
    
    labels = results[method]['labels']
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    
    # 为每个簇分配颜色（最大簇为橙黄色）
    other_colors = [c for c in cluster_colors if c != '#FF8C00']
    label_to_color = {}
    other_idx = 0
    for label in unique_labels:
        if label == largest_cluster:
            label_to_color[label] = '#FF8C00'
        else:
            label_to_color[label] = other_colors[other_idx % len(other_colors)]
            other_idx += 1
    
    # 按簇分组采样
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    sample_indices = []
    
    for label in unique_labels:
        cluster_size = len(cluster_indices[label])
        if label == largest_cluster:
            n_sample = min(cluster_size, int(MAX_LINES_TOTAL * 0.6))
        else:
            other_total = sum(len(cluster_indices[l]) for l in unique_labels if l != largest_cluster)
            if other_total > 0:
                n_sample = min(cluster_size, max(1, int(MAX_LINES_TOTAL * 0.4 * cluster_size / other_total)))
            else:
                n_sample = min(cluster_size, 3)
        sampled = np.random.choice(cluster_indices[label], n_sample, replace=False)
        sample_indices.extend(sampled)
    
    if len(sample_indices) > MAX_LINES_TOTAL:
        sample_indices = np.random.choice(sample_indices, MAX_LINES_TOTAL, replace=False)
    
    # 绘制曲线
    for i in sample_indices:
        color = label_to_color[labels[i]]
        if labels[i] == largest_cluster:
            ax.plot(time_points, X_scaled[i, :], color=color, linewidth=1.2, alpha=0.5)
        else:
            ax.plot(time_points, X_scaled[i, :], color=color, linewidth=0.6, alpha=0.5)
    
    # 图例
    legend_handles = []
    for label in unique_labels:
        count = np.sum(labels == label)
        color = label_to_color[label]
        if label == largest_cluster:
            label_name = f'Major Cluster (n={count})'
        else:
            label_name = f'Cluster {label+1} (n={count})'
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label_name))
    ax.legend(handles=legend_handles, loc='best', frameon=True, fancybox=True, fontsize=8)
    
    ax.set_title(f'{method}', fontweight='bold')
    ax.set_xlabel('Trading Day', fontsize=10)
    ax.set_ylabel('Standardized Price Index', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_xlim(1, n_days)

plt.tight_layout()
plt.savefig('result/hierarchical/cluster_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/cluster_curves_comparison.pdf', format='pdf', bbox_inches='tight')
print("  Saved: cluster_curves_comparison.png/pdf")

# ==========================================
# 情况2：排除最大簇，只画其他簇
# ==========================================
print("  Generating plots WITHOUT major cluster...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Stock Price Index Curves Colored by Cluster Assignment (Excluding Major Cluster)', fontsize=14, fontweight='bold', y=0.98)

for idx, method in enumerate(methods_to_plot):
    row = idx // 2
    col = idx % 2
    ax = axes2[row, col]
    
    labels = results[method]['labels']
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    major_count = np.sum(labels == largest_cluster)
    
    # 只取非最大簇
    other_labels = [l for l in unique_labels if l != largest_cluster]
    
    # 为其他簇分配颜色
    label_to_color = {}
    for i, label in enumerate(other_labels):
        label_to_color[label] = cluster_colors[i % len(cluster_colors)]
    
    # 按簇分组采样
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
    
    # 图例
    legend_handles = []
    for label in other_labels:
        count = np.sum(labels == label)
        color = label_to_color[label]
        label_name = f'Cluster {label+1} (n={count})'
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label_name))
    ax.legend(handles=legend_handles, loc='best', frameon=True, fancybox=True, fontsize=8)
    
    ax.set_title(f'{method} (Major Cluster n={major_count} excluded)', fontweight='bold')
    ax.set_xlabel('Trading Day', fontsize=10)
    ax.set_ylabel('Standardized Price Index', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_xlim(1, n_days)

plt.tight_layout()
plt.savefig('result/hierarchical/cluster_curves_comparison_no_major.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/cluster_curves_comparison_no_major.pdf', format='pdf', bbox_inches='tight')
print("  Saved: cluster_curves_comparison_no_major.png/pdf")

print("  All cluster curve plots generated successfully!")


# ==========================================
# 12. 树状图展示（学术风格）
# ==========================================
print("\n[11] Generating dendrograms...")

from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

# 设置树状图颜色
set_link_color_palette(['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3'])

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Hierarchical Clustering Dendrograms', fontsize=14, fontweight='bold', y=0.98)

n_display = min(300, len(stock_names))

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
# 15. 金融视角的聚类评估指标（改进版）
# ==========================================
print("\n[14] Computing clustering quality metrics...")

# 从价格指数计算日收益率
returns_clean = price_clean.pct_change().dropna()
print(f"收益率数据形状: {returns_clean.shape}")

def evaluate_clustering_quality(labels, returns_df):
    """
    综合评估聚类质量，不依赖选股策略
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # 1. 簇内/簇间相关性分析
    intra_corrs = []
    inter_corrs = []
    
    for c in unique_clusters:
        mask = (labels == c)
        if mask.sum() < 2:
            continue
        cluster_returns = returns_df.loc[:, mask]
        # 簇内平均相关性（添加 .copy()）
        corr_matrix = cluster_returns.corr().values.copy()
        np.fill_diagonal(corr_matrix, np.nan)
        intra_corrs.append(np.nanmean(corr_matrix))
    
    for i in range(len(unique_clusters)):
        for j in range(i+1, len(unique_clusters)):
            mask_i = (labels == unique_clusters[i])
            mask_j = (labels == unique_clusters[j])
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                continue
            # 簇间相关性（使用簇中心）
            center_i = returns_df.loc[:, mask_i].mean(axis=1)
            center_j = returns_df.loc[:, mask_j].mean(axis=1)
            inter_corrs.append(np.corrcoef(center_i, center_j)[0, 1])
    
    intra_corr = np.mean(intra_corrs) if intra_corrs else 0
    inter_corr = np.mean(inter_corrs) if inter_corrs else 0
    
    # 2. 构建簇内等权重组合
    portfolio_returns = pd.Series(0, index=returns_df.index)
    for c in unique_clusters:
        mask = (labels == c)
        cluster_returns = returns_df.loc[:, mask]
        if cluster_returns.empty:
            continue
        cluster_weight = 1 / n_clusters
        cluster_port = cluster_returns.mean(axis=1)
        portfolio_returns += cluster_port * cluster_weight
    
    sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    
    # 3. 多元化比率
    cluster_vols = []
    for c in unique_clusters:
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cluster_port = returns_df.loc[:, mask].mean(axis=1)
        cluster_vols.append(cluster_port.std())
    
    weighted_avg_vol = np.mean(cluster_vols) if cluster_vols else 0
    port_vol = portfolio_returns.std()
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1
    
    return {
        'Sharpe Ratio': sharpe,
        'Diversification Ratio': div_ratio,
        'Intra-cluster Correlation': intra_corr,
        'Inter-cluster Correlation': inter_corr,
        'Cluster Separation': inter_corr - intra_corr,
        'Num Clusters': n_clusters
    }


# 存储评估结果
financial_results = {}

for method, scores in results.items():
    labels = scores['labels']
    quality = evaluate_clustering_quality(labels, returns_clean)
    financial_results[method] = quality
    print(f"  {method}: Sharpe={quality['Sharpe Ratio']:.4f}, "
          f"DivRatio={quality['Diversification Ratio']:.4f}, "
          f"IntraCorr={quality['Intra-cluster Correlation']:.3f}, "
          f"InterCorr={quality['Inter-cluster Correlation']:.3f}")

# 输出对比表格
print("\n" + "=" * 80)
print("Clustering Quality Assessment (Full Portfolio Evaluation)")
print("=" * 80)
print(f"{'Method':<18} {'Sharpe':<8} {'DivRatio':<10} {'IntraCorr':<10} {'InterCorr':<10} {'Separation':<12}")
print("-" * 80)
for method, metrics in financial_results.items():
    print(f"{method:<18} {metrics['Sharpe Ratio']:.3f}   {metrics['Diversification Ratio']:.3f}     "
          f"{metrics['Intra-cluster Correlation']:.3f}     {metrics['Inter-cluster Correlation']:.3f}     "
          f"{metrics['Cluster Separation']:.3f}")
print("=" * 80)
# ==========================================
# 15.5 辅助函数：Entanglement 和 Dendrogram Purity
# ==========================================

def compute_entanglement(linkage_matrix, labels):
    """
    计算纠缠度 (Entanglement)
    参考: Han et al. (2023) - The impact of isolation kernel on agglomerative hierarchical clustering
    
    参数:
        linkage_matrix: 层次聚类的链接矩阵
        labels: 真实标签（如行业标签）
    
    返回:
        entanglement: 纠缠度，越低越好
    """
    from collections import defaultdict, Counter
    n = len(labels)
    
    # 构建合并历史的数据结构
    class Node:
        def __init__(self, idx, size=1):
            self.idx = idx
            self.left = None
            self.right = None
            self.size = size
            self.label_counts = defaultdict(int)
            self.entanglement = 0
            self.depth = 0
    
    # 初始化叶子节点
    nodes = [Node(i) for i in range(n)]
    for i, label in enumerate(labels):
        nodes[i].label_counts[label] = 1
    
    # 构建树
    for i, merge in enumerate(linkage_matrix):
        left_idx = int(merge[0])
        right_idx = int(merge[1])
        
        left_node = nodes[left_idx]
        right_node = nodes[right_idx]
        
        # 创建新节点
        new_node = Node(n + i, left_node.size + right_node.size)
        new_node.left = left_node
        new_node.right = right_node
        
        # 合并标签计数
        new_node.label_counts = left_node.label_counts.copy()
        for label, count in right_node.label_counts.items():
            new_node.label_counts[label] += count
        
        # 找到主导标签
        dominant_label = max(new_node.label_counts.items(), key=lambda x: x[1])[0]
        dominant_count = new_node.label_counts[dominant_label]
        
        # 计算纠缠：左右子节点中非主导标签的比例
        left_impurity = 1 - (left_node.label_counts.get(dominant_label, 0) / left_node.size) if left_node.size > 0 else 0
        right_impurity = 1 - (right_node.label_counts.get(dominant_label, 0) / right_node.size) if right_node.size > 0 else 0
        
        new_node.entanglement = (left_impurity + right_impurity) / 2
        
        nodes.append(new_node)
    
    # 计算加权平均纠缠度
    root = nodes[-1]
    total_entanglement = 0
    total_weight = 0
    
    def accumulate(node):
        nonlocal total_entanglement, total_weight
        weight = node.size
        total_entanglement += node.entanglement * weight
        total_weight += weight
        if node.left:
            accumulate(node.left)
        if node.right:
            accumulate(node.right)
    
    accumulate(root)
    
    return total_entanglement / total_weight if total_weight > 0 else 0


def compute_dendrogram_purity(linkage_matrix, labels):
    """
    计算树状图纯度 (Dendrogram Purity)
    参考: Heller & Ghahramani (2005) - Bayesian hierarchical clustering
    
    对于同一真实类别的每对点，找到它们在树状图中首次合并的节点，
    计算该节点子树中属于该类别的比例。
    
    参数:
        linkage_matrix: 层次聚类的链接矩阵
        labels: 真实标签（如行业标签）
    
    返回:
        purity: 树状图纯度，越高越好
    """
    from collections import defaultdict
    n = len(labels)
    
    # 构建父子关系
    parent = {}
    children = {}
    node_labels = {}
    node_size = {}
    
    # 初始化叶子节点
    for i in range(n):
        node_labels[i] = defaultdict(int)
        node_labels[i][labels[i]] = 1
        node_size[i] = 1
        children[i] = []
    
    # 处理合并
    for i, merge in enumerate(linkage_matrix):
        node_id = n + i
        left = int(merge[0])
        right = int(merge[1])
        
        parent[left] = node_id
        parent[right] = node_id
        children[node_id] = [left, right]
        
        # 合并标签计数
        node_labels[node_id] = node_labels[left].copy()
        for label, count in node_labels[right].items():
            node_labels[node_id][label] = node_labels[node_id].get(label, 0) + count
        node_size[node_id] = node_size[left] + node_size[right]
    
    # 计算纯度
    total_purity = 0
    pair_count = 0
    
    # 对每个真实类别
    label_groups = defaultdict(list)
    for i, label in enumerate(labels):
        label_groups[label].append(i)
    
    # 预计算 LCA（使用深度优先搜索预处理）
    # 简化版本：直接查找每对点的 LCA
    for label, members in label_groups.items():
        if len(members) < 2:
            continue
        
        # 对于少量点，直接计算
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                a, b = members[i], members[j]
                
                # 找到 LCA
                # 构建 a 的路径
                path_a = set()
                curr = a
                while curr in parent:
                    path_a.add(curr)
                    curr = parent[curr]
                path_a.add(curr)
                
                # 找到 b 的最近公共祖先
                curr = b
                while curr not in path_a:
                    curr = parent[curr]
                lca = curr
                
                # 计算纯度
                label_count = node_labels[lca].get(label, 0)
                total_in_subtree = node_size[lca]
                purity = label_count / total_in_subtree if total_in_subtree > 0 else 0
                
                total_purity += purity
                pair_count += 1
    
    return total_purity / pair_count if pair_count > 0 else 0

# ==========================================
# 16. 层次聚类质量评估：Entanglement & Dendrogram Purity
# ==========================================
print("\n[15] Computing dendrogram quality metrics (Entanglement & Purity)...")

# 加载行业标签数据
print("  Loading industry labels from SP500_Sector.csv...")
sector_df = pd.read_csv('dataset/SP500_Sector.csv')

# 创建股票代码到行业的映射字典
ticker_to_sector = dict(zip(sector_df['Symbol'], sector_df['GICS Sector']))
print(f"  Loaded {len(ticker_to_sector)} tickers with GICS Sector labels")
print(f"  GICS Sectors: {sector_df['GICS Sector'].nunique()} categories")

# 将股票名称映射到行业
print("  Mapping stocks to GICS sectors...")
industry_labels = []
missing_tickers = []

for stock in stock_names:
    if stock in ticker_to_sector:
        industry_labels.append(ticker_to_sector[stock])
    else:
        industry_labels.append('Unknown')
        missing_tickers.append(stock)

print(f"  Successfully mapped: {len([l for l in industry_labels if l != 'Unknown'])}/{len(stock_names)}")
if missing_tickers:
    print(f"  Missing tickers (first 10): {missing_tickers[:10]}")

# 编码为数值
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
industry_encoded = le.fit_transform(industry_labels)
print(f"  Encoded into {len(le.classes_)} classes (including 'Unknown')")

# 显示行业分布
print("\n  GICS Sector distribution in dataset:")
sector_counts = pd.Series(industry_labels).value_counts()
for sector, count in sector_counts.head(10).items():
    print(f"    {sector}: {count}")

# 存储层次聚类质量指标
hierarchical_quality = {}

for method, scores in results.items():
    linkage_matrix = scores['linkage']
    
    # 使用真实行业标签计算纠缠度和纯度
    entanglement = compute_entanglement(linkage_matrix, industry_encoded)
    dendrogram_purity = compute_dendrogram_purity(linkage_matrix, industry_encoded)
    
    hierarchical_quality[method] = {
        'Entanglement': entanglement,
        'Dendrogram Purity': dendrogram_purity
    }
    print(f"  {method}: Entanglement={entanglement:.4f}, Dendrogram Purity={dendrogram_purity:.4f}")

print("\n" + "=" * 70)
print("Hierarchical Clustering Quality Assessment")
print("(Based on GICS Sector ground truth labels)")
print("=" * 70)
print(f"{'Method':<20} {'Entanglement (↓)':<20} {'Dendrogram Purity (↑)':<25}")
print("-" * 70)
for method, metrics in hierarchical_quality.items():
    print(f"{method:<20} {metrics['Entanglement']:.4f}               {metrics['Dendrogram Purity']:.4f}")
print("=" * 70)

# 可视化
fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))

methods_hier = list(hierarchical_quality.keys())
entanglement_vals = [hierarchical_quality[m]['Entanglement'] for m in methods_hier]
purity_vals = [hierarchical_quality[m]['Dendrogram Purity'] for m in methods_hier]

# 纠缠度图（越低越好）
bars1 = axes6[0].bar(methods_hier, entanglement_vals, color=['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00'])
axes6[0].set_ylabel('Entanglement (Lower is Better)', fontsize=11)
axes6[0].set_title('Entanglement Score', fontweight='bold')
axes6[0].grid(axis='y', alpha=0.3)
for bar, v in zip(bars1, entanglement_vals):
    axes6[0].text(bar.get_x() + bar.get_width()/2., v + 0.01, f'{v:.4f}', ha='center', fontsize=9)

# 树状图纯度（越高越好）
bars2 = axes6[1].bar(methods_hier, purity_vals, color=['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00'])
axes6[1].set_ylabel('Dendrogram Purity (Higher is Better)', fontsize=11)
axes6[1].set_title('Dendrogram Purity', fontweight='bold')
axes6[1].grid(axis='y', alpha=0.3)
for bar, v in zip(bars2, purity_vals):
    axes6[1].text(bar.get_x() + bar.get_width()/2., v + 0.02, f'{v:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('result/hierarchical/hierarchical_quality.png', dpi=300, bbox_inches='tight')
plt.savefig('result/hierarchical/hierarchical_quality.pdf', format='pdf', bbox_inches='tight')
print("\nHierarchical quality chart saved: result/hierarchical/hierarchical_quality.png/pdf")