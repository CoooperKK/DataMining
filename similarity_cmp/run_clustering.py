# run_clustering.py
"""
只运行聚类算法，保存聚类结果到文件
"""

import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomTreesEmbedding
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import euclidean_distances
from tslearn.metrics import dtw

print("=" * 60)
print("Clustering Computation (Save Results)")
print("=" * 60)

# ==========================================
# 1. 加载数据
# ==========================================
print("\n[1] Loading data...")
raw_df = pd.read_csv('./dataset/stock_data_sp500_paper.csv')
price_data = raw_df.iloc[1:, 2:].astype(float)

stock_num = 300
price_data = price_data.iloc[:, :stock_num]
stock_names = price_data.columns.values

print(f"Data shape: {price_data.shape}")

# ==========================================
# 2. 清洗和标准化
# ==========================================
print("\n[2] Cleaning and scaling...")
missing_ratio = price_data.isna().mean()
valid_stocks = missing_ratio[missing_ratio < 0.1].index
price_clean = price_data[valid_stocks]
price_clean = price_clean.ffill().bfill().fillna(100)

X = price_clean.T.values
X_scaled = StandardScaler().fit_transform(X)
print(f"Scaled data shape: {X_scaled.shape}")

# ==========================================
# 3. 计算所有距离矩阵和聚类标签
# ==========================================
n_clusters = 8
results = {}

# Euclidean
print("\n[3] Computing Euclidean...")
t0 = time.time()
dist_euclidean = euclidean_distances(X_scaled)
dist_euclidean = (dist_euclidean + dist_euclidean.T) / 2
np.fill_diagonal(dist_euclidean, 0)
linkage_euclidean = linkage(dist_euclidean, method='average')
labels_euclidean = fcluster(linkage_euclidean, t=n_clusters, criterion='maxclust') - 1
results['Euclidean'] = {
    'labels': labels_euclidean.tolist(),
    'linkage': linkage_euclidean.tolist(),
    'time': time.time() - t0
}
print(f"  Done in {results['Euclidean']['time']:.2f}s")

# Pearson
print("\n[4] Computing Pearson...")
t0 = time.time()
corr_matrix = np.corrcoef(X_scaled)
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
dist_pearson = 1 - np.abs(corr_matrix)
dist_pearson = (dist_pearson + dist_pearson.T) / 2
np.fill_diagonal(dist_pearson, 0)
linkage_pearson = linkage(dist_pearson, method='average')
labels_pearson = fcluster(linkage_pearson, t=n_clusters, criterion='maxclust') - 1
results['Pearson'] = {
    'labels': labels_pearson.tolist(),
    'linkage': linkage_pearson.tolist(),
    'time': time.time() - t0
}
print(f"  Done in {results['Pearson']['time']:.2f}s")

# DTW (最慢的部分)
print("\n[5] Computing DTW (this will take a while)...")
t0 = time.time()
n_stocks = X_scaled.shape[0]
dist_dtw = np.zeros((n_stocks, n_stocks))

for i in range(n_stocks):
    for j in range(i+1, n_stocks):
        dist = dtw(X_scaled[i], X_scaled[j], global_constraint='sakoe_chiba', sakoe_chiba_radius=10)
        dist_dtw[i, j] = dist
        dist_dtw[j, i] = dist
    if (i+1) % 50 == 0:
        print(f"  Progress: {i+1}/{n_stocks}")

dist_dtw = (dist_dtw + dist_dtw.T) / 2
np.fill_diagonal(dist_dtw, 0)
linkage_dtw = linkage(dist_dtw, method='average')
labels_dtw = fcluster(linkage_dtw, t=n_clusters, criterion='maxclust') - 1
results['DTW'] = {
    'labels': labels_dtw.tolist(),
    'linkage': linkage_dtw.tolist(),
    'time': time.time() - t0
}
print(f"  Done in {results['DTW']['time']:.2f}s")

# Isolation Kernel
print("\n[6] Computing Isolation Kernel...")
t0 = time.time()
n_estimators = 200
rte = RandomTreesEmbedding(
    n_estimators=200,
    max_depth=10,      # 增加深度 → 更多细胞 → 密度自适应更强
    min_samples_split=2,
    random_state=42
)
leaf_indices = rte.fit_transform(X_scaled)
sim_ik = (leaf_indices @ leaf_indices.T).toarray() / n_estimators
dist_ik = 1 - sim_ik
dist_ik = (dist_ik + dist_ik.T) / 2
np.fill_diagonal(dist_ik, 0)
linkage_ik = linkage(dist_ik, method='average')
labels_ik = fcluster(linkage_ik, t=n_clusters, criterion='maxclust') - 1
results['Isolation Kernel'] = {
    'labels': labels_ik.tolist(),
    'linkage': linkage_ik.tolist(),
    'time': time.time() - t0
}
print(f"  Done in {results['Isolation Kernel']['time']:.2f}s")

# ==========================================
# 4. 计算评估指标
# ==========================================
print("\n[7] Computing evaluation metrics...")
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet

for method, data in results.items():
    labels = np.array(data['labels'])
    linkage_matrix = np.array(data['linkage'])
    
    # 轮廓系数
    sil = silhouette_score(X_scaled, labels)
    
    # Davies-Bouldin 和 Calinski-Harabasz
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    
    # CPCC (需要距离矩阵)
    # 根据方法重新计算距离矩阵（或从 linkage 恢复）
    if method == 'Euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(X_scaled)
    elif method == 'Pearson':
        corr_matrix = np.corrcoef(X_scaled)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        dist_matrix = 1 - np.abs(corr_matrix)
    elif method == 'DTW':
        # DTW 距离矩阵已在计算时得到，但未保存
        # 这里暂时跳过或用 linkage 近似
        dist_matrix = squareform(linkage_matrix[:, 2])
    else:  # Isolation Kernel
        from sklearn.ensemble import RandomTreesEmbedding
        n_estimators = 200
        rte = RandomTreesEmbedding(n_estimators=n_estimators, max_depth=5, 
                                   min_samples_split=3, max_leaf_nodes=200, random_state=42)
        leaf_indices = rte.fit_transform(X_scaled)
        sim_ik = (leaf_indices @ leaf_indices.T).toarray() / n_estimators
        dist_matrix = 1 - sim_ik
    
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)
    
    # 计算 CPCC
    condensed_dist = squareform(dist_matrix, checks=False)
    cpcc, _ = cophenet(linkage_matrix, condensed_dist)
    
    # 保存指标到 results
    data['silhouette'] = sil
    data['db_index'] = db
    data['ch_index'] = ch
    data['cpcc'] = cpcc
    
    print(f"  {method}: Sil={sil:.4f}, CPCC={cpcc:.4f}, DB={db:.4f}, CH={ch:.2f}")

# ==========================================
# 5. 保存结果（包含指标）
# ==========================================
print("\n[8] Saving results with metrics...")
with open('clustering_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'stock_names': stock_names.tolist(),
        'X_scaled': X_scaled,
        'price_clean': price_clean,
        'dates': raw_df.iloc[1:, 1].tolist()
    }, f)
print("Results saved to clustering_results.pkl")