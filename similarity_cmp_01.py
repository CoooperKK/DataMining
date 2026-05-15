import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.metrics import dtw_path

# 解决画图中文乱码问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 可选：设置时间窗口大小（None表示使用全部数据）
# ==========================================
TIME_WINDOW = None  # 例如: 500 表示只使用前500个交易日
# TIME_WINDOW = 500  # 取消注释启用

# ==========================================
# Step 1: 数据加载与对数收益率转换
# ==========================================
print("Step 1: 正在解析自定义数据...")

# 读取数据
raw_df = pd.read_csv('./dataset/stock_data_sp500.csv')

# 1. 提取行业信息
industries = raw_df.iloc[0, 2:].to_dict()

# 2. 提取涨跌幅数据
returns_df = raw_df.iloc[1:, 2:].astype(float).dropna(axis=1, how='all')
stock_names = returns_df.columns.values


# ----- 添加这行：只看最近一年约 250 个交易日 -----
returns_df = returns_df.iloc[-250:, :]  # 或使用 -252（一年交易日约 252 天）
# -------------------------------------------

print(f"原始数据形状: {returns_df.shape}")

# ----- 修改点 1：安全处理原始涨跌幅，避免 log(0) -----
# 将小于 -0.999 的值限制为 -0.999
safe_returns = returns_df.clip(lower=-0.999, upper=None)
# --------------------------------------------------

# 转换为对数收益率
print(" -> 转换为对数收益率...")
log_returns_df = np.log(1 + safe_returns)   # 注意变量名已改

# 可选：应用时间窗口
if TIME_WINDOW is not None:
    log_returns_df = log_returns_df.iloc[:TIME_WINDOW, :]
    print(f" -> 应用时间窗口: 前 {TIME_WINDOW} 个交易日")

# 转置：行=股票，列=交易日
X_raw = log_returns_df.T.values

# ----- 修改点 2：最终清理 NaN/Inf -----
nan_inf_count = np.isnan(X_raw).sum() + np.isinf(X_raw).sum()
if nan_inf_count > 0:
    print(f" -> 发现无效数值 {nan_inf_count} 个，将填充为 0")
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
# ------------------------------------

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"解析完毕：共有 {len(stock_names)} 只股票，{X_scaled.shape[1]} 个交易日数据。")

# ==========================================
# Step 2: 多算法对比实验
# ==========================================
n_clusters = 5
results = {}

# ==========================================
# 1. Euclidean Distance - KMeans
# ==========================================
print(" -> 正在运行 Euclidean K-Means...")
km_ed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_ed = km_ed.fit_predict(X_scaled)
results['Euclidean'] = silhouette_score(X_scaled, labels_ed)
print(f"    Euclidean 轮廓系数: {results['Euclidean']:.4f}")

# ==========================================
# 2. Pearson Correlation - 层次聚类（正确实现）
# ==========================================
print(" -> 正在运行 Pearson Correlation...")
# 计算相关系数矩阵
corr_matrix = np.corrcoef(X_scaled)
# 处理可能的 NaN
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
# 将相关系数转换为距离：距离 = 1 - |相关系数| 或 1 - 相关系数
# 使用 1 - |r| 可以同时捕捉正负相关
dist_pearson = 1 - np.abs(corr_matrix)
# 确保距离矩阵对称且非负
dist_pearson = (dist_pearson + dist_pearson.T) / 2
np.fill_diagonal(dist_pearson, 0)

# 使用层次聚类
agglo = AgglomerativeClustering(
    n_clusters=n_clusters, 
    metric='precomputed', 
    linkage='average'
)
labels_pearson = agglo.fit_predict(dist_pearson)
results['Pearson'] = silhouette_score(dist_pearson, labels_pearson, metric='precomputed')
print(f"    Pearson 轮廓系数: {results['Pearson']:.4f}")

# ==========================================
# 3. DTW (Dynamic Time Warping)
# ==========================================
print(" -> 正在运行 DTW...")
t0 = time.time()

# 使用 PAA 降维
n_paa_segments = min(100, X_scaled.shape[1])
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
X_paa = paa.fit_transform(X_scaled)

print(f"    PAA 降维: {X_scaled.shape[1]} -> {n_paa_segments} 维")

# DTW K-Means 聚类
km_dtw = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    max_iter=5,
    metric_params={
        "global_constraint": "sakoe_chiba",
        "sakoe_chiba_radius": 3  # 放宽窗口，提高准确性
    },
    n_jobs=1,
    random_state=42
)

labels_dtw = km_dtw.fit_predict(X_paa)
results['DTW'] = silhouette_score(X_scaled, labels_dtw)
print(f"    DTW 轮廓系数: {results['DTW']:.4f}")
print(f"    DTW 耗时: {time.time()-t0:.2f} 秒")

# ==========================================
# 4. Isolation Kernel (IK)
# ==========================================
print(" -> 正在运行 Isolation Kernel...")

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
from sklearn.cluster import SpectralClustering
spec = SpectralClustering(
    n_clusters=n_clusters,
    affinity='precomputed',
    random_state=42,
    assign_labels='kmeans'
)
labels_ik = spec.fit_predict(sim_ik)
results['Isolation Kernel'] = silhouette_score(dist_ik, labels_ik, metric='precomputed')
print(f"    Isolation Kernel 轮廓系数: {results['Isolation Kernel']:.4f}")

# ==========================================
# Step 3: 结果输出
# ==========================================
print("\n" + "="*50)
print("聚类评估结果 (轮廓系数)")
print("="*50)
print(f"{'相似度度量':<20} {'轮廓系数':>12}")
print("-"*50)
for method, score in results.items():
    # 添加评价标记
    if score >= 0.3:
        flag = "★ 良好"
    elif score >= 0.15:
        flag = "○ 可接受"
    elif score >= 0:
        flag = "△ 较弱"
    else:
        flag = "✗ 无效"
    print(f"{method:<20} {score:>10.4f}   {flag}")
print("="*50)

# 找出最佳方法
best_method = max(results, key=results.get)
print(f"\n最佳相似度度量: {best_method} (轮廓系数 = {results[best_method]:.4f})")

# ==========================================
# 可选：可视化（简化版）
# ==========================================
try:
    # PCA降维可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # 使用最佳方法的聚类结果
    if best_method == 'Euclidean':
        best_labels = labels_ed
    elif best_method == 'Pearson':
        best_labels = labels_pearson
    elif best_method == 'DTW':
        best_labels = labels_dtw
    else:
        best_labels = labels_ik
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='tab20', s=20, alpha=0.7)
    plt.title(f'PCA可视化 (聚类方法: {best_method})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter, label='簇标签')
    
    # 热力图（相关性矩阵按聚类排序）
    plt.subplot(1, 2, 2)
    sorted_idx = np.argsort(best_labels)
    corr_sorted = np.corrcoef(X_scaled[sorted_idx])
    sns.heatmap(corr_sorted, cmap='RdYlBu_r', center=0, cbar_kws={'label': '相关系数'})
    plt.title('按聚类排序的相关性热力图')
    
    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存: clustering_results.png")
    plt.show()
except Exception as e:
    print(f"\n可视化失败: {e}")

print("\n✅ 程序运行完成！")




# 检查数据是否有重复
df = pd.read_csv('./dataset/stock_data_sp500.csv')
returns = df.iloc[1:, 2:].astype(float)
print(f"数据形状: {returns.shape}")

# 检查每只股票是否与其他股票完全相同
unique_series = []
for col in returns.columns:
    is_dup = any(returns[col].equals(returns[other]) for other in returns.columns if other != col)
    if is_dup:
        print(f"发现重复: {col}")

# 检查方差是否过小（可能全是0）
var = returns.var()
print(f"方差<=0.001的股票数: {(var <= 0.001).sum()}")

# 检查两两相关系数分布
corr = returns.corr().values.copy()  # 添加 .copy()
np.fill_diagonal(corr, np.nan)
print(f"平均相关系数: {np.nanmean(corr):.4f}")
print(f"相关系数标准差: {np.nanstd(corr):.4f}")