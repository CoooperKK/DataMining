import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans

from tslearn.piecewise import PiecewiseAggregateApproximation


# 解决画图中文乱码问题
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Step 1: 数据加载与格式适配 (Data Loading)
# ==========================================
print("Step 1: 正在解析自定义数据...")

# 读取数据
# 注意：根据你的描述，第一行是代码，第二行是行业
raw_df = pd.read_csv('stock_data.csv')

# 1. 提取行业信息 (索引为0的那行)
# 假设列名为 '时间', '云南白药'...
# 我们要把这行数据取出来，作为后续验证使用
industries = raw_df.iloc[0, 2:].to_dict() # 股票代码 -> 行业

# 2. 提取涨跌幅数据 (索引从1开始的部分)
# 去掉前两列 (序号和时间)
returns_df = raw_df.iloc[1:, 2:].astype(float).dropna(axis=1, how='all')
stock_names = returns_df.columns.values

# 3. 数据预处理
# 转置使得 行=股票，列=交易日
X_raw = returns_df.T.values
# 标准化 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"解析完毕：共有 {len(stock_names)} 只股票，{X_scaled.shape[1]} 个交易日数据。")

# ==========================================
# Step 2: 多算法对比实验 (Experiments)
# ==========================================
n_clusters = 8 # 建议簇数设多一点，因为行业比较多
results = {}

# --- 1. Euclidean ---
print(" -> 正在运行 Euclidean K-Means...")
km_ed = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_ed = km_ed.fit_predict(X_scaled)
results['Euclidean'] = {'labels': labels_ed, 'score': silhouette_score(X_scaled, labels_ed)}

# --- 2. Pearson (1 - Correlation) ---
print(" -> 正在运行 Pearson Correlation...")
corr_matrix = np.corrcoef(X_scaled)
corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
dist_pearson = 1 - corr_matrix
labels_p = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average').fit_predict(dist_pearson)
results['Pearson'] = {'labels': labels_p, 'score': silhouette_score(dist_pearson, labels_p, metric='precomputed')}

# --- 3. 极速降维版 DTW ---
print(f" -> 正在运行降维优化版 DTW...")
t0 = time.time()

# 【核心优化 1】：使用 PAA 将 1216 维压缩到 100 维
# 这样 DTW 计算量会缩小 (1216/100)^2 ≈ 147 倍！
n_paa_segments = 100 
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
X_paa = paa.fit_transform(X_scaled) # 压缩后的数据

# 【核心优化 2】：限制迭代次数和窗口
km_dtw = TimeSeriesKMeans(
    n_clusters=n_clusters, 
    metric="dtw", 
    max_iter=3,           # 再次缩减迭代次数
    metric_params={
        "global_constraint": "sakoe_chiba", 
        "sakoe_chiba_radius": 3  # 进一步收窄搜索窗口
    },
    n_jobs=1,             # 针对你的 Windows 警告，建议先设为 1 避免多进程开销，或设为 2
    random_state=42
)

# 运行聚类
labels_dtw = km_dtw.fit_predict(X_paa)

print(f" DTW 运行完成！耗时: {time.time()-t0:.2f} 秒")

# 评估时依然使用原始高维数据确保严谨性
results['DTW'] = {'labels': labels_dtw, 'score': silhouette_score(X_scaled, labels_dtw)}


# --- 4. Isolation Kernel (基于 Random Trees 模拟) ---
print(" -> 正在运行 Isolation Kernel (IK)...")
from sklearn.ensemble import RandomTreesEmbedding
rte = RandomTreesEmbedding(n_estimators=200, random_state=42, max_depth=5)
leaf_indices = rte.fit_transform(X_scaled)
sim_ik = (leaf_indices * leaf_indices.T).toarray() / 200.0
labels_ik = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42).fit_predict(sim_ik)
results['Isolation Kernel'] = {'labels': labels_ik, 'score': silhouette_score(1-sim_ik, labels_ik, metric='precomputed')}

# ==========================================
# Step 3: 性能看板输出
# ==========================================
print("\n" + "="*40)
print("🏆 聚类评估看板 (轮廓系数)")
for m, r in results.items(): print(f"{m:>16} : {r['score']:.4f}")
print("="*40)

# ==========================================
# Step 4: 行业交叉洞察分析 (Financial Insights)
# ==========================================
print("\n分析 IK 聚类结果中的行业构成：")
best_labels = results['Isolation Kernel']['labels']
cluster_analysis = []

for i in range(n_clusters):
    member_indices = np.where(best_labels == i)[0]
    member_names = stock_names[member_indices]
    member_industries = [industries.get(name, "未知") for name in member_names]
    
    print(f"\n[簇 Cluster {i}] (共 {len(member_names)} 只股票):")
    print(f" - 成员: {', '.join(member_names[:10])}...")
    # 统计行业分布
    ind_counts = pd.Series(member_industries).value_counts()
    print(f" - 主要行业构成: {ind_counts.index[0]} ({ind_counts.values[0]}只)")
    
    # 寻找“叛徒”（非主流行业的股票）—— 这是你 Insight 的来源
    if len(ind_counts) > 1:
        print(f" - 跨行业洞察：该簇还包含了来自 {ind_counts.index[1]} 的资产，表现出跨行业联动性。")

# ==========================================
# Step 5: 可视化
# ==========================================
# 绘制热力图展示 IK 的效果
plt.figure(figsize=(10, 8))
# 按标签排序，查看块状结构
sorted_idx = np.argsort(best_labels)
sns.heatmap(corr_matrix[sorted_idx][:, sorted_idx], cmap='RdYlBu_r', center=0)
plt.title("经过 IK 聚类重新排序后的相关性热力图")
plt.show()