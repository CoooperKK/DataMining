#!/usr/bin/env python3
"""
按真实行业分组聚类评估脚本
功能：
1. 读取 stock_data.csv，解析行业标签
2. 按行业进行分组（真实行业作为聚类标签）
3. 计算轮廓系数，评估行业分类的聚类质量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 数据加载与解析
# ==========================================
print("=" * 60)
print("按真实行业分组聚类评估")
print("=" * 60)

# 读取数据
df = pd.read_csv('stock_data.csv')

print(f"\n原始数据形状: {df.shape}")
print(f"列数（股票数量+2列）: {df.shape[1]}")

# 解析数据结构
# 第一行是股票名称（表头）
stock_names = df.columns[2:].tolist()
print(f"股票数量: {len(stock_names)}")

# 第二行是行业分类
industry_row = df.iloc[0, 2:]  # 第二行，从第三列开始
industries = {}
industry_list = []
for stock, ind in zip(stock_names, industry_row):
    if pd.notna(ind):
        industries[stock] = str(ind)
        industry_list.append(str(ind))
    else:
        industries[stock] = "未知"
        industry_list.append("未知")

unique_industries = list(set(industry_list))
print(f"行业数量: {len(unique_industries)}")
print(f"行业列表: {unique_industries[:20]}...")

# 统计各行业股票数量
industry_counts = pd.Series(industry_list).value_counts()
print(f"\n各行业股票数量统计（前10）:")
for ind, count in industry_counts.head(10).items():
    print(f"  {ind}: {count} 只")

# 从第三行开始是时间序列数据
returns_data = df.iloc[1:, 2:].astype(float)
dates = df.iloc[1:, 1].tolist()  # 日期列
print(f"\n时间序列数据形状: {returns_data.shape}")
print(f"时间范围: {dates[0]} 到 {dates[-1]}")
print(f"交易日数量: {len(dates)}")

# ==========================================
# 数据预处理
# ==========================================
print("\n" + "=" * 40)
print("数据预处理")
print("=" * 40)

# 转置：每行是一只股票，每列是一个交易日
X_raw = returns_data.T.values
print(f"原始数据矩阵: {X_raw.shape[0]} 只股票 × {X_raw.shape[1]} 个交易日")

# 检查缺失值
nan_counts = np.isnan(X_raw).sum(axis=1)
print(f"缺失值统计: 均值={nan_counts.mean():.1f}, 最大={nan_counts.max()}")
if nan_counts.any():
    # 用向前填充处理缺失值
    X_raw = pd.DataFrame(X_raw).fillna(method='ffill').fillna(method='bfill').values

# 标准化 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"数据标准化完成")

# ==========================================
# 将行业标签编码为数值
# ==========================================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
industry_labels = le.fit_transform(industry_list)
n_industry_clusters = len(le.classes_)
print(f"\n行业类别数: {n_industry_clusters}")

# 显示行业与编码的对应关系
print("\n行业编码映射:")
for i, ind in enumerate(le.classes_[:10]):
    print(f"  {ind} -> {i}")
if len(le.classes_) > 10:
    print(f"  ... 共 {len(le.classes_)} 个行业")

# ==========================================
# 计算聚类评估指标
# ==========================================
print("\n" + "=" * 40)
print("行业分组聚类质量评估")
print("=" * 40)

# 计算距离矩阵（用于轮廓系数）
from sklearn.metrics.pairwise import euclidean_distances

# 1. Silhouette Score（轮廓系数）
try:
    sil_score = silhouette_score(X_scaled, industry_labels, metric='euclidean')
    print(f"\n1. Silhouette Score (轮廓系数): {sil_score:.4f}")
    print(f"   评价: {'>0.5表示良好' if sil_score > 0.5 else '>0.25表示弱结构' if sil_score > 0.25 else '<0.25表示无明显聚类结构'}")
except Exception as e:
    print(f"   计算失败: {e}")

# 2. Calinski-Harabasz Index（CH指数，越高越好）
try:
    ch_score = calinski_harabasz_score(X_scaled, industry_labels)
    print(f"\n2. Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"   评价: 值越高表示簇间离散度越大，簇内离散度越小")
except Exception as e:
    print(f"   计算失败: {e}")

# 3. Davies-Bouldin Index（DB指数，越低越好）
try:
    db_score = davies_bouldin_score(X_scaled, industry_labels)
    print(f"\n3. Davies-Bouldin Index: {db_score:.4f}")
    print(f"   评价: 0-1之间，值越小表示聚类效果越好")
except Exception as e:
    print(f"   计算失败: {e}")

# ==========================================
# 按行业分组的详细分析
# ==========================================
print("\n" + "=" * 40)
print("各行业内部紧凑度分析")
print("=" * 40)

# 计算每个行业的内部平均距离（紧凑度）
industry_names = le.classes_
industry_compactness = {}
industry_sizes = {}

for i, ind_name in enumerate(industry_names):
    mask = industry_labels == i
    if mask.sum() >= 2:
        cluster_data = X_scaled[mask]
        # 计算簇内平均成对距离
        from sklearn.metrics.pairwise import pairwise_distances
        intra_distances = pairwise_distances(cluster_data)
        # 只取上三角，避免重复
        avg_intra_dist = intra_distances[np.triu_indices_from(intra_distances, k=1)].mean()
        industry_compactness[ind_name] = avg_intra_dist
        industry_sizes[ind_name] = mask.sum()
    else:
        industry_compactness[ind_name] = np.nan
        industry_sizes[ind_name] = mask.sum()

# 按紧凑度排序（距离越小表示行业内股票越相似）
sorted_industries = sorted(industry_compactness.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))

print("\n行业内紧凑度排名（距离越小，同行业股票越相似）:")
print(f"{'排名':<4} {'行业':<20} {'股票数':<8} {'平均内距':<12} {'评价'}")
print("-" * 60)
for rank, (ind_name, dist) in enumerate(sorted_industries[:15], 1):
    if not np.isnan(dist):
        size = industry_sizes[ind_name]
        evaluation = "非常紧凑" if dist < 0.5 else "较紧凑" if dist < 0.8 else "松散"
        print(f"{rank:<4} {ind_name:<20} {size:<8} {dist:.4f}       {evaluation}")

# ==========================================
# 找出最不紧凑的行业（需要改进的地方）
# ==========================================
print("\n行业内松散度排名（距离越大，同行业股票越分散）:")
print(f"{'排名':<4} {'行业':<20} {'股票数':<8} {'平均内距':<12} {'评价'}")
print("-" * 60)
for rank, (ind_name, dist) in enumerate(sorted_industries[-15:][::-1], 1):
    if not np.isnan(dist):
        size = industry_sizes[ind_name]
        evaluation = "非常分散" if dist > 1.0 else "较分散" if dist > 0.8 else "一般"
        print(f"{rank:<4} {ind_name:<20} {size:<8} {dist:.4f}       {evaluation}")

# ==========================================
# 可视化：PCA降维展示行业分布
# ==========================================
print("\n" + "=" * 40)
print("可视化：行业分布")
print("=" * 40)

# PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 只显示主要的10个行业
top_industries = industry_counts.head(8).index.tolist()
colors = plt.cm.tab20(np.linspace(0, 1, len(top_industries)))
industry_to_color = {ind: colors[i] for i, ind in enumerate(top_industries)}
industry_to_color['其他'] = (0.7, 0.7, 0.7, 0.5)

# 为每个股票分配颜色
colors_list = []
for ind in industry_list:
    if ind in top_industries:
        colors_list.append(industry_to_color[ind])
    else:
        colors_list.append(industry_to_color['其他'])

plt.figure(figsize=(14, 10))
for i, ind in enumerate(top_industries):
    mask = [industry_list[j] == ind for j in range(len(industry_list))]
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=[industry_to_color[ind]], label=ind, alpha=0.7, s=30)

# 添加其他行业
mask_other = [industry_list[j] not in top_industries for j in range(len(industry_list))]
plt.scatter(X_pca[mask_other, 0], X_pca[mask_other, 1], 
            c=[industry_to_color['其他']], label='其他行业', alpha=0.5, s=20)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('基于真实行业标签的PCA可视化')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('industry_clustering_pca.png', dpi=150, bbox_inches='tight')
print("PCA可视化已保存: industry_clustering_pca.png")

# ==========================================
# 计算每个行业的轮廓系数
# ==========================================
print("\n" + "=" * 40)
print("各行业轮廓系数分析")
print("=" * 40)

# 计算每个样本的轮廓系数
try:
    sample_silhouette = silhouette_samples(X_scaled, industry_labels)
    
    industry_silhouette = {}
    for i, ind_name in enumerate(industry_names):
        mask = industry_labels == i
        if mask.sum() >= 2:
            industry_silhouette[ind_name] = sample_silhouette[mask].mean()
        else:
            industry_silhouette[ind_name] = np.nan
    
    # 排序显示
    sorted_sil = sorted(industry_silhouette.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
    
    print(f"\n各行业轮廓系数排名（越高表示该行业内股票越相似）:")
    print(f"{'排名':<4} {'行业':<20} {'股票数':<8} {'轮廓系数':<12} {'评价'}")
    print("-" * 60)
    for rank, (ind_name, sil) in enumerate(sorted_sil[:15], 1):
        if not np.isnan(sil):
            size = industry_sizes[ind_name]
            evaluation = "优秀" if sil > 0.3 else "良好" if sil > 0.15 else "较差" if sil > 0 else "很差"
            print(f"{rank:<4} {ind_name:<20} {size:<8} {sil:.4f}       {evaluation}")
    
    print("\n轮廓系数较差的行业（需要关注）:")
    for rank, (ind_name, sil) in enumerate(sorted_sil[-10:][::-1], 1):
        if not np.isnan(sil):
            size = industry_sizes[ind_name]
            print(f"  {ind_name}: 轮廓系数={sil:.4f} (股票数={size})")

except Exception as e:
    print(f"计算样本轮廓系数失败: {e}")

# ==========================================
# 对比分析：真实行业 vs 随机分组
# ==========================================
print("\n" + "=" * 40)
print("对比分析：真实行业 vs 随机分组")
print("=" * 40)

# 随机分组（保持相同的簇数）
np.random.seed(42)
random_labels = np.random.randint(0, n_industry_clusters, size=len(industry_labels))

# 计算随机分组的轮廓系数
random_sil = silhouette_score(X_scaled, random_labels)
print(f"\n真实行业分组轮廓系数: {sil_score:.4f}")
print(f"随机分组轮廓系数:     {random_sil:.4f}")
print(f"差异: {sil_score - random_sil:.4f}")

if sil_score > random_sil:
    print("✅ 真实行业分组优于随机分组，说明行业分类有一定意义")
else:
    print("⚠️ 真实行业分组不如随机分组，说明数据中的行业结构很弱")

# ==========================================
# 总结报告
# ==========================================
print("\n" + "=" * 60)
print("评估总结")
print("=" * 60)

print(f"""
1. 数据集概况:
   - 股票数量: {len(stock_names)}
   - 行业类别数: {n_industry_clusters}
   - 交易日数: {X_scaled.shape[1]}
   - 时间范围: {dates[0]} 到 {dates[-1]}

2. 聚类质量指标:
   - Silhouette Score (轮廓系数): {sil_score:.4f}
   - Calinski-Harabasz Index: {ch_score:.2f if 'ch_score' in dir() else 'N/A'}
   - Davies-Bouldin Index: {db_score:.4f if 'db_score' in dir() else 'N/A'}

3. 与随机分组对比:
   - 真实行业比随机分组 {"好" if sil_score > random_sil else "差"} {abs(sil_score - random_sil):.4f} 分

4. 紧凑度最高的行业:
   - {sorted_industries[0][0]}: 平均内距={sorted_industries[0][1]:.4f}
   - {sorted_industries[1][0]}: 平均内距={sorted_industries[1][1]:.4f}
   - {sorted_industries[2][0]}: 平均内距={sorted_industries[2][1]:.4f}

5. 轮廓系数最高的行业:
   - {sorted_sil[0][0]}: 轮廓系数={sorted_sil[0][1]:.4f}
   - {sorted_sil[1][0]}: 轮廓系数={sorted_sil[1][1]:.4f}
   - {sorted_sil[2][0]}: 轮廓系数={sorted_sil[2][1]:.4f}

6. 结论:
   {'真实行业分组具有中等到良好的聚类结构' if sil_score > 0.15 else '真实行业分组聚类结构较弱' if sil_score > 0.05 else '行业标签与股价行为关联很弱'}
   
   建议:
   - 可以将轮廓系数高的行业作为"行为一致性"的验证基准
   - 可以尝试其他聚类算法（如层次聚类）与真实行业进行对比
""")

print("\n✅ 脚本运行完成！")
print("生成文件: industry_clustering_pca.png")