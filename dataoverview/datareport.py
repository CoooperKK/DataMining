# data_report.py
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('./dataset/stock_data_sp500_paper.csv')

# 基本信息
rows, cols = df.shape
stocks = cols - 2  # 去掉 index, date 两列
dates = rows - 1   # 去掉第一行行业行

# 价格数据部分
price_data = df.iloc[1:, 2:].astype(float)
price_min = price_data.min().min()
price_max = price_data.max().max()

# 缺失值情况
missing_ratio = price_data.isna().sum().sum() / (price_data.shape[0] * price_data.shape[1])

# 输出表格（可直接复制）
print("\n" + "="*60)
print("数据集 stock_data_sp500_paper.csv 数据形态报告")
print("="*60)
print(f"{'属性':<20} {'数值':<30}")
print("-"*60)
print(f"{'股票数量':<20} {stocks}")
print(f"{'交易日数量':<20} {dates}")
print(f"{'数据范围 (价格指数)':<20} [{price_min:.2f}, {price_max:.2f}]")
print(f"{'缺失值比例':<20} {missing_ratio:.4%}")
print(f"{'数据形状':<20} {price_data.shape[0]}行 × {price_data.shape[1]}列")
print("="*60)

# 预处理流程说明（供PPT参考）
print("\n预处理流程说明：")
print("1. 从 Kaggle 下载 S&P 500 历史日线数据（2013-02-08 至 2018-02-08）")
print("2. 提取 Close 价格列，删除起始日期不一致的公司（35家）")
print("3. 转换为价格指数（每只股票首日价格归一化为100）")
print("4. SMA-10 平滑处理（窗口=10，Simple Moving Average）")
print("5. 删除缺失率 >10% 的股票，前向+后向填充剩余缺失值")
print("6. 最终保留 350 只股票，1260 个交易日")

# 保存到 CSV（供 Excel 打开）
summary_data = {
    '属性': ['股票数量', '交易日数量', '价格指数范围', '缺失值比例', '数据形状'],
    '数值': [stocks, dates, f'[{price_min:.2f}, {price_max:.2f}]', f'{missing_ratio:.4%}', f'{price_data.shape[0]}行 × {price_data.shape[1]}列']
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('./result/data_report.csv', index=False)
print("\n已保存数据报告至 result/data_report.csv")