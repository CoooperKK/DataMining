# diagnose_sector_data.py
import pandas as pd

print("=" * 60)
print("S&P 500 Sector Data Diagnosis")
print("=" * 60)

# 读取数据
df = pd.read_csv('dataset/SP500_Sector.csv')

print(f"\n[1] Basic Info")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\n[2] First 5 rows")
print(df.head())

print(f"\n[3] Data types")
print(df.dtypes)

print(f"\n[4] Unique values in first column (tickers)")
ticker_col = df.columns[0]
print(f"Column name: '{ticker_col}'")
print(f"Unique tickers: {df[ticker_col].nunique()}")
print(f"First 10 tickers: {df[ticker_col].head(10).tolist()}")

print(f"\n[5] Check for sector columns")
sector_cols = [col for col in df.columns if 'sector' in col.lower() or 'Sector' in col or 'industry' in col.lower()]
print(f"Potential sector columns: {sector_cols}")

for col in sector_cols:
    print(f"\n  Column: '{col}'")
    print(f"    Unique values: {df[col].nunique()}")
    print(f"    Sample: {df[col].dropna().head(5).tolist()}")

print(f"\n[6] Check first row as column names")
print(df.iloc[0].tolist()[:5])

print("\n[7] Summary")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")