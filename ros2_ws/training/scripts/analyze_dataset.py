import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "dataset" / "label.csv"

df = pd.read_csv(CSV_PATH)

print("\n=== DIMENSIONES DEL DATASET ===")
print(df.shape)

print("\n=== COLUMNAS ===")
print(df.columns.tolist())

print("\n=== PRIMERAS 10 FILAS ===")
print(df.head(10).to_string())

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

print("\n=== FILAS DUPLICADAS ===")
print(df.duplicated().sum())

print("\n=== ESTADISTICAS DESCRIPTIVAS ===")
print(df.describe().to_string())
