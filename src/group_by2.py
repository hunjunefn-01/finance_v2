#.\Scripts\python.exe .\src\group.py
import pandas as pd
import json

# 1️⃣ TSV 파일 불러오기
df = pd.read_csv(r"C:\Users\hunju\dev\finance_venv_313\log\category.tsv", sep="\t")

# 2️⃣ 조합별 건수 집계 (거래유형 제거)
summary = (
    df.groupby(["주요_카테고리", "세부_카테고리"])
      .size()
      .reset_index(name="건수")
)

# 3️⃣ 건수 기준 내림차순 정렬 및 순번 재설정
summary = summary.sort_values(by="건수", ascending=False).reset_index(drop=True)
summary.index = summary.index + 1
summary["순번"] = summary.index

# 4️⃣ TSV 파일로 저장
summary.to_csv(
    r"C:\Users\hunju\dev\finance_venv_313\log\category_summary_sorted.tsv",
    sep="\t",
    index=False,
    encoding="utf-8-sig"
)

# 5️⃣ JSON 생성 (거래유형 제거, 건수 10 이상만)
nested_json = {}
for _, row in summary.iterrows():
    건수 = int(row["건수"])
    if 건수 < 10:
        continue

    주요_카테고리 = row["주요_카테고리"]
    세부_카테고리 = row["세부_카테고리"]

    if 주요_카테고리 not in nested_json:
        nested_json[주요_카테고리] = []
    if 세부_카테고리 not in nested_json[주요_카테고리]:
        nested_json[주요_카테고리].append(세부_카테고리)

# 6️⃣ JSON 파일로 저장
with open(
    r"C:\Users\hunju\dev\finance_venv_313\log\category_summary.json",
    "w", encoding="utf-8"
) as f:
    json.dump(nested_json, f, ensure_ascii=False, indent=2)
