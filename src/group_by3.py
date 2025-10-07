#.\Scripts\python.exe .\src\group.py
import pandas as pd
import json

# 1️⃣ TSV 파일 불러오기
df = pd.read_csv(r"C:\Users\hunju\dev\finance_venv_313\log\category.tsv", sep="\t")

# 2️⃣ 조합별 건수 집계
summary = (
    df.groupby(["거래_유형", "주요_카테고리", "세부_카테고리"])
      .size()
      .reset_index(name="건수")
)

# 3️⃣ 건수 기준 내림차순 정렬 및 순번 재설정
summary = summary.sort_values(by="건수", ascending=False).reset_index(drop=True)
summary.index = summary.index + 1  # 1부터 시작
summary["순번"] = summary.index  # 순번 컬럼 추가

# 4️⃣ CSV 파일로 저장 (순번, 건수 포함)
summary.to_csv(
    r"C:\Users\hunju\dev\finance_venv_313\log\category_summary_sorted.tsv",  # 파일 확장자 변경
    sep="\t",  # 구분자 탭으로 지정
    index=False,
    encoding="utf-8-sig"
)

# 5️⃣ JSON용 중첩 구조 생성 (순번/건수 없이, 세부 카테고리 리스트로)
nested_json = {}
for _, row in summary.iterrows():
    건수 = int(row["건수"])
    
    # 건수가 n 이상인 경우만 처리
    if 건수 < 1:
        continue
    
    거래_유형 = row["거래_유형"]
    주요_카테고리 = row["주요_카테고리"]
    세부_카테고리 = row["세부_카테고리"]

    if 거래_유형 not in nested_json:
        nested_json[거래_유형] = {}
    if 주요_카테고리 not in nested_json[거래_유형]:
        nested_json[거래_유형][주요_카테고리] = []

    # 중복 없이 추가
    if 세부_카테고리 not in nested_json[거래_유형][주요_카테고리]:
        nested_json[거래_유형][주요_카테고리].append(세부_카테고리)

# 6️⃣ JSON 파일로 저장
with open(
    r"C:\Users\hunju\dev\finance_venv_313\log\category_summary.json",
    "w", encoding="utf-8"
) as f:
    json.dump(nested_json, f, ensure_ascii=False, indent=2)

# 확인용 출력
# print(json.dumps(nested_json, ensure_ascii=False, indent=2))