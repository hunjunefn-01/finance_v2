import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# --- 1. 환경 설정 및 표준 헤더 정의 ---
STANDARD_HEADERS = [
    '출처', '거래_일시', '거래_유형', '거래_상대방/가맹점', 
    '출금_금액', '입금_금액', '잔액', '상세_정보'
]

# 출처(Source Name)와 파일명 매핑 (사용자 실행 로그 반영)
FILE_MAPPING = {
    '케이': '케이뱅크.tsv',
    '토스': '토스뱅크.tsv',
    '신한': '신한은행.tsv',
    '카카오': '카카오뱅크.tsv',
    '현대': '현대카드.tsv',
    '농협_혜진': '농협_혜진.tsv', # '추가'를 '농협_혜진'으로 반영
}

# 프로젝트 경로 확인 및 디렉토리 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_PATH = PROJECT_ROOT / 'data'
LOG_DIR_PATH = PROJECT_ROOT / 'log'

# 디렉토리 생성 (없을 경우)
os.makedirs(DATA_DIR_PATH, exist_ok=True)
os.makedirs(LOG_DIR_PATH, exist_ok=True)

# 금액/잔액 컬럼 이름 정의 (clean_amount 일괄 적용을 위해 사용)
AMOUNT_COLS_MAP = {
    '케이': ['입금금액', '출금금액', '잔액'],
    '토스': ['거래 금액', '거래 후 잔액'],
    '신한': ['입금', '출금', '잔액'],
    '카카오': ['거래금액', '거래 후 잔액'],
    '현대': ['이용금액'],
    '농협_혜진': ['출금금액', '입금금액', '거래후잔액'] # '농협_혜진' 반영
}


# --- 2. 유틸 함수 정의 ---

def clean_amount(series: pd.Series) -> pd.Series:
    """금액/잔액 문자열에서 쉼표, 통화 기호를 제거하고 Int64로 변환"""
    # 숫자, 마이너스 부호를 제외한 모든 문자 제거
    cleaned = series.astype(str).str.replace(r'[^0-9\.\-]+', '', regex=True)
    
    # Int64 타입 (NaN 지원)으로 변환
    return pd.to_numeric(cleaned, errors='coerce').astype('Int64')


# --- 3. 파일별 변환 로직 ---

def process_file(source_name: str) -> pd.DataFrame | None:
    """개별 TSV 파일을 읽어 표준화된 DataFrame으로 변환"""
    try:
        file_path = DATA_DIR_PATH / FILE_MAPPING[source_name]
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        print(f"[{source_name}] 파일 읽기 시작: {file_path}")
    except Exception as e:
        print(f"⚠️ [{source_name}] 파일 읽기 오류: {e}")
        return None

    # 1. 표준 DataFrame 초기화 및 '출처' 단일 할당
    standard_df = pd.DataFrame(columns=STANDARD_HEADERS, index=df.index)
    standard_df['출처'] = source_name
    
    # 2. 공통 금액/잔액 컬럼 전처리
    if source_name in AMOUNT_COLS_MAP:
        for col in AMOUNT_COLS_MAP[source_name]:
            if col in df.columns:
                df[col] = clean_amount(df[col])


    # 3. 데이터 변환 및 매핑 (각 금융기관별 로직)
    
    # --- A. 케이뱅크 ---
    if source_name == '케이':
        standard_df['거래_일시'] = pd.to_datetime(df['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        standard_df['거래_유형'] = df['거래구분'].fillna('')
        standard_df['거래_상대방/가맹점'] = df['적요내용'].fillna(df['상대 예금주명']).fillna('')
        standard_df['출금_금액'] = df['출금금액'].fillna(0)
        standard_df['입금_금액'] = df['입금금액'].fillna(0)
        standard_df['잔액'] = df['잔액']
        
        details = df[['상대 은행', '상대 계좌번호', '메모']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details

        
    # --- B. 토스뱅크 ---
    elif source_name == '토스':
        standard_df['거래_일시'] = pd.to_datetime(df['거래 일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        standard_df['거래_유형'] = df['거래 유형'] + ' (' + df['적요'] + ')'
        standard_df['거래_상대방/가맹점'] = df['적요']
        standard_df['출금_금액'] = df['거래 금액'].apply(lambda x: abs(x) if x < 0 else 0)
        standard_df['입금_금액'] = df['거래 금액'].apply(lambda x: x if x > 0 else 0)
        standard_df['잔액'] = df['거래 후 잔액']
        
        details = df[['거래 기관', '계좌번호', '메모']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details

        
    # --- C. 신한은행 ---
    elif source_name == '신한':
        df.columns = df.columns.str.replace(r'\(원\)', '', regex=True).str.strip()
        
        standard_df['거래_일시'] = pd.to_datetime(df['거래일자'] + ' ' + df['거래시간'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        standard_df['거래_유형'] = df['적요']
        standard_df['거래_상대방/가맹점'] = df['내용']
        standard_df['출금_금액'] = df['출금'].fillna(0)
        standard_df['입금_금액'] = df['입금'].fillna(0)
        standard_df['잔액'] = df['잔액']
        standard_df['상세_정보'] = df['거래점'].astype(str).fillna('')

        
    # --- D. 카카오뱅크 ---
    elif source_name == '카카오':
        standard_df['거래_일시'] = pd.to_datetime(df['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        standard_df['거래_유형'] = df['구분'] + ' (' + df['거래구분'] + ')'
        standard_df['거래_상대방/가맹점'] = df['내용']
        standard_df['출금_금액'] = df['거래금액'].apply(lambda x: abs(x) if x < 0 else 0)
        standard_df['입금_금액'] = df['거래금액'].apply(lambda x: x if x > 0 else 0)
        standard_df['잔액'] = df['거래 후 잔액']
        standard_df['상세_정보'] = df['메모'].astype(str).fillna('')
        
        
    # --- E. 현대카드 ---
    elif source_name == '현대':
        date_str = df['이용일'].str.replace(r'[년월일]', '-', regex=True).str.strip('-')
        standard_df['거래_일시'] = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        
        standard_df['거래_유형'] = '카드결제 (' + df['카드구분'] + ')'
        standard_df['거래_상대방/가맹점'] = df['가맹점명']
        standard_df['출금_금액'] = df['이용금액'].fillna(0)
        standard_df['입금_금액'] = 0
        standard_df['잔액'] = pd.NA
        
        details = df[['확정일', '카드명(카드번호 뒤 4자리)', '사업자번호', '승인번호', '할부 개월']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details

        
    # --- F. 농협_혜진 데이터 ---
    elif source_name == '농협_혜진':
        try:
            standard_df['거래_일시'] = pd.to_datetime(
                df['거래일시'].str.replace(r'\s+', ' ', regex=True).str.strip(), 
                format='%Y/%m/%d %H:%M:%S', 
                errors='coerce'
            )
        except Exception:
            standard_df['거래_일시'] = df['거래일시']
            
        standard_df['거래_상대방/가맹점'] = df['거래내용'].fillna('')
        
        standard_df['거래_유형'] = np.where(
            pd.notna(df['출금금액']) & (df['출금금액'] > 0), 
            '출금',
            np.where(
                pd.notna(df['입금금액']) & (df['입금금액'] > 0), 
                '입금', 
                '기타'
            )
        )
        
        standard_df['출금_금액'] = df['출금금액'].fillna(0)
        standard_df['입금_금액'] = df['입금금액'].fillna(0)
        standard_df['잔액'] = df['거래후잔액']
        
        details = df[['거래기록사항', '거래점', '거래메모']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details
        
    else:
        print(f"⚠️ [{source_name}] 처리할 수 없는 출처입니다. FILE_MAPPING에 등록되어 있으나, process_file 로직이 없습니다.")
        return None

    # 4. 최종 정제 및 반환
    standard_df = standard_df[STANDARD_HEADERS].dropna(subset=['거래_일시']).reset_index(drop=True)
    return standard_df


# --- 4. 메인 통합 실행 로직 ---

def integrate_all_transactions():
    """모든 파일을 읽어 통합 데이터셋을 생성하고 결과를 출력 및 저장"""
    combined_df_list = []
    
    # 4.1. 파일 처리 루프
    for source_name in FILE_MAPPING.keys():
        standard_data = process_file(source_name)
        if standard_data is not None and not standard_data.empty:
            combined_df_list.append(standard_data)
    
    # 4.2. 최종 통합 및 정렬
    if not combined_df_list:
        print("\n**통합할 데이터가 없습니다. 파일 경로 또는 내용을 확인해주세요.**")
        return None
        
    final_df = pd.concat(combined_df_list, ignore_index=True)
    # 거래 일시를 기준으로 최신순 정렬
    final_df = final_df.sort_values(by='거래_일시', ascending=False).reset_index(drop=True)

    # 4.3. 결과 출력 및 저장
    print("\n" + "="*50)
    print("## ✅ 최종 통합된 금융 거래 내역 (표준화 완료)")
    print(f"**총 거래 건수:** {len(final_df)}건")
    print(f"**통합 데이터셋의 헤더:** {list(final_df.columns)}")
    print("="*50)
    
    # 4.3.1. TSV 파일로 저장 (log 폴더에)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file_name = f'integrated_transactions_{timestamp}.tsv'
    output_path = LOG_DIR_PATH / output_file_name
    
    try:
        # TSV 형식으로 저장 (sep='\t', index=False)
        final_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print(f"\n**💾 통합 데이터 전체 내역을 성공적으로 저장했습니다.**")
        print(f"**저장 경로:** {output_path}")
    except Exception as e:
        print(f"\n⚠️ 데이터 저장 실패: {e}")

    return final_df


if __name__ == "__main__":
    integrated_data = integrate_all_transactions()
    
    if integrated_data is not None:
        print("\n\n*통합된 데이터를 변수 'integrated_data'에 저장했습니다.*")