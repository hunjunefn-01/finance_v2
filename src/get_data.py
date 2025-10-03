import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# --- 1. 환경 설정 및 표준 헤더 정의 ---
# 통합할 표준 헤더
STANDARD_HEADERS = [
    '출처', '거래_일시', '거래_유형', '거래_상대방/가맹점', 
    '출금_금액', '입금_금액', '잔액', '상세_정보'
]

# 출처(Source Name)와 파일명 매핑
FILE_MAPPING = {
    '케이': '케이뱅크.tsv',
    '토스': '토스뱅크.tsv',
    '신한': '신한은행.tsv',
    '카카오': '카카오뱅크.tsv',
    '현대': '현대카드.tsv'
}

# DATA_DIR_PATH = Path(r'C:\Users\dgw49\Documents\Aeong\finance_venv_313\data') 

# 프로젝트 경로 확인
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_PATH = PROJECT_ROOT / 'data'
LOG_DIR_PATH = PROJECT_ROOT / 'log'

# 디렉토리 생성
os.makedirs(DATA_DIR_PATH, exist_ok=True)
os.makedirs(LOG_DIR_PATH, exist_ok=True)

# --- 2. 유틸 함수 정의 ---

# 금액 및 잔액을 숫자(Int64)로 변환하는 함수
def clean_amount(series):
    """금액/잔액 문자열에서 쉼표, 통화 기호를 제거하고 float/Int64로 변환"""
    # 숫자(정수/소수), 마이너스 부호를 제외한 모든 문자 제거
    cleaned = series.astype(str).str.replace(r'[^\d\.\-\+]', '', regex=True)
    # 빈 문자열을 0으로 대체하고 float로 변환
    numeric = cleaned.replace('', 0).astype(float)
    # 정수 부분만 필요하므로 Int64로 변환 (NaN을 지원하는 정수 타입)
    return numeric.astype('Int64')

# --- 3. 파일별 변환 로직 (함수) ---

def process_file(source_name: str, file_path: Path) -> pd.DataFrame:
    """단일 파일을 읽고 표준 헤더 DataFrame으로 변환"""
    print(f"[{source_name}] 파일 읽기 시작: {file_path}")
    
    # TSV 데이터 읽기 (인코딩 문제 방지를 위해 'utf-8' 또는 'cp949' 시도)
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='cp949', on_bad_lines='skip')
        except Exception as e:
            print(f"⚠️ {source_name} 파일 읽기 실패: {e}")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"❌ {source_name} 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ {source_name} 파일 처리 중 알 수 없는 오류 발생: {e}")
        return pd.DataFrame()

    # 표준 헤더 DataFrame 초기화
    standard_df = pd.DataFrame(columns=STANDARD_HEADERS)
    standard_df['출처'] = source_name
    
    # 공통: 모든 컬럼 헤더의 앞뒤 공백 제거
    df.columns = df.columns.str.strip() 

    # --- A. 케이뱅크 (Source: '케이') ---
    if source_name == '케이':
        standard_df['거래_일시'] = pd.to_datetime(df['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        
        df['입금금액'] = clean_amount(df['입금금액'])
        df['출금금액'] = clean_amount(df['출금금액'])
        df['잔액'] = clean_amount(df['잔액'])
        
        standard_df['거래_유형'] = df['거래구분'].fillna('')
        standard_df['거래_상대방/가맹점'] = df['적요내용'].fillna(df['상대 예금주명']).fillna('')
        standard_df['출금_금액'] = df['출금금액'].fillna(0)
        standard_df['입금_금액'] = df['입금금액'].fillna(0)
        standard_df['잔액'] = df['잔액']
        standard_df['출처'] = source_name

        
        details = df[['상대 은행', '상대 계좌번호', '메모']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details
        
    # --- B. 토스뱅크 (Source: '토스') ---
    elif source_name == '토스':
        standard_df['거래_일시'] = pd.to_datetime(df['거래 일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        
        df['거래 금액'] = clean_amount(df['거래 금액'])
        df['거래 후 잔액'] = clean_amount(df['거래 후 잔액'])
        
        standard_df['거래_유형'] = df['거래 유형'] + ' (' + df['적요'] + ')'
        standard_df['거래_상대방/가맹점'] = df['적요']
        standard_df['출금_금액'] = df['거래 금액'].apply(lambda x: abs(x) if x < 0 else 0)
        standard_df['입금_금액'] = df['거래 금액'].apply(lambda x: x if x > 0 else 0)
        standard_df['잔액'] = df['거래 후 잔액']
        standard_df['출처'] = source_name
        
        details = df[['거래 기관', '계좌번호', '메모']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details
        
    # --- C. 신한은행 (Source: '신한') ---
    elif source_name == '신한':
        # 헤더에서 '(원)' 제거
        df.columns = df.columns.str.replace(r'\(원\)', '', regex=True).str.strip()
        
        standard_df['거래_일시'] = pd.to_datetime(df['거래일자'] + ' ' + df['거래시간'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        df['입금'] = clean_amount(df['입금'])
        df['출금'] = clean_amount(df['출금'])
        df['잔액'] = clean_amount(df['잔액'])
        
        standard_df['거래_유형'] = df['적요']
        standard_df['거래_상대방/가맹점'] = df['내용']
        standard_df['출금_금액'] = df['출금'].fillna(0)
        standard_df['입금_금액'] = df['입금'].fillna(0)
        standard_df['잔액'] = df['잔액']
        standard_df['출처'] = source_name
        
        standard_df['상세_정보'] = df['거래점'].astype(str).fillna('')
        
    # --- D. 카카오뱅크 (Source: '카카오') ---
    elif source_name == '카카오':
        standard_df['거래_일시'] = pd.to_datetime(df['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        
        df['거래금액'] = clean_amount(df['거래금액'])
        df['거래 후 잔액'] = clean_amount(df['거래 후 잔액'])
        
        standard_df['거래_유형'] = df['구분'] + ' (' + df['거래구분'] + ')'
        standard_df['거래_상대방/가맹점'] = df['내용']
        standard_df['출금_금액'] = df['거래금액'].apply(lambda x: abs(x) if x < 0 else 0)
        standard_df['입금_금액'] = df['거래금액'].apply(lambda x: x if x > 0 else 0)
        standard_df['잔액'] = df['거래 후 잔액']
        standard_df['출처'] = source_name
        
        standard_df['상세_정보'] = df['메모'].astype(str).fillna('')
        
    # --- E. 현대카드 (Source: '현대') ---
    elif source_name == '현대':
        # 날짜 형식 '2025년 09월 30일' -> '2025-09-30' 변환
        date_str = df['이용일'].str.replace(r'[년월일]', '-', regex=True).str.strip('-')
        standard_df['거래_일시'] = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        
        df['이용금액'] = clean_amount(df['이용금액'])
        
        standard_df['거래_유형'] = '카드결제 (' + df['카드구분'] + ')'
        standard_df['거래_상대방/가맹점'] = df['가맹점명']
        standard_df['출금_금액'] = df['이용금액'].fillna(0)
        standard_df['입금_금액'] = 0
        standard_df['잔액'] = pd.NA # 카드 내역은 잔액이 없음
        standard_df['출처'] = source_name
        
        # 상세 정보 통합
        details = df[['확정일', '카드명(카드번호 뒤 4자리)', '사업자번호', '승인번호', '할부 개월']].astype(str).fillna('').agg(
            lambda x: ', '.join(x[x != '']), axis=1
        ).str.replace(r',\s*$', '', regex=True)
        standard_df['상세_정보'] = details

    # 원본 데이터프레임과 병합하여 표준 헤더에 맞게 데이터 재정렬
    # (주의: 원본 데이터가 많을 경우 이 부분이 느려질 수 있으나, 현재 로직은 매핑된 컬럼만 사용)
    # 필요한 컬럼만 선택하여 반환
    return standard_df[STANDARD_HEADERS].dropna(subset=['거래_일시']).reset_index(drop=True)


# --- 4. 메인 통합 실행 로직 ---

def integrate_all_transactions():
    """모든 파일을 읽어 통합 데이터셋을 생성하고 결과를 출력"""
    combined_df_list = []
    
    # 4.1. 파일 처리 루프
    for source_name, file_name in FILE_MAPPING.items():
        file_path = DATA_DIR_PATH / file_name
        standard_data = process_file(source_name, file_path)
        if not standard_data.empty:
            combined_df_list.append(standard_data)
    
    # 4.2. 최종 통합 및 정렬
    if not combined_df_list:
        print("\n**통합할 데이터가 없습니다. 파일 경로 또는 내용을 확인해주세요.**")
        return None
        
    final_df = pd.concat(combined_df_list, ignore_index=True)
    # 거래 일시를 기준으로 최신순 정렬
    final_df = final_df.sort_values(by='거래_일시', ascending=False).reset_index(drop=True)

    # 4.3. 결과 출력 및 저장 로직 추가
    print("\n" + "="*50)
    print("## ✅ 최종 통합된 금융 거래 내역 (표준화 완료)")
    print(f"**총 거래 건수:** {len(final_df)}건")
    print(f"**통합 데이터셋의 헤더:** {list(final_df.columns)}")
    print("="*50)

    # 4.3.1. 상위 행 미리보기
    final_df_display = final_df.head(100).copy() 
    
    # 금액/잔액에 쉼표 추가하여 보기 좋게 변환
    for col in ['출금_금액', '입금_금액', '잔액']:
        final_df_display[col] = final_df_display[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')

    # 결과 미리보기
    # print(final_df_display.to_markdown(index=False))
    
    # 4.3.2. TSV 파일로 저장 (log 폴더에)
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
    # 코드 실행
    integrated_data = integrate_all_transactions()
    
    # 이후 integrated_data를 활용한 분석 로직 추가 가능
    if integrated_data is not None:
        print("\n\n*통합된 데이터를 변수 'integrated_data'에 저장했습니다.*")
        # 예시: 월별 출금 합계
        # integrated_data['월'] = integrated_data['거래_일시'].dt.to_period('M')
        # monthly_spending = integrated_data.groupby('월')['출금_금액'].sum()
        # print("\n월별 출금 합계:\n", monthly_spending)
