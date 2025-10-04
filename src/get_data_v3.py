import re
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# 1. 환경 설정 및 상수 정의
# ----------------------------------------------------------------------

# 프로젝트 경로 설정 (현재 스크립트가 {프로젝트경로}/src 내에 있다고 가정)
# __file__ 변수가 존재하는 로컬 환경에서 실행할 경우:
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# (src의 상위 디렉토리가 프로젝트 루트가 됨)
# 인터랙티브 환경에서 __file__이 없을 경우: 현재 경로(.)를 프로젝트 루트로 간주
PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in locals() else Path('.')

DATA_DIR_PATH = PROJECT_ROOT / 'data' # 원천 파일 경로
LOG_DIR_PATH = PROJECT_ROOT / 'log' # 로그 저장 경로
LOG_DIR_PATH.mkdir(exist_ok=True) # 로그 디렉토리가 없으면 생성

# ContentFetchId 매핑 (실행 환경에서 파일을 불러오는 데 사용)
FILE_MAP = {
    "카카오뱅크": "카카오뱅크.tsv",
    "케이뱅크": "케이뱅크.tsv",
    "토스뱅크": "토스뱅크.tsv",
    "현대카드": "현대카드.tsv",
    "농협_혜진": "농협_혜진.tsv",
    "신한은행": "신한은행.tsv"
}
# 최종 통합 컬럼 (총 9개)
FINAL_COLUMNS = ['거래일시', '사용구분', '사용내역', '거래상대방', '입금액', '출금액', '잔액', '메모', '출처']


# ----------------------------------------------------------------------
# 2. 헬퍼 함수 정의
# ----------------------------------------------------------------------

def clean_amount(series):
    """숫자형으로 변환 전, 불필요한 문자를 제거하고 숫자로 변환합니다. (빈 값/문자열은 np.nan 처리)"""
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        # 쉼표, '원', 공백 등 제거
        cleaned = series.astype(str).str.replace(r'[,\원\s]', '', regex=True)
        # 빈 문자열을 np.nan으로 변경하여 float 변환 시 오류 방지
        return pd.to_numeric(cleaned.replace('', np.nan), errors='coerce')
    return series

def combine_and_clean_str(df, series_names, sep=' '):
    """지정된 컬럼들을 결합하고, 결과가 빈 문자열이면 pd.NA로 처리합니다."""
    series_to_combine = []
    for name in series_names:
        if name in df.columns:
            # 문자열로 변환하고 NaN/None을 빈 문자열로 대체 (pd.NA로 대체 시 'nan' 문자열 방지)
            s = df[name].astype(str).replace({'nan': '', 'None': ''}).str.strip()
            series_to_combine.append(s)
        else:
            series_to_combine.append(pd.Series([''] * len(df), index=df.index))
            
    # 빈 문자열을 제외하고 결합
    combined = [sep.join(filter(None, row)) for row in zip(*series_to_combine)]
    combined = pd.Series(combined, index=df.index, dtype='object')

    # 불필요한 공백 제거 후, 최종적으로 빈 문자열을 pd.NA로 변환 (결과적으로 nan으로 저장됨)
    return combined.str.replace(r'\s+', ' ', regex=True).str.strip().replace('', pd.NA)


# ----------------------------------------------------------------------
# 3. 핵심 전처리 로직 (각 파일별 9단계 통합)
# ----------------------------------------------------------------------

def process_all_files(file_map):
    """6개 파일을 읽고 전처리 규칙에 따라 하나의 통합 DataFrame으로 결합합니다."""
    list_of_dfs = []
    
    def load_file(source_name):
        # 수정된 부분: DATA_DIR_PATH를 사용하여 전체 경로를 지정합니다.
        file_name = file_map[source_name]
        data_path = DATA_DIR_PATH / file_name

        # 전체 경로를 사용하여 파일을 로드합니다.
        return pd.read_csv(data_path, sep='\t', encoding='utf-8', na_values=['', ' ', 'nan'], keep_default_na=True)

    # --- 1. 카카오뱅크 (총 9단계) ---
    try:
        df_kakao = load_file("카카오뱅크")
        df_kakao_standard = pd.DataFrame(index=df_kakao.index)
        df_kakao['거래금액_clean'] = clean_amount(df_kakao['거래금액'])

        # 1단계: 거래일시 생성
        df_kakao_standard['거래일시'] = pd.to_datetime(df_kakao['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce').dt.strftime('%Y.%m.%d %H:%M:%S').replace('NaT', pd.NA)
        # 2단계: 사용구분 생성
        df_kakao_standard['사용구분'] = df_kakao['구분'].replace('', pd.NA)
        # 3단계: 사용내역 생성
        df_kakao_standard['사용내역'] = df_kakao['거래구분'].replace('', pd.NA)
        # 4단계: 거래상대방 생성
        df_kakao_standard['거래상대방'] = df_kakao['내용'].replace('', pd.NA)
        # 5단계: 입금액 생성
        df_kakao_standard['입금액'] = df_kakao['거래금액_clean'].apply(lambda x: x if x > 0 else 0).astype(float)
        # 6단계: 출금액 생성
        df_kakao_standard['출금액'] = df_kakao['거래금액_clean'].apply(lambda x: abs(x) if x < 0 else 0).astype(float)
        # 7단계: 잔액 생성
        df_kakao_standard['잔액'] = clean_amount(df_kakao['거래 후 잔액'])
        # 8단계: 메모 생성
        df_kakao_standard['메모'] = df_kakao['메모'].replace('', pd.NA)
        # 9단계: 파일 출처 추가
        df_kakao_standard['출처'] = '카카오뱅크'
        
        list_of_dfs.append(df_kakao_standard[FINAL_COLUMNS].copy())
    except Exception as e:
        print(f"⚠️ [카카오뱅크] 처리 중 오류 발생: {e}")

    # --- 2. 케이뱅크 (총 9단계) ---
    try:
        df_kbank = load_file("케이뱅크")
        df_kbank_standard = pd.DataFrame(index=df_kbank.index)
        df_kbank['입금금액_clean'] = clean_amount(df_kbank['입금금액']).fillna(0)
        df_kbank['출금금액_clean'] = clean_amount(df_kbank['출금금액']).fillna(0)
        
        # 1단계: 거래일시 생성
        df_kbank_standard['거래일시'] = pd.to_datetime(df_kbank['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce').dt.strftime('%Y.%m.%d %H:%M:%S').replace('NaT', pd.NA)
        # 2단계: 사용구분 생성
        df_kbank_standard['사용구분'] = np.select([df_kbank['입금금액_clean'] > 0, df_kbank['출금금액_clean'] > 0], ['입금', '출금'], default=pd.NA) 
        # 3단계: 사용내역 생성
        # df_kbank_standard['사용내역'] = combine_and_clean_str(df_kbank, ['거래구분', '적요내용'])
        df_kbank_standard['사용내역'] = df_kbank['거래구분']
        # 4단계: 거래상대방 생성
        # df_kbank_standard['거래상대방'] = combine_and_clean_str(df_kbank, ['상대 은행', '상대 예금주명', '상대 계좌번호'])
        df_kbank_standard['거래상대방'] = df_kbank['적요내용']
        # 5단계: 입금액 생성
        df_kbank_standard['입금액'] = df_kbank['입금금액_clean']
        # 6단계: 출금액 생성
        df_kbank_standard['출금액'] = df_kbank['출금금액_clean']
        # 7단계: 잔액 생성
        df_kbank_standard['잔액'] = clean_amount(df_kbank['잔액'])
        # 8단계: 메모 생성
        # df_kbank_standard['메모'] = df_kbank['메모'].replace('', pd.NA)
        df_kbank_standard['메모'] = combine_and_clean_str(df_kbank, ['메모', '상대 은행', '상대 예금주명', '상대 계좌번호'])
        # 9단계: 파일 출처 추가
        df_kbank_standard['출처'] = '케이뱅크'
        
        list_of_dfs.append(df_kbank_standard[FINAL_COLUMNS].copy())
    except Exception as e:
        print(f"⚠️ [케이뱅크] 처리 중 오류 발생: {e}")

    # --- 3. 토스뱅크 (총 9단계) ---
    try:
        df_toss = load_file("토스뱅크")
        df_toss_standard = pd.DataFrame(index=df_toss.index)
        df_toss['거래 금액_clean'] = clean_amount(df_toss['거래 금액'])

        # 1단계: 거래일시 생성
        df_toss_standard['거래일시'] = pd.to_datetime(df_toss['거래 일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce').dt.strftime('%Y.%m.%d %H:%M:%S').replace('NaT', pd.NA)
        # 2단계: 사용구분 생성
        deposit_mask = df_toss['거래 금액_clean'] > 0
        withdraw_mask = df_toss['거래 금액_clean'] < 0
        df_toss_standard['사용구분'] = np.select([deposit_mask, withdraw_mask],['입금_' + df_toss['거래 유형'].astype(str), '출금_' + df_toss['거래 유형'].astype(str)],default=pd.NA)
        df_toss_standard['사용구분'] = df_toss_standard['사용구분'].str.replace('_nan', '').replace('nan', pd.NA) 
        # 3단계: 사용내역 생성
        # df_toss_standard['사용내역'] = combine_and_clean_str(df_toss, ['거래 기관', '적요'])
        df_toss_standard['사용내역'] = df_toss['거래 기관']
        # 4단계: 거래상대방 생성
        # df_toss_standard['거래상대방'] = combine_and_clean_str(df_toss, ['거래 기관', '계좌번호'])
        df_toss_standard['거래상대방'] = df_toss['적요']
        # 5단계: 입금액 생성
        df_toss_standard['입금액'] = df_toss['거래 금액_clean'].apply(lambda x: x if x > 0 else 0).astype(float)
        # 6단계: 출금액 생성
        df_toss_standard['출금액'] = df_toss['거래 금액_clean'].apply(lambda x: abs(x) if x < 0 else 0).astype(float)
        # 7단계: 잔액 생성
        df_toss_standard['잔액'] = clean_amount(df_toss['거래 후 잔액'])
        # 8단계: 메모 생성
        # df_toss_standard['메모'] = df_toss['메모'].replace('', pd.NA)
        df_toss_standard['메모'] = combine_and_clean_str(df_toss, ['메모', '계좌번호'])
        # 9단계: 파일 출처 추가
        df_toss_standard['출처'] = '토스뱅크'
        
        list_of_dfs.append(df_toss_standard[FINAL_COLUMNS].copy())
    except Exception as e:
        print(f"⚠️ [토스뱅크] 처리 중 오류 발생: {e}")

    # --- 4. 현대카드 (총 9단계) ---
    try:
        df_hcard = load_file("현대카드")
        df_hcard_standard = pd.DataFrame(index=df_hcard.index)
        
        # 1단계: 거래일시 생성
        date_str = df_hcard['이용일'].astype(str).str.replace(r'[년월일]', '', regex=True).str.strip()
        date_str_clean = date_str.str.replace(' ', '.', regex=False)
        datetime_combined = date_str_clean + ' 00:00:00'
        df_hcard_standard['거래일시'] = pd.to_datetime(datetime_combined, format='%Y.%m.%d %H:%M:%S', errors='coerce').dt.strftime('%Y.%m.%d %H:%M:%S').replace('NaT', pd.NA)

        df_hcard['이용금액_clean'] = clean_amount(df_hcard['이용금액']).fillna(0)
        # 2단계: 사용구분 생성
        df_hcard_standard['사용구분'] = '출금'
        # 3단계: 사용내역 생성
        df_hcard_standard['사용내역'] = combine_and_clean_str(df_hcard, ['카드구분', '카드명(카드번호 뒤 4자리)'])
        # 4단계: 거래상대방 생성
        # df_hcard_standard['거래상대방'] = combine_and_clean_str(df_hcard, ['가맹점명', '사업자번호'])
        df_hcard_standard['거래상대방'] = df_hcard['가맹점명']
        # 5단계: 입금액 생성
        df_hcard_standard['입금액'] = 0.0
        # 6단계: 출금액 생성
        df_hcard_standard['출금액'] = df_hcard['이용금액_clean']
        # 7단계: 잔액 생성
        df_hcard_standard['잔액'] = pd.NA
        # 8단계: 메모 생성
        df_hcard['메모_할부_정리'] = df_hcard['할부 개월'].astype(str).str.strip()
        df_hcard['메모_할부_정리'] = df_hcard['메모_할부_정리'].replace('00개월', '일시불')
        df_hcard_standard['메모'] = combine_and_clean_str(df_hcard, ['승인번호', '메모_할부_정리', '사업자번호'])
        # 9단계: 파일 출처 추가
        df_hcard_standard['출처'] = '현대카드'
        
        list_of_dfs.append(df_hcard_standard[FINAL_COLUMNS].copy())
    except Exception as e:
        print(f"⚠️ [현대카드] 처리 중 오류 발생: {e}")

    # --- 5. 농협_혜진 (총 9단계) ---
    try:
        df_nh = load_file("농협_혜진")
        df_nh_standard = pd.DataFrame(index=df_nh.index)

        # 1단계: 거래일시 생성
        datetime_series = df_nh['거래일시'].astype(str).str.replace(r'\/', '.', regex=True).str.strip()
        df_nh_standard['거래일시'] = pd.to_datetime(datetime_series, format='%Y.%m.%d %H:%M:%S', errors='coerce').dt.strftime('%Y.%m.%d %H:%M:%S').replace('NaT', pd.NA)
        
        df_nh['입금금액_clean'] = clean_amount(df_nh['입금금액']).fillna(0)
        df_nh['출금금액_clean'] = clean_amount(df_nh['출금금액']).fillna(0)
        # 2단계: 사용구분 생성
        # df_nh_standard['사용구분'] = np.select([df_nh['입금금액_clean'] > 0, df_nh['출금금액_clean'] > 0], ['입금', '출금'], default=pd.NA)
        df_nh_standard['사용구분'] = np.select(
            [
                df_nh['출금금액_clean'] < 0,  # 조건 1: 출금금액이 음수인 경우 (취소/환불)
                df_nh['입금금액_clean'] > 0,  # 조건 2: 입금금액이 양수인 경우
                df_nh['출금금액_clean'] > 0   # 조건 3: 출금금액이 양수인 경우
            ],
            [
                '취소',
                '입금', 
                '출금'
            ],
            default=pd.NA
        )

        # 3단계: 사용내역 생성
        df_nh_standard['사용내역'] = df_nh['거래내용'].replace('', pd.NA)
        # 4단계: 거래상대방 생성
        # df_nh_standard['거래상대방'] = combine_and_clean_str(df_nh, ['거래기록사항', '거래점'])
        df_nh_standard['거래상대방'] = df_nh['거래기록사항'].replace('', pd.NA)
        # 5단계: 입금액 생성
        df_nh_standard['입금액'] = df_nh['입금금액_clean']
        # 6단계: 출금액 생성
        df_nh_standard['출금액'] = df_nh['출금금액_clean']
        # 7단계: 잔액 생성
        df_nh_standard['잔액'] = clean_amount(df_nh['거래후잔액'])
        # 8단계: 메모 생성
        # df_nh_standard['메모'] = df_nh['거래메모'].replace('', pd.NA)
        df_nh_standard['메모'] = combine_and_clean_str(df_nh, ['거래점', '거래메모'])
        # 9단계: 파일 출처 추가
        df_nh_standard['출처'] = '농협_혜진'
        
        list_of_dfs.append(df_nh_standard[FINAL_COLUMNS].copy())
    except Exception as e:
        print(f"⚠️ [농협_혜진] 처리 중 오류 발생: {e}")

    # --- 6. 신한은행 (총 9단계) ---
    try:
        df_shinhan = load_file("신한은행")
        df_shinhan_standard = pd.DataFrame(index=df_shinhan.index)
        df_shinhan.columns = df_shinhan.columns.str.replace(r'\(원\)', '', regex=True).str.strip()
        
        # 1단계: 거래일시 생성
        datetime_combined = df_shinhan['거래일자'].astype(str) + ' ' + df_shinhan['거래시간'].astype(str)
        df_shinhan_standard['거래일시'] = pd.to_datetime(datetime_combined, format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.strftime('%Y.%m.%d %H:%M:%S').replace('NaT', pd.NA)

        df_shinhan['입금_clean'] = clean_amount(df_shinhan['입금']).fillna(0)
        df_shinhan['출금_clean'] = clean_amount(df_shinhan['출금']).fillna(0)
        # 2단계: 사용구분 생성
        df_shinhan_standard['사용구분'] = np.select([df_shinhan['입금_clean'] > 0, df_shinhan['출금_clean'] > 0], ['입금', '출금'], default=pd.NA)
        # 3단계: 사용내역 생성
        # df_shinhan_standard['사용내역'] = combine_and_clean_str(df_shinhan, ['적요', '내용'])
        df_shinhan_standard['사용내역'] = df_shinhan['적요']
        # 4단계: 거래상대방 생성
        df_shinhan_standard['거래상대방'] = combine_and_clean_str(df_shinhan, ['내용', '거래점'])
        # 5단계: 입금액 생성
        df_shinhan_standard['입금액'] = df_shinhan['입금_clean']
        # 6단계: 출금액 생성
        df_shinhan_standard['출금액'] = df_shinhan['출금_clean']
        # 7단계: 잔액 생성
        df_shinhan_standard['잔액'] = clean_amount(df_shinhan['잔액'])
        # 8단계: 메모 생성
        df_shinhan_standard['메모'] = pd.NA
        # 9단계: 파일 출처 추가
        df_shinhan_standard['출처'] = '신한은행'
        
        list_of_dfs.append(df_shinhan_standard[FINAL_COLUMNS].copy())
    except Exception as e:
        print(f"⚠️ [신한은행] 처리 중 오류 발생: {e}")

    
    # 최종 통합 및 타입 정리
    if list_of_dfs:
        df_combined = pd.concat(list_of_dfs, ignore_index=True)
        # 날짜 타입 재정의 (정렬 및 분석 용이)
        df_combined['거래일시'] = pd.to_datetime(df_combined['거래일시'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        df_combined['입금액'] = df_combined['입금액'].astype(float)
        df_combined['출금액'] = df_combined['출금액'].astype(float)
        return df_combined
    else:
        return pd.DataFrame(columns=FINAL_COLUMNS)

# ----------------------------------------------------------------------
# 4. 메인 실행 파이프라인 (저장 및 분석 모드 통합)
# ----------------------------------------------------------------------

def run_data_integration_pipeline():
    """
    1. 6개 원천 파일을 통합 및 전처리합니다.
    2. 통합된 DataFrame을 'log/' 경로에 타임스탬프가 찍힌 TSV 파일로 저장합니다.
    3. 통합된 DataFrame을 반환하여 즉시 분석에 사용할 수 있도록 합니다.
    """
    
    # 4.1. 파일 처리 및 통합
    final_df = process_all_files(FILE_MAP) 

    if final_df.empty:
        print("\n**통합할 데이터가 없습니다. 파일 경로 또는 내용을 확인해주세요.**")
        return None
        
    # 4.1.1. 필터링 로직 추가
    FILTER_DATE_STR = '2025.08.04 00:00:00'
    filter_start_date = pd.to_datetime(FILTER_DATE_STR, format='%Y.%m.%d %H:%M:%S', errors='coerce')

    # 주의: 거래일시 컬럼이 NaT인 값(파싱 오류)은 필터링에서 제외됩니다.
    final_df = final_df[final_df['거래일시'] >= filter_start_date].reset_index(drop=True)
    
    if final_df.empty:
        print(f"\n**필터링 기준({FILTER_DATE_STR} 이후)을 만족하는 거래 내역이 없습니다.**")
        return None

    # 4.2. 최종 정렬
    # 거래 일시를 기준으로 최신순 정렬
    final_df = final_df.sort_values(
        # by=['출처', '거래일시'], 
        # ascending=[True, False]
        by=['거래일시'], 
        ascending=[False]
    ).reset_index(drop=True)

    # 4.3. 파일 저장 (사용자 요청: tsv 파일로 저장)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file_name = f'integrated_transactions_{timestamp}.tsv'
    output_path = LOG_DIR_PATH / output_file_name 
    
    try:
        final_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print(f"**총 거래 건수:** {len(final_df)}건")
        print(f"**✅ 통합 데이터 전체 내역을 TSV 파일로 성공적으로 저장했습니다.**")
        print(f"**저장 경로:** {output_path}")
    except Exception as e:
        print(f"\n⚠️ 데이터 저장 실패: {e}")
    
    # 4.4. 데이터 분석 준비 (DataFrame 반환)
    print("\n## 데이터 분석 준비 완료")
    print("**통합된 DataFrame을 바로 사용할 수 있도록 반환합니다.**")
    print("="*50)
    
    return final_df


if __name__ == "__main__":
    integrated_data = run_data_integration_pipeline() 
    
    if integrated_data is not None:
        print("\n\n*이제 'integrated_data' 변수를 사용하여 데이터 분석을 시작하세요.*")
    pass