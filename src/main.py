import re
import os
import time
import json
import pandas as pd
from pathlib import Path

import get_data_v3 
from api_genai import GEMINI_CLIENT, RPM_DELAY_SECONDS, classify_payments_batch 

# ----------------------------------------------------------------------
# 1. 환경 설정 및 상수 정의
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in locals() else Path('.')
LOG_DIR_PATH = PROJECT_ROOT / 'log'
LOG_DIR_PATH.mkdir(exist_ok=True)

# AI 배치 처리 설정
MAX_BATCH_SIZE = 20
# get_data_v3에서 사용하는 원본 컬럼 목록
FINAL_COLUMNS = ['거래일시', '사용구분', '사용내역', '거래상대방', '입금액', '출금액', '잔액', '메모', '출처']
# AI가 리턴하는 결과 컬럼 목록
AI_RESULT_COLUMNS = ['인풋_문장', '거래_유형', '주요_카테고리', '세부_카테고리', '판단_사유']


def run_full_pipeline():
    """
    데이터 통합, AI 분류(인덱스 기반), 결과 병합 및 최종 TSV 저장을 오케스트레이션합니다.
    """
    
    # 1단계: 원본 데이터 통합 및 준비 (get_data_v3.py 호출)
    print("1단계: 원본 데이터 통합 및 전처리 (get_data_v3.py 호출)")
    # df_original은 reset_index(drop=True)를 거쳐 0부터 시작하는 깨끗한 인덱스를 가짐
    df_original = get_data_v3.run_data_integration_pipeline()

    if df_original is None or df_original.empty:
        print("❌ 통합 데이터가 없어 파이프라인을 종료합니다.")
        return

    # AI 입력 데이터 준비: TSV 문자열 리스트
    data_length = len(df_original)
    
    # AI 입력 데이터는 TSV 문자열 형식이어야 합니다.
    # 인덱스 대신 순수 데이터만 포함하여 AI에 전달합니다.
    data_for_ai = df_original[FINAL_COLUMNS].to_csv(
        sep='\t', header=False, index=False
    ).strip().split('\n')
    
    # 2단계: AI 배치 처리 및 결과 수집 (api_genai.py 호출)
    print(f"2단계: AI 배치 분류 시작 (인덱스 기반 병합, 총 {data_length}건)")

    df_ai_results_list = [] # 각 배치 결과를 담을 DataFrame 리스트
    
    for i in range(0, data_length, MAX_BATCH_SIZE):
        batch = data_for_ai[i:i + MAX_BATCH_SIZE]
        batch_num = i // MAX_BATCH_SIZE + 1
        total_batches = int((data_length + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE)
        
        print(f"\n--- [배치 {batch_num} / 총 {total_batches} ({len(batch)}건 처리)] ---")
        
        if GEMINI_CLIENT:
            result_json_string = classify_payments_batch(batch, GEMINI_CLIENT)
            time.sleep(2.0) 
        else:
            print("❌ AI 클라이언트 초기화 실패. 분류를 건너뜁니다.")
            break

        # JSON 파싱 및 DataFrame 생성
        try:
            # AI 응답 파싱 안정화: 정규식을 사용해 유효한 JSON 배열 '[...]'만 추출
            match = re.search(r'\[.*\]', result_json_string, re.DOTALL)
            
            if not match:
                print(f"❌ 배치 {batch_num} JSON 배열([...] 포맷)을 찾을 수 없습니다. 원본 응답: {result_json_string[:100]}...")
                time.sleep(RPM_DELAY_SECONDS)
                continue
                
            cleaned_json_string = match.group(0)
            parsed_json_list = json.loads(cleaned_json_string)
            
            # 핵심: JSON 리스트를 바로 DataFrame으로 변환
            df_batch_result = pd.DataFrame(parsed_json_list)
            
            # 입력 데이터 순서와 AI 결과 순서가 동일하므로, 인덱스를 재정의하고 추가
            # 인덱스를 0부터 시작하도록 재설정
            df_batch_result = df_batch_result.reset_index(drop=True) 

            # AI 결과 컬럼만 선택하고 리스트에 추가
            if not df_batch_result.empty:
                 # 필요한 AI 컬럼만 선택 (인풋_문장 포함)
                selected_columns = [col for col in AI_RESULT_COLUMNS if col in df_batch_result.columns]
                df_ai_results_list.append(df_batch_result[selected_columns])
                print(f"✅ 배치 {batch_num} 결과 {len(df_batch_result)}건 통합 완료.")

        except json.JSONDecodeError as e:
            print(f"❌ 배치 {batch_num} JSON 파싱 오류 발생. 원인: {e}. 원본 응답 확인 필요.")
            time.sleep(RPM_DELAY_SECONDS)
            continue
        except Exception as e:
            print(f"❌ 배치 {batch_num} 처리 중 예외 발생: {e}")
            time.sleep(RPM_DELAY_SECONDS)
            continue
            
        time.sleep(RPM_DELAY_SECONDS) # API 호출 간 지연 시간

    
    # 3단계: 최종 결과 병합 및 파일 저장
    print("3단계: 최종 결과 병합 및 TSV 파일 저장")

    if not df_ai_results_list:
        print("❌ AI 분류 결과가 없어 최종 파일을 생성할 수 없습니다.")
        return

    # 3.1. 최종 AI 결과 DataFrame 생성
    # ignore_index=True로 모든 배치 결과를 0부터 시작하는 연속 인덱스로 합침
    df_ai_final = pd.concat(df_ai_results_list, ignore_index=True)

    # 3.2. 원본 DataFrame과 AI 결과 DataFrame 병합
    # 핵심: 인덱스 기반으로 횡방향(axis=1) 병합. 순서가 보장되어야 함.
    # 병합 전에 df_original이 df_ai_final과 행 개수가 같아야 함.
    if len(df_original) != len(df_ai_final):
        print(f"⚠️ 경고: 원본 데이터({len(df_original)}건)와 AI 결과({len(df_ai_final)}건)의 개수가 불일치합니다. TSV 파일에는 병합된 결과({len(df_ai_final)}건)만 포함됩니다.")
    
    # AI 결과의 '인풋_문장'은 최종 파일에 불필요하므로 제외
    df_ai_final_clean = df_ai_final.drop(columns=['인풋_문장'], errors='ignore')
    
    # 원본 DataFrame의 인덱스를 '순번' 컬럼으로 사용하여 최종 TSV에 포함
    df_original['순번'] = df_original.index + 1
    
    # 인덱스 기반 병합
    # concat(axis=1)은 같은 인덱스(0, 1, 2, ...)를 가진 두 DataFrame을 옆으로 붙여줍니다.
    df_final_merged = pd.concat([df_original, df_ai_final_clean], axis=1)

    # 3.3. 최종 TSV 파일 저장
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    tsv_file_name = f'final_integrated_classified_{timestamp}.tsv'
    tsv_output_path = LOG_DIR_PATH / tsv_file_name 

    try:
        # 최종 컬럼 순서 재정의: 순번 + 원본 컬럼 + AI 컬럼
        # AI_COLUMNS는 '거래_유형', '주요_카테고리', '세부_카테고리', '판단_사유'
        AI_COLUMNS_ONLY = [c for c in df_ai_final_clean.columns if c != '인풋_문장']
        final_columns_order = ['순번'] + FINAL_COLUMNS + AI_COLUMNS_ONLY
        
        df_final_merged[final_columns_order].to_csv(tsv_output_path, sep='\t', index=False, encoding='utf-8')
        print(f"**🌟 최종 TSV 파일 저장 성공:** {tsv_output_path} ({len(df_final_merged)}건)")
    except Exception as e:
        print(f"❌ 최종 TSV 파일 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    run_full_pipeline()