import os
import time
import json
import random # ✨ [추가] 랜덤 지연(Jitter)을 위해 random 모듈 import
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError # ✨ [추가] API 오류 처리를 위해 import

# ----------------------------------------------------------------------
# 1. 환경 설정 및 API 클라이언트 초기화 (기존 로직 유지)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent if '__file__' in locals() else Path('.')
DOTENV_PATH = PROJECT_ROOT / '.env'

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)
    print(f"✅ .env 파일에서 환경 변수를 성공적으로 로드했습니다. 경로: {DOTENV_PATH}")
else:
    print(f"❌ .env 파일을 찾을 수 없습니다: {DOTENV_PATH}")

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_CLIENT = None
# TARGET_MODEL = 'gemini-2.5-flash'
TARGET_MODEL = 'gemini-2.5-flash-lite'
RPM_DELAY_SECONDS = 1.5
MAX_RETRIES = 3 # ✨ [추가] 503 오류 발생 시 최대 3번 재시도 설정 (총 4번 시도)

if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client()
        print(f"✅ Gemini API 클라이언트 초기화 완료. 모델: {TARGET_MODEL}")
    except Exception as e:
        print(f"⚠️ Gemini API 클라이언트 초기화 실패: {e}")
        
# ----------------------------------------------------------------------
# 2. JSON 스키마 및 프롬프트 정의 (AI 자율 판단으로 업데이트)
# ----------------------------------------------------------------------

# 2.1. 시스템 프롬프트 (AI가 카테고리를 자유롭게 판단하도록 요청)

DATA_HEADER = "거래일시	사용구분	사용내역	거래상대방	입금액	출금액	잔액	메모	출처"

DATA_CATEGORY = {
  "식비": [
    "외식/음식점/맘스푸드",
    "카페/음료/간식/디저트",
    "배달음식",
    "편의점",
    "마트",
    "주류 및 모임 비용",
    "기타 식비"
  ],
  "쇼핑": [
    "백화점 등 오프라인 쇼핑",
    "온라인 쇼핑 및 다이소",
    "기타 쇼핑"
  ],
  "이체/송금": [
    "개인 간 송금 (타인)",
    "내 계좌 간 이체 (본인)",
    "환불/취소",
    "자동이체",
    "기타 이체/송금"
  ],
  "금융": [
    "카드 대금 납부 (신용카드)",
    "투자, 저축 (증권/펀드)",
    "대출 원리금 (이자)",
    "은행 수수료, 이용료",
    "간편결제",
    "아파트 관리비",
    "가스 요금 결제",
    "기타 주거/공과금",
    "보험료",
    "기타 금융 활동"
  ],
  "소득": [
    "월급",
    "이자, 리워드, 캐시백, 포인트 등",
    "세금 환급",
    "기타 소득"
  ],
  "교통": [
    "대중교통 대금 결제 및 충전",
    "유류비, 세차 등 주유소 비용",
    "고속도로 통행료",
    "주차비",
    "기타 교통비"
  ],
  "통신/구독": [
    "휴대폰 통신비",
    "인터넷 통신비",
    "월 서비스 구독료",
    "기타 통신/구독"
  ],
  "여가/문화": [
    "영화 공연 관광 공원 등 관람비 또는 입장료",
    "도서 구매 비용",
    "여행 경비(비행기, 숙박비 등)",
    "기타 여가/문화"
  ],
  "건강": [
    "보험",
    "병원 약국 진료비",
    "올리브영 등 화장품 구매",
    "필라무드 등 운동비 결제",
    "기타 건강"
  ],
  "기타": [
    "경조사비",
    "과태료",
    "수수료 (비금융)",
    "기타 미분류"
  ]
}

SYSTEM_PROMPT = """
1. 지침
당신은 금융 거래 내역을 전문적으로 분류하는 AI입니다.

2. 데이터
사용자로부터 '탭'(\t)으로 구분된 금융 거래 내역 데이터 리스트를 받습니다.
입력 데이터의 컬럼 순서는 다음과 같습니다:
{DATA_HEADER}

3. 카테고리
데이터 분류 기준에 대한 카테고리는 "주요_카테고리": [세부_카테고리] 의 JSON 으로 구성되어 있습니다.
카테고리 리스트는 다음과 같습니다:
{DATA_CATEGORY}

당신은 이 데이터를 행 단위로 분석하여 각 거래에 대해 '거래_유형', '주요_카테고리', '세부_카테고리', '판단_사유'를 추론해야 합니다.
결과 JSON 리스트의 각 객체에는 원본 문장('인풋_문장')을 포함해야 하며, 어떠한 설명 없이 오직 JSON 리스트 형식으로만 응답해야 합니다.

다음을 순차적으로 수행하세요.

1. 제공되는 모든 결제 문자열 리스트를 분석하여
2. 각 항목에 대해 거래가 '입금'인지 '출금'인지 판단하고
3. 해당 거래에 가장 적합한 '주요 카테고리'와 '세부 카테고리'를 기준에 따라 부여하고
4. 그 사유를 1문장으로 명확히 작성해야 합니다.

* 결과는 반드시 JSON 배열 형식으로만 응답해야 합니다.
"""

# 2.2. JSON 스키마 정의: 출력 형식 강제 (필드 4개 + 인풋 문장 1개)
JSON_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "인풋_문장": types.Schema(type=types.Type.STRING, description="입력으로 받은 원본 결제 문자열"),
            "거래_유형": types.Schema(type=types.Type.STRING, enum=['입금', '출금', '취소']),
            "주요_카테고리": types.Schema(type=types.Type.STRING, description="AI가 판단한 가장 적합한 대분류 카테고리"),
            "세부_카테고리": types.Schema(type=types.Type.STRING, description="AI가 판단한 가장 적합한 소분류 카테고리"),
            "판단_사유": types.Schema(type=types.Type.STRING, description="위의 카테고리를 부여한 구체적인 이유")
        },
        required=["인풋_문장", "거래_유형", "주요_카테고리", "세부_카테고리", "판단_사유"]
    )
)

# ----------------------------------------------------------------------
# 3. 메인 배치 분류 함수
# ----------------------------------------------------------------------

def classify_payments_batch(payment_strings: list, client: genai.Client):
    """
    결제 문장 리스트를 받아 Gemini 2.5 Flash로 분류하고 JSON 배열을 반환합니다.
    """
    if not client:
        print("API 클라이언트가 초기화되지 않아 분류를 수행할 수 없습니다.")
        return json.dumps([])

    prompt = (
        f"다음 {len(payment_strings)}개의 결제 내역을 분류하세요. "
        f"결과는 다음 리스트와 개수가 일치하는 JSON 배열이어야 합니다: \n{json.dumps(payment_strings, ensure_ascii=False)}"
    )
    
    # ✨ [수정] 재시도 로직 추가
    for attempt in range(MAX_RETRIES + 1):
      try:
          response = client.models.generate_content(
              model=TARGET_MODEL,
              contents=[{"role": "user", "parts": [{"text": prompt}]}],
              config=types.GenerateContentConfig(
                  system_instruction=SYSTEM_PROMPT,
                  response_mime_type="application/json",
                  response_schema=JSON_SCHEMA
              )
          )
          
          time.sleep(RPM_DELAY_SECONDS) 
          print(f"✅ {len(payment_strings)}건 처리 완료. 다음 호출까지 {RPM_DELAY_SECONDS}초 대기.")

          return response.text

      except APIError as e:
          # 503 UNAVAILABLE 오류이면서 재시도 횟수가 남은 경우
          if "503" in str(e) and attempt < MAX_RETRIES:
              # 지수적 백오프 계산: 2^attempt + Jitter (랜덤 지연)
              wait_time = (2 ** attempt) + random.uniform(0.1, 1.0) 
              
              print(f"⚠️ [503 UNAVAILABLE] 오류 발생. {attempt + 1}차 재시도: {wait_time:.2f}초 후 재시도합니다.")
              time.sleep(wait_time)
              
          else:
              # 503이 아닌 다른 오류이거나, 최대 재시도 횟수를 초과한 경우
              print(f"❌ API 호출 실패 (최대 시도 횟수 초과 또는 치명적인 오류): {e}")
              # 오류 처리 로직 (예: 빈 JSON 반환 또는 오류 발생)
              return "[]" 
              
      except Exception as e:
          # 기타 예기치 못한 파이썬 오류
          print(f"❌ 예기치 않은 오류 발생: {e}")
          return "[]"
    
    # 모든 시도가 실패하고 for 루프를 빠져나온 경우
    return "[]"