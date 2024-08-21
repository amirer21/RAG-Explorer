# Langchain docs 사이트에 가서, output_parsers 를 참고한다.
# https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.output_parsers
# https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html#langchain_core.output_parsers.string.StrOutputParser
# 모델 생성 -> 프롬프트 템플릿 생성 -> StrOutputParser -> 연결 -> 실행
import openai
import time
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # .env 파일을 불러오는 패키지
import os
#os.environ["OPENAI_API_KEY"] = "your-api-key"
# .env 파일의 환경 변수를 로드
load_dotenv(dotenv_path='openapi_key.env')
# .env 파일에서 API 키를 가져옴
api_key = os.getenv("OPENAI_API_KEY")
# API 키 확인
if api_key:
    print("API 키가 정상적으로 로드되었습니다.")
else:
    print("API 키를 로드할 수 없습니다. .env 파일을 확인하세요.")

# API 키를 직접 설정
openai.api_key = api_key

if not openai.api_key:
    raise ValueError("API 키를 로드할 수 없습니다. .env 파일을 확인하세요.")
# 시작 시간 기록
start_time = time.time()

# 1. 모델 생성
model = ChatOpenAI(model="gpt-4o")

# 2. 프롬프트 템플릿 생성
message = "문장을 입력하면 영어로 변환해줘. {text}"
res = ChatPromptTemplate.from_template(message)

print("========== 템플릿 생성 결과 ==========")
promt = res.invoke({"text" : "안녕하세요."})
print(f"prompt 결과 : {promt}")

# 3. StrOutputParser 생성
output_parser = StrOutputParser()

# 4. 체인 연결
chain = res | model | output_parser  # 파이프 (|)로 체인 연결 !!!

# 5. 체인 실행
result = chain.invoke({"text": "안녕하세요."})
print(f"result :: {result}")

# result :: Hello.