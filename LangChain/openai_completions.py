# pip install -U langchain-openai
# https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI

import openai
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
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # api_key="...",
#     # base_url="...",
#     # organization="...",
#     # other params...
# )

#https://platform.openai.com/docs/guides/text-generation
#https://platform.openai.com/docs/api-reference/chat/create?lang=python

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = api_key
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "user", "content": "오늘 날씨는 어때?"}
  ],
  max_tokens=100,
  temperature=0.5  # 결과 파라미터 창의성
)

print(completion.choices[0].message.content)
print(completion.choices[0].message)
