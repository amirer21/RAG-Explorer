# case 2. langchain_core.messages

# https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.messages

import openai
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
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
model = ChatOpenAI(
    model="gpt-4o",
)

messages = [
    SystemMessage(content="SystemMessage"),
    HumanMessage(content="Hello!"),
]


result = model.invoke(messages)
print(f"result :: {result}")
# result :: content='Hi there! How can I help you today?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 15, 'total_tokens': 25}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None} id='run-f9134e87-4721-4213-9da5-d9d1f9a55b6a-0' usage_metadata={'input_tokens': 15, 'output_tokens': 10, 'total_tokens': 25}

content = result.content
print(f"content :: {content}")
# content :: Hi there! How can I help you today?