# case 1. Open AI invoke()
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
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)

print(llm)
"""
client=<openai.resources.chat.completions.Completions object at 0x000001C0C7EDEC10> async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001C0C7EEB190> root_client=<openai.OpenAI object at 0x000001C0C7C87C90>
 root_async_client=<openai.AsyncOpenAI object at 0x000001C0C7EDEDD0> model_name='gpt-4o' temperature=0.0 openai_api_key=SecretStr('**********') openai_proxy=''
"""

print(f"model {type(llm)}")

result = llm.invoke("서울시 날씨는 어때?")
print(result)

# 파트너 사의 모델, 속성을 그대로 사용하므로, 다른 결과가 아니다.
"""
content='현재 시점에서 실시간 날씨 정보를 제공할 수는 없습니다. 서울시의 최신 날씨 정보를 확인하려면 
기상청 웹사이트나 날씨 애플리케이션을 이용해 보세요. 네이버 날씨, 다음 날씨, 또는 구글에서 "서울 날씨"를 검색해도 실시간 정보를 얻을 수 있습니다.
' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 74, 'prompt_tokens': 15, 'total_tokens': 89}, 
'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27', 'finish_reason': 'stop', 'logprobs': None} 
id='run-87f6a75d-986f-42e6-b463-24d08216023d-0' usage_metadata={'input_tokens': 15, 'output_tokens': 74, 'total_tokens': 89}
"""

#content
print(f"content :: {result.content}")
"""
content :: 현재 시점에서 실시간 날씨 정보를 제공할 수는 없지만, 서울의 날씨를 확인하려면 다음과 같은 방법을 사용할 수 있습니다:

1. **기상청 웹사이트**: 대한민국 기상청 웹사이트에서 최신 날씨 정보를 확인할 수 있습니다.
2. **날씨 애플리케이션**: 스마트폰의 날씨 애플리케이션을 통해 실시간 날씨 정보를 확인할 수 있습니다.
3. **포털 사이트**: 네이버, 다음 등 포털 사이트에서도 날씨 정보를 제공합니다.
4. **뉴스**: TV나 라디오 뉴스에서도 날씨 정보를 확인할 수 있습니다.

이 방법들을 통해 서울의 현재 날씨를 확인해 보세요.
"""