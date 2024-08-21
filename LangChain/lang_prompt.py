# case 4 ChatPromptTemplate 프롬프트 생성

# https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.prompts

# 예제 https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html#langchain_core.prompts.chat.ChatPromptTemplate
import openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import io
from contextlib import redirect_stdout
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


#help_text = help(ChatPromptTemplate)

# Create a StringIO buffer to capture the help output
with io.StringIO() as buffer:
    # Use redirect_stdout to capture the output of help() into the buffer
    with redirect_stdout(buffer):
        help(ChatPromptTemplate)
    
    # Get the content of the buffer and write it directly to the file
    with open("chat_prompt_template_help.txt", "w", encoding="utf-8") as file:
        file.write(buffer.getvalue())

print("Help content saved to chat_prompt_template_help.txt")

############################################################
# case 4-1 ChatPromptTemplate 프롬프트 생성

template = " {날씨}에 대해서 알려줘."
res = ChatPromptTemplate.from_template(template)

print("========== 템플릿 생성 결과 ==========")
promt = res.invoke({"날씨" : "제주도"})
print(f"프롬프트 응답값(질문에 대한 응답 결과가 이님. 휴먼 메시지가 출력됨) : {promt}")
# ========== 템플릿 생성 결과 ==========
# 프롬프트 응답값(질문에 대한 응답 결과가 이님. 휴먼 메시지가 출력됨) : messages=[HumanMessage(content=' 제주도에 대해서 알려줘.')]

############################################################
# case 4-2 ChatPromptTemplate 프롬프트 생성 {} {} 다양하게 지정

template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?"
    }
)

print(f"프롬프트 응답값 : {prompt_value}")
# 프롬프트 응답값 : messages=[SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), 
# HumanMessage(content='Hello, how are you doing?'), AIMessage(content="I'm doing well, thanks!"), HumanMessage(content='What is your name?')]


############################################################
# case 4-3 ChatPromptTemplate 프롬프트 생성 : HumanMessage 활용

my_str = "날씨 {제주도}관련 {상품}에 대해서 알려줘."
res = ChatPromptTemplate.from_template(my_str)

print("========== 템플릿 생성 결과 ==========")
promt = res.invoke({"제주도" : "돌하르방", "상품" : "돌"} )
print(f"프롬프트 응답값 : {promt}")
# ========== 템플릿 생성 결과 ==========
# 프롬프트 응답값 : messages=[HumanMessage(content='날씨 돌하르방관련 돌에 대해서 알려줘.')]

############################################################
# case 4-4 ChatPromptTemplate 프롬프트 생성 : Prompt 생성

# messages = [
#     ("system", "당신은 {topic}에 대한 개발을 하는 개발자입니다."),
#     HumanMessage(content="파이썬 예제 코드를 작성해줘."),

# ]
# res = ChatPromptTemplate.from_template(messages)

# print("========== 템플릿 생성 결과 ==========")
# promt = res.invoke({"topic" : "RAG"})
# print(f"프롬프트 응답값 : {promt}")