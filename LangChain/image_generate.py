import openai
import requests   # url 페이지를 읽어오는 것
from PIL import Image  # 이미지 표시
from io import BytesIO  #이미지를 바이트 단위로 입출력
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


response=openai.images.generate(
  model="dall-e-3",#모델
  prompt="A cute baby sea otter",
  n=1, # 이미지 개수
  size="1024x1024" #이미지 사이즈
)
#print(response.data[0].url)
url  = response.data[0].url   # 이미지 생성 주소를 리턴 한다
res_img  = requests.get(url)  #요청된 웹 페이지 이미지를 다운로드 해서 읽어온다.
m_img  = Image.open( BytesIO (res_img.content))  # 읽어온 이미지를 byte로 변환해서 오픈한다.
m_img.show()  #이미지를 확인 한다.