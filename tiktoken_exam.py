# 참고 URL: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

# 모듈 설치
# pip install --upgrade tiktoken
# pip install --upgrade openai

# 1. tiktoken 모듈을 가져오기
import tiktoken  # tiktoken 모듈을 불러와서 사용할 준비를 합니다.

# 2. 인코딩 로드하기
encoding = tiktoken.get_encoding("cl100k_base")  # "cl100k_base"라는 인코딩 방식을 가져옵니다.
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # GPT-3.5-turbo 모델을 위한 인코딩을 가져옵니다.

# 3. 텍스트를 인코딩(토큰으로 변환)하기
encode_exam = encoding.encode("tiktoken is great!")  # "tiktoken is great!" 텍스트를 토큰으로 변환합니다.
print(f"Encoded: {encode_exam}")
#Encoded: [83, 1609, 5963, 374, 2294, 0]  # "tiktoken is great!" 문장이 토큰 리스트로 변환된 결과입니다.

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """주어진 텍스트 문자열의 토큰 개수를 반환합니다."""
    encoding = tiktoken.get_encoding(encoding_name)  # 주어진 인코딩 이름으로 인코딩을 가져옵니다.
    num_tokens = len(encoding.encode(string))  # 문자열을 토큰으로 변환하고, 그 길이를 계산합니다.
    return num_tokens  # 토큰 개수를 반환합니다.

num_token = num_tokens_from_string("tiktoken is great!", "cl100k_base")
print(f"Number of tokens: {num_token}")
# Number of tokens: 6  # "tiktoken is great!" 문장은 6개의 토큰으로 변환되었습니다.

# 4. 토큰을 텍스트로 디코딩하기
decode_exam = encoding.decode([83, 1609, 5963, 374, 2294, 0])  # 토큰 리스트를 다시 텍스트로 변환합니다.
print(f"Decoded: {decode_exam}")
# Decoded: tiktoken is great!  # 토큰 리스트가 원래의 텍스트 "tiktoken is great!"로 디코딩되었습니다.

single_decode = [encoding.decode_single_token_bytes(token) for token in [83, 1609, 5963, 374, 2294, 0]]
print(f"Decoded: {single_decode}")
# Decoded: [b't', b'ik', b'token', b' is', b' great', b'!']  # 각 토큰이 어떤 바이트로 디코딩되는지 확인할 수 있습니다.

# 5. 다양한 인코딩 방식 비교하기

def compare_encodings(example_string: str) -> None:
    """세 가지 인코딩 방식에 대한 비교를 출력합니다."""
    # 예시 문자열 출력
    print(f'\nExample string: "{example_string}"')
    # 각 인코딩에 대해 토큰 개수, 토큰 정수, 토큰 바이트를 출력합니다.
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)  # 인코딩을 로드합니다.
        token_integers = encoding.encode(example_string)  # 문자열을 토큰 정수로 변환합니다.
        num_tokens = len(token_integers)  # 토큰의 개수를 계산합니다.
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]  # 각 토큰을 바이트로 디코딩합니다.
        print()
        print(f"{encoding_name}: {num_tokens} tokens")  # 인코딩 방식별 토큰 개수 출력
        print(f"token integers: {token_integers}")  # 토큰 정수 리스트 출력
        print(f"token bytes: {token_bytes}")  # 토큰 바이트 리스트 출력

# 예시 문자열로 인코딩 비교 실행
compare_encodings("antidisestablishmentarianism")
"""
Example string: "antidisestablishmentarianism"

r50k_base: 5 tokens
token integers: [415, 29207, 44390, 3699, 1042]
token bytes: [b'ant', b'idis', b'establishment', b'arian', b'ism']

p50k_base: 5 tokens
token integers: [415, 29207, 44390, 3699, 1042]
token bytes: [b'ant', b'idis', b'establishment', b'arian', b'ism']

cl100k_base: 6 tokens
token integers: [519, 85342, 34500, 479, 8997, 2191]
token bytes: [b'ant', b'idis', b'establish', b'ment', b'arian', b'ism']
"""  
# "antidisestablishmentarianism" 단어가 세 가지 인코딩 방식에 따라 어떻게 토큰화되고, 바이트로 변환되는지를 비교합니다.

compare_encodings("2 + 2 = 4")
"""
Example string: "2 + 2 = 4"

r50k_base: 5 tokens
token integers: [17, 1343, 362, 796, 604]
token bytes: [b'2', b' +', b' 2', b' =', b' 4']

p50k_base: 5 tokens
token integers: [17, 1343, 362, 796, 604]
token bytes: [b'2', b' +', b' 2', b' =', b' 4']

cl100k_base: 7 tokens
token integers: [17, 489, 220, 17, 284, 220, 19]
token bytes: [b'2', b' +', b' ', b'2', b' =', b' ', b'4']
"""
# "2 + 2 = 4" 문자열이 세 가지 인코딩 방식에 따라 다르게 토큰화되는지 확인합니다.

compare_encodings("안녕하세요, 반갑습니다.")
"""
Example string: "안녕하세요, 반갑습니다."

r50k_base: 29 tokens
token integers: [168, 243, 230, 167, 227, 243, 47991, 246, 168, 226, 116, 168, 248, 242, 11, 31619, 108, 246, 166, 108, 239, 168, 232, 113, 46695, 230, 46695, 97, 13]
token bytes: [b'\xec', b'\x95', b'\x88', b'\xeb', b'\x85', b'\x95', b'\xed\x95', b'\x98', b'\xec', b'\x84', b'\xb8', b'\xec', b'\x9a', b'\x94', b',', b' \xeb', b'\xb0', b'\x98', b'\xea', b'\xb0', b'\x91', b'\xec', b'\x8a', b'\xb5', b'\xeb\x8b', b'\x88', b'\xeb\x8b', b'\xa4', b'.']

p50k_base: 29 tokens
token integers: [168, 243, 230, 167, 227, 243, 47991, 246, 168, 226, 116, 168, 248, 242, 11, 31619, 108, 246, 166, 108, 239, 168, 232, 113, 46695, 230, 46695, 97, 13]
token bytes: [b'\xec', b'\x95', b'\x88', b'\xeb', b'\x85', b'\x95', b'\xed\x95', b'\x98', b'\xec', b'\x84', b'\xb8', b'\xec', b'\x9a', b'\x94', b',', b' \xeb', b'\xb0', b'\x98', b'\xea', b'\xb0', b'\x91', b'\xec', b'\x8a', b'\xb5', b'\xeb\x8b', b'\x88', b'\xeb\x8b', b'\xa4', b'.']

cl100k_base: 11 tokens
token integers: [31495, 230, 75265, 243, 92245, 11, 64857, 14705, 239, 39331, 13]
token bytes: [b'\xec\x95', b'\x88', b'\xeb\x85', b'\x95', b'\xed\x95\x98\xec\x84\xb8\xec\x9a\x94', b',', b' \xeb\xb0\x98', b'\xea\xb0', b'\x91', b'\xec\x8a\xb5\xeb\x8b\x88\xeb\x8b\xa4', b'.']
"""
# "안녕하세요, 반갑습니다."라는 한국어 문장이 세 가지 인코딩 방식에서 어떻게 다르게 토큰화되고 디코딩되는지 비교합니다.
