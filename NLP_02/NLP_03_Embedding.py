import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

# [1] 샘플 텍스트 데이터 준비
# 음식에 대한 후기 샘플 데이터
sentences = [
    '이 음식은 정말 맛있습니다',
    '별로 맛이 없어요',
    '이 음식을 추천합니다',
    '맛이 매우 별로입니다'
]

# [2] 토큰화 (Tokenization)
# 텍스트 데이터를 정수 시퀀스로 변환하기 위한 Tokenizer 객체 생성 및 학습
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)  # 문장을 정수 시퀀스로 변환

# [3] 패딩 (Padding)
# 정수 시퀀스의 길이를 고정된 값으로 맞추기 위해 패딩 추가
maxlen = 5
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

# [4] 임베딩 레이어 정의 (Embedding Layer)
# 단어의 정수 인덱스를 고정된 크기의 벡터로 변환하기 위한 임베딩 레이어 정의
model = Sequential([
    Embedding(input_dim=100, output_dim=8, input_length=maxlen)  # input_length를 통해 입력 시퀀스 길이 설정
])

# [5] 임베딩 벡터 생성 (Generate Embedding Vectors)
# 정의된 임베딩 레이어를 사용하여 입력된 패딩 시퀀스의 임베딩 벡터 생성
embedding_output = model.predict(padded_sequences)

# [6] 임베딩 결과 출력 (Output Embedding Vectors)
print("임베딩 벡터 출력 (각 텍스트에 대해):")
print(embedding_output)
