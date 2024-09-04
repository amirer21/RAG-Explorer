# 패키지 설치 
# !pip install tensorflow konlpy transformers tf-keras

# 자연어 전처리

# [1] 토큰화 (Tokenization) : 문장을 형태소 단위로 분리
import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 분석할 문장
sentence = "나는 매일 밤 꿈을 꾸고 깨고 나면 대부분을 기억한다."

# 형태소 분석기로 문장을 토큰화
okt = Okt()
tokens = okt.morphs(sentence)
print("Tokens:", tokens)

# [2] 불용어 제거 (Stopword Removal) : 의미가 없는 불용어(예: 조사, 접속사 등)를 제거
stopwords = ['는', '을', '고', '면']
tokens = [word for word in tokens if word not in stopwords]
print("Tokens after stopword removal:", tokens)

# [3] 어간 추출 (Stemming) : 단어의 어간을 추출, 여기서는 기본형으로 변환했으므로 추가 단계 없음
# (한글에서는 형태소 분석기로 기본형 변환을 수행하였으므로 어간 추출 단계는 필요 없음)

# [4] 시퀀스 변환 및 패딩 (Sequence Conversion and Padding) : 문장을 숫자로 변환하고 고정 길이로 패딩
from tensorflow.keras.utils import to_categorical

# 토큰 리스트를 시퀀스로 변환하기 위한 토크나이저 설정
tokenizer = Tokenizer()
tokenizer.fit_on_texts([tokens])  # 텍스트를 학습하여 단어에 인덱스를 부여
sequences = tokenizer.texts_to_sequences([tokens])  # 텍스트를 숫자 시퀀스로 변환
print("Sequences:", sequences)

# 시퀀스 길이를 고정된 길이로 패딩 (모델 입력 크기 일관성 유지)
max_len = 10
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
print("Padded sequences:", padded_sequences)

# [5] 원-핫 인코딩 (One-Hot Encoding) : 단어 인덱스를 원-핫 인코딩 벡터로 변환
one_hot_encoded = to_categorical(padded_sequences, num_classes=len(tokenizer.word_index) + 1)
print("One-hot encoded:\n", one_hot_encoded)

# [6] 텐서로 변환 (Convert to Tensor) : 텐서로 변환하여 딥러닝 모델에 입력할 수 있도록 준비
tensor = tf.convert_to_tensor(padded_sequences, dtype=tf.int32)
print("Tensor:", tensor)
