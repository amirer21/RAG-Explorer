import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# 예제 데이터 준비
texts = ["I love sunset", "Sky is beautiful", "I hate rain", "Cloud is beautiful"]
labels = [1, 1, 0, 1]  # 레이블: 1은 Positive, 0은 Negative를 의미

# 텍스트 토크나이징 및 패딩
tokenizer = Tokenizer(num_words=1000)  # 사전에 사용할 최대 단어 수를 1000으로 제한
tokenizer.fit_on_texts(texts)  # 텍스트에 나오는 단어들에 번호를 부여 (토크나이징)
sequences = tokenizer.texts_to_sequences(texts)  # 텍스트를 숫자의 시퀀스로 변환
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')
# 시퀀스 길이를 10으로 맞추고, 부족한 부분은 'post'(뒤)에 0을 추가

# 데이터 분할 (학습/검증)
train_texts, val_texts, train_labels, val_labels = train_test_split(padded_sequences, labels, test_size=0.2)
# 데이터를 80%는 학습용, 20%는 검증용으로 분할

# 레이블을 NumPy 배열로 변환
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# 신경망 모델 정의
model = tf.keras.Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),  # 임베딩 레이어: 단어를 16차원 벡터로 변환
    GlobalAveragePooling1D(),  # 임베딩 벡터의 평균을 구함
    Dense(16, activation='relu'),  # 은닉층: 16개의 노드, ReLU 활성화 함수
    Dense(1, activation='sigmoid')  # 출력층: 1개의 노드, 시그모이드 활성화 함수 (이진 분류)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 이진 교차 엔트로피 손실 함수 사용
              metrics=['accuracy'])  # 모델 성능 지표로 정확도를 사용

# 모델 학습
model.fit(train_texts, train_labels, validation_data=(val_texts, val_labels), epochs=10)
# 모델을 10번의 에포크 동안 학습하고, 검증 데이터로 성능을 평가

# 예측
sample_text = ["I love to take a picture of sunset"]  # 새로운 샘플 텍스트 입력
sample_sequence = tokenizer.texts_to_sequences(sample_text)  # 샘플 텍스트를 시퀀스로 변환
padded_sample = pad_sequences(sample_sequence, maxlen=10, padding='post')
# 시퀀스 길이를 학습 데이터와 동일하게 10으로 맞추고, 부족한 부분은 0으로 채움
prediction = model.predict(padded_sample)  # 모델을 사용해 예측 수행
print("Predicted label (Simple Neural Network):", "Positive" if prediction > 0.5 else "Negative")
# 예측 결과가 0.5보다 크면 Positive, 그렇지 않으면 Negative로 출력

"""
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step - accuracy: 0.6667 - loss: 0.6890 - val_accuracy: 1.0000 - val_loss: 0.6765
Epoch 2/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step - accuracy: 0.6667 - loss: 0.6875 - val_accuracy: 1.0000 - val_loss: 0.6730
Epoch 3/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - accuracy: 0.6667 - loss: 0.6861 - val_accuracy: 1.0000 - val_loss: 0.6695
Epoch 4/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step - accuracy: 0.6667 - loss: 0.6848 - val_accuracy: 1.0000 - val_loss: 0.6662
Epoch 5/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step - accuracy: 0.6667 - loss: 0.6834 - val_accuracy: 1.0000 - val_loss: 0.6628
Epoch 6/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - accuracy: 0.6667 - loss: 0.6821 - val_accuracy: 1.0000 - val_loss: 0.6595
Epoch 7/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step - accuracy: 0.6667 - loss: 0.6808 - val_accuracy: 1.0000 - val_loss: 0.6561
Epoch 8/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - accuracy: 0.6667 - loss: 0.6795 - val_accuracy: 1.0000 - val_loss: 0.6528
Epoch 9/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - accuracy: 0.6667 - loss: 0.6782 - val_accuracy: 1.0000 - val_loss: 0.6494
Epoch 10/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step - accuracy: 0.6667 - loss: 0.6769 - val_accuracy: 1.0000 - val_loss: 0.6461
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
Predicted label (Simple Neural Network): Positive
"""