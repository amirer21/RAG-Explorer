# python module install
# pip install --upgrade tensorflow keras
# pip install tensorflow transformers
# pip install scikit-learn

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 예제 데이터 준비
texts = ["I love sunset", "Sky is beautiful", "I hate rain", "Cloud is beautiful"]
labels = ["positive", "positive", "negative", "positive"]  # 레이블로 긍정(positive)과 부정(negative) 값을 정의

# 레이블 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)  # 텍스트 레이블을 정수 값으로 변환 (0: Negative, 1: Positive)

# 데이터 분할 (학습/검증)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels_encoded, test_size=0.2)
# 데이터를 학습용과 검증용으로 80:20 비율로 분할

# 사전 학습된 모델과 토크나이저 로드
model_name = "distilbert-base-uncased"  # BERT 경량 버전의 모델 이름 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 해당 모델의 토크나이저 불러오기
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# 사전 학습된 BERT 모델을 불러와, 2개의 레이블(긍정/부정) 분류 모델로 설정

# 텍스트 토크나이징
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
# 학습 및 검증 데이터를 BERT 모델이 처리할 수 있도록 토큰화

# TensorFlow 데이터셋 준비
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(100).batch(2)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(2)
# TensorFlow 데이터셋으로 변환, 학습 데이터는 셔플하고 배치 크기는 2로 설정

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 모델을 컴파일, 옵티마이저는 Adam, 손실 함수는 이진 교차 엔트로피 사용, 정확도를 평가 지표로 설정

# 모델 학습
model.fit(train_dataset, validation_data=val_dataset, epochs=3)
# 모델을 학습, 3번의 에포크 동안 학습을 진행하고 검증 데이터로 성능을 평가

# 예측
sample_text = ["I love to take a picture of sunset"]
sample_encoding = tokenizer(sample_text, truncation=True, padding=True, max_length=128, return_tensors="tf")
logits = model(sample_encoding).logits
predicted_label = np.argmax(logits, axis=1)
# 새로운 텍스트에 대해 예측, 텍스트를 토큰화하고 모델을 통해 예측한 뒤, 가장 높은 확률의 레이블 선택

print(f"Predicted label: {label_encoder.inverse_transform(predicted_label)[0]}")
# 예측된 레이블을 다시 원래 텍스트 레이블로 변환하여 출력

"""
2/2 [==============================] - 15s 2s/step - loss: 0.5507 - accuracy: 1.0000 - val_loss: 0.9798 - val_accuracy: 0.0000e+00
Epoch 2/3
2/2 [==============================] - 1s 655ms/step - loss: 0.4002 - accuracy: 1.0000 - val_loss: 1.3466 - val_accuracy: 0.0000e+00
Epoch 3/3
2/2 [==============================] - 1s 664ms/step - loss: 0.2355 - accuracy: 1.0000 - val_loss: 1.8205 - val_accuracy: 0.0000e+00
Predicted label: positive
"""
# 학습과 예측 결과를 출력, 최종적으로 "positive" 레이블이 예측됨