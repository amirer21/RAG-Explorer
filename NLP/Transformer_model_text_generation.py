from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# 사전 학습된 Transformer 모델과 토크나이저 로드
model_name = "distilbert-base-uncased"  # 사용할 사전 학습된 모델 이름 설정 (DistilBERT)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # 해당 모델에 맞는 토크나이저 불러오기
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)  # 사전 학습된 DistilBERT 모델 불러오기

# 텍스트 데이터 준비
sample_text = ["I enjoy learning new things"]  # 예측할 입력 문장 준비
input_encoding = tokenizer(sample_text, truncation=True, padding=True, max_length=128, return_tensors="tf")
# 문장을 토큰화하고 패딩 및 자르기(truncation) 적용하여 입력 데이터를 모델에 맞게 인코딩
# max_length=128로 설정하여 최대 128개의 토큰까지만 사용

# Transformer 모델을 사용해 예측
logits = model(input_encoding).logits  # 모델에 인코딩된 텍스트를 입력하여 로짓(logits) 출력
predicted_label = np.argmax(logits, axis=-1)  # 로짓에서 가장 큰 값을 가지는 인덱스를 예측된 레이블로 선택

# 예측 결과 출력
print("Predicted label (Transformer):", "Positive" if predicted_label == 1 else "Negative")
# 예측된 레이블이 1이면 "Positive", 그렇지 않으면 "Negative"로 출력
# 코드 실행 결과: Predicted label (Transformer): Positive