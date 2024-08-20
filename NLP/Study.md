NLP(Natural Language Processing, 자연어 처리)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 인공지능(AI)의 한 분야입니다.
NLP는 텍스트와 음성 데이터를 분석하고, 이해하고, 생성하는 데 사용되며, 컴퓨터와 인간 간의 상호작용을 자연스럽게 만듭니다.

자연어 처리는 다음과 같은 여러 하위 분야로 나눌 수 있습니다:

1. **텍스트 전처리 (Text Preprocessing)**: 텍스트 데이터를 분석하기 전에 불필요한 부분을 제거하고, 필요한 형태로 변환하는 단계입니다. 예를 들어, 소문자 변환, 불용어 제거, 형태소 분석 등이 포함됩니다.

2. **형태소 분석 (Morphological Analysis)**: 단어를 구성하는 어근, 접두사, 접미사 등을 분석하여 문장에서의 역할을 이해합니다.

3. **구문 분석 (Syntax Analysis)**: 문장의 문법적 구조를 파악하여 문법적으로 올바른 문장인지 분석합니다. 예를 들어, 의존 구문 분석이 있습니다.

4. **의미 분석 (Semantic Analysis)**: 단어와 문장의 의미를 분석하는 단계로, 텍스트의 의미를 이해하는 것이 목표입니다. 예를 들어, 의미 롤 라벨링이 있습니다.

5. **대화 처리 (Dialogue Management)**: 대화의 흐름을 관리하고, 사용자와의 상호작용을 처리하는 기술입니다. 챗봇이나 가상 비서에 사용됩니다.

6. **자연어 생성 (Natural Language Generation, NLG)**: 데이터를 기반으로 자연스러운 언어를 생성하는 기술로, 보고서 작성, 뉴스 요약 등에 사용됩니다.

7. **감정 분석 (Sentiment Analysis)**: 텍스트에서 감정을 분석하여 긍정적, 부정적, 중립적 감정을 분류하는 기술입니다.

NLP는 일상에서 자주 사용되는 여러 기술에 활용됩니다. 예를 들어, 음성 인식, 기계 번역(예: 구글 번역), 텍스트 요약, 자동 응답 생성, 검색 엔진의 질의 응답 시스템 등이 있습니다. NLP 기술은 AI와 머신러닝의 발전에 따라 점점 더 정교해지고 있으며, 다양한 산업 분야에서 중요한 역할을 하고 있습니다.


이 코드들은 각각 자연어 처리(NLP)와 관련된 다양한 예제를 보여줍니다. 각 코드가 무엇을 다루는지 간단하게 설명드리겠습니다.

### \NLP\NLP_exam.py
1. **Simple Neural Network for Text Classification (간단한 신경망을 사용한 텍스트 분류 예제)**:
   ```python
   import tensorflow as tf
   from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   from sklearn.model_selection import train_test_split
   import numpy as np
   ```
   - **설명**: 이 코드는 간단한 신경망을 사용하여 텍스트를 긍정(Positive) 또는 부정(Negative)으로 분류하는 예제입니다. `Embedding`, `Dense` 등의 레이어를 사용하여 신경망을 구성하고, 텍스트 데이터를 학습하여 분류 모델을 만듭니다.


---------

다음 두 코드 모두 텍스트 데이터를 사용하여 긍정(Positive) 또는 부정(Negative)으로 분류하는 예제입니다.
각각 사용된 모델과 방법론이 다른데, 각 코드의 차이점과 각각이 다루는 예제를 간단히 설명하겠습니다.

### \NLP\Transformer_model_text_classification.py

### Transformer_model_text_classification.py (Simple Neural Network for Text Classification)
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# 이하 생략...
```

#### **특징:**
- **모델 유형**: 이 코드는 기본 신경망(Neural Network)을 사용하여 텍스트 분류를 수행합니다.
- **데이터 준비**: 직접 제공된 간단한 텍스트 데이터를 사용하여, `Tokenizer`를 통해 텍스트를 토큰화하고, `pad_sequences`로 시퀀스 길이를 맞추는 과정을 포함합니다.
- **신경망 구성**: `Embedding` 레이어를 사용해 텍스트를 벡터로 변환하고, `GlobalAveragePooling1D`으로 평균을 구한 후, `Dense` 레이어를 통해 이진 분류를 수행합니다.
- **예제의 목적**: 이 코드는 기본적인 신경망 모델을 구축하고 텍스트 데이터를 분류하는 방법을 보여줍니다. 이 모델은 주로 간단한 데이터셋과 작은 모델로 실험할 때 사용됩니다.

---------

### Transformer_model_text_generation.py (Transformer-based Text Classification)
```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# 이하 생략...
```

#### **특징:**
- **모델 유형**: 이 코드는 Transformer 기반의 사전 학습된 모델(DistilBERT)을 사용하여 텍스트 분류를 수행합니다.
- **데이터 준비**: BERT와 같은 사전 학습된 모델에 맞춰 `AutoTokenizer`를 사용해 텍스트를 토큰화하고, 모델에 입력될 수 있는 형태로 인코딩합니다.
- **Transformer 모델 사용**: 이 예제에서는 `TFAutoModelForSequenceClassification`을 사용하여 사전 학습된 DistilBERT 모델을 불러와 텍스트 분류 작업을 수행합니다.
- **예제의 목적**: 이 코드는 고성능의 사전 학습된 Transformer 모델을 활용해 텍스트 분류 작업을 수행하는 방법을 보여줍니다. 이는 더 복잡하고 규모가 큰 데이터셋에서도 높은 성능을 기대할 수 있는 접근 방식입니다.

### **주요 차이점:**
1. **모델 복잡도와 성능**:
   - 첫 번째 코드는 간단한 신경망을 사용하며, 작은 데이터셋과 기본적인 텍스트 분류 작업에 적합합니다.
   - 두 번째 코드는 사전 학습된 BERT 모델을 활용하며, 대규모 데이터셋과 복잡한 자연어 처리 작업에서도 높은 성능을 발휘할 수 있습니다.

2. **데이터 전처리**:
   - 첫 번째 코드에서는 `Tokenizer`와 `pad_sequences`를 사용하여 텍스트 데이터를 신경망에 맞게 변환합니다.
   - 두 번째 코드에서는 `AutoTokenizer`를 사용하여 텍스트 데이터를 BERT 모델에 맞게 토큰화하고 인코딩합니다.

3. **모델 사용**:
   - 첫 번째 코드에서는 처음부터 신경망을 정의하고 학습합니다.
   - 두 번째 코드에서는 이미 학습된 BERT 모델을 불러와 재사용합니다.

### **결론:**
- **첫 번째 코드**는 **기본적인 신경망을 사용한 텍스트 분류**를 배우고자 할 때 유용한 예제입니다.
- **두 번째 코드**는 **사전 학습된 Transformer 모델(BERT)을 사용한 텍스트 분류**에 관한 예제로, 더 복잡한 NLP 작업에서 높은 성능을 기대할 수 있는 접근 방식입니다.