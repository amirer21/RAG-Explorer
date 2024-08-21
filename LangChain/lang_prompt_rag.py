# Langchain docs 사이트에 가서, output_parsers 를 참고한다.
# https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.output_parsers
# https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html#langchain_core.output_parsers.string.StrOutputParser
from dotenv import load_dotenv  # .env 파일을 불러오는 패키지
import os
import openai
import time
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
#os.environ["OPENAI_API_KEY"] = "your-api-key"
# .env 파일의 환경 변수를 로드
#load_dotenv()
load_dotenv(dotenv_path='openapi_key.env')
# .env 파일에서 API 키를 가져옴
api_key = os.getenv("OPENAI_API_KEY")
# API 키 확인
print(f"Loaded API Key: {api_key}")
if not api_key:
    raise ValueError("API 키를 로드할 수 없습니다. .env 파일을 확인하세요.")

openai.api_key = api_key

# 시작 시간 기록
start_time = time.time()

model = ChatOpenAI(model="gpt-4o")
# 프롬프트 템플릿 정의 (별도의 Runnable 체인 필요 없음)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "당신은  {topic}에 대한 개발을 하는 개발자입니다. " ),
    ("human"," 기본코드를 {cnt}개정도 작성해 주세요 ")
 ])
# LangChain Expression Language (LCEL):   LangChain에서 제공하는 파이프라인 스타일 연산
chain = prompt_template | model | StrOutputParser()  # 파이프 (|)로 체인 연결 !!!
# 체인 실행
result = chain.invoke({"topic": "RAG", "cnt": 3})

# 종료 시간 기록
end_time = time.time()

# 출력
print(result)

# 실행 시간 출력
execution_time = end_time - start_time
print(f"실행 시간: {execution_time:.2f}초")  # 실행 시간: 16.01초
"""
RAG(회귀 분석을 위한 유전자 알고리즘, Regression Analysis using Genetic Algorithm)에 대한 기본 코드를 작성해드리겠습니다. 이 코드는 Python의 주요 라이브러리인 `numpy`와 `sklearn`을 사용하여 간단한 유전자 알고리즘을 구현하고 이를 회귀 분석 문제에 적용하는 예제입니다. 다음은 세 가지 주요 부분으로 구성된 기본 코드입니다.

1. **유전자 알고리즘 초기화 및 유전자 생성**
2. **적합도 평가 및 선택**
3. **교차 및 돌연변이 연산**

### 1. 유전자 알고리즘 초기화 및 유전자 생성

```python
import numpy as np

# 유전자 알고리즘 파라미터 설정
population_size = 100
num_generations = 50
mutation_rate = 0.01
num_genes = 3  # 예: 회귀 계수의 수

# 초기 유전자 생성
def initialize_population(population_size, num_genes):
    population = np.random.randn(population_size, num_genes)
    return population

population = initialize_population(population_size, num_genes)
print("초기 유전자 집합:\n", population)
```

### 2. 적합도 평가 및 선택

```python
from sklearn.metrics import mean_squared_error

# 가상의 데이터셋 생성 (X, y)
X = np.random.rand(100, num_genes)  # 100개의 샘플, num_genes개의 특성
true_coefficients = np.array([3.5, -2.0, 1.0])
y = X @ true_coefficients + np.random.randn(100) * 0.5  # 실제 y 값

# 적합도 함수 정의 (여기서는 MSE 사용)
def fitness_function(population, X, y):
    predictions = X @ population.T
    fitness = -np.mean((predictions - y[:, np.newaxis])**2, axis=0)  # MSE의 음수
    return fitness

fitness_scores = fitness_function(population, X, y)
print("적합도 점수:\n", fitness_scores)
```

### 3. 교차 및 돌연변이 연산

```python
# 선택 연산 (룰렛휠 선택)
def select_parents(population, fitness_scores):
    probabilities = fitness_scores / np.sum(fitness_scores)
    parents_indices = np.random.choice(np.arange(population.shape[0]), size=population.shape[0], p=probabilities)
    return population[parents_indices]

# 교차 연산 (단순 한 지점 교차)
def crossover(parents, population_size):
    offspring = []
    for _ in range(population_size // 2):
        parent1, parent2 = parents[np.random.randint(0, len(parents), 2)]
        crossover_point = np.random.randint(1, num_genes)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        offspring.append(child1)
        offspring.append(child2)
    return np.array(offspring)

# 돌연변이 연산
def mutate(offspring, mutation_rate):
    for child in offspring:
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(num_genes)
            child[mutation_point] += np.random.randn()
    return offspring

# 세대 반복
for generation in range(num_generations):
    fitness_scores = fitness_function(population, X, y)
    parents = select_parents(population, fitness_scores)
    offspring = crossover(parents, population_size)
    population = mutate(offspring, mutation_rate)
    best_fitness = np.max(fitness_scores)
    print(f"세대 {generation+1}, 최고 적합도: {best_fitness}")

# 최종 결과
best_individual = population[np.argmax(fitness_scores)]
print("최적의 회귀 계수:", best_individual)
```

이제 이 코드들을 하나로 합쳐 실행하면, 유전자 알고리즘을 통해 회귀 계수를 찾는 과정을 시뮬레이션할 수 있습니다. 이 코드는 매우 기본적인 형태이며, 실제 문제에 적용할 때는 더 많은 최적화와 조정이 필요할 수 있습니다.
"""