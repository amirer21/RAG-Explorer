import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def preprocess_texts(train_texts, train_labels, new_texts, num_words=1000, max_len=20):
    """
    Converts text data into integer sequences and applies padding for model input preparation.
    
    Args:
        train_texts (list): Training text data.
        train_labels (list): Labels for training data.
        new_texts (list): New text data for prediction.
        num_words (int): Maximum number of words to use.
        max_len (int): Maximum length of sequences.
    
    Returns:
        tuple: Preprocessed training data (x_train, y_train) and new data (x_new).
    """
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_texts)

    x_train = tokenizer.texts_to_sequences(train_texts)
    x_new = tokenizer.texts_to_sequences(new_texts)

    x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
    x_new = pad_sequences(x_new, maxlen=max_len, padding='post')

    y_train = to_categorical(train_labels, 2)  # Convert labels to one-hot encoding

    return x_train, y_train, x_new

def train_and_evaluate_model(model, x_train, y_train):
    """
    Trains and evaluates a model.
    
    Args:
        model (Sequential): Model to train.
        x_train (array): Training data.
        y_train (array): Training labels.
    
    Returns:
        tuple: Training history and model accuracy.
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=2, verbose=0)
    scores = model.evaluate(x_train, y_train, verbose=0)
    return history, scores[1]

def build_dnn_model(input_dim, max_len):
    """
    Builds a simple DNN model.
    
    Args:
        input_dim (int): Dimension of input data (number of words).
        max_len (int): Maximum length of input sequences.
    
    Returns:
        Sequential: Defined DNN model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def build_cnn_model(input_dim, max_len):
    """
    Builds a CNN model with Conv1D and MaxPooling1D layers.
    
    Args:
        input_dim (int): Dimension of input data (number of words).
        max_len (int): Maximum length of input sequences.
    
    Returns:
        Sequential: Defined CNN model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def build_rnn_model(input_dim, max_len):
    """
    Builds an RNN model with an LSTM layer.
    
    Args:
        input_dim (int): Dimension of input data (number of words).
        max_len (int): Maximum length of input sequences.
    
    Returns:
        Sequential: Defined RNN model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def build_gru_model(input_dim, max_len):
    """
    Builds a GRU model.
    
    Args:
        input_dim (int): Dimension of input data (number of words).
        max_len (int): Maximum length of input sequences.
    
    Returns:
        Sequential: Defined GRU model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(GRU(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

# Preparing data
# Example text data and labels for training and testing
train_texts = ["이 음식은 정말 맛있습니다", "별로 맛이 없어요", "이 음식을 추천합니다", "맛이 매우 별로입니다"]
train_labels = [1, 0, 1, 0]
new_texts = ["이 음식 정말 대단해요!", "돈이 아까워요."]

# Preprocess data
x_train, y_train, x_new = preprocess_texts(train_texts, train_labels, new_texts)

# Define a list of models to train and evaluate
models = [
    ("DNN", build_dnn_model(1000, 20)),
    ("CNN", build_cnn_model(1000, 20)),
    ("RNN", build_rnn_model(1000, 20)),
    ("GRU", build_gru_model(1000, 20))
]

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models:
    print(f"{name} 모델 학습 중...")
    history, accuracy = train_and_evaluate_model(model, x_train, y_train)
    results[name] = (history, accuracy)

# Compare and visualize results
for name, (history, accuracy) in results.items():
    print(f"{name} 모델 - 학습 정확도: {accuracy * 100:.2f}%")

# Plotting the training accuracy of each model
plt.figure(figsize=(10, 5))
for name, (history, _) in results.items():
    plt.plot(history.history['accuracy'], label=f'{name} Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
