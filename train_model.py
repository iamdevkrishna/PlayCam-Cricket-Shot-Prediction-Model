import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

DATA_PATH = "MP_Data"
SEQUENCE_LENGTH = 30
FEATURES = 1662
LABELS = os.listdir(DATA_PATH)

# Encode labels
label_map = {label: num for num, label in enumerate(LABELS)}
print("Labels map:", label_map)

sequences, labels = [], []

# Load data
for label in LABELS:
    dir_path = os.path.join(DATA_PATH, label)
    for file in os.listdir(dir_path):
        sequence = np.load(os.path.join(dir_path, file))
        if sequence.shape == (SEQUENCE_LENGTH, FEATURES):
            sequences.append(sequence)
            labels.append(label_map[label])

X = np.array(sequences)
y = np.array(labels)

# One-hot encode labels using LabelBinarizer
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Splitting dataset in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model Building Start
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(LABELS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Saving the Model
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[checkpoint])

#Evaluating Model
val_loss, val_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Accuracy: {val_acc*100:.2f}%")
