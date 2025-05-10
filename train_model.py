import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Load Data
LOAD_FEATURES_PATH = "D:/PA Project/"
X = np.load(os.path.join(LOAD_FEATURES_PATH, "X.npy"))
y = np.load(os.path.join(LOAD_FEATURES_PATH, "y.npy"))

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save Label Encoder Classes
np.save(os.path.join(LOAD_FEATURES_PATH, "label_classes.npy"), label_encoder.classes_)

# Reshape for CNN Input
X = X[..., np.newaxis]  # Adding channel dimension

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 216, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, y_categorical, epochs=30, batch_size=16, validation_split=0.2)

# Save Model
model.save(os.path.join(LOAD_FEATURES_PATH, "raga_model.h5"))   

print("Model Training Completed! Saved as raga_model.h5")
