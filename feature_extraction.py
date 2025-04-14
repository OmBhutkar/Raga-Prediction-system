import os
import librosa
import numpy as np
import librosa.display

# Dataset Path
DATASET_PATH = "D:/PA Project/dataset"
SAVE_FEATURES_PATH = "D:/PA Project/"

# Extract Features (Mel Spectrograms)
def extract_features(file_path, max_pad=216): 
    y, sr = librosa.load(file_path, sr=22050)  # Load Audio
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Padding or Truncating to Standard Size
    pad_width = max_pad - mel_spec.shape[1]
    if pad_width > 0:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec = mel_spec[:, :max_pad]
    
    return mel_spec

# Process Entire Dataset
def process_dataset(dataset_path):
    X, y = [], []
    labels = os.listdir(dataset_path)  # Get Raga Names
    
    for label in labels:
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue
        
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

# Load Dataset
X, y = process_dataset(DATASET_PATH)

# Save Processed Data
np.save(os.path.join(SAVE_FEATURES_PATH, "X.npy"), X)
np.save(os.path.join(SAVE_FEATURES_PATH, "y.npy"), y)

print("Feature Extraction Completed! Data saved as X.npy and y.npy")
