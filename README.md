# 🎶 Raga Prediction System 🎶

Welcome to the **Raga Prediction System**!  
This is a web-based application built to identify and classify **Indian Classical Music Ragas** from audio inputs.  

Whether you're a 🎼 musician, 🎓 student, or just a 🎧 music enthusiast, this tool can help you analyze and understand ragas in a fun and interactive way.  

---

## ✨ Features

- 🎵 **Predict from File** → Upload a `.wav` audio file and get an instant raga prediction.  
- 🔴 **Live Recording** → Record a 10-second audio clip directly in your browser for real-time raga identification.  
- 👤 **User Profiles** → Create a profile to track your prediction history.  
- 📊 **Prediction Dashboard** → View your total predictions and most frequently identified ragas.  
- 📈 **Detailed Analysis** → For each prediction, the system shows:  
  - 🎼 Predicted Raga name + description  
  - 📊 Confidence scores for top predictions  
  - 🎶 Pitch contour & waveform visualization  
- 📄 **PDF Reports** → Download a beautifully formatted report of your prediction.  

---

## 🚀 How to Run the Project

Follow these steps to run the project locally:

### 1. 🛠️ Prerequisites
- Python **3.8+**  
- `pip` package manager  

---

### 2. 📂 Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
````

---

### 3. 🌳 Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 4. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

⚠️ For `sounddevice`, you may need system packages:

* **Ubuntu/Debian:** `sudo apt-get install libportaudio2`
* **macOS (Homebrew):** `brew install portaudio`

---

### 5. 🧠 Prepare Model & Data

Ensure the following files exist in your project root:

* `raga_model.h5` → trained model
* `label_classes.npy` → class labels
* `X.npy`, `y.npy` → training data (optional, if retraining)

📂 Example structure:

```
.
├── app.py
├── train_model.py
├── feature_extraction.py
├── templates/
│   ├── index.html
│   ├── login.html
│   └── profile.html
├── static/
│   └── (CSS, JS, Images)
├── raga_model.h5
└── label_classes.npy
```

---

### 6. 🎉 Run the Application

```bash
python app.py
```

Expected output:

```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

---

### 7. 🌐 Open in Browser

Go to:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

✨ That’s it! You’re ready to explore the magical world of ragas with the **Raga Prediction System**.
Enjoy making music! 🎶😊

```

---

👉 Replace your current `README.md` with this version and push to GitHub.  
Would you like me to also **add a section at the bottom** explaining how to use **Git LFS** for uploading `raga_model.h5`?
```
