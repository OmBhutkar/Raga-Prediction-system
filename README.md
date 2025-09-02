````markdown
# 🎶 Raga Prediction System 🎶

Welcome to the Raga Prediction System! This is a web-based application built to identify and classify Indian Classical Music Ragas from audio inputs. Whether you're a musician, a student, or just a music enthusiast, this tool can help you analyze and understand ragas in a fun, interactive way. 🎤

## ✨ Features

-   **🎵 Predict from File**: Upload a `.wav` audio file and get an instant raga prediction.
-   **🔴 Live Recording**: Record a 10-second audio clip directly in your browser for real-time raga identification.
-   **👤 User Profiles**: Create a simple user profile to track your prediction history.
-   **📊 Prediction Dashboard**: View your total predictions and a list of your most frequently identified ragas.
-   **📈 Detailed Analysis**: For each prediction, the system generates:
    -   The predicted Raga name and a brief description.
    -   Confidence scores for the top predictions.
    -   Visualizations of the audio's pitch contour and waveform.
-   **📄 PDF Reports**: Download a beautifully formatted PDF report summarizing the entire analysis for each prediction.

## 🚀 How to Run the Project

Follow these simple steps to get the Raga Prediction System up and running on your local machine.

### 1. **Prerequisites** 🛠️

Make sure you have Python (version 3.8 or higher) and `pip` installed on your system.

### 2. **Clone the Repository** 📂

Open your terminal or command prompt and clone the project repository:

```bash
git clone <your-repository-url>
cd <repository-name>
````

### 3\. **Set Up a Virtual Environment (Recommended)** 🌳

It's a good practice to create a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4\. **Install Dependencies** 📦

Install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**Note for `sounddevice` users:** You might need to install system-level dependencies for this library.

  - **On Debian/Ubuntu:** `sudo apt-get install libportaudio2`
  - **On macOS (using Homebrew):** `brew install portaudio`

### 5\. **Prepare Your Data and Model** 🧠

Make sure you have the following files in your project's root directory:

  - `raga_model.h5` (the trained model)
  - `label_classes.npy` (the raga labels)
  - `X.npy` and `y.npy` (if you plan to retrain the model)

Your project structure should look like this:

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
│   └── (your css, js, images)
├── raga_model.h5
└── label_classes.npy
```

### 6\. **Run the Application\!** 🎉

Now, you're ready to start the Flask web server.

```bash
python app.py
```

You should see an output like this, indicating that the server is running:

```
 * Running on [http://127.0.0.1:5000/](http://127.0.0.1:5000/) (Press CTRL+C to quit)
```

### 7\. **Open in Your Browser** 🌐

Open your favorite web browser and navigate to the following address:

**[http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000)**
