from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import numpy as np
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
import time
import sounddevice as sd
from scipy.io.wavfile import write
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import csv
import uuid
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "ragas_secret_key"  # Added secret key for session management

# Ensure necessary directories exist
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/waveforms", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/user_data", exist_ok=True)

# Define constants and paths
USER_DATA_PATH = "static/user_data/user_predictions.csv"
MODEL_PATH = "raga_model.h5"  # Changed from absolute path to relative path
LABELS_PATH = "label_classes.npy"  # Changed from absolute path to relative path

# Initialize the CSV file if it doesn't exist
if not os.path.exists(USER_DATA_PATH):
    with open(USER_DATA_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'timestamp', 'raga', 'confidence', 'file_name', 'report_path'])

# Load Trained Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_classes = np.load(LABELS_PATH, allow_pickle=True)
except Exception as e:
    print(f"Error loading model or labels: {e}")
    # Initialize with default values to prevent app crashes
    model = None
    label_classes = []

# Dictionary of Raga Descriptions
raga_info = {
    "Asavari": "A morning raga known for its serious and meditative nature.\n\n Author: Vishnu Narayan Bhatkhande",
    "Bageshree": "A late-night raga that expresses deep emotions and longing.\n\n Author: Ravi Shankar",
    "Bhairavi": "A soulful and devotional raga, usually performed at the end of a concert.\n\n Author: Omkarnath Thakur",
    "Bhoop": "A pentatonic raga associated with peace and devotion.\n\n Author: Vishnu Digambar Paluskar",
    "Bhoopali": "A joyful and uplifting raga known for its simplicity and purity.\n\n Author: Kumar Gandharva",
    "Darbari": "A serious and heavy raga, typically performed late at night.\n\n Author: Miyan Tansen",
    "DKanada": "A mix of Darbari Kanada and Malhar, known for its rich and expressive melody.\n\n Author: Bhimsen Joshi",
    "Malkauns": "A powerful and meditative raga, often performed at midnight.\n\n Author: Sharangadeva",
    "Sarang": "A bright and lively raga associated with the afternoon.\n\n Author: Ramashreya Jha",
    "Yaman": "An evening raga known for its calm and serene mood.\n\n Author: Amir Khusrau"
}
# Function to save prediction data
def save_prediction(user_id, raga, confidence, file_name=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = f"static/reports/{raga}_report.pdf"
    
    with open(USER_DATA_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, timestamp, raga, confidence, file_name, report_path])

# Function to Extract Features
def extract_features(file_path, max_pad=216):
    try:
        y, sr = librosa.load(file_path, sr=22050)

        # Remove silent parts
        y, _ = librosa.effects.trim(y)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        pad_width = max_pad - mel_spec.shape[1]
        if pad_width > 0:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :max_pad]

        return mel_spec, y, sr
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None

# Function to Save Pitch Contour Plot
def save_pitch_contour(y, sr, filename_prefix="pitch_contour"):
    try:
        timestamp = int(time.time() * 1000)
        filename = f"{filename_prefix}_{timestamp}.png"
        
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Pitch Contour")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(f"static/{filename}")
        plt.close()
        
        return filename
    except Exception as e:
        print(f"Error saving pitch contour: {e}")
        return None

# Function to save waveform with pitch
def save_waveform_with_pitch(y, sr, filename_prefix="waveform_pitch"):
    try:
        timestamp = int(time.time() * 1000)
        filename = f"{filename_prefix}_{timestamp}.png"
        
        # Create a figure with 1 subplot (only pitch contour, removed waveform)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get pitch contour (highest magnitude at each frame)
        pitch_contour = []
        times = librosa.times_like(pitches)
        
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            # Only include frequencies that are likely to be pitch
            if pitch > 0 and magnitudes[index, i] > 0.025:
                pitch_contour.append(pitch)
            else:
                pitch_contour.append(float('nan'))  # Use NaN for silent/non-pitched regions
        
        # Plot the pitch contour with improved styling
        ax.plot(times[:len(pitch_contour)], pitch_contour, color='#e74c3c', linewidth=1.5, label='Pitch')
        ax.set_title("Pitch Contour")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        
        # Add shaded areas for pitch ranges to help identify the raga characteristics
        ax.axhspan(60, 130, alpha=0.1, color='blue', label='Low register')
        ax.axhspan(130, 250, alpha=0.1, color='green', label='Mid register')
        ax.axhspan(250, 500, alpha=0.1, color='red', label='High register')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add musical note markers on the y-axis
        note_freqs = {
            'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88
        }
        
        # Set y-ticks to match musical notes
        ax.set_yticks(list(note_freqs.values()))
        ax.set_yticklabels(list(note_freqs.keys()))
        
        # Limit y-axis to relevant range for Indian classical music
        ax.set_ylim(60, 600)
        
        # Add a colorbar for intensity
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Intensity (relative)')
        
        plt.tight_layout()
        plt.savefig(f"static/waveforms/{filename}", dpi=100)
        plt.close()
        
        return f"waveforms/{filename}"
    except Exception as e:
        print(f"Error saving waveform with pitch: {e}")
        return None

# Function to Generate PDF Report with waveform
def generate_pdf(raga, description, duration, sample_rate, image_path, waveform_path, top_predictions=None):
    try:
        pdf_filename = f"static/reports/{raga}_report.pdf"
        c = canvas.Canvas(pdf_filename, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(200, 750, "🎵 Raga Prediction Report 🎶")
        
        # Basic information
        c.setFont("Helvetica", 14)
        c.drawString(50, 700, f"🎼 Predicted Raga: {raga}")
        c.drawString(50, 670, f"📖 Description: {description}")
        c.drawString(50, 640, f"🎵 Duration: {duration}")
        c.drawString(50, 610, f"🎚 Sample Rate: {sample_rate}")
        
        # Confidence scores section with improved formatting
        if top_predictions and len(top_predictions) > 0:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 580, "🔍 Confidence Scores:")
            
            # Create a table-like structure for confidence scores
            c.setFont("Helvetica", 12)
            y_position = 550
            
            # Draw header line
            c.line(70, y_position + 15, 300, y_position + 15)
            c.drawString(70, y_position, "Raga")
            c.drawString(200, y_position, "Confidence")
            c.line(70, y_position - 5, 300, y_position - 5)
            
            y_position -= 25
            
            # Draw each prediction with proper alignment
            for pred in top_predictions:
                c.drawString(70, y_position, f"{pred['raga']}")
                c.drawString(200, y_position, f"{pred['confidence']}%")
                y_position -= 20
                
            # Draw bottom line
            c.line(70, y_position + 5, 300, y_position + 5)
            
            # Adjust the image position based on the number of predictions
            image_y_position = y_position - 200  # Adjusted for larger waveform image
        else:
            # Default image position if no predictions
            image_y_position = 450
        
        # Add Enhanced Waveform & Pitch Contour Image (give it more space)
        if waveform_path and os.path.exists(f"static/{waveform_path}"):
            c.drawString(50, image_y_position + 190, "📊 Audio Analysis:")
            c.drawImage(f"static/{waveform_path}", 50, image_y_position, width=500, height=180)
            
            # Add legend for the pitch contour
            c.setFont("Helvetica", 8)
            c.drawString(60, image_y_position - 10, "Blue: Low register | Green: Mid register | Red: High register")
            c.setFont("Helvetica", 12)
        
        # Add page break for better formatting
        c.showPage()
        
        # Original Pitch Contour Image (on the second page)
        if image_path and os.path.exists(image_path):
            c.drawString(50, 700, "📈 Original Pitch Contour Analysis:")
            c.drawImage(image_path, 50, 500, width=500, height=180)
        
        # Add explanatory notes about raga characteristics
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 450, "🎵 Understanding Raga Characteristics:")
        c.setFont("Helvetica", 12)
        
        # Add explanatory text
        raga_explanation = """
        Ragas are characterized by specific patterns of ascending (aroha) and descending (avaroha) 
        movements, distinctive phrases (pakad), emphasized notes (vadi, samvadi), and specific 
        pitch relationships. The pitch contour visualization helps identify these unique patterns.
        """
        
        # Split text into lines for PDF
        text_object = c.beginText(50, 420)
        text_object.setFont("Helvetica", 10)
        for line in raga_explanation.strip().split('\n'):
            text_object.textLine(line.strip())
        c.drawText(text_object)
        
        # Add specific traits of the identified raga
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 370, f"Key Characteristics of {raga}:")
        c.setFont("Helvetica", 10)
        
        # Additional raga-specific information
        raga_traits = {
            "Asavari": ["• Uses komal re, ga, dha, ni (flat 2nd, 3rd, 6th, 7th)", 
                        "• Emphasizes movements around komal ga and dha", 
                        "• Associated with early morning meditation",
                        "• Raga Asavari – Author :- Vishnu Narayan Bhatkhande"],
            "Bageshree": ["• Features komal ga, dha, ni (flat 3rd, 6th, 7th)", 
                          "• Has characteristic phrase dha-ni-sa-ga-ma", 
                          "• Expresses deep longing and emotion",
                          "• Raga Bageshree – Author: Ravi Shankar"],
            "Bhairavi": ["• All notes except sa and pa are komal (flat)",
                         "• Has rich, devotional quality with ornate movements",
                         "• Often performed as a conclusion to concerts",
                         "• Raga Bhairavi – Author: Omkarnath Thakur"],
            "Bhoop": ["• Simple pentatonic scale (sa, re, ga, pa, dha)",
                      "• Features characteristic phrase ga-re-sa",
                      "• Creates peaceful, devotional atmosphere",
                      "• Raga Bhoop – Author: Vishnu Digambar Paluskar"],
            "Bhoopali": ["• Pentatonic melody with sa, re, ga, pa, dha",
                         "• Features characteristic ascending movements",
                         "• Known for joyful and uplifting qualities"
                         "• Raga Bhoopali – Author: Kumar Gandharva"],
            "Darbari": ["• Features heavy oscillations on re and dha",
                        "• Slow, serious movements with unique phrases",
                        "• Complex melodic structure with characteristic descent",
                        "• Raga Darbari – Author: Miyan Tansen"],
            "DKanada": ["• Combines elements of Darbari Kanada and Malhar",
                        "• Features slow, meandering phrases with particular emphasis on komal re",
                        "• Rich and expressive with complex ornamentations"
                        "• Raga DKanada – Author: Bhimsen Joshi"],
            "Malkauns": ["• Pentatonic scale with komal ga and dha",
                         "• Powerful oscillations between ma and komal ga",
                         "• Meditative quality with distinctive ascending patterns",
                         "• Raga Malkauns – Author: Sharangadeva"],
            "Sarang": ["• Strong emphasis on the note ma",
                       "• Bright, lively melodic phrases with quick movements",
                       "• Associated with afternoon performances",
                       "• Raga Sarang – Author: Ramashreya Jha"],
            "Yaman": ["• Features teevra ma (sharp 4th)",
                      "• Characteristic phrases emphasizing teevra ma and ni",
                      "• Creates serene and peaceful atmosphere",
                      "• Raga Yaman – Author: Amir Khusrau"]
        }
        
        # Draw the raga traits
        y_pos = 340
        for trait in raga_traits.get(raga, ["• No specific traits available"]):
            c.drawString(60, y_pos, trait)
            y_pos -= 20
        
        # Add footer
        c.setFont("Helvetica", 10)
        c.drawString(width/2 - 100, 30, "Generated by Raga Prediction System")
        
        c.save()
        return pdf_filename
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

# Routes - Fixed the duplicate route issue
@app.route("/")
def home():
    # Pass the session data to the template
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        # Generate a unique user ID
        user_id = str(uuid.uuid4())
        
        # Get the username from the form
        username = request.form.get('username', 'User')
        
        # Store both user_id and username in session
        session['user_id'] = user_id
        session['username'] = username
        
        # Redirect to home page after login
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route("/profile")
def profile():
    if 'user_id' not in session:
        # Redirect to login page with a flash message
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_predictions = []
    
    # Read the CSV and filter for this user
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['user_id'] == user_id:
                    user_predictions.append(row)
    
    # Get statistics
    total_predictions = len(user_predictions)
    raga_counts = {}
    for pred in user_predictions:
        raga = pred['raga']
        if raga in raga_counts:
            raga_counts[raga] += 1
        else:
            raga_counts[raga] = 1
    
    # Sort by most frequent
    most_frequent_ragas = sorted(raga_counts.items(), key=lambda x: x[1], reverse=True)
    
    return render_template("profile.html", 
                          predictions=user_predictions, 
                          total_predictions=total_predictions, 
                          most_frequent_ragas=most_frequent_ragas)

@app.route("/logout")
def logout():
    # Clear all session data
    session.clear()
    # Redirect to home page
    return redirect(url_for('home'))

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500
        
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["audio"]
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # Extract Features
        features, y, sr = extract_features(file_path)
        if features is None:
            return jsonify({"error": "Failed to extract features from audio"}), 400
            
        duration = librosa.get_duration(y=y, sr=sr) / 60
        features = features[np.newaxis, ..., np.newaxis]

        # Predict Raga
        prediction = model.predict(features)[0]  # Get prediction probabilities
        top_indices = np.argsort(prediction)[-3:][::-1]  # Get top 3 predictions

        top_predictions = [
            {"raga": label_classes[i].title(), "confidence": round(prediction[i] * 100, 2)}
            for i in top_indices
        ]

        predicted_label = top_predictions[0]["raga"]
        raga_description = raga_info.get(predicted_label, "No description available.")

        # Generate Pitch Contour Plot (keep for backward compatibility)
        pitch_contour_filename = save_pitch_contour(y, sr)
        pitch_contour_path = f"static/{pitch_contour_filename}" if pitch_contour_filename else None
        
        # Generate Enhanced Waveform with Pitch Visualization
        waveform_path = save_waveform_with_pitch(y, sr)

        # Generate PDF Report with confidence scores
        pdf_filename = generate_pdf(
            predicted_label, 
            raga_description, 
            f"{duration:.2f} min", 
            f"{sr} Hz", 
            pitch_contour_path,
            waveform_path,
            top_predictions
        )
        
        # Save prediction if user is logged in
        if 'user_id' in session:
            save_prediction(
                session['user_id'],
                predicted_label,
                top_predictions[0]["confidence"],
                filename
            )

        return jsonify({
            "raga": predicted_label,
            "description": raga_description,
            "duration": f"{duration:.2f} min",
            "sample_rate": f"{sr} Hz",
            "pitch_contour_image": pitch_contour_path,
            "waveform_image": waveform_path,
            "pdf_report": pdf_filename,
            "top_predictions": top_predictions
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/live_predict", methods=["GET"])
def live_predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500
        
    try:
        fs = 22050
        duration = 10
        print("🎤 Listening for Live Raga Prediction...")

        # Record Live Audio
        audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        write("live_audio.wav", fs, audio_data)

        # Extract Features
        features, y, sr = extract_features("live_audio.wav")
        if features is None:
            return jsonify({"error": "Voice is not clear, try again!"}), 400

        features = features[np.newaxis, ..., np.newaxis]

        # Predict
        prediction = model.predict(features)[0]
        top_indices = np.argsort(prediction)[-3:][::-1]

        top_predictions = [
            {"raga": label_classes[i].title(), "confidence": round(prediction[i] * 100, 2)}
            for i in top_indices
        ]

        predicted_label = top_predictions[0]["raga"]
        raga_description = raga_info.get(predicted_label, "No description available.")

        # Generate Pitch Contour Plot (keep for backward compatibility)
        pitch_contour_filename = save_pitch_contour(y, sr)
        pitch_contour_path = f"static/{pitch_contour_filename}" if pitch_contour_filename else None
        
        # Generate Enhanced Waveform with Pitch Visualization
        waveform_path = save_waveform_with_pitch(y, sr)

        # Generate PDF Report with confidence scores
        pdf_filename = generate_pdf(
            predicted_label, 
            raga_description, 
            "Live Recording (10 sec)", 
            f"{sr} Hz", 
            pitch_contour_path,
            waveform_path,
            top_predictions
        )
        
        # Save prediction if user is logged in
        if 'user_id' in session:
            save_prediction(
                session['user_id'],
                predicted_label,
                top_predictions[0]["confidence"],
                "live_recording"
            )

        return jsonify({
            "raga": predicted_label,
            "description": raga_description,
            "pitch_contour_image": pitch_contour_path,
            "waveform_image": waveform_path,
            "pdf_report": pdf_filename,
            "top_predictions": top_predictions
        })
    except Exception as e:
        return jsonify({"error": f"Live prediction failed: {str(e)}"}), 500

# Route to Download PDF Report
@app.route("/download_report/<raga>")
def download_report(raga):
    pdf_path = f"static/reports/{raga}_report.pdf"
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return "Report not found", 404

# New route to get all available ragas (useful for voice commands)
@app.route("/available_ragas", methods=["GET"])
def available_ragas():
    return jsonify({
        "ragas": list(raga_info.keys())
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True)  # Changed to threaded=True for better performance