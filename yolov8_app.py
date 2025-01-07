import os
from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import cv2
from werkzeug.utils import secure_filename
from flask import send_from_directory

# Initialize Flask app
app = Flask(__name__)

@app.route('/runs/<path:filename>')
def serve_runs(filename):
    return send_from_directory('runs', filename)


# Set upload folder
UPLOAD_FOLDER = "uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('models/yolov8m.pt')

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for image prediction
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform inference
        results = model.predict(source=filepath, save=True)
        annotated_image_path = f"detect/predict/{os.path.splitext(filename)[0]}.jpg"

        # Debug: Print the path for verification
        print("Annotated image path:", annotated_image_path)

        return render_template('result.html', image_path=annotated_image_path)


# Route for live webcam
@app.route('/live_webcam')
def live_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Show the live feed with annotations
        cv2.imshow('Live Webcam Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
