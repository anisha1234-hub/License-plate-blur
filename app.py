import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def blur_license_plate(image_path):
    image = cv2.imread(image_path)
    plate_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_russian_plate_number.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in plates:
        plate = image[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(plate, (25, 25), 30)
        image[y:y+h, x:x+w] = blurred

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.jpg')
    cv2.imwrite(output_path, image)
    return output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_path = blur_license_plate(filepath)
        return render_template('index.html', uploaded=True, image_path=processed_path)

    return render_template('index.html', uploaded=False)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

