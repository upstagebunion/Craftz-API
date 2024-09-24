# app/main.py

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_uploads import configure_uploads, IMAGES, UploadSet
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
import os

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Asegurarse de que el directorio de subida existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Configurar el set de subidas
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'  # Carpeta donde se guardan las imágenes subidas
configure_uploads(app, photos)

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        # Aquí procesarás la imagen después (detección de playera)
        return f"Imagen subida exitosamente: {filepath}"  # Confirmación básica
    return render_template('upload.html')  # Página con formulario de subida


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Revisar si se ha subido un archivo
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Procesar la imagen para detectar la playera
            processed_image_path = process_image(file_path)
            
            return f'File uploaded and processed: {processed_image_path}'
    return render_template('upload.html')

def process_image(image_path):
    # Cargar la imagen usando OpenCV
    img = cv2.imread(image_path)
    
    # Convertir la imagen a espacio de color HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir un rango de color para detectar la playera (esto puede variar)
    # Supongamos que la playera es de color azul:
    lower_color = (90, 50, 50)
    upper_color = (130, 255, 255)

    # Crear una máscara que detecte solo los píxeles dentro del rango de color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(img, img, mask=mask)

    # Guardar la imagen procesada
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, result)

    return processed_image_path

if __name__ == '__main__':
    app.run(debug=True)