from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import os
from steganography import steganography, extract_watermark

app = Flask(__name__)

# Set the upload folder and allowed extensions for uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/encrypt', methods=['GET', 'POST'])
def encrypt():
    if request.method == 'POST':
        # Check if the request contains files
        if 'cover_image' not in request.files or 'watermark' not in request.files:
            return redirect(request.url)
        
        cover_image = request.files['cover_image']
        watermark = request.files['watermark']
        
        # Check if the files are empty or not allowed
        if cover_image.filename == '' or not allowed_file(cover_image.filename):
            return redirect(request.url)
        if watermark.filename == '' or not allowed_file(watermark.filename):
            return redirect(request.url)
        
        # Save the files to the upload folder
        cover_image_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_image.filename)
        watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark.filename)
        cover_image.save(cover_image_path)
        watermark.save(watermark_path)
        
        # Load the images and perform steganography
        img1 = cv2.imread(cover_image_path)
        img2 = cv2.imread(watermark_path)
        alpha = request.form['alpha']
        alpha = float(alpha)
        stego_image, _, _ = steganography(img1, img2, alpha)
        
        # Save the stego image to the upload folder
        stego_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stego_' + cover_image.filename)
        cv2.imwrite(stego_image_path, stego_image)
        
        return redirect(url_for('result', filename='stego_' + cover_image.filename))
    
    return render_template('encrypt.html')

@app.route('/decrypt', methods=['GET', 'POST'])
def decrypt():
    if request.method == 'POST':
        # Check if the request contains files
        if 'cover_image' not in request.files or 'stego_image' not in request.files:
            return redirect(request.url)
        
        cover_image = request.files['cover_image']
        stego_image = request.files['stego_image']
        
        # Check if the files are empty or not allowed
        if cover_image.filename == '' or not allowed_file(cover_image.filename):
            return redirect(request.url)
        if stego_image.filename == '' or not allowed_file(stego_image.filename):
            return redirect(request.url)
        
        # Save the files to the upload folder
        cover_image_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_image.filename)
        stego_image_path = os.path.join(app.config['UPLOAD_FOLDER'], stego_image.filename)
        cover_image.save(cover_image_path)
        stego_image.save(stego_image_path)
    
        # Load the images and extract the watermark
        img1 = cv2.imread(cover_image_path)
        img2 = cv2.imread(stego_image_path)
        alpha = request.form['alpha']
        alpha = float(alpha)
        watermark, corr_coef = extract_watermark(img1, img2, alpha)
        
        # Save the extracted watermark to the upload folder
        watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], 'watermark_' + stego_image.filename)
        cv2.imwrite(watermark_path, watermark)
        
        # Render the result template with the extracted watermark and correlation coefficient
        return render_template('result.html', filename='watermark_' + stego_image.filename, corr_coef=corr_coef)

    return render_template('decrypt.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)