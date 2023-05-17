from flask import Flask, render_template, request, redirect, url_for, send_file
from steganography import steganography, extract_watermark
import cv2
import io
import os
import numpy as np
from werkzeug.utils import secure_filename
import math
import imageio.v2 as imageio
import uuid

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Home page
# @app.route('/')
# def index():
#     return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encryptdecrypt')
def encryptdecrypt():
    return render_template('encryptdecrypt.html')

@app.route('/alphablendingblog')
def alphablendingblog():
    return render_template('alphablendingblog.html')

@app.route('/psnrblog')
def psnrblog():
    return render_template('psnrblog.html')

# # Embed watermark page
@app.route('/encrypt', methods=['GET', 'POST'])
def encrypt():
    if request.method == 'POST':
        # Get form data
        # alpha = float(request.form['alpha'])
        alpha = float(request.form['alpha'])
        img1 = request.files['cover_image']
        img2 = request.files['watermark']
        
        
       # Read the contents of the file object into a NumPy array
        cover_image_array = np.frombuffer(img2.read(), dtype=np.uint8)

        # Reshape the array to its original dimensions (assuming it's an image)
        # and convert it to BGR format (OpenCV's default)
        cover_image = cv2.imdecode(cover_image_array, cv2.IMREAD_COLOR)

        # Save the image to disk as a JPEG file
        save_path = 'static/image/aayoo.jpg'
        cv2.imwrite(save_path, cover_image)
        # if(request.files['alpha']=='' and request.files['cover_image'] == '' and request.files['watermark'] ==''):
        #     # if(alpha=='' and img1 =="" and img2 ==""):
        #     render_template('home.html')
        # save_dir='C:/Users/dines/Desktop/project 2/static/image'
        # filename= 'aayoo' + '.jpg'
        # save_path =os.path.join(save_dir,filename)
        # img1.save(save_path)
        # save_path =os.path.join('static','image','aayoo.jpg')
        # imageio.imwrite(save_path,img2)
        # cv2.imwrite(save_path,img2)
        # Read the images using OpenCV
        # img1_cv = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # img2_cv = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # Call the steganography function to embed the watermark
        # watermarked_image, psnr, mse = steganography(img1_cv, img2_cv, alpha)
        watermarked_image, psnr, mse = steganography(img1, img2, alpha)
        # Create a buffer to store the image
        # buf = io.BytesIO()
        # Save the image to the buffer
        # cv2.imencode('.jpeg', final_img)[1].tofile(buf)
        # buf.seek(0)
        # cv2.imwrite('output.jpg', final_img)
        # Return the image and PSNR value to the client
        # return send_file(buf, mimetype='image/jpg', as_attachment=True, attachment_filename='output.jpg'), psnr
        
        # final_img.save(os.path.join(app.config['UPLOAD_FOLDER'],final_img.filename))
        cv2.imwrite('static/image/watermarked.jpg', watermarked_image)
        # print(final_img.filename)
        # final_img.save(os.path.join(app.config['UPLOAD_FOLDER'],final_img.filename))
        # Return the image and PSNR value to the client
        
        #here it is for showing image preview
        #cover_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['cover_image'].filename))
        #watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['watermark'].filename))
        #watermarked_image_path = os.path.join(app.config['STATIC_FOLDER'], 'image', 'watermarked.jpg')

        #return render_template("encrypt.html", final_watermarked_image='watermarked.jpg', psnr=psnr, mse= mse,cover_image=cover_image_path,watermark=watermark_path,watermarked_image=watermarked_image_path)
        # return render_template("home.html", final_img =  cv2.imwrite('output.jpg', final_img))
        return render_template("encrypt.html", final_watermarked_image='watermarked.jpg', psnr=psnr, mse= mse)
    else:
        return render_template('encrypt.html')

# Extract watermark page
@app.route('/decrypt', methods=['GET', 'POST'])
def decrypt():
    if request.method == 'POST':
        # Get form data
        alpha = float(request.form['alpha'])
        img1 = request.files['watermarked_image']
        img2 = request.files['cover_image']
        # # Read the images using OpenCV
        # img1_cv = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # img2_cv = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # Call the extract_watermark function to extract the watermark
        watermark, psnr, mse = extract_watermark(img1, img2, alpha)
        # Create a buffer to store the image
        # buf = io.BytesIO()
        # Save the image to the buffer
        # cv2.imencode('.png', watermark)[1].tofile(buf)
        # buf.seek(0)
        # # Invert the color of the watermark
        # watermark = cv2.bitwise_not(watermark)
        
        cv2.imwrite('static/image/decrypted.jpg', watermark)

        # Return the image to the client
        return render_template("decrypt.html", final_img='decrypted.jpg', psnr=psnr, mse= mse)
    else:
        return render_template('decrypt.html')

@app.route('/download_processed_image')
def download_processed_image():
    return send_file('static/image/watermarked.jpg', as_attachment=True)

@app.route('/download_decrypted_image')
def download_decrypted_image():
    return send_file('static/image/decrypted.jpg', as_attachment=True)

# @app.route('/test_files')
# def test_file():
#     return url_for('cover.jpg')

# @app.route() add error page


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)