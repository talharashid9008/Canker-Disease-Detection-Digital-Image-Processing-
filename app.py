# Flask example
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
import cv2


# Load pre-trained model
model = tf.keras.models.load_model('C:\\Users\\Talha\\OneDrive\\Documents\\Dip_Canker\\cankerMod.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Create Flask app
app = Flask(__name__)

# Create route for image uploads
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get uploaded image
        img_file = request.files['image']
        _, ext = os.path.splitext(img_file.filename)
        if ext not in ('.png', '.jpg', '.jpeg','.JPG'):
            return "Invalid image file format. Allowed formats are png, jpg and jpeg."
        file = request.files['image']
        filename = "uploaded.jpg"
        filepath = os.path.join('static', filename)
        file.save(filepath)
        # Load the image
        img = cv2.imread("C:\\Users\\Talha\\OneDrive\\Documents\\Dip_Canker\\static\\uploaded.jpg")

        # Apply the Laplacian filter to enhance edges
        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        # Add the original image and the Laplacian image together to sharpen the image
        #sharpened = cv2.addWeighted(img, 1.5, laplacian, -0.5, 0)
        sharpened = cv2.addWeighted(img, 1.1, laplacian, -0.1, 0, dtype=cv2.CV_8U)

        filename = "uploaded.jpg"
        filepath = os.path.join('static', filename)
        #file.save(filepath)
        cv2.imwrite(filepath, sharpened)
        
        # Get the path of the saved image
        img_path = os.path.abspath(filepath)
        # Load the saved image and make predictions
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)

        # Render result page
        return render_template('result.html' , predicted_class=predicted_class, confidence=confidence)

    return render_template('upload.html')
# Create route for image details
@app.route('/image_details')
def image_details():
    # Get the path to the last uploaded image
    filename = "uploaded.jpg"
    last_image_path = os.path.join('static', filename)
    #************************************************************
        
    # read the image
    img = cv2.imread('C:\\Users\\Talha\\OneDrive\\Documents\\Dip_Canker\\static\\uploaded.jpg')#image.jpg

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([15,45,65])
    upper_brown = np.array([110,140,175])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    safe_leaf = cv2.bitwise_and(img, img, mask=mask)
    filename = "uploaded.jpg"
    filepath = os.path.join('static', filename)
    cv2.imwrite(filepath, safe_leaf)
    #************************************************************

    # Render image details page
    return render_template('image_details.html')



if __name__ == '__main__':
    app.run(debug=True)
