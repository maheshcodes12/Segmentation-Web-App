import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function from your Colab notebook
def getX(img):
    height = 256
    width = 256
    # initialize the feature matrix
    X = np.empty([65536, 4], dtype=float)
    i = 0 

    # black and white image for computing the entropy
    grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgEntropy = entropy(grayScaleImg, disk(3))

    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            blueVal = float(pixel[0])
            greenVal = float(pixel[1])
            redVal = float(pixel[2])
            entropyVal = float(imgEntropy[y, x])

            values = np.array([redVal, greenVal, blueVal, entropyVal])
            X[i] = values
            i+=1

    return X

# Function from your Colab notebook
def getSegmentation(X):
    # classify the pixels
    model = torch.load('models/G_model_geo_2topClass.h5',weights_only=False)
    X = torch.Tensor(X)
    y_vals = model(X)
    cat_y = torch.argmax(y_vals, dim=1)
    labels = cat_y.numpy()

    # now build the matrix of labels
    imgLabels = np.empty([256, 256], dtype=np.uint8)
    i = 0
    for y in range(256):
        for x in range(256):
            imgLabels[y, x] = labels[i]
            i += 1

    # now build the segmented image from the labels 
    segmentedImg = np.ones([256, 256, 3], dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            if imgLabels[y, x] == 0: # building -> red
                segmentedImg[y, x][0] = 255
                segmentedImg[y, x][1] = 0
                segmentedImg[y, x][2] = 0
  
            elif imgLabels[y, x] == 1: # road -> yellow
                segmentedImg[y, x][0] = 255
                segmentedImg[y, x][1] = 255
                segmentedImg[y, x][2] = 0
      
            elif imgLabels[y, x] == 2: # pavement -> darker yellow
                segmentedImg[y, x][0] = 192
                segmentedImg[y, x][1] = 192
                segmentedImg[y, x][2] = 0
  
            elif imgLabels[y, x] == 3: # vegetation -> green
                segmentedImg[y, x][0] = 0
                segmentedImg[y, x][1] = 255
                segmentedImg[y, x][2] = 0

            elif imgLabels[y, x] == 4: # bare soil -> grey
                segmentedImg[y, x][0] = 128
                segmentedImg[y, x][1] = 128
                segmentedImg[y, x][2] = 128
  
            else: # water -> blue
                segmentedImg[y, x][0] = 0
                segmentedImg[y, x][1] = 0
                segmentedImg[y, x][2] = 255

    return segmentedImg

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            
            # Resize to 256x256 if needed
            if img.shape[0] != 256 or img.shape[1] != 256:
                img = cv2.resize(img, (256, 256))
            
            # Get features and segmentation
            X = getX(img)
            segmented_img = getSegmentation(X)
            
            # Save the result
            result_filename = 'segmented_' + filename
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
            
            return render_template('index.html', 
                                 original_image=filepath, 
                                 segmented_image=result_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)