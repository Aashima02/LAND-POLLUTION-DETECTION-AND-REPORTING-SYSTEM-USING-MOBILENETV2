# LAND-POLLUTION-DETECTION-AND-REPORTING-SYSTEM-USING-MOBILENETV2
# Indian Sign Language Detection using Deep Learning - A CNN Approach

## PROJECT OUTLINE:

Land contamination, caused by inappropriate waste disposal, industrial activity, and unsustainable farming practices, is a major danger to the environment, human health, and biodiversity. Current approaches for identifying land contamination, which frequently rely on manual inspections, are inefficient, prone to mistake, and lack scalability. The goal of this project, "Land Pollution Detection and Reporting System using MobileNetV2," is to construct an automated system that can identify, evaluate, and monitor land pollution in real-time by using machine learning models and cutting-edge computer vision techniques. The technology is able to precisely identify and measure pollution levels by evaluating visual data from sources such as fixed cameras or unmanned aerial vehicles. The technology automatically notifies the appropriate authorities when pollution levels above a certain threshold, allowing for prompt actions.
The approach consists of gathering and preprocessing a collection of pollution photos, training a Convolutional Neural Network (CNN), and integrating the model into a real-time detection system. The technology will be extensively tested in both controlled and real-world conditions. This approach intends to greatly increase the efficacy and efficiency of land pollution control, supporting environmental sustainability, with possible future improvements including identifying more pollutant kinds and extending deployment regions.

## METHODOLOGY:

This project develops a real-time Land Pollution Detection and Reporting System using deep learning and computer vision. A Convolutional Neural Network (CNN) is trained on a large dataset of annotated images showing varying pollution levels, using preprocessing and tuning techniques to improve accuracy. Integrated into a real-time framework, the system analyzes live video from CCTV and drones, identifying pollution across different environmental conditions. If pollution in any frame exceeds a 70% threshold, the system automatically sends alerts to authorities with pollution severity and exact location details, enabling prompt action. The system undergoes extensive testing in diverse environments to assess its real-world performance, with feedback used for ongoing improvements. Future plans include expanding to monitor air and water pollutants, enabling a broader environmental monitoring network. This project offers a scalable solution for pollution control and real-time environmental protection through proactive monitoring and timely reporting.

### SAMPLE DATASET:
<img height="400" alt="dataset" src="https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/1340ab0b-0418-4a70-82a4-03ca07a6f864">


## REQUIREMENTS:

* A suitable python environment
* Python packages:
    
    * tensorflow
    * keras
    * opencv
    * smtp

The above packages can be manually installed using the pip commands as follows:
```python
pip install tensorflow
pip install keras
pip install opencv-python
```

## PROGRAM:
### cnn.py
```python
import os
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained garbage detection model
model = load_model('garbage_detection_model.h5')

# Email configuration
sender_email = "projectgarbage001@gmail.com"  # Replace with your email
receiver_email = "projectsanitaryoffice001@gmail.com"  # Replace with the receiver's email
email_password = "vpcx wybd xcvr xiwa"  # Replace with your App Password or email password

# Function to send an email alert
def send_email_alert(garbage_percentage, location="Location"):
    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"Garbage Alert: {garbage_percentage:.2f}% in {location}"

        # Email body content
        body = f"Alert! Garbage detection in {location} has exceeded 70%.\n\nCurrent garbage detected: {garbage_percentage:.2f}%. Take action to clear as soon as possible"
        msg.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server and send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Start TLS for security
        server.login(sender_email, email_password)  # Login to the SMTP server
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)  # Send email
        server.quit()  # Quit the server

        print(f"Alert email sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to detect garbage in a batch of patches
def detect_garbage_in_patches(patches, batch_size=32):
    preprocessed_patches = preprocess_input(patches)
    predictions = model.predict(preprocessed_patches, batch_size=batch_size)
    return predictions

# Function to analyze the image for pollution percentage using batch processing
def analyze_pollution_percentage(image, patch_size=(100, 100), threshold=0.5, batch_size=32):
    height, width, _ = image.shape
    patches = []
    total_patches = 0

    # Collect patches from the image
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):  # Fixed the loop range syntax
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]  # Correct slicing
            if patch.shape[0] != patch_size[1] or patch.shape[1] != patch_size[0]:
                continue
            resized_patch = cv2.resize(patch, (224, 224))
            patches.append(img_to_array(resized_patch))
            total_patches += 1

    # Convert patches to a numpy array
    patches_array = np.array(patches)

    # Batch processing of patches
    predictions = detect_garbage_in_patches(patches_array, batch_size=batch_size)

    # Count the number of garbage patches based on threshold
    garbage_count = np.sum(predictions < threshold)

    # Calculate the percentage of garbage detected
    if total_patches > 0:
        garbage_percentage = (garbage_count / total_patches) * 100
    else:
        garbage_percentage = 0

    return garbage_percentage

# Function to visualize the percentage of garbage detected on the image
def visualize_pollution_percentage(image, alert_sent, patch_size=(100, 100), threshold=0.5, batch_size=32):
    garbage_percentage = analyze_pollution_percentage(image, patch_size, threshold, batch_size)
    
    # Send an email alert if garbage exceeds 70% and alert has not been sent yet
    if garbage_percentage > 70 and not alert_sent:
        send_email_alert(garbage_percentage)
        alert_sent = True  # Mark alert as sent for this input

    # Display the pollution percentage on the image
    label = f"Garbage: {garbage_percentage:.2f}%"
    color = (0, 0, 255) if garbage_percentage > 70 else (0, 255, 0)  # Red if > 70%, Green otherwise
    
    cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image, alert_sent

# Function to process video input and analyze each frame for garbage detection
def process_video(input_source, is_file=False, patch_size=(100, 100), threshold=0.5, batch_size=32):
    global alert_sent
    alert_sent = False  # Reset alert flag for new video

    cap = cv2.VideoCapture(input_source if is_file else int(input_source))
    
    if not cap.isOpened():
        print("Error: Unable to access the video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or no more frames.")
            break
        
        analyzed_frame, alert_sent = visualize_pollution_percentage(frame, alert_sent, patch_size, threshold, batch_size)
        out.write(analyzed_frame)
        cv2.imshow('Garbage Detection', analyzed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to process a single image for garbage detection
def process_image(image_path, patch_size=(100, 100), threshold=0.5, batch_size=32):
    global alert_sent
    alert_sent = False  # Reset alert flag for new image

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    analyzed_image, alert_sent = visualize_pollution_percentage(image, alert_sent, patch_size, threshold, batch_size)

    # Display the final image with pollution percentage
    cv2.imshow('Garbage Detection', analyzed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to evaluate model accuracy
def evaluate_model_accuracy():
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Assuming you have test data in 'test/' directory with subdirectories 'garbage/' and 'not_garbage/'
    test_generator = test_datagen.flow_from_directory(
        'image_dataset/test',  # Directory with test images
        target_size=(224, 224),  # Resizing the images
        batch_size=32,
        class_mode='binary'
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    print(f"Model Accuracy on test data: {test_accuracy * 100:.2f}%")

# Main function to handle user input
def main():
    while True:
        print("\nSelect input source for garbage detection:")
        print("1. Camera")
        print("2. Video file")
        print("3. Image file")
        print("4. Evaluate model accuracy")
        print("5. Exit")
        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            process_video(0)  # Default camera input
        elif choice == '2':
            file_path = input("Enter the video file path: ")
            process_video(input_source=file_path, is_file=True)
        elif choice == '3':
            image_path = input("Enter the image file path: ")
            process_image(image_path)
        elif choice == '4':
            evaluate_model_accuracy()  # Call the accuracy evaluation function
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

```


## FLOW OF THE PROJECT:


1. The system receives live video feeds, which act as the primary input for pollution detection.
2. Frames are extracted from the video feed, and labels are added to identify different pollution levels.
3. The extracted frames undergo preprocessing, including resizing, normalization, and augmentation, to prepare them for model training.
4. The frames are split into training and testing sets, and a Convolutional Neural Network (CNN) is trained using these preprocessed frames for effective pollution detection.
5. The trained model is evaluated on test data to assess its accuracy and performance in identifying pollution levels.
6. After successful training and evaluation, the system processes live video feeds in real time, identifying pollution and triggering alerts when necessary.

![image](https://github.com/user-attachments/assets/e2bae321-c408-4472-a730-1a8d2d984da2)



## OUTPUT:

### Sample Detection Images:
![image](https://github.com/user-attachments/assets/f14b9a81-4e75-4095-97bc-b4d30da09f92)

![image](https://github.com/user-attachments/assets/f0e18b49-04f7-4f20-aed0-7fbd617f174e)

![image](https://github.com/user-attachments/assets/a945e9bd-9699-4d89-b96b-0ff69520e69b)


## RESULT:
Video frames were extracted and processed within a bounding box to isolate regions of interest, converting them into images compatible with the dataset for accurate pollution detection. Achieving an accuracy of 96.83%, this model effectively identifies and classifies pollution levels in real-time video feeds, enabling reliable alerts for authorities based on pollution severity and location.




















# Indian Sign Language Detection using Deep Learning - A CNN Approach

## PROJECT OUTLINE:

Humans converse through a vast collection of languages, allowing them to exchange information with one another. Sign language on the other hand, involves using visual signs and gestures that are vastly used by the people who have hearing and speaking disabilities.  There are over 300 sign languages over the world, that are actively used. The Indian Sign Language was standardized in 2001 and has been actively used all over the country since then. Having a difference in the mode of communication between the hearing people and the people with hearing losses causes a huge barrier and does not allow for easy conversing between the two communities. In addition the absence of certified interpreters also adds to improper communications and lag of understanding. To remove such barriers, researchers in various fields are actively looking for solutions so as the bridge the gap between the two worlds. The purpose of the project is to eradicate such barriers and lead to active an active communication mode between two parties. The research for developing such a system for the Indian Sign Language isn’t very much advanced. In this project, we consider the third edition of the Indian Sign Language (ISL) system and advance it to detect the alphabetic signs The goal of this project is to develop a deep learning system that helps in image classification using CNN to interpret signs generally used in the ISL.

## METHODOLOGY:

This experiment developed an efficient deep learning model to transform the Indian Sign Language signs for letters to textual format. We adapt an ISL dataset, preprocess them and feed them into a Convolution Neural Network(CNN). This CNN model is specifically designed for image processing. At the initial level of the CNN model, we’ve used a convolutional layer for feature extraction and MaxPooling layers for object recognition tasks. This set of outputs is later fed into a flattening layer, whose main function is to convert the 2-dimensional arrays from pooled feature maps into a long continuous linear vector. Such use of multiple layers within the CNN model helps in improving the accuracy and thus providing perfect interpretations.

### SAMPLE DATASET:
<img height="400" alt="dataset" src="https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/1340ab0b-0418-4a70-82a4-03ca07a6f864">


## REQUIREMENTS:

* A suitable python environment
* Python packages:
    
    * tensorflow
    * keras
    * opencv

The above packages can be manually installed using the pip commands as follows:
```python
pip install tensorflow
pip install keras
pip install opencv-python
```

## PROGRAM:
### cnn.py
```python
# Part 1 - Building the CNN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
# from tensorflow.keras import metrics
# # from metrics import accuracy
# from .accuracy import accuracy


# Initialing the CNN
classifier = Sequential()
# Step 1 - Convolution Layer 
classifier.add(Convolution2D(32, 3,  3, input_shape = (256, 256, 3), activation = 'relu'))
#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))
# Adding second convolution layer
classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
# 3rd convolution layer
classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
# Flattening Layer
classifier.add(Flatten())
# Final Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation = 'softmax'))

# Compile the model
classifier.compile(
              optimizer = optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=25,
        validation_data = test_set,
        validation_steps = 6500
      )

'''#Saving the model
import h5py
classifier.save('Trained_model.h5')'''


print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

### signdetect.py:
```python
import cv2
import numpy as np

def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
              return 'D'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
              return 'N'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'P'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       elif result[0][19] == 1:
              return 'T'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
              return 'W'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'
       

       

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()
```


## FLOW OF THE PROJECT:


1. The cnn.py file contains the main model of the project, containing multiple layers for feature extraction and object detecction.
2. The code is saved as a trained model in a Hierarchial Data Format(HDF5) file format.
3. Run signdetect.py, whose ultimate goal is to recognise the signs.
4. The opencv package creates a bounding box, where the signs are captured and converted to lower blue and upper blue hsv channels for identification.
5. The image is compared with the dataset and the letters are visualised in textual formats.

<img width="500" alt="flow" src="https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/bb74d302-9184-4e4a-8a18-308cd14993e3">


## CNN MODEL ARCHITECTURE:

![CNN ARCHITETURE](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/d1383de4-dcf4-49d3-9ced-9676f206e0ec)


## OUTPUT:

### Sample Detection Images:
![g](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/d9e56c1d-9474-4021-b85e-c1220b0b2937)

![u](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/60c2a7da-ce67-4622-95ac-127d3f60f22f)

![x](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/c31450a3-f71c-47f2-90da-00d88f2b6f8a)

### Accuracy Plot:
![accuracy plot](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/7b9d51fc-92c6-4642-9634-4e39a85834d2)


### Loss Plot:
![loss plot](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/1607d36c-baf9-4544-a54b-83ba05cd964d)



## RESULT:
In this study, we’ve made use of the third edition of the Indian Sign Language (ISL) - 2021 as the standard dataset. The input for sign detection was fed into a bounding box, whose region was converted as an image recognizable by the camera, and similar to the dataset. With the accuracy of 90% and loss of 28%, this model is able to detect individual signs for alphabets. 
