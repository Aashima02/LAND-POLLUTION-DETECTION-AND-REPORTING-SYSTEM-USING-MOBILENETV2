# LAND-POLLUTION-DETECTION-AND-REPORTING-SYSTEM-USING-MOBILENETV2

## PROJECT OUTLINE:

Land contamination, caused by inappropriate waste disposal, industrial activity, and unsustainable farming practices, is a major danger to the environment, human health, and biodiversity. Current approaches for identifying land contamination, which frequently rely on manual inspections, are inefficient, prone to mistake, and lack scalability. The goal of this project, "Land Pollution Detection and Reporting System using MobileNetV2," is to construct an automated system that can identify, evaluate, and monitor land pollution in real-time by using machine learning models and cutting-edge computer vision techniques. The technology is able to precisely identify and measure pollution levels by evaluating visual data from sources such as fixed cameras or unmanned aerial vehicles. The technology automatically notifies the appropriate authorities when pollution levels above a certain threshold, allowing for prompt actions.
The approach consists of gathering and preprocessing a collection of pollution photos, training a Convolutional Neural Network (CNN), and integrating the model into a real-time detection system. The technology will be extensively tested in both controlled and real-world conditions. This approach intends to greatly increase the efficacy and efficiency of land pollution control, supporting environmental sustainability, with possible future improvements including identifying more pollutant kinds and extending deployment regions.

## METHODOLOGY:

This project develops a real-time Land Pollution Detection and Reporting System using deep learning and computer vision. A Convolutional Neural Network (CNN) is trained on a large dataset of annotated images showing varying pollution levels, using preprocessing and tuning techniques to improve accuracy. Integrated into a real-time framework, the system analyzes live video from CCTV and drones, identifying pollution across different environmental conditions. If pollution in any frame exceeds a 70% threshold, the system automatically sends alerts to authorities with pollution severity and exact location details, enabling prompt action. The system undergoes extensive testing in diverse environments to assess its real-world performance, with feedback used for ongoing improvements. Future plans include expanding to monitor air and water pollutants, enabling a broader environmental monitoring network. This project offers a scalable solution for pollution control and real-time environmental protection through proactive monitoring and timely reporting.

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




















