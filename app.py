from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# 원격지의 ipcamera도 사용이 가능하다. 
# url = 'rtsp://admin:Admin001@gold33.iptime.org:555/2'  # 가든 85
# camera = cv2.VideoCapture(url)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)


while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    input_image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    input_image = (input_image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(input_image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Display label and confidence score on the image
    label = f"{class_name}: {confidence_score * 100:.2f}%"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Predicted online", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
