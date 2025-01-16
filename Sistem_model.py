import cv2
import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
from time import sleep


# Load model
model = tf.keras.models.load_model("waste_classifier_model.h5")

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Preprocess image
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)

    # Predict category
    prediction = model.predict(input_data)
    class_idx = np.argmax(prediction)
    categories = ["Plastic", "Paper", "Metal", "Glass"]
    predicted_category = categories[class_idx]

    # Display result
    cv2.putText(frame, f"Category: {predicted_category}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

PIO.setmode(GPIO.BCM)
servo_pin = 17
GPIO.setup(servo_pin, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)
servo.start(0)

def move_servo(position):
    servo.ChangeDutyCycle(position)
    sleep(1)


if predicted_category == "Plastic":
    move_servo(5)  
elif predicted_category == "Paper":
    move_servo(7)  
elif predicted_category == "Metal":
    move_servo(11)  
elif predicted_category == "Glass":
    move_servo(15)  


servo.stop()
GPIO.cleanup()