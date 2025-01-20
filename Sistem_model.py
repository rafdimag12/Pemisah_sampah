import tensorflow as tf
import numpy as np
import cv2
import RPi.GPIO as GPIO
from picamera import PiCamera
from time import sleep

# Konfigurasi GPIO untuk Relay
RELAY_1_PIN = 17  # Ganti dengan pin GPIO yang digunakan untuk Relay 1
RELAY_2_PIN = 27  # Ganti dengan pin GPIO yang digunakan untuk Relay 2

# Inisialisasi GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_1_PIN, GPIO.OUT)
GPIO.setup(RELAY_2_PIN, GPIO.OUT)

# Matikan semua relay di awal
GPIO.output(RELAY_1_PIN, GPIO.LOW)
GPIO.output(RELAY_2_PIN, GPIO.LOW)

# Load model machine learning
model = tf.keras.models.load_model('waste_classifier_supervised.h5')

# Labels untuk klasifikasi
labels = ['Plastic', 'Paper', 'Metal']

# Inisialisasi kamera Raspberry Pi
camera = PiCamera()
camera.resolution = (150, 150)

def classify_image(image):
    """
    Klasifikasikan gambar menjadi Plastik, Kertas, atau Logam.
    """
    # Preproses gambar
    image = cv2.resize(image, (150, 150))
    image = image / 255.0  # Normalisasi piksel
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch

    # Prediksi kategori
    predictions = model.predict(image)
    class_index = np.argmax(predictions)  # Index kategori dengan probabilitas tertinggi
    return labels[class_index]

def control_relay(category):
    """
    Kontrol relay berdasarkan kategori sampah.
    """
    if category == 'Plastic':
        GPIO.output(RELAY_1_PIN, GPIO.HIGH)  # Relay 1 nyala
        GPIO.output(RELAY_2_PIN, GPIO.LOW)   # Relay 2 mati
        
    elif category == 'Paper':
        GPIO.output(RELAY_1_PIN, GPIO.LOW)   # Relay 1 mati
        GPIO.output(RELAY_2_PIN, GPIO.HIGH)  # Relay 2 nyala
        
    elif category == 'Metal':
        GPIO.output(RELAY_1_PIN, GPIO.LOW)   # Semua relay mati
        GPIO.output(RELAY_2_PIN, GPIO.LOW)
        

# Loop utama untuk klasifikasi dan kontrol relay
try:
    while True:
        print("Mengambil gambar...")
        camera.capture('image.jpg')  # Ambil gambar dari kamera
        img = cv2.imread('image.jpg')  # Baca gambar dari file
        category = classify_image(img)  # Klasifikasikan gambar

        print(f"Kategori sampah terdeteksi: {category}")
        control_relay(category)  # Kontrol relay berdasarkan kategori

        sleep(2)  # Tunggu sebelum iterasi berikutnya
except KeyboardInterrupt:
    print("Program dihentikan.")
finally:
    # Matikan semua relay sebelum keluar
    GPIO.output(RELAY_1_PIN, GPIO.LOW)
    GPIO.output(RELAY_2_PIN, GPIO.LOW)
    GPIO.cleanup()
    camera.close()
