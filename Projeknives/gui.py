import tkinter as tk
import cv2
import os
from ultralytics import YOLO
from detectedknifevid import YOLOVideoProcessor
from detectknivesphoto import YOLOCaptureProcessor
import allpath as ap

# Ana pencereyi oluştur
window = tk.Tk()
window.geometry("640x480")
# Button fonksiyonu
def video_tiklandi():
    # YOLOVideoProcessor sınıfını başlat
    processor = YOLOVideoProcessor(ap.model_path, ap.video_path, ap.output_path)
    processor.process_video()

def video_foto_tiklandi():

    processor = YOLOCaptureProcessor(ap.model_path, ap.output_image_path)
    processor.start_capture()  
        

# Düğme ekle
video_button = tk.Button(window, text="Video link", command=video_tiklandi)
video_button.pack(pady=120)

video_foto_button = tk.Button(window, text="Video foto", command=video_foto_tiklandi)
video_foto_button.pack(pady=20)

# Pencereyi çalıştır
window.mainloop()
