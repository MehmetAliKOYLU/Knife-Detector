
import tkinter as tk
import cv2
import os
from ultralytics import YOLO
from detectedknifevid import YOLOVideoProcessor
from detectknivesphoto import YOLOCaptureProcessor
import allpath as ap

# Ana pencereyi oluştur
window = tk.Tk()
window.geometry("800x600")  # Pencere boyutu genişletildi
window.configure(bg="black")  # Arka planı siyah yaptık

# Başlık ekle
title_label = tk.Label(window, text="BICAK ALGILAMA PROGRAMI", font=("Arial", 20), fg="yellow", bg="black")
title_label.pack(pady=20)

# Button fonksiyonu
def video_tiklandi():
    processor = YOLOVideoProcessor(ap.model_path, ap.video_path, ap.output_path,.6)
    processor.process_video()

def video_foto_tiklandi():
    processor = YOLOCaptureProcessor(ap.model_path, ap.output_image_path,.6)
    processor.start_capture()

# Butonlar ekle ve yan yana hizala
button_frame = tk.Frame(window, bg="black")
button_frame.pack(pady=100)

video_button = tk.Button(button_frame, text="Video link", command=video_tiklandi, width=20, height=2, bg="yellow", fg="black")
video_button.grid(row=0, column=0, padx=50)

video_foto_button = tk.Button(button_frame, text="Video foto", command=video_foto_tiklandi, width=20, height=2, bg="yellow", fg="black")
video_foto_button.grid(row=0, column=1, padx=50)

# Alt yazı ekle
footer_label = tk.Label(window, text="Bu proje mehmet ali tarafından yapılmıştır.", font=("Arial", 10), fg="yellow", bg="black")
footer_label.pack(side="bottom", pady=20)

# Pencereyi çalıştır
window.mainloop()
