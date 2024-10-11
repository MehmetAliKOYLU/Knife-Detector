#from roboflow import Roboflow
#rf = Roboflow(api_key="TYmJJ0qGMpiydakhO3By")
#project = rf.workspace("sona-thc0d").project("knives-zcsqv")
#version = project.version(5)
#dataset = version.download("yolov8")

import os
import torch
from ultralytics import YOLO
from IPython.display import display, Image
from IPython import display # type: ignore

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if __name__ == "__main__":


    model = YOLO("yolov8n.pt")
    
    # train 1,2 3 sirasiyla lr = 0.01  epoch degerleri ise 45 50 100 yapildi .
    # train 4 lr 0.001 epoch 45 
    # train 5 lr 0.001 epoch 50 
    # train 6 lr 0.001 epoch 100
    # train 7 lr 0.001 epoch 100 benim ekledigim ek verilerle
    # train 8 lr 0.001 epoch 100 ben ve yucellerden aldigim verilerle 
    # train 9 yolov8s modelini kullanarak tekrar yapilmis hali 
    # train 10 lr 0.0001 ile 0.0005 arasi 100 veri set 10 patience yolovn
    # onceki train 1 ve 2 az deger veridigi icin silindi
    # train 1 lr 0.0001 ile 0.0005 epoch 70 ben ve yucellerden aldigim verilerle ,5 patience  yolov8m \
    # train 2 lr 0.0001 ile 0.0005 epoch 100 veriseti valid degismis halde yolov8n ile tekrar denenior
   
     #yapildi karsilastirilip en iyi sonucu veren hali kullanilacak

    results = model.train(data="data.yaml", epochs=100, imgsz=640, lr0=0.0001, lrf=0.0005, device=device, patience=10)
    #results = model.train(data="data.yaml", epochs=100, imgsz=640,lr0=0.001,device=device, workers=4)  # num_workers = 0
