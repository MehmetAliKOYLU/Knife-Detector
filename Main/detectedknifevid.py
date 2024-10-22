import cv2
import os
from ultralytics import YOLO

class YOLOVideoProcessor:
    def __init__(self, model_path, video_path, output_path, threshold=0.7):
        self.model_path = model_path # Modelin dosya yolunu tanimladim
        self.video_path = video_path #Okuyacagi videonun dosya yolunu bulmasi icin yaptim
        self.output_path = output_path # Cikti olarak yeni videoyu nereye kaydedecegini anlattim
        self.threshold = threshold # algilamada yuzde kac deger ustu gordugu nesnelere tepki vermesi gerektigini anlattim
        self.model = YOLO(model_path).to('cuda')  #ekran kartimi kullanmak icin yaptim
        self.cap = cv2.VideoCapture(video_path) #Cv2 video yakalama ozelligi ile dosya yolunda bulunan videoyu okumasini sagladim
            #Videonun genislik yukseklik ve fps degerlerini aldim
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.video_format = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(self.output_path, self.video_format, self.fps, 
                                             (self.frame_width, self.frame_height))
    
    def process_frame(self, frame):

        results = self.model(frame)[0] # Sonuc kismina bicak algilandigi zaman kaydedecek

        for result in results.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = result

            if score > self.threshold: #Eger score degeri treshold degerinden yuksekse yani %70 ustundeyse bicak algilandi.
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, "Bicak Algilandi", (int(x1), int(y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        return frame

    def process_video(self):

        while True:
            ret, frame = self.cap.read() 
            if not ret:  #Videoyu okuyup okuyamadigina gore bir uyari mesaji gonderdim
                print("Goruntu islenemedi.")
                break

            processed_frame = self.process_frame(frame)
            self.output_writer.write(processed_frame)
            cv2.imshow('YOLO Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def release_resources(self):
        self.cap.release()
        self.output_writer.release()
        cv2.destroyAllWindows()

