import cv2
import os
from ultralytics import YOLO

class YOLOCaptureProcessor:
    def __init__(self, model_path, output_image_path, threshold=0.7, knife_class_id=0):
        self.model_path = model_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.knife_class_id = knife_class_id
        self.model = YOLO(model_path).to('cuda')  # CUDA desteğin varsa bu şekilde kullanabilirsin

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def process_frame(self, frame):
        results = self.model(frame)[0] #frame 0 bicak 0. indekste
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
                cv2.putText(frame, "Bicak Algilandi", (int(x1), int(y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                # Bıçak tespit edilirse resmi kaydet
                if int(class_id) == self.knife_class_id:
                    cv2.imwrite(self.output_image_path, frame)
                    print(f"Bıçak tespit edildi ve '{self.output_image_path}' olarak kaydedildi.")
        return frame

    def start_capture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Goruntu islenemedi.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('YOLO Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()
