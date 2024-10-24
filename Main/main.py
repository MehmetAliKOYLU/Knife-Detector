import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from ultralytics import YOLO 
import sqlite3
import datetime
import allpath as ap
import pandas as pd  # Excel kaydetmek için gerekli
import os

# Kendi eğittiğiniz YOLO modelini yükleyin
save_dir = "detected_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Klasör yoksa oluşturulur
model_path = ap.model_path  # Model dosyasının doğru yolunu buraya yazın
model = YOLO(model_path)  # Eğittiğiniz modeli yükleyin

# Veritabanı bağlantısı oluştur
conn = sqlite3.connect('knife_detection.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY, detection_time TEXT)''')
conn.commit()

last_saved = None  # Veritabanına son kaydın zamanını izlemek için
threshold = 0.60  # Eşik değeri

def detect_from_image():
    global last_saved
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        results = model(img)  # Kendi modelinizi kullanarak resmi işleyin
        
        detection_flag = False  # Tespit edildiğinde True olacak
        
        for result in results:  # Tespit edilen nesneleri işleyin
            boxes = result.boxes  # Tespit edilen kutuları al
            for box in boxes:
                conf = box.conf.item()  # Güven skorunu float olarak al
                if conf >= threshold:  # Yalnızca eşik değerinden büyük tespitler
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Koordinatları al
                    
                    # Sınıf etiketini güvenli şekilde al
                    label = result.names[int(box.cls)] if hasattr(result, 'names') else 'Unknown'
                    
                    # Görüntüye kutu ve etiket ekle
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    detection_flag = True
        
        # Resmi ekranda daha uygun boyutlarda göstermek için yeniden boyutlandırma ekleyelim
        max_height = 800  # Maksimum yükseklik (piksellerde)
        max_width = 1200  # Maksimum genişlik (piksellerde)
        h, w, _ = img.shape

        if h > max_height or w > max_width:
            scale = min(max_width / w, max_height / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Tespit edildiyse ve 'k' tuşuna basıldıysa veritabanına kaydet
        if detection_flag and (last_saved is None or cv2.waitKey(1) & 0xFF == ord('k')):
            now = datetime.datetime.now()
            c.execute("INSERT INTO detections (detection_time) VALUES (?)", (now,))
            conn.commit()
            cv2.imwrite(f"detected_image_{now}.jpg", img)  # Tespit edilen görüntüyü kaydet
            last_saved = now  # Kaydedilen zamanı güncelle
        
        cv2.imshow("Resim - Bıçak Algılama", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_from_video():
    global last_saved
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)  # Model ile videodaki her kareyi işleyin
            detection_flag = False  # Tespit edildiğinde True olacak
            
            for result in results:  # Tespit edilen nesneleri işleyin
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf.item()  # Güven skorunu float olarak al
                    if conf >= threshold:  # Eşik değerine göre filtrele
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Sınıf etiketini güvenli şekilde al
                        label = result.names[int(box.cls)] if hasattr(result, 'names') else 'Unknown'
                        
                        # Görüntüye kutu ve etiket ekle
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        detection_flag = True
            
            # Tespit edildiyse ve 'k' tuşuna basıldıysa veritabanına kaydet
            if detection_flag and (last_saved is None or cv2.waitKey(1) & 0xFF == ord('k')):
                now = datetime.datetime.now()
                c.execute("INSERT INTO detections (detection_time) VALUES (?)", (now,))
                conn.commit()
                cv2.imwrite(f"detected_video_{now}.jpg", frame)  # Tespit edilen görüntüyü kaydet
                last_saved = now  # Kaydedilen zamanı güncelle
            
            cv2.imshow('Video - Bıçak Algılama', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



def detect_from_camera():
    global last_saved  # Son kaydın zamanını takip etmek için
    saved_once = False  # Tek seferlik kayıt kontrolü
    
    cap = cv2.VideoCapture(0)  # Kameradan görüntü al
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # Bıçak tespiti yap
        detection_flag = False  # Tespit edildiğinde True olacak
        
        for result in results:  # Tespit edilen nesneleri işleyin
            boxes = result.boxes  # Tespit edilen kutuları al
            for box in boxes:
                conf = box.conf.item()  # Güven skorunu float olarak al
                if conf >= threshold:  # Yalnızca eşik değerinden büyük tespitler
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Koordinatları al
                    
                    # Sınıf etiketini güvenli şekilde al
                    label = result.names[int(box.cls)] if hasattr(result, 'names') else 'Unknown'
                    
                    # Görüntüye kutu ve etiket ekle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    detection_flag = True
        
        # Tespit edildiyse ve daha önce kayıt yapılmadıysa veritabanına kaydet ve resmi kaydet
        if detection_flag and not saved_once:
            now = datetime.datetime.now()
            c.execute("INSERT INTO detections (detection_time) VALUES (?)", (now,))
            conn.commit()
            
            # Kaydetme işlemi için dosya adı belirleme
            image_filename = os.path.join(save_dir, f"detected_camera_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(image_filename, frame)  # Tespit edilen görüntüyü kaydet
            
            saved_once = True  # Kaydın yalnızca bir kere yapılmasını sağla
            last_saved = now  # Kaydedilen zamanı güncelle
        
        # Ekranda bilgilendirici metin göster
        cv2.putText(frame, "Press 'k' to enable saving again", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Canlı görüntüyü ve tespit edilen bıçakları göster
        cv2.imshow('Kamera - Bıçak Algılama', frame)
        
        # 'q' tuşuna basılınca döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 'k' tuşuna basıldığında tekrar kaydetmeye hazır hale getir
        if cv2.waitKey(1) & 0xFF == ord('k'):
            saved_once = False  # Kaydetmeye tekrar izin ver
        
    cap.release()
    cv2.destroyAllWindows()
# Veritabanındaki verileri Treeview'a yükleyen fonksiyon
def load_data():
    for row in tree.get_children():
        tree.delete(row)  # Önce mevcut verileri temizle
    c.execute("SELECT * FROM detections")
    rows = c.fetchall()
    for row in rows:
        tree.insert("", tk.END, values=row)

# Veritabanındaki seçili kaydı silen fonksiyon
def delete_selected_record():
    selected_item = tree.selection()
    if not selected_item:
        messagebox.showwarning("Uyarı", "Silmek için bir kayıt seçmelisiniz!")
        return
    record_id = tree.item(selected_item)['values'][0]  # ID'yi alıyoruz
    c.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    conn.commit()
    load_data()  # Silindikten sonra verileri güncelle
    messagebox.showinfo("Başarılı", "Seçili kayıt başarıyla silindi.")

# Veritabanındaki tüm kayıtları silen fonksiyon
def delete_all_records():
    confirm = messagebox.askyesno("Emin misiniz?", "Tüm kayıtları silmek istediğinize emin misiniz?")
    if confirm:
        c.execute("DELETE FROM detections")
        conn.commit()
        load_data()  # Silindikten sonra verileri güncelle
        messagebox.showinfo("Başarılı", "Tüm kayıtlar başarıyla silindi.")

# Veritabanındaki tüm verileri Excel dosyasına kaydeden fonksiyon
def save_to_excel():
    c.execute("SELECT * FROM detections")
    rows = c.fetchall()
    df = pd.DataFrame(rows, columns=["ID", "Detection Time"])
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if save_path:
        df.to_excel(save_path, index=False)
        messagebox.showinfo("Başarılı", "Veriler Excel dosyasına başarıyla kaydedildi.")

def quit_app():
    conn.close()
    root.quit()

# Tkinter arayüzü
root = tk.Tk()
root.title("Bıçak Algılama Sistemi")

# Tema ve stil kullanarak daha modern bir arayüz ekleyelim
style = ttk.Style()
style.theme_use('clam')  # Farklı bir tema seçebilirsiniz (örneğin 'alt', 'default', 'clam')

# Veritabanı verilerini göstermek için Treeview ekliyoruz
tree = ttk.Treeview(root, columns=("ID", "Detection Time"), show="headings")
tree.heading("ID", text="ID")
tree.heading("Detection Time", text="Detection Time")
tree.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# Verileri yükle butonu
load_button = ttk.Button(root, text="Verileri Yükle", command=load_data)
load_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Seçili kayıt silme butonu
delete_button = ttk.Button(root, text="Seçili Kaydı Sil", command=delete_selected_record)
delete_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

# Tüm kayıtları silme butonu
delete_all_button = ttk.Button(root, text="Tüm Kayıtları Sil", command=delete_all_records)
delete_all_button.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

# Excel'e kaydet butonu
save_button = ttk.Button(root, text="Excel'e Kaydet", command=save_to_excel)
save_button.grid(row=1, column=3, padx=10, pady=10, sticky="ew")

# Çıkış butonu
exit_button = ttk.Button(root, text="Çıkış", command=quit_app)
exit_button.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

# Algılama butonları
image_button = ttk.Button(root, text="Resimden Bıçak Algıla", command=detect_from_image)
image_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

video_button = ttk.Button(root, text="Videodan Bıçak Algıla", command=detect_from_video)
video_button.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

camera_button = ttk.Button(root, text="Kameradan Bıçak Algıla ve Kaydet", command=detect_from_camera)
camera_button.grid(row=3, column=2, padx=10, pady=10, sticky="ew")

exit_button.grid(row=3, column=3, padx=10, pady=10, sticky="ew")

root.mainloop()
