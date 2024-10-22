import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Modeli tanımla
model = YOLO("yolov8n.pt")

# Eğer training_history.pkl dosyası varsa onu yükle, yoksa yeni bir dictionary başlat
if os.path.exists('training_history.pkl'):
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)
else:
    history = {'train_loss': [], 'val_loss': [], 'metrics': []}

# Eğitim fonksiyonu
def train_model(optimizer_name, lr, epochs=100):
    print(f"Training with optimizer: {optimizer_name}, learning rate: {lr}")
    
    # Seçilen optimizer ismi dinamik olarak kullanılıyor
    results = model.train(
        data="./training/data.yaml",
        epochs=epochs,
        optimizer=optimizer_name,  # 'Adam' yerine dinamik optimizer ismi
        imgsz=640,
        lr0=lr,
        device=device,
        patience=10,
        workers=8
    )
    
    # Eğitim sırasında sonuçlara results direkt olarak erişelim
    try:
        train_loss = results.loss['train']  # Eğitim kaybı
        val_loss = results.loss['val']  # Doğrulama kaybı
    except KeyError:
        print("Train veya val loss bulunamadı. Sonuçları kontrol edin.")
        train_loss, val_loss = None, None

    # Eğitim süresince kaydedilen metrikleri dictionary'ye ekle
    if train_loss is not None and val_loss is not None:
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['metrics'].append(results.metrics)  # Diğer metrikler
    
    # Her model eğitimi sonrası modelin ağırlıklarını kaydet
    torch.save(model.model.state_dict(), f"./model_weights_{optimizer_name}_lr{lr}.pt")
    
    # Her eğitim bittikten sonra pickle dosyasına yeni sonuçları ekleyerek kaydet
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

# Eğitim ve sonuçları çizdirme kısmı
if __name__ == '__main__':
    # Kullanmak istediğiniz optimizer isimlerini ve learning rate değerlerini tanımlayın
    optimizer_name = 'RMSProp'  
    learning_rate = 0.001 ## burada kaldin

    # Modeli eğit ve sonuçları kaydet
    train_model(optimizer_name, learning_rate)

    # Eğitim sonrası sonuçları çizdirme (isteğe bağlı)
    for key, value in history.items():
        plt.figure(figsize=(10, 5))
        for i, losses in enumerate(value):
            plt.plot(losses, label=f'{optimizer_name} - lr{learning_rate}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{key} per Optimizer and Learning Rate')
        plt.legend()
        plt.show()
