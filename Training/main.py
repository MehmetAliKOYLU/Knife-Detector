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
# if os.path.exists('training_history.pkl'):
#     with open('training_history.pkl', 'rb') as f:
#         history = pickle.load(f)
# else:
#     history = {'train_loss': [], 'val_loss': [], 'metrics': []}

# Eğitim fonksiyonu
def train_model(optimizer_name, lr, epochs=100):
    print(f"Training with optimizer: {optimizer_name}, learning rate: {lr}")
    
    # Seçilen optimizer ismi dinamik olarak kullanılıyor
    results = model.train(
        data="./training/data.yaml",
        epochs=1,
        optimizer=optimizer_name,  # 'Adam' yerine dinamik optimizer ismi
        imgsz=640,
        lr0=lr,
        device=device,
        patience=10,
        workers=8
    )
    
    print("-"*100)
    #print(results.history.keys())
    print("-"*100)
    print(results.__dict__)
    
    train_loss_history = []
    val_loss_history = []
    mAP_history = []

    # Eğitim sonuçlarını inceleyip kaydetme
    for epoch in range(len(results)):
        train_loss = results['train']['loss'][epoch]  # Her epoch için eğitim kaybı
        val_loss = results['val']['loss'][epoch]  # Her epoch için validasyon kaybı
        mAP_50_95 = results['metrics']['mAP_50_95'][epoch]  # Her epoch için mAP

        # Geçmişi kaydet
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        mAP_history.append(mAP_50_95)


    with open('first_training_history.pkl', 'wb') as f:
        pickle.dump({
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'mAP': mAP_history
        }, f)
    # Eğitim sırasında sonuçlara results direkt olarak erişelim
    # try:
    #     # Box Loss (Bounding Box kaybı)
    #     train_box_loss = results.box_loss
    #     # Class Loss (Sınıflandırma kaybı)
    #     train_cls_loss = results.cls_loss
    #     # DFL Loss (Distribution Focal Loss kaybı)
    #     train_dfl_loss = results.dfl_loss

    #     val_box_loss = results.metrics['val/box_loss']
    #     val_cls_loss = results.metrics['val/cls_loss']
    #     val_dfl_loss = results.metrics['val/dfl_loss']

    #     train_loss = results.history['train']  # Eğitim kaybı
    #     val_loss = results.history['val']  # Doğrulama kaybı
    # except KeyError:
    #     print("Train veya val loss bulunamadı. Sonuçları kontrol edin.")
    #     val_box_loss, val_cls_loss, val_dfl_loss = None, None, None

    # Eğitim süresince kaydedilen metrikleri dictionary'ye ekle
        # if train_loss is not None and val_loss is not None:
        #     history['train_loss'].append(train_loss)
        #     history['val_loss'].append(val_loss)
        #     history['metrics'].append(results.metrics)  # Diğer metrikler
        
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
