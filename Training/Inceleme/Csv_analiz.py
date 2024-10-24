import os
import pandas as pd
import matplotlib.pyplot as plt

# Dizin varsa kontrol et, yoksa oluştur
save_directory = "C:/Users/koylu/Desktop/Proje/Training"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# mAP50 değeri en yüksek olan satırı bulma fonksiyonu
def get_max_map50_info(dataframe):
    max_map50_index = dataframe['metrics/mAP50(B)'].idxmax()
    max_map50_info = dataframe.loc[max_map50_index].copy()  # Kopyalama yaparak hata çözülüyor
    return max_map50_info

# Klasördeki tüm CSV dosyalarını okuyarak en iyi mAP50 değerini toplayan fonksiyon
def collect_best_map50_data(directory_path):
    results = []

    # Klasördeki tüm CSV dosyalarını oku
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            
            # CSV dosyasını oku
            df = pd.read_csv(file_path)
            
            # En yüksek mAP50 değerini al
            max_map50_info = get_max_map50_info(df)
            max_map50_info['algorithm'] = filename  # Dosya adını algoritma ismi olarak kaydet
            
            # Sonuçları listeye ekle
            results.append(max_map50_info)
    
    # Tüm sonuçları tek bir DataFrame olarak birleştir
    combined_df = pd.DataFrame(results)

    # mAP50(B) değerlerine göre küçükten büyüğe sıralama
    combined_df.sort_values(by='metrics/mAP50(B)', inplace=True)

    return combined_df

# CSV dosyalarının bulunduğu klasör yolu
directory_path = './Training/CSV'
best_map50_df = collect_best_map50_data(directory_path)

# Sonuçları göster
print(best_map50_df)

# Grafik 4: mAP50 ve Precision Değerleri
plt.figure(figsize=(12, 6))
plt.plot(best_map50_df['algorithm'], best_map50_df['metrics/mAP50(B)'], marker='o', linestyle='-', color='blue', label='mAP50')
plt.plot(best_map50_df['algorithm'], best_map50_df['metrics/precision(B)'], marker='x', linestyle='--', color='green', label='Precision')
plt.xticks(rotation=90)
plt.ylim(0.9, 1.0)
plt.xlabel('Algorithm')
plt.ylabel('Score (mAP50 and Precision)')
plt.title('Algorithm vs mAP50 and Precision (Scores > 0.9)')
plt.legend()
plt.tight_layout()

# Grafiği kaydet
plt.savefig(os.path.join(save_directory, "map50_precision.png"))

# Grafik kaydedildikten sonra ekrana gösterme (isteğe bağlı)
plt.show()
