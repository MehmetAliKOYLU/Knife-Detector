# Knife-Detector

Knife-Detector, YOLOv8 kullanarak görüntülerde bıçakları tespit etmek amacıyla geliştirilmiş bir nesne algılama modelidir. Bu proje, bıçak tespitini hızlı ve etkili bir şekilde gerçekleştirmek için derin öğrenme tekniklerinden yararlanır ve güvenlik sistemleri gibi çeşitli uygulamalarda kullanılabilir.

## Proje Hakkında

Bu proje, güvenlik amacıyla video ve görüntülerde bıçak tespiti yapmak için YOLOv8 nesne tespiti algoritmasını kullanır. Eğitim, çeşitli veri artırma teknikleri ile geliştirilmiş bir veri seti üzerinde gerçekleştirilmiştir. Model, hem eğitim hem de doğrulama sırasında yüksek doğruluk ve performans elde etmeye çalışır.

### Kullanılan Teknolojiler

- **Python**: Programlama dili
- **YOLOv8**: Nesne algılama modeli
- **Ultralytics**: YOLOv8 çerçevesi
- **OpenCV**: Görüntü işleme
- **Pandas**: Veri analizi
- **Matplotlib**: Grafik ve görselleştirme

## Özellikler

- Bıçak tespiti için optimize edilmiş YOLOv8 modeli
- Video ve görüntü işleme desteği
- Gerçek zamanlı algılama (RTSP/Canlı video akışı desteği)
- Gelişmiş veri artırma teknikleri
- Model eğitimi ve değerlendirme için hazır betikler

## Kurulum

Bu projeyi çalıştırmak için aşağıdaki adımları takip edin:

### Gereksinimler

- Python 3.8 veya daha üst sürüm
- YOLOv8 kurulumu (Ultralytics)
- OpenCV
- Diğer bağımlılıklar (requirements.txt dosyasında belirtilmiştir)

### Adımlar

1. **Projenin Klonlanması**

   ```bash
   git clone https://github.com/MehmetAliKOYLU/Knife-Detector.git
   cd Knife-Detector
