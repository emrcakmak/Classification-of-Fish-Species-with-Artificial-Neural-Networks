# Classification-of-Fish-Species-with-Artificial-Neural-Networks

Bu projede, Kaggle'da yer alan "A Large Scale Fish Dataset" kullanılarak balık türlerinin sınıflandırılması amacıyla bir yapay sinir ağı (ANN) modeli geliştirilmiştir. Çalışma, verisetindeki görüntülerin ön işlenmesi, sınıflandırma modeli oluşturulması ve modelin performans değerlendirmesini kapsamaktadır.

# Veri Ön İşleme

İlk aşamada, verisetinde yer alan balık görüntüleri load_img() fonksiyonu ile yüklenip, img_to_array() fonksiyonu ile numpy dizilerine dönüştürülerek gri tonlamaya (grayscale) indirgenmiştir. Görüntü boyutları, modelin eğitim performansını optimize etmek için 224x224 piksele küçültülmüştür. Görüntülerin tamamı normalize edilerek 0-1 aralığına getirilmiş, böylece daha verimli bir model eğitimi sağlanmıştır.

# Etiketlerin Hazırlanması

Verisetindeki etiketler (balık türleri), modelin anlayabileceği bir formata dönüştürülmüştür. Bunun için LabelEncoder kullanılarak etiketler sayısal değerlere çevrilmiş ve ardından to_categorical() ile kategorik hale getirilmiştir:

# Model Mimarisi
Modelin mimarisi, 224x224 boyutlarındaki görüntülerin işlenmesine uygun şekilde oluşturulmuş olup, tam bağlantılı (dense) katmanlardan oluşan bir yapay sinir ağı (ANN) kullanılmıştır. Modelde dört gizli katman bulunmakta ve her katmanda ReLU aktivasyon fonksiyonu tercih edilmiştir. Modelin çıktısında, 9 farklı balık türünü sınıflandırmak üzere softmax aktivasyon fonksiyonu kullanılmıştır.

Modelin yapısı şu şekildedir:

model = Sequential()
model.add(Flatten(input_shape=(224, 224)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))


Model, adam optimizasyon algoritması ve categorical_crossentropy kayıp fonksiyonu ile derlenmiştir:

# Model Eğitimi
Model, 10 epoch boyunca eğitilmiş ve doğrulama (validation) aşamasında %80 doğruluk (accuracy) elde edilmiştir. Bu sonuç, modelin balık türlerini başarıyla sınıflandırabildiğini göstermektedir.

# Sonuçlar
Modelin eğitim sonrası doğrulama doğruluğu %0.8 civarındadır. Yapılan testler sonucunda, modelin performansı tatmin edici bulunmuş ve daha ileri geliştirme için temel oluşturmuştur.


Projenin Kaggle Linki: https://www.kaggle.com/code/ckmkemr/fish-classification-with-ann




