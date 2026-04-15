# Deteksi Kendaraan Menggunakan YOLOv10 dan YOLOv11

## Gambaran Project
Proyek ini bertujuan untuk mendeteksi kendaraan besar (mobil, bus, dan truk) di jalan sebagai potensi bahaya lalu lintas menggunakan metode Object Detection berbasis Deep Learning

Dua model yang diimplementasikan dan dibandingkan:
- YOLOv10  
- YOLOv11  

Tujuan utama adalah menganalisis performa kedua model berdasarkan kualitas kinerja deteksi keseluruhan dan efisiensi komputasi.

## Tujuan
- Mengimplementasikan YOLOv10 dan YOLOv11 untuk deteksi objek 
- Membandingkan performa model menggunakan metrik evaluasi yang relevan 
- Meningkatkan performa model melalui proses fine-tuning  

## Dataset
- Dataset: Traffic Vehicles Object Detection (Kaggle)  
- Kelas yang digunakan: car, bus, truck  
- Total gambar setelah pramosresan: 1784
- Link Akses ke Dataset: https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection

## Cara Menggunakan Dataset
1. Download dataset dari link yang disertakan di atas
2. Lakukan pramosresan sesuai bagian "Pengolahan Data" (filtering dan oversampling)  
3. Letakkan dataset hasil pramosresan ke dalam folder berikut:

project/
│── filtered_dataset/

filtered_dataset/
│── images/
│   ├── train/
│   ├── val/
│   └── test/
│── labels/
│   ├── train/
│   ├── val/
│   └── test/

4. Pastikan file `data.yaml` sudah mengarah ke folder `filtered_dataset`

**Catatan:** 
- Dataset yang digunakan dalam training adalah hasil preprocessing yang disimpan dalam folder `filtered_dataset`
- Dataset yang digunakan pada video demo sama dengan Dataset yang digunakan untuk pelatihan
- Penambahan link akses dan instruksi penggunaan dataset dilakukan untuk meningkatkan reproducibility project
- README telah diperbarui untuk meningkatkan kejelasan dokumentasi 
- Perubahan ini tidak mempengaruhi dataset, model, maupun hasil eksperimen yang ditampilkan pada video demo
- Dataset telah disesuaikan ke format YOLO (format .txt)

## Pengolahan Data
- Memfilter hanya kelas yang relevan  
- Melakukan *oversampling* untuk mengatasi ketidakseimbangan data  
- Pembagian data:
  - Pelatihan: 70%  
  - Validasi: 20%  
  - Pengujian: 10%  

## Model yang Digunakan
- YOLOv10 (Baseline)  
- YOLOv11 (Baseline)  
- YOLOv11 (Fine-Tuned)  

## Konfigurasi Pelatihan
- Epoch: 40 (*baseline*), 50 (*fine-tuned*)  
- Ukuran gambar: 640x640  
- Batch size: 4  
- Perangkat: NVIDIA GTX 1650 Ti  
- Framework: Ultralytics YOLO  
- Random seed: 42  

## Strategi Fine-Tuning

Pada model YOLOv11, dilakukan proses fine-tuning untuk meningkatkan performa deteksi. Penyesuaian yang dilakukan meliputi:

- Penyesuaian learning rate untuk mempercepat konvergensi model
- Penambahan jumlah epoch (dari 40 menjadi 50)  
- Penerapan partial layer freezing untuk mempertahankan fitur dasar dari model pre-trained  

Pendekatan ini bertujuan untuk meningkatkan kemampuan model dalam mengenali objek kendaraan pada dataset yang telah diproses dan meningkatkan generalisasi terhadap distribusi data yang telah difilter

## Hasil Evaluasi

| Model | Precision | Recall | mAP@50 | mAP@50-95 | FPS |
|------|----------|--------|--------|-----------|-----|
| YOLOv10 | 0,7637 | 0,8864 | 0,8894 | 0,5864 | 60,89 |
| YOLOv11 | 0,7593 | 0,9232 | 0,8961 | 0,5903 | 59,43 |
| YOLOv11 (Fine-Tuned) | 0,7479 | 0,9316 | 0,9032 | 0,5933 | 60,83 |

**Catatan Ringkasan Hasil Evaluasi**: 
Model YOLOv11 (Fine-Tuned) memiliki hasil kualitas kinerja deteksi yang terbaik secara keseluruhan (nilai Recall, mAP@50, dan mAP@50-95 tertinggi). Sedangkan, model YOLOv10 memiliki hasil kecepatan inferensi yang tercepat (nilai FPS tertinggi di antara seluruh model). Hal ini menunjukkan adanya trade-off antara kualitas kinerja deteksi dengan kecepatan inferensi secara real-time 

## Contoh Hasil Deteksi

Sampel Gambar berikut menunjukkan hasil deteksi kendaraan menggunakan model YOLOv10, YOLOv11, dan YOLOv11 (fine-tuned dengan kesesuaian hiperparameter learning rate, epoch, serta partial layer freezing untuk meningkatkan performa), 
dimana model berhasil mengidentifikasi objek mobil, bus, dan truk dengan bounding box.

![Hasil Deteksi](results/Sample_Detection.jpg)

## Log Pelatihan

Berikut merupakan hasil pelatihan dari masing-masing model:

### YOLOv10 Baseline
![YOLOv10](results/train10_results.png)

### YOLOv11 Baseline
![YOLOv11](results/train11_results.png)

### YOLOv11 Fine-Tuned
![YOLOv11 FT](results/train11_finetuned_results.png)

Note: Eksperimen dilakukan melalui beberapa skenario, yaitu baseline (YOLOv10 dan YOLOv11) serta fine-tuning pada YOLOv11 dengan penyesuaian hyperparameter seperti learning rate, jumlah epoch, dan layer freezing. Proses pelatihan menghasilkan kurva performa (loss, precision, recall, dan mAP) yang digunakan untuk menganalisis proses konvergensi model.

## Temuan Utama
- YOLOv11 memberikan akurasi deteksi lebih tinggi (Recall dan mAP)  
- YOLOv10 memiliki kecepatan inferensi sedikit lebih tinggi (FPS)  
- Proses fine-tuning meningkatkan performa YOLOv11  

## Bobot Model
Tersedia pada folder `/weights/`:
- YOLOv10_baseline_best.pt   
- YOLOv11_baseline_best.pt  
- YOLOv11_finetuned_best.pt  

## Setup Environment

Project ini telah diuji dengan konfigurasi berikut:
- Python: 3.10  
- OS: Windows  
- GPU: NVIDIA GTX 1650 Ti (opsional)

**Catatan**:
Project ini dikembangkan menggunakan versi library berikut:
- ultralytics 8.4.33
- torch 2.11.0
- torchvision 0.26.0
- numpy 2.2.6
- opencv-python 4.13.0.92
- matplotlib 3.10.8
- PyYAML 6.0.3

## Cara Menjalankan

Ikuti langkah berikut untuk menjalankan project:

1. Clone Repository
```bash
git clone https://github.com/Relll06HUB/Project-1-DL.git
cd Project-1-DL
```

2. Install dependensi:

```bash
pip install -r requirements.txt

```

Alternatif (Opsional):

```bash
pip install ultralytics
```

3. Menjalankan inferensi menggunakan command line:

```bash
yolo task=detect mode=predict model=weights/YOLOv11_finetuned_best.pt source=images/Test1.jpg
```

4. Menjalankan inferensi menggunakan Python:

```python
from ultralytics import YOLO

model = YOLO("weights/YOLOv11_finetuned_best.pt")
results = model("images/Test1.jpg", save=True)
```

**Catatan:**
- Hasil deteksi akan otomatis tersimpan pada folder `runs/detect/`
- File input source dapat diganti dengan gambar lain sesuai kenyamanan dan kebutuhan 
- Untuk jalankan inferensi dengan beberapa gambar sekaligus, bisa gunakan algoritma list seperti yang dilampirkan berikut ini:

```python
results = model(["images/Test1.jpg", "images/Test2.jpg"], save=True) 
```

## License
- Project ini dibuat untuk keperluan akademik dan tidak ditujukan untuk penggunaan komersial
- Dataset mengikuti lisensi dari sumber aslinya (Kaggle)

## Video Demo

Video demonstrasi proses inferensi model dapat diakses melalui link berikut:

https://drive.google.com/file/d/1B6ovYBIXKnmHFt-tLvoJlu0L06dlSvG9/view?usp=sharing

## Struktur Project
project/
│── weights/            # Model hasil pelatihan
│   ├── YOLOv10_baseline_best.pt
│   ├── YOLOv11_baseline_best.pt
│   └── YOLOv11_finetuned_best.pt
│
│── results/		# Contoh Hasil Deteksi & Log Pelatihan
│   ├── Sample_Detection.jpg
│   ├── train10_results.png
│   ├── train11_results.png
│   └── train11_finetuned_results.png
│
│── runs/               # Output Pelatihan (tidak disertakan)
│── Project1.ipynb      # Notebook utama
│── data.yaml		# Konfigurasi dataset
│── requirements.txt    # Konfigurasi library proses percobaan      
│── README.md		# Dokumentasi seluruh project

**Catatan**:
- Folder `results/` berisi contoh hasil deteksi dan log pelatihan
- Folder `runs/` merupakan output otomatis dari proses training dan inferensi
- Jika berhasil, hasil akan tersimpan di folder `runs/detect/` dan dapat dilihat secara visual
