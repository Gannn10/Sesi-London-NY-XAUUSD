# Blueprint Evolusi Bot Scalping XAU/USD 

Dokumen ini berisi rancangan arsitektur untuk meng-upgrade bot scalping harian (`london_ny_bot.py`) dari sistem *Rule-Based* menjadi sistem bertenaga AI kelas institusional, dengan mengadopsi modul-modul dari repositori `xaubot-ai`.

## 1. Ekstraksi Fitur dengan Smart Money Concept (SMC) & Polars
Sebagai pengganti indikator standar (EMA/RSI tunggal), bot akan membaca aliran dana institusi.
* **Logika Inti:** Mendeteksi *Fair Value Gaps* (FVG), *Order Blocks* (OB), dan *Liquidity Sweeps* di timeframe M5. 
* **Teknologi:** Menggunakan `Polars` alih-alih `Pandas`. Komputasi *vectorized* dari Polars sangat ringan dan cepat, sehingga proses *feature engineering* ratusan ribu baris data historis tidak akan menyebabkan *overheating* atau membebani memori laptop saat dijalankan secara lokal.
* **Tujuan:** Menghasilkan kolom data (fitur) baru yang menjadi "makanan" berkualitas tinggi untuk model XGBoost.

## 2. Pelatihan XGBoost dengan Triple Barrier Method
Mengubah cara AI memahami target *profit* dan *loss* khusus untuk gaya *scalping*.
* **Logika Inti:** Model tidak dilatih untuk menebak arah pasti, melainkan menebak mana dari 3 penghalang ini yang akan tersentuh lebih dulu:
    1.  *Barrier Atas:* Take Profit (Sinyal 1)
    2.  *Barrier Bawah:* Stop Loss (Sinyal -1)
    3.  *Barrier Waktu:* Waktu habis/Sesi tutup (Sinyal 0)
* **Tujuan:** Memastikan AI tetap mempertahankan DNA *scalping*-nya (ambil untung cepat) dan tidak menahan posisi terlalu lama hingga berubah menjadi *swing trade*.

## 3. Filter Eksekusi: HMM Regime Detector
Kecerdasan untuk mendeteksi kapan bot harus "diam".
* **Logika Inti:** Menggunakan *Hidden Markov Model* (HMM) untuk membaca volatilitas tersembunyi dan mengklasifikasikan pasar ke dalam rezim tertentu (misal: *Trending Volatile*, *Choppy*, *Sideways*).
* **Tujuan:** Mencegah bot memuntahkan *order* (meskipun sinyal XGBoost menyuruh BUY/SELL) saat kondisi *market* sedang tidak jelas arahnya atau dipenuhi manipulasi bandar.

## 4. Pertahanan Saldo: Smart Breakeven & Kelly Position Scaler
Modul manajemen risiko dinamis untuk melindungi dan melipatgandakan *equity*.
* **Smart Breakeven:** Fitur *exit* dinamis yang langsung memindahkan Stop Loss ke titik *Entry* (BEP) seketika saat posisi sudah *running profit* 10-15 pips.
* **Kelly Scaler:** Algoritma yang secara otomatis menghitung *Lot Size* berdasarkan tingkat persentase probabilitas dari tebakan XGBoost. Lot akan dibesarkan (misal 0.03) jika probabilitas profit > 85%, dan dikecilkan (misal 0.01) jika probabilitas hanya 60%.

---

## Roadmap Implementasi Bertahap

Untuk menghindari kerumitan, proses *coding* akan dibagi menjadi 3 fase:

- [ ] **Fase 1 (Data & Otak):** Mengadopsi skrip `smc_polars.py` untuk mengolah data historis MT5, dilanjutkan dengan pembuatan *script* `triple_barrier.py` untuk memberi label data. Melatih model XGBoost secara *offline*.
- [ ] **Fase 2 (Injeksi Prediksi):** Memodifikasi fungsi pencari sinyal di `london_ny_bot.py` agar mengeksekusi *order* berdasarkan file `xgboost_model.pkl` hasil Fase 1.
- [ ] **Fase 3 (Pengaman Institusional):** Memasukkan modul `regime_detector.py` (HMM) sebagai filter *entry* dan algoritma *Kelly Criterion* untuk menentukan besaran *lot size* secara otomatis.