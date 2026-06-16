# Upgrade Bot London/NY dengan Pengaman Institusional

Dokumen ini berisi rencana implementasi untuk menyuntikkan fitur-fitur pengaman dari *Phase 3* (yang sudah kita pasang di bot Asia) ke dalam `london_ny_bot.py`.

## Proposed Changes

---

### 1. Menambahkan HMM Regime Detector
Menyalin skrip pendeteksi volatilitas agar bot London/NY bisa mendeteksi pasar *Choppy* (bolak-balik tanpa arah jelas) dan berhenti *trading* sementara.

#### [NEW] [LondonNewyork/regime_detector.py](file:///d:/Bot%20XAUUSD/scalping/LondonNewyork/regime_detector.py)
Akan di-copy dari `regime_detector.py` yang ada di root, agar bot London/NY tetap rapi dalam foldernedy sendiri.

#### [MODIFY] [LondonNewyork/london_ny_bot.py](file:///d:/Bot%20XAUUSD/scalping/LondonNewyork/london_ny_bot.py)
* **Import**: Mengimport `detect_market_regime`.
* **Execution**: Di dalam *main loop*, sebelum mengevaluasi sinyal XGBoost, bot akan memanggil `detect_market_regime(df_m5_pd)`. Jika mengembalikan `"CHOPPY"`, bot akan men-skip eksekusi order pada siklus tersebut.

---

### 2. Smart Breakeven & Kelly Scaler
Mengamankan profit dan memaksimalkan lot saat AI sedang sangat yakin.

#### [MODIFY] [LondonNewyork/london_ny_bot.py](file:///d:/Bot%20XAUUSD/scalping/LondonNewyork/london_ny_bot.py)
* **Smart Breakeven**: Menambahkan fungsi `terapkan_smart_breakeven()` yang akan digerakkan di setiap iterasi *loop* utama. Jika posisi yang sedang *running* sudah mencapai profit $\ge$ 15 pips (+150 point), Stop Loss akan dipindah ke harga *Entry*.
* **Kelly Position Scaler**: Memodifikasi blok eksekusi `mt5.order_send`. Alih-alih selalu menggunakan `cfg.LOT_SIZE`, bot akan mengecek probabilitas prediksi:
  - Jika probabilitas $> 85\%$, lot dikali 3.
  - Jika probabilitas $> 70\%$, lot dikali 2.
  - Jika tidak, gunakan standar `cfg.LOT_SIZE`.

## Verification Plan
1. Menjalankan skrip *patch* untuk menyuntikkan kode.
2. Memverifikasi tidak ada *syntax error* menggunakan `python -m py_compile`.
3. Mengonfirmasi `LondonNewyork/london_ny_bot.py` dapat memanggil `hmmlearn` tanpa *error* karena *library* sudah diinstal sebelumnya.
