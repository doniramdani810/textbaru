# Gunakan base image Python yang ringan
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk optimasi cache
COPY requirements.txt .

# Install semua library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek Anda ke dalam container
# Pastikan server.py, best_model.pth, dan mappings.json berada di folder yang sama
COPY . .

# Ekspos port yang akan digunakan oleh aplikasi
# Hugging Face Spaces secara default menggunakan port 7860 untuk aplikasi Docker
EXPOSE 7860

# Perintah untuk menjalankan aplikasi saat container dimulai
# Menggunakan Gunicorn sebagai production server yang lebih andal
# server:app berarti: jalankan objek 'app' dari file 'server.py'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "server:app"]