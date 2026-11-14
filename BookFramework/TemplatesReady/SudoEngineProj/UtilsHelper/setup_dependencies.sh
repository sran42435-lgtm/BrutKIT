#!/bin/bash

# =========================================================
# SCRIPT AUTOMASI INSTALASI DEPENDENSI SUDOKU EXTRACTOR
# Target: Ubuntu / Debian / WSL
# Kebutuhan: OpenCV 5, Tesseract OCR, C++17 Compiler, CMake
# =========================================================

set -e # Berhenti jika ada error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Cek apakah dijalankan sebagai root/sudo
if [ "$EUID" -eq 0 ]; then
  log_err "Jangan jalankan script ini dengan 'sudo'. Jalankan sebagai user biasa, sudo akan diminta otomatis."
fi

log_info "Memulai instalasi dependensi..."

# ---------------------------------------------------------
# 1. UPDATE SISTEM & INSTALL TOOLS DASAR
# ---------------------------------------------------------
log_info "1. Memperbarui sistem dan menginstal tools dasar..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    yasm \
    libtbb-dev

# ---------------------------------------------------------
# 2. INSTALL DEPENDENSI OPENCV (IMAGE/VIDEO I/O)
# ---------------------------------------------------------
log_info "2. Menginstal library pendukung gambar dan video untuk OpenCV..."
sudo apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev

# ---------------------------------------------------------
# 3. BUILD & INSTALL OPENCV 5 DARI SOURCE
# ---------------------------------------------------------
OPENCV_VERSION="5.x"
BUILD_DIR="$HOME/opencv_build"

log_info "3. Mengunduh dan membangun OpenCV versi ${OPENCV_VERSION}..."
log_warn "Proses ini memakan waktu cukup lama (15-45 menit tergantung CPU)."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone repository jika belum ada
if [ ! -d "opencv" ]; then
    git clone --branch 5.x --depth 1 https://github.com/opencv/opencv.git
fi

mkdir -p opencv/build
cd opencv/build

# Konfigurasi CMake khusus untuk kebutuhan programmu
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_GTK=ON \
      -D WITH_OPENGL=OFF \
      -D BUILD_opencv_calib3d=ON \
      -D BUILD_opencv_imgproc=ON \
      -D BUILD_opencv_imgcodecs=ON \
      -D BUILD_opencv_core=ON \
      ..

# Compile menggunakan semua core CPU yang tersedia
CORES=$(nproc)
log_info "Menggunakan ${CORES} core CPU untuk compile..."
make -j${CORES}

# Install ke sistem
sudo make install
sudo ldconfig

log_info "OpenCV 5 berhasil diinstal!"

# ---------------------------------------------------------
# 4. INSTALL TESSERACT OCR
# ---------------------------------------------------------
log_info "4. Menginstal Tesseract OCR..."
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev

# Verifikasi Tesseract
if command -v tesseract &> /dev/null; then
    TESS_VER=$(tesseract --version | head -n 1)
    log_info "Tesseract terinstal: ${TESS_VER}"
else
    log_err "Tesseract gagal diinstal."
fi

# ---------------------------------------------------------
# 5. CLEANUP (OPSIONAL)
# ---------------------------------------------------------
log_info "5. Membersihkan sisa cache apt..."
sudo apt-get autoremove -y
sudo apt-get clean

# ---------------------------------------------------------
# SELESAI
# ---------------------------------------------------------
echo ""
echo "========================================================="
log_info "SEMUA DEPENDENSI BERHASIL DIINSTAL!"
echo "========================================================="
echo ""
echo "Langkah selanjutnya untuk menjalankan program:"
echo ""
echo "1. Compile program:"
echo "   g++ extract_img.cpp -std=c++17 -O2 \$(pkg-config --cflags --libs opencv4 || pkg-config --cflags --libs opencv) -o extract_img"
echo ""
echo "   ATAU jika pkg-config tidak mendeteksi opencv5 secara langsung:"
echo "   g++ extract_img.cpp -std=c++17 -O2 -I/usr/local/include/opencv5 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -o extract_img"
echo ""
echo "2. Jalankan program:"
echo "   ./extract_img"
echo ""
echo "========================================================="

# Kembali ke direktori awal
cd ~
