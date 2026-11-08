#!/usr/bin/env python3
"""
Satranç Tahtası Otomatik Görüntü Yakalama (Stereo)
====================================================
İki kameradan canlı görüntü alır, satranç tahtası tespiti yapar.
Her iki kamerada da board stabil (hareketsiz + net) ve tespit
başarılı olduğunda otomatik olarak fotoğraf çeker ve önizler.

Not: Chessboard tespiti ya tamamdır ya değildir (kısmi tespit yok).
Bu nedenle her iki kamerada board var/yok kontrolü yeterlidir.

Stabilite Kontrolü (Hibrit):
    1. Frame Differencing  → Hareket algılama (piksel farkı ortalaması)
    2. Laplacian Variance  → Netlik ölçümü (blur algılama)

Tuşlar:
    S  → Önizlemedeki fotoğrafı kaydet ve yakalamaya dön
    D  → Önizlemeyi iptal et, yakalamaya dön
    Q  → Çıkış

Kullanım:
    python 11_capture_chessboard_auto.py

Not: Kamera parlaklik/pozlama ayarlari open_cameras → camera_settings.json
(veya varsayilanlar) ile uygulanir. Once 02_kamera_temel_ayarlar_arayuzu.py
calistirmaniz onerilir.
"""

import cv2
import numpy as np
import os
import winsound

from config import (
    LEFT_CAM_INDEX, RIGHT_CAM_INDEX, FLIP_LEFT_CAM, FLIP_RIGHT_CAM,
    LEFT_IMAGE_DIR_CHESS, RIGHT_IMAGE_DIR_CHESS, CALIB_CRITERIA
)
from camera_utils import open_cameras

# ── Satranç Tahtası Ayarları ──────────────────────────────────────────────────
CHESS_ROWS   = 9       # İç köşe sayısı (yatay)
CHESS_COLS   = 6       # İç köşe sayısı (dikey)
PATTERN_SIZE = (CHESS_ROWS, CHESS_COLS)

WINDOW_NAME  = "Chessboard Auto Capture"
DISPLAY_W    = 768          # Tek kamera gösterim genişliği (px)
DISPLAY_H    = 432          # Tek kamera gösterim yüksekliği (px)

# ── Stabilite / Blur Parametreleri ────────────────────────────────────────────
# Bu değerler ortam ışığı ve kamera gürültüsüne göre fine-tune edilebilir.
AUTO_SAVE               = True # True ise önizleme sormadan otomatik kaydeder
MOTION_THRESHOLD        = 3.0   # Frame farkı ort. bu altındaysa hareket yok (0–255)
SHARPNESS_THRESHOLD     = 80.0  # Laplacian varyansı bu üstündeyse görüntü net
STABLE_FRAMES_REQUIRED  = 10    # Art arda bu kadar frame stabil olmalı (~0.33s @30fps)


# ── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────────

def compute_motion(gray_curr, gray_prev):
    """
    İki gri görüntü arasındaki ortalama mutlak piksel farkını döndürür.
    Düşük değer → hareket yok.
    """
    if gray_prev is None:
        return 999.0
    diff = cv2.absdiff(gray_curr, gray_prev)
    return float(np.mean(diff))


def compute_sharpness(gray):
    """
    Görüntünün Laplacian varyansını döndürür.
    Yüksek değer → net görüntü; düşük değer → blur.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def detect_chessboard(gray):
    """Satranç tahtası köşelerini tespit eder. Bulursa (True, corners) döner."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CALIB_CRITERIA)
    return ret, corners


def draw_status_bar(canvas, text, color=(0, 200, 0)):
    """Ekranın altına durum çubuğu çizer."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.putText(canvas, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def beep_capture():
    """Yakalama anında kısa bir bip sesi çıkarır (sadece Windows)."""
    try:
        winsound.Beep(1000, 120)
    except Exception:
        pass


# ── Ana Fonksiyon ──────────────────────────────────────────────────────────────

def main():
    capl, capr = open_cameras(LEFT_CAM_INDEX, RIGHT_CAM_INDEX)
    if not (capl.isOpened() and capr.isOpened()):
        print("HATA: Kameralar acilamadi!")
        return

    os.makedirs(LEFT_IMAGE_DIR_CHESS,  exist_ok=True)
    os.makedirs(RIGHT_IMAGE_DIR_CHESS, exist_ok=True)

    existing    = [f for f in os.listdir(LEFT_IMAGE_DIR_CHESS) if f.endswith('.png')]
    saved_count = len(existing)

    from config import FRAME_WIDTH, FRAME_HEIGHT
    print("=" * 60)
    print("  Chessboard Auto Capture")
    print(f"  Cozunurluk hedefi: {FRAME_WIDTH}x{FRAME_HEIGHT} (camera_settings.json uygulanir)")
    print(f"  Pattern: {CHESS_ROWS}x{CHESS_COLS} ic kose")
    print("  Board'u sabit tutun → otomatik yakalama")
    print("  S=Kaydet  D=Iptal  Q=Cikis")
    print(f"  Stabilite: motion<{MOTION_THRESHOLD}  netlik>{SHARPNESS_THRESHOLD}  {STABLE_FRAMES_REQUIRED} frame")
    print("=" * 60)

    preview_mode  = False
    preview_l     = None
    preview_r     = None

    prev_gray_l   = None   # Önceki frame (sol) – motion için
    prev_gray_r   = None   # Önceki frame (sag) – motion için
    stable_count  = 0      # Art arda stabil frame sayısı

    while True:

        # ── CANLI MOD ──────────────────────────────────────────────────────────
        if not preview_mode:
            ret_l = capl.grab()
            ret_r = capr.grab()
            if not (ret_l and ret_r):
                continue

            _, frame_l = capl.retrieve()
            _, frame_r = capr.retrieve()

            if FLIP_LEFT_CAM:
                frame_l = cv2.flip(frame_l, -1)
            if FLIP_RIGHT_CAM:
                frame_r = cv2.flip(frame_r, -1)

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            # ── Stabilite Kontrolü ─────────────────────────────────────────
            motion_l   = compute_motion(gray_l, prev_gray_l)
            motion_r   = compute_motion(gray_r, prev_gray_r)
            motion_max = max(motion_l, motion_r)

            sharp_l   = compute_sharpness(gray_l)
            sharp_r   = compute_sharpness(gray_r)
            sharp_min = min(sharp_l, sharp_r)

            motion_ok    = motion_max < MOTION_THRESHOLD
            sharpness_ok = sharp_min > SHARPNESS_THRESHOLD

            if motion_ok and sharpness_ok:
                stable_count += 1
            else:
                stable_count = 0

            prev_gray_l = gray_l.copy()
            prev_gray_r = gray_r.copy()

            # ── Board Tespiti ──────────────────────────────────────────────
            # Not: Chessboard ya tüm köşeleri bulur ya da hiç bulamaz.
            # Bu nedenle her iki kamerada tespit var/yok kontrolü yeterli.
            det_l, corners_l = detect_chessboard(gray_l)
            det_r, corners_r = detect_chessboard(gray_r)

            both_detected = det_l and det_r

            # Gösterim kopyaları
            show_l = frame_l.copy()
            show_r = frame_r.copy()
            if det_l:
                cv2.drawChessboardCorners(show_l, PATTERN_SIZE, corners_l, det_l)
            if det_r:
                cv2.drawChessboardCorners(show_r, PATTERN_SIZE, corners_r, det_r)

            canvas = np.hstack([
                cv2.resize(show_l, (DISPLAY_W, DISPLAY_H)),
                cv2.resize(show_r, (DISPLAY_W, DISPLAY_H))
            ])

            # ── Otomatik Yakalama Tetikle ──────────────────────────────────
            all_ok = both_detected and stable_count >= STABLE_FRAMES_REQUIRED

            if all_ok:
                beep_capture()
                if AUTO_SAVE:
                    fname_l = os.path.join(LEFT_IMAGE_DIR_CHESS,  f"chessboard-L{saved_count}.png")
                    fname_r = os.path.join(RIGHT_IMAGE_DIR_CHESS, f"chessboard-R{saved_count}.png")
                    cv2.imwrite(fname_l, frame_l)
                    cv2.imwrite(fname_r, frame_r)
                    saved_count += 1
                    print(f"\n[AUTO-SAVE] #{saved_count}: {fname_l}")
                    stable_count = 0
                    cv2.waitKey(500) # Yeni goruntuye gecmeden once ufak bekleme
                    continue
                else:
                    preview_l    = frame_l.copy()
                    preview_r    = frame_r.copy()
                    preview_mode = True
                    stable_count = 0
                    print(f"\n[AUTO] Otomatik yakalama! (S=Kaydet, D=Iptal)")
                    print(f"       motion={motion_max:.2f}  netlik={sharp_min:.1f}")
                    continue

            # ── Durum Çubuğu ──────────────────────────────────────────────
            ready_to_fire = motion_ok and sharpness_ok and both_detected

            board_status = "SOL:OK SAG:OK" if both_detected else \
                           f"SOL:{'OK' if det_l else '---'} SAG:{'OK' if det_r else '---'}"

            status = (
                f"MOT:{motion_max:.1f}({'OK' if motion_ok else '!'}) "
                f"NET:{sharp_min:.0f}({'OK' if sharpness_ok else 'blur'}) "
                f"STBL:{stable_count}/{STABLE_FRAMES_REQUIRED}  "
                f"{board_status}  Kayit:{saved_count}"
            )

            bar_color = (0, 220, 0) if ready_to_fire else (0, 140, 255)
            draw_status_bar(canvas, status, bar_color)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

        # ── ÖNİZLEME MODU ──────────────────────────────────────────────────────
        else:
            prev_l = preview_l.copy()
            prev_r = preview_r.copy()

            g_l = cv2.cvtColor(preview_l, cv2.COLOR_BGR2GRAY)
            g_r = cv2.cvtColor(preview_r, cv2.COLOR_BGR2GRAY)
            d_l, c_l = detect_chessboard(g_l)
            d_r, c_r = detect_chessboard(g_r)
            if d_l:
                cv2.drawChessboardCorners(prev_l, PATTERN_SIZE, c_l, d_l)
            if d_r:
                cv2.drawChessboardCorners(prev_r, PATTERN_SIZE, c_r, d_r)

            canvas = np.hstack([
                cv2.resize(prev_l, (DISPLAY_W, DISPLAY_H)),
                cv2.resize(prev_r, (DISPLAY_W, DISPLAY_H))
            ])
            draw_status_bar(
                canvas,
                f"ONIZLEME #{saved_count}  |  [S] Kaydet  [D] Iptal  [Q] Cikis",
                (0, 255, 255)
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                fname_l = os.path.join(LEFT_IMAGE_DIR_CHESS,  f"chessboard-L{saved_count}.png")
                fname_r = os.path.join(RIGHT_IMAGE_DIR_CHESS, f"chessboard-R{saved_count}.png")
                cv2.imwrite(fname_l, preview_l)
                cv2.imwrite(fname_r, preview_r)
                saved_count += 1
                print(f"[KAYIT] #{saved_count}: {fname_l}")
                print(f"         {fname_r}")
                preview_mode = False

            elif key == ord('d') or key == ord('D'):
                print("[IPTAL] Fotograf iptal edildi.")
                preview_mode = False

            elif key == ord('q') or key == ord('Q'):
                break

    capl.release()
    capr.release()
    cv2.destroyAllWindows()
    print(f"\nToplam {saved_count} cift fotograf kaydedildi.")


if __name__ == '__main__':
    main()
