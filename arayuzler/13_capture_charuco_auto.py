#!/usr/bin/env python3
"""
ChArUco Board Otomatik Görüntü Yakalama (Stereo)
=================================================
İki kameradan canlı görüntü alır, ChArUco board tespiti yapar.
Board her iki kamerada da stabil (hareketsiz + net) ve eşit sayıda
köşe tespit edildiğinde otomatik olarak fotoğraf çeker ve önizler.

Stabilite Kontrolü (Hibrit):
    1. Frame Differencing  → Hareket algılama (piksel farkı ortalaması)
    2. Laplacian Variance  → Netlik ölçümü (blur algılama)

Tuşlar:
    S  → Önizlemedeki fotoğrafı kaydet ve yakalamaya dön
    D  → Önizlemeyi iptal et, yakalamaya dön
    Q  → Çıkış

Kullanım:
    python 13_capture_charuco_auto.py

Not: Kamera ayarlari open_cameras → camera_settings.json (veya varsayilanlar).
Once 02_kamera_temel_ayarlar_arayuzu.py onerilir.
"""

import cv2
import numpy as np
import os
import time
import winsound

from config import (
    create_charuco_board, LEFT_CAM_INDEX, RIGHT_CAM_INDEX,
    FLIP_LEFT_CAM, FLIP_RIGHT_CAM, LEFT_IMAGE_DIR, RIGHT_IMAGE_DIR,
    get_detect_tuning, preprocess_gray,
)
from camera_utils import open_cameras

# ── Görüntüleme Ayarları ──────────────────────────────────────────────────────
WINDOW_NAME  = "ChArUco Auto Capture"
MIN_CORNERS  = get_detect_tuning('stereo').get('min_corners', 12)
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
    board, detector, _ = create_charuco_board()

    capl, capr = open_cameras(LEFT_CAM_INDEX, RIGHT_CAM_INDEX)
    if not (capl.isOpened() and capr.isOpened()):
        print("HATA: Kameralar acilamadi!")
        return

    os.makedirs(LEFT_IMAGE_DIR, exist_ok=True)
    os.makedirs(RIGHT_IMAGE_DIR, exist_ok=True)

    # Mevcut dosyalardan sayacı başlat
    existing     = [f for f in os.listdir(LEFT_IMAGE_DIR) if f.endswith('.png')]
    saved_count  = len(existing)

    from config import FRAME_WIDTH, FRAME_HEIGHT
    print("=" * 60)
    print("  ChArUco Auto Capture")
    print(f"  Cozunurluk hedefi: {FRAME_WIDTH}x{FRAME_HEIGHT} (camera_settings.json uygulanir)")
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
            proc_l = preprocess_gray(gray_l, profile='stereo')
            proc_r = preprocess_gray(gray_r, profile='stereo')

            # ── Stabilite Kontrolü ─────────────────────────────────────────
            motion_l    = compute_motion(gray_l, prev_gray_l)
            motion_r    = compute_motion(gray_r, prev_gray_r)
            motion_max  = max(motion_l, motion_r)

            sharp_l = compute_sharpness(gray_l)
            sharp_r = compute_sharpness(gray_r)
            sharp_min = min(sharp_l, sharp_r)

            motion_ok   = motion_max < MOTION_THRESHOLD
            sharpness_ok = sharp_min > SHARPNESS_THRESHOLD

            if motion_ok and sharpness_ok:
                stable_count += 1
            else:
                stable_count = 0

            prev_gray_l = gray_l.copy()
            prev_gray_r = gray_r.copy()

            # ── Board Tespiti ──────────────────────────────────────────────
            corners_l, ids_l, _, _ = detector.detectBoard(proc_l)
            corners_r, ids_r, _, _ = detector.detectBoard(proc_r)

            det_l = corners_l is not None and len(corners_l) >= MIN_CORNERS
            det_r = corners_r is not None and len(corners_r) >= MIN_CORNERS

            # Köşe sayısı eşitlik kontrolü
            n_l = len(corners_l) if det_l else 0
            n_r = len(corners_r) if det_r else 0
            corners_equal = det_l and det_r and (n_l == n_r)

            # Gösterim kopyaları
            show_l = frame_l.copy()
            show_r = frame_r.copy()
            if det_l:
                cv2.aruco.drawDetectedCornersCharuco(show_l, corners_l, ids_l)
            if det_r:
                cv2.aruco.drawDetectedCornersCharuco(show_r, corners_r, ids_r)

            # Boyutlandır ve birleştir
            canvas = np.hstack([
                cv2.resize(show_l, (DISPLAY_W, DISPLAY_H)),
                cv2.resize(show_r, (DISPLAY_W, DISPLAY_H))
            ])

            # ── Otomatik Yakalama Tetikle ──────────────────────────────────
            all_ok = corners_equal and stable_count >= STABLE_FRAMES_REQUIRED

            if all_ok:
                beep_capture()
                if AUTO_SAVE:
                    fname_l = os.path.join(LEFT_IMAGE_DIR,  f"charuco-L{saved_count}.png")
                    fname_r = os.path.join(RIGHT_IMAGE_DIR, f"charuco-R{saved_count}.png")
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
                    print(f"       SOL={n_l} kose  SAG={n_r} kose  "
                          f"motion={motion_max:.2f}  netlik={sharp_min:.1f}")
                    continue

            # ── Durum Çubuğu ──────────────────────────────────────────────
            # Renk: tüm koşullar tamam → yeşil, değil → turuncu
            ready_to_fire = motion_ok and sharpness_ok and corners_equal

            if corners_equal:
                corner_info = f"SOL={n_l} SAG={n_r} ESIT"
            elif det_l and det_r:
                corner_info = f"SOL={n_l} SAG={n_r} FARKLI!"
            else:
                corner_info = f"SOL={'OK' if det_l else '---'}  SAG={'OK' if det_r else '---'}"

            status = (
                f"MOT:{motion_max:.1f}{'<OK' if motion_ok else '>!'} "
                f"NET:{sharp_min:.0f}{'<OK' if sharpness_ok else '>!'} "  # typo intentional: sharpness ok if high
                f"STBL:{stable_count}/{STABLE_FRAMES_REQUIRED}  "
                f"{corner_info}  Kayit:{saved_count}"
            )
            # Fix display logic for sharpness (high = ok)
            status = (
                f"MOT:{motion_max:.1f}({'OK' if motion_ok else '!'}) "
                f"NET:{sharp_min:.0f}({'OK' if sharpness_ok else 'blur'}) "
                f"STBL:{stable_count}/{STABLE_FRAMES_REQUIRED}  "
                f"{corner_info}  Kayit:{saved_count}"
            )

            bar_color = (0, 220, 0) if ready_to_fire else (0, 140, 255)
            draw_status_bar(canvas, status, bar_color)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

        # ── ÖNİZLEME MODU ──────────────────────────────────────────────────────
        else:
            prev_show_l = preview_l.copy()
            prev_show_r = preview_r.copy()

            # Tespiti önizlemede de göster
            g_l = preprocess_gray(cv2.cvtColor(preview_l, cv2.COLOR_BGR2GRAY), profile='stereo')
            g_r = preprocess_gray(cv2.cvtColor(preview_r, cv2.COLOR_BGR2GRAY), profile='stereo')
            c_l, i_l, _, _ = detector.detectBoard(g_l)
            c_r, i_r, _, _ = detector.detectBoard(g_r)
            if c_l is not None:
                cv2.aruco.drawDetectedCornersCharuco(prev_show_l, c_l, i_l)
            if c_r is not None:
                cv2.aruco.drawDetectedCornersCharuco(prev_show_r, c_r, i_r)

            canvas = np.hstack([
                cv2.resize(prev_show_l, (DISPLAY_W, DISPLAY_H)),
                cv2.resize(prev_show_r, (DISPLAY_W, DISPLAY_H))
            ])
            draw_status_bar(
                canvas,
                f"ONIZLEME #{saved_count}  |  [S] Kaydet  [D] Iptal  [Q] Cikis",
                (0, 255, 255)
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                fname_l = os.path.join(LEFT_IMAGE_DIR,  f"charuco-L{saved_count}.png")
                fname_r = os.path.join(RIGHT_IMAGE_DIR, f"charuco-R{saved_count}.png")
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
