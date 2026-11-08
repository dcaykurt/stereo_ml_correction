"""
Kamera Temel Ayarlar Arayuzu (Stereo)
=====================================
Sol/sag kamera parlaklik, pozlama, kontrast, doygunluk (donanim CAP_PROP).
Canli onizleme + gri histogram; ayarlar camera_settings.json dosyasina kaydedilir.
Capture, tespit ve worker modulleri open_cameras ile bu dosyayi okur.

Kullanim:
    python 02_kamera_temel_ayarlar_arayuzu.py

Kaydedilen ayarlar (camera_settings.json) otomatik foto cekme
(11_ / 13_) ve diger open_cameras kullanan scriptlerde uygulanir.
Dosya yoksa camera_utils varsayilanlari kullanilir.
"""

import json
import sys

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QGroupBox, QGridLayout, QPushButton, QMessageBox,
)

from config import (
    LEFT_CAM_INDEX, RIGHT_CAM_INDEX,
    FLIP_LEFT_CAM, FLIP_RIGHT_CAM, FRAME_WIDTH, FRAME_HEIGHT,
)
from camera_utils import (
    SETTINGS_FILE,
    configure_capture_resolution,
    load_camera_settings,
    lock_manual_exposure,
    warmup_cameras,
)

# 640x480 (4:3) onizleme
_PREVIEW_W = 480
_PREVIEW_H = 360
_HIST_W = 480
_HIST_H = 140
_HIST_EVERY_N = 3  # her N karede bir histogram (CPU)
_DEFAULTS = {"brightness": 112, "exposure": -5, "contrast": 32, "saturation": 32}


def _profile_from_settings(settings, profile):
    settings = settings or {}
    return {**_DEFAULTS, **settings.get(profile, {})}


def _make_luma_histogram(bgr, width=_HIST_W, height=_HIST_H):
    """Gri histogram + ortalama / clip uyarisi."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)

    if bgr is None or bgr.size == 0:
        return canvas

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    peak = float(hist.max()) if hist.max() > 0 else 1.0
    mean_v = float(np.mean(gray))
    dark_pct = float(np.mean(gray < 8) * 100.0)
    bright_pct = float(np.mean(gray > 247) * 100.0)

    plot_h = height - 28
    for i in range(256):
        x0 = int(i * width / 256)
        x1 = int((i + 1) * width / 256)
        bar_h = int((hist[i] / peak) * (plot_h - 4))
        if bar_h <= 0:
            continue
        color = (90, 200, 120)
        if i < 8:
            color = (80, 80, 220)
        elif i > 247:
            color = (60, 180, 255)
        cv2.rectangle(canvas, (x0, plot_h - bar_h), (max(x0 + 1, x1 - 1), plot_h), color, -1)

    # ortalama cizgisi
    mx = int(mean_v * width / 256.0)
    cv2.line(canvas, (mx, 0), (mx, plot_h), (255, 220, 80), 1)

    warn = ""
    if dark_pct > 8:
        warn += f" karanlik%{dark_pct:.0f}"
    if bright_pct > 8:
        warn += f" clip%{bright_pct:.0f}"
    label = f"mean={mean_v:.0f}{warn}"
    cv2.putText(
        canvas, label, (6, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return canvas


class KameraTemelAyarlarWorker(QThread):
    # RGB onizleme + BGR histogram goruntuleri
    frame_ready = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        settings = load_camera_settings()

        left = _profile_from_settings(settings, "left")
        right = _profile_from_settings(settings, "right")

        self.l_bright = left["brightness"]
        self.l_exp = left["exposure"]
        self.l_cont = left["contrast"]
        self.l_sat = left["saturation"]

        self.r_bright = right["brightness"]
        self.r_exp = right["exposure"]
        self.r_cont = right["contrast"]
        self.r_sat = right["saturation"]

        self.params_changed = True
        self._frame_i = 0
        self._hist_l = _make_luma_histogram(None)
        self._hist_r = _make_luma_histogram(None)

    def _apply_stereo(self, capl, capr):
        lock_manual_exposure(capl)
        lock_manual_exposure(capr)

        capl.set(cv2.CAP_PROP_BRIGHTNESS, self.l_bright)
        capl.set(cv2.CAP_PROP_EXPOSURE, self.l_exp)
        capl.set(cv2.CAP_PROP_CONTRAST, self.l_cont)
        capl.set(cv2.CAP_PROP_SATURATION, self.l_sat)

        capr.set(cv2.CAP_PROP_BRIGHTNESS, self.r_bright)
        capr.set(cv2.CAP_PROP_EXPOSURE, self.r_exp)
        capr.set(cv2.CAP_PROP_CONTRAST, self.r_cont)
        capr.set(cv2.CAP_PROP_SATURATION, self.r_sat)

    def run(self):
        capl = cv2.VideoCapture(LEFT_CAM_INDEX, cv2.CAP_DSHOW)
        capr = cv2.VideoCapture(RIGHT_CAM_INDEX, cv2.CAP_DSHOW)

        for cap, label in (
            (capl, f"sol (index {LEFT_CAM_INDEX})"),
            (capr, f"sag (index {RIGHT_CAM_INDEX})"),
        ):
            expected, actual = configure_capture_resolution(cap, FRAME_WIDTH, FRAME_HEIGHT)
            if expected != actual:
                print(
                    f"[{label}] Cozunurluk uyusmuyor: istenen {expected[0]}x{expected[1]}, "
                    f"gercek {actual[0]}x{actual[1]}"
                )

        if not (capl.isOpened() and capr.isOpened()):
            blank = np.zeros((_PREVIEW_H, _PREVIEW_W, 3), dtype=np.uint8)
            hist = _make_luma_histogram(None)
            self.frame_ready.emit(blank, blank, hist, hist)
            return

        self._apply_stereo(capl, capr)
        warmup_cameras(capl, capr)
        self.params_changed = False

        self.running = True
        while self.running:
            if self.params_changed:
                if capl.isOpened() and capr.isOpened():
                    self._apply_stereo(capl, capr)
                self.params_changed = False

            if not (capl.isOpened() and capr.isOpened()):
                continue

            r1, r2 = capl.grab(), capr.grab()
            if not (r1 and r2):
                continue

            _, fl = capl.retrieve()
            _, fr = capr.retrieve()
            if fl is None or fr is None:
                continue

            if FLIP_LEFT_CAM:
                fl = cv2.flip(fl, -1)
            if FLIP_RIGHT_CAM:
                fr = cv2.flip(fr, -1)

            self._frame_i += 1
            if self._frame_i % _HIST_EVERY_N == 0:
                self._hist_l = _make_luma_histogram(fl)
                self._hist_r = _make_luma_histogram(fr)

            show_l = cv2.cvtColor(fl, cv2.COLOR_BGR2RGB)
            show_r = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(show_l, show_r, self._hist_l, self._hist_r)

        capl.release()
        capr.release()

    def stop(self):
        self.running = False
        self.wait()


class KameraTemelAyarlarArayuzu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Kamera Temel Ayarlar (Stereo) — {FRAME_WIDTH}x{FRAME_HEIGHT}")
        self.setMinimumWidth(1020)

        main = QWidget()
        self.setCentralWidget(main)
        main_layout = QVBoxLayout(main)

        cam_panel = QHBoxLayout()

        left_col = QVBoxLayout()
        self.lbl_l = QLabel()
        self.lbl_l.setFixedSize(_PREVIEW_W, _PREVIEW_H)
        self.lbl_l.setStyleSheet("background-color: black;")
        self.lbl_hist_l = QLabel()
        self.lbl_hist_l.setFixedSize(_HIST_W, _HIST_H)
        self.lbl_hist_l.setStyleSheet("background-color: #1c1c1c;")
        left_col.addWidget(QLabel(f"<b>Sol Kamera (index {LEFT_CAM_INDEX})</b>"))
        left_col.addWidget(self.lbl_l)
        left_col.addWidget(QLabel("Histogram (gri) — sari cizgi = ortalama"))
        left_col.addWidget(self.lbl_hist_l)
        cam_panel.addLayout(left_col)

        right_col = QVBoxLayout()
        self.lbl_r = QLabel()
        self.lbl_r.setFixedSize(_PREVIEW_W, _PREVIEW_H)
        self.lbl_r.setStyleSheet("background-color: black;")
        self.lbl_hist_r = QLabel()
        self.lbl_hist_r.setFixedSize(_HIST_W, _HIST_H)
        self.lbl_hist_r.setStyleSheet("background-color: #1c1c1c;")
        right_col.addWidget(QLabel(f"<b>Sag Kamera (index {RIGHT_CAM_INDEX})</b>"))
        right_col.addWidget(self.lbl_r)
        right_col.addWidget(QLabel("Histogram (gri) — sari cizgi = ortalama"))
        right_col.addWidget(self.lbl_hist_r)
        cam_panel.addLayout(right_col)

        main_layout.addLayout(cam_panel)

        tip = QLabel(
            "Ipucu: Sol/sag histogram tepe ve mean degerlerini yaklastirin. "
            "Karanlik%/clip% uyarilari asiri karanlik veya yanik bolgeyi gosterir."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #555; margin-bottom: 6px;")
        main_layout.addWidget(tip)

        settings_panel = QHBoxLayout()
        self.worker = KameraTemelAyarlarWorker()

        g_left = QGroupBox("Sol Kamera Ayarlari")
        gl_left = QGridLayout(g_left)
        self.sl_l_bright = self._add_slider(gl_left, 0, "Parlaklik", 0, 255, self.worker.l_bright)
        self.sl_l_exp = self._add_slider(gl_left, 1, "Pozlama", -13, 0, self.worker.l_exp)
        self.sl_l_cont = self._add_slider(gl_left, 2, "Kontrast", 0, 100, self.worker.l_cont)
        self.sl_l_sat = self._add_slider(gl_left, 3, "Doygunluk", 0, 100, self.worker.l_sat)
        settings_panel.addWidget(g_left)

        g_right = QGroupBox("Sag Kamera Ayarlari")
        gl_right = QGridLayout(g_right)
        self.sl_r_bright = self._add_slider(gl_right, 0, "Parlaklik", 0, 255, self.worker.r_bright)
        self.sl_r_exp = self._add_slider(gl_right, 1, "Pozlama", -13, 0, self.worker.r_exp)
        self.sl_r_cont = self._add_slider(gl_right, 2, "Kontrast", 0, 100, self.worker.r_cont)
        self.sl_r_sat = self._add_slider(gl_right, 3, "Doygunluk", 0, 100, self.worker.r_sat)
        settings_panel.addWidget(g_right)

        btn_panel = QVBoxLayout()
        btn_save = QPushButton("Ayarlari Kaydet")
        btn_save.setMinimumHeight(60)
        btn_save.setStyleSheet(
            "font-size: 16px; font-weight: bold; background-color: #28a745; color: white;"
        )
        btn_save.clicked.connect(self.save_settings)
        btn_panel.addStretch()
        btn_panel.addWidget(btn_save)
        btn_panel.addStretch()
        settings_panel.addLayout(btn_panel)

        main_layout.addLayout(settings_panel)

        self.worker.frame_ready.connect(self.update_frames)
        self.worker.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.sync_params)
        self.timer.start(200)

    def _add_slider(self, grid, row, label, vmin, vmax, vdefault):
        grid.addWidget(QLabel(label), row, 0)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(vmin, vmax)
        sl.setValue(int(vdefault))
        lbl = QLabel(str(int(vdefault)))
        sl.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
        grid.addWidget(sl, row, 1)
        grid.addWidget(lbl, row, 2)
        return sl

    def sync_params(self):
        w = self.worker
        new = {
            "l_bright": self.sl_l_bright.value(),
            "l_exp": self.sl_l_exp.value(),
            "l_cont": self.sl_l_cont.value(),
            "l_sat": self.sl_l_sat.value(),
            "r_bright": self.sl_r_bright.value(),
            "r_exp": self.sl_r_exp.value(),
            "r_cont": self.sl_r_cont.value(),
            "r_sat": self.sl_r_sat.value(),
        }
        changed = any(getattr(w, k) != v for k, v in new.items())
        if not changed:
            return
        for k, v in new.items():
            setattr(w, k, v)
        w.params_changed = True

    def cv2qt(self, img, tw, th):
        ih, iw, ch = img.shape
        qimg = QImage(img.data, iw, ih, ch * iw, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.scaled(tw, th, Qt.KeepAspectRatio))

    def update_frames(self, show_l, show_r, hist_l, hist_r):
        self.lbl_l.setPixmap(self.cv2qt(show_l, _PREVIEW_W, _PREVIEW_H))
        self.lbl_r.setPixmap(self.cv2qt(show_r, _PREVIEW_W, _PREVIEW_H))
        # histogram BGR cizildi → RGB
        hl = cv2.cvtColor(hist_l, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hist_r, cv2.COLOR_BGR2RGB)
        self.lbl_hist_l.setPixmap(self.cv2qt(hl, _HIST_W, _HIST_H))
        self.lbl_hist_r.setPixmap(self.cv2qt(hr, _HIST_W, _HIST_H))

    def save_settings(self):
        existing = load_camera_settings() or {}
        data = {
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "left": {
                "brightness": self.worker.l_bright,
                "exposure": self.worker.l_exp,
                "contrast": self.worker.l_cont,
                "saturation": self.worker.l_sat,
            },
            "right": {
                "brightness": self.worker.r_bright,
                "exposure": self.worker.r_exp,
                "contrast": self.worker.r_cont,
                "saturation": self.worker.r_sat,
            },
        }
        # Varsa 3. kamera profilini koru (bu arayuz dokunmaz)
        if "third" in existing:
            data["third"] = existing["third"]

        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(
                self,
                "Basarili",
                f"Sol/sag ayarlar kaydedildi:\n{SETTINGS_FILE}\n\n"
                f"Hedef cozunurluk: {FRAME_WIDTH}x{FRAME_HEIGHT}\n"
                "Capture, tespit ve olcum acilisinda open_cameras bu degerleri yukler.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydedilirken hata olustu:\n{e}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = KameraTemelAyarlarArayuzu()
    w.show()
    sys.exit(app.exec())
