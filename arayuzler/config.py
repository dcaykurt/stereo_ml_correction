"""
Stereo Kalibrasyon Paketi — Merkezi Konfigürasyon
=================================================
Bu klasörü zip'leyip başka PC'ye taşıyabilirsiniz.
Tüm yollar bu dosyanın bulunduğu klasöre göredir.

Önemli:
  BOARD_TYPE, kamera indexleri, tahta ölçüleri burada ayarlanır.
  Donanım parlaklık/pozlama: camera_settings.json
  (02_kamera_temel_ayarlar_arayuzu.py ile kaydedilir; yoksa varsayılanlar kullanılır)
"""

import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
#  Board Tipi: "chessboard" veya "charuco"
# ──────────────────────────────────────────────
BOARD_TYPE = "chessboard"

# ──────────────────────────────────────────────
#  ChArUco Board
# ──────────────────────────────────────────────
ARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_1000
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.0374   # metre (37.4 mm)
MARKER_LENGTH = 0.0186   # metre (18.6 mm)

# ──────────────────────────────────────────────
#  Satranç Tahtası (Chessboard)
# ──────────────────────────────────────────────
CHESS_ROWS = 9                 # iç köşe yatay
CHESS_COLS = 6                 # iç köşe dikey
CHESS_SQUARE_SIZE = 0.0268     # metre (26.8 mm)

# ──────────────────────────────────────────────
#  Görüntü klasörleri
# ──────────────────────────────────────────────
LEFT_IMAGE_DIR = os.path.join(BASE_DIR, "left_charuco")
RIGHT_IMAGE_DIR = os.path.join(BASE_DIR, "right_charuco")
LEFT_IMAGE_DIR_CHESS = os.path.join(BASE_DIR, "left_chessboard")
RIGHT_IMAGE_DIR_CHESS = os.path.join(BASE_DIR, "right_chessboard")

# ──────────────────────────────────────────────
#  XML çıktıları
# ──────────────────────────────────────────────
CALIB_OUTPUT_DIR = os.path.join(BASE_DIR, "calibration_output")


def get_output_paths():
    out_dir = os.path.join(CALIB_OUTPUT_DIR, BOARD_TYPE)
    return {
        "left_xml": os.path.join(out_dir, "CameraLeft.xml"),
        "right_xml": os.path.join(out_dir, "CameraRight.xml"),
        "stereo_xml": os.path.join(out_dir, "StereoCamera.xml"),
    }


def get_board_dirs():
    """(left_dir, right_dir, unused_third_placeholder)"""
    if BOARD_TYPE == "charuco":
        return LEFT_IMAGE_DIR, RIGHT_IMAGE_DIR, None
    if BOARD_TYPE == "chessboard":
        return LEFT_IMAGE_DIR_CHESS, RIGHT_IMAGE_DIR_CHESS, None
    raise ValueError(f"Gecersiz BOARD_TYPE: {BOARD_TYPE}")


_paths = get_output_paths()
LEFT_XML = _paths["left_xml"]
RIGHT_XML = _paths["right_xml"]
STEREO_XML = _paths["stereo_xml"]
WORKER_XML_PATH = STEREO_XML

# ──────────────────────────────────────────────
#  Kamera donanımı
# ──────────────────────────────────────────────
LEFT_CAM_INDEX = 0
RIGHT_CAM_INDEX = 1
THIRD_CAM_INDEX = 2  # bu pakette kullanılmaz; 04 arayüzü uyumu için
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

FLIP_LEFT_CAM = False
FLIP_RIGHT_CAM = True

# 04_charuco_tespit arayüzü AprilTag önizlemesi için (stereo kalibrasyon kullanmaz)
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# ──────────────────────────────────────────────
#  StereoSGBM (görüntüleme / sonraki kullanım)
# ──────────────────────────────────────────────
SGBM_BLOCK_SIZE = 11
SGBM_MIN_DISPARITY = -2
SGBM_NUM_DISPARITIES = 16 * 8
SGBM_UNIQUENESS_RATIO = 10
SGBM_SPECKLE_WINDOW = 100
SGBM_SPECKLE_RANGE = 32
SGBM_DISP12_MAX_DIFF = 5

# ──────────────────────────────────────────────
#  Optimizasyon
# ──────────────────────────────────────────────
CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# ──────────────────────────────────────────────
#  ChArUco tespit ayarları (04_... kaydedebilir)
# ──────────────────────────────────────────────
# --- begin DETECT_TUNING_STEREO (charuco_tespit_ayarlari_arayuzu) ---
DETECT_TUNING_STEREO = {
    'blur_enabled': True,
    'blur_k': 1,
    'clahe_clip': 0.5,
    'clahe_enabled': True,
    'clahe_grid': 22,
    'max_perimeter': 6.0,
    'min_corner_dist': 0.08,
    'min_corners': 12,
    'min_perimeter': 0.01,
    'poly_accuracy': 0.12,
    'thresh_constant': 2,
    'thresh_win_max': 135,
    'thresh_win_min': 46,
    'thresh_win_step': 10,
}
# --- end DETECT_TUNING_STEREO ---
# --- begin DETECT_TUNING_THIRD_CAM (charuco_tespit_ayarlari_arayuzu) ---
DETECT_TUNING_THIRD_CAM = {
    'blur_enabled': True,
    'blur_k': 1,
    'clahe_clip': 0.5,
    'clahe_enabled': True,
    'clahe_grid': 22,
    'max_perimeter': 6.0,
    'min_corner_dist': 0.08,
    'min_corners': 1,
    'min_perimeter': 0.01,
    'poly_accuracy': 0.12,
    'thresh_constant': 2,
    'thresh_win_max': 135,
    'thresh_win_min': 46,
    'thresh_win_step': 10,
}
# --- end DETECT_TUNING_THIRD_CAM ---


def get_detect_tuning(profile="stereo"):
    if profile == "third_cam":
        return dict(DETECT_TUNING_THIRD_CAM)
    return dict(DETECT_TUNING_STEREO)


def apply_detector_parameters(params, tuning=None, profile="stereo"):
    if tuning is None:
        tuning = get_detect_tuning(profile)
    tw_min = int(tuning["thresh_win_min"])
    tw_max = max(int(tuning["thresh_win_max"]), tw_min)
    tw_step = max(2, int(tuning["thresh_win_step"]))
    if tw_step > (tw_max - tw_min) and tw_max > tw_min:
        tw_step = max(2, tw_max - tw_min)
    params.adaptiveThreshWinSizeMin = tw_min
    params.adaptiveThreshWinSizeMax = tw_max
    params.adaptiveThreshWinSizeStep = tw_step
    params.adaptiveThreshConstant = float(tuning["thresh_constant"])
    params.minMarkerPerimeterRate = float(tuning["min_perimeter"])
    params.maxMarkerPerimeterRate = float(tuning["max_perimeter"])
    params.polygonalApproxAccuracyRate = float(tuning["poly_accuracy"])
    params.minCornerDistanceRate = float(tuning["min_corner_dist"])
    return params


def preprocess_gray(gray, tuning=None, profile="stereo"):
    if tuning is None:
        tuning = get_detect_tuning(profile)
    if tuning.get("blur_enabled", True):
        k = int(tuning.get("blur_k", 1))
        if k >= 3:
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
    if tuning.get("clahe_enabled", True):
        grid = max(2, int(tuning.get("clahe_grid", 22)))
        clahe = cv2.createCLAHE(
            clipLimit=float(tuning.get("clahe_clip", 0.5)),
            tileGridSize=(grid, grid),
        )
        gray = clahe.apply(gray)
    return gray


def create_charuco_board(profile="stereo"):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    apply_detector_parameters(detector_params, profile=profile)
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, detector, dictionary
