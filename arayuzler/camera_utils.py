import cv2
import json
import os

from config import FRAME_WIDTH, FRAME_HEIGHT

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_settings.json")


def load_camera_settings():
    """camera_settings.json dosyasından kamera ayarlarını yükler."""
    if not os.path.exists(SETTINGS_FILE):
        return None
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[UYARI] Kamera ayarlari okunamadi: {e}")
        return None


_DEFAULT_CAMERA_PROPS = {
    'brightness': 112,
    'exposure': -5,
    'contrast': 32,
    'saturation': 32,
}


def lock_manual_exposure(cap):
    """
    Auto-exposure'i manuel moda alir.
    Aksi halde slider/JSON exposure hizla ezilir.
    OpenCV/DirectShow'da 0.25 genelde manual, 0.75 auto.
    """
    if cap is None or not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)


def _apply_cap_props(cap, cfg, defaults=None):
    """Tek kameraya JSON profil ayarlarini uygular."""
    if defaults is None:
        defaults = _DEFAULT_CAMERA_PROPS
    if cfg is None:
        cfg = defaults
    lock_manual_exposure(cap)
    for key, prop in (
        ('brightness', cv2.CAP_PROP_BRIGHTNESS),
        ('exposure', cv2.CAP_PROP_EXPOSURE),
        ('contrast', cv2.CAP_PROP_CONTRAST),
        ('saturation', cv2.CAP_PROP_SATURATION),
        ('gain', cv2.CAP_PROP_GAIN),
    ):
        if key in cfg:
            cap.set(prop, cfg[key])
        elif key in defaults:
            cap.set(prop, defaults[key])


def apply_camera_settings(capl, capr):
    """Yüklenen ayarları sol ve sağ kameralara uygular.
    camera_settings.json yoksa veya bozuksa _DEFAULT_CAMERA_PROPS kullanılır.
    """
    settings = load_camera_settings()
    if not settings:
        print(
            f"[kamera] {SETTINGS_FILE} yok/okunamadi → "
            f"varsayilan ayarlar: {_DEFAULT_CAMERA_PROPS}"
        )
        for cap in (capl, capr):
            _apply_cap_props(cap, None)
        return

    if 'left' in settings:
        _apply_cap_props(capl, settings['left'])
    else:
        _apply_cap_props(capl, None)
    if 'right' in settings:
        _apply_cap_props(capr, settings['right'])
    else:
        _apply_cap_props(capr, None)


def apply_third_camera_settings(cap3):
    """Yüklenen ayarları 3. kameraya uygular."""
    settings = load_camera_settings()
    if not settings:
        _apply_cap_props(cap3, None)
        return
    _apply_cap_props(cap3, settings.get('third', {}))


def read_actual_resolution(cap):
    """CAP_PROP sonrası gerçek çözünürlük (bazen istenenle farklıdır)."""
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def configure_capture_resolution(cap, width=None, height=None):
    """
    Çözünürlük isteği gönderir, bir kare okuyup gerçek boyutu döndürür.
    Returns: (istenen_w, istenen_h), (gercek_w, gercek_h)
    """
    width = FRAME_WIDTH if width is None else width
    height = FRAME_HEIGHT if height is None else height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.read()
    return (width, height), read_actual_resolution(cap)


def resolution_warning(label, expected_wh, actual_wh):
    if expected_wh == actual_wh:
        return None
    return (
        f"[{label}] Cozunurluk uyusmuyor: istenen {expected_wh[0]}x{expected_wh[1]}, "
        f"gercek {actual_wh[0]}x{actual_wh[1]}. Kalibrasyon farkli cozunurlukteyse 3B olcek hatali olur."
    )


def read_fs_int(node):
    """FileStorage skaler int (imageWidth vb.) — release oncesi okunmali."""
    if node.empty():
        return None
    return int(node.real())


def scale_intrinsics(mtx, from_wh, to_wh):
    """Kalibrasyon matrisini baska goruntu boyutuna olcekle."""
    if from_wh == to_wh:
        return mtx.copy()
    sx = to_wh[0] / float(from_wh[0])
    sy = to_wh[1] / float(from_wh[1])
    scaled = mtx.copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] *= sx
    scaled[1, 2] *= sy
    return scaled


def load_intrinsics_from_xml(xml_path, frame_wh=None):
    """
    mtx, dist ve XML'deki imageWidth/Height yukler.
    frame_wh verilirse ve XML boyutu farkliysa mtx olceklenir (uyari basilir).
    """
    fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)

    node_mtx = fs.getNode('cameraMatrix')
    if node_mtx.empty():
        node_mtx = fs.getNode('mtx')
    mtx = node_mtx.mat() if not node_mtx.empty() else None

    node_dist = fs.getNode('distCoeffs')
    if node_dist.empty():
        node_dist = fs.getNode('dist')
    dist = node_dist.mat() if not node_dist.empty() else None

    calib_w = calib_h = None
    nw = fs.getNode('imageWidth')
    nh = fs.getNode('imageHeight')
    if not nw.empty() and not nh.empty():
        calib_w, calib_h = read_fs_int(nw), read_fs_int(nh)

    fs.release()

    if mtx is None:
        raise ValueError(f"Kamera matrisi yuklenemedi: {xml_path}")

    calib_wh = (calib_w, calib_h) if calib_w and calib_h else None
    if frame_wh and calib_wh and frame_wh != calib_wh:
        msg = resolution_warning(os.path.basename(xml_path), calib_wh, frame_wh)
        if msg:
            print(msg + " → intrinsics olcekleniyor.")
        mtx = scale_intrinsics(mtx, calib_wh, frame_wh)

    return mtx, dist, calib_wh


def open_third_camera(third_idx, width=None, height=None):
    """3. kamera (AprilTag referans) acar."""
    width = FRAME_WIDTH if width is None else width
    height = FRAME_HEIGHT if height is None else height
    cap3 = cv2.VideoCapture(third_idx, cv2.CAP_DSHOW)
    expected, actual = configure_capture_resolution(cap3, width, height)
    warn = resolution_warning(f"3. kamera (index {third_idx})", expected, actual)
    if warn:
        print(warn)
    apply_third_camera_settings(cap3)
    warmup_cameras(cap3)
    return cap3


def open_cameras(left_idx, right_idx, width=None, height=None):
    """
    Kameralari acar; cozunurluk config.FRAME_WIDTH x FRAME_HEIGHT (varsayilan).
    camera_settings.json uygulayan, AE kilitleyen, warmup yapan ortak giris.
    DirectShow gercek cozunurlugu farkli verebilir — konsola uyarir.
    """
    width = FRAME_WIDTH if width is None else width
    height = FRAME_HEIGHT if height is None else height

    capl = cv2.VideoCapture(left_idx, cv2.CAP_DSHOW)
    capr = cv2.VideoCapture(right_idx, cv2.CAP_DSHOW)

    for cap, label in ((capl, f"sol (index {left_idx})"), (capr, f"sag (index {right_idx})")):
        expected, actual = configure_capture_resolution(cap, width, height)
        warn = resolution_warning(label, expected, actual)
        if warn:
            print(warn)

    apply_camera_settings(capl, capr)
    warmup_cameras(capl, capr)
    print(f"[kamera] Ayarlar uygulandi ({SETTINGS_FILE}), cozunurluk hedefi {width}x{height}")
    return capl, capr


def warmup_cameras(*caps, frames=15):
    """DirectShow icin ilk kareleri atar."""
    for _ in range(frames):
        for cap in caps:
            if cap is not None and cap.isOpened():
                cap.read()
