"""charuco_tespit_ayarlari_arayuzu -> config.py kalici kayit (DETECT_TUNING_* bloklari)."""

import os
import re

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")

_MARKER_BEGIN = {
    "stereo": "# --- begin DETECT_TUNING_STEREO (charuco_tespit_ayarlari_arayuzu) ---",
    "third_cam": "# --- begin DETECT_TUNING_THIRD_CAM (charuco_tespit_ayarlari_arayuzu) ---",
}
_MARKER_END = {
    "stereo": "# --- end DETECT_TUNING_STEREO ---",
    "third_cam": "# --- end DETECT_TUNING_THIRD_CAM ---",
}
_VAR_NAME = {
    "stereo": "DETECT_TUNING_STEREO",
    "third_cam": "DETECT_TUNING_THIRD_CAM",
}


def _format_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    return repr(v)


def _format_tuning_block(var_name, tuning):
    lines = [f"{var_name} = {{"]
    for key in sorted(tuning.keys()):
        lines.append(f"    '{key}': {_format_value(tuning[key])},")
    lines.append("}")
    return "\n".join(lines)


def persist_detect_tuning(profile, tuning):
    """
    profile: 'stereo' | 'third_cam'
    tuning: dict (charuco_tespit_ayarlari_arayuzu'nden gelen ayarlar)
    """
    if profile not in _MARKER_BEGIN:
        raise ValueError(f"Gecersiz profil: {profile}")

    begin = _MARKER_BEGIN[profile]
    end = _MARKER_END[profile]
    var_name = _VAR_NAME[profile]

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    if begin not in text or end not in text:
        raise RuntimeError(f"config.py icinde {profile} isaretleyicileri bulunamadi")

    new_block = (
        f"{begin}\n"
        f"{_format_tuning_block(var_name, tuning)}\n"
        f"{end}"
    )

    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end),
        re.DOTALL,
    )
    new_text, n = pattern.subn(new_block, text, count=1)
    if n != 1:
        raise RuntimeError(f"{profile} blogu guncellenemedi")

    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)
