import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False, None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, None

    return True, frame


def scan_cameras(max_index=10):
    print("Kamera tarama başlıyor...\n")

    found = []

    for i in range(max_index):
        ok, frame = test_camera(i)

        if ok:
            found.append(i)
            print(f"[OK] Kamera bulundu: index {i}  | frame size: {frame.shape}")

            # küçük preview göstermek istersen:
            cv2.imshow(f"Camera {i}", frame)
        else:
            print(f"[--] index {i}: yok / erişilemedi")

    print("\nTarama tamamlandı.")
    print("Bulunan kameralar:", found)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scan_cameras(max_index=10)