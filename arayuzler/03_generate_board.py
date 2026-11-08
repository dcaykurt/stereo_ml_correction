#!/usr/bin/env python3
"""
ChArUco Board Oluşturucu
=========================
Yazdırılabilir bir ChArUco board görüntüsü oluşturur.

Kullanım:
    python generate_board.py
"""

import cv2
from config import create_charuco_board, SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH


def generate_board_image(output_path='charuco_board.png', size_px=(1400, 1000), margin=20):
    """Yazdırılabilir ChArUco board görüntüsü oluşturur ve kaydeder."""
    board, _, _ = create_charuco_board()
    board_image = board.generateImage(size_px, marginSize=margin)
    cv2.imwrite(output_path, board_image)

    print(f"ChArUco board kaydedildi: {output_path}")
    print(f"  Board: {SQUARES_X}x{SQUARES_Y}, Kare: {SQUARE_LENGTH*1000:.1f}mm, Marker: {MARKER_LENGTH*1000:.1f}mm")
    print("Yazdırdıktan sonra bir kareyi ölçüp config.py'deki SQUARE_LENGTH'i güncelleyin.")

    cv2.imshow('ChArUco Board', board_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    generate_board_image()
