"""Optional local preview; headless operation never calls GUI functions."""

from __future__ import annotations
import cv2
import numpy as np


class OpenCVDisplay:
    def __init__(self, title: str) -> None:
        self.title = title

    def show(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyAllWindows()


class HeadlessDisplay:
    def show(self, frame: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        pass
