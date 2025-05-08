import cv2 as cv
import os
from datetime import datetime


class ImageLogger:
    def __init__(self):
        self.folder_path = self._create_folder()
        print(f"Logging to {self.folder_path}")

    def append(self, image):
        now = datetime.now()
        filename = now.strftime('%H%M%S_') + f'{now.microsecond // 1000:03d}' + '.jpg'
        path = os.path.join(self.folder_path, filename)
        cv.imwrite(path, image)

    def _create_folder(self):
        now = datetime.now()
        folder_name = now.strftime('%Y%m%d_%H%M%S')
        folder_path = os.path.join('logs', folder_name)
        os.makedirs(os.path.dirname(folder_path))
        return folder_path
