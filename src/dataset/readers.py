"""Storage-specific frame readers; dataset classes own labels and sampling."""

import os

from PIL import Image, UnidentifiedImageError

from src.dataset.data_transform_cpu import DEPTH_MODALITIES


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class FrameReader:
    def __init__(self):
        self._names = {}

    def names(self, path):
        if path not in self._names:
            try:
                names = sorted(entry.name for entry in os.scandir(path)
                               if entry.is_file(follow_symlinks=False)
                               and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS)
            except OSError as exc:
                raise RuntimeError(f'Failed to list sample directory {path!r}.') from exc
            if not names:
                raise RuntimeError(f'No image frames in {path!r}.')
            self._names[path] = tuple(names)
        return self._names[path]

    def load(self, path, indices, modality):
        names = self.names(path)
        frames = []
        for index in indices:
            index = int(index)
            if not 0 <= index < len(names):
                raise IndexError(f'Frame {index} outside {len(names)} frames in {path!r}.')
            frame_path = os.path.join(path, names[index])
            try:
                with Image.open(frame_path) as image:
                    frames.append(image.copy() if modality in DEPTH_MODALITIES else image.convert('RGB'))
            except (OSError, UnidentifiedImageError) as exc:
                raise RuntimeError(f'Failed to read frame {frame_path!r}.') from exc
        return frames

