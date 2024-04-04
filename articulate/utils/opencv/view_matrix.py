r"""
    View 2D matrix in real-time using opencv.
"""


__all__ = ['MatrixViewer']


import cv2
import numpy as np


class MatrixViewer:
    r"""
    View 2d matrix in real-time.
    """
    def __init__(self, shape, value_range=(-1, 1), box_size=(40, 40)):
        r"""
        :param shape: Maximum matrix shape (nrows, ncols).
        :param value_range: Value range (min, max).
        :param box_size: Box size for each entry (h, w).
        """
        self.shape = shape
        self.vrange = value_range
        self.box_size = [int(_) for _ in box_size]
        self.c = None
        self.m = None
        self.im = None
        self.fontsize = self.box_size[0] / 128

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.c = -np.ones(self.shape)
        self.m = np.ones(self.shape) * np.nan
        self.im = np.ones((self.box_size[0] * self.shape[0], self.box_size[1] * self.shape[1], 3), dtype=np.uint8) * 255
        cv2.namedWindow('Matrix Viewer', cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        cv2.destroyWindow('Matrix Viewer')
        self.im = None

    def update(self, matrix):
        r"""
        Update the viewer.

        :param matrix: 2D Matrix.
        """
        if self.im is None:
            print('[Error] MatrixViewer is not connected.')
            return
        assert len(matrix.shape) == 2 and matrix.shape[0] <= self.shape[0] and matrix.shape[1] <= self.shape[1], \
            'Matrix is not 2D or larger than the init value in MatrixViewer.'
        m = np.zeros(self.shape) + np.nan
        m[:matrix.shape[0], :matrix.shape[1]] = np.array(matrix)
        c = np.clip(255 * (m - self.vrange[0]) / (self.vrange[1] - self.vrange[0]), 0, 255).round()
        c[np.isnan(c)] = -1
        change = np.logical_or(np.abs(m - self.m) > 5e-3, c != self.c)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if change[i, j]:
                    cv2.rectangle(self.im,
                                  (j * self.box_size[1], i * self.box_size[0]),
                                  ((j + 1) * self.box_size[1], (i + 1) * self.box_size[0]),
                                  (255, 255, 255) if c[i, j] == -1 else (255 - c[i, j], 0, c[i, j]),
                                  -1)
                    if c[i, j] != -1:
                        cv2.putText(self.im,
                                   '%.2f' % m[i, j],
                                   (j * self.box_size[1] + 4, int((i + 0.55) * self.box_size[0])),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   self.fontsize,
                                   (255, 255, 255),
                                   1)
        self.c = c
        self.m = m
        cv2.imshow('Matrix Viewer', self.im)
        cv2.waitKey(1)


# example
if __name__ == '__main__':
    viewer = MatrixViewer((3, 6), box_size=(50, 50))
    viewer.connect()
    viewer.update(np.array([[1, 0.75, 0.5, 0.25, 0],
                           [-1, -0.75, -0.5, -0.25, 0]]))
    input()
