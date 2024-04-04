r"""
    View 2D field in real-time using opencv.
"""


__all__ = ['Field2DViewer']


import cv2
import numpy as np


class Field2DViewer:
    r"""
    View 2d field in real-time.

    Origin at left-top, x points to right, y points to down.
    """
    def __init__(self, shape, line_scale=1., box_size=40):
        r"""
        :param shape: Maximum field shape (nrows, ncols).
        :param line_scale: Scale of the line length.
        :param box_size: Box size for each entry.
        """
        self.shape = (shape[0], shape[1], 2)
        self.box_size = int(box_size)
        self.line_scale = line_scale * box_size
        self.im = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.im = np.ones((self.box_size * self.shape[0], self.box_size * self.shape[1], 3), dtype=np.uint8) * 255
        cv2.arrowedLine(self.im,
                        (self.box_size // 2, self.box_size // 2),
                        (self.box_size // 2 + self.box_size, self.box_size // 2),
                        (0, 0, 255), max(self.box_size // 25, 1))
        cv2.arrowedLine(self.im,
                        (self.box_size // 2, self.box_size // 2),
                        (self.box_size // 2, self.box_size // 2 + self.box_size),
                        (0, 255, 0), max(self.box_size // 25, 1))
        cv2.namedWindow('Field2D Viewer', cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        cv2.destroyWindow('Field2D Viewer')
        self.im = None

    def update(self, field):
        r"""
        Update the viewer.

        :param field: 2D field in shape [M, N, 2].
        """
        if self.im is None:
            print('[Error] Field2D is not connected.')
            return
        assert len(field.shape) == 3 and field.shape[0] <= self.shape[0] and field.shape[1] <= self.shape[1] and field.shape[2] == 2, \
            'Field is not 2D or larger than the init value in Field2DViewer.'
        im = self.im.copy()
        m = np.array(field)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                cv2.arrowedLine(im,
                              (int((j + 0.5) * self.box_size), int((i + 0.5) * self.box_size)),
                              (int((j + 0.5) * self.box_size) + int(m[i, j, 0] * self.line_scale), int((i + 0.5) * self.box_size) + int(m[i, j, 1] * self.line_scale)),
                              (0, 0, 0),
                              max(self.box_size // 25, 1))
        cv2.imshow('Field2D Viewer', im)
        cv2.waitKey(1)

    def pause(self):
        r"""
        Pause the viewer.
        """
        cv2.waitKey(0)


# example
if __name__ == '__main__':
    viewer = Field2DViewer((3, 6), line_scale=0.5, box_size=100)
    viewer.connect()
    viewer.update(np.array([[[1, 1], [1, 1]],
                           [[1, 1], [1, 2]],
                           [[1, 1], [-0.5, 1]]]))
    viewer.pause()
