r"""
    View 0-1 probability in real-time using opencv.
"""


__all__ = ['ProbabilityViewer']


import cv2
import numpy as np


class ProbabilityViewer:
    r"""
    View 0-1 probability in real-time.
    """
    def __init__(self, n, value_range=(0, 1), box_size=(50, 800), name_pairs=None):
        r"""
        :param n: Number of variables.
        :param value_range: Value range (min, max).
        :param box_size: Box size for each variable (h, w).
        :param name_pairs: List of pairs of strings [(b0, e0), (b1, e1), ...] with length n.
        """
        self.n = n
        self.vrange = value_range
        self.box_size = [int(_) for _ in box_size]
        self.im = None
        self.wpadding = int(box_size[1] / 5)
        self.hpadding = int(box_size[0] / 10)
        self.fontsize = box_size[0] / 80
        assert self.wpadding > 4 and self.hpadding > 1, 'Box size is too small.'
        assert name_pairs is None or len(name_pairs) == n, 'Name pairs should have length n.'
        self.empty_im = np.ones((box_size[0] * n, box_size[1], 3), dtype=np.uint8) * 255

        for i in range(n):
            cv2.rectangle(self.empty_im,
                          (self.wpadding - 4, i * self.box_size[0] + self.hpadding),
                          (self.box_size[1] - self.wpadding + 4, (i + 1) * self.box_size[0] - self.hpadding),
                          (222, 222, 222),
                          -1)
            if name_pairs is not None:
                cv2.putText(self.empty_im,
                           name_pairs[i][0],
                           (1, int((i + 0.55) * self.box_size[0] + self.hpadding)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           self.fontsize,
                           (0, 0, 0),
                           1)
                cv2.putText(self.empty_im,
                           name_pairs[i][1],
                           (self.box_size[1] - self.wpadding + 6, int((i + 0.55) * self.box_size[0] + self.hpadding)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           self.fontsize,
                           (0, 0, 0),
                           1)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.im = self.empty_im.copy()
        cv2.namedWindow('Probability Viewer', cv2.WINDOW_AUTOSIZE)
        cv2.startWindowThread()

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        cv2.destroyWindow('Probability Viewer')
        self.im = None

    def update(self, values):
        r"""
        Update the viewer.

        :param values: List of n values.
        """
        if self.im is None:
            print('[Error] ProbabilityViewer is not connected.')
            return
        assert len(values) == self.n, 'Number of values is not equal to the init value in ProbabilityViewer.'
        self.im = self.empty_im.copy()
        for i in range(len(values)):
            v = np.clip((values[i] - self.vrange[0]) / (self.vrange[1] - self.vrange[0]), 0, 1)
            x = int(self.wpadding + (self.box_size[1] - self.wpadding * 2) * v)
            cv2.rectangle(self.im,
                          (x - 4, i * self.box_size[0] + self.hpadding),
                          (x + 4, (i + 1) * self.box_size[0] - self.hpadding),
                          (8, 8, 8),
                          -1)
        cv2.imshow('Probability Viewer', self.im)
        cv2.waitKey(1)


# example
if __name__ == '__main__':
    viewer = ProbabilityViewer(3, name_pairs=[('11111111', '222222'), ('11111111', '222222'), ('11111111', '222222')])
    viewer.connect()
    viewer.update(np.array([1, 0.5, 0.25]))
    input()
