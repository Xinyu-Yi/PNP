r"""
    View streaming 1D data in real-time using pygame.
"""


__all__ = ['StreamingDataViewer']


import pygame
import numpy as np
from collections import deque
import matplotlib
from scipy.fftpack import fft as scipy_fft


class StreamingDataViewer:
    r"""
    View 1D streaming data in real-time.
    """
    W = 800
    H = 600
    font_size = 18
    colors = (np.array(matplotlib.colormaps['tab10'].colors) * 255).astype(int)

    def __init__(self, n=1, y_range=(0, 10), window_length=100, names=None, fft=False):
        r"""
        Note: x-range will be [0, window_length] for time domain and [0, fps/2] for frequency domain.
              y-range will be [y_range[0], y_range[1]] for time domain and [0, y_range[1]] for frequency domain.

        :param n: Number of data to simultaneously plot.
        :param y_range: Data range (min, max).
        :param window_length: Number of historical data points simultaneously shown in screen for each plot.
        :param names: List of str. Line names.
        :param fft: Perform fast fourier transform on the window of data and plot the frequency.
        """
        self.n = n
        self.window_length = window_length
        self.y_range = y_range if not fft else (-0.01, y_range[1])
        self.ys = None
        self.screen = None
        self.names = names
        self.dx = self.W / (window_length - 1) * (2 if fft else 1)
        self.dy = self.H / (self.y_range[1] - self.y_range[0])
        self.line_width = max(self.H // 200, 1)
        self.fft = fft

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _fft(self, data):
        r"""
        https://blog.csdn.net/weixin_43589323/article/details/127562996
        """
        return np.abs(scipy_fft(data))[:self.window_length//2] / self.window_length * 2

    def connect(self):
        r"""
        Connect to the viewer.
        """
        pygame.init()
        self.ys = [deque([0.] * self.window_length, maxlen=self.window_length) for _ in range(self.n)]
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption('Streaming Data Viewer (%s Domain): x_length=%d, y_range=(%.1f, %.1f)' %
                                   ('Frequency' if self.fft else 'Time',
                                    self.window_length, self.y_range[0], self.y_range[1]))
        if self.names is not None:
            font = pygame.font.SysFont('arial', self.font_size)
            self.names = [font.render(chr(9644) + ' ' + self.names[i], True, self.colors[i % 10]) for i in range(len(self.names))]

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.screen is not None:
            self.screen = None
            pygame.quit()

    def plot(self, values):
        r"""
        Plot all current values.

        :param values: Iterable in length n.
        """
        if self.screen is None:
            print('[Error] StreamingDataViewer is not connected.')
            return
        assert len(values) == self.n, 'Number of data is not equal to the init value in StreamingDataViewer.'
        self.screen.fill((255, 255, 255))
        for i, v in enumerate(values):
            self.ys[i].append(float(v))
            data = np.asarray(self.ys[i])
            if self.fft:
                data = self._fft(data)
            points = [(j * self.dx, self.H - (v - self.y_range[0]) * self.dy) for j, v in enumerate(data)]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors[i % 10], False, points, width=self.line_width)
        if self.names is not None:
            for i in range(len(self.names)):
                self.screen.blit(self.names[i], (10, i * self.font_size))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.disconnect()


# example
if __name__ == '__main__':
    import time
    f1 = 5
    f2 = 10
    f3 = 29
    with StreamingDataViewer(3, y_range=(-1, 1), window_length=60, fft=True) as viewer:
        for t in range(600):
            viewer.plot([0.5 * np.sin(2 * np.pi * f1 * t / 60),
                         0.8 * np.cos(2 * np.pi * f2 * t / 60),
                         1.0 * np.cos(2 * np.pi * f3 * t / 60)])
            time.sleep(1 / 60)
