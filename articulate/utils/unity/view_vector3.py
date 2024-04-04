r"""
    View vector3 in real-time using Unity3D. This is the server script.
"""

__all__ = ['Vector3Viewer']


import time
import numpy as np
import matplotlib
import socket
import cv2


class Vector3Viewer:
    r"""
    View vector3 in real-time / offline using Unity3D.
    """
    colors = matplotlib.colormaps['tab10'].colors
    ip = '127.0.0.1'
    port = 8888

    def __init__(self, n=1, overlap=True, names=None):
        r"""
        :param n: Number of vectors to simultaneously show.
        :param names: List of str for names. No special char like #, !, @, $.
        """
        assert n <= len(self.colors), 'Vectors are more than colors in Vector3Viewer.'
        assert names is None or n <= len(names), 'Vectors are more than names in Vector3Viewer.'
        self.n = n
        self.offsets = [(((n - 1) / 2 - i) * 1.2 if not overlap else 0, 0, 0) for i in range(n)]
        self.names = names
        self.conn = None
        self.server_for_unity = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        r"""
        Connect to the viewer.
        """
        self.server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_for_unity.bind((self.ip, self.port))
        self.server_for_unity.listen(1)
        print('Vector3Viewer server start at', (self.ip, self.port), '. Waiting for unity3d to connect.')
        self.conn, addr = self.server_for_unity.accept()
        s = str(self.n) + '#' + \
            ','.join(['%g' % v for v in np.array(self.colors)[:self.n].ravel()]) + '#' + \
            (','.join(self.names) if self.names is not None else '') + '$'
        self.conn.send(s.encode('utf8'))
        assert self.conn.recv(32).decode('utf8') == '1', 'Vector3Viewer failed to connect to unity.'
        print('Vector3Viewer connected to', addr)

    def disconnect(self):
        r"""
        Disconnect to the viewer.
        """
        if self.conn is not None:
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
        if self.server_for_unity is not None:
            self.server_for_unity.close()
        self.conn = None
        self.server_for_unity = None

    def update_all(self, vectors: list, render=True):
        r"""
        Update all vectors together.

        :param vectors: List of vector3 tensor/ndarray that can all reshape to [3].
        :param render: Render the frame after all vectors have been updated.
        """
        assert len(vectors) == self.n, 'Number of vectors is not equal to the init value in Vector3Viewer.'
        for i, v in enumerate(vectors):
            self.update(v, i, render=False)
        if render:
            self._render()

    def update(self, vector, index=0, render=True):
        r"""
        Update the ith vector.

        :param vector: Tensor or ndarray that can reshape to [3].
        :param index: The index of the vector to update.
        :param render: Render the frame after the vector has been updated.
        """
        assert self.conn is not None, 'Vector3Viewer is not connected.'
        o = np.array(self.offsets[index])
        v = np.array(vector).reshape(3)
        s = str(index) + '#' + \
            ','.join(['%g' % v for v in o.ravel()]) + '#' + \
            ','.join(['%g' % v for v in v.ravel()]) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self._render()

    def _render(self):
        r"""
        Render the frame in unity.
        """
        self.conn.send('!$'.encode('utf8'))

    def view_offline(self, vectors: list, fps=60):
        r"""
        View vector sequences offline.

        :param vectors: List of vector tensor/ndarray that can all reshape to [N, 3].
        :param fps: Sequence fps.
        """
        is_connected = self.conn is not None
        if not is_connected:
            self.connect()
        for i in range(vectors[0].reshape(-1, 3).shape[0]):
            t = time.time()
            self.update_all([r[i] for r in vectors])
            time.sleep(max(t + 1 / fps - time.time(), 0))
        if not is_connected:
            self.disconnect()


if __name__ == '__main__':
    theta = 0
    with Vector3Viewer(3, True, ['a', 'bb', 'ccc']) as viewer:
        while True:
            theta += 0.1
            viewer.update_all([np.array([np.cos(theta), 0 , np.sin(theta)]),
                               np.array([np.cos(theta), np.sin(theta), 0]),
                               np.array([0, np.cos(theta), np.sin(theta)])])
            time.sleep(0.02)
