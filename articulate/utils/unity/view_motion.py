r"""
    View human motions in real-time using Unity3D. This is the server script.
"""

__all__ = ['MotionViewer']


import time
import numpy as np
import matplotlib
import socket
import cv2


class MotionViewer:
    r"""
    View human motions in real-time / offline using Unity3D.
    """
    colors = matplotlib.colormaps['tab10'].colors
    ip = '127.0.0.1'
    port = 8888

    def __init__(self, n=1, overlap=True, names=None):
        r"""
        :param n: Number of human motions to simultaneously show.
        :param names: List of str. Subject names. No special char like #, !, @, $.
        """
        assert n <= len(self.colors), 'Subjects are more than colors in MotionViewer.'
        assert names is None or n <= len(names), 'Subjects are more than names in MotionViewer.'
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
        print('MotionViewer server start at', (self.ip, self.port), '. Waiting for unity3d to connect.')
        self.conn, addr = self.server_for_unity.accept()
        s = str(self.n) + '#' + \
            ','.join(['%g' % v for v in np.array(self.colors)[:self.n].ravel()]) + '#' + \
            (','.join(self.names) if self.names is not None else '') + '$'
        self.conn.send(s.encode('utf8'))
        assert self.conn.recv(32).decode('utf8') == '1', 'MotionViewer failed to connect to unity.'
        print('MotionViewer connected to', addr)

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

    def update_all(self, poses: list, trans: list, render=True):
        r"""
        Update all subject's motions together.

        :param poses: List of pose tensor/ndarray that can all reshape to [24, 3, 3].
        :param trans: List of tran tensor/ndarray that can all reshape to [3].
        :param render: Render the frame after all subjects have been updated.
        """
        assert len(poses) == len(trans) == self.n, 'Number of motions is not equal to the init value in MotionViewer.'
        for i, (pose, tran) in enumerate(zip(poses, trans)):
            self.update(pose, tran, i, render=False)
        if render:
            self.render()

    def update(self, pose, tran, index=0, render=True):
        r"""
        Update the ith subject's motion using smpl pose and tran.

        :param pose: Tensor or ndarray that can reshape to [24, 3, 3] for smpl pose.
        :param tran: Tensor or ndarray that can reshape to [3] for smpl tran.
        :param index: The index of the subject to update.
        :param render: Render the frame after the subject has been updated.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        pose = np.array(pose).reshape((24, 3, 3))
        tran = np.array(tran).reshape(3) + np.array(self.offsets[index])
        pose = np.stack([cv2.Rodrigues(_)[0] for _ in pose])
        s = str(index) + '#' + \
            ','.join(['%g' % v for v in pose.ravel()]) + '#' + \
            ','.join(['%g' % v for v in tran.ravel()]) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def draw_line(self, start, end, color=(0, 0, 0), width=0.01, render=True):
        r"""
        Draw a line.

        :param start: Tensor or ndarray that can reshape to [3] for the starting point.
        :param end: Tensor or ndarray that can reshape to [3] for the ending point.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param width: Line width.
        :param render: Render the frame after the line has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        start = np.array(start)
        end = np.array(end)
        color = np.array(color)
        s = 'L#' + \
            ','.join(['%g' % v for v in start.ravel()]) + '#' + \
            ','.join(['%g' % v for v in end.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(width) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def clear_line(self, render=True):
        r"""
        Clear all lines.

        :param render: Render the frame after the line has been cleared.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        self.conn.send('l$'.encode('utf8'))
        if render:
            self.render()

    def draw_point(self, position, color=(0, 0, 0), radius=0.2, render=True):
        r"""
        Draw a point.

        :param position: Tensor or ndarray that can reshape to [3] for the point position.
        :param color: Tensor or ndarray that can reshape to [3] for RGB or [4] for RGBA in [0, 1].
        :param radius: Point size.
        :param render: Render the frame after the line has been drawn.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        position = np.array(position)
        color = np.array(color)
        s = 'P#' + \
            ','.join(['%g' % v for v in position.ravel()]) + '#' + \
            ','.join(['%g' % v for v in color.ravel()]) + '#' + \
            str(radius) + '$'
        self.conn.send(s.encode('utf8'))
        if render:
            self.render()

    def clear_point(self, render=True):
        r"""
        Clear all points.

        :param render: Render the frame after the line has been cleared.
        """
        assert self.conn is not None, 'MotionViewer is not connected.'
        self.conn.send('p$'.encode('utf8'))
        if render:
            self.render()

    def render(self):
        r"""
        Render the frame in unity.
        """
        self.conn.send('!$'.encode('utf8'))

    def view_offline(self, poses: list, trans: list, fps=60):
        r"""
        View motion sequences offline.

        :param poses: List of pose tensor/ndarray that can all reshape to [N, 24, 3, 3].
        :param trans: List of tran tensor/ndarray that can all reshape to [N, 3].
        :param fps: Sequence fps.
        """
        is_connected = self.conn is not None
        if not is_connected:
            self.connect()
        for i in range(trans[0].reshape(-1, 3).shape[0]):
            t = time.time()
            self.update_all([r[i] for r in poses], [r[i] for r in trans])
            time.sleep(max(t + 1 / fps - time.time(), 0))
        if not is_connected:
            self.disconnect()
