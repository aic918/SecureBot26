"""SecureBot web panel, camera view: the robot's webcam as JPEG frames for the browser.

The camera is opened only while a page is showing the camera view (it asks for a frame a few
times a second) and is let go WATCH_S seconds after the last request, so face capture and the
live face node can have it the rest of the time. It is never opened while one of those holds
it (`busy()` says so, e.g. FaceEnroller busy capturing or securebot_live_node running).
While busy, the view shows the live face node's own newest frame instead when it saves one
(`preview()`, the laps / explore launches' faces:=true: launch_helpers.LIVE_PREVIEW).
"""
import os
import threading
import time

from securebot_ros.launch_helpers import LIVE_PREVIEW

WATCH_S = 3.0          # let the camera go this long after the last frame request
FPS = 8.0
JPEG_QUALITY = 70
WIDTH = 640
RETRY_S = 3.0          # after a failure, report it this long before trying again
PREVIEW_MAX_AGE = 3.0  # s: an older live-node frame means that node is not saving any more


def live_preview(path=LIVE_PREVIEW, max_age=PREVIEW_MAX_AGE, clock=time.time):
    """The live face node's newest frame (JPEG bytes) if it is recent, else None."""
    try:
        if clock() - os.path.getmtime(path) > max_age:
            return None
        with open(path, 'rb') as f:
            return f.read() or None
    except OSError:
        return None


class CameraFeed:
    def __init__(self, index=0, busy=lambda: None, opener=None, clock=time.monotonic,
                 start_thread=None, preview=live_preview):
        self.index, self.busy, self.clock, self.preview = index, busy, clock, preview
        self.opener = opener or self._open_cv2
        self.start_thread = start_thread or (
            lambda f: threading.Thread(target=f, daemon=True).start())
        self.lock = threading.Lock()
        self.jpeg, self.error = None, ''
        self.wanted_until = 0.0
        self.running = False
        self.failed_at = 0.0

    @staticmethod
    def _open_cv2(index):
        import cv2
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def frame(self):
        """The newest JPEG (bytes) or None; starts the camera if it is off.
        RuntimeError (with why) when the camera cannot be used."""
        with self.lock:
            self.wanted_until = self.clock() + WATCH_S
            if not self.running:
                if self.error and self.clock() < self.failed_at + RETRY_S:
                    raise RuntimeError(self.error)           # don't hammer a missing camera
                why = self.busy()
                if why:
                    jpeg = self.preview()
                    if jpeg:
                        return jpeg
                    raise RuntimeError(why)
                self.running, self.error, self.jpeg = True, '', None
                self.start_thread(self._run)
            return self.jpeg

    def _run(self):
        import cv2
        cap = self.opener(self.index)
        if cap is None:
            with self.lock:
                self.running, self.failed_at = False, self.clock()
                self.error = (f'Cannot open the camera (/dev/video{self.index}): is it plugged '
                              'in, or is another program using it?')
            return
        try:
            misses = 0
            while True:
                with self.lock:
                    if self.clock() > self.wanted_until:
                        break
                ok, img = cap.read()
                if not ok or img is None:
                    misses += 1
                    if misses > 20:
                        with self.lock:
                            self.error = 'The camera stopped sending pictures'
                            self.failed_at = self.clock()
                        break
                    time.sleep(0.05)
                    continue
                misses = 0
                if img.shape[1] > WIDTH:
                    img = cv2.resize(img, (WIDTH, img.shape[0] * WIDTH // img.shape[1]))
                ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self.lock:
                        self.jpeg = buf.tobytes()
                time.sleep(1.0 / FPS)
        finally:
            cap.release()
            with self.lock:
                self.running = False
                self.jpeg = None

    def snapshot(self, timeout=4.0, sleep=time.sleep):
        """One JPEG for an alert, waiting up to `timeout` s for the camera to start; None
        when there is no picture to be had."""
        end = self.clock() + timeout
        while True:
            try:
                jpeg = self.frame()
            except RuntimeError:
                jpeg = None
            if jpeg:
                return jpeg
            if self.clock() >= end:
                return None
            sleep(0.1)

    def state(self):
        with self.lock:
            return {'on': self.running, 'error': self.error}
