import cv2
import threading

class Camera:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Camera, cls).__new__(cls)
                cls._instance._init_camera()
            return cls._instance

    def _init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
        self.running = True
        
        # Start background thread to continuously read frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Flip for mirror effect
                self.frame = cv2.flip(frame, 1)

    def get_frame(self):
        return self.frame

    def __del__(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
