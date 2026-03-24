import cv2
import mediapipe as mp
import time
import os
import sys

class PoseProcessor:
    def __init__(self):
        self.current_pose = None
        self.pose_landmarker = None
        self._init_model()

    def get_resource_path(self, relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
        return os.path.join(base_dir, relative_path)

    def _init_model(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        def _save_result(result, output_image: mp.Image, timestamp_ms: int):
            # Salva os pontos articulares da primeira pessoa detectada
            if result.pose_landmarks:
                self.current_pose = result.pose_landmarks[0]
            else:
                self.current_pose = None

        model_path = self.get_resource_path('assets/models/pose_landmarker_lite.task')

        if os.path.exists(model_path):
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=_save_result
            )
            self.pose_landmarker = PoseLandmarker.create_from_options(options)
        else:
            print(f"AVISO: Modelo de Pose não encontrado em -> {model_path}")

    def process(self, frame):
        """Envia o frame para processamento e retorna a última pose calculada."""
        if not self.pose_landmarker:
            return self.current_pose

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)

        try:
            self.pose_landmarker.detect_async(mp_image, ts)
        except Exception:
            pass

        return self.current_pose