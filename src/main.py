import os
import cv2
import time
import numpy as np
from core.optFlow import OpticalFlowEngine
from core.background import BackgroundProcessor
from effects.geometry import ArrowEffect, GridWarpEffect
from effects.fluid import FluidPaintEffect
from effects.debug import ShowMaskEffect
from effects.trails import MotionTrailEffect
from effects.clones import SolidCloneEffect
from effects.timeTunnel import TimeTunnelEffect, DrosteTunnelEffect
from effects.filters import CartoonEffect, HeatmapEffect, NegativeEffect
from utils.hud import HUD
from utils.benchmarker import FlowBenchmarker

class ExhibitionApp:
    def __init__(self):
        # 1. Initialize Camera
        self.cap = cv2.VideoCapture(4)
        ret, frame = self.cap.read()
        if not ret: raise RuntimeError("Could not initialize camera.")
        
        # 2. Engine Setup
        self.flow_engine = OpticalFlowEngine(scale=0.5)
        self.bg_processor = BackgroundProcessor()
        
        # 3. Effects Playlist
        self.effects = [
            #HeatmapEffect(),     # Adds a colorful, thermal-camera vibe
            #CartoonEffect(),     # Adds a comic-book aesthetic
            #NegativeEffect(),    # Classic high-contrast look
            TimeTunnelEffect(max_clones=10, frame_delay=15),
            #SolidCloneEffect(max_clones=4, frame_delay=8),
            #DrosteTunnelEffect(scale_factor=0.94), # Faster recession
            DrosteTunnelEffect(scale_factor=0.98), # Slow, hypnotic recession
            #ShowMaskEffect(),
            FluidPaintEffect(decay=0.985),
            GridWarpEffect(step=40, amplitude=10.0),
            #MotionTrailEffect(trail_length=0.95),
            ArrowEffect(step=30)
        ]
        self.current_idx = 0
        self.effect_duration = 30.0
        self.start_time = time.time()
        
        # 4. Window & HUD Setup
        h, w = frame.shape[:2]
        self.hud = HUD(w, h)
        self.window_name = "Exhibition"
        
        # Use WINDOW_NORMAL to keep window controls (Minimize/Resize/Close)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Resize to a large standard resolution to fill the screen automatically
        # You can change (1280, 720) to (1920, 1080) if you have a Full HD screen
        cv2.resizeWindow(self.window_name, 1280, 720)

        # 5. Optical Flow Setup
        self.flow_methods = ["DIS", "TVL1", "FARNEBACK"]
        self.flow_idx = 0
        self.flow_engine = OpticalFlowEngine(method=self.flow_methods[self.flow_idx])

        # 6. Benchmark
        self.benchmarker = FlowBenchmarker("../data/csv/exhibition_data.csv", sample_interval=0.5)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # Mirror view
            
            # --- Processing ---
            t0 = time.time()
            flow = self.flow_engine.update(frame)
            t1 = time.time()
            
            mask = self.bg_processor.get_mask(frame, flow)

            latency = (t1 - t0) * 1000
            self.benchmarker.log(self.flow_engine.method, flow, latency)
            
            # --- Auto Rotation ---
            elapsed = time.time() - self.start_time
            if elapsed > self.effect_duration:
                self.next_effect()
                elapsed = 0

            # --- Render Effect ---
            current_effect = self.effects[self.current_idx]
            try:
                output = current_effect.apply(frame, flow, mask)
            except Exception as e:
                print(f"Error in {current_effect.name}: {e}")
                output = frame

            # --- HUD & Status Logic ---
            is_flow_mode = self.bg_processor.use_flow_mask      # Determine the mode label
            method_label = "MODE: " + ("Motion (Flow)" if is_flow_mode else "Static")

            status_parts = []
            # Only show the Engine if we are in Flow Mode
            if is_flow_mode:
                status_parts.append(f"ENGINE: {self.flow_engine.method}")

            # Always show the background warning if it's missing
            if not is_flow_mode and self.bg_processor.bg_base is None:
                status_parts.append("No BG captured (Press 'b')")

            full_status = " | ".join(status_parts)
            self.hud.render(
                output, 
                current_effect.name, 
                method_label, 
                remaining_time=(self.effect_duration - elapsed),
                extra_info=full_status
            )

            cv2.imshow(self.window_name, output)
            
            if self.handle_input():
                break

        self.cleanup()

    def next_effect(self):
        self.current_idx = (self.current_idx + 1) % len(self.effects)
        self.effects[self.current_idx].reset()
        self.start_time = time.time()

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]: return True
        elif key == ord('n'): self.next_effect()
        elif key == ord('m'): self.bg_processor.set_mode(not self.bg_processor.use_flow_mask)
        elif key == ord('b'): self.bg_processor.capture_static_model(self.cap)
        elif key == ord('d'): self.hud.toggle()
        elif key == ord('r'): self.effects[self.current_idx].reset()
        if key == ord('o'): # Cycle Optical Flow Engines
            self.flow_idx = (self.flow_idx + 1) % len(self.flow_methods)
            new_method = self.flow_methods[self.flow_idx]
            self.flow_engine.set_method(new_method)
            print(f"Switched Flow Engine to: {new_method}")

        return False
    
    def cleanup(self):
        """
        Safely shuts down the application and resets the camera.
        """
        print("Cleaning up and resetting camera settings...")
        
        # 1. Reset Camera Settings (Linux/Ubuntu v4l2 commands)
        # These commands revert the C922 to its default Auto modes.
        try:
            # 3 = Auto Exposure on many cameras (V4L2_EXPOSURE_AUTO)
            os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=3") 
            # 1 = Enable Auto Focus
            os.system("v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=1")
            print("Camera Auto-features restored.")
        except Exception as e:
            print(f"Note: Could not reset camera via terminal: {e}")

        # 2. Release hardware
        if self.cap.isOpened():
            self.cap.release()
            
        # 3. Close GUI
        cv2.destroyAllWindows()
        print("Exhibition closed successfully.")

# --- THE EXECUTION PART ---
if __name__ == "__main__":
    app = ExhibitionApp()
    try:
        app.run()
    finally:
        # The 'finally' block ensures cleanup runs even if the code crashes
        app.cleanup()