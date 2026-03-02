import cv2

class HUD:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.active = True
        
        # Standard exhibition styling
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.primary_color = (0, 255, 255)  # Yellow (Titles)
        self.secondary_color = (220, 220, 220) # Off-white (Info)
        self.warning_color = (150, 150, 255) # Light Red/Pink (Alerts)
        
    def toggle(self):
        self.active = not self.active

    def render(self, frame, effect_name, method_name, remaining_time=None, extra_info=""):
        """
        Draws the interface.
        :param extra_info: Optional status message (e.g., 'No BG Captured')
        """
        if not self.active:
            return

        # 1. Top Left: Effect Status
        self._draw_text(frame, f"EFEITO: {effect_name}", (20, 40), 0.8, self.primary_color)
        
        if remaining_time is not None:
            self._draw_text(frame, f"Prox. em: {remaining_time:.1f}s", (20, 70), 0.6, self.secondary_color)
            
        self._draw_text(frame, method_name, (20, 100), 0.5, self.secondary_color)

        # 2. Status / Warning Line (Just below the method name)
        if extra_info:
            self._draw_text(frame, extra_info, (20, 130), 0.6, self.warning_color)

        # 3. Bottom: Controls (Dynamic positioning based on screen height)
        h = frame.shape[0]
        help_text = "n:Prox | m:Modo | b:Fundo | d:HUD | q:Sair"
        self._draw_text(frame, help_text, (20, h - 20), 0.5, self.secondary_color)

    def _draw_text(self, img, text, pos, scale, color):
        """Helper to draw text with a subtle black outline for contrast"""
        # Outline (Black)
        cv2.putText(img, text, pos, self.font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        # Foreground (Color)
        cv2.putText(img, text, pos, self.font, scale, color, 1, cv2.LINE_AA)