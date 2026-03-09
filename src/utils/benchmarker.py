import numpy as np
import time
import csv
import os
import cv2

class FlowBenchmarker:
    def __init__(self, filename="flow_study_results.csv", sample_interval=0.5, enabled=False):
        """
        :param enabled: Define como False para eventos ao vivo, evitando I/O de disco e uso de CPU.
        """
        self.enabled = enabled
        
        # Se estiver desativado, nem inicializamos as variáveis de arquivo
        if not self.enabled:
            return

        self.filename = filename
        self.sample_interval = sample_interval
        self.last_log_time = 0
        self.headers = ["Timestamp", "Method", "Latency_ms", "Energy", "Sparsity_pct"]
        
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, method_name, flow, latency_ms):
        """
        Smart logging: aborta imediatamente se desativado.
        """
        # Aborta logo de cara, poupando CPU e Disco!
        if not self.enabled:
            return

        now = time.time()
        
        # 1. THROTTLE: Stop if we logged too recently
        if (now - self.last_log_time) < self.sample_interval:
            return

        # 2. CALCULATE: Only do the math if we are going to save it
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        energy = np.mean(mag)
        
        # Sparsity: % of pixels that are moving (magnitude > 1.0)
        sparsity = (np.count_nonzero(mag > 1.0) / mag.size) * 100
        
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(now))

        # 3. WRITE: Append to CSV
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, method_name, f"{latency_ms:.2f}", f"{energy:.4f}", f"{sparsity:.2f}"])
            
            self.last_log_time = now
            
        except PermissionError:
            pass