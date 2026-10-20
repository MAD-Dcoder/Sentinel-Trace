import numpy as np
from pipeline.tracker_pipeline import VideoTrackingPipeline

pipeline = VideoTrackingPipeline(camera_id="Cam_01_Doca")

# Frame de teste preto com um rectângulo branco para simular conteúdo
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2_frame, detections_count = pipeline.process_frame(test_frame)

print(f"Pipeline operacional para a câmara: {pipeline.camera_id}")
print(f"Frame processado com sucesso! Deteções encontradas: {detections_count}")