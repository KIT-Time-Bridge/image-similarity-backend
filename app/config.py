# TimBridge_Project/ImageSimilarityServer/config.py
# 시스템 동작을 위한 config

import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTOR_MODEL_NAME = os.getenv("DETECTOR_MODEL_NAME", "retinaface_mnet_v2")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'backbone', 'R100_Glint360K', 'model.onnx')

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))