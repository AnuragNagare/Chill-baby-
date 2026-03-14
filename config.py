import os
from dotenv import load_dotenv

load_dotenv()

# ── Clarifai food recognition ───────────────────────────────────────────────
FOOD_API_KEY = os.getenv('FOOD_API_KEY', '')
MODEL_ID     = os.getenv('MODEL_ID', '')

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv('OPENAI_API_KEY', '')
OPENAI_API_BASE    = "https://babii.openai.azure.com/"
OPENAI_API_VERSION = "2023-07-01-preview"
OPENAI_ENGINE      = "babii-chat-gpt-4-32"

# ── Foodvisor (defined but not actively used yet) ────────────────────────────
FOODVISOR_API = os.getenv('FOODVISOR_API', '')
FOODVISOR_URL = "https://vision.foodvisor.io/api/1.0/en/analysis"

# ── MongoDB ──────────────────────────────────────────────────────────────────
DB_URL  = os.getenv('DB_URL', 'mongodb://localhost:27017/')
DB_NAME = "mealtimecammy"

# ── Eye / Sleep detection thresholds ─────────────────────────────────────────
EYE_AR_THRESH       = 0.15
EYE_AR_CONSEC_FRAMES = 40

# ── Video processing ─────────────────────────────────────────────────────────
FRAME_RESIZE_WIDTH = 540        # resize all incoming frames to this width
EMOTION_EVERY_N_FRAMES = 30    # run FER every N WebRTC frames
FOOD_CAPTURE_INTERVAL_S = 3    # seconds between canvas food snapshots (frontend)

# ── Recording ────────────────────────────────────────────────────────────────
STATIC_VIDEO_FOLDER = './static/videos/'
FFMPEG_PATH         = 'ffmpeg/ffmpeg'

# ── Phase 2: YOLOv8 Child Detection ──────────────────────────────────────────
YOLO_MODEL_PATH         = 'yolov8n.pt'   # auto-downloaded on first run (~6 MB)
YOLO_DETECT_EVERY_N     = 15             # run detection every N WebRTC frames
YOLO_CONFIDENCE_THRESH  = 0.50           # minimum confidence to count as detected
YOLO_PERSON_CLASS_ID    = 0              # COCO class 0 = person

# ── Phase 3: YAMNet Audio Detection ──────────────────────────────────────────
YAMNET_MODEL_URL        = 'https://tfhub.dev/google/yamnet/1'
YAMNET_SAMPLE_RATE      = 16000          # YAMNet requires 16 kHz mono audio
YAMNET_BUFFER_SECONDS   = 1.0            # accumulate this many seconds before classifying
YAMNET_COUGH_CLASS_ID   = 370            # YAMNet class index for 'Cough'
YAMNET_SNEEZE_CLASS_ID  = 411            # YAMNet class index for 'Sneeze'
YAMNET_CONFIDENCE_THRESH = 0.30          # minimum score to raise an alert

# ── Cough analysis: noise & severity ──────────────────────────────────────────
COUGH_NOISE_GATE_RMS    = 0.01           # skip if RMS below this (too quiet)
COUGH_SNR_MIN_DB       = 3.0             # skip if SNR below this (too noisy)
COUGH_SEVERITY_MILD_RMS = 0.03           # RMS thresholds for severity
COUGH_SEVERITY_MODERATE_RMS = 0.08
COUGH_SEVERITY_SEVERE_RMS   = 0.15
COUGH_BURST_WINDOW_SEC = 3.0             # window for counting paroxysmal coughs
COUGH_BURST_COUNT_SEVERE = 4              # 4+ coughs in window → severe
