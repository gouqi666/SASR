import pyaudio
from collections import UserDict

audio_config = {
    "frames_per_buffer": 1024,
    "format": pyaudio.paFloat32,
    "channels": 1,
    "rate": 16000
}

stop_config = {
    # 静音多少秒停止
    "SILENCE_THRESHOLD_SEC": 1,
    # 多小声算静音
    "SILENCE_THRESHOLD": 0.5
}
