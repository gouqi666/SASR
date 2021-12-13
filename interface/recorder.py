import time

import pyaudio
import wave
from config import audio_config, stop_config
import numpy as np
import librosa
import matplotlib.pyplot as plt


class SilenceDetector:
    def __init__(self):
        self.chunks_threshold = int(np.ceil(audio_config['rate'] / audio_config['frames_per_buffer']
                                            * stop_config['SILENCE_THRESHOLD_SEC']))
        self.silence_count = 0

    @staticmethod
    def checkSilence(audio):
        return np.max(np.abs(audio)) < stop_config["SILENCE_THRESHOLD"]

    def stop(self, audio):
        if self.checkSilence(audio):
            self.silence_count += 1
        else:
            self.silence_count = 0
        return self.silence_count >= self.chunks_threshold


class AudioRecorderWithAutoStop:
    def __init__(self):
        self.s = SilenceDetector()
        self.audio = []

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        self.audio.append(data)
        if self.s.stop(data):
            return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def record(self, return_silence_tail=False):
        p = pyaudio.PyAudio()
        stream = p.open(**audio_config,
                        output=False,
                        input=True,
                        stream_callback=self.callback)
        stream.start_stream()
        while stream.is_active():
            pass
        stream.stop_stream()
        stream.close()
        p.terminate()
        if return_silence_tail:
            return np.concatenate(self.audio)
        else:
            return np.concatenate(self.audio)[:-self.s.chunks_threshold * audio_config['frames_per_buffer']]


if __name__ == '__main__':
    print(AudioRecorderWithAutoStop().record().shape)
