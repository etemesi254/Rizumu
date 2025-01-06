import sys
import wave
import pyaudio
import numpy as np
import torch
import torchaudio
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QLabel, QHBoxLayout, QComboBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QColor
import queue

from rizumu.pl_model import RizumuLightning


class AudioProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Processor")
        self.setGeometry(100, 100, 800, 500)

        self.is_processing = False
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.buffer = queue.Queue()
        self.recorded_frames = []

        self.rate = 44100
        self.chunk = 4096
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.model = RizumuLightning.load_from_checkpoint("/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=49-step=53350.ckpt")


        self.setup_ui()

    def setup_ui(self):
        self.layout = QVBoxLayout()

        devices_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Device:")
        self.input_combo = QComboBox()
        self.populate_devices(self.input_combo, True)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_combo)
        devices_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Device:")
        self.output_combo = QComboBox()
        self.populate_devices(self.output_combo, False)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_combo)
        devices_layout.addLayout(output_layout)

        self.layout.addLayout(devices_layout)

        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.toggle_processing)
        button_layout.addWidget(self.start_button)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        button_layout.addWidget(self.record_button)

        self.save_button = QPushButton("Save Recording")
        self.save_button.clicked.connect(self.save_recording)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.layout.addLayout(button_layout)

        waveforms_layout = QHBoxLayout()

        input_wave_layout = QVBoxLayout()
        input_wave_layout.addWidget(QLabel("Input:"))
        self.input_waveform = WaveformWidget()
        input_wave_layout.addWidget(self.input_waveform)
        waveforms_layout.addLayout(input_wave_layout)

        output_wave_layout = QVBoxLayout()
        output_wave_layout.addWidget(QLabel("Output:"))
        self.output_waveform = WaveformWidget()
        output_wave_layout.addWidget(self.output_waveform)
        waveforms_layout.addLayout(output_wave_layout)

        self.layout.addLayout(waveforms_layout)
        self.setLayout(self.layout)

    def populate_devices(self, combo, is_input):
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if (is_input and device_info['maxInputChannels'] > 0) or \
                    (not is_input and device_info['maxOutputChannels'] > 0):
                combo.addItem(device_info['name'], i)

    def process_audio(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.input_waveform.update_waveform(audio_data)

        processed_data = self.apply_model(audio_data)

        if self.is_recording:
            self.recorded_frames.append(processed_data.tobytes())

        self.output_waveform.update_waveform(processed_data)
        return (processed_data.tobytes(), pyaudio.paContinue)

    def apply_model(self, audio_data):
        tensor = torch.from_numpy(audio_data).clone()
        tensor = tensor.unsqueeze(0)
        output: torch.Tensor = self.model(tensor).squeeze()

        return output.detach().numpy() * 3.0
        #return audio_data

    def toggle_processing(self):
        if not self.is_processing:
            self.start_processing()
        else:
            self.stop_processing()

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_processing(self):
        self.is_processing = True
        self.start_button.setText("Stop Processing")
        self.record_button.setEnabled(True)
        self.input_combo.setEnabled(False)
        self.output_combo.setEnabled(False)

        input_device = self.input_combo.currentData()
        output_device = self.output_combo.currentData()

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=self.chunk,
            input_device_index=input_device,
            output_device_index=output_device,
            stream_callback=self.process_audio
        )

        self.stream.start_stream()
        self.status_label.setText("Status: Processing")

    def stop_processing(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        self.is_processing = False
        self.start_button.setText("Start Processing")
        self.record_button.setEnabled(False)
        self.input_combo.setEnabled(True)
        self.output_combo.setEnabled(True)
        self.status_label.setText("Status: Stopped")

        if self.is_recording:
            self.stop_recording()

    def start_recording(self):
        self.recorded_frames = []
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.save_button.setEnabled(False)
        self.status_label.setText("Status: Processing and Recording")

    def stop_recording(self):
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self.save_button.setEnabled(True)
        self.status_label.setText("Status: Processing")

    def save_recording(self):
        if not self.recorded_frames:
            return

        filename = "processed_audio.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(4)  # 4 bytes for float32
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.recorded_frames))

        self.status_label.setText(f"Status: Recording saved as {filename}")
        self.save_button.setEnabled(False)


class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(100)
        self.waveform_data = np.zeros(1024)

    def update_waveform(self, audio_data):
        self.waveform_data = audio_data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.setPen(QColor(0, 0, 0))

        width = self.width()
        height = self.height()

        max_sample = max(np.max(np.abs(self.waveform_data)), 1)
        scale = height / (2 * max_sample)

        points_per_pixel = max(len(self.waveform_data) // width, 1)

        for x in range(width - 1):
            start_idx = x * points_per_pixel
            end_idx = start_idx + points_per_pixel

            if start_idx >= len(self.waveform_data):
                break

            chunk = self.waveform_data[start_idx:end_idx]
            if len(chunk) == 0:
                continue

            y = height / 2 - np.mean(chunk) * scale
            next_y = height / 2 - np.mean(self.waveform_data[end_idx:end_idx + points_per_pixel]) * scale

            painter.drawLine(x, int(y), x + 1, int(next_y))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    processor = AudioProcessor()
    processor.show()
    sys.exit(app.exec())