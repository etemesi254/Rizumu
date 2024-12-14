import sys
import wave
import pyaudio
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QColor


class AudioRecorder(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Recorder with Waveform")
        self.setGeometry(100, 100, 600, 400)

        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        print(self.audio.get_device_info_by_index(2))

        # Set up audio stream
        self.stream = None
        self.rate = 44100  # Sample rate
        self.chunk = 1024  # Chunk size
        self.format = pyaudio.paInt16  # Audio format (16-bit)
        self.channels = 2  # Mono audio

        # Create GUI elements
        self.layout = QVBoxLayout()

        self.status_label = QLabel("Status: Ready to record")
        self.layout.addWidget(self.status_label)

        # Create a layout for buttons
        button_layout = QHBoxLayout()

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)

        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("Save Recording")
        self.save_button.clicked.connect(self.save_recording)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.layout.addLayout(button_layout)

        # Create the waveform widget
        self.waveform_widget = WaveformWidget()
        self.layout.addWidget(self.waveform_widget)

        self.setLayout(self.layout)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self.status_label.setText("Status: Recording...")
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)

        # Start the stream
        self.stream = self.audio.open(format=self.format,
                                      channels=1,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk,input_device_index=2)

        # Start the recording in a timer loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.record_audio)
        self.timer.start(100)

    def record_audio(self):
        if self.is_recording:
            data = self.stream.read(self.chunk,exception_on_overflow = False)

            self.frames.append(data)

            # Convert the audio data into an array of samples for waveform rendering
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.waveform_widget.update_waveform(audio_data)

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.status_label.setText("Status: Recording stopped")
        self.record_button.setText("Start Recording")
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)

        self.timer.stop()

    def save_recording(self):
        filename = "recording.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

        self.status_label.setText(f"Status: Recording saved as {filename}")
        self.frames = []


class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(100)
        self.waveform_data = []

    def update_waveform(self, audio_data):
        # Add new audio data to the waveform
        self.waveform_data.extend(audio_data)
        self.update()

    def paintEvent(self, event):
        if len(self.waveform_data) == 0:
            return

        # Create a QPainter object
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Clear the widget
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        # Set up drawing parameters
        painter.setPen(QColor(0, 0, 0))  # Black color for the waveform
        width = self.width()
        height = self.height()

        # Scale the waveform data to fit within the widget
        samples = np.array(self.waveform_data)
        max_sample = np.max(np.abs(samples))
        if max_sample == 0:
            max_sample = 1  # Avoid division by zero

        # Calculate the scaling factor to fit the waveform in the widget
        scale = height / max_sample

        # Draw the waveform
        for i in range(1, len(samples)):
            x1 = (i - 1) * width / len(samples)
            y1 = height / 2 - (samples[i - 1] * scale)
            x2 = i * width / len(samples)
            y2 = height / 2 - (samples[i] * scale)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Keep the waveform data within a reasonable range
        if len(self.waveform_data) > width:
            self.waveform_data = self.waveform_data[int(len(self.waveform_data) / 2):]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    recorder.show()
    sys.exit(app.exec())
