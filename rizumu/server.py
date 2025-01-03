import os
import tempfile
from typing import Tuple

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from rizumu.pl_model import RizumuLightning

# Import your Lightning model
# Assuming your model is in speech_extractor.py
# from speech_extractor import SpeechExtractorModel

app = FastAPI(title="Speech Extractor API")


class SpeechExtractorService:
    def __init__(self, model_path: str):
        # Load your model here
        self.model = RizumuLightning.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @torch.no_grad()
    def extract_speech(self, audio_path: str) -> Tuple[np.ndarray, int]:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)

        # Process through your model
        extracted_speech = self.model(waveform)
        # Replace the above line with your actual model inference

        # For demonstration, just returning the input
        # extracted_speech = waveform

        return (extracted_speech.cpu().numpy(), sample_rate)


# Initialize the service
# Replace with your model checkpoint path
MODEL_PATH = "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=49-step=53350.ckpt"
service = SpeechExtractorService(MODEL_PATH)


@app.post("/extract-speech")
async def extract_speech(audio_file: UploadFile = File(...)):
    """
    Extract speech from an uploaded audio file.
    Returns the processed audio as a WAV file.
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        input_path = os.path.join(temp_dir, "input.wav")
        with open(input_path, "wb") as f:
            f.write(await audio_file.read())

        # Process the audio
        extracted_speech,sample_rate = service.extract_speech(input_path)

        # Save the processed audio
        output_path = os.path.join(temp_dir, "extracted_speech.wav")
        torchaudio.save(
            output_path,
            torch.from_numpy(extracted_speech),
            sample_rate=sample_rate  # Adjust to match your model's sample rate
        )

        # Return the processed file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="extracted_speech.wav"
        )


@app.get("/health")
async def health_check():
    """Check if the service is running."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)
