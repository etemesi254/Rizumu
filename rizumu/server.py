import os
import tempfile
from typing import Tuple

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware

from rizumu.pl_model import RizumuLightning

origins = [
    "*",
]

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
        extracted_speech, sample_rate = service.extract_speech(input_path)

        # Save the processed audio
        output_path = os.path.join(temp_dir, "extracted_speech.wav")
        torchaudio.save(
            output_path,
            torch.from_numpy(extracted_speech),
            sample_rate=sample_rate,
            encoding="PCM_F",
            format="wav"
        )

        file_data = open(output_path, "rb").read()
        # Return the processed file
        return Response(
            file_data,
            media_type="audio/wav",

        )


@app.get("/health")
async def health_check():
    """Check if the service is running."""
    return {"status": "healthy"}


def run():
    import uvicorn

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host="0.0.0.0", port=9000)


if __name__ == "__main__":
    # run()
    model = RizumuLightning.load_from_checkpoint(
        "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=49-step=53350.ckpt")
    model.eval()
    audio, sr = torchaudio.load("/Users/etemesi/Datasets/312/mix.wav")
    output = model(audio)
    torchaudio.save(
        "./tt.wav",
        output.detach().cpu(),
        sample_rate=44100,
        encoding="PCM_F",
        format="wav"
    )
