import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware

from rizumu.convert_file import convert_file
from rizumu.pl_model import RizumuLightning

origins = [
    "*",
]

app = FastAPI(title="Speech Extractor API")

# Initialize the service
# Replace with your model checkpoint path
MODEL_PATH = "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=49-step=53350.ckpt"
model = RizumuLightning.load_from_checkpoint(MODEL_PATH)


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
        output_path = os.path.join(temp_dir, "output.wav")
        convert_file(input_path, output_path, model)

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
    run()
