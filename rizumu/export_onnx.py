import torch
import torchaudio
import soundfile as sf
from rizumu.pl_model import RizumuLightning


def export_onnx(filename: str = "./rizumu_onnx.onnx"):
    model = RizumuLightning.load_from_checkpoint(
        "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=38-step=41613.ckpt")
    #model = RizumuLightning(labels=["mix","speech"],output_label_name="speech",mix_name="mix",num_splits=1)
    # input and output are dynamic, so we can send anything
    sample_input = torch.randn(1, 23000)
    # model.to_onnx(filename, sample_input,
    #               dynamo_export=True,
    #               external_data=False,
    #               report=True,
    #               input_names=["input"],
    #               output_names=["output"],
    #               dynamic_axes={"input": {0: "channels", 1: "length"},
    #                             "output": {0: "channels", 1: "length"}})

    input,sr = torchaudio.load("/Users/etemesi/Datasets/dnr_v2/cv/10702/mix.wav")
    output = model(input).detach()
    torchaudio.save("./hello2.wav", output,sr,encoding="PCM_F",format="wav")


if __name__ == "__main__":
    export_onnx()
