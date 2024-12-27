# def export_onnx(filename: str = "./rizumu_onnx.onnx"):
#     model = RizumuLightning.load_from_checkpoint(
#         "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=49-step=53350.ckpt")
#     #model = RizumuLightning(labels=["mix","speech"],output_label_name="speech",mix_name="mix",num_splits=1)
#     # input and output are dynamic, so we can send anything
#     sample_input,sr = torchaudio.load("/Users/etemesi/Datasets/dnr_v2/cv/488/mix.wav")
#
#     #torch.onnx.export(do_constant_folding=)
#     model.to_onnx(filename, sample_input,
#                   dynamo_export=True,
#                   external_data=False,
#                   report=True,
#                   do_constant_folding=False,
#
#                   input_names=["input"],
#                   output_names=["output"],
#                   dynamic_axes={"input": {0: "channels", 1: "length"},
#                                 "output": {0: "channels", 1: "length"}})
#
#     # output = model(input).detach()
#     # torchaudio.save("./hello2.wav", output,sr,encoding="PCM_F",format="wav")
import torchaudio

from rizumu.pl_model import RizumuLightning


# if __name__ == "__main__":
#    export_onnx()

def load_onnx_and_run(model, audio_file):
    import onnxruntime
    audio, sr = torchaudio.load(audio_file)
    audio = audio.detach().numpy()
    sess = onnxruntime.InferenceSession(model)
    output = sess.run(["output"], {'input': audio})[0]
    return output

def load_normal_model(audio_file):
    model = RizumuLightning.load_from_checkpoint(
             "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=49-step=53350.ckpt")
    audio, sr = torchaudio.load(audio_file)


    output = model(audio).detach()
    return output.numpy()


if __name__ == '__main__':
    import numpy as np
    onnx = load_onnx_and_run("/Users/etemesi/PycharmProjects/Rizumu/rizumu/rizumu_onnx.onnx",
                             "/Users/etemesi/Datasets/dnr_v2/cv/488/mix.wav")
    output_run = load_normal_model("/Users/etemesi/Datasets/dnr_v2/cv/488/mix.wav")

    print(np.testing.assert_allclose(onnx, output_run))
