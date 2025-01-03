import numpy as np
import torch.nn.functional
import torchaudio

from rizumu.pl_model import RizumuLightning

def export_onnx(filename: str = "./rizumu_onnx.onnx"):
    model = RizumuLightning.load_from_checkpoint(
        "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=51-step=59466.ckpt")
    #model = RizumuLightning(labels=["mix","speech"],output_label_name="speech",mix_name="mix",num_splits=1)
    # input and output are dynamic, so we can send anything
    sample_input,sr = torchaudio.load("/Volumes/Untitled/DNR/dnr/dnr/dnr/tt/312/mix.wav")

    #torch.onnx.export(do_constant_folding=)
    # model.to_onnx(filename, sample_input,
    #               dynamo_export=True,
    #               external_data=False,
    #               report=True,
    #               do_constant_folding=False,
    #
    #               input_names=["input"],
    #               output_names=["output"],
    #               dynamic_axes={"input": {0: "channels", 1: "length"},
    #                             "output": {0: "channels", 1: "length"}})

    output = model(sample_input).detach()
    torchaudio.save("./newer_model.wav", output,sr,encoding="PCM_F",format="wav")

def calculate_sdr(target_tensor, output_tensor) -> float:
    """
     Calculate the signal to distortion ratio between target and output tensor
    :param target_tensor: The true expected output
    :param output_tensor: The predicted output
    :return:  The signal to distortion ratio between target and output tensor
    """

    target_tensor = target_tensor.detach().cpu().numpy()
    output_tensor = output_tensor.detach().cpu().numpy()

    target_power = np.sum(target_tensor ** 2)
    noise_power = np.sum((target_tensor - output_tensor) ** 2)

    if noise_power == 0:
        return float('inf')  # Handle the case where the noise power is zero to prevent division by zero

    sdr = 10 * np.log10(target_power / noise_power)
    return sdr

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
    # import numpy as np
    # onnx = load_onnx_and_run("/Users/etemesi/PycharmProjects/Rizumu/rizumu/rizumu_onnx.onnx",
    #                          "/Users/etemesi/Datasets/dnr_v2/cv/488/mix.wav")
    # output_run = load_normal_model("/Users/etemesi/Datasets/dnr_v2/cv/488/mix.wav")
    #
    # print(np.testing.assert_allclose(onnx, output_run))
    export_onnx()
    a,_ = torchaudio.load("/Volumes/Untitled/DNR/dnr/dnr/dnr/tt/312/speech.wav")
    b,_ =torchaudio.load("/Users/etemesi/PycharmProjects/Rizumu/rizumu/older_model.wav")
    c,_ = torchaudio.load("/Users/etemesi/PycharmProjects/Rizumu/rizumu/newer_model.wav")
    d = torch.nn.functional.mse_loss(a,b)
    e = torch.nn.functional.mse_loss(a,c)

    print(f"SDR :{calculate_sdr(a,b)} loss: {d}")
    print(f"SDR :{calculate_sdr(a,c)} loss: {e}")

