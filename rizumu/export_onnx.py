from rizumu.model import RizumuModelV2
from rizumu.pl_model import RizumuLightning
import torch

def export_onnx(filename: str = "./rizumu_onnx.onnx"):
    model = RizumuLightning.load_from_checkpoint("/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=0-step=715.ckpt")
    # input and output are dynamic, so we can send anything
    sample_input = torch.randn(1,23000)
    model.to_onnx(filename,sample_input,
                    dynamo_export=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "channels", 1: "length"},
                                    "output": {0: "channels", 1: "length"}})




if __name__ == "__main__":
    export_onnx()