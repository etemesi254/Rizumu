from rizumu.model import RizumuModelV2
from rizumu.pl_model import RizumuLightning


def export_onnx(filename: str = "./rizumu_onnx.onnx"):
    model = RizumuLightning.load_from_checkpoint("/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=8-step=6435.ckpt")
    model.to_onnx(filename,dynamo_export=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "channels", 1: "length"},
                                    "output": {0: "channels", 1: "length"}})




if __name__ == "__main__":
    export_onnx()