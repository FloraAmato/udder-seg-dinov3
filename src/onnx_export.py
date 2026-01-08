import os
import torch

def export_onnx(model, dummy_input, model_name):
    # Only export once in DDP
    if int(os.environ.get("RANK", "0")) != 0:
        return

    out_path = f"checkpoints/{model_name.lower()}/best.onnx"

    kwargs = dict(
        input_names=["image"],
        output_names=["logits"],
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
        dynamic_axes={
            "image":  {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

    with torch.no_grad():
        torch.onnx.export(model, dummy_input, out_path, dynamo=False, **kwargs)
