# udder-seg-dinov3

This repository contains the code release for the scientific paper **Lightweight Thermal Udder Segmentation via Structured Pruning for On-Device Farm Deployment**. It includes training scripts, as well as an inference module designed for NVIDIA Jetson devices, exposed as a REST API.

## Table of Contents

1. [Training](#training)
2. [Configuration File](#configuration-file)
3. [Inference on NVIDIA Jetson](#inference-on-nvidia-jetson)
4. [Training Logs and Model Weights](#official-training-logs-and-model-weights)

## Training

The [`train+prune.py`](train+prune.py) script is the main entry point for training the model. This script trains the model on your dataset, applies pruning techniques to optimize the model for deployment, and finally fine-tunes it.

### Prerequisites

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** DINOv3 backbone weights are automatically downloaded from [Hugging Face Hub](https://huggingface.co/collections/timm/timm-dinov3) via the [timm](https://github.com/huggingface/pytorch-image-models) library. No manual weight download is required!

### Running the Script

To run the training process, execute the following command:

```bash
python train+prune.py
```

### Key Parameters

The script uses several parameters defined in the [`config.ini`](config.ini) file. These parameters control the training process, as well as other things like logging.

## Configuration File

Below is a snippet of the structure of the file:

```ini
[GENERAL]
SEED = 42
IN_CHANS = 1  # Input channels (1 for grayscale thermal images)

[MODEL]
MODEL_ENCODER = convnext
MODEL_NAME = convnext_base.dinov3_lvd1689m

[TRAINING]
BATCH_SIZE = 8
LR = 0.005
MAX_EPOCHS = 200

[PRUNING]
PRUNING_RATIO = 0.30  # 30% of weights removed (~50% param reduction)
ROUND_TO = 8          # Channel rounding for Jetson/TensorRT
```

### Available DINOv3 Models

The following DINOv3 models are available via timm:

**ConvNeXt variants** (use with `MODEL_ENCODER = convnext`):
- `convnext_tiny.dinov3_lvd1689m`
- `convnext_small.dinov3_lvd1689m`
- `convnext_base.dinov3_lvd1689m`

**ViT variants** (use with `MODEL_ENCODER = vit`):
- `vit_small_patch16_dinov3.sat493m`
- `vit_base_patch16_dinov3.sat493m`

To see all available DINOv3 models in timm, run:
```python
import timm
print(timm.list_models('*dinov3*'))
```

### Input Channel Handling

timm handles input channels natively via the `in_chans` parameter. Simply set `IN_CHANS = 1` in the config file for grayscale thermal images - no external input adapters are needed. timm automatically adapts the first convolutional layer when `in_chans != 3`.

Modify these values and the others as needed to suit your dataset and hardware.

## Inference on NVIDIA Jetson

The [`jetson`](jetson/) folder contains the code for deploying the model on NVIDIA Jetson devices. The model is exposed as a REST API for inference. Note that in this release, we specifically address the Nvidia Jetson Orin Nano devices, but it can be easily changed to address other Jetson devices.

### Folder Structure

- [`app/serve_cuda.py`](jetson/app/serve_cuda.py): The main script for running the REST API server.
- [`client.py`](jetson/client.py): A client script to send inference requests to the server for testing purposes.
- [`models/`](jetson/models/): The location of your ONNX model for inference.
- [`Dockerfile`](jetson/Dockerfile): Docker configuration for building and deploying the application.

### Running the Inference Server

1. Place your ONNX model in the [`jetson/models/`](jetson/models/) folder.
2. Move to the `jetson` folder.
2. Build the Docker image:

   ```bash
   docker build -t jetson-inference .
   ```

3. Run the Docker container:

   ```bash
   docker run -d \
   --runtime nvidia \
   --network host \
   -e MODEL_PATH=/models/your_model.onnx \
   -e WARMUP_RUNS=5 \
   -v $(pwd)/models:/models:ro \
   jetson-inference
   ```

4. The REST API segmentation endpoint will be available at `http://<JETSON_IP>:8000/segment`.

### Sending Inference Requests

For testing, you can use the provided [`client.py`](jetson/client.py) script to send inference requests:

```bash
python client.py --url http://<JETSON_IP>:8000/segment --image path/to/image.png
```

We also provide an [`example image`](jetson/example.png) for this purpose.

## Official Training Logs and Model Weights

- **Training Logs**: [WandB](https://wandb.ai/mat094/udder-seg-dinov3)
- **Model Weights**: [Google Drive](https://drive.google.com/file/d/1AMtpJtvlrIduNFDs8Cr5rfEyHy6eVEuD/view?usp=sharing)

This repository is released under the MIT [`LICENSE`](LICENSE). For any questions, please contact the authors of the paper or open an issue here on the official repository.
