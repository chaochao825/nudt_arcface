# ARCFACE Face Recognition - Attack & Defense System

This project implements a comprehensive face recognition attack and defense system using the ARCFACE model.

## Features

### Supported Attack Methods
1. **BIM** (Basic Iterative Method)
2. **DIM** (Diverse Input Method)
3. **TIM** (Translation-Invariant Method)
4. **PGD** (Projected Gradient Descent)
5. **CW** (Carlini & Wagner)
6. **DeepFool**

### Supported Defense Methods
1. **HGD** (High-level Guided Denoising)
2. **TVM** (Total Variation Minimization)
3. **LivenessDetection** (Face Spoofing Detection)
4. **FeatureSpacePurification**
5. **EnsembleDefense**

### Supported Datasets
1. **VGGFace2**
2. **CelebA**
3. **CASIA-WebFace**
4. **LFW** (Labeled Faces in the Wild)
5. **YaleB**
6. **MegaFace**

## Project Structure

```
nudt_arcface/
├── attacks/                # Attack methods implementation
│   ├── bim.py
│   ├── dim.py
│   ├── tim.py
│   ├── pgd.py
│   ├── cw.py
│   └── deepfool.py
├── defends/               # Defense methods implementation
│   ├── hgd.py
│   ├── tvm.py
│   ├── liveness_detection.py
│   ├── feature_space_purification.py
│   └── ensemble_defense.py
├── datasets/              # Dataset loaders
│   ├── vggface2.py
│   ├── celeba.py
│   ├── webface.py
│   ├── lfw.py
│   ├── yaleb.py
│   └── megaface.py
├── face_recognizer/       # Model implementation
│   ├── arcface_model.py
│   └── main.py
├── utils/                 # Utility functions
│   ├── sse.py            # SSE output formatting
│   └── yaml_rw.py        # YAML config handling
├── main.py               # Entry point
├── Dockerfile            # Docker image configuration
├── requirements.txt      # Python dependencies
├── docker_run_scripts.sh # Docker run examples
└── ENV_VARIABLES.md      # Environment variables documentation
```

## Installation

### Local Installation

```bash
# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build Docker image
docker build -t nudt_arcface:latest .
```

## Usage

### Running Locally

#### Attack Mode
```bash
python main.py \
  --input_path ./input \
  --output_path ./output \
  --process adv \
  --attack_method bim \
  --epsilon 0.031 \
  --step_size 0.008 \
  --max_iterations 10 \
  --device 0
```

#### Defense Mode
```bash
python main.py \
  --input_path ./input \
  --output_path ./output \
  --process defend \
  --defend_method hgd \
  --device 0
```

### Running with Docker

#### BIM Attack Example
```bash
docker run --rm --gpus all \
  -v /path/to/input:/project/input:ro \
  -v /path/to/output:/project/output:rw \
  -e PROCESS=adv \
  -e ATTACK_METHOD=bim \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  nudt_arcface:latest
```

#### HGD Defense Example
```bash
docker run --rm \
  -v /path/to/input:/project/input:ro \
  -v /path/to/output:/project/output:rw \
  -e PROCESS=defend \
  -e DEFEND_METHOD=hgd \
  nudt_arcface:latest
```

See `docker_run_scripts.sh` for more examples.

## Input Directory Structure

```
input/
├── model/
│   └── model_weights.pth    # Pretrained model weights
└── data/
    └── dataset_name/         # Dataset folder
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Output Format

### SSE (Server-Sent Events) Format

All outputs follow the SSE format for real-time progress monitoring:

```
event: input_path_validated
data: {"status": "success", "message": "Input path is valid and complete.", "file_name": "/project/input"}

event: adv_samples_gen_validated
data: {"status": "success", "message": "adv sample is generated.", "file_name": "/project/output/adv_image1.jpg"}
```

## Environment Variables

See `ENV_VARIABLES.md` for complete documentation of all supported environment variables.

### Key Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| PROCESS | Operation mode | `attack` | `adv`, `attack`, `defend`, `train` |
| ATTACK_METHOD | Attack to use | `bim` | `bim`, `dim`, `tim`, `pgd`, `cw`, `deepfool` |
| DEFEND_METHOD | Defense to use | `hgd` | `hgd`, `tvm`, `livenessdetection`, etc. |
| EPSILON | Attack strength | `0.031` | `0.0 - 1.0` |
| MAX_ITERATIONS | Attack iterations | `10` | `1 - 1000` |
| DEVICE | GPU device ID | `0` | `-1` (CPU) or `0+` (GPU) |

## Model Information

### ARCFACE Architecture
- **Backbone**: ResNet-50
- **Embedding Size**: 512 dimensions
- **Input Size**: 112x112 pixels
- **Output**: Identity classification logits

## Examples

### Example 1: PGD Attack on LFW Dataset
```bash
docker run --rm --gpus all \
  -v /data/lfw:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=adv \
  -e DATA=lfw \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.062 \
  -e STEP_SIZE=0.016 \
  -e MAX_ITERATIONS=20 \
  -e DEVICE=0 \
  nudt_arcface:latest
```

### Example 2: Ensemble Defense
```bash
docker run --rm \
  -v /data/adversarial:/project/input:ro \
  -v /data/defended:/project/output:rw \
  -e PROCESS=defend \
  -e DEFEND_METHOD=ensembledefense \
  nudt_arcface:latest
```

### Example 3: C&W Attack with Custom Parameters
```bash
docker run --rm --gpus all \
  -v /data/celeba:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=adv \
  -e DATA=celeba \
  -e ATTACK_METHOD=cw \
  -e MAX_ITERATIONS=100 \
  -e DEVICE=0 \
  nudt_arcface:latest
```


