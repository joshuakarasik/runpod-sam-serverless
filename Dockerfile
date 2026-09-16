# CUDA 12.8 + PyTorch 2.8 is required for Blackwell (RTX PRO 6000, sm_120).
# The previous 2.1.0-cuda11.8 image only compiled sm_50–sm_90, so RunPod
# workers scheduled onto Blackwell MIG slices crashed every job.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Force CUDA 12.8 wheels even if the base image shipped an older torch.
# Do not install torch from default PyPI — those wheels lack sm_120.
RUN pip uninstall -y torch torchvision torchaudio || true \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('archs', torch.cuda.get_arch_list()); assert torch.version.cuda, 'torch has no CUDA'; maj, mn = map(int, torch.version.cuda.split('.')[:2]); assert (maj, mn) >= (12, 8), torch.version.cuda; p = torch.__version__.split('+')[0].split('.'); assert (int(p[0]), int(p[1])) >= (2, 7), torch.__version__"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM ViT-H weights from the official source
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth

COPY handler.py .

CMD ["python", "-u", "handler.py"]
