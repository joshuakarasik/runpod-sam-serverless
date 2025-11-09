FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Copy only what we need
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM model weights from official source
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth

COPY handler.py .

# Explicit entrypoint
CMD ["python", "-u", "handler.py"]

