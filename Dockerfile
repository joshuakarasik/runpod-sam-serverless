FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /app

# Copy application files (not the model - we'll download it)
COPY handler.py requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM model weights from official source
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth

CMD ["python", "-u", "handler.py"]

