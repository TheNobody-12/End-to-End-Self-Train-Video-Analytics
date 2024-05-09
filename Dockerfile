FROM pytorch/torchserve:latest-gpu
# RUN mkdir -p /app/templates
USER root
RUN apt-get install --reinstall apt
# RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip3 install -r requirements.txt
RUN dvc init -f
RUN dvc repro

EXPOSE 5000
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

RUN torchserve --stop
RUN torch-model-archiver -f --model-name yolov8n --version 1.0 --serialized-file torchserve/models/trained_model_60epoch.onnx --export-path torchserve/model-store --handler torchserve/custom_handler.py

# CMD ["/bin/bash", "./start_script.sh"]
# CMD ["torchserve", "--start", "--ncs", "--model-store", "torchserve/model-store", "--models", "spaceship=spaceship.mar"]
# torchserve --start --ncs --model-store torchserve/model-store --models yolov8n=yolov8n.mar --ts-config torchserve/config.properties