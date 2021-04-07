FROM ubuntu:18.04
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev ffmpeg libsm6 libxext6 libxrender-dev
WORKDIR /app
COPY ./requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "object_detection/object_detection.py", "yolo_tiny_configs/"]