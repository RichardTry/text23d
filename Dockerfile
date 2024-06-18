# syntax=docker/dockerfile:1

FROM python:3.10-bullseye
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app

COPY ./run.py /app/run.py
COPY ./bot.py /app/bot.py
COPY ./tiny_nerf/ /app/tiny_nerf/
COPY ./stable_diffusion_guidance/ /app/stable_diffusion_guidance/
COPY ./camera_view_generator/ /app/camera_view_generator/

CMD ["python3", "/app/bot.py"]