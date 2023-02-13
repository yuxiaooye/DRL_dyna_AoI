FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 git\
 && sudo rm -rf /var/lib/apt/lists/*

# Install from PyPI.
RUN pip install gym wandb tensorboardX json5 matplotlib pandas shapely folium geopandas movingpandas -i https://pypi.tuna.tsinghua.edu.cn/simple
