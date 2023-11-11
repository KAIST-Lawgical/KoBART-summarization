FROM nvcr.io/nvidia/pytorch:22.05-py3

EXPOSE 6006 6007 6008 6009
RUN python3 -m pip --no-cache-dir install --upgrade \
    boto3 \
    pandas \
    konlpy \
    datasets \
    transformers==4.16.2 \
    pytorch-lightning==1.5.10 \
    streamlit==1.2.0 \
    torchmetrics==0.6.0 \
    omegaconf==2.1.0 \
    gpustat==0.6.0 \
    gdown \
    && \
apt update && \
apt install -y \
    tmux \
    htop \
    ncdu \
    vim \
    && \
apt clean && \
apt autoremove && \
rm -rf /var/lib/apt/lists/* /tmp/* && \
mkdir /KoBART-summarization
COPY . /KoBART-summarization/
WORKDIR /KoBART-summarization
