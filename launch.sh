#!/bin/bash
set -ex

export http_proxy=sys-proxy-rd-relay.byted.org:8118 https_proxy=sys-proxy-rd-relay.byted.org:8118 no_proxy=byted.org

# pip install --force-reinstall --no-deps torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


cd /opt/tiger/Megatron-LM/
#cd /mlx_devbox/users/liujuncai/repo/1291/Megatron-LM/

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq/huangqi/demo_gpt_text.zip
sudo apt install unzip
unzip demo_gpt_text.zip -d demo_gpt_text

# pip install --force-reinstall -v ./
pip install pybind11

# sudo cp /opt/tiger/Megatron-LM/examples/collate.py /usr/local/lib/python3.7/dist-packages/torch/utils/data/_utils/collate.py

export PYTHONPATH=$PYTHONPATH:/opt/tiger/janus

export ARNOLD_ID=${ARNOLD_ID:-0}
if [[ "$ENABLE_PROFILE" == "1" && "$ARNOLD_ID" == "0" && "$DISABLE_NSYS" != "1" ]]; then
  cd /opt/tiger && wget http://tosv.byted.org/obj/bin/toscli -O toscli && chmod a+x toscli && cd -
  curl -fsSL https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | sudo apt-key add -
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
  sudo apt-get update -y && sudo apt-get -y install cuda-nsight-systems-11-3 --no-install-recommends
  nsys launch bash examples/$@
else
  bash examples/$@
fi

# # run it
# if ! command -v TORCHRUN &> /dev/null
# then
#     echo "==<TORCHRUN could not be found, use cruise included script>=="
#     /opt/tiger/cruise/cruise/tools/TORCHRUN $@
#     exit
# else
#     TORCHRUN $@
# fi