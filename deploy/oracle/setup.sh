#!/usr/bin/env bash
# One-command setup of the AHUIKE API on Ubuntu 22.04 (Oracle Cloud Always Free, VM.Standard.A1.Flex).
#   HF_TOKEN=hf_xxx bash setup.sh
# Takes 20-30 minutes: compiling llama.cpp (~10-15 min) and downloading the 4.9 GB model.
set -euo pipefail
: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face read token first}"
GGUF="${AHUIKE_GGUF:-SamEgwu/AHUIKE-N-ATLaS-8B-GGUF-Powered-by-Awarri/AHUIKE-N-ATLaS-8B-Q4_K_M.gguf}"

echo "== system packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-dev build-essential cmake git iptables-persistent

echo "== code"
cd "$HOME"
if [ -d ahuike ]; then git -C ahuike pull --ff-only; else git clone https://github.com/EgwuSamuel/ahuike.git; fi

echo "== python environment (compiles llama.cpp)"
python3 -m venv "$HOME/venv"
"$HOME/venv/bin/pip" install --upgrade pip
CMAKE_ARGS="-DGGML_NATIVE=ON" "$HOME/venv/bin/pip" install -r "$HOME/ahuike/api/requirements.txt"

echo "== settings"
sudo tee /etc/ahuike.env >/dev/null <<ENV
HF_TOKEN=$HF_TOKEN
AHUIKE_GGUF=$GGUF
AHUIKE_LOG=/home/ubuntu/ahuike/results/api_requests.jsonl
HF_HOME=/home/ubuntu/.cache/huggingface
ENV
sudo chmod 600 /etc/ahuike.env

echo "== model download (4.9 GB)"
set -a; source /etc/ahuike.env; set +a
"$HOME/venv/bin/python" - <<PY
import os
from huggingface_hub import hf_hub_download
repo, fname = os.environ["AHUIKE_GGUF"].rsplit("/", 1)
print(hf_hub_download(repo, fname, token=os.environ["HF_TOKEN"]))
PY

echo "== firewall: open port 8000 (Oracle Ubuntu images block it by default)"
if ! sudo iptables -C INPUT -p tcp --dport 8000 -j ACCEPT 2>/dev/null; then
  sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
  sudo netfilter-persistent save
fi

echo "== service"
sudo cp "$HOME/ahuike/deploy/oracle/ahuike-api.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ahuike-api

IP=$(curl -s https://ifconfig.me || echo "<public-ip>")
echo
echo "Done. The model loads in about a minute. Then open:  http://$IP:8000/docs"
echo "Logs: sudo journalctl -u ahuike-api -f"
