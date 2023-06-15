# 1 apt install *
apt-get update
apt-get install -yq --no-install-recommends $(awk '{print $1'} packages.txt | tr "\n" " ")

# 2 pip install
python3 -m pip install -r requirements.txt

# 3 get ofa
cd ./attachments
git clone https://github.com/OFA-Sys/OFA.git
cd ..

# 4 get weights
wget -c https://huggingface.co/jxu124/tio-checkpoint-zoo/resolve/main/ckpts/tio-huge/checkpoint.best_score_0.7630.pt -o ./attachments/checkpoint.pt

# 5 start!
python3 app.py --share
