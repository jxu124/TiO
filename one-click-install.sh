# 1 apt install *
apt-get update
apt-get install -yq --no-install-recommends $(awk '{print $1'} packages.txt | tr "\n" " ")

# 2 pip install
python3 -m pip install -r requirements.txt

# 3 start!
# python3 app.py --share
