# download om model
wget -nc --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RU1UBVH5EBbVV4CVAPuNokSzpfx9A3Ug' -O mot_v2.om
# download sample video london.mp4
wget -nc --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ntbudc1JB8HzEw38pwZKPXukrgADiKdS' -O ../data/london.mp4
cd ../src
python3 main.py --conf_thres 0.35 --input_video ../data/london.mp4