# Project ISSAC - Language ID - Asian Language
- Train Languages: Mandarin(zh) English(en) Indonesian(id) Tagalog(tl) Out-of-languages[Lao(lo), Burmese(my)]<br>
- Test Languages: Mandarin(zh) English(en) Indonesian(id) Tagalog(tl) Out-of-languages[Thai(th), Vietnamese(vi)]<br> 
## Installation:
1. Install Kaldi
```bash
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi
cd kaldi/tools/; 
# Run this next line to check for dependencies, and then install them
extras/check_dependencies.sh
make; cd ../src; ./configure; make depend; make
```
2. sph2pipe
```
cd sph2pipe_v2.5
gcc -o sph2pipe *.c -lm

# Add the sph2pipe to the user environment variables
vim ~/.bashrc
export PATH=/home3/lirui/sph2pipe_v2.5:$PATH
source ~/.bashrc
```
## Download the project
1. Clone the project from GitHub into your workspace
```
git clone https://github.com/lirui-cyber/ISSAC_Lid_Asian.git
```
2. Point to your kaldi <br>
Open ```ISSAC_Lid_Asian/path.sh``` file, change **KALDI_ROOT** to your kaldi directory,e.g.
```
KALDI_ROOT=/home/asrxiv/w2021/kaldi-cuda11
```
## Data preparation
The data folder contains:<br>
- **Train sets**: train_zh_en_id_tl_lo_my {zh_train, en_train, id_train, tl_train, lo_train, my_train}<br>
- **Test sets**: zh_test, en_test, id_test, tl_test, th_test, vi_test <br>
- **Noise Rats data**: rats_noise_channel_AEH, rats_noise_channel_BCDFG
### Modify the path 
You can use the ```sed``` command to replace the path in the wav.scp file with your path <br>
```
egs:
Original train data path: /home3/theanhtran/corpus/2022_Asia_ISSAC/VoxLingua107/en/-F5o0jnJgo0__U__S0---0015.950-0025.980.wav
Your path: /data/VoxLingua107/en/-F5o0jnJgo0__U__S0---0015.950-0025.980.wav
sed -i "s#/home3/theanhtran/corpus/2022_Asia_ISSAC/#/data/#g" data/train_zh_en_id_tl_lo_my/wav.scp

Original Rats data path:/home3/andrew219/python_scripts/extract_rats_noise/rats_channels/channel_A/10002_20705_alv_A.wav
Your path: /data/rats_channels/channel_A/10002_20705_alv_A.wav
sed -i "s#/home3/andrew219/python_scripts/extract_rats_noise/#/data/#g" data/rats_noise_channel_AEH/wav.scp
```

### Add noise 
- Clean training set: train_zh_en_id_tl_lo_my<br>
- Noise training set: train_zh_en_id_tl_lo_my_5th [clean,20,15,10,5snrs]
```
  bash add_noise.sh
```
## Training pipeline
Before execution, please check the parameters in ```xsa_config``` <br>
You need to change two parameters:<br>
- **userroot**: Project root 
- **model_path**: The path of pretrained-model xlsr_53_56k.pt. <br>
You can download the model from this link below:  https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt <br>
### Set up Conda environment
```
conda create -n xsa python=3.8 numpy pandas
conda activate xsa
```
- install pytorch
```
conda install pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
- install librosa, kaldiio
```
pip install librosa
pip install kaldiio 
```
- install fairseq
```
git clone -b v0.12.2 https://github.com/facebookresearch/fairseq.git  tool/fairseq
pip install -e tool/fairseq
```
- install s3prl
```
git clone -b v0.3.4 https://github.com/s3prl/s3prl.git tool/s3prl
sed -i '60d' tool/s3prl/setup.py
pip install -e tool/s3prl/
mv tool/expert.py tool/s3prl/s3prl/upstream/wav2vec2/
```

### Extracting wav2vec2 features
```
python3 extract_features.py
```
### Training 
```
python3 train_xsa.py
```
## Test pipeline
You can change "check_point" variable in xsa_config.json file, Change to the epoch you want to use.
```
python3 test.py
```

## Notice
All the required parameters in the script are written in the xsa_config.json file.
