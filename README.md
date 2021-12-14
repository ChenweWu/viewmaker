## 1) Install Dependencies

```console
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other dependencies:
```console
pip install -r requirements.txt
```

## 2) Running experiments

Start by running
```console
source init_env.sh
```

Now, you can run experiments for the different modalities as follows:

```console
scripts/run_ecg.py config/ecg/pretrain_viewmaker_ptb_xl_simclr.json --gpu-device 0
```

Scripts contributed for COS429 Final Project:

The `scripts` directory holds:
- `run_ecg.py`: for pretraining and running linear evaluation on PTB-XL with spectrogram inputs
- `run_ecg_1d.py`: for pretraining and running linear evaluation on PTB-XL with 1D ECG time series inputs

The `config/ecg` directory holds all experiment configuration files. The first field in every config file is `exp_base` which specifies the base directory to save experiment outputs, which you should change for your own setup.

The `src/datasets` directory holds:
- `ptb_xl.py`: for loading PTB-XL batch inputs in spectrogram format
- `ptb_xl_1d.py`: for loading PTB-XL batch inputs in 1D time series signal format

The `src/models` directory holds:
- `resnet_1d.py`: for running a ResNet18 on 1D inputs. Taken from 3KG codebase.
- `viewmaker_1d.py`: for running a Viewmaker network on 1D inputs. Inspired by resnet_1d.py.

The `src/systems` directory holds:
- `ecg_systems.py`: for initializing pretraining and transfer learning models with spectrogram inputs
- `ecg_1d_systems.py`: initializing pretraining and transfer learning models with 1D time series signal inputs

