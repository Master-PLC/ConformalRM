# SelectiveRM
Official implementation of "Optimal Transport for Reward Modeling from Noisy Feedback"

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

You can run the following command to train the SelectiveRM model.

### Stage 1: Download preference data

Download the preference data from huggingface into `rawdata` directory.

```bash
python download.py --data_name hs
```

### Stage 2: Embedding Extraction

Extract embeddings from a pretrained LLM reward model. This produces safetensors files with `embeddings` (Tensor[N, D]) and `labels` (Tensor[N]) keys. This stage requires access to the pretrained model like `FsfairX-LLaMA3-RM-v0.1`. You can download it into `ckpt` directory from [here](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1). And then run:

```bash
python data_prepare.py --data_name hs --subset train
python data_prepare.py --data_name hs --subset test
```

Expected output location: `./embeddings/normal/{model_name}_{data_name}_{train|test}.safetensors`

### Stage 3: Binary Noise Simulation

Simulate binary noise on the labels with ratio $\rho_{01}$ and $\rho_{10}$. This stage requires access to the Stage 2 safetensors files. Run:

```bash
python simulate_binary_noise.py --r01 0.2 --r10 0.2 --data_name hs
```
Expected output location: `./embeddings/binary_noisy/{model_name}_{data_name}_{r01}_{r10}.safetensors`

### Stage 4: Estimate Noise Ratio with Cleanlab

To facilitate SelectiveRM, we need to estimate the noise ratio $\rho$ from the data. We use the Cleanlab library to estimate the noise ratio. This stage requires access to the Stage 3 safetensors files. Run:

```bash
python noise_ratio_est.py --r01 0.2 --r10 0.2 --data_name hs
```

You need to log the estimated noise ratio $\rho$ in the `configs.py` or just pass it into the `selectiverm.py` script with parameter `--m` which equals to $1 - \rho$.

### Stage 5: SelectiveRM Training & Evaluation

```bash
# log into configs.py
python selectiverm.py --r01 0.2 --r10 0.2 --data_name hs

# or pass it into the script
python selectiverm.py --r01 0.2 --r10 0.2 --data_name hs --m 1-rho --mount_input fixed
```

This trains the SelectiveRM model on the binary-noisy data and evaluates it on the test set.
