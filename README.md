- [1. Install](#1-install)
- [2. Usage](#2-usage)
  - [2.1. Train](#21-train)
    - [2.1.1. Example](#211-example)
  - [2.2. Generate](#22-generate)
    - [2.2.1. Example](#221-example)
  - [2.3. Convert to decimal text file to binary text file](#23-convert-to-decimal-text-file-to-binary-text-file)
- [3. Cite](#3-cite)

---

This repo contains the official implementation for the paper [Transformers in Pseudo-Random Number Generation: A Dual Perspective on Theory and Practice](https://arxiv.org/abs/2508.01134)

---

## 1. Install

1. Create a virtual python environment.

```bash
conda create -n transformer_prng python=3.12
conda activate transformer_prng
```

2. Install [pytorch](https://pytorch.org/get-started/locally/) that matches your CUDA version.

For example:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
```

3. Install dependencies.

```bash
pip install "ls_mlkit @ git+https://github.com/zeroDtree/my_pkg_py.git" --no-cache-dir
pip install -r requirements.txt --no-cache-dir
```

## 2. Usage

### 2.1. Train

```bash
python main.py {training config}
```

#### 2.1.1. Example

```
python main.py ++dataset.name=mt19937-12 ++train.num_train_epochs=50
```

After training, the model will be saved in `./checkpoint/`

### 2.2. Generate

Generate random number to a file.

```
python generator.py {model_path} {num_sampling} {num_bits} {file_path}
```

#### 2.2.1. Example

```
python generator.py ./checkpoint/finetuned-mt19937-12-openai_community_gpt2/ 4096 12 mt12bit.txt
```

### 2.3. Convert to decimal text file to binary text file

```
python txt2bxt.py {text_file_path} {binary_text_file_path}
```

## 3. Cite

```
@article{li2025transformers,
  title={Transformers in Pseudo-Random Number Generation: A Dual Perspective on Theory and Practice},
  author={Li, Ran and Zeng, Lingshu},
  journal={arXiv preprint arXiv:2508.01134},
  year={2025}
}
```
