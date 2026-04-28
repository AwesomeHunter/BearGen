# Neural Networks Project

This project was created by a friend and me during a university Neural Networks course.

## Project Goal

Train Quick, Draw!-based classifiers and use them to filter teddy bear sketches, then train GAN/VAE models to generate new teddy bear-like images and videos.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

- `classifier_d8.ipynb`: trains/evaluates the 8-class classifier and saves `models/classifier_d8.model`.
- `classifier_d15.ipynb`: trains/evaluates the 15-class classifier and saves `models/classifier_15.model`.
- `data_filtering.ipynb`: loads both classifiers, filters `datasets/teddy-bear.bin`, and writes `clean_datasets/teddy_bear.npy`.
- `VAE.ipynb`: trains the VAE on cleaned data, saves periodic checkpoints, and updates canonical `models/Vae.model`.
- `GAN.ipynb`: trains the GAN on cleaned data, saves periodic checkpoints, and updates canonical `models/Gen.model` and `models/Dis.model`.
- `generate_bears.ipynb`: loads trained models, generates/scorers samples, and exports videos.

## Sample Generated Videos

### GAN

![Random GAN generation](assets/video_rand_gan.gif)

### VAE

![Random VAE generation](assets/video_rand_vae.gif)
