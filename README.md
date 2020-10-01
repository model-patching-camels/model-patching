# Model Patching: Closing the Subgroup Performance Gap with Data Augmentation

## Setup

Create a Python environment and install dependencies. All our models and training are implemented in Tensorflow 2.0, and we use Weights and Biases for logging. You'll need to create a Weights and Biases account to run our code.
```bash
# Create a Conda environment
conda create -n model_patching python=3.6
conda activate model_patching

# Install dependencies
pip install -r requirements.txt
```


Download datasets from our Google Cloud Bucket and unzip,
```bash
# Download CelebA: ~1.6 GiB
wget https://storage.googleapis.com/model-patching/celeba_tfrecord_128.zip
# Download Waterbirds: ~0.4 GiB
wget https://storage.googleapis.com/model-patching/waterbirds_tfrecord_224.zip
```

For convenience, we include a release of the MNIST-Correlation dataset (see paper for details) that we created. This is intended for use in your research, and is not required to re-run our experiments on MNIST-Correlation.
```bash
wget https://storage.googleapis.com/model-patching/mnist_correlation_npy.zip
```


## Stage 1: Learning Augmentations with a Subgroup Transformation Model

For Stage 1 with CycleGAN Augmented Model Patching (CAMEL), we include configs for training CycleGAN models. Typically, we train one model per class, where the model learns transformations between the subgroups of the class. This is not necessary, and you could alternatively train e.g. a single StarGAN model for all classes and subgroups in your setting.  

#### Training from Scratch
It is not necessary to train these models to reproduce our results, and you can just reuse the augmented datasets that we provide if you want to skip this step.
```bash
# Training a single CycleGAN model for MNIST-Correlation
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/mnist-correlation/config.yaml

# Training CycleGAN models on Waterbirds
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/waterbirds/config-1.yaml
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/waterbirds/config-2.yaml

# Training CycleGAN models on CelebA-Undersampled
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/celeba/config-1.yaml
python -m augmentation.methods.cyclegan.train --config augmentation/configs/stage-1/celeba/config-2.yaml
```

All of these configs are supplied to the `augmentations/methods/cyclegan/train.py` file. You can find a template configuration file to run your own CycleGAN training experiments at `augmentation/configs/template_cyclegan_training.yaml`.

#### Reusing our CycleGANs
We provide `.tfrecord` datasets that can be used to replicate the outputs of Stage 1,
```bash
# Downloads logs for Stage-1: ~38 GiB
wget https://storage.googleapis.com/model-patching/stage-1-tfrecords.zip
```


## Stage 2: Training a Robust End-Model

For Stage 2, we include configs for training classifiers with consistency regularization and Group DRO [Sagawa et al., ICLR 2020], as well as standard ERM training. 

```bash
# Training {CAMEL, Group DRO, ERM} on {MNIST-Correlation, Waterbirds, CelebA-Undersampled}
python -m augmentation.methods.robust.train --config augmentation/configs/stage-2/{mnist-correlation,waterbirds,celeba}/{camel,gdro,erm}/config.yaml
```

These configs are supplied to the `augmentations/methods/robust/train.py` file. You can find a template configuration file to run end-model training experiments at `augmentation/configs/template_robust_training.yaml`.

We include an implementation of the Group DRO trainer as well as various consistency penalties at `augmentation/methods/robust/utils.py`. They should be easy to port over to your own codebase.
