
# vit-Facial-Expression-Recognition

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the [FER 2013](https://www.kaggle.com/datasets/msambare/fer2013),[MMI Facial Expression Database](https://mmifacedb.eu/), and [AffectNet dataset](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) datasets.
It achieves the following results on the evaluation set:
- Loss: 0.4503
- Accuracy: 0.8434

## Model description

The vit-face-expression model is a Vision Transformer fine-tuned for the task of facial emotion recognition. 

It is trained on the FER2013, MMI facial Expression, and AffectNet datasets, which consist of facial images categorized into seven different emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral


## Data Preprocessing

The input images are preprocessed before being fed into the model. The preprocessing steps include:
- **Resizing:** Images are resized to the specified input size.
- **Normalization:** Pixel values are normalized to a specific range.
- **Data Augmentation:** Random transformations such as rotations, flips, and zooms are applied to augment the training dataset.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 256
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 1000
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.3548        | 0.17  | 100  | 0.8024          | 0.7418   |
| 1.047         | 0.34  | 200  | 0.6823          | 0.7653   |
| 0.9398        | 0.51  | 300  | 0.6264          | 0.7827   |
| 0.8618        | 0.67  | 400  | 0.5857          | 0.7973   |
| 0.8363        | 0.84  | 500  | 0.5532          | 0.8104   |
| 0.8018        | 1.01  | 600  | 0.5279          | 0.8196   |
| 0.7567        | 1.18  | 700  | 0.5110          | 0.8248   |
| 0.7521        | 1.35  | 800  | 0.5080          | 0.8259   |
| 0.741         | 1.52  | 900  | 0.5002          | 0.8271   |
| 0.7229        | 1.69  | 1000 | 0.4967          | 0.8263   |
| 0.7157        | 1.85  | 1100 | 0.4876          | 0.8326   |
| 0.6868        | 2.02  | 1200 | 0.4836          | 0.8342   |
| 0.6605        | 2.19  | 1300 | 0.4711          | 0.8384   |
| 0.6449        | 2.36  | 1400 | 0.4608          | 0.8406   |
| 0.6085        | 2.53  | 1500 | 0.4503          | 0.8434   |
| 0.6178        | 2.7   | 1600 | 0.4434          | 0.8478   |
| 0.6166        | 2.87  | 1700 | 0.4420          | 0.8486   |


### Framework versions

- Transformers 4.36.0
- Pytorch 2.0.0
- Datasets 2.1.0
- Tokenizers 0.15.0
