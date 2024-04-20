# fine-tuning-foundation-model
Apply Lightweight Fine-Tuning to a Foundation Model


## Dataset
We use the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) for this excersise. We will use the dataset to fine tune the foundation model.

The data is used to take in text as input and will predict 6 emotional states.

## Foundation Model.
We will use the [microsoft/DialogRPT-updown](https://huggingface.co/microsoft/DialogRPT-updown) to predict emotional text for the given data.

For the model we intialize the model with 6 labels for each emotional state.

## Training and Evaluating

We use the Trainer library to fine-tune and evaluate our model against the dataset.

### Training
For the model, we will reduce the trainable parameters using the PEFT module.

### Evaluating 
To evaluate the model, we will load the exisiting model stored in the directory against the test dataset.


## How to fine-tune?
To fine-tune the model, clone the repository locally and run the `LightweightFineTuning.ipynb`. 
The 


## Parameters and Results
For this model, I have used the following LoRA config

### LoRA config
```
from peft import LoraConfig
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.2
)
```

### Training Parameters
```
from transformers import TrainingArguments
training_arguments = TrainingArguments(
    output_dir="./data/models",
    learning_rate=2e-5,# Set the learning rate
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="epoch", # Evaluate and save the model after each epoch
    evaluation_strategy="epoch",
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
)
```

### Training Results
Validation Dataset. (Note the model was trained twice with 5 epochs for a total of 10 epoch)
```
{
    'eval_loss': 0.384300,
    'eval_accuracy': 0.916000,
    'eval_runtime': 21.1295,
    'eval_samples_per_second': 94.655,
    'eval_steps_per_second': 23.664
}
```

Test Dataset
```
{
    'eval_loss': 0.3568718731403351,
    'eval_accuracy': 0.9115,
    'eval_runtime': 22.693,
    'eval_samples_per_second': 88.133,
    'eval_steps_per_second': 22.033
}
```
