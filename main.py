from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import Trainer,TrainingArguments
from datasets import list_datasets,load_dataset
from sklearn.metrics import accuracy_score, f1_score
import os
import torch

os.environ['WANDB_DISABLED'] = "true"

NUM_LABELS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataSet:
    def __init__(self,tokenizer,name='emotion'):
        self.tokenizer = tokenizer
        self.name = name
        self.dataset = self.load()
        self.encoded_dataset = self.dataset.map(self.tokenize,batched=True,batch_size=None)
    
    def load(self):
        return load_dataset(self.name)

    def tokenize(self,batch):
        return self.tokenizer(batch["text"],padding=True,truncation=True)


def get_model(model_ckpt):
    return (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=NUM_LABELS).to(DEVICE))

def get_tokenizer(model_ckpt):
    return AutoTokenizer.from_pretrained(model_ckpt)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def main():
    
    
    model_ckpt = "distilbert-base-uncased"
    model = get_model(model_ckpt)
    tokenizer = get_tokenizer(model_ckpt)
    dataset = DataSet(tokenizer)
    
    batch_size = 64
    output_dir = f"outputs/{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=output_dir,
                                    num_train_epochs=1,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    push_to_hub=False)
    
    trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=dataset.encoded_dataset["train"],
                  eval_dataset=dataset.encoded_dataset["validation"],
                  tokenizer=tokenizer)
    trainer.train()
    
if __name__ == '__main__':
    main()
