from transformers import AutoModelForSequenceClassification,AutoModel
from transformers import AutoTokenizer
import data
import abc
import torch
from transformers import Trainer,TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(metaclass=abc.ABCMeta):
    def __init__(self,model_name,mode,epochs):
        self.model_name = model_name
        self.mode = mode
        self.testing = False
        self.tokenizer = self.get_tokenizer()
        self.data = self.get_data()
        self.set_labels()
        self.batch_size = 128
        if self.mode == 'train':
            self.model = self.get_model()
            self.epochs = epochs
            self.learning_rate = 2e-5
            self.weight_decay = 0.01
    
    @abc.abstractmethod
    def get_data(self):
        pass

    def set_labels(self):
        label_dict = {
            'tweet_eval':['negative','neutral','positive'],
            'amazon_reviews_multi':['1 star','2 stars','3 stars','4 stars','5 stars'],
            'financial_phrasebank':['negative','neutral','positive']
        }
        self.labels = label_dict[self.data.path]
        
    def get_model(self):
        return (AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.data.num_labels,
            cache_dir='cache_dir'
        ).to(DEVICE))
        
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir='cache_dir'
        )
        
    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=f"outputs/{self.model_name}_{self.data.path}",
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            evaluation_strategy='epoch',
            save_strategy='no',
            disable_tqdm=False,
            push_to_hub=False,
            dataloader_num_workers=4,
            save_total_limit=1
        )

    def save_confusion_matrix(self,y_true,y_pred):
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot(cmap="Greens", values_format=".2f", ax=ax, colorbar=False)
        plt.title(f"{self.model_name}  -  {self.data.path}")
        plt.savefig(f'images/{self.model_name}_{self.data.path}_cmatrix_{self.mode}')

    def compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="weighted")
        recall = recall_score(labels, preds,average='weighted')
        if self.testing:
            self.save_confusion_matrix(labels,preds)
        return {"accuracy": acc, "f1": f1,"recall": recall, "precision":precision,}

    def train(self,save_model=False):
        trainer = Trainer(
            model=self.model,
            args=self.get_training_arguments(),
            compute_metrics=self.compute_metrics,
            train_dataset=self.data.encoded_dataset["train"],
            eval_dataset=self.data.encoded_dataset["validation"],
            tokenizer=self.tokenizer)
        trainer.train()

        if save_model:
            trainer.save_model(
                f'trained_models/{self.model_name}_{self.data.path}'
            )
        self.testing = True
        pred = trainer.predict(self.data.encoded_dataset["test"])
        print(pred.metrics)


    def get_test_arguments(self):
        return TrainingArguments(
            output_dir=f"outputs/{self.model_name}_{self.data.path}_test",
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            disable_tqdm=False,
            push_to_hub=False,
            do_train=False,
            do_predict=True
        )

    def test(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            f'trained_models/{self.model_name}_{self.data.path}',
            cache_dir='cache_dir'
        )
        trainer = Trainer(
            model = model,
            args = self.get_test_arguments(),
            compute_metrics = self.compute_metrics
        )
        
        preds_output = trainer.predict(self.data.encoded_dataset["test"])
        print(preds_output.metrics)


class AmazonModel(Model):
    def __init__(self,model_name,mode='train',epochs=1):
        super().__init__(model_name,mode,epochs)
        
    def get_data(self):
        return data.AmazonData(self.tokenizer)
    
class FinancialModel(Model):
    def __init__(self,model_name,mode='train',epochs=1):
        super().__init__(model_name,mode,epochs)
        
    def get_data(self):
        return data.FinancialData(self.tokenizer)
    
class TweetModel(Model):
    def __init__(self,model_name,mode='train',epochs=1):
        super().__init__(model_name,mode,epochs)
        
    def get_data(self):
        return data.TweetData(self.tokenizer)
