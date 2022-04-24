from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import data
import abc
import torch
from transformers import Trainer,TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(metaclass=abc.ABCMeta):
    def __init__(self,path):
        self.path = path
        self.tokenizer = self.get_tokenizer()
        self.data = self.get_data()
        self.model = self.get_model()
        self.epochs = 1
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.batch_size = 64
    
    @abc.abstractmethod
    def get_data(self):
        pass
        
    def get_model(self):
        return (AutoModelForSequenceClassification.from_pretrained(self.path,num_labels=self.data.num_labels).to(DEVICE))
        
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.path)
        
    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=f"outputs/{self.path}_{self.data.path}",
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch",
            disable_tqdm=False,
            push_to_hub=False
        )

    def compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.get_training_arguments(),
            compute_metrics=self.compute_metrics,
            train_dataset=self.data.encoded_dataset["train"],
            eval_dataset=self.data.encoded_dataset["validation"],
            tokenizer=self.tokenizer)
        trainer.train()

    def test():
        pass
    
        #     preds_output = trainer.predict(emotions_encoded["validation"])
        #     preds_output.metrics
        #     y_preds = np.argmax(preds_output.predictions, axis=1)

class AmazonModel(Model):
    def __init__(self,path):
        super().__init__(path)
        
    def get_data(self):
        return data.AmazonData(self.tokenizer)
    
class FinancialModel(Model):
    def __init__(self,path):
        super().__init__(path)
        
    def get_data(self):
        return data.FinancialData(self.tokenizer)
    
class TweetModel(Model):
    def __init__(self,path):
        super().__init__(path)
        
    def get_data(self):
        return data.TweetData(self.tokenizer)