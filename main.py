import os
import models
from datasets import list_datasets,load_dataset,Dataset,dataset_dict
from transformers import AutoModelForSequenceClassification,AutoModel

os.environ['WANDB_DISABLED'] = "true"
    

class TweetPipeline:
    def __init__(self):
        self.bert = models.TweetModel(path="bert-base-cased")
        self.roberta = models.TweetModel(path="roberta-base")
        # self.xlnet = 
    
def test():
    trainer = AutoModelForSequenceClassification.from_pretrained(f'trained_models/{self.path}_{self.data.path}')
    print(trainer)
    
        #     preds_output = trainer.predict(emotions_encoded["validation"])
        #     preds_output.metrics
        #     y_preds = np.argmax(preds_output.predictions, axis=1)
    
def main():
    
    bert_tweet = models.TweetModel(path="bert-base-cased")
    
    bert_tweet.train()
    
    
    
    # roberta_tweet = models.TweetModel(path="roberta-base")
    # xlnet_tweet = models.TweetModel(path="xlnet-base-cased")
    
    # bert_amazon = models.AmazonModel(path="bert-base-cased")
    # roberta_amazon = models.AmazonModel(path="roberta-base")
    # xlnet_amazon = models.AmazonModel(path="xlnet-base-cased")
    
    # bert_financial = models.FinancialModel(path="bert-base-cased")
    # roberta_financial = models.FinancialModel(path="roberta-base")
    # xlnet_financial = models.FinancialModel(path="xlnet-base-cased")
    
    
    
    
    
    
    
        
    
if __name__ == '__main__':

        
    main()
    
    # # 
    # # print(ds_dict)