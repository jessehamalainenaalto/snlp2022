import os
import models
from datasets import list_datasets,load_dataset,Dataset,dataset_dict
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import numpy as np

os.environ['WANDB_DISABLED'] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
EPOCHS = 10

def train_bert():
    model_name = "bert-base-cased"

    bert_amazon = models.AmazonModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    bert_amazon.train(save_model=True)

    bert_financial = models.FinancialModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    bert_financial.train(save_model=True)

    bert_tweet = models.TweetModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    bert_tweet.train(save_model=True)

def train_xlnet():
    model_name = 'xlnet-base-cased'

    xlnet_amazon = models.AmazonModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    xlnet_amazon.train(save_model=True)

    xlnet_financial = models.FinancialModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    xlnet_financial.train(save_model=True)

    xlnet_tweet = models.TweetModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    xlnet_tweet.train(save_model=True)

def train_roberta():
    model_name = 'roberta-base'

    bert_amazon = models.AmazonModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    bert_amazon.train(save_model=True)

    roberta_financial = models.FinancialModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    roberta_financial.train(save_model=True)

    roberta_tweet = models.TweetModel(
        model_name=model_name,
        mode='train',
        epochs=EPOCHS)
    roberta_tweet.train(save_model=True)


def main():
    ''' Run python -m main in the snlp2022 folder'''
    train_bert()
    train_roberta()
    train_xlnet()
    
    
if __name__ == '__main__':

    main()