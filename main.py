import os
import models

os.environ['WANDB_DISABLED'] = "true"
    
def main():
    '''
    TODO:
    =====
    Model names to run:
    - bert-base-uncased
    - roberta-base
    - xlnet-base-cased
    
    Cased vs. uncased:
    In BERT uncased, the text has been lowercased before WordPiece tokenization step while in BERT cased, the text is same as the input text (no changes).
    '''

    
    # bert = models.TweetModel(path="bert-base-cased")
    bert = models.AmazonModel(path="bert-base-cased")
    
    bert.train()

    
    # batch_size = 64
    # output_dir = f"outputs/{bert.path}_{bert.amazon_data.path}"
    # training_args = TrainingArguments(output_dir=output_dir,
    #                                     num_train_epochs=1,
    #                                     learning_rate=2e-5,
    #                                     per_device_train_batch_size=batch_size,
    #                                     per_device_eval_batch_size=batch_size,
    #                                     weight_decay=0.01,
    #                                     evaluation_strategy="epoch",
    #                                     disable_tqdm=False,
    #                                     push_to_hub=False)
        
    # trainer = Trainer(model=bert.amazon_model, args=training_args,
    #             compute_metrics=compute_metrics,
    #             train_dataset=bert.amazon_data.encoded_dataset["train"],
    #             eval_dataset=bert.amazon_data.encoded_dataset["validation"],
    #             tokenizer=bert.tokenizer)
    # trainer.train()

    # trainer.save_model(f'trained_models/{bert.path}_{bert.amazon_data.path}')

        
        
    
if __name__ == '__main__':
    # ds = load_dataset('financial_phrasebank',name='sentences_50agree')
    
    
    # ds = load_dataset('amazon_reviews_multi',name='en')
    
    
    
    # print(ds['train'][:])#.value_counts())
    
    # df = ds['train'][:]
    # df['label'] = df['stars'].apply(lambda x: x-1)
    
    # newds = Dataset.from_pandas(df)
    # print(newds)
        
    # ds.set_format(type='pandas')
    # df
    main()
    
    # ds_dict = dataset_dict.DatasetDict()
    # print(ds_dict)