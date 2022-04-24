from datasets import list_datasets,load_dataset,dataset_dict,Dataset
import abc

class Data(metaclass=abc.ABCMeta):
    '''
    
    '''
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.dataset = self.load()
        # self.dataset.set_format(type='pandas')
        # print(self.dataset['train']['review_body'])
        
        # self.preprocess()
        self.encoded_dataset = self.dataset.map(self.tokenize,batched=True,batch_size=None)
        
    def load(self):
        if self.name == None:
            return load_dataset(self.path)
        else:
            return load_dataset(self.path,self.name)
    
    @abc.abstractmethod
    def tokenize(self,batch):
        pass
    
    # @abc.abstractmethod
    def preprocess(self):
        pass
        
class AmazonData(Data):
    def __init__(self,tokenizer):
        self.path = 'amazon_reviews_multi'
        self.name = 'en'
        self.num_labels = 5
        super().__init__(tokenizer)
        
    def tokenize(self,batch):
        return self.tokenizer(batch['review_body'],padding=True,truncation=True)

    # def preprocess(self):
    #     tmp_ds = self.dataset
    #     tmp_ds.set_format(type='pandas')
    #     for dataset_type in ['train','validation','test']:
    #         pass
        
class FinancialData(Data):
    def __init__(self,tokenizer,path,num_labels,name=None):
        super().__init__(tokenizer,path,num_labels,name)
        
    def preprocess(self):
        print()
        
class TweetData(Data):
    def __init__(self,tokenizer):
        self.path = 'tweet_eval'
        self.name = 'sentiment'
        self.num_labels = 3
        super().__init__(tokenizer)
            
    def tokenize(self,batch):
        return self.tokenizer(batch["text"],padding=True,truncation=True)
            
    def preprocess(self):
        print()
        