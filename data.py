from datasets import list_datasets,load_dataset,dataset_dict,Dataset
import abc

class Data(metaclass=abc.ABCMeta):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.dataset = self.load()
        self.preprocess()
        self.encoded_dataset = self.dataset.map(self.tokenize,batched=True,batch_size=None)
        
    def load(self):
        if self.name == None:
            return load_dataset(self.path)
        else:
            return load_dataset(self.path,self.name)
    
    @abc.abstractmethod
    def tokenize(self,batch):
        pass
    
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

    def preprocess(self):
        ds = self.dataset
        ds.set_format(type='pandas')
        
        df_train = ds['train'][:]
        df_train['label'] = df_train['stars'].apply(lambda x: x-1)
        
        df_validation = ds['validation'][:]
        df_validation['label'] = df_validation['stars'].apply(lambda x: x-1)
        
        df_test = ds['test'][:]
        df_test['label'] = df_test['stars'].apply(lambda x: x-1)

        ds_dict = dataset_dict.DatasetDict()
        ds_dict['train'] = Dataset.from_pandas(df_train)
        ds_dict['validation'] = Dataset.from_pandas(df_validation)
        ds_dict['test'] = Dataset.from_pandas(df_test)
        
        self.dataset = ds_dict
            
class FinancialData(Data):
    def __init__(self,tokenizer):
        self.path = 'financial_phrasebank'
        self.name = 'sentences_50agree'
        self.num_labels = 3
        super().__init__(tokenizer)
        
    def tokenize(self,batch):
        return self.tokenizer(batch["sentence"],padding=True,truncation=True)
    
    def preprocess(self):
        ds = self.dataset
        ds.set_format(type='pandas')
        df = ds['train'][:]
        df = df.sample(frac=1,random_state=1).copy()
        
        n_train = int(len(df)*0.8)
        n_val = int(len(df)*0.1)
        n_test = len(df) - (n_train+n_val)
        
        df_train = df[:n_train].copy()
        df_validation = df[n_train:n_train+n_val].copy()
        df_test = df[n_train+n_val:].copy()
        
        ds_dict = dataset_dict.DatasetDict()
        ds_dict['train'] = Dataset.from_pandas(df_train)
        ds_dict['validation'] = Dataset.from_pandas(df_validation)
        ds_dict['test'] = Dataset.from_pandas(df_test)
        
        self.dataset = ds_dict
        
class TweetData(Data):
    def __init__(self,tokenizer):
        self.path = 'tweet_eval'
        self.name = 'sentiment'
        self.num_labels = 3
        super().__init__(tokenizer)
            
    def tokenize(self,batch):
        return self.tokenizer(batch["text"],padding=True,truncation=True)
        