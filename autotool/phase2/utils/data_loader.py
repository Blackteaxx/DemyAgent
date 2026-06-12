import torch 
from transformers import AutoTokenizer, AutoModel
import os
import sys
import gc

current_dir = os.getcwd()
data_path = os.path.join(current_dir, "data")
sys.path.append(data_path)
os.environ['HF_HOME'] = os.path.join(current_dir, "HF_CACHE")


from phase2_train.utils.data_utils import (
    AGNewsProcessor,
    AmazonProcessor,
    BaseProcessor,
    SST2Processor,
    TRECProcessor,
    BB_Math_Processor
)

PROCESSORS = {
    "sst-2": SST2Processor,
    "agnews": AGNewsProcessor,
    "trec": TRECProcessor,
    "amazon": AmazonProcessor,
}

from llm import LLMWrapper


###############################################################################################
###############################################################################################
###############################################################################################


def row_wise_l2_norm(exemplars: torch.Tensor) -> torch.Tensor:
    n_ex, H, W = exemplars.shape
    with torch.no_grad():
        row_norms = torch.norm(exemplars, dim=-1, keepdim=True).clamp(min=1e-9)
        return exemplars / row_norms


def entire_exemplar_l2_norm(exemplars: torch.Tensor) -> torch.Tensor:
    n_ex, H, W = exemplars.shape
    with torch.no_grad():
        norms = torch.norm(exemplars.view(n_ex, -1), dim=-1, keepdim=True)  # (#ex,1)
        norms = norms.reshape(n_ex,1,1).clamp(min=1e-9)
        
        return exemplars / norms


def global_mean_std_norm(exemplars: torch.Tensor) -> torch.Tensor:
    mean_ = exemplars.mean()
    std_  = exemplars.std().clamp(min=1e-9)
    return (exemplars - mean_) / std_

exemplar_norm_function = row_wise_l2_norm


###############################################################################################



###############################################################################################
###############################################################################################

class DataLoader:
    def __init__(self, data_name="agnews", seed=42, embed_max_len=500, 
                 train_count=50, val_count=50, test_count=1000, device='cuda', model_name = 'gpt2-medium',
                 embed_with_LLM_flag=False, multi_gpu_flag=True):
        self.seed = seed
        self.data_name = data_name
        self.embed_max_len = embed_max_len
        self.train_count, self.val_count, self.test_count = train_count, val_count, test_count
        self.device = device
        
        # Embedding model
        if embed_with_LLM_flag:
            self.embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if multi_gpu_flag:
                self.embed_language_model = AutoModel.from_pretrained(model_name, device_map='auto')
            else:
                self.embed_language_model = AutoModel.from_pretrained(model_name)
            #
            self.embed_tokenizer.padding_side = "right"
            self.embed_tokenizer.pad_token_id = self.embed_tokenizer.eos_token_id
            self.embed_language_model.config.pad_token_id = self.embed_language_model.config.eos_token_id

            for param in self.embed_language_model.parameters():
                param.requires_grad = False
            #   
            self.embed_language_model.eval()
        else:
            self.embed_tokenizer = AutoTokenizer.from_pretrained('GPT2')
            if multi_gpu_flag:
                self.embed_language_model = AutoModel.from_pretrained('GPT2', device_map='auto')
            else:
                self.embed_language_model = AutoModel.from_pretrained('GPT2')
            #
            self.embed_tokenizer.padding_side = "right"
            self.embed_tokenizer.pad_token_id = self.embed_tokenizer.eos_token_id
            self.embed_language_model.config.pad_token_id = self.embed_language_model.config.eos_token_id

            for param in self.embed_language_model.parameters():
                param.requires_grad = False
            #   
            self.embed_language_model.eval()

        # Task processor
        if data_name in ["agnews", "sst-2", "trec", "amazon"]:
            self.processor = PROCESSORS[data_name](seed=seed, mode='labeled')
        elif data_name in ["winowhy", "epistemic_reasoning", "hyperbaton", "timedial", "aqua"]:
            self.processor = BB_Math_Processor(dataset_name=data_name, seed=seed, mode='labeled')
        else:
            raise NotImplementedError
        # Truncating datasets
        self.data_trunc()

        # Embed training exemplars
        self.exemplars_embs, self.query_emb_list, self.test_emb_list = self.embed_data(row_norm_flag=True)
        self.H, self.W = self.exemplars_embs.shape[1], self.exemplars_embs.shape[2]

        ### Delete embedding model to free memory
        del self.embed_tokenizer
        del self.embed_language_model
        torch.cuda.empty_cache()
        gc.collect()

        # LM wrapper
        self.model_name = model_name
        self.llm = LLMWrapper(model_name, batch_size=8, calibrate=True, multi_gpu_flag=multi_gpu_flag, 
                                     **self.processor.model_kwargs)   # **proc.model_kwargs -> labels

    def data_trunc(self):
        self.processor.train_dataset = self.processor.train_dataset[:self.train_count]
        self.processor.val_dataset = self.processor.val_dataset[:self.val_count]
        self.processor.test_dataset = self.processor.test_dataset[:self.test_count]
        assert len(self.processor.train_dataset) == self.train_count and \
            len(self.processor.val_dataset) == self.val_count and len(self.processor.test_dataset) == self.test_count
        
    def embed_data(self, row_norm_flag=True):
        # Normalization
        format_sentence = self.processor.fill_train_template
        #
        training_texts = [format_sentence(sample) for sample in self.processor.train_dataset]
        exemplars_emb_list = self.embed_texts_with_bert(training_texts, self.embed_tokenizer, self.embed_language_model, 
                                                         device=self.device, max_len=self.embed_max_len, norm_flag=row_norm_flag)
        exemplars_embs = torch.stack(exemplars_emb_list, dim=0).to(self.device)
        exemplars_embs.requires_grad = False
        
        ###
        context_only_emb = self.processor.fill_representation_template
        query_embs = [context_only_emb(sample) for sample in self.processor.val_dataset]
        query_emb_list = self.embed_texts_with_bert(query_embs, self.embed_tokenizer, self.embed_language_model, 
                                                    device=self.device, max_len=self.embed_max_len, norm_flag=row_norm_flag)
        #
        test_embs = [context_only_emb(sample) for sample in self.processor.test_dataset]
        test_emb_list = self.embed_texts_with_bert(test_embs, self.embed_tokenizer, self.embed_language_model, 
                                                    device=self.device, max_len=self.embed_max_len, norm_flag=row_norm_flag)
        
        return exemplars_embs, query_emb_list, test_emb_list

    def eval_exemplars(self, exemplar_indices, eval_target_indices=None):
        
        if eval_target_indices is None:
            prompts, cali_prompts = self.processor.create_prompts(train_indices=exemplar_indices, 
                                                                  train_split='train', split='test', custom_split=None)
            llm_outputs = self.llm.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = self.processor.extract_predictions(llm_outputs, split='test', custom_split=None)
        else:
            eval_target_split = [self.processor.val_dataset[i] for i in eval_target_indices]
            prompts, cali_prompts = self.processor.create_prompts(train_indices=exemplar_indices, 
                                                                  train_split='train', split='custom', custom_split=eval_target_split)
            llm_outputs = self.llm.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = self.processor.extract_predictions(llm_outputs, split='custom', custom_split=eval_target_split)
        
        return eval_result['acc']
    
    def valid_exemplars_with_training_data(self, exemplar_indices, valid_indices=None):
        
        valid_target_split = [self.processor.val_dataset[i] for i in valid_indices]
        prompts, cali_prompts = self.processor.create_prompts(train_indices=exemplar_indices, 
                                                                train_split='train', split='custom', custom_split=valid_target_split)
        llm_outputs = self.llm.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = self.processor.extract_predictions(llm_outputs, split='custom', custom_split=valid_target_split)
        
        return eval_result['acc']
    
    def test_exemplars_with_testing_data(self, exemplar_indices, test_indices=None):
        
        test_target_split = [self.processor.test_dataset[i] for i in test_indices]
        prompts, cali_prompts = self.processor.create_prompts(train_indices=exemplar_indices, 
                                                                train_split='train', split='custom', custom_split=test_target_split)
        llm_outputs = self.llm.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = self.processor.extract_predictions(llm_outputs, split='custom', custom_split=test_target_split)
        
        return eval_result['acc']

    ####
    def pad_or_trunc(self, seq_emb, max_len, tokenizer=None, embed_model=None):
        ###
        with torch.no_grad():
            pad_token_id = tokenizer.pad_token_id
            padding_emb = embed_model.get_input_embeddings()(torch.tensor(pad_token_id).to(seq_emb.device))
        self.all_pad_max_len = padding_emb.unsqueeze(0).repeat(max_len, 1).to(seq_emb.device)

        ###
        seq_len, hidden_dim = seq_emb.shape
        if seq_len > max_len:
            return seq_emb[:max_len, :]
        elif seq_len < max_len:
            pad_len = max_len - seq_len
            #
            if tokenizer is None or embed_model is None:
                pad = torch.zeros(pad_len, hidden_dim, dtype=seq_emb.dtype, device=seq_emb.device)
            else:
                # Obtain the embedding of the pad_token
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    raise ValueError("The tokenizer does not have a pad_token defined.")
                with torch.no_grad():
                    padding_emb = embed_model.get_input_embeddings()(torch.tensor(pad_token_id).to(seq_emb.device))
                pad = padding_emb.unsqueeze(0).repeat(pad_len, 1).to(seq_emb.device)
            #
            return torch.cat([pad, seq_emb], dim=0)
        else:
            return seq_emb

    def embed_texts_with_bert(self, texts, tokenizer, bert_model, device, max_len=200, norm_flag=True):
        results = []
        with torch.no_grad():
            for txt in texts:
                enc = tokenizer(txt, return_tensors='pt', truncation=True, max_length=max_len)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = bert_model(**enc)
                    seq_emb = out.last_hidden_state[0].detach()  # shape(seq_len, hidden_dim)
                seq_emb = self.pad_or_trunc(seq_emb, max_len, tokenizer=tokenizer, embed_model=bert_model).to(self.device)
                #
                seq_emb.requires_grad = False
                #
                if norm_flag:
                    results.append(exemplar_norm_function(seq_emb.unsqueeze(0)).squeeze(0))
                else:
                    results.append(seq_emb)
        
        return results


