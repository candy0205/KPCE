

from transformers import BertTokenizer, BertForQuestionAnswering
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt_MRCModel(nn.Module):
    def __init__(self, model_name, tokenizer, embed_size=768, max_ques_len=35, max_seq_len = 256):
        super(Prompt_MRCModel, self).__init__()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.topic_model = BertForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.embed_size = embed_size
        self.hidden_size = self.embed_size
        self.max_ques_len = 30
        self.max_topic_len = 5
        
        self.prompt_embeddings = torch.nn.Embedding(self.max_ques_len, self.embed_size)
        #self.topic_embeddings = torch.nn.Embedding(self.max_topic_len, self.embed_size)
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        
        #self.fc = nn.Linear(max_ques_len, max_ques_len)
        
    
    
    def generate_default_inputs(self, batch, ques_embed, device):

        input_ids = batch['input_ids']
        bz = batch['input_ids'].shape[0]
        block_flag = 1
        #print(ques_embed.size())
        #block_flag = self.tokenizer.build_inputs_with_special_tokens(block_flag_a)
        raw_embeds = self.model.bert.embeddings.word_embeddings(input_ids.to(device)).squeeze(1)
        topic_embeds = self.topic_model.bert.embeddings.word_embeddings(ques_embed.to(device)).squeeze(1)
        #print(raw_embeds.size())
        #print(topic_embeds.size())
        replace_embeds = self.prompt_embeddings(
            torch.LongTensor(list(range(self.max_ques_len))).to(device))
        replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]
        replace_embeds = self.lstm_head(replace_embeds)[0]
        replace_embeds = self.mlp_head(replace_embeds).squeeze()
        #replace_embeds = replace_embeds.squeeze(0)
        for bidx in range(bz):
            for i in range(1,31):
                raw_embeds[bidx, i, :] = replace_embeds[i-1, :]
                
        for bidx in range(bz):
            for i in range(31,36):
                raw_embeds[bidx, i, :] = topic_embeds[bidx, i-31, :]
                
        inputs = {'inputs_embeds': raw_embeds.to(device), 'attention_mask': batch['attention_mask'].squeeze(1).to(device)}
        inputs['token_type_ids'] = batch['token_type_ids'].squeeze(1).to(device)
        return inputs

    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):

        return self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          labels=labels,
                          token_type_ids=token_type_ids)

    def mlm_train_step(self, batch, ques_embed, start_positions, end_positions, device):

        inputs_prompt = self.generate_default_inputs(batch, ques_embed, device)
        bert_out = self.model(**inputs_prompt, start_positions=start_positions, end_positions=end_positions)
        return bert_out
    
    