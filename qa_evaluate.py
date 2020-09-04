import torch
import torch.nn as nn

import torchtext
from torchtext.data import Field

from torchtext import data
from transformers import T5Tokenizer, T5Model

tokenizer = T5Tokenizer.from_pretrained('t5-small')

init_token = tokenizer.pad_token
eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = tokenizer.max_model_input_sizes['t5-small']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

SRC = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

TRG = data.Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

class T5Network(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.t5 = t5 = T5Model.from_pretrained('t5-small')
        
        self.out = nn.Linear(t5.config.to_dict()['d_model'], t5.config.to_dict()['vocab_size'])
                
    def forward(self, src, trg):
        
        embedded = self.t5(input_ids=src, decoder_input_ids=trg)
        
        output = self.out(embedded[0])
        
        return output
    
models = []

for i in range(4):
    new_model = T5Network().cuda()
    new_model.load_state_dict(torch.load(f'saved_models/marco_model_{i+1}.pt'))
    models.append(new_model)
    
for i in range(4):
    new_model = T5Network().cuda()
    new_model.load_state_dict(torch.load(f'saved_models/squad_model_{i+1}.pt'))
    models.append(new_model)
    
def model_ensemble_output(models, src_tensor, trg_tensor):
    
    outputs = []
    for i in range(len(models)):
        outputs.append(models[i](src_tensor, trg_tensor))
        
    return sum(outputs)

def translate_sentence(sentence, src_field, trg_field, models, max_len = 50):
    for m in models:
        m.eval()

    src_indexes = [init_token_idx] + sentence + [eos_token_idx]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).cuda()

    trg_indexes = [init_token_idx]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = model_ensemble_output(models, src_tensor, trg_tensor)
            
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == eos_token_idx:
            break
            
    return trg_indexes

# CONTEXT = 'The COVID‑19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 (COVID‑19), caused by severe acute respiratory syndrome coronavirus 2 (SARS‑CoV‑2). The outbreak was first identified in December 2019 in Wuhan, China. The World Health Organization declared the outbreak a Public Health Emergency of International Concern on 30 January 2020 and a pandemic on 11 March. As of 24 August 2020, more than 23.4 million cases of COVID‑19 have been reported in more than 188 countries and territories, resulting in more than 808,000 deaths; more than 15.1 million people have recovered.'
# QUERIES = ['where was the outbreak first identified ?',
#            'when was the outbreak first identified ?',
#            'how many people have died from covid-19 ?',
#            'how many people have recovered from covid-19 ?',
#            'when did the world health organization declare an emergency ?',
#            'when did the world health organization declare a pandemic ?']

def str_result(tokens):
    tokens = tokens[1:-1]
    joined = ''.join(tokens)
    sep_token = joined[0]
    split = joined.split(sep_token)
    final = ' '.join(split[1:])
    
    return final

def qa_result(context, query):
    text = "context : " + context.lower() + " query : " + query.lower()
    tokens = tokenizer.tokenize(text)

    pred_tokens = translate_sentence(tokenizer.convert_tokens_to_ids(tokens), SRC, TRG, models)
    final_result = str_result(tokenizer.convert_ids_to_tokens(pred_tokens))
    
    return final_result

# for query in QUERIES:
#     text = "context : " + CONTEXT.lower() + " query : " + query.lower()
#     tokens = tokenizer.tokenize(text)
    
#     print(f"INPUT TEXT\n{text}")

#     pred_tokens = translate_sentence(tokenizer.convert_tokens_to_ids(tokens), SRC, TRG, models)

#     print("\nPREDICTION")
#     final_result = str_result(tokenizer.convert_ids_to_tokens(pred_tokens))
#     print(final_result)
#     print('\n\n')