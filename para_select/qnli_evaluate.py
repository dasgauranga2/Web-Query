import torch
import torch.nn as nn

import random

from transformers import BertTokenizer, BertModel
from torchtext import data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

len(tokenizer.vocab)

tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')

indexes = tokenizer.convert_tokens_to_ids(tokens)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

class BERTSentiment(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.output = nn.Linear(embedding_dim, 1)
                
    def forward(self, text):
                
        embedded = self.bert(text)[0]
        
        logits = embedded[:,0,:]
        
        final_logits = self.output(logits)
        
        return final_logits
    
models = []

for i in range(2):
    new_model = BERTSentiment().cuda()
    new_model.load_state_dict(torch.load(f'saved_models/qnli_model_{i+1}.pt'))
    models.append(new_model)
    
def model_ensemble_output(models, tensor):
    
    outputs = []
    weights = [1,1]
    for i in range(len(models)):
        outputs.append(weights[i] * models[i](tensor))
        
    return sum(outputs)

def predict_sentiment(models, tokenizer, sentence):
    for m in models:
        m.eval()
    
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).cuda()
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model_ensemble_output(models, tensor))
    return prediction.item()

def para_scores(para, question):
    lines = para.split('.')
    
    print(f"QUESTION : {question}\n")
    for line in lines[:-1]:
        input_text = question + ' [SEP] ' + line.lower()
        probs = predict_sentiment(models, tokenizer, input_text)

        print(f"LINE : {line}\nPRED : {probs:.4f}\n")
                
#question = 'how many people have died from coronavirus?'

#para = 'Coronavirus disease 2019 (COVID‑19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). It was first identified in December 2019 in Wuhan, Hubei, China, and has resulted in an ongoing pandemic. As of 4 September 2020, more than 26 million cases have been reported across 188 countries and territories, resulting in more than 867,000 deaths. More than 17 million people have recovered.'
#para = 'Common symptoms include fever, cough, fatigue, shortness of breath or breathing difficulties, and loss of smell and taste. While most people have mild symptoms, some people develop acute respiratory distress syndrome (ARDS) possibly precipitated by cytokine storm,[14] multi-organ failure, septic shock, and blood clots. The time from exposure to onset of symptoms is typically around five days, but may range from two to fourteen days.'
#para = 'The virus is spread primarily via nose and mouth secretions including small droplets produced by coughing, sneezing, and talking. The droplets usually do not travel through air over long distances. However, those standing in close proximity may inhale these droplets and become infected. People may also become infected by touching a contaminated surface and then touching their face. The transmission may also occur through smaller droplets that are able to stay suspended in the air for longer periods of time in enclosed spaces. It is most contagious during the first three days after the onset of symptoms, although spread is possible before symptoms appear, and from people who do not show symptoms. The standard method of diagnosis is by real-time reverse transcription polymerase chain reaction (rRT-PCR) from a nasopharyngeal swab. Chest CT imaging may also be helpful for diagnosis in individuals where there is a high suspicion of infection based on symptoms and risk factors; however, guidelines do not recommend using CT imaging for routine screening.'
#para = 'Recommended measures to prevent infection include frequent hand washing, maintaining physical distance from others (especially from those with symptoms), quarantine (especially for those with symptoms), covering coughs, and keeping unwashed hands away from the face. The use of cloth face coverings such as a scarf or a bandana has been recommended by health officials in public settings to minimise the risk of transmissions, with some authorities requiring their use. Health officials also stated that medical-grade face masks, such as N95 masks, should be used only by healthcare workers, first responders, and those who directly care for infected individuals.'

#para_scores(para, question)