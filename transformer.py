import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from datasets import load_dataset
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy

import time


seed = 47 
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device="cuda"

seq_len=128
embed_dim=1024 
num_heads=8
feed_forward_ratio=4
batch_size=20
temp=1
lr=1e-4
reference_model_update_every=5
num_epochs=2000
save_weight = True
max_data_len=10**4
num_pred_tokens=10
model_name = 'Model_'+str(seed)+'_'+str(embed_dim)+'_'+str(num_epochs)+'_'+str(max_data_len)+'.pth'

print(f"seq_len: {seq_len}")
print(f"embed_dim: {embed_dim}")
print(f"num_heads: {num_heads}")
print(f"feed_forward_ratio: {feed_forward_ratio}")
print(f"batch_size: {batch_size}")
print(f"temp: {temp}")
print(f"lr: {lr}")
print(f"reference_model_update_every: {reference_model_update_every}")
print(f"num_epochs: {num_epochs}")
print(f"max_data_len: {max_data_len}")
print(f"num_pred_tokens: {num_pred_tokens}")
print(f"model_name: {model_name}")


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim,num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self,q,k,v,mask=None): # q,k,v has shape [batch_size,seq_len,embed_dim]
        batch_size=q.shape[0]
        seq_len = q.shape[1]
        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim).permute(0,2,1,3) # [batch_size,num_heads,seq_len,head_dim]
        kT = k.reshape(batch_size,seq_len,self.num_heads,self.head_dim).permute(0,2,3,1) # [batch_size,num_heads,head_dim,seq_len]
        v = v.reshape(batch_size,seq_len,self.num_heads,self.head_dim).permute(0,2,1,3) # [batch_size,num_heads,seq_len,head_dim]
        attention_logits=q@kT/torch.sqrt(torch.tensor(self.embed_dim)) # [batch_size,num_heads,seq_len,seq_len]
        if mask is not None:
            attention_logits=attention_logits.masked_fill(mask==0, -torch.inf)
        attn_weights=F.softmax(attention_logits, dim=-1)
        #print(f"attn_weights has shape {attn_weights.shape}")
        #print(f"v has shape {v.shape}")
        atten_values=attn_weights@v # [batch_size,num_heads,seq_len,head_dim]
        return atten_values.permute(0,2,1,3).reshape(batch_size,seq_len,self.embed_dim), attn_weights



class positional_encoding(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
    
    def forward(self,x): # x has shape [batch_size, seq_len, seq_len]
        #x=F.one_hot(x,num_classes=self.embed_dim).float()
        batch_size, seq_len, embed_dim=x.shape
        pe=torch.arange(0,seq_len).unsqueeze(1) # [seq_len,1]
        embed=torch.arange(embed_dim)
        embed1=torch.where(embed%2==0,0,1)*torch.sin(pe*(100**(-embed/embed_dim)).unsqueeze(0)) # [seq_len,embed_dim]
        embed2=torch.where(embed%2==0,1,0)*torch.sin(pe*(100**(-embed/embed_dim)).unsqueeze(0)) # [seq_len,embed_dim]
        pe_embed=(embed1+embed2).unsqueeze(0).repeat(batch_size,1,1).to(device) # [batch_size,seq_len,embed_dim]
        
        return x+pe_embed



# load dataset and tokenizer from Huggingface
dataset_name="stanfordnlp/SHP"
tokenizer_name="gpt2"
dataset=load_dataset(dataset_name, split="train", trust_remote_code=True)
dataset=dataset.select(range(min(max_data_len,len(dataset)))) # select first max_data_len samples

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token
vocab_size=tokenizer.vocab_size

# print(dir(tokenizer))  # all the methods and attributes of the tokenizer
# print(f"max_len for the tokenizer: {tokenizer.model_max_length}")

sample_data=next(iter(dataset)) # it is a continuous text
# for item in sample_data.keys():
    # print(item) # to see the keys - here only key is "text"
# print(f"example history: {sample_data['history']}")
# print(f"example human_ref_A: {sample_data['human_ref_A']}")
# print(f"example human_ref_B: {sample_data['human_ref_B']}")
# print(f"example score_A: {sample_data['score_A']}")
# print(f"example score_B: {sample_data['score_B']}")


class tokenized_text_dataset(data.Dataset):
    def __init__(self, dataset,tokenizer, seq_len):
        super().__init__()
        self.dataset=dataset
        self.tokenizer=tokenizer
        self.seq_len=seq_len
        self.tokenized_dataset=[]
        for item in dataset: 
            
            history=item["history"]
            human_ref_A=item["human_ref_A"]
            human_ref_B=item["human_ref_B"]
            score_A=item["score_A"]
            score_B=item["score_B"]
            if score_A>score_B:
                chosen=human_ref_A
                rejected=human_ref_B
            else:
                chosen=human_ref_B
                rejected=human_ref_A
        
            chosen_text=history+chosen
            rejected_text=history+rejected
            prompt_tokens=self.tokenizer(history,truncation=True, max_length=tokenizer.model_max_length, padding="max_length")["input_ids"]
            chosen_tokens=self.tokenizer(chosen_text,truncation=True, max_length=tokenizer.model_max_length, padding="max_length")["input_ids"] 
            rejected_tokens=self.tokenizer(rejected_text,truncation=True, max_length=tokenizer.model_max_length, padding="max_length")["input_ids"] 
            
            self.tokenized_dataset.append({
                "prompt":prompt_tokens,
                "chosen":chosen_tokens,
                "rejected":rejected_tokens
            })
        self.size=len(self.tokenized_dataset)

    def __len__(self):
        return self.size
    
    def __getitem__(self,idx):
        
        prompt_token=torch.tensor(self.tokenized_dataset[idx]["prompt"][:self.seq_len],dtype=torch.long)
        
        chosen_tokens=torch.tensor(self.tokenized_dataset[idx]["chosen"][:self.seq_len-num_pred_tokens+1]+[self.tokenizer.pad_token_id]*(num_pred_tokens-1),dtype=torch.long) # important to make it a tensor explicitly
        rejected_tokens=torch.tensor(self.tokenized_dataset[idx]["rejected"][:self.seq_len-num_pred_tokens+1]+[self.tokenizer.pad_token_id]*(num_pred_tokens-1), dtype=torch.long) # important to make it a tensor explicitly

        chosen_label=torch.tensor(self.tokenized_dataset[idx]["chosen"][1:self.seq_len]+[self.tokenizer.pad_token_id],dtype=torch.long) # important to make it a tensor explicitly
        rejected_label=torch.tensor(self.tokenized_dataset[idx]["rejected"][1:self.seq_len]+[self.tokenizer.pad_token_id], dtype=torch.long) # important to make it a tensor explicitly

        return chosen_tokens, rejected_tokens, chosen_label, rejected_label, prompt_token
    
text_dataset=tokenized_text_dataset(train_dataset,tokenizer,seq_len)
val_dataset=tokenized_text_dataset(val_dataset,tokenizer,seq_len)
dataloader=data.DataLoader(text_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=32)
val_dataloader=data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=32)
chosen_tokens, rejected_tokens, chosen_label, rejected_label, prompt_token=next(iter(dataloader)) # each has shape (batch_size, seq_len)

# print(f"chosen_tokens has shape {chosen_tokens.shape}")
# print(f"rejected_tokens has shape {rejected_tokens.shape}")
# print(f"chosen_label has shape {chosen_label.shape}")
# print(f"rejected_label has shape {rejected_label.shape}")


class transformer(nn.Module):
    def __init__(self, vocab_size, positional_encoding, embed_dim, num_heads, feed_forward_ratio):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feed_forward_ratio = feed_forward_ratio
        self.embed_layer=nn.Linear(embed_dim,3*embed_dim)
        self.MultiheadAttention=MultiheadAttention(embed_dim,num_heads)
        self.layer_norm=nn.LayerNorm(embed_dim)
        self.ffn_layers=nn.Sequential(
            nn.Linear(embed_dim,feed_forward_ratio*embed_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_ratio*embed_dim,embed_dim)
        )
        self.input_embedding=nn.Embedding(vocab_size, embed_dim)
        self.output_embedding=nn.Linear(embed_dim, vocab_size)
        self.positional_encoding=positional_encoding(embed_dim)
       
    
    def forward(self, x): #x has shape [batch_size,seq_len] each element is an integer in range(0,vocab_size)
        x=self.input_embedding(x) # [batch_size,seq_len] -> [batch_size,seq_len,embed_dim]
        x=self.positional_encoding(x)
        #print(f"x has shape {x.shape}")
        qkv=self.embed_layer(x) # [batch_size,seq_len,embed_dim] -> [batch_size,seq_len,3*embed_dim] 
        q, k, v= qkv.chunk(3,dim=-1) 
        batch_size, seq_len, _=x.shape
        mask=torch.tril(torch.ones(seq_len,seq_len), diagonal=0).unsqueeze(0).unsqueeze(0).repeat(batch_size,self.num_heads,1,1).to(device) # [1,1,seq_len,seq_len]
        #print(f"mask has shape {mask.shape}")
        attn_value, _=self.MultiheadAttention(q,k,v,mask)
        #print(f"attn_value has shape {attn_value.shape}")
        #print(f"x has shape {x.shape}")
        x=x+attn_value
        x=self.layer_norm(x)
        x= self.ffn_layers(x)
        x=self.layer_norm(x)
        x=self.output_embedding(x) # [batch_size,seq_len,embed_dim] -> [batch_size,seq_len,vocab_size]
        return x # [batch_size,seq_len,vocab_size] 
    

policy_model=transformer(vocab_size, positional_encoding, embed_dim, num_heads, feed_forward_ratio)
reference_model=transformer(vocab_size, positional_encoding, embed_dim, num_heads, feed_forward_ratio)
optimizer=optim.Adam(policy_model.parameters(),lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

def initialize_weights(model):
    for name, pram in model.named_parameters():
        if pram.requires_grad:
            if "weight" in name:
                if pram.dim() >= 2:
                    fan_in, fan_out = pram.shape[0], pram.shape[1]
                else:
                    fan_in = fan_out = pram.shape[0]
                pram.data.normal_(0, torch.sqrt(torch.tensor(2/(fan_in+fan_out))))
            elif "bias" in name:
                pram.data.fill_(0)

initialize_weights(policy_model)
initialize_weights(reference_model)

class loss_function(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp=temp

    def forward(self, policy_chosen_logprob, policy_rejected_logprob, reference_chosen_logprob, reference_rejected_logprob): # each one has shape [batch_size,seq_len]
        mask=(policy_chosen_logprob!=0).float()
        # print(f"-policy_chosen_logprob: {-((policy_chosen_logprob).sum(dim=-1)/mask.sum(dim=-1)).mean()}")
        # print(f"-reference_chosen_logprob: {-((reference_chosen_logprob).sum(dim=-1)/mask.sum(dim=-1)).mean()}")
        # print(f"-policy_rejected_logprob: {-((policy_rejected_logprob).sum(dim=-1)/mask.sum(dim=-1)).mean()}")
        # print(f"-reference_rejected_logprob: {-((reference_rejected_logprob).sum(dim=-1)/mask.sum(dim=-1)).mean()}")
        
        return -((policy_chosen_logprob).sum(dim=-1)/mask.sum(dim=-1)).mean() # T0
        # return -((policy_chosen_logprob-(torch.exp(policy_chosen_logprob)-1)).sum(dim=-1)/mask.sum(dim=-1)).mean()  #T1
        # return -(((policy_chosen_logprob-((torch.exp(policy_chosen_logprob)-1)-0.5*(torch.exp(policy_chosen_logprob)-1)**2))).sum(dim=-1)/mask.sum(dim=-1)).mean() #T2
     
        # return -(F.logsigmoid(self.temp*((policy_chosen_logprob-reference_chosen_logprob)-(policy_rejected_logprob-reference_rejected_logprob))).sum(dim=-1)/mask.sum(dim=-1)).mean() # [batch_size,seq_len] # spin
        
class validation_loss_function(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp=temp

    def forward(self, policy_chosen_logprob, chosen_label): # each one has shape [batch_size,seq_len]
        mask=(chosen_label!=0).float()
        return -((policy_chosen_logprob).sum(dim=-1)/mask.sum(dim=-1)).mean()
        
    
class calculate_logprob(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels, pad_token_id): # logits has shape [batch_size,seq_len,vocab_size], labels has shape [batch_size,seq_len]
        #batch_size, seq_len, vocab_size=logits.shape
        log_prob=F.log_softmax(logits, dim=-1) # [batch_size,seq_len,vocab_size]
        required_log_prob=torch.gather(log_prob,-1,labels.unsqueeze(-1)).squeeze(-1) # [batch_size,seq_len,vocab_size] -> [batch_size,seq_len]
        mask=(labels!=pad_token_id).float() # [batch_size,seq_len]
        log_prob=required_log_prob*mask # [batch_size,seq_len]
        return log_prob # [batch_size, seq_len]

calculate_logprob=calculate_logprob()
loss_function=loss_function(temp)
validation_loss_function=validation_loss_function(temp)

if os.path.exists(model_name):
    policy_model.load_state_dict(torch.load(model_name))
    reference_model.load_state_dict(torch.load(model_name))
    print(f"Loaded model from {model_name}")
else:
    print(f"Model {model_name} does not exist")

class trainer(nn.Module):
    def __init__(self, policy_model, reference_model , optimizer,scheduler, loss_function, validation_loss_function, dataloader, val_dataloader, positional_encoding, tokenizer, embed_dim, vocab_size, num_epochs, reference_model_update_every, model_name, save_weight):
        super().__init__()
        self.policy_model=policy_model
        self.reference_model=reference_model
        with torch.no_grad():
            for param in self.policy_model.parameters():
                noise = torch.randn_like(param) * 0.1  # add small noise
                param.add_(noise)
        
        self.optimizer=optimizer
        self.loss_function=loss_function
        self.validation_loss_function=validation_loss_function
        self.dataloader=dataloader
        self.val_dataloader=val_dataloader
        self.num_epochs=num_epochs
        self.positional_encoding=positional_encoding(embed_dim)
        self.scheduler=scheduler
        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.tokenizer=tokenizer
        self.reference_model_update_every=reference_model_update_every
        self.model_name=model_name
        self.save_weight=save_weight

    def validate(self):
        self.policy_model.eval()
        val_loss = 0
        with torch.no_grad():
            for chosen_tokens, rejected_tokens, chosen_label, rejected_label, _ in self.val_dataloader:
                chosen_tokens=chosen_tokens.to(device)
                rejected_tokens=rejected_tokens.to(device)
                chosen_label=chosen_label.to(device)
                rejected_label=rejected_label.to(device)
                
                policy_chosen_logits=self.policy_model(chosen_tokens)
                
                policy_chosen_logprob=calculate_logprob(policy_chosen_logits, chosen_label, self.tokenizer.pad_token_id)

                loss=self.validation_loss_function(policy_chosen_logprob, chosen_label)
                val_loss += loss.item()
        
        return val_loss / len(self.val_dataloader)

    def forward(self):
        self.policy_model=self.policy_model.to(device)
        self.policy_model.train()
        self.reference_model=self.reference_model.to(device)
        self.reference_model.eval()
        
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            # Training loop
            self.policy_model.train()
            epoch_loss=0
            # for chosen_tokens, rejected_tokens, chosen_label, rejected_label, prompt_token in self.dataloader:
            for chosen_tokens, _, chosen_label, _, prompt_token in self.dataloader:
                self.optimizer.zero_grad()
                chosen_tokens=chosen_tokens.to(device)
                # rejected_tokens=rejected_tokens.to(device)
                chosen_label=chosen_label.to(device)
                # rejected_label=rejected_label.to(device)
                prompt_token=prompt_token.to(device)
                with torch.no_grad():
                    ref_model_output = self.reference_model(chosen_tokens)
                    rejected_tokens = torch.argmax(ref_model_output, dim=-1).to(device)
                    rejected_tokens = torch.cat((prompt_token[:,0].unsqueeze(1), rejected_tokens[:,:-1]), dim=-1)
                    # print(f"rejected_tokens has shape {rejected_tokens.shape}")
                    # print(f"ref_model_output has shape {ref_model_output.shape}")
                    rejected_label = torch.cat((rejected_tokens[:, 1:], torch.tensor([self.tokenizer.pad_token_id]).unsqueeze(0).repeat(rejected_tokens.shape[0], 1).to(device)), dim=-1).to(device)
                
                policy_chosen_logits=self.policy_model(chosen_tokens)
                policy_rejected_logits=self.policy_model(rejected_tokens)
                with torch.no_grad():
                    reference_chosen_logits=self.reference_model(chosen_tokens)
                    reference_rejected_logits=self.reference_model(rejected_tokens)
                
                policy_chosen_logprob=calculate_logprob(policy_chosen_logits, chosen_label, self.tokenizer.pad_token_id)
                policy_rejected_logprob=calculate_logprob(policy_rejected_logits, rejected_label, self.tokenizer.pad_token_id)
                reference_chosen_logprob=calculate_logprob(reference_chosen_logits, chosen_label, self.tokenizer.pad_token_id)
                reference_rejected_logprob=calculate_logprob(reference_rejected_logits, rejected_label, self.tokenizer.pad_token_id)

                loss=self.loss_function(policy_chosen_logprob, policy_rejected_logprob, reference_chosen_logprob, reference_rejected_logprob)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=10)
                self.optimizer.step()
                epoch_loss+=loss.item()
            
            # Validation loop
            val_loss = self.validate()
            
            self.scheduler.step()
            if epoch%self.reference_model_update_every==0:
                self.reference_model.load_state_dict(self.policy_model.state_dict())
            
            print(f"Epoch: {epoch+1}, train loss: {epoch_loss/len(self.dataloader):.4f}, val loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(self.policy_model.state_dict(), 'best_model.pth')
                if self.save_weight:
                    torch.save(self.policy_model.state_dict(), self.model_name)


trainer=trainer(policy_model,reference_model, optimizer,scheduler, loss_function, validation_loss_function,
                dataloader, val_dataloader, positional_encoding,tokenizer, embed_dim, vocab_size, 
                num_epochs,reference_model_update_every, model_name, save_weight)
trainer()

