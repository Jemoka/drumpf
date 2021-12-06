from nltk.tokenize import word_tokenize
from collections import defaultdict

from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from tqdm import tqdm
import math

import random
import re

import wandb
import json

print("Welp I am too tired to do anything so here goes nothing.")
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hyperparametre_defaults = dict(
    actor_lr = 5e-5, 
    critic_lr = 1e-6,
    max_length = 50,
    epochs = 100,
    train_split = 0.99,
    batch_size = 4,
)

run = wandb.init(project="drumpf", entity="jemoka", config=hyperparametre_defaults)
# run = wandb.init(project="drumpf", entity="jemoka", config=hyperparametre_defaults, mode="disabled")
config = wandb.config

np2tens = lambda x: torch.tensor(x)

print("Setting up constants.")
ACTOR_LR = config.actor_lr
CRITIC_LR = config.critic_lr
MAX_LENGTH = config.max_length
TRAIN_SPLIT = config.train_split
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs

print("Getting data.")
with open("./data_parsed.json", "r") as df:
    data_raw = json.load(df)

data_train = data_raw[:int(TRAIN_SPLIT*len(data_raw))]
data_val = data_raw[int(TRAIN_SPLIT*len(data_raw)):]

data_train_batches = []
for i in range(0, len(data_train), BATCH_SIZE):
    data_train_batches.append(data_train[i:i+BATCH_SIZE])

data_val_batches = []
for i in range(0, len(data_val), BATCH_SIZE):
    data_val_batches.append(data_val[i:i+BATCH_SIZE])

print("Establishing tokenizers and actor model.")
bart_config = BartConfig.from_pretrained("facebook/bart-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", config=bart_config)

print("Establishing critic model.")
class Critic(nn.Module):
    def __init__(self, vocab_size):
        super(Critic,self).__init__()

        self.d1 = nn.Linear(vocab_size, 64)
        self.d2 = nn.Linear(64, 32)
        self.flatten = nn.Flatten()
        self.d3 = nn.Linear(1600, 128)
        self.d4 = nn.Linear(128, 64)
        self.d5 = nn.Linear(64, 32)
        self.d6 = nn.Linear(32, 1)

    def forward(self,x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.flatten(x)
        x = F.relu(self.d3(x))
        x = F.relu(self.d4(x))
        x = F.relu(self.d5(x))
        x = self.d6(x)
        return x

critic_model = Critic(len(bart_tokenizer))

print("Creating optimizers and moving models.")
bart_model.to(DEVICE)
bart_model.train()
actor_optim = AdamW(bart_model.parameters(), lr=ACTOR_LR)

critic_model.to(DEVICE)
critic_model.train()
critic_optim = AdamW(critic_model.parameters(), lr=CRITIC_LR)

print("Starting to train.")
max_token = len(bart_tokenizer)-1

for ep in range(EPOCHS):
    print(f"Training epoch {ep}")

    bar = tqdm(enumerate(data_train_batches), total=len(data_train_batches))
    for i, batch in bar:

        # Generate the noise to input to the model
        input_sentences = []
        for _ in range(BATCH_SIZE):
            target_sentence = [0]
            for _ in range(random.randint(1, MAX_LENGTH-2)):
                target_sentence.append(random.randint(3, max_token))
            target_sentence.append(2)
            input_sentences.append(target_sentence)

        # Encode each input sentence 
        input_sentence_encoded = [bart_tokenizer.encode(i)[:MAX_LENGTH] for i in input_sentences]
        # Pad the encoded result to the MAX_LENGTH
        input_sentence_padded = np2tens([i + [1 for _ in range(MAX_LENGTH-len(i))] for i in input_sentence_encoded]).to(DEVICE)
        # Mask the attention such that only non-padded values are available
        input_sentence_mask = np2tens([[1 for _ in range(len(i))] + [0 for _ in range(MAX_LENGTH-len(i))] for i in input_sentence_encoded]).to(DEVICE)

        # Pass these sentences through the model
        model_sentences_padded_expanded = bart_model(input_sentence_padded, attention_mask=input_sentence_mask)["logits"]

        # Encode each output sentence 
        target_output_sentence_encoded = [bart_tokenizer.encode(i)[:MAX_LENGTH] for i in batch]
        # Pad the encoded result to the MAX_LENGTH
        target_sentence_padded = np2tens([i + [1 for _ in range(MAX_LENGTH-len(i))] for i in target_output_sentence_encoded]).to(DEVICE)
        # Expand the target sentences
        target_sentence_padded_expanded = F.one_hot(target_sentence_padded, max_token+1).float()

        # Select for the predicted outputs
        actions = torch.stack([torch.argmax(i, axis=1) for i in model_sentences_padded_expanded.detach()])
        # Stringify the outputs
        logits_string = [bart_tokenizer.decode(i) for i in actions]
        # Return the final string
        logits_string = [re.sub("<s>", "", i.split("</s>")[0]) for i in logits_string]

        # Calculate critic outputs
        critic_output_model = critic_model(F.softmax(model_sentences_padded_expanded, dim=2))
        critic_output_target = critic_model(target_sentence_padded_expanded)

        # we want to maximize loss
        critic_loss = -1*(critic_output_target-critic_output_model).mean()
        model_loss = -1*critic_output_model.mean()

        # Log some stuff
        if i % 10 == 0:
            try: 
                run.log({"model_loss": model_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "sample": wandb.Html(logits_string[0])})
            except IsADirectoryError:
                pass

        bar.set_description(f"model loss: {round(model_loss.item(),5)}, critic loss: {round(critic_loss.item(),5)}")

        # Freeze critic model to backprop model
        for param in critic_model.parameters():
            param.requires_grad = False
        model_loss.backward(retain_graph=True)

        # Unreeze critic model to backprop model
        for param in critic_model.parameters():
            param.requires_grad = True
        # Freeze bart model to backprop critic
        for param in bart_model.parameters():
            param.requires_grad = False
        critic_loss.backward()

        actor_optim.step()
        critic_optim.step()

        actor_optim.zero_grad()
        critic_optim.zero_grad()

