from nltk.tokenize import word_tokenize
from collections import defaultdict

from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize

from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.optim import RMSprop

import statistics
import numpy as np
from tqdm import tqdm
import math

import random
import re

import wandb
import json

torch.autograd.set_detect_anomaly(True)

print("Welp I am too tired to do anything so here goes nothing.")
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hyperparametre_defaults = dict(
    actor_lr = 1e-5, 
    critic_lr = 1e-5,
    max_length = 50,
    epochs = 10000,
    train_split = 0.99,
    batch_size = 28,
    descriminator_accumulate = 16
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
ACCUMULATE = config.descriminator_accumulate

print("Getting data.")
with open("./data_parsed.json", "r") as df:
    data_raw = json.load(df)

print("Setting up reward.")
detokenizer = TreebankWordDetokenizer()
dataset_sents = [detokenizer.detokenize(i) for i in data_raw]
dataset_words = [j.lower() for i in data_raw for j in word_tokenize(i)]
usage = defaultdict(int)

# We count the usage of each word
for word in dataset_words:
    usage[word] += 1

# We get the mean and stddev usage and normalize the
# usages by them
usage_mean = statistics.mean(usage.values())
usage_stddev = statistics.stdev(usage.values())

# Finally, we normalize every value based on this
# difference. We encourage results to be higher
# than mean so we don't abs value. Also we will
# take the sigmoid of the output to normalize it
# between 0 and 1

for key in usage.keys():
    usage[key] = np.tanh((usage[key]-usage_mean)/usage_stddev)

# Overall simplification reward
def reward(src):
    words_src = [i.lower() for i in word_tokenize(src)]

    try: 
        usage_src = sum([usage[i] for i in words_src])/len(words_src)
    except ZeroDivisionError:
        usage_src = 0

    return np.tanh(usage_src)

print("Setting up dataset.")
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

        self.process = nn.Linear(vocab_size, 512)
        encoderLayer = nn.TransformerEncoderLayer(512, 8)
        self.encoder = nn.TransformerEncoder(encoderLayer, 4)

        self.output = nn.Linear(512, 1)

    def forward(self,x):
        x = self.process(x)
        encoded = self.encoder(x.transpose(0,1))
        x = self.output(encoded)
        return encoded

critic_model = Critic(len(bart_tokenizer))

print("Creating optimizers and moving models.")
bart_model.to(DEVICE)
bart_model.train()
actor_optim = RMSprop(bart_model.parameters(), lr=ACTOR_LR)

critic_model.to(DEVICE)
critic_model.train()
critic_optim = RMSprop(critic_model.parameters(), lr=CRITIC_LR)

run.watch([bart_model, critic_model])

print("Starting to train.")
max_token = len(bart_tokenizer)-1

for ep in range(EPOCHS):
    print(f"Training epoch {ep}")

    bar = tqdm(enumerate(data_train_batches), total=len(data_train_batches))
    for i, batch in bar:
        # Encode each input sentence 
        input_sentence_encoded = [bart_tokenizer.encode(i)[:MAX_LENGTH] for i in batch]
        # Pad the encoded result to the MAX_LENGTH
        input_sentence_padded = np2tens([i + [1 for _ in range(MAX_LENGTH-len(i))] for i in input_sentence_encoded]).to(DEVICE)
        # Mask the attention such that only non-padded values are available
        input_sentence_mask = np2tens([[1 for _ in range(len(i))] + [0 for _ in range(MAX_LENGTH-len(i))] for i in input_sentence_encoded]).to(DEVICE)

        # Pass these sentences through the model
        model_sentences_padded_expanded = bart_model(input_sentence_padded, attention_mask=input_sentence_mask)["logits"]

        # Select for the predicted outputs
        actions = torch.stack([torch.argmax(i, axis=1) for i in model_sentences_padded_expanded.detach()])
        # Stringify the outputs
        logits_string = [bart_tokenizer.decode(i) for i in actions]
        # Return the final string
        logits_string = [re.sub("<s>", "", i.split("</s>")[0]) for i in logits_string]

        # Calculate critic outputs
        critic_output_model = critic_model(model_sentences_padded_expanded)

        # Calculate the relative rewards
        critic_targets = np2tens([[reward(i)] for i in logits_string]).to(DEVICE)

        # First, backprop critics' loss
        critic_loss = ((critic_targets-critic_output_model)**2).mean()
        critic_loss.backward(retain_graph=True)
        actor_optim.zero_grad()

        # Then, backprop the model's loss
        model_loss = -1*(critic_output_model.mean())
        model_loss.backward()
        critic_optim.zero_grad()

        # Zero the dangling critic loss
        critic_optim.step()
        if i % ACCUMULATE == 0:
            actor_optim.step() # train the critic wayy more than the actor

        critic_optim.zero_grad()
        actor_optim.zero_grad()

        # Clip the weights
        for p in critic_model.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Log some stuff
        if i % 10 == 0:
            try: 
                run.log({"model_loss": model_loss.item(),
                            "critic_loss": critic_loss.item(),
                            "reward": critic_targets[0].item(),
                            "sample": wandb.Html(logits_string[0])})
            except IsADirectoryError:
                pass

        # Set model parametres
        bar.set_description(f"model loss: {round(model_loss.item(),5)}, critic loss: {round(critic_loss.item(),5)}")
