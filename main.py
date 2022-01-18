from nltk.tokenize import word_tokenize
from collections import defaultdict

from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer, BertModel
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize

from nltk.corpus import brown

from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding, CrossEntropyLoss
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

import csv

torch.autograd.set_detect_anomaly(True)

print("Welp I am too tired to do anything so here goes nothing.")
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hyperparametre_defaults = dict(
    actor_lr = 1e-5, 
    critic_lr = 1e-5,
    max_length = 200,
    epochs = 10000,
    train_split = 0.90,
    batch_size = 4,
    # batch_size = 24,
    actor_model = None,
    critic_model = None
    # critic_model = None
)

# run = wandb.init(project="drumpf", entity="jemoka", config=hyperparametre_defaults)
run = wandb.init(project="drumpf", entity="jemoka", config=hyperparametre_defaults, mode="disabled")
config = wandb.config

# A few random utility function
np2tens = lambda x: torch.tensor(x)

def find(tensor, value, axis=0):
    x = tensor==2
    nonz = (x > 0)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis).indices

# similarity_model = SentenceTransformer('stsb-bert-base')
# def semantic_similarity(a,b):
#     global similarity_model
#     a,b = similarity_model.encode([a,b])
#     return util.pytorch_cos_sim(a,b)[0][0].item()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

print("Setting up constants.")
ACTOR_LR = config.actor_lr
CRITIC_LR = config.critic_lr
MAX_LENGTH = config.max_length
TRAIN_SPLIT = config.train_split
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
ACTOR = config.actor_model
CRITIC = config.critic_model

print("Getting data.")

data_raw = []
with open("./data/coordinance.csv") as df:
    reader = csv.reader(df)
    next(reader)

    for row in tqdm(reader):
                        # ah yes good code
        data_raw.append([eval(row[0]), float(row[1])])

 data_raw = list(zip(data_text, data_score))

random.shuffle(data_raw)

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
pretrain = 3 
critic_model = None
if CRITIC:
    pretrain = 0
    critic_model = torch.load(f"./models/critic/{CRITIC}")
else:
    bert_config = BertConfig.from_pretrained("bert-base-cased")
    bert_config.num_labels = 1
    critic_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    critic_model = BertForSequenceClassification(bert_config)

print("Creating optimizers and moving models.")
bart_model.to(DEVICE)
bart_model.train()
actor_optim = Adam(bart_model.parameters(), lr=ACTOR_LR)

critic_model.to(DEVICE)
critic_model.train()
critic_optim = Adam(critic_model.parameters(), lr=CRITIC_LR)

run.watch([bart_model, critic_model])

max_token = len(bart_tokenizer)-1

print("Defining validation...")
def accuracy(model, batch):
    # seperate in out batch
    batch_in, batch_out = zip(*batch)

    batched_texts_paired = [i[0]+"[SEP]"+i[1] for i in batch_in]
    tokenized =  critic_tokenizer(batched_texts_paired, padding="max_length", truncation=True, max_length=MAX_LENGTH*2, return_tensors="pt").to(DEVICE)

    # expand and predict
    labels = torch.unsqueeze(np2tens(batch_out), dim=1).to(DEVICE).float()
    critic_outputs = critic_model(**tokenized, labels=labels)

    # First, backprop critics' loss
    critic_loss = critic_outputs["loss"]

    # compare them!
    return critic_loss.item()

print("Starting to pre-train critic.")
for ep in range(pretrain):
    print(f"Training epoch {ep}")
    random.shuffle(data_train_batches)
    bar = tqdm(enumerate(data_train_batches), total=len(data_train_batches))
    for i, batch in bar:
        if i%20 == 0:
            run.log({"critic_val": accuracy(critic_model, random.sample(data_val_batches, 1)[0])})
        batch_texts, batch_scores = zip(*batch)
        #         # Encode each input sentence 
        # input_sentence_encoded = [bart_tokenizer.encode(i)[:MAX_LENGTH] for i in batch_texts]
        # # Pad the encoded result to the MAX_LENGTH
        # input_sentence_padded = np2tens([i + [1 for _ in range(MAX_LENGTH-len(i))] for i in input_sentence_encoded]).to(DEVICE)
        # # Mask the attention such that only non-padded values are available
        # input_sentence_mask = np2tens([[1 for _ in range(len(i))] + [0 for _ in range(MAX_LENGTH-len(i))] for i in input_sentence_encoded]).to(DEVICE)

        # one-hot encode the inputs to the model
        # input_sentences_one_hot = F.one_hot(input_sentence_padded, num_classes=max_token+1).to(DEVICE).float()
        # Calculate the relative rewards
        # critic_targets = F.one_hot(np2tens(batch_scores), num_classes=3).to(DEVICE)
        # Encode input sentences
        batched_texts_paired = [i[0]+"[SEP]"+i[1] for i in batch_texts]
        tokenized =  critic_tokenizer(batched_texts_paired, padding="max_length", truncation=True, max_length=MAX_LENGTH*2, return_tensors="pt").to(DEVICE)

        # expand and predict
        labels = torch.unsqueeze(np2tens(batch_scores), dim=1).to(DEVICE).float()
        critic_outputs = critic_model(**tokenized, labels=labels)

        # First, backprop critics' loss
        critic_loss = critic_outputs["loss"]

        critic_loss.backward()
        actor_optim.zero_grad()
        critic_optim.step() # train the critic wayy more than the actor
        critic_optim.zero_grad()

        # Log some stuff
        try: 
            run.log({"critic_loss": critic_loss.item(),
                    "reward": batch_scores[0]})
        except IsADirectoryError:
            pass

if not CRITIC:
    torch.save(critic_model, f"./models/critic/{run.name}")


def predict_on_batch(batch):
        batched_texts_paired = [i[0]+"[SEP]"+i[1] for i in batch]
        tokenized =  critic_tokenizer(batched_texts_paired, padding="max_length", truncation=True, max_length=MAX_LENGTH*2, return_tensors="pt").to(DEVICE)

        # expand and predict
        critic_outputs = critic_model(**tokenized)

        print(critic_outputs["logits"])

while True:
    a = input("1>> ")
    b = input("2>> ")
    predict_on_batch([(a,b)])

# Initialize a loss function

print("Starting to train model.")
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

        # Pass these sentences through the model, calculate the topic modeling loss
        model_outputs = bart_model(input_sentence_padded, attention_mask=input_sentence_mask, output_hidden_states=True)
        model_sentences_padded_expanded = model_outputs["logits"]

        # Select for the predicted outputs
        actions = torch.stack([torch.argmax(i, axis=1) for i in model_sentences_padded_expanded.detach()])
        # Stringify the outputs
        logits_string = [bart_tokenizer.decode(i) for i in actions]
        # Find the </s> tokens
        eos_token_pos = find(actions, 2, 1)
        # Find out the masked attention values
        output_sentence_mask = np2tens([[1 for _ in range(i)] + [0 for _ in range(MAX_LENGTH-i)] for i in eos_token_pos]).to(DEVICE)
        # Return the final string
        logits_string = [re.sub("<s>", "", i.split("</s>")[0]) for i in logits_string]
        # Return the logits probabilites
        logits_probs = torch.softmax(model_sentences_padded_expanded, dim=2)
        # Gather the logits values of the selected actions
        action_logits = model_sentences_padded_expanded.gather(2, torch.unsqueeze(actions, 2))
        # Take the log of it
        action_log_logits = torch.abs(action_logits)

        # Calculate critic outputs
        critic_output_model = critic_model(logits_probs, mask=output_sentence_mask)

        # Calculate the relative rewards
        critic_targets = np2tens([[reward(i)] for i in logits_string]).to(DEVICE)

        # Calculate losses
        model_loss_critic = (1-(critic_output_model.mean()))*5 # *5 to prevent vanishing gradients
        model_loss_topicmodeling = maskedCrossEntropy(model_sentences_padded_expanded, 
                                                      input_sentence_padded)

        # Calculate group loss
        model_loss = (model_loss_critic + model_loss_topicmodeling)

        model_loss.backward()
        critic_optim.zero_grad()
        actor_optim.step()

        critic_optim.zero_grad()
        actor_optim.zero_grad()

        # Log some stuff
        if i % 10 == 0:
            try: 
                run.log({"model_loss": model_loss.item(),
                         # "critic_loss": critic_loss.item(),
                         "model_loss_critic": model_loss_critic.item(),
                         "model_loss_topicmodeling": model_loss_topicmodeling.item(),
                         "model_loss": model_loss.item(),
                         "reward": critic_targets[0].item(),
                         "sample": wandb.Html(batch[0]+"<br />"+logits_string[0])})
            except IsADirectoryError:
                pass

        # Set model parametres
        # bar.set_description(f"model loss: {round(model_loss.item(),5)}, critic loss: {round(critic_loss.item(),5)}")
        bar.set_description(f"model loss: {round(model_loss.item(),5)}")

torch.save(bart_model, f"./models/actor/{run.name}")

