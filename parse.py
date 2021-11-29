import glob
import json
from nltk.tokenize import sent_tokenize

datafiles = glob.glob("./data_raw/*.txt")

sentences_raw = []
for f in datafiles:
    with open(f, "r") as df:
        sentences_raw = sentences_raw+sent_tokenize(df.read())
    
sentences_raw = list(set(sentences_raw))
with open("./data_parsed.json", "w") as df:
    json.dump(sentences_raw, df)


