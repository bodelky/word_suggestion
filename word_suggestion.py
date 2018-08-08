import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time
import datetime
import pandas as pd
import os.path
import os
from pathlib import Path
from gensim import corpora


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u0020"
        u"\u2003"
        u"\xa0"
                               "]+", flags=re.UNICODE)


def wrapword(value):
    with open('single_stopwords.txt',encoding='utf8') as f:
        stop_words = f.readlines()
    stop_words = [x.strip() for x in stop_words]
    
    def _wrapword(value):
        bd = icu.BreakIterator.createWordInstance(icu.Locale('th'))
        bd.setText(value)
        start = bd.first()
        for end in bd:
            yield value[start:end]
            start = end
    
    return [word for word in _wrapword(value) if word not in stop_words and emoji_pattern.match(word) is None and word != '\n']



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def accuracy_calculate(predict, targets):
    top1_correct = 0
    topN_correct = 0
    accuracy1 = correct / len()
#     if targets
    return accuracy1, accuracy2


def generate_onehot(word_split, word_id):
    
    word_index = [[word_id[w]] for w in word_split]
    word_tensor = torch.tensor(word_index)
    y_onehot = torch.FloatTensor(len(word_split), len(word_id))
    y_onehot.zero_()
    
    y_onehot.scatter_(1, word_tensor, 1)
    y_onehot = y_onehot.view(len(word_split), 1, -1)
    return y_onehot


def generate_sentence(text_split, window_size) :
    
    for i in range(0, len(text_split)-window_size) :
        yield text_split[i:i+window_size-1] , text_split[i+1:i+window_size]
        
        
def timer(start, stop):
    hour = int((stop-start) / 3600 )
    minute = int(((stop-start) % 3600) / 60)
    sec = int((stop-start) % 60)
    return '%d hour %d min %d sec' % (hour, minute, sec)



class LSTMTagger(nn.Module):

    def __init__(self, vocab_size, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(vocab_size, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim, device='cuda'),
                torch.zeros(1, 1, self.hidden_dim, device='cuda'))

    def forward(self, onehot):
        lstm_out, self.hidden = self.lstm(onehot, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(onehot), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# open file
with open("BOT_API/api_bot_concept_26.txt",encoding='utf8') as f:
    text = json.load(f)

title_text = ""
detail_text = ""
for items in text['data']:
    if items['items'] != []:
        for item in items['items']:
            title_text += item['sentence']
            detail_text += item['detail']
            
text = title_text
text_split  = wrapword(title_text)
text_split = text_split[:1000]
unique_word = corpora.Dictionary([text_split])
word_id = unique_word.token2id


# create model
learning_rate = 0.1
Hidden = 256
model = LSTMTagger(len(word_id), Hidden, len(word_id))
model = model.cuda()
loss_function = nn.NLLLoss()
loss_function = loss_function.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
min_loss = 100
min_loss_epoch = 0
n = 5
window_size = n + 1
epoch = 300

# Train
for i in range(epoch):
    for x,y in generate_sentence(text_split, window_size):
        
        model.zero_grad()
        model.hidden = model.init_hidden()
        x = generate_onehot(x, word_id)
        x = x.cuda()
        tag_scores = model(x)
        
        y = prepare_sequence(y, word_id)
        y = y.cuda()

        loss = loss_function(tag_scores, y)
        loss.backward()
        optimizer.step()
        
    if loss < min_loss :
        min_loss = loss
        min_loss_epoch = i
        
    if i % 10 == 0:
        print(i, ' loss :', float(loss))

        
end_time = time.time()

print('\n',timer(start_time, end_time))
print('min loss', float(min_loss))
print('epoch', min_loss_epoch)


top1_correct = 0
topN_correct = 0
count = 0


# test
with torch.no_grad():
    for x, targets in generate_sentence(text_split, window_size) :
        count += 1
        model.hidden = model.init_hidden()
        x_onehot = generate_onehot(x, word_id)
        tag_scores = model(x_onehot.cuda())
        
        int_predict = tag_scores.topk(10, dim=1)[1]
        word_predict = [list(map(lambda x: unique_word.get(x.item()), num)) for num in int_predict]
        
        for inputs, word, pred in zip(x, targets, word_predict) : 
            if word == pred[0] :
                top1_correct += 1
#             else :
#                 print('in TOP 1', inputs, '|', 'Target :', word, '| Predict :', pred[0])
            if word in pred :
                topN_correct += 1
#             else:
#                 print('in TOP N', inputs, '|','Target :', word, '| Predict :', pred)

    
    accuracy_top1 = (top1_correct / (count*n)) * 100
    accuracy_topN = (topN_correct / (count*n)) * 100
    print('acc top1 :',accuracy_top1)
    print('acc topN :',accuracy_topN)
    
    
## Write to file
# df = pd.DataFrame([['LSTM', len(text_split), Hidden, epoch, learning_rate, 'Adam', float(loss), accuracy_top1, accuracy_topN, timer(start_time, end_time), '',float(min_loss), min_loss_epoch]], 
#                       columns=['Model','Total Word','Hidden','Epoch','Learning rate','Optimizer','Loss','Accuracy TOP 1','Accuracy TOP 10','Time used','','Min Loss','Epoch'])
    
        
# # Write to File
# my_file = Path("log_word_suggestion.csv")
# if my_file.exists() :
#     df.to_csv('log_word_suggestion.csv', mode='a', encoding='utf-8', header=False, index=False)
# else :
#     df.to_csv('log_word_suggestion.csv', mode='a', encoding='utf-8', header=True, index=False)