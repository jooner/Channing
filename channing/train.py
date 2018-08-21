from tim import get_words
from prep import *
from model import CBOW

from tqdm import tqdm

import torch
from torch.autograd import Variable

EMB_DIM = 50

article = get_words()
tokens = check(PreProcessor().clean(article))
# vec_txt = encode(tokens)
x, y = prep_train(tokens)
print("We out here!")
embs, w2x = w2v_dict_to_torch_emb(w2v)
print("Made it!")
cbow = CBOW(len(w2v), EMB_DIM, embs)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=0.001)
print("Almost there!")
# train


# TODO: save model with torch every few epochs
for epoch in range(50):
    losses = []
    total_loss = 0
    for context, target in tqdm(zip(x, y)):
        cbow.zero_grad()
        context = list(map(lambda w: w2x[w], context))
        log_probs = cbow(Variable(torch.LongTensor(context)))
        loss = loss_function(log_probs, Variable(
            # TODO: fix this part to be a one-hot vector
            torch.LongTensor([w2x[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print(total_loss)
    losses.append(total_loss)
