import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CONTEXT_SIZE = 2
# EMBEDDING_DIM = 10

word_to_ix = {}
ix_to_word = {}


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i 
    return index

def get_max_prob_result(input, ix_to_word):
    return ix_to_word[get_index_of_max(input)]


# This is where your code will come 
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# vocab = set(raw_text)
# vocab_size = len(vocab)
#
#
# for i, word in enumerate(vocab):
#     word_to_ix[word] = i
#     ix_to_word[i] = word
#
#
# data = []
# for i in range(2, len(raw_text) - 2):
#     context = [raw_text[i - 2], raw_text[i - 1],
#                raw_text[i + 1], raw_text[i + 2]]
#     target = raw_text[i]
#     data.append((context, target))
# # print(data[:5])



class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pre_trained=None):
        super(CBOW, self).__init__()
        if pre_trained:
            self.embeddings = pre_trained
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * 4, 128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim = -1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_embedding(self, word):
        word = Variable(torch.LongTensor([word_to_ix[word]]))
        return self.embeddings(word).view(1, -1)


# model = CBOW(vocab_size, EMBEDDING_DIM)
#
# loss_function = nn.NLLLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#
#
#
# for epoch in range(50):
#     total_loss = 0
#     for context, target in data:
#         context_vector = make_context_vector(context, word_to_ix)
#         model.zero_grad()
#         log_probs = model(context_vector)
#         loss = loss_function(log_probs, Variable(
#             torch.LongTensor([word_to_ix[target]])))
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.data
#


# TEST

# context = ['People','create','to', 'direct']
# context_vector = make_context_vector(context, word_to_ix)
# a = model(context_vector).data.numpy()
# print('Raw text: {}\n'.format(' '.join(raw_text)))
# print('Context: {}\n'.format(context))
# print('Prediction: {}'.format(get_max_prob_result(a[0], ix_to_word)))
#


















