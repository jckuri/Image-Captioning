import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # From the image at "images/encoder-decoder.png"
        # We can infer the Decoder RNN should have the following parts:
        # 1. The embeddings with dimensions: vocab_size and embed_size.
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # 2. The RNN with dimensions: embed_size, hidden_size, and num_layers.
        self.rnn = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        # 3. The linear layer with dimensions: hidden_size and vocab_size.
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # From the link https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        # We can infer the following processes:
        # 1. Generate the embedding vector.
        embeds = self.embeddings(captions[:, :-1])
        # 2. Concatenate the features vector with the embedding vector. That will be the input of the RNN.
        rnn_input = torch.cat((features.unsqueeze(dim = 1), embeds), dim = 1)
        # 3. Feed the input to the RNN.
        rnn_output, rnn_hidden = self.rnn(rnn_input)
        # 4. Linearly transform the output of the RNN and return it.
        return self.linear(rnn_output)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # We define the list of words to be collected.
        words = []
        # We iterate max_len number of times.
        for i in range(max_len):
            # We pass the inputs and states to the RNN. And we obtain its output and the new states.
            rnn_output, states = self.rnn(inputs, states)
            # Its output is passed through the linear layer to obtain the scores.
            scores = self.linear(rnn_output)
            # We get the most likely word and its probability, from the scores.
            prob, word = scores.max(2)
            # We collect the most likely word into our list of words.
            words.append(word.item())
            # The next input will be the embedding of the most likely word, to close the RNN loop.
            inputs = self.embeddings(word)
        # We return the list of words.
        return words
