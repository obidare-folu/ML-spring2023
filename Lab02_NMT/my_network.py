import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            bidirectional=True,
        )

        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)

        embedded = self.dropout(embedded)

        encoder_states, (hidden, cell) = self.rnn(embedded)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)

        self.rnn = nn.LSTM(
            input_size=hid_dim * 2 + emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
        )

        self.energy = nn.Linear(in_features=hid_dim * 3, out_features=1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.out = nn.Linear(in_features=hid_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, encoder_states,  hidden, cell):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        sequence_length = encoder_states.shape[0]
        hidden_resized = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(
            self.energy(torch.cat((hidden_resized, encoder_states), dim=2))
        )
        attention = self.softmax(energy)

        context = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context, embedded), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.out(output).squeeze(0)

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimention instead of zero
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_states, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs
