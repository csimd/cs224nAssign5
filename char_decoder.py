#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        super(CharDecoder, self).__init__()
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        self.tgt_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.tgt_vocab.char2id),bias=True)
        self.decoderCharEmb = nn.Embedding(len(self.tgt_vocab.char2id),\
         char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.cel = nn.CrossEntropyLoss(ignore_index=target_vocab.char2id['<pad>'], reduction='sum')

        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embeddings = self.decoderCharEmb(input)
        rax_embed, dec_hidden = self.charDecoder(char_embeddings, dec_hidden)
        scores = self.char_output_projection(rax_embed)
        return scores, dec_hidden
        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, dec_hidden = self.forward(char_sequence[:-1], dec_hidden)
        leng, batch_sz, vocab_sz = scores.size()
        scores = scores.reshape(leng*batch_sz, vocab_sz)
        modified_chars = char_sequence[1:]
        modified_chars = modified_chars.contiguous()
        modified_chars = modified_chars.view(leng*batch_sz)
        return self.cel(scores, modified_chars)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_sz = initialStates[0].size()[1]
        hidden_sz = initialStates[0].size()[2]
        outputs = [""] * batch_sz
        current_chars = [self.tgt_vocab.char2id['{']] * batch_sz

        for t in range(max_length):
            chars_tensor = (torch.tensor(current_chars, device=device)).unsqueeze(dim=0)
            scores, initialStates = self.forward(chars_tensor, initialStates)
            scores = scores.permute(1, 0, 2)
            i = 0
            for score in scores:
                p_t = (softmax(score, dim=1)).squeeze()
                cur_ind = p_t.argmax().item()
                cur_char = self.tgt_vocab.id2char[cur_ind]
                current_chars[i] = cur_ind
                outputs[i] += cur_char
                i +=1

        decodedWords = []
        for w in outputs:
            i = w.find('}')
            if i!= -1:
                w = w[:i]
            decodedWords.append(w)

        return decodedWords

        ### END YOUR CODE
