import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        """

        Args:
            input_seqs: (max_len, batch_size, memory_size)  // memory_size = 4
            input_lengths:

        Returns:
            outputs.transpose(0, 1): (batch, seq_len, hidden_size)
            hidden: (1, batch, hidden_size)

        """
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size() + (embedded.size(-1),))
        # (embedded.size(-1),): add a ',' to define a tuple
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(outputs)
        return outputs.transpose(0, 1), hidden


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    # encoding
    # story = data['context_arr']
    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs):
        # Forward multiple hop mechanism
        """

        Args:
            story: (batch_size, max_len + 1, memory_size)  // +1 is ['$$$$', '$$$$', '$$$$', '$$$$']
            kb_len:
                sample: [0, 0, 0, 0, 0, 0, 0, 0] (batch_size)
            conv_len:
                sample: [45, 16, 15, 15, 2, 1, 1, 1] (batch_size, conv_len)
            hidden: (1, batch, hidden_size)
            dh_outputs: (batch, seq_len, hidden_size)

        Returns:
            self.sigmoid(prob_logit): Global memory pointer (batch_size, max_len + 1)
            u[-1]: q^(K+1) (batch, hidden_size)

        """
        u = [hidden.squeeze(0)]
        story_size = story.size()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # (batch_size, max_len + 1, hidden_size)
            if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
            embed_A = self.dropout_layer(embed_A)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  # used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)  # p: (batch_size, max_len + 1)

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)  # (batch_size, hidden_size)
            u_k = u[-1] + o_k
            u.append(u_k)
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1]

    # decoding
    def forward(self, query_vector, global_pointer):
        """

        Args:
            query_vector: (batch_size, hidden)
            global_pointer: (batch_size, max_len + 1)

        Returns:
            prob_soft: Softmaxed prob_logits
            prob_logits: (batch_size, max_len + 1)

        """
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]  # (batch_size, max_len + 1, hidden_size)
            if not args["ablationG"]:
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  # used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)  # (batch_size, max_len + 1)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_story[hop + 1]
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = shared_emb
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2 * embedding_dim, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length,
                batch_size, use_teacher_forcing, get_decoded_words, global_pointer):
        # Initialize variables for vocab and pointer
        """

        Args:
            extKnow: instantiation of class ExternalKnowledge
            story_size: the size of dialog history. (batch_size, max_len + 1, memory_size)
            story_lengths: The length of dialog history. (batch_size)
            copy_list:
                the list of the words in dialogue history preparing to be copied. not number.
                (batch_size, dialogue history len)
            encode_hidden:
                The concatenation of encoder output and read knowledge output. (batch_size, 2 * hidden size)
            target_batches:
                sketch_response (batch_size, max_len of all target response in this batch)
            max_target_length: The length of target response (batch_size)
            batch_size:
            use_teacher_forcing:
            get_decoded_words:
            global_pointer: (batch_size, max_len + 1)

        Returns:
            all_decoder_outputs_vocab: The logits by sketch rnn (max_rep_len, batch_size, vocab_len)
            all_decoder_outputs_ptr:
                The logits by external knowledge, it is transformed into the local memory pointer by softmax.
                (max_rep_len, batch_size, vocab_len)
            decoded_fine
            decoded_coarse

        """
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))  # record
        decoded_fine, decoded_coarse = [], []

        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)

        # Start to generate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input))  # b * e
            if len(embed_q.size()) == 1:
                embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0]  # (batch_size, hidden)

            # 直接使用ExternalKnowledge中嵌入矩阵C的参数做公式Softmax(Wh^d_t)的W
            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))  # The logits (batch_size, vocab_len)
            all_decoder_outputs_vocab[t] = p_vocab # 未使用MN的预测词分布
            _, topvi = p_vocab.data.topk(1)  # vocab_num

            # query the external knowledge using the hidden state of sketch RNN
            prob_soft, prob_logits = extKnow(query_vector, global_pointer)
            all_decoder_outputs_ptr[t] = prob_logits

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            # 以下为输出单词，因为训练只需要预测的词汇分布来求loss，所以训练时无需get_decoded_words代码块
            if get_decoded_words:
                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)  # dialog history len
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item()  # topvi[:,0][bi].item(),
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, i][bi] < story_lengths[bi] - 1:
                                cw = copy_list[bi][toppi[:, i][bi].item()]
                                break
                        temp_f.append(cw)

                        if args['record']:
                            memory_mask_for_step[bi, toppi[:, i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                # fine是询问过外部知识的回复，coarse是草图回复；即copy后的和未copy的
                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

            # print("temp_f:{}\ntemp_c:{}".format(temp_f, temp_c))
            # print("decoded_fine:{}\ndecoded_coarse:{}".format(decoded_fine, decoded_coarse))
        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        # scores = F.softmax(scores_, dim=1)
        return scores_


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
