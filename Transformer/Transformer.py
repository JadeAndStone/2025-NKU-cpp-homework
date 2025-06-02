import torch
from torch import nn
import numpy
import math
from torch.utils.data import DataLoader, TensorDataset
import re
import collections
import requests
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,x,*args):
        raise NotImplementedError
class Decoder(nn.Module):
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    def forward(self,X,*args):
        raise NotImplementedError
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络(通过非线性变换，变换输入向量的特征维度，增强拟合能力)"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
class AddNorm(nn.Module):
    """残差连接后进行层规范化(在特征维度上normalize，而BN则是对所有位置进行normalize)"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
# class AttentionDecoder(Decoder):
#     """带有注意力机制解码器的基本接口"""
#     def __init__(self, **kwargs):
#         super(AttentionDecoder, self).__init__(**kwargs)

#     def attention_weights(self):
#         raise NotImplementedError
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 显式调用父类构造函数
    def attention_weights(self):
        raise NotImplementedError
class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    def attention_weights(self):
        return self._attention_weights
def tokenize(text, token='word'):
    text = text.lower().replace('\u202f', ' ').replace('\xa0', ' ')
    if token == 'word':
        return [line.split() for line in text.split('\n')]
    elif token == 'char':
        return [list(line) for line in text.split('\n')]
    
def build_vocab(tokens, min_freq=1):
    # 强制添加特殊标记
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    # 展平二维列表为一维列表
    flat_tokens = [token for line in tokens for token in line]
    
    # 统计词频
    token_freqs = collections.Counter(flat_tokens)
    
    # 合并特殊标记和有效词汇
    unique_tokens = special_tokens.copy()
    unique_tokens += [
        token for token, freq in token_freqs.items() 
        if freq >= min_freq and token not in special_tokens
    ]
    
    # 生成词表
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    return vocab

def truncate_pad(line, num_steps, padding_token):
    return line[:num_steps] + [padding_token] * max(0, num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    lines = [[vocab.get(tok, 0) for tok in line] + [vocab['<eos>']] for line in lines]
    array = torch.tensor([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps):
    # 示例句子
    src_raw, tgt_raw = read_dataset('fra.txt', max_samples=10000)

    src_tokens = tokenize('\n'.join(src_raw))
    tgt_tokens = tokenize('\n'.join(tgt_raw))

    src_vocab = build_vocab(src_tokens)
    tgt_vocab = build_vocab(tgt_tokens)

    src_array, src_valid_len = build_array_nmt(src_tokens, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(tgt_tokens, tgt_vocab, num_steps)

    dataset = TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), src_vocab, tgt_vocab

def masked_cross_entropy_loss(y_hat, y, valid_len):
    weights = torch.ones_like(y)
    weights = sequence_mask(weights, valid_len)
    loss = nn.CrossEntropyLoss(reduction='none')
    unweighted_loss = loss(y_hat.permute(0, 2, 1), y)
    weighted_loss = (unweighted_loss * weights).mean(dim=1)
    return weighted_loss.mean()

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            loss = masked_cross_entropy_loss(Y_hat, Y, Y_valid_len)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch+1}, loss {total_loss:.4f}')
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    net.eval()
    src_tokens = tokenize(src_sentence.lower())[0]
    
    # 确保处理未知词时使用 <unk>
    src_tokens = [src_vocab.get(tok, src_vocab['<unk>']) for tok in src_tokens] + [src_vocab['<eos>']]
    
    # 其余代码保持不变
    src_tensor = torch.tensor(truncate_pad(src_tokens, num_steps, src_vocab['<pad>']), device=device).unsqueeze(0)
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    
    with torch.no_grad():
        enc_outputs = net.encoder(src_tensor, enc_valid_len)
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
        dec_input = torch.tensor([[tgt_vocab['<bos>']]], device=device)
        
        output_seq = []
        for _ in range(num_steps):
            Y, dec_state = net.decoder(dec_input, dec_state)
            pred = Y.argmax(dim=2)
            dec_input = pred
            pred_token = pred.squeeze().item()
            if pred_token == tgt_vocab['<eos>']:
                break
            output_seq.append(pred_token)
    
    # 确保目标词表包含 <unk>
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    return ' '.join([inv_tgt_vocab.get(idx, '<unk>') for idx in output_seq]), None

def bleu(pred_seq, label_seq, k):
    pred_tokens = pred_seq.split()
    label_tokens = label_seq.split()
    
    if len(pred_tokens) == 0 or len(label_tokens) == 0:
        return 0.0
    
    score = 0.0
    for n in range(1, k+1):
        pred_ngrams = collections.Counter([' '.join(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        label_ngrams = collections.Counter([' '.join(label_tokens[i:i+n]) for i in range(len(label_tokens)-n+1)])
        num_match = sum((pred_ngrams & label_ngrams).values())
        num_total = max(1, sum(pred_ngrams.values()))
        score += math.log((num_match + 1e-9) / num_total)
    
    bp = math.exp(min(0, 1 - len(label_tokens)/len(pred_tokens))) if len(pred_tokens) > 0 else 0.0
    return bp * math.exp(score / k)
def read_dataset(file_path='fra.txt', max_samples=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    src_raw, tgt_raw = [], []
    for line in lines[:max_samples]:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            src_raw.append(parts[0])
            tgt_raw.append(parts[1])
    return src_raw, tgt_raw
if __name__ == "__main__":
    # url = 'http://data.d2l.ai/data/fra-eng/fra.txt'
    # with open('fra.txt', 'wb') as f:
    #     f.write(requests.get(url).content)
    # print("Download completed.")
    src_raw, tgt_raw = read_dataset('fra.txt', max_samples=10000)
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, ',
            f'bleu {bleu(translation, fra, k=2):.3f}')