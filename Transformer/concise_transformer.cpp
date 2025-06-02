#include<iostream>
#include<torch/torch.h>
#include<torch/script.h>
#include<vector>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<tuple>
using namespace std;
struct Vocab {
    unordered_map<string, int64_t> stoi;
    vector<string> itos;
    int64_t pad, bos, eos;

    Vocab() {
        itos = {"<unk>", "<pad>", "<bos>", "<eos>", "hello", "world"};
        for (int i = 0; i < itos.size(); ++i)
            stoi[itos[i]] = i;
        pad = stoi["<pad>"];
        bos = stoi["<bos>"];
        eos = stoi["<eos>"];
    }

    Vocab(const vector<vector<string>>& lines, int min_freq = 1) {
        unordered_map<string, int> freq;
        for (const auto& line : lines)
            for (const auto& w : line)
                freq[w]++;

        itos = {"<unk>", "<pad>", "<bos>", "<eos>"};
        for (const auto& kv : freq)
            if (kv.second >= min_freq && find(itos.begin(), itos.end(), kv.first) == itos.end())
                itos.push_back(kv.first);

        for (size_t i = 0; i < itos.size(); ++i)
            stoi[itos[i]] = static_cast<int>(i);

        pad = stoi["<pad>"];
        bos = stoi["<bos>"];
        eos = stoi["<eos>"];
    }

    int64_t operator[](const string& token) const {
        auto it = stoi.find(token);
        return it != stoi.end() ? it->second : stoi.at("<unk>");
    }

    vector<int64_t> operator[](const vector<string>& tokens) const {
        vector<int64_t> ids;
        for (const auto& token : tokens)
            ids.push_back((*this)[token]);
        return ids;
    }

    size_t size() const { return itos.size(); }
};

struct Batch {
    torch::Tensor src_array;
    torch::Tensor src_valid_len;
    torch::Tensor tgt_array;
    torch::Tensor tgt_valid_len;
};

string read_data(const string& file_path) {
    ifstream file(file_path);
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

string preprocess_data(const string& text) {
    string out;
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc == 0xA0 || uc == 0x202F) {
            out += ' ';
        } else {
            out += c;
        }
    }
    return out;
}

void tokenize(const string& text,
              vector<vector<string>>& source,
              vector<vector<string>>& target,
              int num_samples = 10000) {
    istringstream iss(text);
    string line;
    int count = 0;

    while (getline(iss, line) && count < num_samples) {
        size_t tab = line.find('\t');
        if (tab == string::npos) continue;

        string src_line = line.substr(0, tab);
        string tgt_line = line.substr(tab + 1);

        istringstream src_stream(src_line), tgt_stream(tgt_line);
        string word;
        vector<string> src_tokens, tgt_tokens;

        while (src_stream >> word) src_tokens.push_back(word);
        while (tgt_stream >> word) tgt_tokens.push_back(word);

        source.push_back(src_tokens);
        target.push_back(tgt_tokens);
        count++;
    }
}

vector<int64_t> truncate_pad(const vector<int64_t>& line, int64_t num_steps, int64_t pad_id) {
    vector<int64_t> out = line;
    if (out.size() > static_cast<size_t>(num_steps))
        out.resize(num_steps);
    else
        out.insert(out.end(), num_steps - out.size(), pad_id);
    return out;
}

std::pair<torch::Tensor, torch::Tensor> build_array(const std::vector<std::vector<std::string>>& lines,
    const Vocab& vocab,
    int64_t num_steps) {
        std::vector<std::vector<int64_t>> array;
        std::vector<int64_t> valid_lens;

        for (const auto& line : lines) {
        auto ids = vocab[line];
        ids.push_back(vocab.eos);
        auto padded = truncate_pad(ids, num_steps, vocab.pad);
        valid_lens.push_back(std::count_if(padded.begin(), padded.end(), [&](int64_t x){ return x != vocab.pad; }));
        array.push_back(padded);
        }

        std::vector<int64_t> flat_array;
        for (const auto& row : array)
        flat_array.insert(flat_array.end(), row.begin(), row.end());

        torch::Tensor data = torch::from_blob(
        flat_array.data(),
        {static_cast<int64_t>(array.size()), static_cast<int64_t>(array[0].size())},
        torch::kLong
        ).clone();

        torch::Tensor valid_len = torch::tensor(valid_lens, torch::kLong);
        return {data, valid_len};
}

vector<Batch> load_array(const torch::Tensor& src_array, const torch::Tensor& src_valid_len,
                         const torch::Tensor& tgt_array, const torch::Tensor& tgt_valid_len,
                         int64_t batch_size) {
    vector<Batch> batches;
    int64_t num_samples = src_array.size(0);
    for (int64_t i = 0; i < num_samples; i += batch_size) {
        int64_t end = min(i + batch_size, num_samples);
        batches.push_back({
            src_array.slice(0, i, end),
            src_valid_len.slice(0, i, end),
            tgt_array.slice(0, i, end),
            tgt_valid_len.slice(0, i, end)
        });
    }
    return batches;
}

class DataloaderImpl : public torch::nn::Module {
public:
    std::tuple<std::vector<Batch>, Vocab, Vocab> forward(int batch_size, const std::string& file_name, int num_steps = 10) {
        string raw_text = read_data(file_name);
        string text = preprocess_data(raw_text);

        vector<vector<string>> source, target;
        tokenize(text, source, target);

        Vocab src_vocab(source, 1);
        Vocab tgt_vocab(target, 1);

        auto [src_array, src_valid_len] = build_array(source, src_vocab, num_steps);
        auto [tgt_array, tgt_valid_len] = build_array(target, tgt_vocab, num_steps);

        auto batches = load_array(src_array, src_valid_len, tgt_array, tgt_valid_len, batch_size);
        return {batches, src_vocab, tgt_vocab};
    }
};
TORCH_MODULE(Dataloader);




//Encoder和Decoder基类，用于后面实现Attention、Transformer等的EncoderDecoder
class EncoderImpl: public torch::nn::Module{
    public:
        virtual torch::Tensor forward(torch::Tensor x,vector<torch::Tensor> &args)=0;
};
TORCH_MODULE(Encoder); //libtorch很方便的一个宏，能够自动实现每个自定义模块的智能指针
// class DecoderImpl: public torch::nn::Module{
//     public:
//         virtual vector<torch::Tensor> init_state(torch::Tensor encoder_output,vector<torch::Tensor> &args)=0;
//         virtual torch::Tensor forward(torch::Tensor X,vector<torch::Tensor> &decoder_state)=0;   
// };
// TORCH_MODULE(Decoder);
class DecoderImpl : public torch::nn::Module {
    public:
        virtual std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>>
        forward(torch::Tensor X, std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>& state) = 0;
    
        virtual std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>
        init_state(torch::Tensor enc_outputs, torch::Tensor valid_lens) = 0;
    };
TORCH_MODULE(Decoder);
    
class EncoderDecoderImpl : public torch::nn::Module {
    public:
        Encoder encoder{nullptr};
        Decoder decoder{nullptr};

        EncoderDecoderImpl(Encoder encoder, Decoder decoder)
            : encoder(encoder), decoder(decoder) {
            register_module("encoder", encoder);
            register_module("decoder", decoder);
        }

        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>>
        forward(torch::Tensor encoder_X, torch::Tensor decoder_X, std::vector<torch::Tensor>& args) {
            auto enc_outputs = encoder->forward(encoder_X, args);
            auto dec_state = decoder->init_state(enc_outputs, args[0]);
            return decoder->forward(decoder_X, dec_state);
        }
};
TORCH_MODULE(EncoderDecoder);

class PositionWiseFFNImpl: public torch::nn::Module{
    torch::nn::Linear ffn_Linear1{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Linear ffn_Linear2{nullptr};
    public:
        PositionWiseFFNImpl(int ffn_input,int ffn_hidden,int ffn_output):
            ffn_Linear1(torch::nn::Linear(ffn_input,ffn_hidden)),
            ffn_Linear2(torch::nn::Linear(ffn_hidden,ffn_output)){
            relu=torch::nn::ReLU();
            register_module("ffn_Linear1",ffn_Linear1);
            register_module("ffn_Linear2",ffn_Linear2);
            register_module("relu",relu);
        }
        torch::Tensor forward(torch::Tensor X){
            return ffn_Linear2(relu(ffn_Linear1(X)));
        }
};
TORCH_MODULE(PositionWiseFFN);
class AddNormImpl: public torch::nn::Module{
    torch::nn::Dropout dropout{nullptr};
    torch::nn::LayerNorm ln{nullptr};
    public:
        AddNormImpl(vector<int64_t> norm_shape,double dropout_rate):
        //神奇，libtorch没办法跟py一样自动判断你传的norm_shape是int还是list[int]，所以不让你直接传tensor
            dropout(torch::nn::Dropout(dropout_rate)),
            ln(torch::nn::LayerNorm(norm_shape)){
            register_module("dropout",dropout);
            register_module("ln",ln);
        } 
        torch::Tensor forward(torch::Tensor X,torch::Tensor Y){
            return ln(dropout(Y)+X);
        }
};
TORCH_MODULE(AddNorm);
torch::Tensor sequence_mask(torch::Tensor X, c10::optional<torch::Tensor> valid_lens, float value=0.0) {
    int64_t maxlen = X.size(1);
    auto device = X.device();

    // 检查 valid_len 是否存在
    if (!valid_lens.has_value()) {
        return X;
    }

    // 正确取出 tensor，并使用
    auto valid = valid_lens.value();

    auto mask = torch::arange(maxlen, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                    .unsqueeze(0)
                    .lt(valid.unsqueeze(1));  // valid 是 Tensor 了，才能 .unsqueeze

    X=X.masked_fill(~mask.to(torch::kBool), value);
    return X;
}
torch::Tensor masked_softmax(torch::Tensor X, c10::optional<torch::Tensor> valid_lens) {
    if (!valid_lens.has_value()) {
        return torch::softmax(X, -1);
    } else {
        auto shape = X.sizes();
        int64_t last_dim = shape.back();

        auto reshaped = X.reshape({-1, last_dim});
        auto valid = valid_lens.value();

        if (valid.dim() == 1) {
            valid = valid.repeat_interleave(reshaped.size(0) / valid.size(0), 0);
        }
        valid = valid.reshape({-1});

        reshaped = sequence_mask(reshaped, valid, -1e6);
        auto result = torch::softmax(reshaped, -1);
        return result.reshape(shape);
    }
}

class DotProductAttentionImpl: public torch::nn::Module{
    torch::nn::Dropout dropout{nullptr};
    public:
        DotProductAttentionImpl(double dropout_rate):dropout(torch::nn::Dropout(dropout_rate)){
            register_module("dropout",dropout);
        }
        torch::Tensor forward(torch::Tensor query,torch::Tensor key,torch::Tensor value,c10::optional<torch::Tensor> valid_lens = c10::nullopt){
            double d=query.size(-1);
            auto score=torch::bmm(query,key.transpose(1,2))/sqrt(d);
            torch::Tensor attention_weight=masked_softmax(score,valid_lens);
            return torch::bmm(dropout(attention_weight),value);
        }
};
TORCH_MODULE(DotProductAttention);
torch::Tensor transpose_qkv(torch::Tensor X, int64_t num_heads) {
    // X: [batch_size, seq_len, num_hiddens]
    int64_t batch_size = X.size(0);
    int64_t seq_len = X.size(1);
    int64_t num_hiddens = X.size(2);
    int64_t head_dim = num_hiddens / num_heads;

    // [batch_size, seq_len, num_heads, head_dim]
    X = X.view({batch_size, seq_len, num_heads, head_dim});
    
    // [batch_size, num_heads, seq_len, head_dim]
    X = X.permute({0, 2, 1, 3});
    
    // [batch_size * num_heads, seq_len, head_dim]
    X = X.reshape({batch_size * num_heads, seq_len, head_dim});
    return X;
}
torch::Tensor transpose_output(torch::Tensor X, int64_t num_heads) {
    // X: [batch_size * num_heads, seq_len, head_dim]
    int64_t batch_heads = X.size(0);
    int64_t seq_len = X.size(1);
    int64_t head_dim = X.size(2);
    int64_t batch_size = batch_heads / num_heads;

    // [batch_size, num_heads, seq_len, head_dim]
    X = X.view({batch_size, num_heads, seq_len, head_dim});
    
    // [batch_size, seq_len, num_heads, head_dim]
    X = X.permute({0, 2, 1, 3});
    
    // [batch_size, seq_len, num_heads * head_dim]
    X = X.reshape({batch_size, seq_len, num_heads * head_dim});
    return X;
}
class MultiHeadAttentionImpl : public torch::nn::Module {
    public:
        int64_t num_heads;
        int64_t num_hiddens;
        DotProductAttention attention;
    
        torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr}, W_o{nullptr};
    
        MultiHeadAttentionImpl(int64_t key_size, int64_t query_size, int64_t value_size,
                               int64_t num_hiddens, int64_t num_heads,
                               double dropout, bool bias = false)
            : num_heads(num_heads), num_hiddens(num_hiddens),
              attention(DotProductAttention(dropout)) {
    
            W_q = register_module("W_q", torch::nn::Linear(torch::nn::LinearOptions(query_size, num_hiddens).bias(bias)));
            W_k = register_module("W_k", torch::nn::Linear(torch::nn::LinearOptions(key_size, num_hiddens).bias(bias)));
            W_v = register_module("W_v", torch::nn::Linear(torch::nn::LinearOptions(value_size, num_hiddens).bias(bias)));
            W_o = register_module("W_o", torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, num_hiddens).bias(bias)));
            register_module("attention", attention);
        }
    
        torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values,
                              c10::optional<torch::Tensor> valid_lens = c10::nullopt) {
            // queries, keys, values: (batch_size, seq_len, num_hiddens)
            auto q = transpose_qkv(W_q->forward(queries), num_heads);
            auto k = transpose_qkv(W_k->forward(keys), num_heads);
            auto v = transpose_qkv(W_v->forward(values), num_heads);
    
            torch::Tensor valid;
            if (valid_lens.has_value()) {
                // valid_lens shape: (batch_size,) or (batch_size, seq_len)
                // repeat_interleave(batch_size * num_heads)

                // std::cout << "valid_lens: " << valid_lens.value() << std::endl;
                // std::cout << "valid_lens dtype: " << valid_lens.value().dtype() << std::endl;
                // std::cout << "valid_lens shape: " << valid_lens.value().sizes() << std::endl;
                // cout<<"forward"<<endl;
                valid = torch::repeat_interleave(valid_lens.value(), num_heads, 0);
                // std::cout << "Expanded valid_lens: " << valid << std::endl;
            }
    
            // std::cout << "Before attention forward" << std::endl;
            auto output = attention->forward(q, k, v, valid_lens.has_value() ? c10::optional<torch::Tensor>(valid) : c10::nullopt);
            // std::cout << "After attention forward" << std::endl;
            auto output_concat = transpose_output(output, num_heads);
            return W_o->forward(output_concat);
        }
    };
TORCH_MODULE(MultiHeadAttention);
class EncoderBlockImpl : public torch::nn::Module {
    public:
        MultiHeadAttention attention{nullptr};
        AddNorm addnorm1{nullptr};
        PositionWiseFFN ffn{nullptr};
        AddNorm addnorm2{nullptr};
    
        EncoderBlockImpl(int64_t key_size, int64_t query_size, int64_t value_size,
                         int64_t num_hiddens, std::vector<int64_t> norm_shape,
                         int64_t ffn_num_input, int64_t ffn_num_hiddens,
                         int64_t num_heads, double dropout, bool use_bias = false)
            : attention(MultiHeadAttention(key_size, query_size, value_size,
                                           num_hiddens, num_heads, dropout, use_bias)),
              addnorm1(AddNorm(norm_shape, dropout)),
              ffn(PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)),
              addnorm2(AddNorm(norm_shape, dropout)) {
    
            register_module("attention", attention);
            register_module("addnorm1", addnorm1);
            register_module("ffn", ffn);
            register_module("addnorm2", addnorm2);
        }
    
        torch::Tensor forward(torch::Tensor X, c10::optional<torch::Tensor> valid_lens = c10::nullopt) {
            auto Y = addnorm1->forward(X, attention->forward(X, X, X, valid_lens));
            return addnorm2->forward(Y, ffn->forward(Y));
        }
    };
TORCH_MODULE(EncoderBlock);
class PositionalEncodingImpl : public torch::nn::Module {
    public:
        torch::nn::Dropout dropout{nullptr};
        torch::Tensor P;
    
        PositionalEncodingImpl(int64_t num_hiddens, double dropout_rate, int64_t max_len = 1000)
            : dropout(torch::nn::DropoutOptions(dropout_rate)) {
            
            // [1, max_len, num_hiddens]
            P = torch::zeros({1, max_len, num_hiddens});
    
            auto position = torch::arange(max_len, torch::kFloat32).unsqueeze(1);  // [max_len, 1]
            auto div_term = torch::pow(
                10000,
                torch::arange(0, num_hiddens, 2, torch::kFloat32) / num_hiddens);  // [num_hiddens/2]
    
            P.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)}, torch::sin(position / div_term));
            P.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)}, torch::cos(position / div_term));
    
            register_buffer("P", P);
            register_module("dropout", dropout);
        }
    
        torch::Tensor forward(torch::Tensor X) {
            // X: [batch_size, seq_len, num_hiddens]
            // P: [1, max_len, num_hiddens]
    
            // 给输入加上对应长度的位置编码
            X = X + P.index({0, torch::indexing::Slice(0, X.size(1)), torch::indexing::Slice()}).to(X.device());
            return dropout(X);
        }
    };
TORCH_MODULE(PositionalEncoding);
class TransformerEncoderImpl : public EncoderImpl {
    public:
        int64_t num_hiddens;
        torch::nn::Embedding embedding{nullptr};
        torch::nn::ModuleHolder<PositionalEncodingImpl> pos_encoding{nullptr};
        torch::nn::Sequential blks;
        std::vector<torch::Tensor> attention_weights;

        TransformerEncoderImpl(int64_t vocab_size, int64_t key_size, int64_t query_size,
            int64_t value_size, int64_t num_hiddens,
            std::vector<int64_t> norm_shape,
            int64_t ffn_num_input, int64_t ffn_num_hiddens,
            int64_t num_heads, int64_t num_layers,
            double dropout, bool use_bias = false)
            : num_hiddens(num_hiddens),
            embedding(register_module("embedding", torch::nn::Embedding(vocab_size, num_hiddens))) {

            pos_encoding = register_module("pos_encoding", std::make_shared<PositionalEncodingImpl>(num_hiddens, dropout));

            for (int64_t i = 0; i < num_layers; ++i) {
            auto block = EncoderBlock(key_size, query_size, value_size,
                                num_hiddens, norm_shape,
                                ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, use_bias);
            blks->push_back(block);
            register_module("block" + std::to_string(i), block);
            }
            register_module("blks", blks);
            attention_weights.resize(num_layers);
            }

            torch::Tensor forward(torch::Tensor X, std::vector<torch::Tensor>& args) override {
                c10::optional<torch::Tensor> valid_lens = args.size() > 0 ? c10::optional<torch::Tensor>(args[0]) : c10::nullopt;
            
                auto X_emb = embedding->forward(X) * std::sqrt(num_hiddens);
                auto X_pos = pos_encoding->forward(X_emb);
            
                for (size_t i = 0; i < blks->size(); ++i) {
                    X_pos = blks[i]->as<EncoderBlock>()->forward(X_pos, valid_lens);
                }
                return X_pos;
            }
            
};
TORCH_MODULE(TransformerEncoder);
class DecoderBlockImpl : public torch::nn::Module {
    public:
        int64_t i;
        MultiHeadAttention attention1{nullptr};
        AddNorm addnorm1{nullptr};
        MultiHeadAttention attention2{nullptr};
        AddNorm addnorm2{nullptr};
        PositionWiseFFN ffn{nullptr};
        AddNorm addnorm3{nullptr};
    
        DecoderBlockImpl(int64_t key_size, int64_t query_size, int64_t value_size,
                         int64_t num_hiddens, std::vector<int64_t> norm_shape,
                         int64_t ffn_num_input, int64_t ffn_num_hiddens,
                         int64_t num_heads, double dropout, int64_t index)
            : i(index),
              attention1(MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout)),
              addnorm1(AddNorm(norm_shape, dropout)),
              attention2(MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout)),
              addnorm2(AddNorm(norm_shape, dropout)),
              ffn(PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)),
              addnorm3(AddNorm(norm_shape, dropout)) {
            register_module("attention1", attention1);
            register_module("addnorm1", addnorm1);
            register_module("attention2", attention2);
            register_module("addnorm2", addnorm2);
            register_module("ffn", ffn);
            register_module("addnorm3", addnorm3);
        }
    
        std::tuple<torch::Tensor, std::vector<std::optional<torch::Tensor>>> forward(
            torch::Tensor X,
            std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>& state) {
            
            auto& enc_outputs = std::get<0>(state);
            auto& enc_valid_lens = std::get<1>(state);
            auto& decoder_states = std::get<2>(state);
    
            torch::Tensor key_values;
            if (!decoder_states[i].has_value()) {
                key_values = X;
            } else {
                key_values = torch::cat({decoder_states[i].value(), X}, 1);
            }
            decoder_states[i] = key_values;
    
            torch::Tensor dec_valid_lens;
            if (this->is_training()) {
                auto batch_size = X.size(0);
                auto num_steps = X.size(1);
                dec_valid_lens = torch::arange(1, num_steps + 1, X.options())
                                    .repeat({batch_size, 1});
            } else {
                dec_valid_lens = torch::full({X.size(0)}, X.size(1), torch::kLong).to(X.device());
            }
            // std::cout << "[DecoderBlock] dec_valid_lens: " << dec_valid_lens.sizes() << std::endl;
            auto X2 = attention1->forward(X, key_values, key_values, dec_valid_lens);
            auto Y = addnorm1->forward(X, X2);
    
            auto Y2 = attention2->forward(Y, enc_outputs, enc_outputs, enc_valid_lens);
            auto Z = addnorm2->forward(Y, Y2);
    
            auto out = addnorm3->forward(Z, ffn->forward(Z));
            return {out, decoder_states};
        }
    };
TORCH_MODULE(DecoderBlock);
struct AttentionDecoderImpl :public DecoderImpl{
    virtual std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>>
    forward(torch::Tensor X,
            std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>& state) = 0;
};
TORCH_MODULE(AttentionDecoder);

struct TransformerDecoderImpl :public AttentionDecoderImpl {
    int64_t num_hiddens;
    int64_t num_layers;
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Dropout dropout{nullptr};
    std::vector<DecoderBlock> blocks;
    torch::nn::Linear dense{nullptr};

    TransformerDecoderImpl(int64_t vocab_size, int64_t k_size, int64_t q_size, int64_t v_size,
                           int64_t hiddens, std::vector<int64_t> norm_shape,
                           int64_t ffn_input, int64_t ffn_hidden,
                           int64_t heads, int64_t layers, double drop)
        : num_hiddens(hiddens), num_layers(layers),
          embedding(torch::nn::Embedding(vocab_size, hiddens)),
          dropout(torch::nn::Dropout(drop)) {

        register_module("embedding", embedding);
        register_module("dropout", dropout);
        for (int64_t i = 0; i < layers; ++i) {
            auto blk = DecoderBlock(k_size, q_size, v_size, hiddens, norm_shape,
                                    ffn_input, ffn_hidden, heads, drop, i);
            register_module("block" + std::to_string(i), blk);
            blocks.push_back(blk);
        }
        dense = torch::nn::Linear(hiddens, vocab_size);
        register_module("dense", dense);
    }
    
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>>
    forward(torch::Tensor X,
            std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>& state) override {
        // std::cout << "[Decoder] input shape: " << X.sizes() << std::endl;
        X = dropout(embedding(X) * std::sqrt((double)num_hiddens));
        for (auto& blk : blocks) {
            auto result = blk->forward(X, state);
            X = std::get<0>(result);
        }
        return {dense(X), state};
    }

    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::optional<torch::Tensor>>>
    init_state(torch::Tensor enc_outputs, torch::Tensor enc_valid_lens) {
        return {enc_outputs, enc_valid_lens, std::vector<std::optional<torch::Tensor>>(num_layers)};
    }
};
TORCH_MODULE(TransformerDecoder);
torch::Tensor masked_cross_entropy_loss(
    const torch::Tensor& pred,      // [B, T, V]
    const torch::Tensor& label,     // [B, T]
    const torch::Tensor& valid_len  // [B]
) {
    auto weights = torch::ones_like(label, torch::kFloat32);
    int64_t maxlen = label.size(1);
    auto mask = torch::arange(maxlen, torch::kLong)
                    .unsqueeze(0)
                    .to(valid_len.device())
                    .lt(valid_len.unsqueeze(1));
    weights = weights * mask.to(weights.dtype());

    auto pred_flat = pred.reshape({-1, pred.size(2)});     // [B*T, V]
    auto label_flat = label.reshape({-1});                 // [B*T]
    auto weights_flat = weights.reshape({-1});             // [B*T]

    // Debug vocab mismatch
    auto label_max = label_flat.max().item<int64_t>();
    auto label_min = label_flat.min().item<int64_t>();
    // std::cout << "Label min: " << label_min << ", max: " << label_max << std::endl;
    // std::cout << "Pred vocab size: " << pred.size(2) << std::endl;
    // TORCH_CHECK(label_max < pred.size(2), "Label exceeds vocabulary size!");

    auto loss = torch::nn::functional::cross_entropy(
        pred_flat, label_flat,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)
    );

    auto masked_loss = loss * weights_flat;
    return masked_loss.sum() / weights_flat.sum();
}



// void train_seq2seq(EncoderDecoder net,
//     const std::vector<Batch>& data_iter,
//     torch::optim::Optimizer& optimizer,
//     int64_t num_epochs,
//     const Vocab& tgt_vocab,
//     torch::Device device) {
//         net->train();
//         net->to(device);

//         for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
//         double total_loss = 0.0;
//         int64_t token_count = 0;

//         for (const auto& batch : data_iter) {
//         auto encoder_X = batch.src_array.to(device);
//         auto src_valid_len = batch.src_valid_len.to(device);
//         auto tgt_array = batch.tgt_array.to(device);
//         auto tgt_valid_len = batch.tgt_valid_len.to(device);

//         auto bos = torch::full({tgt_array.size(0), 1}, tgt_vocab["<bos>"], torch::kLong).to(device);
//         auto decoder_X = torch::cat({bos, tgt_array.slice(1, 0, -1)}, 1);

//         optimizer.zero_grad();

//         std::vector<torch::Tensor> enc_args = {src_valid_len};
//         // std::cout << "[DEBUG] Epoch: " << epoch << ", Batch start" << std::endl;
//         // std::cout << "encoder_X shape: " << encoder_X.sizes() << std::endl;
//         // std::cout << "decoder_X shape: " << decoder_X.sizes() << std::endl;
//         // std::cout << "Running forward..." << std::endl;
//         auto [logits, _] = net->forward(encoder_X, decoder_X, enc_args);
//         // std::cout << "Running loss..." << std::endl;
//         // cout<<logits.sizes();
//         auto loss = masked_cross_entropy_loss(logits, tgt_array, tgt_valid_len);
//         // std::cout << "Loss value: " << loss.item<double>() << std::endl;
//         // std::cout << "Running backward..." << std::endl;
//         loss.backward();
//         // std::cout << "Step..." << std::endl;
//         optimizer.step();

//         total_loss += loss.item<double>() * tgt_valid_len.sum().item<int64_t>();
//         token_count += tgt_valid_len.sum().item<int64_t>();
//         }

//         std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] Loss per token: " << total_loss / token_count << std::endl;
//         }
// }
void train_seq2seq(EncoderDecoder net,
    const std::vector<Batch>& data_iter,
    torch::optim::Optimizer& optimizer,
    int64_t num_epochs,
    const Vocab& tgt_vocab,
    torch::Device device) {

    ofstream loss_log("loss_log.csv");
    loss_log << "epoch,loss\n";

    net->train();
    net->to(device);

    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        int64_t token_count = 0;

        for (const auto& batch : data_iter) {
            auto encoder_X = batch.src_array.to(device);
            auto src_valid_len = batch.src_valid_len.to(device);
            auto tgt_array = batch.tgt_array.to(device);
            auto tgt_valid_len = batch.tgt_valid_len.to(device);

            auto bos = torch::full({tgt_array.size(0), 1}, tgt_vocab["<bos>"], torch::kLong).to(device);
            auto decoder_X = torch::cat({bos, tgt_array.slice(1, 0, -1)}, 1);

            optimizer.zero_grad();

            std::vector<torch::Tensor> enc_args = {src_valid_len};
            auto [logits, _] = net->forward(encoder_X, decoder_X, enc_args);
            auto loss = masked_cross_entropy_loss(logits, tgt_array, tgt_valid_len);

            loss.backward();
            optimizer.step();

            total_loss += loss.item<double>() * tgt_valid_len.sum().item<int64_t>();
            token_count += tgt_valid_len.sum().item<int64_t>();
        }

        double avg_loss = total_loss / token_count;
        cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] Loss per token: " << avg_loss << endl;

        // 写入 CSV
        loss_log << epoch + 1 << "," << avg_loss << "\n";
    }

    loss_log.close();
}
string predict_seq2seq(EncoderDecoder net,
    const string& src_sentence,
    const Vocab& src_vocab,
    const Vocab& tgt_vocab,
    int64_t num_steps,
    torch::Device device) {
        net->eval();

        istringstream iss(src_sentence);
        string token;
        vector<string> tokens;
        while (iss >> token) tokens.push_back(token);

        vector<int64_t> src_ids;
        for (const auto& tok : tokens)
        src_ids.push_back(src_vocab.stoi.count(tok) ? src_vocab.stoi.at(tok) : 0);
        src_ids.push_back(src_vocab["<eos>"]);

        auto enc_input = torch::tensor(truncate_pad(src_ids, num_steps, src_vocab["<pad>"]), torch::kLong).unsqueeze(0).to(device);
        auto enc_valid_len = torch::tensor({(int64_t)src_ids.size()}, torch::kLong).to(device);

        vector<torch::Tensor> args = {enc_valid_len};
        auto enc_outputs = net->encoder->forward(enc_input, args);
        auto dec_state = net->decoder->init_state(enc_outputs, enc_valid_len);

        auto dec_input = torch::tensor({{tgt_vocab["<bos>"]}}, torch::kLong).to(device);
        vector<int64_t> output;

        for (int i = 0; i < num_steps; ++i) {
            // cout << "[Predict] Step " << i << endl;
            torch::Tensor logits;
            // cout << "[Predict] Calling decoder forward..." << endl;
            tie(logits, dec_state) = net->decoder->forward(dec_input, dec_state);
            // cout << "[Predict] Decoder forward done." << endl;
            auto pred = logits.squeeze(1).argmax(1);
            int64_t pred_token = pred.item<int64_t>();
            // cout << "[Predict] Token: " << pred_token << " (" << tgt_vocab.itos[pred_token] << ")" << endl;
            if (pred_token == tgt_vocab["<eos>"]) break;
            output.push_back(pred_token);
            dec_input = pred.unsqueeze(0);
            auto probs = torch::softmax(logits.squeeze(1), -1);
            auto topk = probs.topk(5, -1);
            auto topk_values = get<0>(topk);
            auto topk_indices = get<1>(topk);
            // cout << "[Step " << i << "] top-5 probs:\n";
            for (int k = 0; k < 5; ++k) {
                int idx = topk_indices[0][k].item<int>();
                float p = topk_values[0][k].item<float>();
                // cout << "  " << tgt_vocab.itos[idx] << " (" << idx << "): " << p << endl;
            }
        }
        string result;
        for (auto idx : output) {
        if (idx >= 0 && idx < (int64_t)tgt_vocab.itos.size())
            result += tgt_vocab.itos[idx] + " ";
        else result += "<unk> ";
        }
        return result;
}









int main(){
    // torch::Tensor x=torch::ones({3,3});
    // torch::Tensor valid_len=torch::tensor({1,2,3});
    // cout<<sequence_mask(x,valid_len,12);    
    // cout<<masked_softmax(x,valid_len);
    // int64_t batch_size = 2, seq_len = 4, num_hiddens = 8, num_heads = 2;
    // torch::Tensor X = torch::randn({batch_size, seq_len, num_hiddens});

    // auto transposed = transpose_qkv(X, num_heads);
    // std::cout << "After transpose_qkv: " << transposed.sizes() << std::endl;

    // auto restored = transpose_output(transposed, num_heads);
    // std::cout << "After transpose_output: " << restored.sizes() << std::endl;
    // 初始化随机种子
    // torch::manual_seed(42);

    // 模拟输入: batch_size=2, seq_len=4, num_hiddens=8
    // int64_t batch_size = 2;
    // int64_t seq_len = 4;
    // int64_t num_hiddens = 8;
    // int64_t num_heads = 2;

    // // 随机输入 query/key/value (batch_size, seq_len, num_hiddens)
    // torch::Tensor queries = torch::randn({batch_size, seq_len, num_hiddens});
    // torch::Tensor keys = torch::randn({batch_size, seq_len, num_hiddens});
    // torch::Tensor values = torch::randn({batch_size, seq_len, num_hiddens});

    // // 创建 MultiHeadAttention 模块
    // MultiHeadAttention mha(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.1);

    // // 不带 valid_lens
    // torch::Tensor output1 = mha->forward(queries, keys, values);
    // std::cout << "Output shape without valid_lens: " << output1.sizes() << std::endl;

    // // 带 valid_lens: 长度为 batch_size，表示每个序列有效长度
    // torch::Tensor valid_lens = torch::tensor({3, 2}, torch::kInt64);
    // torch::Tensor output2 = mha->forward(queries, keys, values, valid_lens);
    // std::cout << "Output shape with valid_lens: " << output2.sizes() << std::endl;
    // cout<<"Running";
//MultiheadAttention test
    // torch::manual_seed(0);

    // int64_t batch_size = 2;
    // int64_t num_queries = 4;
    // int64_t num_kv = 4;
    // int64_t dim = 8;

    // auto query = torch::randn({batch_size, num_queries, dim});
    // auto key = torch::randn({batch_size, num_kv, dim});
    // auto value = torch::randn({batch_size, num_kv, dim});
    // auto valid_lens = torch::tensor({3, 2}, torch::kInt64);  // 不同样本长度

    // auto attn = DotProductAttention(0.1);

    // std::cout << "\n===== Without valid_lens =====\n";
    // auto out1 = attn->forward(query, key, value);
    // std::cout << "Output shape (no lens): " << out1.sizes() << std::endl;

    // std::cout << "\n===== With valid_lens =====\n";
    // auto out2 = attn->forward(query, key, value, valid_lens);
    // std::cout << "Output shape (with lens): " << out2.sizes() << std::endl;
//encoderblock test
    // torch::manual_seed(0);

    // // 模拟输入参数
    // int64_t batch_size = 2;
    // int64_t seq_len = 4;
    // int64_t num_hiddens = 8;
    // int64_t ffn_num_input = 8;
    // int64_t ffn_num_hiddens = 16;
    // int64_t num_heads = 2;
    // double dropout = 0.1;

    // auto X = torch::randn({batch_size, seq_len, num_hiddens});
    // auto valid_lens = torch::tensor({3, 2}, torch::kLong);

    // EncoderBlock encoder_block(
    //     /*key_size=*/num_hiddens,
    //     /*query_size=*/num_hiddens,
    //     /*value_size=*/num_hiddens,
    //     /*num_hiddens=*/num_hiddens,
    //     /*norm_shape=*/vector<int64_t>{num_hiddens},
    //     /*ffn_num_input=*/ffn_num_input,
    //     /*ffn_num_hiddens=*/ffn_num_hiddens,
    //     /*num_heads=*/num_heads,
    //     /*dropout=*/dropout,
    //     /*use_bias=*/false
    // );

    // auto Y = encoder_block->forward(X, valid_lens);

    // std::cout << "Input shape: " << X.sizes() << std::endl;
    // std::cout << "Output shape: " << Y.sizes() << std::endl;
    // torch::manual_seed(0);

    // int64_t batch_size = 2, seq_len = 4, num_hiddens = 8;
    // double dropout = 0.1;

    // PositionalEncoding pe(num_hiddens, dropout);

    // auto X = torch::zeros({batch_size, seq_len, num_hiddens});
    // pe->eval();
    // auto out = pe->forward(X);

    // std::cout << "Output shape: " << out.sizes() << std::endl;
    // std::cout << out << std::endl;
//transformerencoder test
    // torch::manual_seed(0);

    // int64_t batch_size = 2;
    // int64_t seq_len = 4;
    // int64_t vocab_size = 1000;
    // int64_t num_hiddens = 8;
    // double dropout = 0.1;

    // TransformerEncoder encoder(
    //     /*vocab_size=*/1000,
    //     /*key_size=*/8,
    //     /*query_size=*/8,
    //     /*value_size=*/8,
    //     /*num_hiddens=*/8,
    //     /*norm_shape=*/std::vector<int64_t>{8},
    //     /*ffn_num_input=*/8,
    //     /*ffn_num_hiddens=*/16,
    //     /*num_heads=*/2,
    //     /*num_layers=*/1,
    //     /*dropout=*/0.1,
    //     /*use_bias=*/false);

    // // 模拟输入 token 序列，batch_size x seq_len
    // auto X = torch::randint(0, vocab_size, {batch_size, seq_len}, torch::kLong);

    // auto Y = encoder->forward(X);

    // std::cout << "Input tokens:\n" << X << std::endl;
    // std::cout << "Output shape: " << Y.sizes() << std::endl;
    // std::cout << "Output tensor:\n" << Y << std::endl;
//decoderblock test
    // torch::manual_seed(0);

    // // 模拟输入参数
    // int64_t batch_size = 2;
    // int64_t seq_len = 4;
    // int64_t num_hiddens = 8;
    // int64_t ffn_num_input = 8;
    // int64_t ffn_num_hiddens = 16;
    // int64_t num_heads = 2;
    // double dropout = 0.1;

    // // 随机输入X，形状(batch_size, seq_len, num_hiddens)
    // torch::Tensor X = torch::randn({batch_size, seq_len, num_hiddens});

    // // 模拟编码器输出 enc_outputs 和有效长度 enc_valid_lens
    // torch::Tensor enc_outputs = torch::randn({batch_size, seq_len, num_hiddens});
    // torch::Tensor enc_valid_lens = torch::tensor({3, 2}, torch::kLong);

    // // 初始化DecoderBlock，索引0（假设是第0层解码块）
    // DecoderBlock decoder_block(
    //     num_hiddens, num_hiddens, num_hiddens,
    //     num_hiddens, vector<int64_t>{num_hiddens},
    //     ffn_num_input, ffn_num_hiddens,
    //     num_heads, dropout, 0
    // );

    // decoder_block->train();  // 切换训练模式

    // // 初始化state结构
    // std::vector<std::optional<torch::Tensor>> decoder_states(1);  // 初始为nullopt
    // auto state = std::forward_as_tuple(enc_outputs, enc_valid_lens, decoder_states);

    // // 前向传播
    // auto [out, updated_decoder_states] = decoder_block->forward(X, state);

    // std::cout << "Output shape: " << out.sizes() << std::endl;
    // std::cout << "Output tensor:\n" << out << std::endl;
//transformerdecoder test
    // int64_t vocab_size = 1000, h = 32, layers = 2;
    // TransformerDecoder decoder(vocab_size, h, h, h, h, vector<int64_t>{h}, h, 64, 4, layers, 0.1);

    // auto X = torch::tensor({{1, 2, 3}, {4, 5, 0}}, torch::kLong);
    // auto enc_outputs = torch::randn({2, 3, h});
    // auto enc_valid_lens = torch::tensor({3, 2}, torch::kLong);
    // auto state = decoder->init_state(enc_outputs, enc_valid_lens);

    // auto [Y, new_state] = decoder->forward(X, state);
    // std::cout << "Output shape: " << Y.sizes() << std::endl;
//dataloader test
    // Vocab vocab;
    // vector<vector<string>> test_lines = {
    //     {"hello", "world"},
    //     {"world"},
    //     {"hello", "world", "hello"}
    // };

    // int64_t num_steps = 5;
    // auto [data, valid_len] = build_array(test_lines, vocab, num_steps);

    // cout << "Data Tensor:\n" << data << endl;
    // cout << "Shape: " << data.sizes() << endl;
    // cout << "Valid lengths:\n" << valid_len << endl;
    // int batch_size = 2;
    // int num_steps = 5;
    // std::string file_name = "fra.txt";  // 请确认这个文件存在并位于可访问路径

    // Dataloader loader;
    // auto [batches, src_vocab, tgt_vocab] = loader->forward(batch_size, file_name, num_steps);

    // std::cout << "Number of batches: " << batches.size() << "\n";
    // std::cout << "Source vocab size: " << src_vocab.size() << "\n";
    // std::cout << "Target vocab size: " << tgt_vocab.size() << "\n";

    // if (!batches.empty()) {
    //     auto& batch = batches[0];  // 查看第一个 batch

    //     std::cout << "\n=== First Batch ===\n";

    //     std::cout << "Source Tensor:\n" << batch.src_array << "\n";
    //     std::cout << "Source Valid Lengths:\n" << batch.src_valid_len << "\n";

    //     std::cout << "Target Tensor:\n" << batch.tgt_array << "\n";
    //     std::cout << "Target Valid Lengths:\n" << batch.tgt_valid_len << "\n";
    // } else {
    //     std::cout << "No batches loaded. Check file format or content.\n";
    // }
//train
    torch::manual_seed(0);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    system("chcp 65001 > nul");
    Dataloader loader;
    auto [batches, src_vocab, tgt_vocab] = loader->forward(256, "fra.txt", 10);
    // int unk_id = tgt_vocab["<unk>"];
    // int total_tokens = 0;
    // int unk_tokens = 0;
    // for (const auto& batch : batches) {
    //     auto flat = batch.tgt_array.flatten();
    //     total_tokens += flat.size(0);
    //     unk_tokens += flat.eq(unk_id).sum().item<int>();
    // }
    // std::cout << "UNK ratio: " << (float)unk_tokens / total_tokens << std::endl;
    // cout<<batches[0].src_array;
    // cout<<tgt_vocab.size()<<endl<<src_vocab.size();
    cout<<"Train: 0 or Predict: 1"<<endl;
    bool load_only;
    cin>>load_only;
    if (load_only) {
        getchar();
        auto raw_encoder = TransformerEncoder(src_vocab.size(), 32, 32, 32, 32, vector<int64_t>{32}, 32, 64, 4, 2, 0.1,true);
        auto raw_decoder = TransformerDecoder(tgt_vocab.size(), 32, 32, 32, 32, vector<int64_t>{32}, 32, 64, 4, 2, 0.1);
        Encoder encoder = dynamic_pointer_cast<EncoderImpl>(raw_encoder.ptr());
        Decoder decoder = dynamic_pointer_cast<DecoderImpl>(raw_decoder.ptr());
        EncoderDecoder net(encoder, decoder);
        torch::load(net, "transformer_model.pt");
        net->eval();
        string src;
        while(1){
            cout<<"Please enter your English src"<<endl;
            getline(cin,src);
            if(src=="0"){
                return 0;
            }
            string result = predict_seq2seq(net, src, src_vocab, tgt_vocab, 10, device);
            cout << "Input: " << src << "\nOutput: " << result << endl;
        }
        return 0; 
    }
    auto raw_encoder = TransformerEncoder(src_vocab.size(), 32, 32, 32, 32, vector<int64_t>{32}, 32, 64, 4, 2, 0.1,true);
    auto raw_decoder = TransformerDecoder(tgt_vocab.size(), 32, 32, 32, 32, vector<int64_t>{32}, 32, 64, 4, 2, 0.1);
    Encoder encoder=dynamic_pointer_cast<EncoderImpl>(raw_encoder.ptr());
    Decoder decoder=dynamic_pointer_cast<DecoderImpl>(raw_decoder.ptr());
    EncoderDecoder net(encoder, decoder);
    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.003));
    train_seq2seq(net, batches, optimizer, 100, tgt_vocab, device);
    torch::save(net, "transformer_model.pt");
    cout<<"Running";
    string src = "hello world !";
    string translation = predict_seq2seq(net, src, src_vocab, tgt_vocab, 10, device);
    cout << "Input: " << src << "\nOutput: " << translation << endl;
    return 0;
}