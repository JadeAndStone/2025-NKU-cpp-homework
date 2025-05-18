#include<torch/script.h>
#include<torch/torch.h>
#include<iostream>
#include<vector>
#include"MNISTDataset.h"
using namespace std;
class Linear_blockImpl: public torch::nn::Module{
    torch::nn::Linear ln{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};
    public:
        Linear_blockImpl(){}
        Linear_blockImpl(int num_input,int num_output){
            ln=register_module("ln",torch::nn::Linear(torch::nn::LinearOptions(num_input,num_output)));
            bn=register_module("bn",torch::nn::BatchNorm1d(num_output));
        }
        torch::Tensor forward(torch::Tensor x){
            x=torch::relu(ln->forward(x));
            x=bn(x);
            return x;
        }
};
TORCH_MODULE(Linear_block);
class MLPImpl: public torch::nn::Module{
    Linear_block b1,b2,b3;
    torch::nn::Linear out_layer{nullptr};
    int num_hidden[3]={128,64,32};
    public:
        MLPImpl(int num_input,int num_output){
            b1=Linear_block(num_input,num_hidden[0]);
            b2=Linear_block(num_hidden[0],num_hidden[1]);
            b3=Linear_block(num_hidden[1],num_hidden[2]);
            out_layer=torch::nn::Linear(num_hidden[2],num_output);

            b1=register_module("b1",b1);
            b2=register_module("b2",b2);
            b3=register_module("b3",b3);
            out_layer=register_module("out_layer",out_layer);
        }
        torch::Tensor forward(torch::Tensor x){
            x=b1->forward(x);
            x=b2->forward(x);
            x=b3->forward(x);
            return out_layer->forward(x);
        }
};
TORCH_MODULE(MLP);
auto dataloader(torch::data::datasets::MapDataset<MNISTDataset, torch::data::transforms::Stack<torch::data::Example<>>> dataset){
    return torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(64)
    );
}
int main()
{
    // torch::Tensor output;
    // cout << "cuda is_available: " << torch::cuda::is_available() << endl;
    // torch::DeviceType device = at::kCPU; 
    // if (torch::cuda::is_available())
    //     device = at::kCUDA;
    // output = torch::randn({ 3,3 }).to(device);
    // cout<<output<<endl;
    // cout << "Torch version: " << TORCH_VERSION << std::endl;
    torch::DeviceType device=at::kCPU;
    if(torch::cuda::is_available()){
        device=at::kCUDA;
    }
    MLP net(784,10);
    net->to(device);
    auto train_raw_dataset=MNISTDataset(
        "D:/d2l from the scratch/number_mnist/MNIST/raw/train-images-idx3-ubyte",
        "D:/d2l from the scratch/number_mnist/MNIST/raw/train-labels-idx1-ubyte"
    );
    auto test_raw_dataset=MNISTDataset(
        "D:/d2l from the scratch/number_mnist/MNIST/raw/t10k-images-idx3-ubyte",
        "D:/d2l from the scratch/number_mnist/MNIST/raw/t10k-labels-idx1-ubyte"
    );
    
    // std::cout << "Total samples: " << raw_dataset.size().value() << std::endl;
    // for (size_t i = 0; i < 5; ++i) {
    //     auto example = raw_dataset.get(i);
    //     std::cout << "Image size: " << example.data.sizes()
    //               << ", Label: " << example.target.item<int>() << std::endl;
    // }
    
    // 应用于数据加载器的 dataset
    auto train_dataset=train_raw_dataset.map(torch::data::transforms::Stack<>()),
        test_dataset=test_raw_dataset.map(torch::data::transforms::Stack<>());
    
    auto train_loader=dataloader(train_dataset),test_loader=dataloader(test_dataset);

    // int count = 0;
    // for (auto& batch : *data_loader) {
    //     cout << "Got a batch, size: " << batch.data.sizes() << endl;
    //     if (++count >= 3) break;
    // }
    torch::optim::SGD optimizer(net->parameters(),0.01);
    for(int epoch=1;epoch<=10;epoch++){
        auto train_loss=torch::zeros(1,device);
        int num_train=0;
        net->train();
        for(auto &batch:*train_loader){
            optimizer.zero_grad();
            torch::Tensor y_hat=net->forward(batch.data.to(device).reshape({batch.data.size(0),-1}));
            torch::Tensor loss=torch::cross_entropy_loss(y_hat,batch.target);
            loss.backward();
            optimizer.step();
            train_loss+=loss;
            num_train++;
        }
        net->eval();
        auto test_loss=torch::zeros(1,device);
        auto test_accuracy=test_loss.clone();
        int num_test=0;
        for(auto &batch:*test_loader){
            test_loss+=torch::cross_entropy_loss(net->forward(batch.data.to(device).reshape({batch.data.size(0),-1})),batch.target.to(device));
            test_accuracy+=(net->forward(batch.data.to(device).reshape({batch.data.size(0),-1})).argmax(-1)==batch.target).sum().item<float>()/batch.data.size(0);
            num_test++;
        }
        cout<<"epoch: "<<epoch<<endl<<"train_loss: "<<(train_loss/num_train).item<float>()<<endl;
        cout<<"test_loss: "<<(test_loss/num_test).item<float>()<<endl;
        cout<<"test_accuracy: "<<(test_accuracy/num_test).item<float>()<<endl;
    }
    cout<<torch::cuda::is_available();
    // system("echo %PATH%");
    return 0;
}