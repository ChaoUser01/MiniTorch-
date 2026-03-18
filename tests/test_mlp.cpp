#include<iostream>
#include"../include/nn.hpp"

int main(){
    MLP model(3, {4, 4, 1});

    std::vector<Value> x;
    x.emplace_back(2.0f);
    x.emplace_back(3.0f);
    x.emplace_back(-1.0f);

    //Inference
    std::vector<Value> out = model(x);
    //Backprop
    out[0].backward();

    std::cout << "MLP Output: "; out[0].print();
    std::cout << "Parameters count: " << model.parameters().size() << std::endl;
    return 0;
}