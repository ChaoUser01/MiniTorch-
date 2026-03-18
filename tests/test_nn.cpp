#include<iostream>
#include "../include/nn.hpp"
#include"../include/value.hpp"

int main(){
    Neuron n(2);  //Two inputs

    std::vector<Value> x;
    x.emplace_back(Value(2.0f));
    x.emplace_back(Value(3.0f));

    Value out = n(x);
    out.setGrad(1.0f);
    out.backward();

    std::cout << "Neuron Ouptut: "; 
    out.print();
    std::cout << "Parameters have gradients" << std::endl;

    return 0;
}