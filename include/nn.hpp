#ifndef NN_HPP
#define NN_HPP

#include "value.hpp"
#include<vector>
#include<iostream>
#include<random>

class Module{
    public:
    //Every neural netwoek part must be able to return its parameter (weights)
    //so we can update them later (Gradient descent)
    virtual std::vector<Value> operator()(const std::vector<Value>& x) = 0;
    virtual std::vector<Value*> parameters() = 0;

    //Helper to reset all gradient to 0
    void zero_grad(){
        for (Value* p : parameters()){
            p->setGrad(0.0f);
        }
    }
};

/* class Neuron : public Module{
    private:
    std::vector<Value> w; //Weights
    Value b; //Bias

    public:
    // nin = number of inputs from inputs or previous layer
    Neuron(int nin);
    //The forward pass: Calculate (w*x + b) -> relu
    Value operator()(const std::vector<Value>& x);
    std::vector<Value*> parameters();

};
class Layer : public Module{
    private:
        std::vector<Neuron> neurons;
    
    public:
    //nin = number of input per neuron
    //nout = number of neurons in this layer
    Layer(int nin, int nout);
    //Forward pass: returns a vector of outputs 
    std::vector<Value> operator()(const std::vector<Value>& x);
    std::vector<Value*> parameters();
};
class MLP : public Module{
    private:
        std::vector<Layer> layers;
    
    public:
    //nin = size of input layer
    //nouts = list of sizes for subsequent layers
    //Example: MLP(3, {4, 4, 1}) -> Input 3 features -> Hidden 4 -> Hidden 4 -> Ouput 1
    MLP(int nin, std::vector<int>nouts);

    std::vector<Value> operator()(std::vector<Value> x);
    std::vector<Value*> parameters();

}; */ //OLD CODE

//Linear Layer
//Pure w*x + b no activation
class Linear : public Module {
    struct Neuron{
        std::vector<Value> w;
        Value b;
        Neuron(int nin);
        Value operator()(const std::vector<Value>& x);
        std::vector<Value*> parameters();
    };
    std::vector<Neuron> neurons;

public:
    Linear(int nin, int nout);
    std::vector<Value> operator()(const std::vector<Value>& x) override;
    std::vector<Value*> parameters() override;
};

class ReLU : public Module{
public:
    std::vector<Value> operator()(const std::vector<Value>& x) override;
    std::vector<Value*> parameters() override; //Returns empty
};

class Tanh : public Module {
    public:
        std::vector<Value> operator()(const std::vector<Value>& x) override;
        std::vector<Value*> parameters() override;
};

class Sigmoid : public Module {
    public:
        std::vector<Value> operator()(const std::vector<Value>& x) override;
        std::vector<Value*> parameters() override;
};

class Softmax : public Module {
    public:
        std::vector<Value> operator()(const std::vector<Value>& x) override;
        std::vector<Value*> parameters() override { return {};}
};
// Sequential Container
//Chains layer together
class Sequential : public Module{
    std::vector<Module*> modules;
public:
    Sequential(std::vector<Module*> mods);
    std::vector<Value> operator()(const std::vector<Value>& x) override;
    std::vector<Value*> parameters() override;
};
#endif