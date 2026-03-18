#include "nn.hpp"
#include <cstdlib>
#include<cmath>

//Standard Normal Distribution Generator
float randn(){
    //Box-Muller transform to get Gaussian distribution
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() /RAND_MAX;
    return sqrt(-2.0f *log(u1)) *cos(2.0f * 3.1415926f*u2);
}
/* Neuron::Neuron(int nin) : b(0.0f){
    //Kaiming initialization 
    //Scale = sqrt(2 / number of inputs)
    float scale = sqrt(2.0f / (float)nin);
    for (int i = 0; i < nin; ++i){
        float r = ((float)rand() / (float)RAND_MAX) - 0.5f;
        w.emplace_back(r);
    }
}
Value Neuron::operator()(const std::vector<Value>& x){
    //w * x + b
    Value act = b;
    for (size_t i = 0; i < w.size(); ++i){
        //Assuming x and w are the same size.
        Value prod = w[i] * x[i];
        act = act + prod;
    }
    return act.LeakyReLU();
}
std::vector<Value*> Neuron::parameters(){
    std::vector<Value*> params;
    for (auto& weight : w){
        params.push_back(&weight);
    }
    params.push_back(&b);
    return params;
}
 */
/* //Layer Implementation
Layer::Layer(int nin, int nout){
    for(int i = 0; i < nout; i++){
        neurons.emplace_back(Neuron(nin));
    }
}
std::vector<Value> Layer::operator()(const std::vector<Value>& x){
    std::vector<Value> outs;
    for (auto& n : neurons){
        outs.push_back(n(x));
    }
    return outs;

}
std::vector<Value*> Layer::parameters(){
    std::vector<Value*> params;
    for(auto& n : neurons){
        auto n_params = n.parameters();
        params.insert(params.end(), n_params.begin(), n_params.end());

    }
    return params;
}

//MLP Implementation
MLP::MLP(int nin, std::vector<int> nouts){
    std::vector<int> sz = nouts;
    sz.insert(sz.begin(), nin); //[nin, nouts]

    //Iterate though sizes to connect layers
    for(size_t i = 0; i < nouts.size(); ++i){
        //layer i takes sz[i] inputs and produce sz[i+1] outputs
        layers.emplace_back(Layer(sz[i], sz[i+1]));
    }

}
std::vector<Value> MLP::operator()(std::vector<Value> x){
    for (auto& layer : layers){
        x = layer(x); //The output of on layer is the input to the next
    }
    return x;
}
std::vector<Value*> MLP::parameters(){
    std::vector<Value*> params;
    for(auto& layer : layers){
        auto l_params = layer.parameters();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
} */ //OLD LAYER TECHNIQUES AND MULTI LAYER PERCEPTRON
Linear::Neuron::Neuron(int nin) : b(0.0f) {
    // Kaiming Init: Scale by sqrt(2/inputs)
    float scale = sqrt(2.0f / (float)nin);
    for (int i = 0; i < nin; ++i) {
        w.emplace_back(randn() * scale);
    }
}

Value Linear::Neuron::operator()(const std::vector<Value>& x) {
    Value act = b;
    for (size_t i = 0; i < w.size(); ++i) {
        act = act + (w[i] * x[i]);
    }
    return act; // <--- NO RELU HERE! Pure linear math.
}

std::vector<Value*> Linear::Neuron::parameters() {
    std::vector<Value*> params;
    for (auto& weight : w) params.push_back(&weight);
    params.push_back(&b);
    return params;
}

// --- Linear Layer ---
Linear::Linear(int nin, int nout) {
    for (int i = 0; i < nout; ++i) {
        neurons.emplace_back(Neuron(nin));
    }
}

std::vector<Value> Linear::operator()(const std::vector<Value>& x) {
    std::vector<Value> outs;
    for (auto& n : neurons) outs.push_back(n(x));
    return outs;
}

std::vector<Value*> Linear::parameters() {
    std::vector<Value*> params;
    for (auto& n : neurons) {
        auto p = n.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

// ReLU Layer
std::vector<Value> ReLU::operator()(const std::vector<Value>& x) {
    std::vector<Value> outs;
    for (const auto& v : x) {
        // We create a COPY of the value to apply ReLU, 
        // effectively adding a node to the graph
        Value v_copy = v; 
        outs.push_back(v_copy.ReLU());
    }
    return outs;
}

std::vector<Value*> ReLU::parameters() {
    return {}; // No weights in ReLU
}
//Tanh Layer
std::vector<Value> Tanh::operator()(const std::vector<Value>& x){
    std::vector<Value> outs;
    for (const auto& v : x){
        Value v_copy = v;
        outs.push_back(v_copy.tanh());
    }
    return outs;
}

std::vector<Value*> Tanh::parameters() {return {};}

std::vector<Value> Sigmoid::operator()(const std::vector<Value>& x) {
    std::vector<Value> outs;
    for (const auto& v : x) {
        Value v_copy = v;
        // 1.0 / (1.0 + exp(-x))
        Value one(1.0f);
        Value neg_one(-1.0f);
        Value denom = one + (v_copy * neg_one).exp(); // 1 + e^-x
        Value out = one * denom.pow(-1.0f);           // 1 * denom^-1 (division)
        
        outs.push_back(out);
    }
    return outs;
}

std::vector<Value*> Sigmoid::parameters() {return {};}

std::vector<Value> Softmax::operator()(const std::vector<Value>& x){
    std::vector<Value> outs;
    //Calculate exp() for all
    std::vector<Value> exps;
    Value sum_exp(0.0f);

    for (const auto& v : x){
        Value v_copy = v;
        Value e= v_copy.exp();
        exps.push_back(e);
        sum_exp = sum_exp + e;
    }
    //Divide each by sum
    for (const auto& e: exps){
        // (e / sum) = e * sum^-1
        outs.push_back(e * sum_exp.pow(-1.0f));
    }
    return outs;
}

// Sequential Container
Sequential::Sequential(std::vector<Module*> mods) : modules(mods) {}

std::vector<Value> Sequential::operator()(const std::vector<Value>& x) {
    std::vector<Value> out = x;
    for (auto m : modules) {
        out = (*m)(out);
    }
    return out;
}

std::vector<Value*> Sequential::parameters() {
    std::vector<Value*> params;
    for (auto m : modules) {
        auto p = m->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}