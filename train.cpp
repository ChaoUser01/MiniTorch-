#include <iostream>
#include <vector>
#include "include/nn.hpp"
#include <ctime>

int main() {
    srand(time(0)); //seeds random number generate with current time
    // 1. Architecture
    MLP model(2, {4, 4, 1}); 

    // 2. Dataset (XOR problem)
    std::vector<std::vector<Value>> inputs = {
        {Value(2.0f), Value(3.0f)},
        {Value(2.0f), Value(-1.0f)},
        {Value(-0.5f), Value(0.5f)},
        {Value(1.0f), Value(1.0f)}
    };

    std::vector<float> targets = {1.0f, 0.0f, 0.0f, 1.0f};

    // Training Loop
    std::cout << "Starting Training..." << std::endl;

    for (int k = 0; k < 5000; ++k) { 
        
        Value total_loss(0.0f);

        //Forward Pass
        // no update weights here, just measure error.
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<Value> out = model(inputs[i]);
            
            // Difference = Pred - Target
            Value diff = out[0] + Value(-targets[i]); 
            Value loss = diff * diff;
            
            total_loss = total_loss + loss;
        }

        // Backward Pass
        model.zero_grad();
        total_loss.backward();

        //Optimizer (Update Weights)
        float learning_rate = 0.01f; // Lowered learning rate for safety
        
        for (Value* p : model.parameters()) {
            float current = p->getData();
            float change = learning_rate * p->getGrad();
            p->setData(current - change);
        }

        // Print progress every 10 epochs
        if (k % 10 == 0) {
            std::cout << "Epoch " << k << " | Loss: " << total_loss.getData() << std::endl;
        }
    }

    return 0;
}