#include <iostream>
#include <vector>
#include <ctime>
#include "../include/nn.hpp"
#include "../include/optimizer.hpp"
#include "../include/losses.hpp"

int main() {
    srand(time(0));

    // 1. SIMPLIFIED ARCHITECTURE
    // 2 Inputs -> 4 Hidden (ReLU) -> 1 Output (Sigmoid)
    // map the output to 0..1 directly.
    Linear l1(2, 4);
    ReLU r1;
    Linear l2(4, 1); 
    Sigmoid s1;

    std::vector<Module*> layers = { &l1, &r1, &l2, &s1 };
    Sequential model(layers);

    // 2. DATA
    std::vector<std::vector<Value>> inputs = {
        {Value(0.0f), Value(0.0f)},
        {Value(0.0f), Value(1.0f)},
        {Value(1.0f), Value(0.0f)},
        {Value(1.0f), Value(1.0f)}
    };
    // Targets are simple values now, not classes
    std::vector<float> targets = {0.0f, 1.0f, 1.0f, 0.0f};

    // 3. OPTIMIZER 
    SGD optimizer(model.parameters(), 0.5f); 

    std::cout << "--- Starting Sigmoid Regression ---" << std::endl;

    for (int k = 0; k < 5000; ++k) {
        
        Value total_loss(0.0f);
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<Value> out = model(inputs[i]);
            
            // MSE LOSS: (Pred - Target)^2
            // We use the simpler regression loss
            Value diff = out[0] + Value(-targets[i]);
            Value loss = diff * diff;
            
            total_loss = total_loss + loss;
        }

        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();

        if (k % 500 == 0) {
            std::cout << "Epoch " << k << " | Loss: " << total_loss.getData() << std::endl;
        }
    }
    
    // Results
    std::cout << "\n--- Final Results ---" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        float pred = model(inputs[i])[0].getData();
        std::cout << "Input " << i << ": " << pred 
                  << " (Target: " << targets[i] << ")" << std::endl;
    }
    
    return 0;
}