#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "nn.hpp"
#include <vector>

class Optimizer{
    public:
        virtual void step() = 0;
        virtual void zero_grad() = 0;
};

class SGD : public Optimizer{
    private:
        std::vector<Value*> parameters;
        float lr;
    public:
        SGD(std::vector<Value*> params, float learning_rate)
            : parameters(params), lr(learning_rate) {}
        
        void step() override{
            for (Value* p : parameters){
                //w = w - lr * grad
                float current = p->getData();
                float change = lr * p->getGrad();
                p->setData(current - change);
            }
        }

        void zero_grad() override {
            for(Value* p : parameters){
                p->setGrad(0.0f);
            }
        }
};
#endif