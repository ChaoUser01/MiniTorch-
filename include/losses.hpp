#ifndef LOSSES_HPP
#define LOSSES_HPPP

#include "value.hpp"
#include <vector>

namespace Loss {
    //Standard MSE
    inline Value MSE(Value pred, Value target){
        Value diff = pred + (target * Value(-1.0f));
        return diff * diff;
    }

    //Cross Entropy Loss
    inline Value CrossEntropy(const std::vector<Value>& preds, int target){
        //Care only about the probabilities of the correct class
        //The loss is simply -log(p_correct)
        Value p_correct = preds[target];
        return p_correct.log() * Value(-1.0f);

        //ON PAPER CLACULATION FOR SHORTENING THE CROSS ENTROPY CLASS
        // CEL = - sum(ylog(p)) for given classes
        //Target y is a one-hot vector, meaning for Class 2 out of 3 Classes
        // y = [0, 0 , 1] p= [0.1, 0.2, 0.7]
        //Sum : CLE = -(0 * log(0.1) + 0 * log(0.2) + 1 * log(0.7)) = -log(0.7)
        //so this the whole idea behind not adding loops instead a simple grab and log was done
    }
}

#endif