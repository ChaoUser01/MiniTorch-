#include<iostream>
#include"../include/value.hpp"

int main(){
    Value weight(2.5f);
    Value a(2.0f);
    Value b(3.0f);

    //stress test value
    //e = a*b    
    //d = e + e
    Value e = a * b;
    Value d = e + e;
    d.backward();
    std::cout << "THE DIAMOND TEST" << std::endl;
    std::cout << "Value d: "; d.print();
    std::cout << "Grad a: "; a.print();
    std::cout << "Grad b: "; b.print();
    
    //forward pass
    Value term = weight * a;
    Value perceptron = term + b;

    perceptron.setGrad(1.0f);
    perceptron.backward();

    std::cout << "Automatic Gradients" << std::endl;
    std::cout << "Grad of the perceptron: "; perceptron.print();
    std::cout << "Grad of the weight: "; weight.print();
    std::cout << "Grad of the data"; a.print();
    std::cout << "Grad of the bias: "; b.print();

    return 0;
}