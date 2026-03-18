#ifndef VALUE_HPP 
#define VALUE_HPP

#include<iostream>
#include<vector>
#include<functional>
#include<memory>
#include<set>

struct ValueImpl;

class Value{
    private:
        float data; //Actual Numerical Data to Wrap
        float grad; //The Gradient 

        //The Graph of the Network 
        //Store the pointers to the parents
        //Use mutable so we can touch them even in const funtions if needed,
        //Removing const from operators as we need to modify the other.grad too
        std::vector<Value*> children;

        //The AutoGrad Engine
        //This has the Mathematical function for the operation
        std::function<void()> _backward;
        std::shared_ptr<ValueImpl> ptr; //Need to copy a Value, just need to copy this pointer

    public:
        Value(float val); //The constructor
        Value(std::shared_ptr<ValueImpl> ptr);
        float getData() const; //The Accessor for data value
        float getGrad() const;//The Accessor for gradient
        void setData(float v);
        void setGrad(float g); //set the Gradient to g
        Value operator+(const Value& other) const; //operator overloading
        Value operator*(const Value& other) const;
        Value operator-(const Value& other) const;
        Value operator/(const Value& other) const;
        Value pow(float exponent);
        Value exp();
        Value log();
        Value tanh();
        void backward(); //Let there be chain reaction
        //void build_topo(std::vector<Value*>& topo, std::set<Value*>& visited);//Helper for topological Sorting
        void print() const; //Just for printing nicely
        std::shared_ptr<ValueImpl> getPtr() const {return ptr;}

        Value ReLU();
        Value LeakyReLU();
        
        // Memory management - clears the computation graph to free memory
        // Call this after backward() when gradients are no longer needed
        void clearGraph();

};

#endif