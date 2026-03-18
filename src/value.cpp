#include "../include/value.hpp"
#include <algorithm>
#include <cmath>

//The body
struct ValueImpl{
    float data;
    float grad;
    std::vector<std::shared_ptr<ValueImpl>> childern; //Keep the children alive!
    std::function<void()> _backward;

    //constuctor
    ValueImpl(float v) : data(v), grad(0.0f), _backward(nullptr) {}
};

//The handling
Value::Value(float val){
    ptr = std::make_shared<ValueImpl>(val);
}
//Internal constructor
Value::Value(std::shared_ptr<ValueImpl> p) : ptr(p) {}

//Accessor
float Value::getData() const {return ptr->data;}
float Value::getGrad() const {return ptr->grad;}
void Value::setGrad(float g) {ptr->grad = g;}

//Operations
Value Value::operator+(const Value& other) const{
    //New result
    Value out(ptr->data + other.ptr->data);

    //Buildig graph using shared pointers
    //Now out holds a strong reference to this and other
    //cannot die as long as out lives
    out.ptr->childern.push_back(ptr);
    out.ptr->childern.push_back(other.ptr);

    //Backward Logic
    //Capture Smart pointers, not raw this
    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> oth = other.ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, oth, res](){
        self->grad += 1.0f * res->grad;
        oth->grad += 1.0f * res->grad;
    };
    return out;
}

Value Value::operator*(const Value& other) const{
    Value out(ptr->data * other.ptr->data);

    out.ptr->childern.push_back(ptr);
    out.ptr->childern.push_back(other.ptr);

    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> oth = other.ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, oth, res](){
        self->grad += oth->data * res->grad;
        oth->grad += self->data * res->grad;
    };
    return out;
}

Value Value::operator-(const Value& other) const{
    return *this + (other * Value(-1.0f));
}

Value Value::exp(){
    float x = ptr->data;
    Value out(std::exp(x));
    out.ptr->childern.push_back(ptr);

    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, res](){
        //d(;e^x)/dx = e^x = out.data
        self->grad += res->data * res->grad;
    };
    return out;
}

Value Value::pow(float exponent){
    float x = ptr->data;
    Value out(std::pow(x, exponent));
    out.ptr->childern.push_back(ptr);

    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, res, exponent](){
        // d(x^n)/dx = n * x^(n-1)
        float base = self->data;
        //Handle 0^negative edge case safely
        float deriv = exponent * std::pow(base, exponent - 1);
        self->grad += deriv * res->grad;
    };
    return out;
}

Value Value::log(){
    float x = ptr->data;
    Value out(std::log(x));
    out.ptr->childern.push_back(ptr);

    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, res](){
        //dL/dx = 1/x * dL/dy
        float x_val = self->data;
        self->grad += (1.0f / x_val) * res->grad; 
    };
    return out;
}

Value Value::tanh(){
    float x = ptr->data;
    float t = std::tanh(x);
    Value out(x);

    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, res](){
        float t = res->data;
        self->grad += (1.0f -t * t) * res->grad;
    };
    return out;
}

Value Value::ReLU(){
    float v = (ptr->data < 0) ? 0: ptr->data;
    Value out(v);

    out.ptr->childern.push_back(ptr);

    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;

    out.ptr->_backward = [self, res]() {
        if(self->data > 0){
            self->grad += res->grad;
        }
    };
    return out;
}
Value Value::LeakyReLU(){
    // Forward Pass: Leaky logic
    // If negative, don't return 0. Return 0.01 * data.
    float v = (ptr->data < 0) ? (ptr->data * 0.01f) : ptr->data;
    Value out(v);
    
    out.ptr->childern.push_back(ptr);
    
    std::shared_ptr<ValueImpl> self = ptr;
    std::shared_ptr<ValueImpl> res = out.ptr;
    
    // 2. Backward Logic
    out.ptr->_backward = [self, res]() {
        // If data was positive, gradient flows 100%
        // If data was negative, gradient flows at 1% (The Leak)
        float factor = (self->data < 0) ? 0.01f : 1.0f;
        
        self->grad += factor * res->grad;
    };
    
    return out;
}

//Topological Sort Engine
//Helper function
void build_topo(std::shared_ptr<ValueImpl> v,
                std::vector<std::shared_ptr<ValueImpl>>& topo,
                std::set<std::shared_ptr<ValueImpl>>& visited){
                    if (visited.find(v) != visited.end()) return;
                        visited.insert(v);
                    for (auto child: v->childern){
                        build_topo(child, topo, visited);
                    }
                    topo.push_back(v);
                }
void Value::backward(){
    std::vector<std::shared_ptr<ValueImpl>> topo;
    std::set<std::shared_ptr<ValueImpl>> visited;

    build_topo(ptr, topo, visited);
    std::reverse(topo.begin(), topo.end());
    ptr->grad = 1.0f;
    for (auto v : topo){
        if (v->_backward){
            v->_backward();
        }
    }
}
void Value::print() const{
    std::cout << "Value(data=" << ptr->data << ", grad=" << ptr->grad << ")" << std::endl; 
}
void Value::setData(float v){
    ptr->data = v;
}

// Clear computation graph to free memory
// traverses the graph and clears all backward functions and children references
void Value::clearGraph() {
    std::vector<std::shared_ptr<ValueImpl>> topo;
    std::set<std::shared_ptr<ValueImpl>> visited;
    
    // Build topological order to visit all nodes
    build_topo(ptr, topo, visited);
    
    // Clear all nodes in the graph
    for (auto& v : topo) {
        v->childern.clear();      // Release children references
        v->_backward = nullptr;   // Clear backward function, releasing captured refs
    }
}