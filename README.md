# MiniTorch++

A lightweight, educational **automatic differentiation engine** and **neural network library** written in pure C++. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
  - [Value (Autograd Engine)](#value-autograd-engine)
  - [Neural Network Layers](#neural-network-layers)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
- [Mathematical Foundations](#mathematical-foundations)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Backpropagation Algorithm](#backpropagation-algorithm)
  - [Gradient Computation for Each Operation](#gradient-computation-for-each-operation)
- [Usage Examples](#usage-examples)
- [Building & Running](#building--running)
- [API Reference](#api-reference)

---

## Overview

MiniTorch++ implements a **scalar-valued autograd engine** that tracks computations and automatically computes gradients using reverse-mode differentiation (backpropagation). On top of this engine, it provides PyTorch-style neural network primitives.

### Key Features

| Feature | Description |
|---------|-------------|
| **Automatic Differentiation** | Tracks computation graphs and computes gradients automatically |
| **Dynamic Graphs** | Builds graphs on-the-fly during forward pass |
| **PyTorch-like API** | `Module`, `Sequential`, `Linear`, activation layers |
| **Multiple Activations** | ReLU, LeakyReLU, Tanh, Sigmoid, Softmax |
| **Loss Functions** | MSE, Cross-Entropy |
| **Optimizers** | SGD with configurable learning rate |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           MiniTorch++                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│   │    Value     │     │    Module    │     │  Optimizer   │         │
│   │  (Autograd)  │     │  (Abstract)  │     │  (Abstract)  │         │
│   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘         │
│          │                    │                    │                 │
│          │             ┌──────┴──────┐             │                 │
│          │             │             │             │                 │
│          │      ┌──────┴──┐    ┌─────┴─────┐       │                 │
│          │      │ Linear  │    │ Sequential│       │                 │
│          │      └─────────┘    └───────────┘       │                 │
│          │             │                           │                 │
│          │      ┌──────┴───────────────────┐       │                 │
│          │      │     Activation Layers    │       │                 │
│          │      │ ReLU│Tanh│Sigmoid│Softmax│       │                 │
│          │      └──────────────────────────┘       │                 │
│          │                                         │                 │
│          │      ┌──────────────────┐        ┌──────┴──────┐          │
│          │      │   Loss Functions │        │     SGD     │          │
│          │      │   MSE│CrossEntropy│       └─────────────┘          │
│          │      └──────────────────┘                                 │
│          │                                                           │
└──────────┴───────────────────────────────────────────────────────────┘
```

### File Structure

```
MiniTorch++/
├── include/
│   ├── value.hpp      # Value class declaration (autograd core)
│   ├── nn.hpp         # Neural network modules (Linear, activations, Sequential)
│   ├── losses.hpp     # Loss functions (MSE, CrossEntropy)
│   └── optimizer.hpp  # Optimizers (SGD)
├── src/
│   ├── value.cpp      # Value class implementation (operations, backward)
│   └── nn.cpp         # Neural network implementation
├── tests/
│   ├── test_value.cpp # Value/autograd tests
│   ├── test_nn.cpp    # Neural network tests
│   └── test_mlp.cpp   # MLP integration tests
└── train.cpp          # Training example
```

---

## Module Reference

### Value (Autograd Engine)

The `Value` class wraps scalar floats and tracks all operations to enable automatic gradient computation.

```cpp
#include "include/value.hpp"

Value a(2.0f);      // Create a Value wrapping 2.0
Value b(3.0f);      // Create a Value wrapping 3.0
Value c = a + b;    // c = 5.0, builds computation graph
Value d = c * a;    // d = 10.0, extends the graph

d.backward();       // Compute all gradients

// Access results
std::cout << a.getData();  // 2.0 (the value)
std::cout << a.getGrad();  // Gradient ∂d/∂a
```

#### Value Operations

| Operation | Code | Math | Gradient |
|-----------|------|------|----------|
| Addition | `a + b` | $z = a + b$ | $\frac{\partial z}{\partial a} = 1$ |
| Subtraction | `a - b` | $z = a - b$ | $\frac{\partial z}{\partial a} = 1$ |
| Multiplication | `a * b` | $z = a \cdot b$ | $\frac{\partial z}{\partial a} = b$ |
| Division | `a / b` | $z = \frac{a}{b}$ | $\frac{\partial z}{\partial a} = \frac{1}{b}$ |
| Power | `a.pow(n)` | $z = a^n$ | $\frac{\partial z}{\partial a} = n \cdot a^{n-1}$ |
| Exponential | `a.exp()` | $z = e^a$ | $\frac{\partial z}{\partial a} = e^a$ |
| Logarithm | `a.log()` | $z = \ln(a)$ | $\frac{\partial z}{\partial a} = \frac{1}{a}$ |
| Tanh | `a.tanh()` | $z = \tanh(a)$ | $\frac{\partial z}{\partial a} = 1 - z^2$ |
| ReLU | `a.ReLU()` | $z = \max(0, a)$ | $\frac{\partial z}{\partial a} = \mathbb{1}_{a>0}$ |
| LeakyReLU | `a.LeakyReLU()` | $z = \max(0.01a, a)$ | $\frac{\partial z}{\partial a} = \begin{cases}1 & a>0\\0.01 & a\leq0\end{cases}$ |

---

### Neural Network Layers

#### Module (Abstract Base Class)

All neural network components inherit from `Module`:

```cpp
class Module {
public:
    virtual std::vector<Value> operator()(const std::vector<Value>& x) = 0;
    virtual std::vector<Value*> parameters() = 0;
    void zero_grad();  // Reset all gradients to 0
};
```

#### Linear Layer

Fully connected layer computing $y = Wx + b$ (no activation).

```cpp
#include "include/nn.hpp"

Linear fc(3, 4);  // 3 inputs → 4 outputs

std::vector<Value> input = {Value(1.0f), Value(2.0f), Value(3.0f)};
std::vector<Value> output = fc(input);  // output.size() == 4
```

**Weight Initialization**: Uses Kaiming initialization with scale $\sqrt{\frac{2}{n_{in}}}$

---

### Activation Functions

All activations are `Module` subclasses with no learnable parameters.

#### ReLU

```cpp
ReLU relu;
std::vector<Value> out = relu(input);  // max(0, x) element-wise
```

$$\text{ReLU}(x) = \max(0, x)$$

#### Tanh

```cpp
Tanh tanh_layer;
std::vector<Value> out = tanh_layer(input);  // tanh(x) element-wise
```

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

#### Sigmoid

```cpp
Sigmoid sigmoid;
std::vector<Value> out = sigmoid(input);  // σ(x) element-wise
```

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

#### Softmax

```cpp
Softmax softmax;
std::vector<Value> probs = softmax(logits);  // Probabilities sum to 1
```

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

---

### Sequential Container

Chains multiple modules together:

```cpp
Linear fc1(2, 4);
ReLU relu;
Linear fc2(4, 1);

Sequential model({&fc1, &relu, &fc2});

std::vector<Value> input = {Value(1.0f), Value(2.0f)};
std::vector<Value> output = model(input);
```

---

### Loss Functions

Located in namespace `Loss`:

```cpp
#include "include/losses.hpp"
```

#### Mean Squared Error (MSE)

```cpp
Value pred(2.5f);
Value target(3.0f);
Value loss = Loss::MSE(pred, target);  // (pred - target)²
```

$$\text{MSE} = (y_{pred} - y_{target})^2$$

#### Cross-Entropy Loss

For classification with softmax outputs:

```cpp
std::vector<Value> probs = {Value(0.1f), Value(0.2f), Value(0.7f)};
int target_class = 2;
Value loss = Loss::CrossEntropy(probs, target_class);  // -log(probs[2])
```

$$\text{CE} = -\log(p_{target})$$

---

### Optimizers

#### SGD (Stochastic Gradient Descent)

```cpp
#include "include/optimizer.hpp"

Sequential model({...});
SGD optimizer(model.parameters(), 0.01f);  // lr = 0.01

// Training loop
optimizer.zero_grad();    // Reset gradients
loss.backward();          // Compute gradients
optimizer.step();         // Update weights: w = w - lr * grad
```

---

## Mathematical Foundations

### Automatic Differentiation

MiniTorch++ uses **reverse-mode automatic differentiation** (also called backpropagation). Every `Value` object maintains:

1. **`data`**: The scalar value
2. **`grad`**: The gradient $\frac{\partial L}{\partial \text{this}}$
3. **`children`**: Pointers to input nodes (for graph traversal)
4. **`_backward`**: A closure that computes local gradients

### Backpropagation Algorithm

When you call `loss.backward()`:

```
1. Build topological order of computation graph (DFS post-order)
2. Reverse the order (outputs → inputs)
3. Set loss.grad = 1.0 (seed gradient)
4. For each node in reverse topological order:
     Execute node._backward() to propagate gradients to children
```

```cpp
void Value::backward() {
    // Step 1 & 2: Topological sort
    std::vector<std::shared_ptr<ValueImpl>> topo;
    std::set<std::shared_ptr<ValueImpl>> visited;
    build_topo(ptr, topo, visited);
    std::reverse(topo.begin(), topo.end());
    
    // Step 3: Seed
    ptr->grad = 1.0f;
    
    // Step 4: Propagate
    for (auto v : topo) {
        if (v->_backward) v->_backward();
    }
}
```

### Gradient Computation for Each Operation

#### Chain Rule Foundation

For $L \to y \to x$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

In code: `x.grad += local_derivative * y.grad`

#### Addition: $z = x + y$

```cpp
out.ptr->_backward = [self, oth, res]() {
    self->grad += 1.0f * res->grad;  // ∂z/∂x = 1
    oth->grad  += 1.0f * res->grad;  // ∂z/∂y = 1
};
```

#### Multiplication: $z = x \cdot y$

```cpp
out.ptr->_backward = [self, oth, res]() {
    self->grad += oth->data * res->grad;   // ∂z/∂x = y
    oth->grad  += self->data * res->grad;  // ∂z/∂y = x
};
```

#### Power: $z = x^n$

```cpp
out.ptr->_backward = [self, res, exponent]() {
    float deriv = exponent * std::pow(self->data, exponent - 1);
    self->grad += deriv * res->grad;  // ∂z/∂x = n·x^(n-1)
};
```

#### Exponential: $z = e^x$

```cpp
out.ptr->_backward = [self, res]() {
    self->grad += res->data * res->grad;  // ∂z/∂x = e^x = z
};
```

#### ReLU: $z = \max(0, x)$

```cpp
out.ptr->_backward = [self, res]() {
    if (self->data > 0) {
        self->grad += res->grad;  // ∂z/∂x = 1 if x > 0
    }
    // else: ∂z/∂x = 0, nothing added
};
```

---

## Usage Examples

### Training an MLP

```cpp
#include <iostream>
#include "include/nn.hpp"
#include "include/losses.hpp"
#include "include/optimizer.hpp"

int main() {
    srand(time(0));
    
    // 1. Define architecture
    Linear fc1(2, 4);
    ReLU relu;
    Linear fc2(4, 1);
    Sequential model({&fc1, &relu, &fc2});
    
    // 2. Create optimizer
    SGD optimizer(model.parameters(), 0.01f);
    
    // 3. Training data
    std::vector<std::vector<Value>> X = {
        {Value(0.0f), Value(0.0f)},
        {Value(0.0f), Value(1.0f)},
        {Value(1.0f), Value(0.0f)},
        {Value(1.0f), Value(1.0f)}
    };
    std::vector<float> y = {0.0f, 1.0f, 1.0f, 0.0f};  // XOR targets
    
    // 4. Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        Value total_loss(0.0f);
        
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<Value> pred = model(X[i]);
            Value loss = Loss::MSE(pred[0], Value(y[i]));
            total_loss = total_loss + loss;
        }
        
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch 
                      << " Loss: " << total_loss.getData() << std::endl;
        }
    }
    
    return 0;
}
```

### Classification with Softmax

```cpp
Linear fc1(4, 8);
ReLU relu;
Linear fc2(8, 3);
Softmax softmax;
Sequential classifier({&fc1, &relu, &fc2, &softmax});

std::vector<Value> input = {Value(1.0f), Value(2.0f), Value(3.0f), Value(4.0f)};
std::vector<Value> probs = classifier(input);

int target_class = 1;
Value loss = Loss::CrossEntropy(probs, target_class);
loss.backward();
```

---

## Building & Running

### Compile with g++

```bash
# Compile training example
g++ -std=c++17 -I. -o train train.cpp src/value.cpp src/nn.cpp

# Run
./train
```

### Run Tests

```bash
# Compile and run Value tests
g++ -std=c++17 -I. -o test_value tests/test_value.cpp src/value.cpp
./test_value

# Compile and run NN tests
g++ -std=c++17 -I. -o test_nn tests/test_nn.cpp src/value.cpp src/nn.cpp
./test_nn
```

---

## API Reference

### Value Class

| Method | Signature | Description |
|--------|-----------|-------------|
| Constructor | `Value(float val)` | Create a Value wrapping a scalar |
| `getData` | `float getData() const` | Get the underlying scalar value |
| `getGrad` | `float getGrad() const` | Get the gradient |
| `setData` | `void setData(float v)` | Set the value |
| `setGrad` | `void setGrad(float g)` | Set the gradient |
| `backward` | `void backward()` | Compute all gradients via backprop |
| `clearGraph` | `void clearGraph()` | **Release memory** by clearing computation graph |
| `print` | `void print() const` | Print value and gradient |

### Module Class

| Method | Signature | Description |
|--------|-----------|-------------|
| `operator()` | `vector<Value> operator()(const vector<Value>& x)` | Forward pass |
| `parameters` | `vector<Value*> parameters()` | Get all trainable parameters |
| `zero_grad` | `void zero_grad()` | Reset all gradients to 0 |

### Optimizer Class

| Method | Signature | Description |
|--------|-----------|-------------|
| `step` | `void step()` | Update parameters using gradients |
| `zero_grad` | `void zero_grad()` | Reset all gradients to 0 |

---

## License

MIT License - Feel free to use for educational purposes.

---

## Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- Built for learning the fundamentals of deep learning frameworks
