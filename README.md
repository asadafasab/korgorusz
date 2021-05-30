<h1 align="center"><img src="./data/korgorusz_cat.svg" alt="hmmm"><br>Korgorusz</h1>

<b>Korgorusz</b> is a simple machine learning framework build on numpy.

## Features
* modular design
* slow
* no GPU support
* types
* depends only on numpy

<br>


## One of the biggest advantages of Korgorusz is slowness - almost 10(~6.2) times slower than pytorch
![plot time](./data/plot_bar.png)


<br>


## Example
```python
import numpy as np

from korgorusz.optimizers import SGDOptimizer
from korgorusz.layers import ReLU,Linear,Sigmoid
from korgorusz.utils import minibatch,mse,Model

x = np.random.randn(5,4)
y = np.random.randn(5,2)

class ModelLearn(Model):
    def __init__(self):
        super().__init__()
        self.layers=[
            Linear(4, 8),
            ReLU(),
            # ...
            Linear(8, 2),
            Sigmoid()]

    def forward(self, X):
        for l in self.layers:
            X, b = l.forward(X)
            self.add_derivative(b)
        return X
optim = SGDOptimizer(lr=0.01)
ml = ModelLearn()

for e in range(16):
    pred = ml.forward(x)
    loss, d = mse(pred,y)
    ml.backpropagation(d)
    ml.update(ml.layers,optim)
```
More examples in the notebooks.


## Instalation
```bash
python -m venv venv
source venv/bin/activate
pip install korgorusz/.
# or
pip install korgorusz/.[dev]
```

## Tests
```bash
python -m pytest korgorusz    # test suite
python -m mypy korgorusz      # type checks
python -m black korgorusz    # linting
```

## Implemented Algorithms
### Activations
* ReLU
* Softmax
* Sigmoid

### Optimizers
* SGD
* Momentum
* Adam

### Layers
* Linear
* Dropout
* LayerNorm
* Embedding


## Name
Korgorusz is a slavic cat deamon.
