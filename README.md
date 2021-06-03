<h1 align="center"><img src="./data/korgorusz_cat.svg" alt="hmmm"><br>Korgorusz</h1>

<b>Korgorusz</b> is a simple machine learning framework build on top of numpy.

## Features
* Modular design
* Slow
* No GPU support
* Types
* Depends only on numpy

<br>


## One of the biggest advantages of Korgorusz is slowness - almost 10(~6.2) times slower than pytorch
![plot time](./data/plot_bar.png)


<br>


## Example
```python
import numpy as np

from korgorusz.optimizers import SGDOptimizer
from korgorusz.layers import Linear
from korgorusz.activations import ReLU, Sigmoid
from korgorusz.utils import minibatch, mse, Model

x = np.random.randn(5,4)
y = np.random.randn(5,2)

optim = SGDOptimizer(learning_rate=0.01)
model = Model([
    Linear(4, 8),
    ReLU(),
    # ...
    Linear(8, 2),
    Sigmoid()])

for e in range(8):
    pred = model.forward(x)
    loss, d = mse(pred,y)
    ml.backward(d)
    optim.update(model.layers_elements())
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
pytest korgorusz    # test suite
mypy korgorusz      # type checks
black korgorusz     # formatting
pylint korgorusz/ -d R0903 --good-names=l2,x,y,i # linter
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
