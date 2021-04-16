<h1 align="center"><img src="./data/korgorusz_cat.svg" alt="hmmm"><br>Korgorusz<h2 align="center">Basic machine learning tool</h2></h1>

<b>Korgorusz</b> is a simple set of machine learning alghoritms ...

## Features
* modular design
* very slow
* no GPU support
* types
* depends only on numpy

<br>


## One of the biggest advantages of Korgorusz is slowness - almost 10(~6.2) times slower than pytorch
![plot time](./data/plot_bar.png)


<br>


## Example
```python
from korgorusz.optimizers import SGDOptimizer
from korgorusz.layers import ReLU,Linear,Sigmoid
from korgorusz.utils import minibatch,mse,Model

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
optim = SGDOptimizer(lr=lr)
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


### Name
Korgorusz is a slavic cat deamon.
