# Experiments for "Role of Lipschitz constant in Gradient Learning"
This repo contains the experiments for the paper. A description of each notebook is below.

* `Batch Gradient Descent`: This notebook has the regression experiments.
* `Logistic Regression`: This notebook has the binary classification experiments. Note that the graph showing a downward line is pretty old, it is not from a recent run. Newer runs of this notebook simply use the `run_experiment` function defined.
* `New softmax`: This replaced an older notebook, `Softmax`. The old one assumed a multinomial distributed target variable, while the new one assumes one-hot encoded, and performs much better. The experiments under the `MNIST` header are tests--the final one is in a separate notebook. All other multi-class classification experiments are in this notebook.
* `MNIST`: A Keras implementation of softmax regression with one-hot encoded target labels and standardized data was the only implementation that never had errors with division by zero or nan values. Many hours were wasting getting here; do not disturb this, it's the best possible result, even if not ideal.
