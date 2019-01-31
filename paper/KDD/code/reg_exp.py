def run_experiment(x, y, alpha=0.1, K='auto', epsilon=100, print_cost=1000000):
    m = x.shape[0]
    n = x.shape[1] - 1
    
    theta = np.random.randn(n + 1, 1)
    
    x_norm = np.sum(x, axis=0)
    x = x / x_norm
    
    theta_final, it = batch_gradient_descent(theta, x, y, alpha=alpha, epsilon=epsilon, print_cost=print_cost)
    print('Traditional:', it, 'iterations')
    
    if (K == 'auto'):
        K = np.linalg.norm(theta_final)

    L = K / m * np.linalg.norm(np.dot(x.T, x)) - 1 / m * np.linalg.norm(np.dot(y.T, x))
    a = np.abs(1 / L)
    print('Custom learning rate:', a)
    _, it = batch_gradient_descent(theta, x, y, alpha=a, epsilon=epsilon, print_cost=print_cost)
    print('Custom:', it, 'iterations')