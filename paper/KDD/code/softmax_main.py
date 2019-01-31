w = np.random.randn(x_train.shape[1],len(np.unique(y_train)))
lam = 0
iterations = 200
learningRate = 0.1
for i in tqdm(range(iterations)):
    loss,grad = getLoss(w,x_train,y_train,lam)
    w = w - (learningRate * grad)
print(loss)