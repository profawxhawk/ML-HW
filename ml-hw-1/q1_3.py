import linear_regression as lr
def grad_des_model(lr_model,model,param=0):
    lr_model.Weight=lr.np.zeros((2,1))
    for i in range(1000):
        if model=="normal":
            print(lr_model.gradient_descent(0.00057,1))
        if model=="l1":
            (lr_model.gradient_descent_lasso(0.00057,0,param))
        if model=="l2":
            (lr_model.gradient_descent_ridge(0.00057,0,param))
    print(lr_model.Weight)
    lr.plt.scatter(lr_model.TrainX_vector[:,0],lr_model.TrainY_vector,c='r')
    lr.plt.plot(lr_model.TrainX_vector[:,0],lr.np.matmul(lr_model.TrainX_vector,lr_model.Weight),'b')
    lr.plt.show()
if (__name__ == "__main__"):
    lr_model=lr.linear_regression()
    dataset=lr.pd.read_csv('./brain_body_weight.csv')
    lr_model.find_vector_no_folds(dataset)
    print("Normal gradient descent")
    grad_des_model(lr_model,"normal")
    print("Normal gradient descent end")
    
    print("Finding regularization parameters")
    l1_param=lr_model.regularization_hyper_param_tune("l1",0)
    l2_param=lr_model.regularization_hyper_param_tune("l2",0)
    print("Lasso hyperparameter "+str(l1_param))
    print("Ridge hyperparameter "+str(l2_param))
    
    print("L1 gradient descent started")
    grad_des_model(lr_model,"l1",l1_param['alpha'])
    print("L1 gradient descent ended")
    print("L2 gradient descent started")
    grad_des_model(lr_model,"l2",l2_param['alpha'])
    print("L2 gradient descent ended")
    
    
    