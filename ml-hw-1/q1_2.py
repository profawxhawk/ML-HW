import linear_regression as lr
import warnings
warnings.filterwarnings("ignore")
def gradient_with_reg(lr_model,iterations,model,param,least_fold):
        train_cost=[]
        lr_model.Weight=lr.np.zeros((11,1))
        for i in range(iterations):
            if model=="l2":
                train_cost.append(lr_model.gradient_descent_ridge(0.05,least_fold,param['alpha'],test="False"))
            else:
                train_cost.append(lr_model.gradient_descent_lasso(0.05,least_fold,param['alpha'],test="False"))
        avg_train_rmse=(sum(train_cost)/len(train_cost))
        lr.plt.title("RMSE vs Iterations for "+str(model)+" regularization")
        lr.plt.xlabel('Iterations', fontsize=18)
        lr.plt.ylabel('RMSE', fontsize=18)
        lr.plt.plot(lr.np.linspace(0,iterations,len(train_cost)),train_cost,'r')      
        lr.plt.show()
        lr_model.cost_func_val(least_fold+1)
        print('Test RMSE Error for '+str(model)+" "+str(lr_model.Val_rmse))
        
if (__name__ == "__main__"):
    lr_model=lr.linear_regression()
    # filename=lr_model.convert_data_to_csv('./abalone.data')
    dataset=lr.pd.read_csv('./q1.csv')
    lr_model.find_vectors_k_fold(dataset,5)
    least_val=10**5
    least_fold=0
    print("K-folds created")
    for i in range(5):
        # lr_model.find_vector(dataset,fold_count,5)
        lr_model.optimise_weight_normal(i)
        lr_model.cost_func_train(i+1)
        lr_model.cost_func_val(i+1)
        if lr_model.Val_rmse<least_val:
            least_val=lr_model.Val_rmse
            least_fold=i
    print("Choose Fold "+str(least_fold+1))
    l1_param=lr_model.regularization_hyper_param_tune("l1",least_fold)
    l2_param=lr_model.regularization_hyper_param_tune("l2",least_fold)
    print("Lasso hyperparameter "+str(l1_param))
    print("Ridge hyperparameter "+str(l2_param))
    gradient_with_reg(lr_model,100,"l1",l1_param,least_fold)
    gradient_with_reg(lr_model,100,"l2",l2_param,least_fold)
    
    