import linear_regression as lr
from sklearn.linear_model import LinearRegression
if (__name__ == "__main__"):
    lr_model=lr.linear_regression()
    # filename=lr_model.convert_data_to_csv('./abalone.data')
    dataset=lr.pd.read_csv('./q1.csv')
    lr_model.Weight=lr.np.zeros((11,1))
    lr_model.find_vectors_k_fold(dataset,5)
    iterations=1000
    plot_train_rmse=[]
    plot_val_rmse=[]
    avg_train_rmse=[[],[],[],[],[]]
    avg_val_rmse=[[],[],[],[],[]]
    weights=[lr.np.zeros((11,1))]*5
    for j in range(iterations):
        fold_count=1
        for i in range(5):
            # print("Fold "+str(fold_count))
            #lr_model.find_vector(dataset,fold_count,5)
            # lr_model.Weight=lr.np.zeros((11,1))
            lr_model.Weight=weights[i]
            avg_train_rmse[i].append(lr_model.gradient_descent(0.04,fold_count))
            weights[i]=lr_model.Weight
            avg_val_rmse[i].append(lr_model.cost_func_val(fold_count))
            # lr_model.cost_func_val()
            # print('Training RMS Error '+str(lr_model.Train_rmse))
            # print('Validation RMS Error '+str(lr_model.Val_rmse))
            # print()
            fold_count+=1
        ty=(avg_train_rmse[0][j]+avg_train_rmse[1][j]+avg_train_rmse[2][j]+avg_train_rmse[3][j]+avg_train_rmse[4][j])
        vy=(avg_val_rmse[0][j]+avg_val_rmse[1][j]+avg_val_rmse[2][j]+avg_val_rmse[3][j]+avg_val_rmse[4][j])
        print("Average Train rmse: "+ str((ty)/5))
        print("Average Val rmse: "+ str((vy)/5))
        plot_train_rmse.append(((ty)/5))
        plot_val_rmse.append((((vy)/5)))
        print()
    lr.plt.title("RMSE vs Iterations for training set")
    lr.plt.xlabel('Iterations', fontsize=18)
    lr.plt.ylabel('RMSE', fontsize=18)
    lr.plt.plot(lr.np.linspace(0,iterations,len(plot_train_rmse)),plot_train_rmse,'r') 
    lr.plt.show()    
    lr.plt.title("RMSE vs Iterations for validation set")
    lr.plt.xlabel('Iterations', fontsize=18)
    lr.plt.ylabel('RMSE', fontsize=18)
    lr.plt.plot(lr.np.linspace(0,iterations,len(plot_val_rmse)),plot_val_rmse,'b')   
    lr.plt.show()

    
    
    