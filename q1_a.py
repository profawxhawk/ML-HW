import linear_regression as lr
from sklearn.linear_model import LinearRegression
if (__name__ == "__main__"):
    lr_model=lr.linear_regression()
    # filename=lr_model.convert_data_to_csv('./abalone.data')
    dataset=lr.pd.read_csv('./q1.csv')
    lr_model.Weight=lr.np.zeros((11,1))
    lr_model.find_vectors_k_fold(dataset,5)
    iterations=100
    plot_train_rmse=[]
    plot_val_rmse=[]
    for j in range(iterations):
        avg_train_rmse=[]
        avg_val_rmse=[]
        fold_count=1
        for i in range(5):
            # print("Fold "+str(fold_count))
            #lr_model.find_vector(dataset,fold_count,5)
            # lr_model.Weight=lr.np.zeros((11,1))
            avg_train_rmse.append(lr_model.gradient_descent(0.04,fold_count))
            avg_val_rmse.append(lr_model.gradient_descent(0.04,fold_count,"True"))
            # lr_model.cost_func_val()
            # print('Training RMS Error '+str(lr_model.Train_rmse))
            # print('Validation RMS Error '+str(lr_model.Val_rmse))
            # print()
            fold_count+=1
        print("Average Train rmse: "+ str(sum(avg_train_rmse)/len(avg_train_rmse)))
        print("Average Val rmse: "+ str(sum(avg_val_rmse)/len(avg_val_rmse)))
        plot_train_rmse.append((sum(avg_train_rmse)/len(avg_train_rmse)))
        plot_val_rmse.append((sum(avg_val_rmse)/len(avg_val_rmse)))
        print()
    lr.plt.plot(lr.np.linspace(0,iterations,len(plot_train_rmse)),plot_train_rmse,'r')     
    lr.plt.plot(lr.np.linspace(0,iterations,len(plot_val_rmse)),plot_val_rmse,'b')   
    lr.plt.show()

    
    
    