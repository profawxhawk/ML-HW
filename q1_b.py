import linear_regression as lr
if (__name__ == "__main__"):
    lr_model=lr.linear_regression()
    # filename=lr_model.convert_data_to_csv('./abalone.data')
    dataset=lr.pd.read_csv('./q1.csv')
    lr_model.find_vectors_k_fold(dataset,5)
    for i in range(5):
        # lr_model.find_vector(dataset,fold_count,5)
        lr_model.optimise_weight_normal(i)
        lr_model.cost_func_train(i+1)
        lr_model.cost_func_val(i+1)
        print('Training RMS Error '+str(lr_model.Train_rmse))
        print('Validation RMS Error '+str(lr_model.Val_rmse))
        print()
    print(lr_model.Weight)