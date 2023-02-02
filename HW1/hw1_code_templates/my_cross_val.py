import numpy as np
import pandas as pd
import random 

def my_cross_val(model, loss_func, X, y, k=10):
    dataset = np.column_stack([X, y])
    split_data = []
    copy = list(dataset)
    size_each_fold = int(len(copy) / k)

    for i in range(k):
        fold = []
        while len(fold) < size_each_fold:
            index = random.randrange(len(copy))
            fold.append(copy.pop(index))
        split_data.append(fold)
    split_data = pd.DataFrame(split_data)
    split_data = split_data.transpose()
    #return split_data
    mse_by_fold = []
    err_rate_by_fold = []   
    for i in range(k):
        target_fold = split_data.iloc[:, i]
        df = np.empty((0, len(target_fold.iloc[0])))
        for j in range(target_fold.shape[0]):
            df = np.row_stack([df, target_fold.iloc[j]])
        df = pd.DataFrame(df)
        non_fold = non_target_fold(split_data.drop(split_data.columns[i], axis = 1))
        X_fold = df.iloc[:, :-1]
        X_fold = X_fold.to_numpy()
        y_fold = df.iloc[:, -1:]
        X_non_fold = non_fold.iloc[:, :-1]
        X_non_fold = X_non_fold.to_numpy()
        y_non_fold = non_fold.iloc[:, -1:]
        y_fold = y_fold.to_numpy()
        y_fold = y_fold.reshape((len(y_fold), ))
        y_non_fold = y_non_fold.to_numpy()
        y_non_fold = y_non_fold.reshape((len(y_non_fold), ))

        model.fit(X_non_fold, y_non_fold)
        if loss_func == 'mse':
            mse = np.mean((y_fold - model.predict(X_fold))**2)
            mse_by_fold.append(mse)
        elif loss_func == 'err_rate':
            y_hat = model.predict(X_fold)
            err_rate_bool = y_fold != y_hat
            err_rate = sum(err_rate_bool)/len(y_fold)
            err_rate_by_fold.append(err_rate)
 
    if loss_func == 'mse':
        mse_by_fold = [round(num, 4) for num in mse_by_fold]
        summary = {'Mean': round(np.mean(mse_by_fold), 4),'sd': round(np.std(mse_by_fold), 4)}
        mse_by_fold.append(summary)
        return mse_by_fold
    elif loss_func == 'err_rate':
        err_rate_by_fold = [round(num, 4) for num in err_rate_by_fold]
        summary = {'Mean': round(np.mean(err_rate_by_fold), 4),'sd': round(np.std(err_rate_by_fold), 4)}
        err_rate_by_fold.append(summary)
        return err_rate_by_fold    

def non_target_fold (nested_df):
    final_df = np.empty((0, len(nested_df.iloc[0, 0])))
    for i in range(nested_df.shape[1]):
        df = np.empty((0, len(nested_df.iloc[0, 0])))
        for j in range(nested_df.shape[0]):
            df = np.row_stack([df, nested_df.iloc[j, i]])
        
        final_df = np.row_stack([final_df, df])
        final_df = pd.DataFrame(final_df)
    return final_df


    


    



