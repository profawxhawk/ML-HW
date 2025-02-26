import logistic_regression as lr
from sklearn.preprocessing import LabelEncoder # labelencoder 
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import KFold,GridSearchCV
import warnings
warnings.filterwarnings("ignore")
def label_encode(dataset):
    encoder= LabelEncoder()
    dataset=dataset.drop('education',axis=1)
    dataset['salary'] = encoder.fit_transform(dataset['salary'].astype('str'))
    dataset['workclass'] = encoder.fit_transform(dataset['workclass'].astype('str'))
    dataset['marital-status'] = encoder.fit_transform(dataset['marital-status'].astype('str'))
    dataset['occupation'] = encoder.fit_transform(dataset['occupation'].astype('str'))
    dataset['relationship'] = encoder.fit_transform(dataset['relationship'].astype('str'))
    dataset['sex'] = encoder.fit_transform(dataset['sex'].astype('str'))
    dataset['race'] = encoder.fit_transform(dataset['race'].astype('str'))
    dataset['native-country'] = encoder.fit_transform(dataset['native-country'].astype('str'))
    return (dataset)

def one_hot_encode(df):
    
    df=compress_features(df)
    encoder= LabelEncoder()
    df['salary'] = encoder.fit_transform(df['salary'].astype('str'))
    df=lr.pd.get_dummies(df)
    temp=df.salary
    df=df.drop('salary',axis=1)
    df['salary']=temp
    return df
    
    
def compress_features(dt):
    dt.replace('Without-pay', 'Unemployed',inplace=True)
    dt.replace('Never-worked', 'Unemployed',inplace=True)
    dt.replace('Local-gov', 'State_gov',inplace=True)
    dt.replace('State-gov', 'State_gov',inplace=True)
    dt.replace('Self-emp-inc', 'Self_employed',inplace=True)
    dt.replace('Self-emp-not-inc', 'Self_employed',inplace=True)
    dt.replace(['11th','9th','7th-8th', '12th', '1st-4th', '10th','5th-6th', 'Preschool'],['School','School','School','School','School','School','School','School'],inplace=True)
    dt.replace(["Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala","Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago","United-States"],14*['North_America'],inplace=True)
    dt.replace(["Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos","Philippines", "Taiwan", "Thailand", "Vietnam"],11*['Asia'],inplace=True)
    dt.replace(["England", "France", "Germany", "Greece", "Holand-Netherlands","Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland","Yugoslavia"],12*['Europe'],inplace=True)
    dt.replace(["Columbia", "Ecuador", "Peru"],3*["South_America"])
    dt.replace(['Divorced','Separated','Never-married','Widowed','Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'],['not_married','not_married','not_married','not_married','married_','married_','married_'],inplace=True)
    return dt
if (__name__ == "__main__"):
    lr_model=lr.logistic_regression()
    error_l1=[]
    error_l2=[]
    accuracy_l1=[]
    accuracy_l2=[]
    print("loading dataset")
    dataset=lr.pd.read_csv('./q2_train.csv')
    testset=lr.pd.read_csv('./q2_test.csv')
    print("Converting to one hot encode")
    dataset=one_hot_encode(dataset)
    testset=one_hot_encode(testset)
    print(dataset['age'])
    exit()
    lr_model.find_vector_no_folds(dataset,testset)
    print("L1 gradient descent started")
    lr_model.Weight=lr.np.zeros((lr_model.TrainX_vector.shape[1],1))
    for i in range(1000):
        a,b=(lr_model.gradient_descent_lasso(0.1,0.002))
        error_l1.append(a)
        accuracy_l1.append(b)
    print("train accuracy "+str(lr_model.accuracy("train")))
    print("validation accuracy "+str(lr_model.accuracy("val")))
    print("test accuracy "+str(lr_model.accuracy("test")))
    print("L1 gradient descent ended")
    print("L2 gradient descent started")
    lr_model.Weight=lr.np.zeros((lr_model.TrainX_vector.shape[1],1))
    for i in range(1000):
        a,b=(lr_model.gradient_descent_ridge(0.1,0.002))
        error_l2.append(a)
        accuracy_l2.append(b)
    print("train accuracy "+str(lr_model.accuracy("train")))
    print("validation accuracy "+str(lr_model.accuracy("val")))
    print("test accuracy "+str(lr_model.accuracy("test")))
    print("L2 gradient descent ended")
    lr.plt.title("Error for L1 regularization")
    lr.plt.xlabel('iterations', fontsize=18)
    lr.plt.ylabel('error', fontsize=18)
    lr.plt.plot(lr.np.linspace(0,1000,len(error_l1)),error_l1,'r')  

    lr.plt.show()
    lr.plt.xlabel('iterations', fontsize=18)
    lr.plt.ylabel('error', fontsize=18)
    lr.plt.title("Error for L2 regularization")
    lr.plt.plot(lr.np.linspace(0,1000,len(error_l2)),error_l2,'r') 
        
    lr.plt.show()
    lr.plt.xlabel('iterations', fontsize=18)
    lr.plt.ylabel('accuracy', fontsize=18)
    lr.plt.title("Accuracy for L1 regularization")
    lr.plt.plot(lr.np.linspace(0,1000,len(accuracy_l1)),accuracy_l1,'r') 
        
    lr.plt.show()
    lr.plt.xlabel('iterations', fontsize=18)
    lr.plt.ylabel('accuracy', fontsize=18)
    lr.plt.title("Accuracy for L2 regularization")
    lr.plt.plot(lr.np.linspace(0,1000,len(accuracy_l2)),accuracy_l2,'r')     
    
    lr.plt.show()