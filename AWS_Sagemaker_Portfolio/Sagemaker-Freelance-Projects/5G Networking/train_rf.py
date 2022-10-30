from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pandas as pd
import os, argparse
import numpy as np

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
    return model

def load_dataset(path):
    # Load dataset
    data = pd.read_csv(path)
    data1 = data.copy()
    # Split samples and labels
    x = data.drop(['RSRP_bandwith' , 'RSSI_bandwith' , 'RSRQ_bandwith'], axis=1)
    y = data1[['RSRP_bandwith', 'RSSI_bandwith' ,'RSRQ_bandwith']]
    return x,y

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_depth",
        type=int,
    )
    
    parser.add_argument("--random_state", type=int)
    parser.add_argument('--n_estimators', type=int, default=250)
                        
    
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    max_depth= args.max_depth
    random_state= args.random_state
    n_estimators =args.n_estimators
 
    model_dir = args.model_dir
    training_dir = args.training_dir
    validation_dir = args.validation
    
    print(training_dir)
    x_train, y_train = load_dataset(os.path.join(training_dir, 'sagemaker_training_dataset.csv'))
    x_val, y_val     = load_dataset(os.path.join(validation_dir, 'sagemaker_validation_dataset.csv'))
    
    
#     x_train, y_train = load_dataset(training_dir)
#     x_val, y_val     = load_dataset(validation_dir)
    
    cls = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    multi_target_forest = MultiOutputClassifier(cls, n_jobs=2)
    

    multi_target_forest.fit(x_train, y_train)#     Model Training
    y_pred = multi_target_forest.predict(x_val)#     Prediction

    accuracy = multi_target_forest.score(x_val, y_val)#     AUc Score
    
    print("accuracy score ", accuracy)
    
    model = os.path.join(model_dir, 'rf_model.joblib')
    joblib.dump(multi_target_forest, model)
    # See https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html