from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from .utils import get_scores, get_average_cv_score, save_pipeline, log_score
from .data import get_features_target

def cross_validate(pipeline,features,target,k):
    folds_scores = {}
    skf = StratifiedKFold(n_splits=k,shuffle=False)
    folds = skf.split(features,target)
    for i,(train_indices,val_indices) in enumerate(folds):
        x_train = features.iloc[list(train_indices)]
        y_train = target[list(train_indices)]
        x_val = features.iloc[val_indices]
        y_val = target[val_indices]
        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train)
        pipeline.fit(x_train,y_train,**{"model__sample_weight":classes_weights})
        preds = pipeline.predict(x_val)
        scores = get_scores(preds,y_val)
        folds_scores[i] = scores
    avg_score = get_average_cv_score(folds_scores)
    classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=target)
    pipeline.fit(features,target,**{"model__sample_weight":classes_weights})
    return avg_score, folds_scores, pipeline
    
def train(pipeline,data,k,pipeline_filepath,scores_filepath):
    data_map = get_features_target(data)
    avg_score, folds_scores, pipeline = cross_validate(pipeline,data_map["features"],data_map["target"],k)
    print("Training Finished.")
    print(avg_score)
    save_pipeline(pipeline,pipeline_filepath)
    log_score(avg_score,scores_filepath,"validation")
    print(100*"=")