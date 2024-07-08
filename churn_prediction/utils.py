from sklearn.metrics import recall_score,precision_score,roc_curve,roc_auc_score,f1_score,accuracy_score
from pickle import dump, load
import json

def get_scores(preds,labels):
    scores = {}
    scores["accuracy"] = accuracy_score(preds,labels)
    scores["recall"] = recall_score(preds,labels)
    scores["precision"] = precision_score(preds,labels)
    scores["f1"] = f1_score(preds,labels)
    scores["auc"] = roc_auc_score(preds,labels)
    return scores

def log_score(score,filepath,stage):
    stage_score = {}
    for k,v in score.items():
        stage_score[stage+"."+k] = score[k]
    with open(filepath,"w") as f:
        json.dump(stage_score,f)

def get_average_cv_score(folds_scores):
    avg_scores = {}
    k = len(folds_scores)
    for key,v in folds_scores.items():
        for metric,value in v.items():
            if(not metric in avg_scores.keys()):
                avg_scores[metric] = value/k
            else:
                avg_scores[metric] += value/k
    return avg_scores

def save_pipeline(pipeline,filepath):
    with open(filepath,"wb") as f:
        dump(pipeline, f, protocol=5)

def load_pipeline(filepath):
    with open(filepath, rb) as f:
        pipeline = load(f)
    return pipeline