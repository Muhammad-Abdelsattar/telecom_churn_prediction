for .utils import get_scores, log_score

def evaluate(pipeline,features,target):
    preds = pipeline.predict(features)
    scores = get_scores(preds,target)
    