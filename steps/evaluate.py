import pandas as pd
from churn_prediction import data
from churn_prediction import utils
from churn_prediction import evaluation
from omegaconf import OmegaConf

def evaluate(config):
    test_data = pd.read_csv(r"input/prepared_test.csv")
    pipeline = utils.load_pipeline(config["pipeline"]["filepath"])
    data_map = data.get_features_target(test_data)
    scores = evaluation.evaluate(pipeline,data_map["features"],data_map["target"])
    print("========================= Test Scores ==========================")
    print(scores)
    

if __name__ == "__main__":
    config = OmegaConf.load(r"params.yaml")
    evaluate(config)