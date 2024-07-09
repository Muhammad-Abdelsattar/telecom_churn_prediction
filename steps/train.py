import pandas as pd
from omegaconf import OmegaConf
from churn_prediction import training, modeling


def train(config):
    data = pd.read_csv(r"input/prepared_train.csv")
    cat_cols = list(data.select_dtypes(include=object).columns)
    cat_cols.remove("Churn")
    pipeline = modeling.build_pipeline(config["pipeline"],cat_cols)
    k = 5
    pipeline_file_path = config["pipeline"]["filepath"]
    scores_filepath = config["scores"]["filepath"]
    training.train(pipeline,data,k,pipeline_file_path,scores_filepath)
    
if __name__ == "__main__":
    config = OmegaConf.load(r"params.yaml")
    train(config)