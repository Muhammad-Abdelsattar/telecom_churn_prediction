from churn_prediction import data
from omegaconf import OmegaConf

def prepare_data(config):
    prepared_data = data.prepare_data(r"input/train.csv")
    prepared_data.to_csv(r"input/prepared_train.csv", index=False)
    

if __name__ == "__main__":
    config = OmegaConf.load(r"params.yaml")
    prepare_data(config)