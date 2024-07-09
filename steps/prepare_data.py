from churn_prediction import data
from omegaconf import OmegaConf

def prepare_data(config):
    prepared_train = data.prepare_data(r"input/train.csv")
    prepared_test = data.prepare_data(r"input/test.csv")
    prepared_train.to_csv(r"input/prepared_train.csv", index=False)
    prepared_test.to_csv(r"input/prepared_test.csv", index=False)
    

if __name__ == "__main__":
    config = OmegaConf.load(r"params.yaml")
    prepare_data(config)