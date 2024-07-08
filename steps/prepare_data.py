from churn_prediction import data
from omegaconf import OmegaConf

def prepare_data(config):
    prepared_data = data.prepare_data(config["data"]["raw_data_path"])
    prepared_data.to_csv(config["data"]["prepared_data_path"], index=False)
    

if __name__ == "__main__":
    config = OmegaConf.load(r"params.yaml")
    prepare_data(config)