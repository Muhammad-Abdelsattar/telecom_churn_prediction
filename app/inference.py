from pickle import load
import pandas as pd

class InferencePipeline:
    def __init__(self, pipeline_path: str) -> None:
        if not isinstance(pipeline_path, str) or not pipeline_path:
            raise ValueError("pipeline_path must be a non-empty string")
        self.pipeline = self._load_pipeline(pipeline_path=pipeline_path)

    def __call__(self,raw_input):
        data = self.prepare_input(raw_input)
        return self.predict(data)

    def _load_pipeline(self,pipeline_path: str):
        """
        Loads a machine learning pipeline from the specified file path.
        
        Args:
            pipeline_path (str): The file path to the serialized pipeline.
        
        Returns:
            A deserialized machine learning pipeline object.
        """
        try:
            with open(pipeline_path, "rb") as f:
                pipeline = load(f)
        except (FileNotFoundError, IOError, pickle.PickleError) as e:
            raise RuntimeError(f"Failed to load pipeline: {e}")
        return pipeline

    def prepare_input(self,raw_input: list[dict]):
        """
        Prepares the input data for the machine learning pipeline.
        
        Args:
            raw_input (list[dict]): A list of dictionaries, where each dictionary represents a single data point.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the prepared input data.
        """
        prepared_input = {}
        for datapoint in raw_input:
            for k,v in datapoint.items():
                if(k in prepared_input):
                    prepared_input[k].append(v)
                else:
                    prepared_input[k] = list([v])
        return pd.DataFrame(prepared_input)
    
    def predict(self,data):
        return self.pipeline.predict(data).tolist()

