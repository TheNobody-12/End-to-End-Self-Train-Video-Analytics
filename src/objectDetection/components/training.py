from ultralytics import YOLO
import shutil
from objectDetection.entity.config_entity import TrainingConfig
from pathlib import Path
from objectDetection.utils.common import update_datasets_dir
import mlflow
import logging
import torch.nn as nn
import torch
import re
from ultralytics import settings

settings.update({'mlflow': False,
                 'datasets_dir':'artifacts/data_ingestion/dataset'})   

class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
# mlflow.autolog()


def on_train_end(trainer):
    #    if mlflow:
    print('in the on_train_end callbacks')
    print('trainers with trainers.metrics:' + str(trainer.best))
    print('trainers with trainers.metrics:' + str(trainer.last))
    print('trainers with trainers.metrics:' + str(trainer.testset))
    print('trainers with trainers.metrics:' + str(trainer.trainset))
    mlflow.log_artifact(str(trainer.best), "model")

#    mlflow.pytorch.log_model(str(trainer.best), "model_log")
    # End the MLflow run
    # # Convert to ONNX first
    # model.export(format='onnx', path='model.onnx')
    #
    # # Then log the ONNX model to MLflow
    # mlflow.log_artifact('model.onnx', artifact_path="models")

    # Register the model in MLflow Model Registry
    # mlflow.register_model(
    #      "runs:/train/mlflow_simple_new/model",
    #      "my_registered_model"
    #  )

    # # Register the model in MLflow Model Registry
    # mlflow.register_model(
    #     "runs:/my_project/my_experiment/model",
    #     "my_registered_model"
    # )


def on_train_start(trainer):
    #    if mlflow:
    #    print('in the on_train_start callbacks trainer'+trainer)
    print('trainers with trainers.metrics:' + str(trainer.best))
    print('trainers with trainers.metrics:' + str(trainer.last))
    print('trainers with trainers.metrics:' + str(trainer.testset))
    print('trainers with trainers.metrics:' + str(trainer.trainset))


def on_fit_epoch_end(trainers):
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v)
                    for k, v in trainers.metrics.items()}
    print('trainers.metrics with metrics_dict:' + str(metrics_dict))
    mlflow.log_metrics(metrics=metrics_dict, step=trainers.epoch)


def convert_pt_to_pytorch_model(model_path):
    # Load the .pt model
    model = torch.load(model_path)
    # Convert it to a PyTorch model if necessary
    # For example:
    # pytorch_model = YourYOLOModel(your_model_arguments)
    # Copy the parameters from the loaded model to the PyTorch model
    # pytorch_model.load_state_dict(model.state_dict())
    return model

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = YOLO(
            self.config.base_model_path
        )

        self.model.add_callback('on_train_end', on_train_end)
        self.model.add_callback('on_train_start', on_train_start)
        self.model.add_callback('on_fit_epoch_end', on_fit_epoch_end)


    @staticmethod
    def save_model(path: Path, dest_path: Path):
        # model.save(path)
        shutil.copy(path, dest_path)
        logging.info(f"Model saved at {dest_path}")
        mlflow.log_artifact(dest_path)
        # mlflow.register_model(dest_path, "model")


    def train(self):

        logging.info("Starting model training process")
        
        results = self.model.train(data=self.config.training_data, model="yolov8n.yaml",task="detect", \
                         epochs=self.config.params_epochs,workers=8,batch=self.config.params_batch_size,imgsz=self.config.params_image_size,\
                            project=self.config.params_project,exist_ok=True)
        # Log the metrics
        mlflow.log_metrics({
                "Precision": results.results_dict['metrics/precision(B)'],
                "Recall": results.results_dict['metrics/recall(B)'],
                "map": results.results_dict['metrics/mAP50-95(B)'],
                "map50": results.results_dict['metrics/mAP50(B)'],
            })
        # Save the model
        self.save_model(
            path=self.config.best_model_path,
            dest_path=self.config.trained_model_path
        )