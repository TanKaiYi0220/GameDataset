from models.IFRNet import Model as IFRNetModel
from models.IFRNet_Residual import Model as IFRNetResidualModel

MODEL_CONFIGS = {
    "IFRNet": {
        "model_class": IFRNetModel,
        "pretrained_checkpoint": "./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth",
        "default_resume_path": "./output/IFRNet_FineTuning_Resume_0416/checkpoints/best.pth",
        "default_output_dir": "./output/IFRNet_FineTuning_Resume_0416_30",
    },
    "IFRNet_Residual": {
        "model_class": IFRNetResidualModel,
        "pretrained_checkpoint": None,
        "default_resume_path": "./output/IFRNet_Residual_Resume_0416/checkpoints/latest.pth",
        "default_output_dir": "./output/IFRNet_Residual_Resume_0416_10",
    },
}


def get_model_class(name):
    return MODEL_CONFIGS[name]["model_class"]


def get_model_config(name):
    return MODEL_CONFIGS[name]
