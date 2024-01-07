from typing import List
from dataclasses import dataclass,field

@dataclass
class BasicTrainerConfig:
    number_of_epochs:int = 300
    log_loss:int = 100
    warm_up_best_model_epoch = 1e6
    save_model_test_stopping:bool = True
    save_model_epochs:int = 1e6
    save_metric_epochs:int = 1e6
    max_test_size:int = 2000
    do_ema:bool = True
    clip_grad:bool = False
    clip_max_norm:float = 1.

    learning_rate:float = 0.001
    weight_decay:float =  0.0001
    lr_decay:float =  0.999

    device:str = "cuda:0"
    berlin: bool = True
    distributed: bool = False
    debug:bool = False

    metrics: List[str] = field(default_factory=lambda :["mse_histograms",
                                                        "kdmm",
                                                        "categorical_histograms"])
    def __post_init__(self):
        self.save_model_epochs = int(.5*self.number_of_epochs)
        self.save_metric_epochs = self.number_of_epochs - 1
