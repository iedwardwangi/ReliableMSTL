__author__ = "Zirui Wang"

from config import *

def exp_params2model_params(exp_params):
    params = {}
    params["AL_method"] = exp_params["AL_method"]
    params["start_mode"] = exp_params["start_mode"]
    params["b1"] = exp_params["b1"]
    params["rho"] = exp_params["rho"]
    params["beta_1"] = exp_params["beta_1"]
    params["beta_2"] = exp_params["beta_2"]
    params["mu"] = exp_params["mu"]
    params["max_alpha"] = exp_params["max_alpha"]
    params["tau_lambda"] = exp_params["tau_lambda"]
    return params