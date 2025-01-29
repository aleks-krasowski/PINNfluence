DEFAULTS = {
    "lr": 0.001,
    "layers": [2] + [20] * 3 + [1],
    "n_iterations": 400_000,
    "n_iterations_lbfgs": 12_500,
    "num_domain": 2_540,
    "num_boundary": 80,
    "num_initial": 160,
    "save_path": "./model_zoo",
    "device": "cpu",
}
