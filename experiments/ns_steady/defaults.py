DEFAULTS = {
    "lr": 0.001,
    "layers": [2, 64, 64, 64, 64, 3],
    "n_iterations": 100_000,
    "n_iterations_lbfgs": 25_000,
    "num_domain": 7_500,
    "num_boundary": 2_500,
    "save_path": "./model_zoo",
    "broken": False,
    "device": "mps",
}
