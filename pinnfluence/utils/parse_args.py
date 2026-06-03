import argparse

from .defaults import DEFAULTS


def common_arg_parser():
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULTS["seed"], help="Random seed"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULTS["lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULTS["layers"],
        help="Layer sizes",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=DEFAULTS["n_iterations"],
        help="Number of iterations",
    )
    parser.add_argument(
        "--n_iterations_lbfgs",
        type=int,
        default=DEFAULTS["n_iterations_lbfgs"],
        help="Number of LBFGS iterations",
    )
    parser.add_argument(
        "--num_domain",
        type=int,
        default=DEFAULTS["num_domain"],
        help="Number of domain points",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULTS["model_zoo"],
        help="Path to save model",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load model",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="burgers",
        help="Problem/PDE to run experiment for",
        choices=[
            "allen_cahn",
            "burgers",
            "diffusion",
            "drift_diffusion",
            "navier_stokes_nd",
            "poisson_disk",
            "wave",
        ],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Which optimizer to use for training",
        choices=["adam", "L-BFGS", "SOAP", "NNCG"],
        default=DEFAULTS["optimizer"],
    )
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Use float64",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for training.",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--soft_constrained",
        action="store_true",
        default=True,
        help="Use soft constrained training (only supported mode)",
    )
    parser.add_argument(
        "--num_boundary",
        type=int,
        default=0,
        help="Number of boundary points",
    )
    parser.add_argument(
        "--num_initial",
        type=int,
        default=0,
        help="Number of initial points",
    )
    parser.add_argument(
        "--drop_single_point_type",
        type=str,
        default="none",
        help="Type of single point to drop",
        choices=["none", "IC", "BC", "domain"],
    )
    return parser


def parse_pretrain_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a PINN model",
        parents=[common_arg_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use wandb",
    )
    parser.add_argument(
        "--dont_force_retrain",
        action="store_false",
        dest="force_retrain",
        help="Don't force retraining, i.e. continue training from checkpoint if available",
    )
    return parser.parse_args()


def parse_precalculate_args():
    parser = argparse.ArgumentParser(
        description="Precalculate influence scores. Please match the arguments of the respective pretraining run you with to precalculate influences for.",
        parents=[common_arg_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_candidate_points",
        type=int,
        default=10_000,
        help="Number of candidate points",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="PINNfluence",
        choices=["PINNfluence", "grad_dot"],
        help="Scoring strategy",
    )
    parser.add_argument(
        "--precalc_infl_use_holdout_test",
        action="store_true",
        help="Use holdout test set for precalc",
    )
    parser.add_argument(
        "--precalc_infl_sample_uniformly",
        action="store_true",
        help="Sample uniformly for precalc",
    )
    parser.add_argument(
        "--use_train_set",
        action="store_true",
        help="Use train set for precalc",
    )
    parser.add_argument(
        "--left",
        type=str,
        default=None,
        help="Test loss term (required). Options: total_loss, pde_loss, bc_loss, pde_N, bc_N, output_N",
    )
    parser.add_argument(
        "--right",
        type=str,
        default=None,
        help="Training loss term (required). Options: total_loss, pde_loss, bc_loss, pde_N, bc_N",
    )
    parser.add_argument(
        "--self-influence",
        action="store_true",
        help="Calculate self-influence using train_x_all as candidates",
    )
    return parser.parse_args()
