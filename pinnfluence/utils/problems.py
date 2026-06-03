import deepxde as dde
import numpy as np
import torch
from scipy.io import loadmat

from .defaults import DEFAULTS

DATASET_DIR = DEFAULTS["DATASET_DIR"]

# ALLEN CAHN
# see https://github.com/lu-group/pinn-sampling/blob/main/src/allen_cahn/RAR_G.py


def allen_cahn_equation(x, y):
    u = y
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_t = dde.grad.jacobian(y, x, j=1)
    return du_t - 0.001 * du_xx + 5 * (u**3 - u)


def allen_cahn_ic(x):
    x_in = x[:, 0:1]
    if isinstance(x_in, torch.Tensor):
        return torch.square(x_in) * torch.cos(np.pi * x_in)
    else:
        return np.square(x_in) * np.cos(np.pi * x_in)


allen_cahn_conditions = [
    {
        "target_class": "IC",
        "func": allen_cahn_ic,
        "on_initial": lambda _, on_initial: on_initial,
        "component": 0,
    },
    {
        "target_class": "DirichletBC",
        "func": lambda x: -1,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0),
        "component": 0,
    },
    {
        "target_class": "DirichletBC",
        "func": lambda x: -1,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], -1.0),
        "component": 0,
    },
]


def allen_cahn_load_testdata():
    default_float = dde.config.default_float()

    data = loadmat(f"{DATASET_DIR}/allen_cahn.mat")
    t = data["t"].astype(np.dtype(default_float))
    x = data["x"].astype(np.dtype(default_float))
    u = data["u"].astype(np.dtype(default_float))
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y


# BURGERS
# see https://github.com/lu-group/pinn-sampling/blob/main/src/burgers/RAR_G.py


def burgers_equation(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / dde.backend.torch.pi * dy_xx


def burgers_ic(x):
    if isinstance(x, torch.Tensor):
        return -torch.sin(torch.pi * x[:, 0:1])
    else:
        return -np.sin(np.pi * x[:, 0:1])


burgers_conditions = [
    {
        "target_class": "IC",
        "func": burgers_ic,
        "on_initial": lambda _, on_initial: on_initial,
        "component": 0,
    },
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0,
        "on_boundary": lambda _, on_boundary: on_boundary,
        "component": 0,
    },
]


def burgers_load_testdata():
    default_float = dde.config.default_float()
    data = np.load(f"{DATASET_DIR}/burgers.npz")
    t, x, exact = (
        data["t"].astype(np.dtype(default_float)),
        data["x"].astype(np.dtype(default_float)),
        data["usol"].astype(np.dtype(default_float)).T,
    )
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


# DIFFUSION -- Note, this one is called "heat" in the paper
# see: https://github.com/lu-group/pinn-sampling/blob/main/src/diffusion/RAR_G.py


def diffusion_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return (
        dy_t
        - dy_xx
        + torch.exp(-x[:, 1:])
        * (torch.sin(np.pi * x[:, 0:1]) - np.pi**2 * torch.sin(np.pi * x[:, 0:1]))
    )


def diffusion_ic(x):
    if isinstance(x, torch.Tensor):
        return torch.sin(np.pi * x[:, 0:1])
    else:
        return np.sin(np.pi * x[:, 0:1])


diffusion_conditions = [
    {
        "target_class": "IC",
        "func": diffusion_ic,
        "on_initial": lambda _, on_initial: on_initial,
        "component": 0,
    },
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0,
        "on_boundary": lambda _, on_boundary: on_boundary,
        "component": 0,
    },
]


def diffusion_solution(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:2])


# DRIFT DIFFUSION
# see https://gitlab.fe.hhi.de/pinns/modelzoo/-/blob/main/deepxde/drift_diffusion_equation/train.py

# initial concentration
DRIFT_DIFFUSION_U_00 = 1.0
# frequency
DRIFT_DIFFUSION_S = 2
# phase shift
DRIFT_DIFFUSION_R = np.pi / 4
# diffusivity
DRIFT_DIFFUSION_ALPHA = 1
# velocity in x direction
DRIFT_DIFFUSION_BETA = 20
DRIFT_DIFFUSION_GAMMA = 1.0
DRIFT_DIFFUSION_X_START = 0.0
DRIFT_DIFFUSION_X_END = 2 * np.pi
DRIFT_DIFFUSION_T_START = 0.0
DRIFT_DIFFUSION_T_END = 1.0


def drift_diffusion_equation(x, y):
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_t = dde.grad.jacobian(y, x, i=0, j=1)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    return du_t - DRIFT_DIFFUSION_ALPHA * du_xx + DRIFT_DIFFUSION_BETA * du_x


def drift_diffusion_solution(x):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    return (
        DRIFT_DIFFUSION_U_00
        * np.sin(
            DRIFT_DIFFUSION_R + DRIFT_DIFFUSION_S * (x_in - DRIFT_DIFFUSION_BETA * t_in)
        )
        * np.exp(-DRIFT_DIFFUSION_ALPHA * DRIFT_DIFFUSION_S**2 * t_in)
    )


def drift_diffusion_ic(x):
    x_in = x[:, 0:1]
    if isinstance(x_in, torch.Tensor):
        sin = torch.sin
    else:
        sin = np.sin
    return DRIFT_DIFFUSION_U_00 * sin(DRIFT_DIFFUSION_S * x_in + DRIFT_DIFFUSION_R)


drift_diffusion_conditions_sine = [
    {
        "target_class": "IC",
        "func": drift_diffusion_ic,
        "on_initial": lambda _, on_initial: on_initial,
        "component": 0,
    },
    {
        "target_class": "PeriodicBC",
        "component_x": 0,
        "on_boundary": lambda x, on_boundary: (
            on_boundary and np.isclose(x[0], DRIFT_DIFFUSION_X_START)
        ),
        "derivative_order": 0,
        "component": 0,
    },
    {
        "target_class": "PeriodicBC",
        "component_x": 0,
        "on_boundary": lambda x, on_boundary: (
            on_boundary and np.isclose(x[0], DRIFT_DIFFUSION_X_END)
        ),
        "derivative_order": 0,
        "component": 0,
    },
]


def drift_diffusion_feature_transform(x):
    x_trafo = x[:, 0:1] / DRIFT_DIFFUSION_X_END
    t_trafo = x[:, 1:2] / DRIFT_DIFFUSION_T_END
    return torch.cat([x_trafo, t_trafo], dim=1)


# NAVIER STOKES
# see https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html

NS_L = 2.2
NS_W = 0.41
NS_CYLINDER_CENTER = np.array([0.2, 0.2])
NS_RADIUS = 0.05
NS_U_MAX = 0.3
NS_NU = 1e-3
NS_RHO = 1.0

NS_U_STAR = 0.2  # characteristic velocity
NS_L_STAR = 0.1  # characteristic length

REYNOLDS = NS_U_STAR * NS_L_STAR / NS_NU

NS_ND_L = NS_L / NS_L_STAR
NS_ND_W = NS_W / NS_L_STAR
NS_ND_CYL_CENTER = NS_CYLINDER_CENTER / NS_L_STAR
NS_ND_RADIUS = NS_RADIUS / NS_L_STAR

NS_ND_U_INFLOW = NS_U_MAX / NS_U_STAR


def navier_stokes_equation_nd(x, y):
    """Dimensionless steady incompressible Navier–Stokes"""
    u = y[:, 0:1]  # u*
    v = y[:, 1:2]  # v*
    p = y[:, 2:3]  # p*

    # first derivatives wrt (x*, y*)
    du_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dv_dx = dde.grad.jacobian(y, x, i=1, j=0)
    dp_dx = dde.grad.jacobian(y, x, i=2, j=0)

    du_dy = dde.grad.jacobian(y, x, i=0, j=1)
    dv_dy = dde.grad.jacobian(y, x, i=1, j=1)
    dp_dy = dde.grad.jacobian(y, x, i=2, j=1)

    # second derivatives (Laplacian) wrt (x*, y*)
    du_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)

    dv_dxx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_dyy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = du_dx + dv_dy
    x_momentum = u * du_dx + v * du_dy + dp_dx - (1.0 / REYNOLDS) * (du_dxx + du_dyy)
    y_momentum = u * dv_dx + v * dv_dy + dp_dy - (1.0 / REYNOLDS) * (dv_dxx + dv_dyy)

    return [continuity, x_momentum, y_momentum]


def create_navier_stokes_geometry_nd():
    geom_space = dde.geometry.Rectangle([0.0, 0.0], [NS_ND_L, NS_ND_W])
    cylinder = dde.geometry.Disk(NS_ND_CYL_CENTER, NS_ND_RADIUS)
    return geom_space - cylinder


def navier_stokes_parabolic_inflow_u_nd(xy):
    y = xy[:, 1:2]  # y*
    u_star = 4 * NS_ND_U_INFLOW * y * (NS_ND_W - y) / (NS_ND_W**2)
    return u_star


def navier_stokes_outflow_u_nd(X, y, _):
    du_dx = dde.grad.jacobian(y, X, i=0, j=0)  # ∂u*/∂x*
    p = y[:, 2:3]  # p*
    return (1.0 / REYNOLDS) * du_dx - p


# For v: (1/Re) ∂v*/∂x* = 0
def navier_stokes_outflow_v_nd(X, y, _):
    dv_dx = dde.grad.jacobian(y, X, i=1, j=0)  # ∂v*/∂x*
    return (1.0 / REYNOLDS) * dv_dx


navier_stokes_conditions_nd = [
    # no-slip u*
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0.0,
        "on_boundary": lambda x, on_boundary: (
            on_boundary
            and (
                np.isclose(x[1], 0.0)
                or np.isclose(x[1], NS_ND_W)
                or (
                    np.sqrt(
                        (x[0] - NS_ND_CYL_CENTER[0]) ** 2
                        + (x[1] - NS_ND_CYL_CENTER[1]) ** 2
                    )
                    <= NS_ND_RADIUS
                )
            )
        ),
        "component": 0,
    },
    # no-slip v*
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0.0,
        "on_boundary": lambda x, on_boundary: (
            on_boundary
            and (
                np.isclose(x[1], 0.0)
                or np.isclose(x[1], NS_ND_W)
                or (
                    np.sqrt(
                        (x[0] - NS_ND_CYL_CENTER[0]) ** 2
                        + (x[1] - NS_ND_CYL_CENTER[1]) ** 2
                    )
                    <= NS_ND_RADIUS
                )
            )
        ),
        "component": 1,
    },
    # inflow u*
    {
        "target_class": "DirichletBC",
        "func": navier_stokes_parabolic_inflow_u_nd,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
        "component": 0,
    },
    # inflow v* = 0
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0.0,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
        "component": 1,
    },
    # outflow traction-like for u*
    {
        "target_class": "OperatorBC",
        "func": navier_stokes_outflow_u_nd,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], NS_ND_L),
    },
    # outflow for v*
    {
        "target_class": "OperatorBC",
        "func": navier_stokes_outflow_v_nd,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], NS_ND_L),
    },
]


def navier_stokes_load_testdata_nd():
    data = np.load(f"{DATASET_DIR}/ns_steady.npy", allow_pickle=True).item()
    u_dim = np.array(data["u"])
    v_dim = np.array(data["v"])
    p_dim = np.array(data["p"])
    coords_dim = np.array(data["coords"]).astype(np.float32)

    # non-dimensionalize
    coords_star = coords_dim / NS_L_STAR
    u_star = u_dim / NS_U_STAR
    v_star = v_dim / NS_U_STAR
    p_star = p_dim / (NS_RHO * NS_U_STAR**2)

    Y_star = np.vstack([u_star, v_star, p_star]).T
    return coords_star, Y_star


def navier_stokes_input_transform_nd(x):
    x_ = x[:, 0:1] / NS_ND_L
    y_ = x[:, 1:2] / NS_ND_W
    return torch.cat([x_, y_], dim=1)


# Poisson disk https://www.mathworks.com/help/pde/ug/poissons-equation-with-point-source-and-adaptive-mesh-refinement.html

# One centered source via annulus trick: solve Laplace on R_in < r < R_out
# with Dirichlet u(R_out)=0 and u(R_in) set so that the solution matches q*delta at the center.

POISSON_CENTER = np.array([0.0, 0.0])
POISSON_RADIUS = 1.0
POISSON_SIGMA = 0.02


def poisson_disk_equation(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    r2 = x[:, 0:1] ** 2 + x[:, 1:2] ** 2

    # delta function approximation
    sigma_sq = POISSON_SIGMA**2
    norm_const = 1.0 / (2.0 * torch.pi * sigma_sq)

    delta_gauss = norm_const * torch.exp(-r2 / (2.0 * sigma_sq))

    return dy_xx + dy_yy + delta_gauss


def create_poisson_disk_geometry():
    return dde.geometry.Disk(POISSON_CENTER, POISSON_RADIUS)


def poisson_disk_solution(x):
    # green's function for 2D laplace operator / poisson equation
    # https://en.wikipedia.org/wiki/Green's_function#Table_of_Green's_functions
    r = np.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    r = np.maximum(r, 1e-12)  # avoid log(0)
    return -np.log(r) / (2.0 * np.pi)


poisson_disk_conditions = [
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0,
        "on_boundary": lambda x, on_boundary: on_boundary,
        "component": 0,
    },
]


def sample_singular_point():
    return np.array([[0.0, 0.0]])


# WAVE
# see https://github.com/lu-group/pinn-sampling/blob/main/src/wave/RAR_G.py


def wave_equation(x, y):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - 4.0 * dy_xx


def wave_solution(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(
        4 * np.pi * x[:, 0:1]
    ) * np.cos(8 * np.pi * x[:, 1:2])


def wave_ic(x):
    x_in = x[:, 0:1]
    if isinstance(x_in, torch.Tensor):
        return torch.sin(np.pi * x_in) + 0.5 * torch.sin(4 * np.pi * x_in)
    else:
        return np.sin(np.pi * x_in) + 0.5 * np.sin(4 * np.pi * x_in)


def wave_initial_velocity_bc(x, y, _):
    du_t = dde.grad.jacobian(y, x, i=0, j=1)
    return du_t


wave_conditions = [
    # u(x, 0) = sin(pi * x) + 0.5 * sin(4 * pi * x)
    {
        "target_class": "IC",
        "func": wave_ic,
        "on_initial": lambda _, on_initial: on_initial,
        "component": 0,
    },
    # u(0, t) = 0
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
        "component": 0,
    },
    # u(1, t) = 0
    {
        "target_class": "DirichletBC",
        "func": lambda x: 0,
        "on_boundary": lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
        "component": 0,
    },
    # du/dx(x, 0) = 0
    {
        "target_class": "OperatorBC",
        "func": wave_initial_velocity_bc,
        "on_boundary": lambda x, _: np.isclose(x[1], 0),
    },
]


problems = {
    "allen_cahn": {
        "equation": allen_cahn_equation,
        "conditions": allen_cahn_conditions,
        "load_testdata": allen_cahn_load_testdata,
        "solution": None,
    },
    "burgers": {
        "equation": burgers_equation,
        "conditions": burgers_conditions,
        "load_testdata": burgers_load_testdata,
        "solution": None,
    },
    "diffusion": {
        "equation": diffusion_equation,
        "conditions": diffusion_conditions,
        "load_testdata": None,
        "solution": diffusion_solution,
    },
    "drift_diffusion": {
        "equation": drift_diffusion_equation,
        "conditions": drift_diffusion_conditions_sine,
        "load_testdata": None,
        "solution": drift_diffusion_solution,
        "x_start": DRIFT_DIFFUSION_X_START,
        "x_end": DRIFT_DIFFUSION_X_END,
        "t_start": DRIFT_DIFFUSION_T_START,
        "t_end": DRIFT_DIFFUSION_T_END,
        "feature_transform": drift_diffusion_feature_transform,
    },
    "navier_stokes_nd": {
        "equation": navier_stokes_equation_nd,
        "load_testdata": navier_stokes_load_testdata_nd,
        "solution": None,
        "conditions": navier_stokes_conditions_nd,
        "use_geometry": True,
        "create_geometry": create_navier_stokes_geometry_nd,
        "feature_transform": navier_stokes_input_transform_nd,
    },
    "poisson_disk": {
        "equation": poisson_disk_equation,
        "conditions": poisson_disk_conditions,
        "load_testdata": None,
        "solution": poisson_disk_solution,
        "use_geometry": True,
        "create_geometry": create_poisson_disk_geometry,
        "sample_points": sample_singular_point(),
    },
    "wave": {
        "equation": wave_equation,
        "conditions": wave_conditions,
        "load_testdata": None,
        "solution": wave_solution,
        "x_start": 0,  # NOTE the different x_start
    },
}
