# PINNfluence: Influence Functions for PINNs

This repository is the official implementation of [PINNfluence: Influence Functions for PINNs](https://arxiv.org/abs/2409.08958). 

Authors: Jonas R. Naujoks<sup>1</sup>, Aleksander Krasowski<sup>1</sup>, Moritz Weckbecker, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek, René P. Klausen

<sup>1</sup> Equal contribution.

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

Note that we use a slightly [modified version of captum](https://github.com/aleks-krasowski/captum) which allows for computing influence on arbitrary function outputs $f(x,\cdot)$, instead of being limited to loss terms:

$$
Inf_{f(x,\cdot)} (z_i) := \nabla_{\hat\theta} f(x;\hat\theta) \cdot H_{\hat\theta}^{-1} \cdot \nabla_{\hat\theta} L(z_i; \hat\theta) 
$$

Please furthermore set up the deepxde backend from inside the environment to PyTorch.

```bash 
python -m deepxde.backend.set_default_backend pytorch
```

## Training

To train the model(s) in the paper, run this command:

```bash 
# good 
python train.py --save_path ./model_zoo/good

# broken
python train.py --broken --save_path ./model_zoo/broken

# bad
python train.py --num_domain 1500 --num_boundary 500 --save_path ./model_zoo/broken
```

To calculate influences please run the following. The results will be stored in the `save_path` argument provided.
To speed up calculation a high batch size and a GPU or TPU is recommended. It should be automatically selected if available.

```bash 
# good 
# please adjust model checkpoint if necessary
python calc_influences.py --checkpoint ./model_zoo/good/lbfgs-116386.pt --save_path ./model_zoo/good_influences --train_x_path ./model_zoo/good/train_x.npy --batch_size <what_your_hardware_allows>

# broken 
python calc_influences.py --broken --checkpoint ./model_zoo/broken/lbfgs-122314.pt --save_path ./model_zoo/broken_influences --train_x_path ./model_zoo/broken/train_x.npy --batch_size <what_your_hardware_allows>

# bad
python calc_influences.py --checkpoint ./model_zoo/bad/lbfgs-107271.pt --save_path ./model_zoo/bad_influences --train_x_path ./model_zoo/bad/train_x.npy --batch_size <what_your_hardware_allows>
```

## Evaluation

Please run the `eval.ipynb` to generate all figures and results used in the paper. Feel free to experiment around and create more figures.

## Pre-trained Models

As the models are fairly small they can be found directly in the `model_zoo` directory.

They were trained with the arguments provided in the **Training** section.

## Citing 

Please use the following citation when referencing PINNfluence in literature:

```bibtex
@misc{naujoks2024pinnfluenceinfluencefunctionsphysicsinformed,
      title={PINNfluence: Influence Functions for Physics-Informed Neural Networks}, 
      author={Jonas R. Naujoks and Aleksander Krasowski and Moritz Weckbecker and Thomas Wiegand and Sebastian Lapuschkin and Wojciech Samek and René P. Klausen},
      year={2024},
      eprint={2409.08958},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.08958}, 
}
```

## License

This project is licensed under the BSD-3-Clause-Clear License. Please see the [LICENSE](./LICENSE) file for more information.