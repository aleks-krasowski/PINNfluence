import numpy as np 

jobs = [] 

for ratio in np.arange(0, 0.11, 0.01):
    for n_iter in [10_000, 50_000]:
        for method in ['random', 'IF', 'RAR']:
            for seed in range(5):
                job = f"--ratio {ratio} --method {method} --n_iterations {n_iter} --seed {seed}"
                jobs.append(job)

                job = f"--ratio {ratio} --method {method} --n_iterations {n_iter} --seed {seed} --hard_constrained"
                jobs.append(job)

with open("batch_low_ratios.txt", 'w') as fp:
    for job in jobs:
        fp.write(job)
        fp.write("\n")
