# 2JITsu

A comparative analysis of the pytorch and JAX jit (torch.compile) capabilities, as well as the speed of both frameworks' generated code for training and inference.

---

## Structure

We structure the files into a folder for each evaluated model and subfolders for jax and pytorch. Each subfolder for jax/pytorch will contain an idiomatic implementation of the given model (in another subfolder) and a jupyter notebook at the top level. The jupyter notebook will contain annotated driver code for the experiments, but is duplicated in a python file (also at the top level).

The results for each model will be printed to a file `results.csv` where we have the following schema:

```csv
model name, framework, evaluation (jit, train, inference), total iterations, warmup iterations, average time, notes
```

By default, the average time is taken from the last (total iterations - warmup iterations) runs and averaged using the arithmatic mean.