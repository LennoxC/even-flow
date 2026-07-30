# even-flow

```
Even flow, thoughts arrive like butterflies
Oh, he don't know, so he chases them away
Someday yet, he'll begin his life again
Whispering hands, gently lead him away
```
*- Eddie Vedder, 1991*

Eddie was talking about flow matching generative models (*"even flow"*) which act as a prior for Bayesian inverse problems, generating physically realistic samples that are consistent with data (*"thoughts arrive like butterflies"*) while still exploring the solution space and generating diverse solutions to effectively model uncertainty (*"Oh, he don't know, so he chases them away"*).

This repository contains pytorch implementations of deep learning modules, used by Earth Sciences New Zealand for experimenting with generative AI for data assimilation.

## Run Tests

Running the tests is the easiest way to check that some basic models still compile and can do a forward pass. Note that pytest is only installed in the test pixi environment. This can also be run from the VSCode command pallet, as the `.vscode/tasks.json` has been included for your convenience. `cmd+shift+p` > `Tasks: Run Task` > `run all tests` > `Continue without scanning the task output`.

Tests are grouped:
- fast
- detailed

You can alternatively choose to just run the fast tests, either with the -m flag, or selecting "run fast tests" from the vscode tasks. Some of the detailed tests require backprop which may not be feasible on slow computers. The fast tests still perform a forward pass.

```
pixi run --environment test pytest ./src/even_flow/tests
```

