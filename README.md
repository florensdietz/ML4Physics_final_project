# ML4Physics_final_project
This repository reproduces the results from the paper "Learning Physics with Graph Networks" which has to be presented in the ML4Physics course.

Use the notebook "n_body_data_generation.ipynb" to create some simulated data which can be used for training the GN model.
There you can also create the data that is used to test if the model generalizes well to case where more bodies are simulated.

Right now, the model is trained to minimize the the L1 loss of the updated velocity $v_{t+1}$ between the simulated data and the predicted data. It might be better to change that to optimize for the updates $\Delta v$.

To do:
1. Symbolic regression
2. Create some plots for presentation
