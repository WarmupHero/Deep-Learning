=========================================================================================================================================================================

TABLE OF CONTENTS

1. Project Overview
   - Project Overview
   - High level pipeline mapping
   - Library dependencies
   - Execution sequence

2. Experiments Configurability Overview
   - What is/is not configurable from JSON
   - Summary

3. Analysis details
   - requirements.py
   - analysis.py
   - Summary

=========================================================================================================================================================================

# 1. Project Overview

This project implements a from-scratch feed-forward neural network for two tasks: binary classification and regression. 
The pipeline begins by downloading the datasets, then preprocessing them through cleaning, optional scaling, and train/validation/test splitting. 
Experiment settings such as architectures, optimizers, learning rates, batch sizes, and early stopping are loaded from JSON configuration files. 
The selected network is then trained and evaluated, and the results are saved for later use. 
Separate scripts read those saved results to generate the required plots and short analyses.

=========================================================================================================================================================================

# High level pipeline mapping
- utils.py
  Stores shared constants and helper values such as file paths and random seed settings.
  uses as input: none
  outputs: none
  used as input: main.py
                 src/fetch_data.py
                 src/config_loader.py
                 src/preprocessing.py 
                 src/network.py
                 src/train.py
                 src/requirements.py
                 src/analysis.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- fetch_data.py
  Downloads the raw datasets used in the project and stores them locally as CSV files.
  uses as input: datasets/banknote_auth.csv
                 datasets/energy_efficiency.csv
                 src/utils.py
  outputs: none
  used as input: main.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- preprocessing.py
  Receives the raw dataset, performs minimal preprocessing such as duplicate removal and optional scaling, and splits the data into training, validation, and test sets.
  uses as input: datasets/banknote_auth.csv
                 datasets/energy_efficiency.csv
                 src/scalers.py
                 src/visualizations.py
                 src/utils.py
  outputs: none
  used as input: main.py

- src/scalers.py
  Implements the custom standard scaler ((x-μ)/σ) used during preprocessing.
  uses as input: none
  outputs: none
  used as input: src/preprocessing.py


- src/visualizations.py
  Generates the preprocessing and exploratory plots used to inspect the datasets.
  uses as input: none
  outputs: none
  used as input: src/preprocessing.py


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- network.py
  Builds the feed-forward neural network based on the selected architecture configuration.
  uses as input: src/layers.py
                 src/activations.py
                 src/utils.py
  outputs: none
  used as input: main.py

- layers.py
  Defines the dense layer operations used inside the network.
  uses as input: none
  outputs: none
  used as input: src/network.py

- activations.py
  Implements the supported activation functions used by the network layers.
  uses as input: none
  outputs: none
  used as input: src/network.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- main.py
  Orchestrates the full experiment pipeline by connecting data loading, preprocessing, model creation, training, and result saving.
  Includes option show_eda to show or not show graphs in preprocessing step when running through terminal.
  uses as input: src/fetch_data.py
                 src/config_loader.py
                 src/preprocessing.py
                 src/network.py
                 src/losses.py
                 src/optimizers.py
                 src/train.py
                 src/utils.py
  outputs: report/main_results_full.json
           report/main_summary.csv
  used as input: none

- losses.py
  Defines the loss functions used for classification and regression.
  uses as input: none
  outputs: none
  used as input: main.py

- optimizers.py
  Implements the optimizers used to update network parameters during training.
  uses as input: none
  outputs: none
  used as input: main.py

- metrics.py
  Computes the evaluation metrics used to measure model performance.
  uses as input: none
  outputs: none
  used as input: src/train.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- train.py
  Runs the training loop, validation checks, early stopping, and final evaluation.
  uses as input: src/metrics.py
                 src/utils.py
  outputs: none
  used as input: main.py

- src/metrics.py
  Computes the task-specific evaluation metrics used to assess model performance.
    uses as input: none
    outputs: none
    used as input: src/train.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- requirements.py
  Reads the saved experiment results and generates plots.
  uses as input: report/main_results_full.json (generated by main.py)
                 src/utils.py
  outputs: report/analysis.txt
  used as input: none

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

- analysis.py
  Reads the saved experiment results and produces a summary analysis.
  uses as input: report/main_results_full.json (generated by main.py)
                 src/utils.py
  outputs: report/analysis.txt
  used as input: none

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# A detailed section on configurables is dedicated later in this README file
- config_loader.py
  Loads the JSON experiment settings that define architectures, optimizers, learning rates, batch sizes, and training options.
  uses as input: configs/classification_experiments.json
                 configs/regression_experiments.json
                 src/utils.py
  outputs: none
  used as input: main.py

- configs/classification_experiments.json
    uses as input: none
    outputs: none
    used as input: src/config_loader.py

- configs/regression_experiments.json
    uses as input: none
    outputs: none
    used as input: src/config_loader.py

=========================================================================================================================================================================

# Library dependencies
This repository expects the following library versions:

Python: 3.12.7
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.9.2
seaborn==0.13.2
ucimlrepo==0.0.7 # !pip install ucimlrepo

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Execution sequence

1. Open Command Prompt.

2. Navigate to the project root directory:
   cd path\to\root_dir

3. Run the main experiment pipeline:
   python main.py
   - load/download the datasets
   - preprocess the data
   - train all configured experiments
   - save results to:
     - report\main_results_full.json
     - report\main_summary.csv

4. Generate the optimizers, network depth, and learning rate comparisons:
   python -m src.requirements
   This will read the saved results from report\main_results_full.json
   and save the requirement outputs under:
   - report\requirements_1-2-3

5. Generate the summary analysis:
   python -m src.analysis

   This will read the saved results from report\main_results_full.json
   and save the final analysis report to:
   - report\analysis.txt

If main.py has not been run first, those scripts will fail because
report\main_results_full.json will not exist.

=========================================================================================================================================================================

# 2. Experiments Configurability Overview

This codebase uses two JSON configuration files to define the experiment setup:

- `configs/classification_experiments.json`
- `configs/regression_experiments.json`

These files control a meaningful part of the experiment pipeline, but only within the limits of the current implementation. 
In other words, they are configurable, but they do not define a fully general neural-network framework.

## What is configurable from JSON

The following settings can be changed directly in the JSON files.

- **Preprocessing**
  - preprocessing enabled/disabled <!-- Controls optional preprocessing behavior in the current pipeline. If `true`, the dataset is cleaned and can optionally be scaled. 
    If  `false`, duplicates/cleaning/EDA/scaling are skipped, but the dataset is still split into train/validation/test because the training pipeline requires those splits. -->
  - feature scaling on/off <!-- Controls whether input features are standardized with `StandardScaler` using statistics fit on the training split only. Must be `true` or `false`. 
    If preprocessing is disabled, scaling is also effectively skipped. -->

- **Architectures**
  - architecture names (for example, `A1`, `A2`) <!-- Any JSON key / string name is allowed. Not restricted to A1 and A2. User would have to create a new architecture. -->
  - number of layers <!-- Must contain at least 1 layer. -->
  - number of units per layer <!-- Any positive integer `>= 1`. Number of neurons in the layer - the size of that layer’s output. 
    No explicit upper bound in code beyond practical compute/memory limits. -->
  - activation function per layer <!-- Limited to the implemented set: `relu`, `sigmoid`, `tanh`, `linear`. -->

- **Experiment sweep values**
  - optimizer list <!-- Limited to implemented optimizers. Supported names in the current code are `sgd`, `momentum`, `momentumsgd`, `momentum_sgd`, `adabelief`. -->
  - learning rates <!-- Positive numeric values. Common choices are 0.1, 0.01, or 0.001. -->
  - batch sizes <!-- Restricted based on the shape of the dataset. -->

- **Training settings**
  - number of epochs <!-- Any positive integer `>= 1`. -->
  - early stopping on/off <!-- Boolean only: `true` or `false`. -->
  - patience - patience <!-- Number of consecutive non-improving validation checks allowed before stopping. -->
  - minimum improvement threshold (`min_delta`) <!-- Minimum required decrease in validation loss to count as a real improvement. Recommended be small relative to the scale of the chosen loss. -->

- **Problem-level settings**
  - task type <!-- Restricted to `classification` or `regression`. -->
  - loss function <!-- Limited to implemented and validated losses. Classification accepts `bce`, `binary_crossentropy`, `binary_cross_entropy`. Regression accepts `mse`. -->
  - input dimension <!-- Must match the real number of dataset features or execution will fail. -->

## What is not freely configurable from JSON

### 1. Layer types are not fully configurable

Each layer specification includes a `"type"` field, but the current implementation only supports:

- `dense`

This means the JSON files cannot currently define unsupported layer types such as convolutional layers, dropout layers, batch normalization layers, 
or recurrent layers unless the code is extended first.

### 2. Optimizer internals are mostly hard-coded

The JSON files control the learning rate, but most optimizer-specific internal hyperparameters are fixed in code.

- Momentum uses a fixed momentum coefficient
- AdaBelief uses fixed beta and epsilon values

### 3. Evaluation metrics are not selected in JSON

The evaluation metric is determined by the task-specific code, not by the configuration files. 
In the current implementation, classification and regression each use BCE and MSE respectively.

### 4. Random Seed is not controlled by these JSON files

The JSON files do not define dataset preparation choices such as:

- random seed - addressed in utils.py. Any change to the random seed must be made there.

### 5. Display of EDA plots is controlled in main.py

- show_eda = False by default. To view plots while main is executing you may change this to True. 
Preprocessing plots are saved under report/preprocessing graphs regardless of show_eda settings.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Configurability Summary
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The JSON files make the experiment setup mostly configurable. They allow changes to:

- architecture definitions
- optimizer selection
- learning rates
- batch sizes
- training length
- early stopping settings
- supported task and loss choices

However, this environment operates inside a hard-coded implementation that currently supports:
- feed-forward dense networks only
- a fixed set of activations
- a fixed set of losses
- a fixed set of optimizers
- fixed optimizer hyperparameters beyond learning rate
=========================================================================================================================================================================

# 3. Analysis details
# requirements.py and analysis.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

requirements.py

The purpose of requirements.py is to compare optimizers, network depth, and learning rates. It loads the full JSON results file, 
selects matched experiment runs, plots the training-loss and validation-loss curves, and saves the figures. 
For network depth and learning rates comparison, it also writes a short text explanation describing which setting converged faster and which ended with lower loss.

The script works by comparing runs where most settings stay fixed and only one factor changes.

For optimizers comparisons, the following settings are kept fixed:
- problem_name
- architecture
- learning_rate
- batch_size

The selected runs are filtered from the JSON results and sorted in a fixed order: SGD, Momentum, AdaBelief. 
Then the script creates one subplot per optimizer and plots both the training loss and validation loss over epochs.

For network depth comparisons, the following settings are fixed:
- problem_name
- optimizer
- learning_rate
- batch_size

Only the architecture changes, comparing A1 against A2. 
The script plots both runs and also writes a short analysis text file explaining which 
architecture converged faster and which one ended with lower final training and validation loss.

For learning-rate sensitivity analysis, the following settings are fixed:
- problem_name
- architecture
- optimizer
- batch_size

Only the learning rate changes, comparing 0.1 against 0.001. 
The script again produces a figure and a short analysis text file.

One important calculation in requirements.py is the convergence metric. 
Under the current repo scope convergence is measured from the training-loss curve.

The script computes convergence as follows:

    1. Take the final training loss:
       final_train_loss = last value in train_loss_history
    
    2. Build a tolerance around that final value:
       tolerance = max(absolute_floor, relative_tolerance * abs(final_train_loss))
    
       The default values are:
       - relative_tolerance = 0.01
       - absolute_floor = 1e-4
    
       This means the script usually uses a band of 1% around the final training loss, but never allows the tolerance to become smaller than 0.0001.
    
    3. Starting from epoch 1, scan forward and find the first epoch after which all remaining training-loss values stay within that tolerance band around the final training loss.

That first stable epoch is called the convergence epoch.

This gives a practical definition of “when training has essentially settled down.” The same helper also records:
- epochs_ran
- final_train_loss
- final_val_loss
- convergence_epoch

These values are shown in the metric box for the network depth and learning rates plots.

The plotting process is organized into helper functions:
- load_results(path): loads the JSON file
- filter_optimizer_runs(...): selects the 3 matched optimizer runs
- filter_depth_runs(...): selects the 2 matched architecture runs
- filter_learning_rate_runs(...): selects the 2 matched learning-rate runs
- _compute_convergence_metrics(run): computes final losses and convergence epoch
- _plot_loss_curves(...): draws train/validation curves for one run
- _save_and_show(...): saves the figure and displays it
- _write_text_file(...): saves the short written analysis

Finally, the main() function chooses which exact experiments to visualize by calling the three plot functions with fixed arguments. 
To visualize a different experiment, the user needs to change those function arguments, while keeping the comparison logic the same.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

analysis.py

The purpose of analysis.py is to answersthree questions:
1. Which optimizer converges fastest on average?
2. Which optimizer produces the best loss function value?
3. How does network depth affect optimization?

To do this, analysis.py first adds derived metrics to every run.

For each run, it computes:
- best_val_loss = minimum value in val_loss_history
- final_train_loss = last value in train_loss_history
- convergence_epoch = first epoch after which all remaining training-loss values stay within tolerance of the final training loss

The convergence calculation is the same one used in requirements.py:
- final_train_loss = last training loss
- tolerance = max(1e-4, 0.01 * abs(final_train_loss))
- convergence_epoch = first epoch after which the remaining training losses all stay inside that tolerance band

This means “fastest convergence” is measured by the average convergence epoch. Lower is better.

The next step is loss normalization.
This is necessary because classification and regression do not use the same loss function:
- classification uses BCE
- regression uses MSE
which are on different scales so they cannot be averaged directly across both tasks.

To solve this, analysis.py normalizes best validation loss separately inside each problem using **min-max normalization**:

normalized_best_val_loss =
    (best_val_loss - min_loss_in_that_problem) /
    (max_loss_in_that_problem - min_loss_in_that_problem)

This puts each tasks' losses on a 0 to 1 scale before combining them. 
If all losses inside one problem are identical, the script assigns 0.0 to avoid division by zero.

After that, the script groups runs in different ways.

First, it creates task-specific groups:
- classification runs
- regression runs

Inside each task, it groups by:
- optimizer
- architecture

For each group, it computes:
- num_runs
- avg_convergence_epoch
- avg_best_val_loss

Next, it creates combined overall groups across all runs. In that case, it groups by:
- optimizer
- architecture

For each combined group, it computes:
- num_runs
- avg_convergence_epoch
- avg_normalized_best_val_loss

Here normalized loss is used instead of raw loss because the analysis mixes classification and regression together.

The helper functions used in analysis.py are:
- load_results(path): loads the JSON results
- convergence_epoch(...): computes the convergence epoch from training loss
- add_derived_metrics(results): adds best_val_loss, final_train_loss, and convergence_epoch
- add_normalized_best_val_loss(results): normalizes best validation loss within each task
- filter_by_problem(...): separates classification and regression
- group_by_key(...): groups runs by optimizer or architecture
- summarize_task_group(...): computes averages inside one task
- summarize_combined_group(...): computes averages across all runs using normalized loss
- best_group(summary_dict, field_name): selects the group with the smallest value in the chosen field

The rule for selecting the “best” group is:
- smaller avg_convergence_epoch means faster convergence
- smaller avg_best_val_loss means better loss inside one task
- smaller avg_normalized_best_val_loss means better combined overall loss across tasks

For the network-depth question, the script compares A1 and A2 in the combined architecture summary. It checks:
- which architecture has the smaller average convergence epoch
- which architecture has the smaller average normalized best validation loss

Analysis.py formats the results into plain-text tables and direct answers.

The report contains:
- a Method section
- supporting tables for classification
- supporting tables for regression
- supporting combined-overall tables
- final answers for the three summary questions

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Summary:
- requirements.py is for visual comparison of selected matched experiments
- analysis.py is for full-summary reporting across all experiments

requirements.py helps explain individual experiments clearly with plots and short text analyses.
analysis.py helps answer the overall questions using averages across all runs.

=========================================================================================================================================================================
