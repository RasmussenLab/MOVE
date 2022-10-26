# Tutorial

## Random Small

We have provided a tutorial. In this first tutorial, we inspect datasets 
reporting whether 500 fictitious individuals have taken one of 20 imaginary
drugs. We have included a pair of pretend omics datasets, with measurements
for each sample (individual). All these measurements were generated randomly,
but we have added 200 associations between different pairs of drugs and omics
features. Let us find them with MOVE!

### Workspace structure

First, we take a look at how to organize our data and configuration:


```
tutorial/
│
├── data/
│   ├── changes.small.txt              <- Ground-truth associations (200 links)
│   ├── random.small.drugs.tsv         <- Drug dataset (20 drugs)
│   ├── random.small.ids.tsv           <- Sample IDs (500 samples)
│   ├── random.small.proteomics.tsv    <- Proteomics dataset (200 proteins)
│   └── random.small.metagenomics.tsv  <- Metagenomics dataset (1000 taxa)
│
└── config/                            <- Stores user configuration files
    ├── data/
    │   └── random_small.yaml          <- Configuration to read in the necessary
    │                                     data files.
    ├── experiment/                    <- Configuration for experiments (e.g.,
    │   └── random_small__tune.yaml       for tuning hyperparameters).
    │
    └── task/                          <- Configuration for tasks: such as
        |                                 latent space or identify associations
        │                                 using the t-test or Bayesian approach
        ├── random_small__id_assoc_bayes.yaml
        ├── random_small__id_assoc_ttest.yaml
        └── random_small__latent.yaml
```

#### The data folder

All "raw" data files should be placed inside the same directory. These files
are TSVs (tab-separated value tables) containing discrete values (e.g., for
binary or categorical datasets) or continuous values.

Additionally, make sure each sample has an assigned ID and we provide an ID
table containing a list of all valid IDs (must appear in every dataset).

#### The `config` folder

User-defined configuration must be stored in a `config` folder. This folder
can contain a `data` and `task` folder to store the configuration for a
specific dataset or task.

Let us take a look at the configuration for our dataset. It is a YAML file,
specifying: a default layout\*, the directories to look for raw data and store
intermediary and final output files, and the list of categorical and continuous
datasets we have.

```yaml
# DO NOT EDIT

defaults:
  - base_data

# FEEL FREE TO EDIT BELOW

raw_data_path: data/              # where raw data is stored
interim_data_path: interim_data/  # where intermediate files will be stored
processed_data_path: results/     # where result files will be placed

sample_names: random.small.ids  # names/IDs of each sample, must appear in the
                                # other datasets

categorical_inputs:  # a list of categorical datasets and their weights
  - name: random.small.drugs

continuous_inputs:   # a list of continuous-valued datasets and their weights
  - name: random.small.proteomics
  - name: random.small.metagenomics
```

<span style="font-size: 0.8em">\* We do not recommend changing `defaults`
unless you are sure of what you are doing.</span>

Similarly, the `task` folder contains YAML files to configure the tasks of
MOVE. In this tutorial, we provided two examples for running the method to
identify associations using our t-test and Bayesian approach, and an example to
perform latent space analysis.

For example, for the t-test approach (`random_small__id_assoc_ttest.yaml`), we
define the following values: batch size, number of refits, name of dataset to
perturb, target perturb value, configuration for VAE model, and configuration
for training loop.

```yaml

defaults:
  - identify_associations_ttest

batch_size: 10  # number of samples per batch in training loop

num_refits: 10  # number of times to refit (retrain) model

target_dataset: random.small.drugs  # dataset to perturb
target_value: 1                     # value to change to
save_refits: True                   # whether to save refits to interim folder

model:         # model configuration
  num_hidden:  # list of units in each hidden layer of the VAE encoder/decoder
    - 1000

training_loop:    # training loop configuration
  lr: 1e-4        # learning rate
  num_epochs: 40  # number of epochs

```

Note that the `random_small__id_assoc_bayes.yaml` looks pretty similar, but
declares a different `defaults`. This tells MOVE which algorithm to use!

### Running MOVE

#### Encoding data

Make sure you are on the parent directory of the `config` folder (in this
example, it is the `tutorial` folder), and proceed to run:

```bash
>>> cd tutorial
>>> move-dl data=random_small task=encode_data
```

:arrow_up: This command will encode the datasets. The `random.small.drugs`
dataset (defined in `config/data/random_small.yaml`) will be one-hot encoded,
whereas the other two omics datasets will be standardized.

#### Analyzing the latent space

Next, we will train a variational autoencoder and analyze how good it is at
reconstructing our input data and generating an informative latent space. Run:

```bash
>>> move-dl data=random_small task=random_small__latent
```

:arrow_up: This command will create four types of plot:

- Loss curve shows the overall loss, KLD term, binary cross-entropy term, and
sum of squared errors term over number of training epochs.
- Reconstructions metrics boxplot shows a score (accuracy or cosine similarity
for categorical and continuous datasets, respectively) per reconstructed
dataset.
- Latent space scatterplot shows a reduced representation of the latent space.
To generate this visualization, the latent space is reduced to two dimensions 
using TSNE (or another user-defined algorithm, e.g., UMAP).
- Feature importance swarmplot displays the impact perturbing a feature has on
the latent space.

Additionally, TSV files corresponding to each plot will be generated. These can
be used, for example, to re-create the plots manually.

#### Identifying associations

Next step is to find associations between the drugs taken by each individual
and the omics features. Run:

```bash
>>> move-dl data=random_small task=random_small__id_assoc_ttest
```

:arrow_up: This command will create a `results_sig_assoc.tsv` file, listing
each pair of associated features and the corresponding median p-value for such
association. There should be ~120 associations found.

:warning: Note that the value after `task=` matches the name of our
configuration file. We can create multiple configuration files (for example,
changing hyperparameters like learning rate) and call them by their name here.

If you want to run, the Bayesian approach instead. Run:

```bash
>>> move-dl data=random_small task=random_small__id_assoc_bayes
```
Again, it should generate similar results with over 100 associations known.

Take a look at the `changes.small.txt` file and compare your results against
it. Did MOVE find any false positives?

#### Tuning the model's hyperparameters

Additionally, we can improve the reconstructions generated by MOVE by running
a grid search over a set of hyperparameters.

We define the hyperparameters we want to sweep in an `experiment` config file,
such as:

```yaml
# @package _global_

# Define the default configuration for the data and task (model and training)

defaults:
  - override /data: random_small
  - override /task: tune_model

# Configure which hyperarameters to vary
# This will run and log the metrics of 12 models (combination of 3 hyperparams
# with 2-3 levels: 2 * 2 * 3)

# Any field defined in the task configuration can be configured below.

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      task.batch_size: 10, 50
      task.model.num_hidden: "[500],[1000]"
      task.training_loop.num_epochs: 40, 60, 100
```

The above configuration file will generate different combinations of batch size,
hidden layer size, and training epochs. Then each model will run with one of 
these combinations and the reconstructions metrics will be recorded in a TSV
file.
