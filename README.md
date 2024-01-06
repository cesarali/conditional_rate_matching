# __Discrete States Generative Diffusion Model Library__

## __Abstract__

Framework for __training __ that implements generative diffusion models for discrete data.

## __Framework Structure__

For unified training and evaluation and easy implementation of 
new diffusion models we define in our framework two basic concepts,
_Model_ and _Model Trainer_.

__Model__. This concept is implemented as abstract class ``AModel`` in [discrete-diffusion.models]. The UML diagram of the class can be find below.

![AModel](reports/figures/AModel.png)


## __Installation__

In order to set up the necessary environment:

### __Conda Virtual Enviroment__

1. Install [virtualenv] and [virtualenvwrapper].
2. Create conda env for the project:

    ```bash
    conda create -n conditional_rate_matching python=3.10.9
    ```
3. Activate the enviroment

    ```bash
    activate conditional_rate_matching
    ```
4. Install torch enable cuda

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
   
5. Install the project in edit mode:

    ```bash
    python setup.py develop
    pip install -e .
    ```

6. create data folders in main directory:
    `data`
    `data/raw`
    `data/preprocessed`
    look at project organization below

7. For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

    ```sh
    cd evaluation/orca 
    g++ -O2 -std=c++11 -o orca orca.cpp
    ```

## __Project Organization__

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── src
│   └── conditional_rate_matching        <- Actual Python package where the main functionality goes.
│       ├── configs
│       ├── data
│       ├── models
│           ├── generative_models        <- Classes that aggreagets everything needed for a generative models
│           ├── losses                   
│           ├── metrics
│           ├── networks                 <- Neural Networks Architectures
│           ├── pipelines
│           ├── schedulers
│           ├── temporal_networks        <- Neural Networks Architectures with Time for Diffusion
│           └── trainers 
│       └── utils
├── data
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── requirements.txt         <- The python environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.py                <- Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```
