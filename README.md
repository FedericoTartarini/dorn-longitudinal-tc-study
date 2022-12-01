# Dorn - Longitudinal thermal comfort study

This research aims to develop a personalized thermal comfort model for each of the study participants by
collecting: environmental and personal parameters; physiological variables; and, user’s reported thermal
perceptions.

The hypothesis of this study is that personalized thermal comfort models have better prediction accuracy
than aggregate thermal comfort models.

The rationale of this research is to develop personalized comfort models that can be used to control and
optimize thermal comfort conditions in built environments.

## Research questions

This research aims to answers the following questions:
* Can Internet-of-Things (IoT) devices and wearables devices effectively be used to collect sufficient data to
create personalized thermal comfort models while minimizing the impact on users?
* Can personalized thermal comfort models better predict how human perceive their thermal environment,
than existing thermal comfort models?
* How many data points per user need to be collected to develop a reliable and robust comfort model?

## Table of content 

-   [Getting Started](#getting-started)
    -   [Data](#source-data)
    -   [Data Analysis](#data-analysis)
    -   [Manuscript and Presentation](#manuscript-and-presentation)
-   [Prerequisites](#prerequisites)
    -   [Latex](#latex)
-   [Author](#authors)

## Getting Started

The directory is divided in sub-folders. Each of which contains the relative source code. Just clone this repository on your computer or Fork it on GitHub

### Source Data

The objective of this folder is not to replace the database, but instead to share only some of the database data with other researchers. Data should only contain the .csv files that you want to share publicly. 

>It **SHOULD NOT** contain identifiable data. 

### Data Analysis

The source code used to analyse the data is all saved on the `/code/` folder.

## Setup

Install the dependencies using the `Pipfile.lock`

## Training

In order to train Personal Comfort Models, a file `train_pcm.py` exists. Currently supported models and their settings can be found in the `configuration.py` file as keys in the `model_configs` dictionary. Other experiment wide details like number of iterations of metrics for evaluation could also be configured in the same file. Finally, all detailed modeling and evaluation functions are placed in a `utils.py` file.

An example of running the training file is as follows:
```training
python train_pcm.py pcm_tp_rdf_logged_clo_met
```

### Manuscript and Presentation

The manuscript contains all the Latex files needed to generate your manuscript and your presentation. The main source files are located in the `manuscript/src`.

* `main.tex` is the manuscript source file. 

However, in order to keep the code cleaner, the main sections of the paper are all located in `/sections/`.

### Prerequisites

#### Latex

Latex IDE and compiler installed locally on your machine. I recommend using a PyCharm plugin called [TeXiFy IDEA](https://plugins.jetbrains.com/plugin/9473-texify-idea) as IDE and [miktex](https://miktex.org) as Latex compiler  

Alternatively you can push your code to Overleaf using git and only use Overleaf. I would discourage you from doing this! Overleaf should only be used for the review.

## Authors

* **[Federico Tartarini](https://github.com/FedericoTartarini)** - *Initial work*