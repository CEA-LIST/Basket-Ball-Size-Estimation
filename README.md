
# Ball 3D localization challenge

This repository contains the method we developped to participate to the [MMSports 2022 ball 3D localization challenge](https://eval.ai/web/challenges/challenge-page/1688).
It is described in this [report](doc/BasketBallSizeEstimation.pdf).
The source code is available under the CeCILL 2.1 license.

## Installation

Create a virtual environment:
```
virtualenv venv
source venv/bin/activate
```

Install the dependancies:
```
pip install -r requirements.txt
```

## Dataset

Please follow the [instructions](https://github.com/DeepSportRadar/ball-3d-localization-challenge#participating-with-another-codebase) available on the challenge repository to download and generate the dataset pickle files.

## Usage

To train the model, use the command line:
```
CUBLAS_WORKSPACE_CONFIG=:16:8 python train.py
```

A model is saved every 10 epochs in the `output` directory.

To evaluate the model, use the command line:
```
python eval.py dataset_pickle_file_path model_path
```
You can use the `--visualization` command line option to display images of the estimations.

The model used for our submission to the evaluation server is available in the `models` directory.
