# cloudmask-fit
Cloudmask training, prediction and evaluation of the results.

Refer to [wiki](https://github.com/kappazeta/cloudmask-fit/wiki) for more detailed description of the workflow, model architectures and results.

## Setup
1. Create a conda environment.

        conda env create -f environment.yml

2. Copy `config/config_example.json` and adapt it to your needs.

## Usage

Model fitting can be performed as follows:

```
conda activate cm_fit
python3 cm-fit.py -c config/your_config.json
```

Once the model fitting has concluded, the model weights in the `output` directory can be used for prediction.
This can be performed as follows:

```
conda activate cm_fit
python3 cm-fit.py -c config/your_config.json -p input/for_prediction/T35VLF_20200529T094041_tile_1024_4864.nc -w output/a1_029-0.96.hdf5
```

