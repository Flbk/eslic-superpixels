# VIC course project on SuperPixels!

## Requirements

We will try to use a `requirements.txt` file to manage dependencies.

First create a virtual environnment and activate it:

```bash
virtualenv vic-env
source vic-env/bin/activate
```
If you doesn't have `virtualenv` you can try simply `venv` instead.

To install everything run (with your activated env):
```bash
pip install -r requirements.txt
```

## Images and segmentation data

Download the BSD500 dataset [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz).

Put the folder `data` in this repository, it shoud look like this:

```
|-- vic-project-sp
|   |-- data
|   |   |-- groundTruth
|   |   |-- images
|   |-- README.md
|   |-- etc
```

## Evaluation

We have defined special functions to evaluate the two algorithms we implemented.

This file does also provide a script to evaluate over the BSD dataset.

Example to run the evaluation on both <b>slic and energetic slic</b> pixeliser <b>on the first 5 files</b>.

To run on the whole directory, don't specify the argument `--max_evaluation`.

```bash
python evaluation.py --pixeliser slic+eslic --img_dir_path "data/images/train/" --gt_dir_path "data/groundTruth/train/" --max_evaluation 5
```
Output example:

```json
{
    "eslic": {
        "boundary_recall": 0.6216054006637307,
        "undersegmentation_error": 0.31310159908290747
    },
    "slic": {
        "boundary_recall": 0.6200771032791588,
        "undersegmentation_error": 0.3116288106942312
    }
}
```
