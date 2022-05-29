# openie6-revised
This is an revised implementation of OpenIE6: Iterative Grid Labeling and Coordination Analysis for Open Information Extraction. (EMNLP 2020)

[Original implementation](https://github.com/dair-iitd/openie6), which our implementation is based on, is made by the authors of the paper.

## Changes
Original implementation is based on various libraries including pytorch lightning, and torchtext.

We removed its dependency of pytorch-lighthing and torchtext so that the revision can run on native pytorch and huggingface.

## Datasets and Metric
We use OpenIE4 dataset just like the original implementation.

You can download OpenIE4 dataset following instructions from the [original repo](https://github.com/dair-iitd/openie6).

Carb metric scripts can also be downloaded from the [original repo of Carb](https://github.com/dair-iitd/CaRB).


## Usage
Make sure to install all the datasets and metric scripts. 

Then you can use below command to execute the demo.

```
python main.py
```
