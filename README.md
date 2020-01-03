# [ICLR 2020] Order Learning and Its Application to Age Estimation

## Paper
[**Order Learning and Its Application to Age Estimation**](https://openreview.net/pdf?id=HygsuaNFwr)

We propose order learning to determine the order graph of classes, representing ranks or priorities, and classify an object instance into one of the classes. To this end, we design a pairwise comparator to categorize the relationship between two instances into one of three cases: one instance is 'greater than,' 'similar to,' or 'smaller than' the other. Then, by comparing an input instance with reference instances and maximizing the consistency among the comparison results, the class of the input can be estimated reliably. We apply order learning to develop a facial age estimator, which provides the state-of-the-art performance. Moreover, the performance is further improved when the order graph is divided into disjoint chains using gender and ethnic group information or even in an unsupervised manner.

If our code or results are helpful for your research, please cite our paper:
```
@inproceedings{
Lim2020Order,
title={Order Learning and Its Application to Age Estimation},
author={Kyungsun Lim and Nyeong-Ho Shin and Young-Yoon Lee and Chang-Su Kim},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HygsuaNFwr}
}
```

## Code
Coming Soon

## Dataset
We form a ‘balanced dataset’ from MORPH II[1], AFAD[2], and UTK[3]. Before sampling images from MORPH II, AFAD, and UTK, we rectify inconsistent labels by following the strategy in [4]. For each combination of gender in {female, male} and ethnic group in {African, Asian, European}, we sample about 6,000 images. Also, during the sampling, we attempt to make the age distribution as uniform as possible within range [15,80]. The balanced dataset is partitioned into training and test subsets with ratio 8 : 2. Table 1 shows how the balanced dataset is organized.

|<center></center>|<center>MORPH II</center>|<center>AFAD</center>|<center>UTK</center>|<center>Balanced</center>|
|:--------|:--------:|--------:|

## Results





## Reference


