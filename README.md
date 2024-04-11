# Fitzpatrick17k

------------

## Overview
[![fitzpatrick17k](https://pbs.twimg.com/media/E4QqXfQUYAEfNco?format=jpg&name=large)](https://youtu.be/bizJpy5VQmQ)

We annotated 16,577 clinical images sourced from two dermatology atlases — DermaAmin and Atlas Dermatologico — with Fitzpatrick skin type labels with two data annotation services: Scale AI and Centaur Labs. The Fitzpatrick labeling system, while not perfect, is a six-point scale originally developed for classifying sun reactivity of skin phenotype. The Fitzpatrick scale served as the basis for skin color in emojis and, more recently, the Fitzpatrick scale has been used in computer vision applications to evaluate algorithmic fairness and model accuracy. The annotated images represent 114 skin conditions with at least 53 images and a maximum of 653 images per skin condition. 

![fitzpatrick](https://www.datocms-assets.com/45562/1623693822-blogmitskintypes01.png)


We've included the original image sources, the training script `train.py`, and a notebook to compare Fitzpatrick annotations with individual typology angle scores. You can read the dataset and our analysis in our [paper](https://arxiv.org/abs/2104.09957).

We thank Scale AI and Centaur Labs for providing Fitzpatrick Skin Type annotations for all images in this dataset for free.

## Updates July 7 2022

Given the subjectivity of annotating Fitzpatrick Skin Type in images showing skin disease, we evaluated how well experts, crowds, and an algorithm compare with respect to inter-rater reliability. You can find replication files for this analysis in the `annotation_evaluation` folder.

## Data Usage

------------

### Download the dataset

You can find the Fitzpatrick annotations in [fitzpatrick17k.csv](https://github.com/mattgroh/fitzpatrick17k/blob/main/fitzpatrick17k.csv). You can download the images from their original source, which is shared in the `url` column of the Fitzpatrick annotations .csv. Alternatively, fill out this [form](https://forms.gle/4fS35Kg8x9pkG2Bn9) and contact us and we can provide a link to all the images. 

### Replicate our analysis

The results from our paper can be replicated using `train.py` and `ita_fitzpatrick_analysis.ipynb`.

After you download the dataset, edit `train.py` by specifying the image directory of the dataset, and then run ```python train.py 20 full``` where 20 refers to the number of epochs and full refers to the full dataset.

You can check out our comparison of Fitzpatrick annotations and individual typology angle scores with the `ita_fitzpatrick_analysis.ipynb`

### How to cite this dataset and paper
```
@inproceedings{groh2021evaluating,
  title={Evaluating deep neural networks trained on clinical images in dermatology with the fitzpatrick 17k dataset},
  author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1820--1828},
  year={2021}
}

And the second paper:

@article{groh2022towards,
  title={Towards transparency in dermatology image datasets with skin tone annotations by experts, crowds, and an algorithm},
  author={Groh, Matthew and Harris, Caleb and Daneshjou, Roxana and Badri, Omar and Koochek, Arash},
  journal={Proceedings of the ACM on Human-Computer Interaction},
  volume={6},
  number={CSCW2},
  pages={1--26},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

------------
## Licensing

Original images collected from [Atlas Dermatologico](http://atlasdermatologico.com.br/) and [DermaAmin](https://www.dermaamin.com/site/)

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.
