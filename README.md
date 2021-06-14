# Fitzpatrick17k

We've included the original image sources, the training script `train.py`, and a notebook to compare Fitzpatrick annotations with individual typology angle scores. You can read the dataset and our analysis in our [paper](https://arxiv.org/abs/2104.09957).

We thank Scale AI for providing Fitzpatrick annotations for all images in this dataset pro bono.

# Download the dataset

You can find the Fitzpatrick annotations in [fitzpatrick17k.csv](https://github.com/mattgroh/fitzpatrick17k/blob/main/fitzpatrick17k.csv). You can download the images from their original source, which is shared in the `url` column of the Fitzpatrick annotations .csv. Alternatively, you can email us, and we can provide a link.

# Replicate our analysis

The results from our paper can be replicated using `train.py` and `ita_fitzpatrick_analysis.ipynb`.

After you download the dataset, edit `train.py` by specifying the image directory of the dataset, and then run ```python train.py 20 full``` where 20 refers to the number of epochs and full refers to the full dataset.

You can check out our comparison of Fitzpatrick annotations and individual typology angle scores with the `ita_fitzpatrick_analysis.ipynb`

# How to cite this dataset and paper
```
@article{groh2021evaluating,
  title={Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset},
  author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  journal={arXiv preprint arXiv:2104.09957},
  year={2021}
}
```
