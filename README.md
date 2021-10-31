# 3E-Net-Model
3E-Net: Entropy-based Elastic Ensemble of Deep Convolutional Neural Networks for Grading of Invasive Breast Carcinoma Histopathological Microscopic Images

Automated grading systems using deep convolution neural networks (DCNNs) have proven their capability and potential to distinguish between different breast cancer grades using digitized histopathological images. In medical domains where safety is paramount, knowing when a DCNN is not secure in its prediction is important. A DCNN can only make grading decision if it is confident in its competence. Furthermore, due to many computer vision challenging problems (e.g., the high visual variability of the images) which affects the grading performance, measuring machine-confidence is required to not only improve the robustness of automated systems but also assist medical professionals in coping with complex cases. We propose Entropy-based Elastic Ensemble of DCNN models (3E-Net) for grading invasive breast carcinoma microscopy images which provides an initial stage of explainability (using an uncertainty-aware mechanism adopting entropy). Our proposed model has been designed in a way to (1) exclude images that are less sensitive and highly uncertain to our ensemble model, and (2) dynamically grade the non-excluded images using the certain models in the ensemble architecture. We evaluated two variations of 3E-Net on an invasive breast carcinoma dataset.



![3E-Net](https://user-images.githubusercontent.com/20457990/117557820-f3511300-b06e-11eb-82c0-ce7e0da87b85.png)

## Citation
If you use this code for your research, please cite our paper: [3E-Net: Entropy-Based Elastic Ensemble of Deep Convolutional Neural Networks for Grading of Invasive Breast Carcinoma Histopathological Microscopic Images](https://www.mdpi.com/1099-4300/23/5/620)



```
@Article{Senousy3ENet,
AUTHOR = {Senousy, Zakaria and Abdelsamea, Mohammed M. and Mohamed, Mona Mostafa and Gaber, Mohamed Medhat},
TITLE = {3E-Net: Entropy-Based Elastic Ensemble of Deep Convolutional Neural Networks for Grading of Invasive Breast Carcinoma Histopathological Microscopic Images},
JOURNAL = {Entropy},
VOLUME = {23},
YEAR = {2021},
NUMBER = {5},
ARTICLE-NUMBER = {620},
URL = {https://www.mdpi.com/1099-4300/23/5/620},
ISSN = {1099-4300},
ABSTRACT = {Automated grading systems using deep convolution neural networks (DCNNs) have proven their capability and potential to distinguish between different breast cancer grades using digitized histopathological images. In digital breast pathology, it is vital to measure how confident a DCNN is in grading using a machine-confidence metric, especially with the presence of major computer vision challenging problems such as the high visual variability of the images. Such a quantitative metric can be employed not only to improve the robustness of automated systems, but also to assist medical professionals in identifying complex cases. In this paper, we propose Entropy-based Elastic Ensemble of DCNN models (3E-Net) for grading invasive breast carcinoma microscopy images which provides an initial stage of explainability (using an uncertainty-aware mechanism adopting entropy). Our proposed model has been designed in a way to (1) exclude images that are less sensitive and highly uncertain to our ensemble model and (2) dynamically grade the non-excluded images using the certain models in the ensemble architecture. We evaluated two variations of 3E-Net on an invasive breast carcinoma dataset and we achieved grading accuracy of 96.15% and 99.50%.},
DOI = {10.3390/e23050620}
}
```
