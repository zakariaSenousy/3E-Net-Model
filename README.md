# 3E-Net-Model
3E-Net: Entropy-based Elastic Ensemble of Deep Convolutional Neural Networks for Grading of Invasive Breast Carcinoma Histopathological Microscopic Images

Automated grading systems using deep convolution neural networks (DCNNs) have proven their capability and potential to distinguish between different breast cancer grades using digitized histopathological images. In medical domains where safety is paramount, knowing when a DCNN is not secure in its prediction is important. A DCNN can only make grading decision if it is confident in its competence. Furthermore, due to many computer vision challenging problems (e.g., the high visual variability of the images) which affects the grading performance, measuring machine-confidence is required to not only improve the robustness of automated systems but also assist medical professionals in coping with complex cases. We propose Entropy-based Elastic Ensemble of DCNN models (3E-Net) for grading invasive breast carcinoma microscopy images which provides an initial stage of explainability (using an uncertainty-aware mechanism adopting entropy). Our proposed model has been designed in a way to (1) exclude images that are less sensitive and highly uncertain to our ensemble model, and (2) dynamically grade the non-excluded images using the certain models in the ensemble architecture. We evaluated two variations of 3E-Net on an invasive breast carcinoma dataset.



![3E-Net](https://user-images.githubusercontent.com/20457990/117557820-f3511300-b06e-11eb-82c0-ce7e0da87b85.png)
