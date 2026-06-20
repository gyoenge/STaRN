# Multiomics Research 

> Pathology image, radiomics, spatial transcriptomics(ST)를 통합하기 위한 연구 워크스페이스입니다. <br/>
> Pathology image로부터 ST를 예측하는 과제에 radiomics를 결합하는 두 가지 프레임워크를 소개합니다. 

## [RaPaCL](./RaPaCL) 

> Radiomics-Pathomics Contrastive Learning for Interpretable ST Prediction

<p align="justify">
RaPaCL은 병리 이미지 기반 ST 예측 모델에 해석 가능한 radiomics 표현을 통합하기 위한 프레임워크입니다.
Cell-segmented patch에서 추출한 handcrafted radiomics feature와 딥러닝 기반 pathomics feature를 contrastive learning으로 동일한 latent space에 정렬하여, 모델이 조직의 texture·shape·morphology와 같은 해석 가능한 특성을 학습하도록 유도합니다.
정렬된 radiomics-pathomics 표현을 융합하여 spot-wise gene expression을 예측하며, 예측 성능과 해석 가능성을 동시에 향상시키는 것을 목표로 합니다.
</p>

## [STaRN](./STaRN) 

> Spatially-aware Radiomics Network for ST Prediction

<p align="justify">
STaRN은 경량 radiomics 표현을 활용하여 공간적 이웃 정보(spatial context) 를 효과적으로 학습하는 프레임워크입니다.
각 spot과 주변 이웃 spot들을 Summary Table 형태로 구성한 뒤, SAINT 기반 dual-attention encoder를 사용하여 spot 내 feature 상호작용(column attention)과 이웃 간 관계(row attention)를 동시에 모델링합니다.
또한 UNI 및 scFoundation 기반 teacher representation으로부터 distillation을 수행하고, self-contrastive learning을 결합하여 공간적·의미적 맥락을 반영하는 radiomics 표현을 학습합니다. 학습된 표현은 downstream gene expression prediction에 활용됩니다.
</p>

