# Multiomics Research 

> Pathology image, radiomics, spatial transcriptomics(ST)를 통합하기 위한 연구 워크스페이스입니다. 
> Pathology image로부터 ST를 예측하는 과제에 radiomics를 결합하는 두 가지 프레임워크, **RaPaCL**과 **STaRN**을 진행했습니다.

## [RaPaCL](./RaPaCL) 

> 해석가능성을 위한 Radiomics-Pathomics Contrastive Learning
> RaPaCL investigates how interpretable radiomics representations can be aligned with deep pathomics features for explainable spatial transcriptomics prediction.

Pathology image 기반 ST 예측 프레임워크에 해석가능한 radiomics 표현을 통합하여 설명가능성을 부여한 연구입니다. Cell-segmented patch에서 추출한 handcrafted radiomics feature와 딥러닝 기반 pathomics feature를 contrastive learning으로 동일 latent space에 정렬시켜, image encoder가 texture·형태 등 해석 가능한 조직학적 특성을 포착하도록 학습합니다. 정렬된 두 표현을 융합해 spot-wise gene expression을 예측합니다.

## [STaRN](./STaRN) 

> 이웃 정보를 반영하는 경량 Radiomics 표현 학습
> STaRN explores lightweight yet context-aware radiomics representation learning by incorporating spatial and semantic neighborhood information within whole-slide images.

Radiomics feature의 경량성에 주목하여, pathology image로부터 ST를 예측하는 과제에 WSI 내 spot 이웃(spatial・semantic neighbor) 정보를 효과적으로 반영하는 프레임워크입니다. Anchor spot과 이웃 spot들을 하나의 Summary Table로 구성하고, column attention(spot 내 feature 상호작용)과 row attention(spot 간 이웃 맥락)으로 구성된 SAINT 스타일 dual-attention 인코더로 인코딩합니다. UNI·scFoundation 기반 teacher 표현으로의 distillation과 self-contrastive learning을 통해 이웃-인지적이면서 가벼운 radiomics 표현을 학습하고, 이를 gene expression 예측에 활용합니다.

