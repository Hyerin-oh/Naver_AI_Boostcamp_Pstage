# Pstage 1 ] Image Classification

📌 본 대회는 Naver AI Boostcamp에서 개인 프로젝트로 진행되었습니다. 

## 📋 Table of content

- [최종 결과](#Result)<br>
- [대회 개요](#Overview)<br>
- [문제 정의 해결 및 방법](#Solution)<br>
- [CODE 설명](#Code)<br>


<br></br>
## 🎖 최종 결과 <a name = 'Result'></a>
- private LB (116등)
    - F1 score : `0.7064` 
 


<br></br>
## 👁 대회 개요 <a name = 'Overview'></a>
- 대회 명 : 마스크 착용 상태 분류
  <details>
  <summary>자세한 대회 설명</summary>
  <div markdown="1">       

  COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

  감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

  따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.


  </div>
  </details>

- Dataset 설명
  - 384 x 512 크기의 train 18900장 (60%) , public test 6300장 (20%) , private test 6300장 (20%)
  - 총 18개의 class 존재 (3개의 연령대 , 3개의 마스크 착용 여부 , 2개의 성별)
- 평가 Metric : F1 score
<br></br>
![image](https://user-images.githubusercontent.com/68813518/120065024-39b3f380-c0aa-11eb-8367-44ebfb74e245.png)


<br></br>
## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>
- [wrap up report](https://vimhjk.oopy.io/d5f7f6d2-0a5c-442a-bfcf-4694c88b5c5d)서 문제 정의, 해결, 솔루션을 확인하실 수 있습니다. 
- 위 report에는 대회를 참가한 후, 대회에 대한 본인의 회고도 포함되어있습니다. 

<br></br>
## 💻 CODE 설명<a name = 'Code'></a>
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              
### 폴더 구조 

```
├── ipynb                  # 1주차 미션 & baseline code
|    ├── baseline.ipynb
|    ├── EDA_0329.ipynb
|    ├── Augmentation_0330.ipynb
|    ├── Mission_0331.ipynb
|    ├── Mission_0401.ipynb
|    └── Mission_0402.yml        
|
├── py                     # 2주차 baseline code
|    ├── dataset                
|    ├── loss                 
|    ├── model                             
|    ├── train             
|    ├── Multitrain          
|    ├── inference               
|    └── utils             
| 
└── submit_record.tsv     # 최종 제출 tsv
```

###    소스 설명 
- ipynb는 1주차에 받은 baseline code를 기반으로 EDA, Mission , simple Augmentation등을 연습한 파일입니다. 
- `datset.py` : Augmentation 선택 ,  이미지와 Model에 따라 다른 Label 반환
- `loss.py` : 4가지의 loss (CrossEntropy , Focal , Labelsmoothing , F1)
- `model.py` : 서로 다른 backbone , structure의 model 선택 (Multi Model , Single Model 등)
- `train.py` : train & validation 진행
- `Multitrain.py` : sampler, cutmix, mixup 등을 이용해 train & validation 진행
- `inference.py` : 추론 시 사용
- `utils.py` : mixup, cutmix 등 필요한 utils 함수들

###  학습 실행 코드
``` 
python train.py 
  --dataset       # model의 output에 따라 dataset 선택
  --k             # kfold
  --age_filter    # age_filter 적용 여부 
  --augmentation  # augmentation 선택
  --model         # model 선택
  --name          # pth가 저장될 파일명
  --data_dir      # data path
  --model_dir     # model_path
```

###   실행 코드
``` 
python inference.py 
  --model         # train시 사용했던 model 명
  --data_dir      # data path
  --model_dir     # pth가 저장되어있는 path
  --output_dir    # output.csv가 저장될 path
```
