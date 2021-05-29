
# 📑 문장 내 개체간 관계 추출

###### 📌 본 프로젝트는 Naver AI Boostcamp에서 개인 프로젝트로 진행됐습니다.
## 📋 Table of content

- [최종 결과](#Result)<br>
- [대회 개요](#Overview)<br>
- [문제 정의 해결 및 방법](#Solution)<br>
- [CODE 설명](#Code)<br>

<br></br>
## 🎖 최종 결과 <a name = 'Result'></a>
- 23등 (총 135명)
- private LB & public LB : `80.50%` (ACC)

<br></br>
## 👀 대회 개요 <a name = 'Overview'></a>
- 대회 명 : 문장 내 개체간 관계 추출
  <details>
  <summary>자세한 대회 설명</summary>
  <div markdown="1">
    관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.
    이번 대회에서는 문장, 엔티티, 관계에 대한 정보를 통해 ,문장과 엔티티 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 엔티티들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 model이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.
    
    #
    `sentence`: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.

  `entity 1` : 썬 마이크로시스템즈
    
  `entity 2` : 오라클
    #
  
    `relation` : 단체:별칭
    #
  </div>
  </details>
- 문장 내 2개의 entity 사이의 관계를 42개의 label 중 하나로 분류하는 task (relation extraction)
- Dataset 설명
  - 9000개의 train data ( sentence , entity , label ) , 1000개의 test data
  - 총 42개의 label 존재
  
  ``` python
  
  {'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, 
  '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9, '단체:구성원': 10, 
  '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, 
  '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19, '단체:본사_도시': 20, 
  '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, 
  '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29, '인물:거주_주(도)': 30, 
  '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, 
  '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39, '인물:사망_국가': 40, '인물:나이': 41} 
  ```
- 평가 Metric : Accuracy

<br></br>
## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 [wrapup report](https://www.notion.so/Wrap-up-Report-545c4800791745ccb5cee79cbd0c8542)에서 자세하게 기술하고 있습니다. 
- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 

<br></br>
## 💻 CODE 설명 <a name = 'Code'></a>
### 폴더 구조 


```
├── load_data.py       
├── train_trainer.py                
├── train_kfold.py            
├── inference.py
├── inference.py            
└── EDA.ipynb        
```


### 소스 설명 
- `load_data.py` : data를 불러와 dataset으로 만들어주는 파일, special token 사용 여부, siglne/multi sequence 선택 가능
- `train_trainer.py` : huggingface의 trainer를 이용하여 학습시키는 파일 
- `train_kfold.py` : kfold 시 사용하는 train 파일
- `train.py` : train dataset만을 학습할 시 사용
- `inference.py` : 저장된 모델을 불러와 추론 후 submission.csv를 만드는 파일
- `utils.py` : 그 외 모든 기능
- `EDA.ipynb` : Tokenizer에 따른 unk 토큰 분포 비교 , max_len 선정을 위한 문장의 길이 비교 , label의 분포 등 EDA를 위한 파일

### Train 코드 

``` 
python train.py
```

### Inference 코드 

``` 
python inference.py --model_dir=./results/{checkpoint 파일명}
```
