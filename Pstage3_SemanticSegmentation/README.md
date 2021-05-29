
# ♻ 재활용 품목 분류를 위한 Semantic Segmentation
###### 📌 본 대회는 Naver AI Boostcamp에서 팀 프로젝트로 진행되었습니다. 
<br></br>
## 📋 Table of content
- [최종 결과](#Result)<br>
- [대회 개요](#Overview)<br>
- [문제 정의 해결 및 방법](#Solution)<br>
- [CODE 설명](#Code)<br>

<br></br>
## 🎖 최종 결과 <a name = 'Result'></a>
- 1등 (총 21팀)
- private LB : 0.7043 (mIoU)
- [1등 발표 자료](https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble/blob/master/presentation/Pstage3_solution.pdf)는 여기서 확인하실 수 있습니다. 

<br></br>
## 👀 대회 개요 <a name = 'Overview'></a>
- 대회 명 : 재활용 품목 분류를 위한 Semantic Segmentation
  <details>
  <summary>자세한 대회 설명</summary>
  <div markdown="1">       

  환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 우리나라의 분리 수거율은 굉장히 높은 것으로 알려져 있고, 또 최근 이러한 쓰레기 문제가 주목받으며 더욱 많은 사람이 분리수거에 동참하려 하고 있습니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

  따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.


  </div>
  </details>

- Dataset 설명
  - 512 x 512 크기의 train 2617장 (80%) , public test 417장 (10%) , private test 420장(10%) 
  - 총 11개의 class 존재 
     - Background, UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - coco format으로 images , annotations 정보 존재
    - images : id, height , width, filename
    - annotatins : id, segmentation mask , bbox, area, category_id , image_id
- 평가 Metric : mIoU
<br></br>


<br></br>
## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 [wrapup report](https://www.notion.so/Wrap-up-Pstage3-Semantic-Segmentation-2679c48f500a40f5bf7d7ffb227b8e46)에서 자세하게 기술하고 있습니다. 
- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
- 팀프로젝트를 진행하며 협업 툴로 사용했던 [Notion ](https://www.notion.so/1cdc0eddd3d649b68eebd94e27dc8655?v=b17e11d3c44148bc80dddf4c24b9cabf)내용도 해당 링크에 접속하시면 확인하실 수 있습니다.

<br></br>
## 💻 CODE 설명<a name = 'Code'></a>
### 폴더 구조 
```
├── config                  # 실험 config 코드
|    ├── config.yml             # train   
|    └── eval_config.yml        # infernece 
|
├── src                     # source 코드
|    ├── dataset                
|    ├── losses                 
|    ├── scheduler                             
|    ├── train              # 학습
|    ├── add_train          # pseudo data를 이용해서 train할 때
|    ├── eval               # 추론
|    └── utils              # 그 외 
└── main

```
### 소스 코드 설명 
- `datset.py` : train / val dataset 생성 (object aug를 사용할 지 선택 가능)
- `losses.py` : semantic segmentation loss 모아놓은 코드 , import module을 통해 불러와서 train시 사용
- `scheduler.py` : cosine annealing with warm starts를 사용
- `train.py` : train dataset만을 학습할 시 사용
- `add_train.py` : train dataset과 pseudo dataset을 학습할 때 사용
- `eval.py` : 추론 시 사용
- `utils.py` : 그 외 모든 기능 (ex. Dataloader , CRF , Cutout...)

### 학습 실행 코드 
``` 
python main.py 
  --config_path     # config가 있는 파일 경로
  --config          # 본인이 실험하고 싶은 config 파일 내 이름 
  --run_name        # wandb 사용 시 실험 이름
```
### 추론 실행 코드 
```   
cd src          # src로 이동
python .py 
  --eval_config_file_path     # config가 있는 파일 경로
  --eval_config               # 본인이 실험했던 config 파일 내 이름
  --crf                       # crf 적용 여부 true/false
  --save_name                 # output.csv 저장 이름
```
