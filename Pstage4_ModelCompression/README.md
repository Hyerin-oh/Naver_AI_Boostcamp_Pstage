# 부스트캠프 AI TECH <Pstage 4 경량화>

## 📋 Table of content

- [팀 소개](#Team)<br>
- [최종 결과](#Result)<br>
- [대회 개요](#Overview)<br>
- [문제 정의 해결 및 방법](#Solution)<br>
- [CODE 설명](#Code)<br>



## 🙋‍♀️ 팀 소개 🙋‍♂️ <a name = 'Team'></a>

- 모델 최적화 6조 (영기일기이기삼기사기 조)
- 조원 : 박성훈, 엄희준, 오혜린, 이보현, 장보윤
- 프로젝트 일정  21.05.24 ~ 21.06.15

|                                                                                      박성훈                                                                                      |                                                             엄희준                                                             |                                                          오혜린                                                           |                                                            이보현                                                            |                                                            장보윤                                                             |                                                            
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | 
| <a href='https://github.com/seong0905'><img src='https://avatars.githubusercontent.com/u/70629496?v=4' width='200px'/></a> | <a href='https://github.com/eomheejun'><img src='https://avatars.githubusercontent.com/u/50470448?v=4' width='200px'/></a> | <a href='https://github.com/Hyerin-oh'><img src='https://avatars.githubusercontent.com/u/68813518?s=400&u=e5300247dc2b04f5cf57265a6f2e1cc0987e6d08&v=4' width='200px'/></a> | <a href='https://github.com/bonniehyeon'><img src='https://avatars.githubusercontent.com/u/50580028?v=4' width='200px'/></a> | <a href='https://github.com/dataminegames'><img src='https://avatars.githubusercontent.com/u/45453533?v=4' width='200px'/></a> | 





## 🎖 최종 결과 <a name = 'Result'></a>
  - Private LB
    - F1 : 0.4892 / MACS : 1862490 / Score : 1.1343 (9등 / 10팀) 
  - Public LB
    - F1 : 0.5176 / MACS : 1862490 / Score : 0.5156 (3등 / 10팀)

## ♻ 대회 개요 <a name = 'Overview'></a>
최근들어 분야를 막론하고 인공지능 기술은 사람을 뛰어넘은 엄청난 성능을 보여주고 있고, 때문에 여러 산업에서 인공지능을 이용해 그동안 해결하지 못 했던 문제들을 풀려는 노력을 하고 있습니다.

대표적인 예로는 수퍼빈의 수퍼큐브가 있습니다. 수퍼큐브는 수퍼빈에서 만든 인공지능 분리수거 기계로 사람이 기계에 캔과 페트병을 넣으면 내부에서 인공지능을 통해 재활용이 가능한 쓰레기인지를 판단해 보관해주는 방식입니다. 간단한 인공지능을 이용해 그동안 힘들었던 분리수거 문제를 해결한 것입니다. 그렇다면 수퍼큐브를 만들기 위해 필요한 인공지능은 무엇일까요? 당연히 들어온 쓰레기를 분류하는 작업일 것입니다. 하지만 분류만 잘 한다고 해서 사용할 수 있는 것은 아닙니다. 로봇 내부 시스템에 탑재되어 즉각적으로 쓰레기를 분류할 수 있어야만 실제로 사용이 될 수 있습니다.

이번 프로젝트를 통해서는 분리수거 로봇에 가장 기초 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 작고 계산량이 적은 모델을 만들어볼 예정입니다.
# 
- Dataset 설명
  - train 32,599장 (80%) , public test 4,076장 (10%) , private test 4,083장(10%) 
  - 총 9개의 class 존재 
     - Battery, Clothing, Glass, Metal, Paper, Paperpack, Plastic, Plasticbag, Styrofoam
    
- 평가방법
  - f1-score 계산에 성능적 하한을 두어서 채점된 f1-score가 0.5 미만일 경우, f1 항의 score는 1.0으로 고정됩니다.
<p align="center"><img src="https://user-images.githubusercontent.com/70629496/122766550-a3f03a80-d2dc-11eb-9635-90f294db0daa.png"></p>
    

## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 각자의 wrap up report에서 기술하고 있습니다. 
    - [박성훈 wrapup report](https://www.notion.so/Wrap-Up-Report-Stage-4-5af1d1d785ed451881d896005f874dda)
    - [엄희준 wrapup report](https://github.com/eomheejun/boostcamp/blob/main/Pstage/stage4/Code/Wrap%20up%20Report.md)
    - [오혜린 wrapup report](https://hyerin-oh.oopy.io/7a3ef91a-e7f2-41fe-9bef-0c8775672791)
    - [이보현 wrapup report](https://www.notion.so/P4-Wrap-up-Report-840af74460e74caabc975bc1b9f225a0)
    - [장보윤 wrapup report](https://www.notion.so/Stage-4-Model-Optimization-ac04bba349cb4c999ba36842b61e217d)

- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
- 팀프로젝트를 진행하며 협업 툴로 사용했던 [Notion](https://www.notion.so/6-0ab9c18a20a448888053ef9c4642434c)내용도 해당 링크에 접속하시면 확인하실 수 있습니다.

## 💻 코드 설명 <a name = 'Code'></a>

### 1. Train single file
```
python train.py --model "model config file path" --data "data config file path"
```
### 2. AutoML for Architecture Searching(NAS)
```
python tune_architecture.py
```
### 3. AutoML for Hyper Parameter Searching
```
python tune_hyper.py
```
### 4. Decompose Architecture
```
python decompose.py 
```
### 5. Inference(submission.csv)
```
python inference.py --model_config "model config file path" --weight "weight file path" --img_root "/opt/ml/data/test" --data_config "data config file path"
```
### 6. Finetune & Inference
```
python finetune_inference.py --model "model_weight" --data "data_weight" --weight "weight_path" --save_name "save path" --rank_cfg "rank_config.pkl path" --freeze [0 or 1] --img_root "/opt/ml/data/test" --dst "save csv path"
```
