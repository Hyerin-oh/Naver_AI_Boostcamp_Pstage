
### 📑 문장 내 개체간 관계 추출

###### 📌 본 프로젝트는 Naver AI Boostcamp에서 개인 프로젝트로 진행됐습니다.

----
### 🍀  최종 결과 
- 23등 (총 135명)
- private LB & public LB : 80.50% (ACC)

---
### 📝 문제 정의 및 해결 방법
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 [wrapup report](https://www.notion.so/Wrap-up-Report-545c4800791745ccb5cee79cbd0c8542)에서 자세하게 기술하고 있습니다. 
- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 

---
### 💻 CODE 설명
####   - 폴더 구조 


```
├── load_data.py       
├── train_trainer.py                
├── train_kfold.py            
├── inference.py
├── inference.py            
└── EDA.ipynb        

```


####   - 소스 설명 
- `load_data.py` : data를 불러와 dataset으로 만들어주는 파일, special token 사용 여부, siglne/multi sequence 선택 가능
- `train_trainer.py` : huggingface의 trainer를 이용하여 학습시키는 파일 
- `train_kfold.py` : kfold 시 사용하는 train 파일
- `train.py` : train dataset만을 학습할 시 사용
- `inference.py` : 저장된 모델을 불러와 추론 후 submission.csv를 만드는 파일
- `utils.py` : 그 외 모든 기능
- EDA.ipynb : Tokenizer에 따른 unk 토큰 분포 비교 , max_len 선정을 위한 문장의 길이 비교 , label의 분포 등 EDA를 위한 파일

####   - Train 코드 

``` 
python train.py
```

####   - Inference 코드 

``` 
python inference.py --model_dir=./results/{checkpoint 파일명}
```