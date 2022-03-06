# gnn-implement

본 레파지토리는 https://keras.io/examples/graph/gnn_citations/#examine-the-gnn-model-predictions에 있는 graph neural network 모델을 구현한 것이다. 

## 파일 설명 

* Loader.py : 데이터 로드
* Model.py  : GNN 모델
* Run.py    : 실행하는 프로그램

## 데이터

* 논문의 주제(subject), 단어feature, 인용관계
* 출처: "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
```
citations: [target논문인덱스, source논문인덱스]
papers: [논문인덱스, 1424개 단어 포함 여부, 주제(subject)]
train_data: papers 데이터 중 50% 샘플링
test_data: papers 데이터 중 50% 샘플링
x_train: train_data 중, 논문인덱스와 subject를 제외한 피쳐
y_train: train_data 중 subject에 해당하는 레이블
```


## 코드 실행 예시
```
python Run.py --epochs 100 --batch_size 256 --lr 0.01 --dropout_rate 0.5
```
```
python Run.py --epochs 10 --batch_size 32
```

## update

* Last Update Date: 2022/03/
  