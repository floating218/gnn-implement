# gnn-implement

본 레파지토리는 https://keras.io/examples/graph/gnn_citations/#examine-the-gnn-model-predictions에 있는 graph neural network 모델을 구현한 것이다. 

## 파일 설명 

* Loader.py : 데이터 로드
* Model.py  : GNN 모델
* Run.py    : 실행하는 프로그램

## 데이터

* usersha1-artmbid-artname-plays.tsv : lastfm의 음악 스트리밍 데이터  
* 출처: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html


## 코드 실행 예시
```
python Run.py --epochs 100 --batch_size 256 --lr 0.01 --dropout_rate 0.5
```
```
python Run.py --epochs 10 --batch_size 32
```

## update

* Last Update Date: 2022/03/
  