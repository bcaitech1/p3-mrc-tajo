## Readme

## 소개
P stage 3 대회 코드(베이스라인 코드 기반)

Hugging face 에서 제공하는 colab tutorial 파일과 매우 유사하며, 이곳에서 자세한 설명 확인 가능합니다.
https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb

## 파일 구성
retrieval.py             # retreiver 모듈 제공 
arguments.py             # 실행되는 모든 argument 가 dataclass 의 형태로 저장되어있음
trainer_qa.py            # MRC 모델 학습에 필요한 trainer 제공.
utils_qa.py              # 기타 유틸 함수 제공 
train.py                 # MRC 학습 및 평가 
inference.py		     # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성

### train and eval
```
python train.py 
```
train.py 를 실행하면 MRC 학습이 진행됩니다.   
arguments 에 대한 세팅 정보는 arguments.py 와 train.py의 TrainingArguments에서 확인가능합니다. 

### inference - odqa 실행 (test_dataset 사용)
```
python inference.py --output_dir [아웃풋경로] --dataset_name [테스트데이터셋경로] --model_name_or_path [학습체크포인트경로] --do_predict
```
inference.py 를 실행하면 BM 25 Retreiver를 기본으로 ODQA 가 실행됩니다.
Elasticsearch 를 사용하려면 아래 파일을 참고하셔서 세팅하신뒤, inference.py와 retreiver.py의 retrieve_ES, ES 주석처리를 해제해주세요.
https://github.com/bcaitech1/p3-mrc-tajo/blob/master/%EA%B9%80%EB%82%A8%ED%98%81/Elasticsearch.ipynb

## Things to know
1. inference.py에서 BM25 embedding 이 실행됩니다. 실행 후 "BM25_embedding.bin" 과 "BM25_score.bin", "BM25_indice.bin" 이 저장이 됩니다.
    **만약 BM25 retrieval 관련 코드를 수정한다면, 꼭 세 파일을 지우고 다시 실행해주세요!**
    안그러면 존재하는 파일이 load 됩니다.
2. 모델의 경우 --overwrite_cache 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 
3. output_dir 폴더 또한 --overwrite_output_dir 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.
