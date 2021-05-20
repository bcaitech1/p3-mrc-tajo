# 부스트캠프 AI Tech - p3-mrc-tajo

<p align="center"><img src="https://user-images.githubusercontent.com/55614265/116681368-db7eec80-a9e7-11eb-929e-799e78d6bc15.jpg" height="630px" width="495px"/></p>

# Public Leaderboard

<img width="985" alt="스크린샷 2021-05-20 21 25 26" src="https://user-images.githubusercontent.com/55614265/118978280-1904e300-b9b2-11eb-9a58-282175d51e8c.png">

# Private Leaderboard

<img width="985" alt="스크린샷 2021-05-20 23 29 22" src="https://user-images.githubusercontent.com/55614265/119060306-e50adb80-ba0c-11eb-8f4c-e2921308f133.png">

# 팀원

| [김남혁_T1014](https://github.com/skaurl) | [서일_T1093](https://github.com/Usurper47) | [엄희준_T1122](https://github.com/eomheejun) | [우종빈_T1129](https://github.com/JongbinWoo) | [이보현_T1148](https://github.com/bonniehyeon) | [장보윤_T1178](https://github.com/dataminegames) |
| :----------: |  :--------:  |  :---------: |  :---------: | :---------: | :---------: |
| ![55614265](https://user-images.githubusercontent.com/55614265/116680602-d2d9e680-a9e6-11eb-9207-6b8ff06757f7.jpeg) | ![46472729](https://user-images.githubusercontent.com/55614265/116680614-d53c4080-a9e6-11eb-9f38-38e8896c8b9f.jpeg) | ![50470448](https://user-images.githubusercontent.com/55614265/116680649-e2592f80-a9e6-11eb-8f9e-631c15313c5d.png) | ![44800643](https://user-images.githubusercontent.com/55614265/116680664-e6854d00-a9e6-11eb-89b8-1c822f8ec5b9.jpeg) | ![50580028](https://user-images.githubusercontent.com/55614265/116680672-ea18d400-a9e6-11eb-8e7f-d6af940cd263.jpeg) | ![45453533](https://user-images.githubusercontent.com/55614265/116680729-fb61e080-a9e6-11eb-83ee-20539665565f.png) |

# Wiki

[Daily Report](https://github.com/bcaitech1/p3-mrc-tajo/wiki)를 작성하고 있습니다.

# P3-Machine-Reading-Comprehension

# 전체 개요 설명

"한국에서 가장 오래된 나무는 무엇일까?" 이런 궁금한 질문이 있을 때 검색엔진에 가서 물어보신 적이 있을텐데요, 요즘엔 특히나 놀랍도록 정확한 답변을 주기도 합니다. 어떻게 가능한 걸까요? 질의 응답(Question Answering)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 그 중에서도 Open-Domain Question Answering 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가가 되어야하기에 더 어려운 문제입니다.

![fc81eeb5-3cc0-44b6-a5dc-aaf38e4166a5](https://user-images.githubusercontent.com/55614265/116175274-f2ed7980-a74a-11eb-83c8-bba40f25b77e.png)

본 대회에서 우리가 만들 모델은 두 stage로 구성되어 있습니다. 첫 번째 단계는 질문에 관련된 문서를 찾아주는 "retriever"단계이고요, 다음으로는 관련된 문서를 읽고 간결한 답변을 내보내 주는 "reader" 단계입니다. 이 두 단계를 각각 만든 뒤 둘을 이으면, 어려운 질문을 던져도 척척 답변을 해주는 질의응답 시스템을 여러분 손으로 직접 만들게 됩니다. 더 정확한 답변을 내주는 모델을 만드는 팀이 우승을 하게 됩니다.

![cf0936da-a81a-4ea0-897d-4441d12c402a](https://user-images.githubusercontent.com/55614265/116324692-88484680-a7fb-11eb-9de7-8ac98ef69622.png)

# 평가 방법

평가방법은 두가지입니다.

1. Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어집니다. 즉 각 질문마다 0점 아니면 1점입니다. 다만 띄어쓰기나 "."과 같은 문자가 포함돼 있다고 오답으로 처리되면 억울하겠죠? 이런 것은 지우고 일치하는지 확인합니다. 또한 답이 하나가 아닐 수 있는데, 이런 경우는 하나라도 일치하면 정답으로 간주합니다.

![1](https://user-images.githubusercontent.com/55614265/116324960-0efd2380-a7fc-11eb-8ba8-32d4de379ee9.png)


2. F1 Score: EM과 다르게 부분 점수를 제공합니다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 줍니다.

![2](https://user-images.githubusercontent.com/55614265/116324976-14f30480-a7fc-11eb-8388-a822ae07c15a.png)

EM 기준으로 랭킹을 산정하고, F1은 참고용으로만 활용합니다.

# 학습 데이터 개요

다음은 제공하는 데이터셋의 구성을 보여줍니다.

![8acf9df5-ea23-4ad2-a24c-c9a847d24e59](https://user-images.githubusercontent.com/55614265/116324051-52569280-a7fa-11eb-991a-9de3531df2be.png)

MRC 데이터의 경우, Hugging Face에서 제공하는 datasets library로 접근이 가능합니다. 해당 폴더의 directory를 dataset_name 으로 저장한 후, 아래와 같이 불러오기가 가능합니다.

```
# train_dataset을 불러오고 싶은 경우
from datasets import load_from_disk
dataset = load_from_disk("./data/train_dataset/")
print(dataset)
```

Retrieval 과정에서 사용하는 문서 집합(corpus)은 ./data/wikipedia_documents.json 으로 저장되어있습니다. 약 5만 7천개의 unique 한 문서로 이루어져 있습니다. 평가 데이터는 학습데이터와 대부분 동일하나, 리더보드용인 test_dataset 의 데이터에는 id 와 question 만 주어집니다. 즉 Open Domain QA 전용이며, 답을 내기 위해서 ./data/wikipedia_documents.json 을 활용합니다. 데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```
./data/         # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성
    ./dummy_dataset/           # 모델이 작동하는지 테스트 할때 사용하는 dummy 데이터셋. train 과 validation 으로 구성
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

- id: 질문의 고유 id
- question: 질문
- answers: 답변에 대한 정보. 하나의 질문에 하나의 답변만 존재함
- answer_start : 답변의 시작 위치
- text: 답변의 텍스트
- context: 답변이 포함된 문서
- title: 문서의 제목
- document_id: 문서의 고유 id

# 평가 데이터 개요

평가 데이터는 학습데이터와 대부분 동일하나, 리더보드용인 testdataset 의 데이터에는 id 와 question 만 주어집니다. 즉 Open Domain QA 전용이며, 답을 내기 위해서 ./data/wikipediadocuments.json 을 활용합니다. 대회가 종료된 후에는 한 번도 보지 못했던 테스트 데이터를 기반으로 최종 등수가 결정됩니다.
