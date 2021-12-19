# Capstone_brain_teacher2021_2
------------------------------------
해당 Repository는 경희대학교 2021년 2학기 소프트웨어융합캡스톤 디자인 수업의 일환으로 만들어졌습니다.

* 참여자: 신휘명 지도교수: 이원희

## Overview
- 개요: 딥러닝은 초기 인간의 신경망을 모방하고자 발명되었으나 점차 인간의 구조를 모방하기보단 독자적인 방향으로 발전함. 그에 따라 딥러닝 모델을 신경망 측면에서 접근하여 분석하고자 하는 시도가 줄어들었음. 딥러닝은 여전히 분석하기 어려우며, 딥러닝의 이해를 돕기 위해 인간의 두뇌와의 연관성을 찾기 위한 시도를 해볼 수 있음. 딥러닝의 추론 과정과 인간의 추론 과정의 공통점을 찾고자, 인간의 두뇌가 딥러닝의 추론 과정에 개입하여 딥러닝을 개선할 수 있는 방향을 연구하는 것을 목표로 함.
- 주요내용: 딥러닝 압축 기법의 하나인 Knowledge Distillation(이하 KD) 기법을 이용하여 딥러닝 모델과 인간의 두뇌의 협업을 가능하도록 함. 기존의 KD의 teacher model로 쓰였던 일반적인 딥러닝 모델을 인간의 두뇌 신호를 이용하는 fMRI prediction 결과로 대체하여 인간의 두뇌의 활동이 딥러닝에 영향을 주도록 함.
- 목표: KD와 인간의 두뇌로부터 나온 신호를 이용하여 기존 딥러닝 모델의 성능을 향상시킴.
- 방법: 기존 딥러닝 모델의 KD에서 사용되는 딥러닝 teacher model을 사람의 fMRI prediction으로 대체함. fMRI prediction은 공개된 fMRI - 이미지 페어 데이터를 활용하여 딥러닝을 훈련시켜 진행하였으며, student 모델로 쓰일 모델 또한 딥러닝 classification 모델을 사용.

## Contributions
1. FinancialDatareader를 이용하여 주식 종목 불러오기/프로세싱/딥러닝 학습
2. 일반적인 LSTM 기반 방법이 아닌 Classification 방법을 사용하여 등락률을 맞추는 문제로 간략화, 실질적 정확도를 높임
3. 종목 예측에 도움이 되는 추가 지수를 결합하여 정확도 향상
4. 정확도 계산을 위한 세가지 추가 metric을 도입하여 실제 수익률 예측
5. GradCAM을 이용하여 각 추가 지수의 기여도 분석
6. 지속적 학습을 통한 예측 정확도 유지 및 향상 (준비중)

Main code
--------------------
Load & Process Data (각 종목의 종가)
--------------------
* FinancialDataLoader를 이용하여 코스피 종목의 종가를 불러올 수 있다.
split_iter를 이용하여 서로 다른 시작점과 끝점을 가진 split_iter개의 sequantial 데이터가 만들어진다. 
```python
import FinanceDataReader as fdr

read_lines = np.flip(df_kospi.to_numpy(), axis=0)[:100]
...

for line in np.flip(read_lines, axis=0):
 try:
   df = fdr.DataReader(line[0], start_date, end_date)
  df_ratio = df.iloc[:, 3].astype('float32')
  df_log1 = pd.DataFrame(df_ratio)
  df_ratios = np.append(df_ratios, df_ratio.to_numpy())

  for j in range(0,split_iter):
      split_point_start = j * max_test_size
      split_point_end = (split_iter - j + 1) * max_test_size
      df_train1 = df_log1.iloc[-max_train_size+split_point_start:-split_point_end]
      df_test1 = df_log1.iloc[-split_point_end:-split_point_end+max_test_size]
```

Load & Process Data (추가지수)
--------------------
* FinancialDataLoader를 이용하여 종목 뿐 아니라 주식시장에 영향을 미치는 각종 지수의 데이터를 불러올 수 있다.
10가지 추가 지수를 불러와 이전에 불러온 각 종목의 종가 데이터와 날짜가 같은 것끼리 결합(concatenate)한다. 날짜 쌍이 안 맞는 데이터는 버려진다(.dropna).
```python

df2 = fdr.DataReader('KS11', start_date, end_date)

df_ratio2 = df2.iloc[:, 0:1].astype('float32').fillna(0)
df_log2 = pd.DataFrame(df_ratio2)


df_dict = {
    0 : fdr.DataReader('IXIC', start_date, end_date),#나스닥
    1 : fdr.DataReader('KQ11', start_date, end_date),#코스닥
    2 : fdr.DataReader('USD/KRW', start_date, end_date),#달러/원
    3 : fdr.DataReader('KS50', start_date, end_date),#코스피50
    4 : fdr.DataReader('KS100', start_date, end_date),#코스피100
    5 : fdr.DataReader('KS200', start_date, end_date),#코스피200
    6 : fdr.DataReader('NG', start_date, end_date),#천연가스 선물
    7 : fdr.DataReader('ZG', start_date, end_date),#금 선물
    8 : fdr.DataReader('VCB', start_date, end_date),#베트남무역은행
    9 : fdr.DataReader('US1MT=X', start_date, end_date),#미국채권1개월수익률
}

for i in range(len(df_dict)):
  extra_df = df_dict[i]
  df_ratio_extra = extra_df.iloc[:, 0:1].astype('float32').fillna(0) #((extra_df.iloc[:, 0:1].astype('float32') - extra_df.iloc[:, 0:1].shift().astype('float32')) / extra_df.iloc[:, 0:1].shift().astype('float32')).fillna(0)
  df_log_extra = pd.DataFrame(df_ratio_extra)

  df_log2 = pd.concat([df_log2, df_log_extra],axis=1)

  df_train2 = df_log2.iloc[:]
  df_test2 = df_log2.iloc[:]

  df_train =pd.concat([df_train1, df_train2],axis=1).dropna(axis=0)[-min_train_size:]
  df_test = pd.concat([df_test1, df_test2],axis=1).dropna(axis=0)
```

등락률 계산
--------------------
* 합쳐진 학습데이터를 하나씩 읽어 전날대비 등락률로 데이터를 변조하였다.
```python
df_train_ = np.array([])
previous_train = np.zeros(df_train.shape[1])
for num, i in enumerate(df_train.to_numpy()):#[::sample_step]):
    if num == 0:
        df_train_ = np.expand_dims(previous_train, axis=0) 
    else:
        if (previous_train == 0).any():
          print(previous_train)
        new_item = (i - previous_train) / previous_train
        df_train_ = np.append(df_train_, np.expand_dims(new_item, axis=0), axis=0)
    previous_train = i
```

Classification용 타겟 데이터 생성
--------------------
* max_test_size(11로 설정) 일 이후의 종가의 변화율을 계산하여 그것이 2% 이상이면 '상승', -2% 이하면 '하락', 둘 다 아닌 것은 '유지'로 분류하였다.
```python
df_test = df_test.to_numpy()
df_test = np.array([(df_test[-1] - df_test[0])/ df_test[0] >= 0.02, (df_test[-1] - df_test[0])/ df_test[0] < -0.02])
df_test = np.append(df_test, np.expand_dims(np.logical_not(df_test[0]) * np.logical_not(df_test[1]), axis=0), axis=0)
```

모델 변경
--------------------
* 시퀀스 데이터를 처리하기 위해 기존의 2D Conv를 모두 1D Conv로 교체하였다. 
```python
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
```

정확도 측정
--------------------
* 실제 수익성을 알아보기 위해 accurcay를 포함한 세가지 지수(Precision, Crucial-Fail, Soso-Fail)를 계산하였다.
또한, Thresholding을 통해 최적의 precision을 갖는 threshold를 선택하여 사용할 수 있다. (utils.py)

```python
def accuracy(output, target, threshold=0.5):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    output = output > threshold
    target = target.squeeze()
    correct = (target == output)
    #print("C: ", correct)
    correct = torch.all(correct, dim=1)

    predicted_positive_indexes = (output[:,0].nonzero(as_tuple=True)[0])
    true_positive = (target[:,0])[predicted_positive_indexes]

    crucial_fail = (target[:,1])[predicted_positive_indexes]
    soso_fail = (target[:,2])[predicted_positive_indexes]
    
    return correct.sum() / output.shape[0], true_positive.sum() / predicted_positive_indexes.shape[0], crucial_fail.sum() / predicted_positive_indexes.shape[0], soso_fail.sum() / predicted_positive_indexes.shape[0]
    
```

Tables and Graphs
------------------------------------------
* LSTM을 사용하여 주가를 예측한 결과
![image](https://user-images.githubusercontent.com/40812418/122758082-57543180-d2d3-11eb-870b-6d2255fd6b07.png)

* 1D ResNet을 사용하여 주가를 예측한 결과(Accuracy/Precision/Crucial-Fail/Soso-Fail)
![image](https://user-images.githubusercontent.com/40812418/122758439-c2056d00-d2d3-11eb-9c5f-8829488c0483.png)

GradCAM
--------------------
* 각 추가 지수의 기여도를 정량적으로 알아보기 위하여 유명한 딥러닝 분석 기법인 GradCAM을 사용하였다.
각 지수가 생성하는 gradient를 그래프로 나타낼 수 있었으며 그것의 합을 계산하여 그 합이 큰 순으로 순위를 매길 수 있었다.
![image](https://user-images.githubusercontent.com/40812418/122761653-70f77800-d2d7-11eb-9beb-896a30695220.png)


지속적 학습(준비 중)
--------------------
* 크롤링을 통해 최신 데이터를 읽은 후 그 데이터를 사용하여 모델을 fine-tuning하는 것으로 모델의 성능을 유지하고, 향상시킬 수 있다.

## Conclusion
인간의 뇌신호인 fMRI 데이터를 활용한 prediction model을 teacher로 이용하여 knowledge distillation을 수행하는 기법을 개발
위의 기법을 활용하여 baseline과 비교하여 약 2%, 기존 knowledge distillation과 비교하여 약 1%의 성능 향상을 이뤄 냄
인간의 인지 활동에서 발생한 정보가 딥러닝 모델의 성능 향상에 도움을 준 것으로, 인간의 신경망과 인공 신경망이 일종의 연결 고리가 있을 수 있을 가능성도 열어 둘 수 있음
하지만 세부적인 분석은 아직 부족 (fMRI prediction의 어떤 요소가 성능 향상에 도움을 준 것인가? RoI 각 영역과의 연관성? fMRI 외의 다른 비딥러닝 방식의 prediction도 도움이 될 수 있지 않을까? 등등)
위의 해답을 얻기 위해 random soft targe을 활용하는 실험, fMRI로부터 다른 정보를 활용하는 실험 (activation map을 attention mask로 변환하는 등)을 future works로 진행할 수 있음
![image](https://user-images.githubusercontent.com/40812418/146673779-0d62cd0e-872f-4558-9189-2e8a5c4776d6.png)

