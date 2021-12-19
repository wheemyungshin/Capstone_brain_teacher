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
1. fMRI 데이터로부터 피실험자가 관찰하고 있던 이미지의 class를 예측하는 딥러닝 모델 훈련.
2. fMRI 예측 결과를 KD teacher 모델의 soft label output으로 활용하여 KD를 진행하는 프레임워크 개발.
3. 위의 프레임워크를 이용하여 기존의 딥러닝 기반 KD보다 발전된 성능향상을 이뤄냄.
4. 위의 프레임워크를 이용하여 다양한 RoI 영역으로부터 나온 fMRI신호를 활용하고, 각각의 결과를 비교/분석함.

Main code
--------------------

fMRI prediction
--------------------
* fMRI 데이터를 읽어 딥러닝 모델의 input으로 사용할 수 있도록 정재한다.
* 딥러닝 모델은 1차원 convolution을 사용하는 1D-ResNet18 사용.
* 결과물로 soft label output이 numpy의 형태로 저장됨.

```python
data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)
...

        pred_y, true_y, test_y_predicted, test_y_true = feature_prediction(x_train, y_train,
                                            x_train, y_train,
                                            n_voxel=num_voxel[roi],
                                            n_iter=n_iter)
...

        answers = []
        save_numpy = []
        for i in range(len(test_y_predicted)):
            pred_y = softmax(test_y_predicted[i].squeeze())
            correct = (test_y_true[i] == np.argmax(pred_y))
            answers.append(correct)
            save_numpy.append(pred_y)
        
        numpy_name = "soft_labels/"+analysis_id+".npy"
        np.save(numpy_name, np.array(save_numpy))

```

Knowledge Distillation
--------------------
* fMRI prediction 결과로 저장된 numpy 데이터를 불러온 뒤 teacher의 soft label로 사용.
* KD 를 활용하여 ResNet18 딥러닝 classification 모델 학습
```python

    if brain_teacher_target is not None:
        brain_teacher = np.load(brain_teacher_target)
...

        if brain_teacher is not None:
            soft_target = brain_teacher[idx]
            teacher_output = torch.from_numpy(soft_target).cuda()
            kd_criterion = SoftTarget(T=0.1)
            kd_loss = kd_criterion(output, teacher_output)
            kd_losses.update(kd_loss, input.size(0))
            loss += kd_loss        
```

결과
--------------------
* Baseline보다 정확도가 약 2%, 기존 KD를 사용했을 떄보다 1% 이상 상승함.
![image](https://user-images.githubusercontent.com/40812418/146674391-e31d3ac9-d24e-468d-84e5-1301dd1dd2c7.png)


* RoI별 정확도 차이 비교를 위한 추가 실험 결과
![image](https://user-images.githubusercontent.com/40812418/146674410-20ca7b19-f8cf-4723-b4fc-129c4e573500.png)

* 세부적인 RoI 별 정확도 차이 비교를 위한 피실험자별 실험 결과
![image](https://user-images.githubusercontent.com/40812418/146674423-6bd472c8-625f-4b31-9f53-63f82bb7f61d.png)

## Conclusion

* 인간의 뇌신호인 fMRI 데이터를 활용한 prediction model을 teacher로 이용하여 knowledge distillation을 수행하는 기법을 개발
* 위의 기법을 활용하여 baseline과 비교하여 약 2%, 기존 knowledge distillation과 비교하여 약 1%의 성능 향상을 이뤄 냄
* 인간의 인지 활동에서 발생한 정보가 딥러닝 모델의 성능 향상에 도움을 준 것으로, 인간의 신경망과 인공 신경망이 일종의 연결 고리가 있을 수 있을 가능성도 열어 둘 수 있음
* 하지만 세부적인 분석은 아직 부족 (fMRI prediction의 어떤 요소가 성능 향상에 도움을 준 것인가? RoI 각 영역과의 연관성? fMRI 외의 다른 비딥러닝 방식의 prediction도 도움이 될 수 있지 않을까? 등등)
* 위의 해답을 얻기 위해 random soft targe을 활용하는 실험, fMRI로부터 다른 정보를 활용하는 실험 (activation map을 attention mask로 변환하는 등)을 future works로 진행할 수 있음

## Reference
-------------------------------------
* 이 Repository는 https://github.com/KamitaniLab/GenericObjectDecoding 을 참고하여 만들어졌습니다.
* 이 연구는 논문 https://www.nature.com/articles/ncomms15037 로부터 데이터를 제공 받았습니다.
