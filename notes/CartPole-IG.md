## 간단한 XRL 입문 실험 with Captum

### 목적
나는 몇 시간동안 간단한 Explainable RL 프로젝트 샘플에 XAI 방법론을 적용해서 실험을 해보려고 한다. 목표는 captum 입문, 그리고 해당 기법의 실제 작동 원리 이해이다.

### 개요
- 간단한 실험: CartPole + Captum + Integrated Gradients (IG)
- 실험 환경: uv
- CartPole에서 에이전트가 어떤 상태(feature)에서 가장 민감하게 반응하는지 IG로 실험
- 필요한 실험: 시간에 따른 변화 비교 / Action별(왼쪽, 오른쪽) 패턴 비교 / 상태별(안정, 위기 상태) 비교
- 단계:
    - 환경 및 모델 기본 구현
    - 샘플 상태 선택
    - IG 적용
    - 다양한 실험에 대해 시각화
    - (선택) Ablation으로 feature 제거해보기