## 간단한 XRL 입문 실험 with Captum

### 목적
- 간단한 Explainable RL 프로젝트 샘플에 XAI 방법론을 적용해서 실험을 해보려고 함.
- 목표는 captum 입문, 그리고 해당 기법의 실제 작동 원리 이해.

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

### 실험 결과 / 코드 해석

#### 기본 관찰
- 학습을 다 끝내고 난 뒤 분석을 시작할 수 있다...!
    - XAI는 post-hoc 분석 기법으로, 학습이 완료된 모델의 의사결정 과정을 설명하는 것이 핵심.
- 안정 상태보다 위기 상태에서의 Attribution 값이 매우 커진다
    - 에이전트가 언제 위험을 인지하는지 보여주는 결과. 막대가 넘어지기 직전이거나 카트가 경계 근처에 있을 때, 각 feature의 미세한 차이가 생존 여부를 결정. 모델이 모든 정보를 극도로 민감하게 활용.
- Critical 상태에서 Cart Position의 Attribution이 가장 컸지만, Pole Angle의 제거가 성능 하락에 가장 큰 부정적 영향을 미침. 왜? 판단 자체를 제대로 못하게 되니까?
    - Attribution의 의미 재해석:
        - IG Attribution은 "특정 decision에 얼마나 기여했는가"를 측정
        - 높은 attribution = 그 순간 Q-value 계산에 많이 기여했다는 뜻
        - 그러나 전체 태스크 성공에 필수적이라는 뜻은 아님
    - Pole Angle의 특수성:
        - CartPole의 종료 조건: `|angle| > 0.2095 rad (≈12도)`
        - Pole Angle은 "생존/종료"를 직접 결정하는 feature
        - Cart Position은 어떻게 막대를 세울지에 대한 전략적 정보를 제공하지만, Pole Angle은 지금 위험한가?를 판단하는 전략적 정보
    - Pole Angle이 없으면 → 막대가 넘어지는지조차 모름 → 무의미한 행동
    - Cart Position이 없어도 → Pole Angle과 Angular Velocity만으로 어느 정도 균형 유지 가능
    - 즉, Pole Angle은 필수 조건, Cart Position은 성능 최적화 요소

### 추가 실험 아이디어

1. Feature 조합 Ablation:
   - 2개씩 조합하여 제거 (예: Position + Velocity 동시 제거)
   - 상호보완적 feature 쌍 발견 가능

2. 동적 Attribution 분석:
   - 에피소드 진행에 따른 각 feature의 attribution 변화율 계산
   - "위기 감지 시점"을 정량적으로 파악

3. 다른 XAI 기법과 비교:
   - SHAP, Saliency Maps, GradCAM 등과 IG 결과 비교
   - 각 방법의 장단점 이해

### 결론

1. **Attribution ≠ Importance**: IG가 보여주는 것과 실제 feature의 중요도는 다를 수 있으며, 두 관점을 모두 확인해야 함.

2. **Context Matters**: 같은 feature라도 상황(안정/위기)에 따라 의사결정에 미치는 영향이 극적으로 달라짐.

3. XRL의 가치: 단순히 "어떤 feature가 중요한가"를 넘어 "왜 이 상황에서 이런 결정을 내렸는가"를 이해할 수 있게 함. 모델의 실패 사례를 디버깅하는 데 유용함.

4. 실용적 활용:
   - 모델 압축: 덜 중요한 feature를 제거하여 경량화
   - 전이 학습: 어떤 feature가 domain-specific인지 파악