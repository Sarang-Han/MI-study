## MountainCar-SHAP 실험: SHAP로 RL 에이전트 설명하기

### 목적
- SHAP (SHapley Additive exPlanations)의 개념을 깊이 이해
- MountainCar 환경에서 DQN 에이전트의 의사결정 과정 분석
- IG와의 차이점 비교

### SHAP란?

**게임 이론의 Shapley Value:**
- 여러 플레이어가 협력하여 달성한 성과를 각 플레이어에게 **공정하게** 분배
- ML에서는 각 feature를 "플레이어"로, 예측값을 "성과"로 간주

**Shapley Value 계산 원리:**
```
φᵢ = Σ [|S|!(|N|-|S|-1)! / |N|!] × [f(S ∪ {i}) - f(S)]
```
- 모든 가능한 feature 조합 S를 고려
- feature i가 추가될 때의 기여도 계산
- 순서에 관계없이 공정한 기여도 측정

### 실험 구성

1. **환경**: MountainCar-v0
   - State: Position (위치), Velocity (속도)
   - Action: 왼쪽 밀기, 정지, 오른쪽 밀기
   - 목표: 언덕 정상(위치 0.5)에 도달

2. **모델**: DQN (Deep Q-Network)

3. **SHAP Explainer**: KernelExplainer
   - Model-agnostic (모든 모델에 적용 가능)
   - 정확한 Shapley values 근사

4. **분석 항목**:
   - 위치별 의사결정 전략 (왼쪽 언덕, 골짜기, 오른쪽 언덕)
   - Action별 feature 중요도
   - Feature 간 상호작용
   - Decision boundary 시각화

### 시각화 종류

1. **Summary Plot**: 전체 feature 중요도 개요
   - 각 점 = 하나의 샘플
   - X축 = SHAP value (양수/음수)
   - 색상 = feature 값

2. **Bar Plot**: 위치별, action별 평균 feature 중요도

3. **Force Plot**: 개별 의사결정의 상세 분석
   - Base value → 각 feature의 기여 → 최종 예측

4. **Waterfall Plot**: Feature 기여도를 폭포수 형태로 표현

5. **Dependence Plot**: Feature 값과 SHAP value의 관계
   - 비선형 효과 확인
   - Feature 간 상호작용 색상으로 표시

6. **Decision Boundary**: Position-Velocity 공간에서의 action 선택 영역

### SHAP vs IG (Integrated Gradients)

| 측면 | SHAP | IG |
|------|------|-----|
| **기반 이론** | 게임 이론 (Shapley Value) | Gradient 기반 |
| **계산 방법** | 모든 feature 조합 고려 | Baseline→입력 경로의 gradient 적분 |
| **속도** | 느림 (O(2^n) 조합) | 빠름 (gradient 계산) |
| **Feature 상호작용** | 명시적으로 고려 | 간접적 |
| **Model 의존성** | Model-agnostic | Gradient 필요 (미분 가능 모델) |
| **공리 만족** | 모든 Shapley 공리 만족 | Completeness, Sensitivity |
| **적용 범위** | 모든 모델 | 신경망 등 미분 가능 모델 |

### SHAP의 장단점

**장점:**
- ✅ 이론적으로 유일하게 공정한 기여도 측정 (Shapley axioms)
- ✅ Feature 간 상호작용을 자연스럽게 포착
- ✅ 직관적인 해석 (게임 이론의 공정 분배)
- ✅ 일관성 보장 (feature 기여도 변화와 SHAP value 변화 일치)

**단점:**
- ❌ 계산 비용이 매우 높음
- ❌ 많은 background 샘플 필요
- ❌ 근사 오차 발생 가능 (KernelExplainer)
- ❌ Feature가 많으면 비실용적

### 주요 발견 (예상)

실험을 실행하면 다음과 같은 패턴을 발견할 것으로 예상:

1. **Position의 중요성**:
   - 오른쪽 언덕에 가까울수록 "Push Right" action에서 position의 SHAP value 증가
   - 왼쪽 언덕에서는 "Push Left"의 position SHAP value 증가

2. **Velocity의 역할**:
   - 골짜기 중앙에서 velocity가 의사결정에 가장 중요
   - 속도 방향에 따라 어느 언덕으로 갈지 결정

3. **Feature 상호작용**:
   - Position과 Velocity가 함께 고려됨
   - 예: 오른쪽 위치 + 왼쪽 속도 = 복잡한 의사결정

4. **Action별 전략**:
   - Push Right: Position 중심 (목표에 얼마나 가까운가?)
   - Push Left: Velocity 중심 (가속도를 얻기 위해)
   - No Push: 두 feature 균형

### 실용적 활용

1. **모델 디버깅**:
   - 에이전트가 왜 실패했는지 분석
   - 예: "골짜기에서 속도를 충분히 얻지 못함"

2. **안전성 검증**:
   - 위험 상황에서의 의사결정 근거 확인
   - 예: "경계 근처에서 올바른 feature를 보는가?"

3. **Feature Engineering**:
   - 중요한 feature 파악 → 새로운 feature 추가
   - 예: "가속도", "목표까지의 에너지" 등

4. **신뢰성 평가**:
   - 에이전트가 올바른 이유로 올바른 행동을 하는지 확인

### 코드 핵심 포인트

**1. Model Wrapper:**
```python
class ModelWrapper:
    def __call__(self, x):
        # numpy → tensor → model → numpy
        # SHAP가 호출 가능한 인터페이스
```

**2. SHAP Explainer 생성:**
```python
explainer = shap.KernelExplainer(
    model=wrapped_model,
    data=background_samples  # baseline
)
```

**3. SHAP Values 계산:**
```python
shap_values = explainer.shap_values(
    X=test_samples,
    nsamples=100  # Monte Carlo 샘플 수
)
# 결과: [action_dim] x [n_samples, n_features]
```

**4. 시각화:**
```python
shap.summary_plot(shap_values[action_idx], samples)
shap.waterfall_plot(shap_explanation)
shap.dependence_plot(feature_idx, shap_values[action_idx], samples)
```

### 다음 단계

이 실험을 통해:
1. SHAP의 이론과 구현 이해
2. RL 에이전트의 의사결정 패턴 분석
3. IG와의 실질적 차이 경험
4. 다양한 시각화 기법 습득

향후 확장:
- SHAP TreeExplainer (tree 기반 모델용, 훨씬 빠름)
- SHAP DeepExplainer (딥러닝용)
- 다른 환경 (Atari, MuJoCo)에 적용
- SHAP + IG 결합 분석
