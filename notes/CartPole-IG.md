## 간단한 XRL 입문 실험 with Captum

### IG란
- 입력 feature를 baseline에서 현재 입력으로 서서히 변화시킬 때, 출력이 얼마나 변하는지 gradient로 누적해서 각 feature의 기여도를 구하는 방법

    - 입력 벡터 $ x = [x_1, x_2, …, x_n] $
    - baseline $ x’ = [x’_1, …, x’_n] $
    - 출력 (특정 class 확률 or Q값) $ F(x) $

$$
\text{IG}_i(x) = (x_i - x’i) \int{0}^{1} \frac{\partial F(x’ + \alpha(x - x’))}{\partial x_i} \, d\alpha
$$

- $\frac{\partial F}{\partial x_i}$ : 출력 F가 feature $x_i$ 에 얼마나 민감한지 (gradient)
- $\alpha$ : baseline에서 입력까지의 경로 비율 (0 → 1)
- 적분: 경로 전체에서 gradient를 누적
- 앞의 $(x_i - x’_i)$ : 전체 변화량을 보정

직관적인 이해:
1. baseline (ex: 전부 0인 입력)에서 시작
2. 입력을 점점 실제 x로 바꿔가며, 그때마다 “출력 F가 얼마나 변하나?”를 계산
3. 이 변화를 평균 내면 → feature별 기여도.

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

#### 추가 실험 아이디어

1. Feature 조합 Ablation:
   - 2개씩 조합하여 제거 (예: Position + Velocity 동시 제거)
   - 상호보완적 feature 쌍 발견 가능

2. 동적 Attribution 분석:
   - 에피소드 진행에 따른 각 feature의 attribution 변화율 계산
   - "위기 감지 시점"을 정량적으로 파악

3. 다른 XAI 기법과 비교:
   - SHAP, Saliency Maps, GradCAM 등과 IG 결과 비교
   - 각 방법의 장단점 이해

#### 결론

1. **Attribution ≠ Importance**: IG가 보여주는 것과 실제 feature의 중요도는 다를 수 있으며, 두 관점을 모두 확인해야 함.

2. **Context Matters**: 같은 feature라도 상황(안정/위기)에 따라 의사결정에 미치는 영향이 극적으로 달라짐.

3. XRL의 가치: 단순히 "어떤 feature가 중요한가"를 넘어 "왜 이 상황에서 이런 결정을 내렸는가"를 이해할 수 있게 함. 모델의 실패 사례를 디버깅하는 데 유용함.

4. 실용적 활용:
   - 모델 압축: 덜 중요한 feature를 제거하여 경량화
   - 전이 학습: 어떤 feature가 domain-specific인지 파악

### 코드 해석: Integrated Gradients 구현 상세

#### 1. IG를 위한 Forward Function Wrapper

```python
def model_forward(state, action_idx):
    """Returns Q-value for a specific action"""
    q_values = policy_net(state)
    return q_values[:, action_idx]
```

**핵심 포인트:**
- **왜 wrapper가 필요한가?**
  - DQN은 모든 action의 Q-value를 동시에 출력: `[Q(s,left), Q(s,right)]`
  - IG는 **단일 스칼라 출력**에 대한 gradient를 계산해야 함
  - 따라서 "특정 action의 Q-value만" 반환하는 함수가 필요

- **동작 방식:**
  1. `state`를 입력받아 policy_net에 전달 → 모든 action의 Q-values 획득
  2. `action_idx`로 특정 action의 Q-value만 선택하여 반환
  3. 예: `action_idx=0`이면 "왼쪽으로 밀기" action의 Q-value만 반환

- **IG와의 연결:**
  - IG는 이 함수를 미분하여 "각 state feature가 해당 action의 Q-value에 얼마나 기여했는지" 계산

#### 2. Integrated Gradients 초기화

```python
ig = IntegratedGradients(model_forward)
baseline = torch.zeros(1, state_dim).to(device)
```

**핵심 포인트:**
- **IG 객체 생성:**
  - `IntegratedGradients(model_forward)`: wrapper 함수를 IG에 전달
  - 이제 IG는 이 함수를 미분할 준비가 됨

- **Baseline 선택:**
  - `baseline = torch.zeros(1, state_dim)`: 모든 feature가 0인 상태
  - **Baseline의 의미:** "아무 정보도 없는 상태" (중립적 시작점)
  - **왜 0인가?**
    - CartPole의 상태는 대부분 0 근처에서 시작
    - Position=0: 중앙, Angle=0: 수직 → 가장 중립적인 상태
  
- **대안적 Baseline:**
  - 평균 상태: `baseline = torch.FloatTensor(all_states.mean(axis=0))`
  - 무작위 샘플링한 상태들의 평균
  - 선택에 따라 attribution 해석이 달라질 수 있음

#### 3. IG Attribution 계산 - 핵심 메커니즘

```python
def compute_ig_attributions(states, action_idx):
    """주어진 상태들에 대해 IG attribution 계산"""
    attributions_list = []

    for state in states:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        state_tensor.requires_grad = True  # 🔑 핵심!

        # IG 계산
        attribution = ig.attribute(
            state_tensor,
            baselines=baseline,
            additional_forward_args=(action_idx,),
            n_steps=50
        )

        attributions_list.append(attribution.detach().cpu().numpy()[0])

    return np.array(attributions_list)
```

**핵심 포인트:**

**3.1. Gradient 계산 준비**
```python
state_tensor.requires_grad = True
```
- PyTorch에 "이 텐서에 대해 gradient를 추적하라"고 지시
- 이게 없으면 backpropagation이 불가능

**3.2. IG의 수학적 원리 - 코드로 이해하기**

IG의 수식:
$$
\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha \cdot (x - x'))}{\partial x_i} d\alpha
$$

여기서:
- $x$: 실제 입력 상태 (예: `[0.5, 0.1, 0.05, 0.2]`)
- $x'$: baseline (예: `[0, 0, 0, 0]`)
- $F$: 모델 함수 (우리의 `model_forward`)
- $i$: feature index (0=Position, 1=Velocity, 2=Angle, 3=Angular Velocity)

**코드 구현:**
```python
ig.attribute(
    state_tensor,           # x (현재 상태)
    baselines=baseline,     # x' (baseline)
    additional_forward_args=(action_idx,),  # F 함수의 추가 인자
    n_steps=50              # 적분을 50개 구간으로 근사
)
```

**3.3. n_steps=50의 의미**

IG는 적분을 **Riemann Sum**으로 근사:

```
적분을 50개 스텝으로 나눔:
α = 0.00, 0.02, 0.04, ..., 0.98, 1.00

각 α에서:
1. 보간된 상태 계산: x' + α·(x - x')
   - α=0.00: baseline (모든 feature=0)
   - α=0.50: 중간 지점
   - α=1.00: 실제 상태

2. 그 지점에서 gradient 계산: ∂F/∂x_i

3. 모든 gradient를 평균내고 (x_i - x'_i)를 곱함
```

**예시:**
```
실제 상태: [0.5, 0.1, 0.05, 0.2]
Baseline:  [0.0, 0.0, 0.0, 0.0]

α=0.00: [0.00, 0.00, 0.00, 0.00] → gradient 계산
α=0.02: [0.01, 0.002, 0.001, 0.004] → gradient 계산
α=0.04: [0.02, 0.004, 0.002, 0.008] → gradient 계산
...
α=1.00: [0.5, 0.1, 0.05, 0.2] → gradient 계산

→ 50개 gradient의 평균 × (실제값 - baseline)
```

**3.4. 왜 이렇게 복잡하게?**

**단순 Gradient (Saliency):**
```python
# 단순히 현재 지점에서만 gradient 계산
grad = torch.autograd.grad(F(x), x)
```
- 문제: 비선형 함수에서 부정확
- 현재 위치의 기울기만 봄

**Integrated Gradients:**
```python
# baseline → 실제 상태까지의 경로를 따라 gradient 누적
```
- 장점: 전체 경로를 고려하여 더 공정한(fair) attribution
- **Axiom 1 (Completeness):** 모든 feature의 attribution 합 = 최종 예측값 - baseline 예측값
- **Axiom 2 (Sensitivity):** 어떤 feature가 변하면 attribution도 반드시 변함

#### 4. Attribution 분석 함수

```python
def analyze_attributions(states, action_name):
    """양쪽 action에 대한 attribution 분석"""
    # Action 0 (왼쪽)과 Action 1 (오른쪽)에 대한 attribution
    attr_action0 = compute_ig_attributions(states, 0)
    attr_action1 = compute_ig_attributions(states, 1)

    # 평균 attribution
    mean_attr0 = np.mean(attr_action0, axis=0)
    mean_attr1 = np.mean(attr_action1, axis=0)
```

**핵심 포인트:**
- **왜 두 action 모두 계산?**
  - 각 action은 다른 의사결정 패턴을 가짐
  - "왼쪽으로 밀 때"와 "오른쪽으로 밀 때" 중요하게 보는 feature가 다를 수 있음
  - 예: 왼쪽으로 밀 때는 왼쪽 경계 위치를, 오른쪽으로 밀 때는 오른쪽 경계 위치를 더 중시

- **평균을 내는 이유:**
  - 단일 상태의 attribution은 노이즈가 많음
  - 여러 비슷한 상태들의 평균으로 일반적 패턴 파악

#### 5. 시간에 따른 Attribution 추적

```python
# 각 스텝에서 선택된 action에 대한 attribution 계산
time_attributions = []
for i, (state, action) in enumerate(zip(episode_states, episode_actions)):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    state_tensor.requires_grad = True

    attribution = ig.attribute(
        state_tensor,
        baselines=baseline,
        additional_forward_args=(action,),  # 실제로 선택된 action!
        n_steps=50
    )
```

**핵심 포인트:**
- **실제 선택된 action 사용:**
  - `additional_forward_args=(action,)`: 그 순간 에이전트가 **실제로 선택한** action
  - "이 상태에서 왜 이 action을 선택했는가?"를 분석

- **시간적 패턴:**
  - 에피소드 초반: 안정 상태 → attribution 작음
  - 에피소드 중후반: 불안정 증가 → attribution 증가
  - 위기 직전: attribution 급증

#### 6. Ablation Study - IG와의 차이

```python
def evaluate_with_ablation(feature_idx_to_ablate=None, num_episodes=10):
    """특정 feature를 0으로 설정하여 에피소드 실행"""
    for _ in range(num_episodes):
        state, _ = env.reset()
        
        while not done:
            # Feature ablation
            if feature_idx_to_ablate is not None:
                state[feature_idx_to_ablate] = 0  # 🔑 핵심!
            
            # Action selection
            action = policy_net(state_tensor).argmax(1).item()
```

**핵심 포인트:**
- **IG vs Ablation의 차이:**

| 측면 | Integrated Gradients | Ablation Study |
|------|---------------------|----------------|
| **질문** | "이 feature가 Q-value 계산에 얼마나 기여하는가?" | "이 feature 없이도 태스크를 해결할 수 있는가?" |
| **방법** | Gradient 기반 수학적 계산 | 실제 feature 제거 후 성능 측정 |
| **속도** | 빠름 (gradient만 계산) | 느림 (여러 에피소드 실행) |
| **의미** | Local explanation (개별 결정) | Global importance (전체 태스크) |
| **결과** | Attribution value | Performance drop |

- **왜 둘 다 필요한가?**
  - IG: "지금 이 순간 모델이 무엇을 보고 있는가"
  - Ablation: "장기적으로 무엇이 정말 중요한가"
  - 두 관점이 다를 수 있음 (우리 실험에서 발견!)

#### 7. IG 구현의 핵심 요약

**1단계: Model Wrapping**
```python
def model_forward(state, action_idx):
    return policy_net(state)[:, action_idx]
```
→ 특정 action의 Q-value만 반환하도록 래핑

**2단계: Baseline 정의**
```python
baseline = torch.zeros(1, state_dim)
```
→ "아무 정보 없는 상태" 정의

**3단계: Gradient 활성화**
```python
state_tensor.requires_grad = True
```
→ PyTorch에 미분 추적 지시

**4단계: IG 계산 (핵심!)**
```python
attribution = ig.attribute(
    inputs=state_tensor,      # 분석할 상태
    baselines=baseline,       # 시작점
    additional_forward_args=(action_idx,),  # 추가 인자
    n_steps=50                # 적분 근사 정밀도
)
```
→ Baseline부터 실제 상태까지 50개 지점에서 gradient 계산 후 적분

**5단계: 해석**
```python
mean_attribution = np.mean(attributions, axis=0)
```
→ 각 feature별 평균 기여도 확인

#### 8. 주의사항

**주의사항:**
1. **Baseline 선택의 중요성:**
   - 다른 baseline = 다른 attribution 값
   - 도메인 지식을 활용한 의미있는 baseline 선택 필요

2. **n_steps 값:**
   - 너무 작으면 (예: 10): 부정확한 적분 근사
   - 너무 크면 (예: 1000): 계산 시간 증가, 큰 이득 없음
   - 일반적으로 50~200이 적절

3. **메모리 주의:**
   - `requires_grad=True` 상태에서는 계산 그래프가 메모리에 유지됨
   - `attribution.detach()`로 그래프 분리 필수