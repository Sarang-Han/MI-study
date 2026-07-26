# [번역] How To Become A Mechanistic Interpretability Researcher

원문: [How to Become a Mechanistic Interpretability Researcher](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher) (Neel Nanda, Alignment Forum)

발행일: 2025년 9월 3일

## TL;DR

- 이 글은 기계론적 해석가능성(mech interp) 연구를 **직접 하고 싶은** 사람에게 권하는 마음가짐과 과정에 관한 것이다. 명확한 방향 감각을 주는 것이 목표이므로, 의견이 뚜렷한 조언과 구체적인 권고를 제시한다.
  - Mech interp는 레버리지가 크고 임팩트가 있으며, 짧은 피드백 루프와 적당한 수준의 컴퓨팅 자원만으로 혼자서도 배울 수 있다.
  - **최소한의 기본기만 익힌 뒤, 곧바로 연구를 시작하라.** Mech interp는 경험 과학이다.
- 세 단계:
  - [**기본기 익히기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_1__Learning_the_Ropes) **(1개월 이내)** — 필수 요소를 넓이 우선(breadth-first)으로 학습한다.
  - [**미니 프로젝트로 연구를 배우기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_2__Practicing_Research_with_Mini_Projects) — 1~5일짜리 미니 프로젝트로 기초 연구 스킬을 연습하고, 빠른 피드백 루프 스킬에 집중한다.
  - [**본격적인 프로젝트로 발전시키기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_3__Working_Up_To_Full_Research_Projects) — 1~2주짜리 연구 스프린트를 하고, 그중 가장 좋은 것을 이어간다. 더 깊은 스킬과 훌륭한 연구자의 마음가짐을 탐색한다.
- [**1단계:**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_1__Learning_the_Ropes) **기본기 익히기**
  - **깊이보다 넓이. 완벽함이 아니라 괜찮은 기준선을 확보하라.**
  - **기초 학습**: [트랜스포머를 밑바닥부터 코딩해 보기](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Machine_Learning___Transformer_Basics), [핵심 mech interp 기법들](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Mechanistic_Interpretability_Techniques), [분야 전반의 지형](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Using_LLMs_for_Learning), [선형대수 직관](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Machine_Learning___Transformer_Basics), [mech interp 코드 작성법](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Mechanistic_Interpretability_Coding___Tooling) ([ARENA가 좋은 친구다](https://arena-chapter1-transformer-interp.streamlit.app/))
  - **직접 손을 더럽혀라**: 읽기만 해서는 *안 된다*. Mech interp는 근본적으로 경험 과학이다.
  - **한 달이 지나면 다음으로 넘어가라.** "다 됐다"는 느낌이나 기본기를 *전부* 다뤘다는 느낌을 기대하지 말고, 필요할 때 더 배워라. 뭔가 진짜를 시작하지 않으면 훌륭한 연구 통찰과 마주칠 일도 없다.
  - [**LLM을 적극적으로 활용하라**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Using_LLMs_for_Learning) — 완벽하진 않지만, 지금의 당신보다 mech interp를 더 잘한다! (제대로 쓴다면) 결정적인 학습 도구다.
- [**연구 과정 뜯어보기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#The_Big_Picture__Learning_the_Craft_of_Research):
  - [스킬은 여러 가지가 있고](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Unpacking_the_Research_Process), 피드백 루프의 길이로 분류하면 좋다.
    - 빠른 스킬(수 분~수 시간): 실험 작성/실행/디버깅 등
    - 느린 스킬(수 주): 우선순위 정하기, 언제 방향을 틀지 판단하기 등
    - 매우 느린 스킬(수 개월): 좋은 연구 아이디어 만들어내기 등
  - **모든 스킬을 한꺼번에 배우려 하지** ***마라***. 빠른/중간 속도의 스킬에 먼저 집중하고, 천천히 범위를 넓혀라.
  - [연구의 4단계](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand): 아이디어 찾기(**ideation**) → 직관과 감 쌓기(**exploration**) → 가설 검증하기(**understanding**) → 다듬고 글로 정리하기(**distillation**)
- [**2단계:**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_2__Practicing_Research_with_Mini_Projects) **미니 프로젝트** (각 1~5일, 총 2~4주)
  - [탐색(exploration) 마음가짐](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Practicing_Exploration): **단위 시간당 정보 획득량을 최대화하라.** 막혔을 때 빠져나오는 법을 익혀라. 배우고 있기만 하다면 계획이 없어도 괜찮다.
  - [이해(understanding) 마음가짐](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Practicing_Understanding): **모든 연구 결과는 반증되기 전까지 거짓이다.** 결과가 흥미로울수록 거짓일 가능성이 높다. 스스로가 자신의 가장 혹독한 비판자가 되어라.
  - 아이디어의 질(ideation)과 글쓰기(distillation)는 아직 우선순위가 아니다. **안목과 우선순위 판단은 직접 해보면서 배우는 것이다.**
  - 좋은 연구 아이디어를 갖는 법은 익히는 데 아주 오래 걸리므로, **초기 프로젝트를 고를 땐 편법을 써라!** [범위가 잘 잡힌 프로젝트를 골라라](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Choose_A_Project) — 예: 기존 논문 확장하기(아이디어).
  - [**LLM을 적극적으로 활용하라**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Using_LLMs_for_Research_Code) — 제대로 쓸 줄 안다면 연구와 코딩 속도를 *크게* 높여줄 것이다.
- [**3단계:**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_3__Working_Up_To_Full_Research_Projects) **본격적인 프로젝트로**
  - **1~2주 단위 스프린트로 작업하고**, 매 스프린트 후 포스트모템을 하라. *아주 잘* 되고 있는 게 아니라면 다른 프로젝트로 전환하라.
  - [**더 느린 스킬들**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Deepening_Your_Skills)**과** [**핵심 마음가짐**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Key_Research_Mindsets): 신중한 회의주의, 문헌에 대한 인식, 우선순위 설정, 높은 생산성.
  - [**화려한 과학이 아니라**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Doing_Good_Science) **좋은 과학을 하라** — 한계를 솔직히 밝히고, 체리피킹이 아님을 증명하고, 자기 데이터를 읽고, 효과가 있는 단순한 방법을 쓰고, 진짜 베이스라인을 사용하라.
  - 자신의 작업을 [**글로 정리하라**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Write_up_your_work_)! 하나의 서사로 압축한 뒤, 반복적으로 확장해 글로 완성하라.
    - **공개된 좋은 결과물이** [**최고의 자격증명**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Why_aim_for_public_output_)**이다** — 커리어, 박사과정, 멘토 찾기 등 모든 면에서.
    - **글쓰기는 뒷전의 일이 아니다** — 시간을 확보하라. [독자는 당신 생각보다 훨씬 적게 이해한다.](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Common_mistakes)
  - [**연구 아이디어 생성**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Practicing_Ideation)**을 연습하라.** 가능하다면 [멘토의 연구 안목](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Research_Taste_Exercises)을 모방 학습(imitation learning)하려 해보라.
    - [유행은 피하고](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Avoiding_Fads), [mech interp에서 무엇이 새롭고 흥미로운지](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#What_s_New_In_Mech_Interp_)를 생각하라.
- [**멘토에게 먼저 적극적으로 연락하라.**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Advice_on_finding_a_mentor) 좋은 멘토가 있으면 모든 것이 *훨씬* 쉬워진다. 콜드 이메일을 보내고, 멘토링 프로그램에 지원하라.
  - 가장 유명한 연구자가 아니라, 시간을 낼 수 있는 연구자에게 연락하라.
- **커리어:** 이 분야에서 일하고 싶다면, 지원하라! [일자리](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Where_to_apply), [멘토링 프로그램](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Mentoring_programs), [연구비](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Applying_for_grants), [학계 연구실](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Relevant_Academic_Labs).
  - 보너스: [채용 담당자는 무엇을 보는가](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#What_do_hiring_managers_look_for), [좋은 연구 멘토는 실제로 무슨 일을 하는가](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#So_what_does_a_research_mentor_actually_do_), 그리고 [박사과정을 해야 할까](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Should_you_do_a_PhD_)?
- 또한 요즘 내가 이 분야를 어떻게 바라보고 있는지, 그리고 생각이 바뀐 부분들에 대한 여러 생각을 밝힌다. 이 부분은 실용적 조언과 분리해 두었으니, 받아들이든 넘기든 자유롭게 하면 된다.
  - 신규 (2025년 12월 1일): [내가 연구에 취하고 있는 훨씬 더 실용적인 접근](https://neelnanda.io/vision)과 [우리가 유망하다고 보는 연구 방향](https://neelnanda.io/agenda)에 관한 관련 글들을 참고하라.
  - 다루는 내용: [내가 현재 이 분야를 어떻게 정의하는지](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Interlude__What_is_mech_interp_), 왜 [야심찬 리버스 엔지니어링에는 비관적이고 더 실용적인 접근에는 기대를 거는지](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#A_Pragmatic_Vision_for_Mech_Interp), [최근 어떤 연구에 흥미를 느끼는지](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#What_s_New_In_Mech_Interp_) 및 그 위에 무엇을 쌓아 올릴 것을 권하는지.

## Introduction

기계론적 해석가능성(mechanistic interpretability, mech interp)은 — 나의 지독히 편향된 견해로는 — 세상에서 가장 흥미진진한 연구 분야 중 하나다. 우리에겐 이해하지 못하는 엄청나게 복잡한 AI 모델들이 있는데, 그 안에 진짜 구조가 있다는 감질나는 신호들이 보인다. 이 구조를 부분적으로만 이해해도 온갖 가능성의 세계가 열리는데, 정작 머신러닝 연구자의 99%는 이 분야를 등한시하고 있다. 할 일이 정말 많다!

Mech interp는 혼자서 배우기에 유별나게 쉬운 분야라고 생각한다. 교육 자료가 많고, 컴퓨팅 자원도 그리 많이 필요하지 않으며, 피드백 루프가 짧다. 하지만 처음 시작하는 사람에겐 꽤 막막하게 느껴질 수 있다. 이 글은 실력을 쌓고, 이 분야에 발을 들이고, 실제로 연구를 할 수 있는 수준에 도달하는 방법, 그리고 거기서 이 분야의 커리어나 학계 자리로 나아가는 방법에 대한 나의 최신 가이드다.

이 가이드는 의도적으로 매우 의견이 뚜렷하다. 완벽하거나 폭넓은 개관을 주려는 것이 아니라, 잘 작동하리라 생각하는 생산적인 마음가짐과 구체적인 단계를 전달하고 방향 감각을 주는 것이 목표다. (그리고 링크의 상당수가 내 작업인 이유는 그게 내가 가장 잘 아는 것이기 때문이다. 미안!)

### 큰 틀에서의 관점 (High-Level Framing)

Mech interp에 입문하는 것에 대한 나의 핵심 철학은 이렇다: **절대적으로 최소한의 기본기만 최대한 빨리 익히고, 곧바로 연구를 하면서 배우는 쪽으로 전환하라.**

목표는 연구에 손대기 전에 모든 논문을 읽는 것이 아니다. 연구를 하다 보면 빈틈이 보일 테고, 그때 돌아가서 더 배우면 된다. 하지만 프로젝트에 발을 딛고 있으면 학습을 이끌어 줄 방향이 훨씬 더 분명해지고, 지금 배우는 것이 왜 실제로 중요한지 맥락이 잡힌다. 필요한 건 자신이 무엇을 하는지 어느 정도 이해한 채로 프로젝트를 시작할 수 있을 만큼의 기반뿐이다.

처음부터 연구의 질이나 완벽한 프로젝트 아이디어를 갖는 것에 스트레스받지 마라. [연구 안목(research taste)](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/Ldrss6o3tiKT6NdMm)이나 우선순위를 정하는 능력 같은 핵심 스킬은 발전하는 데 시간이 걸린다. 경험을 쌓는 것은 — 설령 엉망인 경험이라도 — 실험을 어떻게 돌리고 해석하는지 같은 기본기를 가르쳐 주고, 그것이 다시 상위 수준의 스킬을 배우는 데 도움이 된다.

나는 이것을 세 단계로 나눈다:

1. [**기본기 익히기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_1__Learning_the_Ropes) — 기초를 넓이 우선(breadth-first)으로 훑고, 아무리 길어도 한 달 뒤에는 2단계로 넘어간다.
2. [**미니 프로젝트로 연구 연습하기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_2__Practicing_Research_with_Mini_Projects) — 버려도 되는 1~5일짜리 연구 프로젝트를 한다. 피드백 루프가 가장 빠른 기초 연구 스킬을 연습하는 데 집중하고, 최고의 아이디어를 갖는다거나 글로 정리하는 것에는 스트레스받지 마라. 2~4주 뒤에는 3단계로 넘어간다.
3. [**본격적인 프로젝트로 발전시키기**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Stage_3__Working_Up_To_Full_Research_Projects) — 1~2주 단위 스프린트로 작업한다. 매 스프린트가 끝나면 포스트모템을 하고 다른 것으로 전환하라. *단*, 아주 잘 되고 있고 탄력이 붙었다면 예외다. 결국에는 더 장기적인 무언가를 하게 될 것이다. 더 깊은 스킬과 연구자의 마음가짐에 대해 생각하기 시작하고, 좋은 아이디어를 내는 연습을 하며, 잘 풀린 스프린트는 좋은 공개 결과물로 정리하는 것을 우선하라.

## Stage 1: Learning the Ropes (기본기 익히기)

이 단계의 목표는 기본기를 익히는 것이다. mech interp 라이브러리로 실험을 작성하는 법, 핵심 개념을 이해하는 것, 분야의 지형을 파악하는 것.

목표는 **학습을 끝내는 것이 아니라**, 나머지 학습을 연구를 하면서 해나갈 수 있을 만큼만 배우는 것이다. 무자비하게 우선순위를 정하라. **길어야 1개월 뒤에는 2단계로 넘어가라.** 어느 부분이 필수(essential)이고 어느 부분이 있으면 좋은 정도(nice to have)인지 표시해 두었다.

**논문만 읽고 있지 마라.** 학구적인 성향의 사람들이 흔히 저지르는 실수가, 코드를 쓰기 전에 손에 잡히는 모든 논문을 몇 달씩 읽는 것이다. 그러지 마라. Mech interp는 경험 과학이고, 직접 손을 더럽혀야 학습에 필요한 핵심 맥락이 생긴다. 논문 읽기와 코딩 튜토리얼·소규모 탐색 연구를 번갈아 가며 하라. 아주 작은 탐색적 프로젝트가 어떤 모습인지 감을 잡으려면 [내 연구 워크스루 영상들](https://www.youtube.com/playlist?list=PL7m7hLIqA0hr4dVOgjNwP2zjQGVHKeB7T)을 보라.

LLM은 핵심 도구다 — 잘 쓰는 법에 대한 조언은 [아래 섹션](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Using_LLMs_for_Learning)을 참고하라.

### 머신러닝 & 트랜스포머 기초 (Machine Learning & Transformer Basics)

*기본적인 Python과 입문 수준의 ML 개념은 이미 안다고 가정한다.*

- 수학:
  - **선형대수가 왕이다 (필수):** 벡터와 행렬로 유창하게 사고할 수 있어야 한다. mech interp나 ML 연구를 하기 위해 배워야 할 일반 수학 중 압도적으로 가치가 높다.
    - *자료:* 3Blue1Brown의 [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).
    - **강력 추천:** [A Mathematical Framework For Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)를 컨텍스트 윈도우에 넣고, 트랜스포머 내부에 대한 당신의 직관을 시험할 연습문제를 LLM에게 만들게 하라.
    - LLM은 선형대수가 진짜로 체득됐는지 확인하는 데 아주 좋다. 배운 것과 개념들 사이의 연결을 요약해 보고 LLM에게 맞는지 물어보라. 예를 들어:
      - SVD를 이해하고 왜 그것이 작동하는지 확실히 하라.
      - 기저 변환(change of basis)이 무엇을 뜻하고 왜 중요한가.
      - 저랭크(low rank) 행렬과 풀랭크(full rank) 행렬이 다른 핵심 지점들.
  - **그 밖의 것들:** 기초 확률론, 정보이론, 최적화, 벡터 미적분.
    - 트랜스포머와 가장 관련 깊은 부분에 대해 LLM 튜터에게 이해도를 퀴즈로 확인받아라.
  - 그 외의 수학 분야는 대체로 굳이 배우지 마라 (재미로 하는 거라면 예외!).
- PyTorch 실전 ML: (필수)
  - 간단한 트랜스포머(GPT-2 같은)를 밑바닥부터 코딩하라. ARENA Chapter 1.1이 훌륭한 코딩 튜토리얼이다[^ue9pdw6v8rj].
    - 이것은 mech interp에 대한 직관 **과** PyTorch 사용 능력을 동시에 길러준다.
    - 이에 대해 기초부터 시작하는 영상 튜토리얼 두 개를 만들어 두었다 — 뭘 해야 할지 모르겠다면 [여기서 시작하라](https://www.youtube.com/watch?v=bOYE6E8JrtU&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz)!
    - 그리고 PyTorch 기초처럼 빠진 배경지식은 LLM으로 채워라.
- 클라우드 GPU:
  - 언어 모델을 돌릴 수 있어야 하는데, (대개) GPU가 필요하다.
  - 빠르게 시작하려면 Google Colab으로 출발해도 되지만, 장기적으로는 매우 제약이 크다. 클라우드 GPU를 빌려 쓰는 법을 익혀라.
    - 최신 MacBook Pro나 강력한 게이밍 GPU가 달린 컴퓨터라면 로컬에서 LLM을 돌릴 수도 있다.
  - *자료:* ARENA에 [가이드](https://arena-chapter0-fundamentals.streamlit.app/#vm-setup-instructions)가 있다. 나는 제공업체로 [runpod.io](http://runpod.io)를 좋아하고, [vast.ai](http://vast.ai/)가 더 저렴하다.
  - nnsight를 쓰면 [그들이 직접 호스팅하는 특정 모델들에 대해 해석가능성 작업](https://nnsight.net/notebooks/tutorials/get_started/start_remote_access/)을 할 수도 있는데, LLaMA 3 405B도 포함되어 있어 더 큰 모델을 다뤄보는 좋은 방법이 된다.

### 기계론적 해석가능성 기법들 (Mechanistic Interpretability Techniques)

Mech interp 연구의 상당 부분은 어떤 맥락에서 어떤 기법을 적용해야 하는지 아는 것처럼 보인다. 시작할 때 머릿속에 정리해 두는 것을 우선해야 할 핵심 사항이다. 교육 자료를 읽는 것과 ARENA 같은 코딩 튜토리얼을 하는 것을 섞어가며 배우게 될 것이다(다음 소절에서 다룬다).

- [Ferrando et al](https://arxiv.org/abs/2405.00208)이 핵심 기법들에 대한 좋은 **개관**이다. 길어서 전부 읽는 것을 우선순위로 둘 필요는 없지만, 훌륭한 참고 자료다.
  - LLM 컨텍스트 윈도우에 넣고 질문하거나, 연습문제를 써 달라고 하라.
- **필수:** 아래 **핵심 기법들**을, GPT-2 Small 같은 간단한 모델에서 직접 코딩할 수 있을 정도로 확실히 이해하라[^hh6mwdeo4zm]:
  - Activation Patching (활성값 패칭)
  - Linear Probes (선형 프로브)
  - Sparse Autoencoder(SAE) 사용하기 (SAE를 *학습시키는* 코드가 아니라 *사용하는* 코드만 쓸 줄 알면 된다)
  - Max Activating Dataset Examples (최대 활성 데이터셋 예시)
  - 있으면 좋은 것:
    - Steering Vectors (조종 벡터)
    - Direct Logit Attribution(DLA) (더 단순한 버전은 logit lens라고 부른다)
  - **핵심 연습:** Ferrando et al을 컨텍스트에 넣은 채로 각 기법을 LLM에게 설명하고 피드백을 요청하라. 전부 맞을 때까지 반복하라.
    - 진짜 피드백을 받으려면 다른 사람이 쓴 것인 척하는 안티-아첨(anti-sycophancy) 프롬프트를 써라. 예: "어떤 사람이 이렇게 주장하는 걸 봤는데 저는 꽤 틀린 것 같습니다. 그 사람이 놓친 부분에 대해 직설적이되 건설적인 피드백을 줄 수 있게 도와주세요. [당신의 설명 삽입]"
- 가치 있는 **블랙박스 해석가능성(black-box interpretability)** 기법들도 많다는 걸 기억하라! (즉, 모델 내부를 쓰지 않는 기법들) 모델의 사고 사슬(chain of thought)을 읽는 것만으로 모델의 알고리즘을 정확히 추측할 수 있는 경우가 많다. 프롬프트를 신중하게 변형하는 것은 가설을 인과적으로 검증하는 강력한 방법이다.
  - 이것들은 추가 도구다. 조사의 올바른 첫 단계는 그냥 모델과 많이 대화하며 그 행동을 관찰하는 것인 경우가 많다. 순수주의자가 되어 "엄밀하지 않다"며 무시하지 마라 — 다른 모든 기법과 마찬가지로 쓸모와 결함이 함께 있을 뿐이다.
    - 프론티어 모델의 "자기 보존(self-preservation)"을 해석하는 [내가 지도한 프로젝트](https://www.alignmentforum.org/posts/wnzkjSmrgWZaBa2aC/self-preservation-or-instruction-ambiguity-examining-the) 하나는 단순한 블랙박스 기법으로 시작했는데 그냥 잘 됐고, 더 화려한 건 전혀 필요하지 않았다.
  - 모델의 입에 말을 넣어주는 [token forcing](https://arxiv.org/abs/2312.12321)(prefill attack이라고도 한다) 같은 좀 더 정교한 블랙박스 기법도 이해하라.

### Mech interp 코딩 & 툴링 (Mechanistic Interpretability Coding & Tooling)

- **목표:** 실험을 돌리고 모델 내부를 가지고 "노는" 것에 익숙해져라. 엔지니어링 기본기를 갖춰라[^sxyjce3nii]. 직접 손을 더럽혀라.
- **ARENA**: ARENA에는 [Callum McDougall이 만든 환상적인 코딩 튜토리얼 모음](https://arena-chapter1-transformer-interp.streamlit.app/)이 있다. 그냥 가서 하면 된다. 다만 양이 엄청나게 많으니 **무자비하게 우선순위를 정하라.**
  - **필수: Chapter 1.2** (Interpretability Basics — 툴링, 직접 관찰, 패칭을 다루는 앞의 세 섹션을 우선하라).
  - *권장:* 1.4.1 (Causal Interventions & Activation Patching — 핵심 기법이다).
  - *해볼 만함:* 1.3.2 (Sparse Autoencoders — 섹션 1은 훑거나 건너뛰어도 된다. 나머지에서 얻어야 할 핵심은 SAE가 무엇인지, 강점과 약점은 무엇인지, 오픈소스 SAE를 어떻게 쓰는지에 대한 직관이다. 학습시키는 건 신경 쓰지 마라).
- **툴링 (필수):** 최소한 하나의 mech interp 라이브러리에 능숙해져라. 실험을 돌릴 때 쓰게 될 도구다.
  - [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): 9B 이하의 작은 모델에서 더 복잡한 해석가능성 실험을 작성하거나 여러 모델을 한꺼번에 다루고 싶을 때 최고다.
    - 2025년 9월 초 기준, TransformerLens [v3](https://github.com/TransformerLensOrg/TransformerLens/releases/tag/v3.0.0a5)가 알파 단계인데, 큰 모델에서도 잘 작동하고 훨씬 유연하다.
  - [nnsight](http://nnsight.net/): 성능이 더 좋고 더 큰 모델에서 잘 작동한다. HuggingFace transformers 같은 표준 LLM 라이브러리를 감싼 래퍼일 뿐이다.
- **LLM API**: LLM을 프로그래밍적으로 호출하는 API 사용법을 익혀라. 어떤 데이터에 대해 정성적인 것을 측정하거나 합성 데이터셋을 생성할 때 대단히 유용하다.
  - 나는 [openrouter.ai](http://openrouter.ai)를 좋아한다. 중요한 LLM 거의 전부를 한 곳에서 쓸 수 있다. GPT-5와 Gemini는 가격이 합리적이고 좋은 기본 선택지이며, 다양한 크기가 있다.
    - Cerebras와 Groq는 일반 제공업체보다 처리량이 *훨씬* 높고 소수의 오픈소스 모델을 서빙하니 확인해 볼 만하다.
  - 연습: API로 LLM에게 행복한 프롬프트 32개와 슬픈 프롬프트 32개를 생성하게 한 뒤, 평균 활성값의 차이[^kte6u8splw](예: 중간 층의 residual stream)를 취해 (예컨대 GPT-2 Small용) 행복 조종 벡터(happiness steering vector)를 만들어 보라. 몇 가지 예시 프롬프트에 대한 응답을 생성하는 동안 이 벡터를 모델의 residual stream에 더하고[^2ob115pcmet], LLM API로 응답이 얼마나 행복해 보이는지 점수를 매기게 해서, 조종했을 때 이 점수가 올라가는지 확인하라.
- **오픈소스 LLM**: 해석 대상으로서 오픈소스 LLM을 많이 다루게 될 것이다. 최고의 오픈소스 LLM은 자주 바뀐다.
  - 2025년 9월 초 기준, Qwen3가 좋은 기본 모델 패밀리다. 각 모델에 추론(reasoning) 모드와 비추론 모드가 있고, 크기 선택지가 넉넉하며, 대부분 dense 모델이다[^1b9r0ass7sd].
    - Gemma 3와 LLaMA 3.3은 괜찮은 비추론 모델이다. gpt-oss와 LLaMA 4에 대해서는 안 좋은 이야기를 들었다.
  - *함정:* 오픈소스 LLM마다 채팅이나 추론 토큰의 토크나이제이션과 포맷이 다른 경우가 많다. 잘못된 토큰 포맷을 쓰면 성능이 *약간만* 떨어져서 알아채기 어려운 채로 결과를 오염시킬 수 있다. 주의를 기울이고, 어디에 문서화되어 있을지 열심히 찾아보고, 공식 평가 결과와 비교하는 식으로 온전성 검사(sanity check)를 하라.

### 문헌 이해하기 (Understanding the literature)

우선순위는 개념과 기초를 이해하는 것이지만, 분야의 지형에 대한 감도 필요하므로 최소한 논문을 어느 정도 읽는 연습은 해야 한다.

- 기억하라, **깊이보다 넓이.** 훑어보고, 무엇이 있는지 감을 잡고, 가장 흥미로운 것만 깊이 파고들어라.
  - 여기서 **LLM**을 적극적으로 써야 한다. 읽을까 고민 중인 것을 주고 요약을 받고, 그 연구에 대해 질문하고, 당신의 이해를 요약해 주고 피드백을 요청하라(안티-아첨 프롬프트와 함께).
    - 스스로 검증할 수 없다면, 여러 LLM에게 물어 교차 확인하고 모두 일관된 말을 하는지 확인하라.
- 여기 [내가 가장 좋아하는 논문 목록](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite)이 있다(2024년 중반 기준). 요약과 의견이 붙어 있다.
  - 이걸 전부 통독하려 하지 *마라*. 요약을 훑고, 초록을 훑고, 몇 개를 골라 LLM과 더 깊이 탐색한 *다음에* 논문 전체를 읽을지 결정하라.
  - [내 YouTube 채널](https://www.youtube.com/@neelnanda2469): [논문 워크스루](https://www.youtube.com/watch?v=KV5gbOmHbjU&list=PL7m7hLIqA0hpsJYYhlt1WbHHgdfRLM2eY&pp=gAQB), [내가 연구하는 모습을 녹화한 것](https://www.youtube.com/watch?v=LP_NTmMvp10&list=PL7m7hLIqA0hr4dVOgjNwP2zjQGVHKeB7T), 그리고 강연들.
- [Open Problems In Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)는 최근에 나온 괜찮은 문헌 리뷰로, 최상급 mech interp 연구자들이 대거 참여했다.
  - 다만 이 논문은 기본적으로 의견이 강하고 서로 잘 동의하지 않는 연구자들이 각자 자기 섹션을 쓰면서 종종 강한 견해를 낸 결과물임을 유의하라. 너무 그대로 받아들이지는 말되, 무엇이 있는지 빠르게 파악하기엔 좋다.
- **깊이 파기(deep dives):** 최소한 논문 한 편은 처음부터 끝까지 꼼꼼히 읽어야 한다. 프로젝트에 극도로 관련 깊은 논문이 몇 편 있을 때 쓰게 될 유용한 스킬이다.
  - 이건 단순히 글자를 읽는 것 이상이다! 요약을 직접 써 보고, LLM의 도움으로 주변 맥락을 이해하려 하고, 그 논문이 왜 존재하는지, 동기가 무엇인지, 어떤 문제를 풀려는지 등을 설명할 수 있어야 한다.
  - 바벨 전략(barbell strategy)을 목표로 하라. 대부분의 논문에는 최소한의 노력만 들이고, 소수에 많은 노력을 쏟아라.
- **LLM**: LLM은 문헌을 탐색하는 데 대단히 유용한 도구지만, 잘못 쓰면 제 발등을 찍기 쉽다.
  - 문헌 검색 엔진처럼 쓸 수 있다(특히 문헌 리뷰 몇 편이나 출발점이 될 논문을 컨텍스트에 넣으면). 사실상 문헌 조사를 하고, 궁금한 질문에 관련된 연구를 찾는 식이다.
    - 논문을 훑는 걸 돕는 도구로도 쓸 수 있다 — 논문을 컨텍스트 윈도우에 넣고[^bzop9pji3nl] 요약을 받거나 질문을 던져라.
    - 환각(hallucination)이 걱정된다면 답변을 인용문으로 뒷받침하게 하거나(그 인용이 실재하고 말이 되는지 확인하라), 답변을 다른 LLM에게 주고 부정확한 부분을 혹독하게 비판해 달라고 할 수 있다. 솔직히 나는 대개 그렇게까지 하지 않는다. 요즘 프론티어 추론 모델은 꽤 괜찮다.
  - 깊이 파기를 돕는 도구로도 쓸 수 있다 — 논문은 직접 읽어야 하지만, 논문을 컨텍스트에 넣은 LLM 채팅창을 열어두고 읽으면서 헷갈릴 때마다 맥락 등을 물어보는 것을 권한다.

### 학습에 LLM 활용하기 (Using LLMs for Learning)

*주: 이 섹션은 빠르게 낡을 것으로 예상한다! 2025년 9월 초 작성.*

LLM은 학습, 특히 새로운 분야를 배울 때 대단히 유용한 도구다. 전문가를 이기기는 어려워하지만, 초심자는 곧잘 이긴다. 이 과정 내내 LLM을 꾸준히 쓰고 있지 않다면 상당한 가치를 그냥 흘려보내고 있는 것 같다.

다만 LLM은 이상한 결함과 강점을 함께 갖고 있으므로, 어떻게 쓸지 의식적으로 고민할 가치가 있다:

- **좋은 모델을 써라:** 최고의 유료 모델은 무료 ChatGPT 같은 것보다 훨씬 낫다. 인색하게 굴지 마라. 가능하다면 월 20달러 구독을 하라. 큰 차이를 만든다. Gemini 2.5 Pro, extended thinking을 켠 Claude 4.1 Opus, GPT-5 Thinking 모두 합리적인 선택이다. (thinking이 없는 GPT-5나 GPT-4o처럼 더 오래된 모델은 쓰지 *마라*. 추론 모델은 큰 업그레이드다.)
  - 구독이 어렵다면 Gemini 2.5 Pro는 무료로도 쓸 수 있고, 무료 중에서는 최고다.
  - Gemini 2.5 Pro는 [AI Studio](https://aistudio.google.com/prompts/new_chat)를 통해 써라. 메인 Gemini 인터페이스보다 훨씬 낫고 무료 사용자 기준 rate limit도 훨씬 관대하다. 항상 compare 모드(헤더의 화살표 두 개 버튼)를 켜서 Pro의 응답 두 개를 나란히 보라.
  - 학습용으로 여러 LLM을 비교하고 왜 현재 Gemini를 선호하는지에 대한 내 MATS 수료생 Paul Bogdan의 [의견](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher?commentId=jDzbZGnjWDMsNjDPQ)을 참고하라.
- **시스템 프롬프트:** 시스템 프롬프트는 큰 차이를 만든다. 무엇을 원하는지, 어떻게 해주길 원하는지 구체적이고 명확하게 써라.
  - LLM은 이걸 잘한다. 나는 그냥 과제가 무엇인지, 내 기준은 무엇인지, 원하지 않는 실패 양상은 무엇인지를 두서없이 늘어놓기만 하고, 그러면 LLM이 프롬프트를 대신 써 준다.
  - 프롬프트가 잘 안 먹히면 LLM에게 무엇이 잘못됐는지 말해주고 프롬프트를 다시 써 줄 수 있는지 보라.
- **관점을 합쳐라:**
  - 같은 질문을 여러 프론티어 LLM에게 던지고, LLM B의 응답을 LLM A에게 주면서 강점과 약점을 평가한 뒤 합쳐 달라고 하라.
    - 어떤 논점이 두 원본 응답 모두에 있다면, 아마 환각이 아니다.
  - LLM의 답을 팩트체크하고 싶다면, 안티-아첨 프롬프트와 함께 다른 LLM에게 넘겨라.
- **안티-아첨(Anti-Sycophancy) 프롬프트:** LLM은 비판적 피드백을 주는 데 서툴다. 피드백받고 싶은 것을 다른 사람이 썼다고 가장해서, 비판하는 쪽이 오히려 아첨하는 행동이 되도록 요청을 프레이밍하라.
  - *"친구가 이 설명을 쓰고 아주 솔직한 피드백을 부탁했어요. 제가 봐주면 오히려 서운해할 거예요. 가장 유용한 피드백을 줄 수 있게 도와주세요."*
  - *"누가 이렇게 주장하는 걸 봤는데, 제겐 꽤 멍청해 보여요. 어떻게 생각하세요?"*
  - *"어떤 멍청이가 이런 걸 썼는데 정말 짜증나네요. 혹독하지만 진실된 반박을 써 주세요."*
- 수동적으로 말고 능동적으로 배워라:
  - 당신의 이해를 자기 말로 LLM에게 **요약**해 주고 비판적 피드백을 요청하라. 논문을 읽거나 새 개념을 배울 때마다 하라.
  - **소크라테스식으로** 가르쳐 달라고 해 보라. 참고: 공식 "study mode"보다 당신이 더 나은 시스템 프롬프트를 설계할 수 있을 것이다.
  - 이해도를 시험할 **연습문제를 생성**해 달라고 하라. 필요에 따라 수학 문제와 코딩 문제 모두.
    - Gemini는 객관식 퀴즈를 만들 수 있는데, 이걸 즐기는 사람들도 있다.
    - 코딩 문제는 ARENA 튜토리얼처럼 테스트가 딸린 형태로, 빈 함수만 있는 템플릿 코드와 함께 요청할 수 있다.
- **컨텍스트 엔지니어링:** 요즘 LLM은 관련 정보가 컨텍스트에 들어 있을 때 훨씬 유용하다. 해당 논문이나 관련 라이브러리의 소스 코드[^207k0k5nobb]를 주면 훨씬 더 도움이 된다.
  - mech interp 질의용으로 저장해 둔 컨텍스트 파일 모음은 [이 폴더](https://drive.google.com/drive/u/0/folders/1GfrgKJwndk-twnJ8K7Ba-TE9i_8wBWAU)를 참고하라. 무엇이 필요한지 모르겠다면 그냥 [이 기본 파일](https://drive.google.com/file/d/18cF3lkU17_elUSv0zk8KSVejM1jGfNnz/view?usp=drive_link)을 쓰면 된다.
  - [aistudio.google.com](http://aistudio.google.com/)을 통한 Gemini 2.5 Pro(100만 토큰 컨텍스트 윈도우)를 추천한다. UI가 더 낫다. compare 모드는 항상 켜 두어라. 답변 두 개를 나란히 받을 수 있다.
- **음성 받아쓰기:** 무료 음성-텍스트 변환 소프트웨어로 LLM에게 받아쓰기를 시키고 편집 없이 그대로 돌려도 LLM은 잘 알아듣는다. 개인적으로는 이쪽이 훨씬 편하다. 특히 머릿속을 쏟아낼 때 그렇다.
  - Mac에서는 [Superwhisper](http://superwhisper.com)가 훌륭하다. Superwhisper는 현재 Windows용이 없지만, Windows 사용자는 [Whispr Flow](https://wisprflow.ai/)를 쓸 수 있다.
- **코딩:** Cursor 같은 LLM 도구는 코딩에 훌륭하지만, 목표가 *학습*이라면 아니다. ARENA 같은 것을 할 때는 브라우저 기반 LLM만 쓰도록 스스로를 제한하고, 오직 튜터로만 써라. 코드를 복사·붙여넣기 하지 마라. 목표는 연습문제를 완료하는 게 아니라 배우는 것이다.

## Interlude: Mech interp란 무엇인가? (What is mech interp?)

*[**그래서 다음에 뭘 해야 하나**](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#The_Big_Picture__Learning_the_Craft_of_Research) 부분으로 건너뛰어도 좋다.*

이쯤에서 mech interp가 *실제로* 무엇인지 짚어볼 만하다. 우리는 대체 여기서 뭘 하고 있는가? 기계론적 해석가능성을 정확히 어떻게 정의할지에 대한 합의된 정의는 없고, 연구자마다 아주 다른 견해를 낼 것이다. 하지만 *내* 작업 정의는 다음과 같다[^979wnkvgpa4].

- **해석가능성(Interpretability)** 은 모델을 이해하는 것, 그 행동에 대한 통찰을 얻는 것, 그 안의 인지, 왜 그리고 어떻게 작동하는지 등을 연구하는 것이다. 이게 중요한 부분이자 분야의 심장이다.
- **기계론적(Mechanistic)** 이란 모델의 내부, 즉 가중치와 활성값을 사용한다는 뜻이다.
- 따라서 **기계론적 해석가능성**이란 모델의 내부를 사용해 모델을 이해하려는 모든 접근을 말한다.
  - 이것은 다른 가치 있는 방향들과 구별된다. 예컨대 내부를 쓰지 않고 모델을 이해하는 **블랙박스 해석가능성**, 그리고 조종 벡터처럼 모델의 내부를 다른 용도로 쓰는 **모델 내부(model internals)** 연구가 있다.

**왜 이런 정의인가?** 임팩트 있는 연구를 하려면 다른 사람들이 놓치고 있는 방향을 찾는 것이 대개 좋다. 나는 머신러닝의 대부분을 "비(非)기계론적 비(非)해석가능성"으로 본다. ML 연구의 99%는 모델의 입력과 출력만 보고, 그 행동을 제어하는 것을 북극성으로 삼는다. 진보는 숫자를 올리는 것으로 정의되지, 왜 작동하는지 설명하는 것으로 정의되지 않는다. 이는 매우 성공적이었지만, 내 생각엔 많은 가치를 남겨두고 있다. 기계론적 해석가능성은 이보다 더 잘해보자는 것이고, [AlphaZero를 해석해서 그랜드마스터에게 체스를 더 잘 두는 법을 가르친 것](https://arxiv.org/abs/2310.16410) 같은 멋진 성과를 여럿 냈다.

**왜 신경 쓰는가?** 물론 우리의 목표가 "위 정의에 맞을 때만 그 일을 한다"는 것은 아니지만, 나는 이 정의가 유용하다고 본다. 이 논의를 하려면 먼저 우리의 실제 목표를 생각해 봐야 한다. 내게 **궁극적인 목표는 인간 수준(혹은 그 이상)의 AI 시스템을 더 안전하게 만드는 것**이다. 내가 mech interp를 하는 이유는, 모델 내부에서 무슨 일이 일어나는지 실용적으로 쓸 만할 만큼은 이해하게 되리라 생각하기 때문이다(그리고 mech interp가 재미있기 때문이기도 하다!). 모델이 어떻게 작동하는지 더 잘 이해하고, 우리에게 거짓말을 하는지 탐지하고, 예상치 못한 실패 양상을 탐지하고 진단하는 것 등이다. 하지만 사람마다 목표는 다르다. 예컨대 오늘의 실세계 유용성, 미학적 아름다움, 과학적 통찰일 수도 있다. 당신의 목표는 무엇인지 생각해 볼 가치가 있다.

이 프레이밍에서 나오는 함의 몇 가지를 짚어두면:

- 나의 궁극적인 **북극성은 실용주의**다 — (신뢰할 수 있게) 유용할 만큼의 이해를 달성하는 것. "모델을 완전히 리버스 엔지니어링한다" 같은 하위 목표는 수단일 뿐이다.
  - 최근 몇 년간 내 연구 우선순위에서 일어난 큰 변화 중 하나는 **리버스 엔지니어링이 올바른 목표가 아니라**는 결론에 이른 것이다. 대신 내부를 이용해 유용한 일을 할 수 있게 해주는 실용적인 작업을 더 직접적으로 시도해야 한다고 생각한다. 이 변화는 [뒤에서](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#A_Pragmatic_Vision_for_Mech_Interp) 더 논한다.
- 이것은 **넓은 정의**다. 역사적으로 이 분야는 모델의 야심찬 리버스 엔지니어링 같은 더 구체적인 아젠다에 집중해 왔다. 하지만 우리 스스로를 제한해서는 안 된다고 생각한다. 중요하면서도 등한시된 다른 방향이 많고, 이 분야는 넓은 영역을 다룰 만큼 충분히 크다[^3zw26zes9dx].
- 이것은 내부를 *사용하는* 것이 아니라 **이해하는** 것에 관한 것이다. 조종 벡터 같은 모델 내부 기법은 모델의 행동을 조형하는 데 유용할 수 있지만, 프롬프팅이나 파인튜닝 같은 강력한 방법들과 경쟁해야 한다. ML에서 이해를 달성할 수 있는 영역은 극히 드물다.
- **순수주의자가 되지 마라.** 내부를 쓰는 것은 목적을 위한 수단이다. 블랙박스 방법이 올바른 도구라면 그걸 써라.

## The Big Picture: 연구라는 기예를 배우기 (Learning the Craft of Research)

자, 튜토리얼을 다 거쳤고, 핵심 개념도 이해했고, 기본적인 실험 코드도 쓸 수 있게 됐다. 이제 어려운 부분이 온다. 실제로 mech interp 연구를 어떻게 하는지 배우는 것이다[^7cxhc64szn8].

물론 이건 본질적으로 배우기 어려운 일이다. 하지만 내 생각에 사람들은 여기서 무엇을 해야 하는지 오해하거나, 모든 것을 한꺼번에 배우려 하거나, 더 일반적으로는 불필요하게 스스로를 힘들게 만드는 경우가 많다. 핵심은 과정을 쪼개고, 관련된 여러 스킬을 이해하고, **피드백 루프가 가장 빠른 조각부터 배우는 데 집중하는 것**이다.

나는 이것을 두 단계로 나눌 것을 제안한다[^9wj0u0qz3q].

**2단계:** 각각 1~5일짜리인 버려도 되는 미니 프로젝트를 여러 개 한다. 최고의 프로젝트를 고르거나 공개 결과물을 내는 것에 스트레스받지 마라. 목표는 피드백 루프가 가장 빠른 스킬을 배우는 것이다.

**3단계:** 이걸 몇 주 한 뒤에는 더 야심차게 가기 시작한다. 프로젝트를 어떻게 고르는지에 더 신경 쓰고, 더 미묘한 스킬을 얻고, 결과를 어떻게 글로 정리할지 익힌다. 여전히 1~2주 단위 스프린트로 반복적으로 작업하되, 잘 풀리면 결국 더 장기적인 프로젝트로 이어지는 것을 권한다.

참고: 1단계에서 2단계로 갈 때와 달리, 2단계에서 3단계로의 전환은 더 큰 프로젝트를 맡고 더 야심차게 되어가면서 꽤 점진적으로 일어나야 한다. 2단계에서 3~4주 정도 뒤가 좋은 기본값이겠지만, 거창한 공식적 전환이 있을 필요는 없다.

**멘토십:** 좋은 멘토는 강력한 가속기이며, 멘토를 찾는 것이 당신의 주요 우선순위여야 한다. 커리어 섹션에서 [좋은 멘토를 찾는 방법](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#Advice_on_finding_a_mentor)과 [멘토가 구체적으로 어떤 가치를 더해줄 수 있는지](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher#So_what_does_a_research_mentor_actually_do_)에 대한 조언을 제공한다. 이 글의 나머지 부분은 대부분 멘토가 없다고 가정하고 쓰되, 적절한 곳에서 멘토를 활용하는 방법을 짚어두겠다.

### 연구 과정 뜯어보기 (Unpacking the Research Process)

나는 연구를 네 개의 구별되는 단계가 순환하는 것으로 생각하면 도움이 된다고 본다. 자세한 내용은 [연구 과정에 대한 내 블로그 글](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand)을 읽어보되, 간단히 정리하면:

- **아이디에이션(Ideation):** 연구 문제나 집중할 일반적인 영역을 고른다.
- **탐색(Exploration):** 아직 구체적인 가설이 없을 수 있다. 어떤 질문을 던져야 하는지 알아내고 그 영역에 대한 더 깊은 직관을 쌓으려는 중이다. 북극성은 정보와 접촉면(surface area)을 얻는 것이다.
- **이해(Understanding):** 구체적인 가설과 그 영역에 대한 어느 정도의 직관적 이해가 생겼을 때 시작된다. 북극성은 그 가설이 참인지 거짓인지 스스로를 납득시키는 것이다.
- **증류(Distillation):** 납득이 됐다면, 북극성은 발견한 것을 세상에 전달할 수 있는 간결하고 엄밀한 진실로 압축하는 것이다. 남을 설득하기에 충분한 실험적 증거를 만들고, 명료하게 글로 쓰고, 공유하라.

이 단계들을 떠받치는 것은 여러 스킬들인데, 얼마나 빨리 적용하고 피드백을 얻을 수 있는지로 나누는 것이 가장 좋다. 우리는 무언가를 하고 피드백을 받으면서 배우므로, 빠른 스킬을 훨씬 더 빨리 익히게 된다. 대략적인 목록과 분류를 아래에 적어둔다.

내 일반적인 조언은 **피드백 루프 순서대로 배우는 것을 우선하라**는 것이다. 좋은 연구 문제를 고르는 안목처럼 느린 스킬이 시작하는 데 필요해 보인다면, 그 스킬이 없다고 스트레스받지 말고 편법을 찾아라(예: 논문에 점진적 확장을 더하기, 멘토에게서 문제를 받기 등).

- 빠른 루프 (수 분~수 시간):
  - 실험 코드 계획하고 작성하기
    - **중간:** 훌륭한 실험 설계하기
    - **중간:** 언제 대충 짠 코드를 쓰고 언제 제대로 된 코드를 쓸지 판단하기
  - 실험 돌리고 디버깅하기
    - **중간/느림:** 미묘한 버그를 발견하고 고치기 (예: 토크나이제이션이 미묘하게 잘못됐다, 하이퍼파라미터 탐색이 충분치 않았다 등)
  - 하나의 실험 결과를 해석하기
    - **중간:** 결과가 당신의 결론을 뒷받침하는지 이해하기
    - **느림:** 결과가 실제로는 주장을 뒷받침하지 않는 미묘한 해석가능성 착시(interpretability illusion) 알아채기
- 중간 루프 (수 일):
  - Mech interp에 대한 개념적 이해 발전시키기
    - **느림:** 자기 자신의 미묘한 혼동을 알아차리고 바로잡기
    - **느림:** 문헌에 대한 깊은 지식 쌓기
  - 막히지 않고 탐색하는 법 알기
  - 결과를 글로 정리하기
    - **느림:** 자신의 작업을 사람들에게 진짜로 명료하게 전달하기
    - **느림:** 자신의 작업이 왜 *흥미로운지*를 사람들에게 전달하기
- 느린 루프 (수 주):
  - 다음에 어떤 실험을 할지 우선순위 정하기
  - 어떤 연구 방향을 계속 밀지, 다른 공략 각도나 다른 프로젝트로 방향을 틀지 판단하기
  - 먼저 프로젝트를 해보지 *않고도* 나쁜 연구 아이디어를 식별하기
- 매우 느린 루프 (수 개월):
  - 좋은 연구 아이디어를 떠올리기. 이것이 "연구 안목"의 핵심이다.

당신의 진행 경로는 단순해야 한다. 먼저 버려도 되는 프로젝트로 탐색과 이해를 떠받치는 빠른/중간 스킬에 집중하라. 그다음 엔드투엔드 프로젝트로 넘어가서 더 깊은 스킬을 의도적으로 연습하고, 아이디에이션과 증류도 함께 연습하라.

### 연구 안목이란 무엇인가? (What is research taste?)

특히 중요하면서도 흐릿한 종류의 스킬을 연구 안목(research taste)이라고 부른다. 나는 이것을 기본적으로, 충분한 연구 경험을 통해 얻게 되는 직관들의 묶음이라고 본다. 좋은 아이디어를 떠올리고, 어떤 아이디어가 유망한지 예측하고, 좋은 연구 방향에 확신을 갖는 것 같은 일을 가능하게 해주는 것들이다. 더 자세한 생각은 [이 주제에 대한 내 글](https://www.alignmentforum.org/posts/Ldrss6o3tiKT6NdMm/my-research-process-understanding-and-cultivating-research)을 참고하라.

대체로 지금은 그냥 무시하고, 아직 안목이 별로 없다는 걸 보완할 방법을 찾고, 빠른~중간 스킬을 배우는 데 집중하라고 말하고 싶다. 그러면 나중에 안목을 배울 훨씬 좋은 기반이 된다. 특히 안목은 멘토가 있으면 훨씬 빨리 배울 수 있으므로, 처음에 멘토가 없다면 다른 것을 우선해야 한다.

하지만 언젠가는 배우고 싶은 것이니, 내내 이것을 염두에 두고 연습하고 교훈을 얻을 기회를 찾는 것이 좋다. 있으면 좋은 것으로 취급하되 스트레스받지 않기를 권한다.

참고로, 여기엔 중요한 함정이 하나 있다. 좋은 안목을 가졌다는 것은 종종 어떤 연구 방향에 대한 자신감과 확신으로 나타난다. 그런데 초심자 연구자들은 나쁜 아이디어에 확신을 갖지 *않는* 능력을 기르기 훨씬 *전에* 이 자신감과 확신을 발달시키는 경우가 많다. 정말 대단할 거라고 확신했던 것을 한두 번 밀어붙였다가 틀렸음을 발견하는 것은 대개 좋은 학습 경험이므로 그리 나쁜 결과는 아니다. 특히 2단계(미니 프로젝트)에서는 더 그렇다. 다만 주의하라.

[^nifk1wb1jum]: 여기서 말하는 건 꽉 채운 한 달, 그러니까 200 작업시간 정도를 뜻한다. 파트타임으로만 할 수 있다면 더 오래 걸려도 괜찮다. 정말 집중하고 있거나 미리 앞서 있다면 더 빨리 넘어가라.

[^ue9pdw6v8rj]: 더 접근하기 쉬운 것을 원한다면, 내 예전 MATS 스칼라 중 한 명은 GPT-5 thinking에게 코딩 연습문제(예: 빈 함수와 좋은 테스트가 있는 Python 스크립트)를 만들게 하는 것을 더 쉬운 진입로로 추천한다.

[^hh6mwdeo4zm]: 이 코딩에 LLM의 도움과 문서/튜토리얼 찾아보기가 잔뜩 필요해도 괜찮다. 이건 암기 시험이 아니다. 핵심은 각 기법의 요체를 친구나 LLM에게 정확히 설명할 수 있는가다.

[^sxyjce3nii]: 참고: 이 커리큘럼은 독립적인 연구를 시작할 수 있게 하는 것을 목표로 한다. 학계 연구실에는 대개 이 정도면 충분하지만, 대부분의 산업계 랩은 엔지니어링 기준이 훨씬 높다. 수백 명의 다른 연구자와 함께 크고 복잡한 코드베이스에서 일해야 하기 때문이다. 다만 그런 스킬은 얻는 데 훨씬 오래 걸린다.

[^kte6u8splw]: 활성값을 수집할 때 프롬프트의 첫 토큰은 제외하는 게 좋다. 이상한 어텐션 싱크(attention sink)이고 노름이 크거나 여러 면에서 변칙적인 경우가 많다.

[^2ob115pcmet]: 함정: 벡터를 더할 때 계수를 여러 개 시도해 보는 것을 잊지 마라. 이건 결정적인 하이퍼파라미터이고, 조종된 모델의 행동은 그 값에 따라 크게 달라진다.

[^1b9r0ass7sd]: Mixture of experts 모델은 파라미터가 많지만 각 토큰마다 일부만 활성화되는데, 해석가능성 연구에는 골칫거리다. 모델이 커지면 더 많고 더 큰 GPU가 필요해서 비싸고 다루기 번거롭다. 가능하면 dense 모델을 선호하라.

[^bzop9pji3nl]: PDF를 다운로드해서 모델에 업로드하거나, 그냥 PDF에서 전체 선택 후 복사해 채팅창에 붙여넣으면 된다. 포맷이 깨진 걸 고칠 필요는 없다. LLM은 이상한 포맷 아티팩트를 무시하는 데 능하다.

[^207k0k5nobb]: repo2txt.com은 GitHub 레포를 하나의 txt 파일로 합쳐주는 유용한 도구다.

[^979wnkvgpa4]: 다른 관점을 원한다면 Open Problems in Mechanistic Interpretability(여러 선도 연구자들이 참여한 최근의 폭넓은 문헌 리뷰)나 Interpretability Dreams(Anthropic, 2년 전)를 확인해 보라.

[^3zw26zes9dx]: 그리고 뒤에서 논의할 이유들로, 지금은 야심찬 리버스 엔지니어링 방향에 대해 훨씬 더 비관적으로 느낀다.

[^7cxhc64szn8]: 이미 다른 분야에서 연구 경력이 있더라도, 기계론적 해석가능성은 충분히 달라서 최소한 일부 본능은 다시 배워야 할 것으로 예상해야 한다. 이 단계는 당신에게도 여전히 매우 유효하다. 다만 더 빨리 배울 수 있기를 바란다.

[^9wj0u0qz3q]: 이 글의 나머지는 연구 학습을 이런 식으로 접근하는 것을 틀로 삼아 쓰였고, 왜 그것이 합리적인 과정이라고 생각하는지를 담고 있다. 당연히 연구를 배우는 유일한 정답은 없다! 내가 무언가를 "실수"라고 비판할 때는, "이런 게 좋은 아이디어가 되는 연구 학습 방식은 존재하지 않는다"가 아니라 "사람들이 이렇게 하는 걸 자주 보는데 그들에게 최적이 아니라고 생각한다"로 해석하라.

