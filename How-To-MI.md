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

