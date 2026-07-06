# PyTorch 스터디

Introduction to PyTorch 실습 스터디 자료입니다. [Learn PyTorch for Deep Learning](https://www.learnpytorch.io)의 챕터 00~05 + 07를 다룹니다.

## 사전지식
다음 내용은 알고 있는 것으로 간주합니다. 만약 모르는 경우 아래 자료를 통해 미리 학습하면 좋습니다.
- Python 기초
  - [CS50's Introduction to Programming with Python](https://www.youtube.com/playlist?list=PLhQjrBD2T3817j24-GogXmWqO5Q5vYy0V)
  - 양이 많으면 Lecture 5, 7, 9는 제외해도 됨
  - 파이썬을 알아도 OOP(Object-Oriented Programming - 객체지향)을 모른다면 [Lecture 8](https://youtu.be/e4fwY9ZsxPw?si=w-4fy77BK_F9_LV1)은 숙지하는게 좋습니다.
  - 참고: [Original CS50](https://www.youtube.com/playlist?list=PLhQjrBD2T380hlTqAU8HfvVepCcjCqTg6)도 좋은 강의입니다. 압축적으로 CS 지식을 배울 수 있습니다.
- shell 환경
  - [The Missing Semester of Your CS Education - Introduction to the Shell](https://missing.csail.mit.edu/2026/course-shell/)
  - 참고: 위 강의록([The Missing Semester of Your CS Education](https://missing.csail.mit.edu))의 다른 내용도 읽어보시면 좋습니다.

## 커리큘럼

| 주차 | 교재 챕터 | 내용 | 과제 |
|---|---|---|---|
| 1 | [00](https://www.learnpytorch.io/00_pytorch_fundamentals/) + [01](https://www.learnpytorch.io/01_pytorch_workflow/) | 텐서 다루기, 학습 루프 전 과정 | [1주차 과제](week1/assignment.ipynb) |
| 2 | [02](https://www.learnpytorch.io/02_pytorch_classification/) | 이진분류, 비선형, 다중분류, 평가 | TBA |
| 3 | [03](https://www.learnpytorch.io/03_pytorch_computer_vision/) | CNN 구조, Conv2d, CV 학습, 평가 | TBA |
| 4 | [04](https://www.learnpytorch.io/04_pytorch_custom_datasets/) + [05](https://www.learnpytorch.io/05_pytorch_going_modular/) + [07(일부)](https://www.learnpytorch.io/07_pytorch_experiment_tracking/) | Custom Datasets, 모듈화, TensorBoard 데모 | TBA |

## 진행 방법

- 각 주차 폴더(`week1/` ~ `week4/`)는 같은 구조입니다.

  ```
  weekN/
    assignment.ipynb        # 과제: 지시문 + 스타터 코드 + TODO
    solution.ipynb          # 발표자/조교용 해답 (과제 마감 후 공개)
  ```

- **발표자**: 교재 내용 + 과제 발표를 준비합니다.
- **참가자**: 발표가 있기 전 내용을 숙지하고 과제를 미리 수행합니다.
- **해답**: `solution.ipynb`은 과제 마감 후에 공개합니다.

## 환경 세팅

### Google Colab (권장)

별도 설치 없이 브라우저에서 바로 실행합니다.

1. 과제 노트북 상단의 **Open in Colab** 버튼을 클릭합니다. (커리큘럼 표의 과제 링크 → GitHub에서 열어도 버튼이 보입니다)
2. `파일 → Drive에 사본 저장`으로 사본을 만들어 작업하세요. 사본이 아니면 변경 내용이 저장되지 않습니다.
3. (선택) GPU 사용: `런타임 → 런타임 유형 변경 → T4 GPU`. 1~2주차는 CPU로 충분하고, 3~4주차는 GPU를 권장합니다.
4. 제출 전 `런타임 → 세션 다시 시작 및 모두 실행`으로 처음부터 끝까지 에러 없이 실행되는지 확인하세요.

참고 사항:

- `torch`, `torchvision`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `tensorboard` 등 필요한 패키지는 Colab에 기본 설치되어 있습니다. 없는 패키지가 필요한 주차는 노트북 안에 `!pip install ...` 셀이 포함되어 있습니다.
- Colab 세션이 초기화되면 노트북이 만든 파일(그래프 `*.png`, 모델 `*.pth` 등)은 사라집니다. **제출용 파일은 좌측 파일 패널에서 미리 다운로드**해 두세요.

### (참고) 로컬 실행

Python 3.10 이상을 권장합니다.

conda 사용 시:

```bash
conda create -n pytorch-study python=3.11 -y
conda activate pytorch-study
pip install torch torchvision matplotlib numpy pandas scikit-learn tensorboard torchinfo tqdm jupyter
```

venv + pip 사용 시:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision matplotlib numpy pandas scikit-learn tensorboard torchinfo tqdm jupyter
```

노트북 실행: `jupyter lab` (또는 `jupyter notebook`)을 실행한 뒤 원하는 `weekN/assignment.ipynb`를 여세요.

#### GPU / device 관련

- **NVIDIA GPU 서버**: CUDA 버전에 맞는 wheel이 필요할 수 있습니다. [pytorch.org/get-started](https://pytorch.org/get-started/locally/)에서 서버의 CUDA 버전에 맞는 설치 명령을 확인하세요.
- **Apple Silicon Mac**: 기본 `pip install torch`로 MPS(GPU) 가속이 지원됩니다.
- GPU가 없어도 이 스터디의 모든 실습은 CPU로 충분히 돌아갑니다.

모든 코드는 **device-agnostic** 컨벤션을 따릅니다:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
# Apple Silicon에서 MPS를 쓰고 싶다면:
# device = "mps" if torch.backends.mps.is_available() else "cpu"
```
