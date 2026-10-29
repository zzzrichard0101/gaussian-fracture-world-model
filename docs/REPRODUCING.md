# 환경과 재현 방법

## GPU 학습을 포함한 Gaussian 표현 pilot

[제한된 beam·표현 감사·모델·수치·한계](REPRESENTATION_PILOT.md). 기존 GaussianFluent CUDA renderer와 Torch 2.0 환경을 사용한다. 아래 주 실행과 두 대조는 CUDA가 필요하다. 첫 CPU 단위 테스트는 CUDA renderer를 import하지 않는다.

```powershell
.\scripts\run-wsl.ps1 python scripts/representation_pilot_evaluate.py
.\scripts\run-wsl.ps1 python scripts/representation_action_controls.py
.\scripts\run-wsl.ps1 python scripts/representation_linear_controls.py
.\scripts\run-wsl.ps1 python scripts/report_representation_pilot.py
.\scripts\run-wsl.ps1 test -q tests/test_representation_pilot.py
.\scripts\run-wsl.ps1 python scripts/publish_results.py
.\scripts\run-wsl.ps1 python scripts/validate_public_docs.py
```

`config/representation-pilot.json`에 물리·분리 seed·표현·MLP·학습 예산·감사 허용치를 저장했다. 주 예측 고정 이후 행동·선형 최적화의 사후 대조를 별도로 실행한다. 가중치와 전체 NPZ는 로컬 `artifacts/representation-pilot`에 남고, 선택한 입력·점수·감사·영상은 공개한다. 전체 실행을 다시 하면 해당 폴더를 덮어쓰므로 별도 보존 후 비교한다. 실제 독립 Gaussian 재구성이나 영상 상태 추정은 포함하지 않는다. 전체 테스트는 276개를 통과했다.

## 전체 물리 예산을 맞춘 제안 비교

[비용 정의·방법·해석](SPATIAL_BUDGET.md). 같은 8개 공개 개발 이력을 사용하며 외부 데이터는 필요하지 않다.

```powershell
.\scripts\run-wsl.ps1 python scripts/spatial_budget_evaluate.py
.\scripts\run-wsl.ps1 python scripts/report_spatial_budget.py
.\scripts\run-wsl.ps1 test -q tests/test_spatial_budget.py
.\scripts\run-wsl.ps1 python scripts/publish_results.py
.\scripts\run-wsl.ps1 python scripts/validate_public_docs.py
```

`config/spatial-budget.json`에 두 예산·독립 seed·방법·혼합 비율·회계 규칙을 고정했다. 방법마다 새 물리 bank를 사용하고 pilot과 적응도 비용으로 계산한다. 작은 예산 결과는 최대 예산 흐름의 앞부분이며 setup 전체를 청구한다. 시간은 최대 예산의 fresh-bank 실행만 측정한다. 공개 `spatial-budget-*` JSON에서 추출 수·범위 밖 추출·실제 물리 평가 수·예산 잔액을 따로 확인할 수 있다. 실행 파일은 이전 산출물을 덮어쓰므로 비교 시 별도 보존한다.

## 반응 기반 제안분포 개발 실험

[세부 조건과 비용](SPATIAL_TRANSPORT.md). 기존 WSL CPU NumPy/SciPy/Matplotlib 환경을 사용한다. 기존 공개 관측 중 8개 개발 이력을 재사용하며 새로운 폐루프 실행은 아니다.

```powershell
.\scripts\run-wsl.ps1 python scripts/spatial_transport_evaluate.py
.\scripts\run-wsl.ps1 python scripts/report_spatial_transport.py
.\scripts\run-wsl.ps1 test -q tests/test_spatial_transport.py
.\scripts\run-wsl.ps1 python scripts/publish_results.py
.\scripts\run-wsl.ps1 python scripts/validate_public_docs.py
```

`config/spatial-transport.json`에 독립 pilot/본 샘플 seed·8개 이력 선택·네 제안·수렴 기준을 저장했다. 공개 `spatial-transport-*` JSON은 고정 예측·개별 평가·적응 비용·관측 출처 해시를 포함한다. 실행 시간은 머신에 따라 달라지며 새 실행은 이전 산출물을 덮어쓰므로 비교할 때 실행 폴더를 별도 보존한다. 전체 테스트는 264개를 통과했다. 위험 증거 재생은 아래의 기존 경로를 사용하며 이번 제안 변경의 성과로 세지 않는다.

## GPU 없이 최신 검증기 재현

최신 연속 영역 검사와 공개 관측 재생에는 CUDA·Houdini·Unreal·촬영 데이터가 필요 없다. 저장소 루트에서 CPU Python 환경으로 실행할 수 있다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/test-core.txt
python -m pytest -q
export WM_PROJECT_ROOT="$PWD"
export WM_RESEARCH_ROOT="$PWD/.runtime/reproduction"
PYTHONPATH="$PWD" python scripts/benchmark_risk_reuse.py \
  --inputs docs/results/risk-reuse-inputs.json --repetitions 2
```

Windows에서는 `source` 대신 `.venv\Scripts\Activate.ps1`을 사용하고 환경 변수는 PowerShell 문법으로 지정한다. 위 재생 입력에는 공개 관측·알려진 상수·이전 판정만 있으며 숨은 정답은 없다. 과거 개인 실행 폴더를 요구하지 않는다. 재생은 같은 행동·상하한·위반 지지점·선택 증거 구조를 확인하고 새 증거를 저장한다. 예측 성능을 새로 평가하는 실험은 아니다. 전체 테스트에는 NumPy·SciPy·h5py·OpenCV가 필요하지만 독립 증거 재검사 CLI 자체는 표준 라이브러리만 사용한다.

새로운 [관측 예산 비교](OBSERVATION_BUDGET.md)도 같은 CPU 환경에서 실행한다. 과거 개인 실행 파일 없이 설정에서 24개 개발 상태를 재생성하고, 새 잡음에서 2·4·8개 관측을 비교한다.

```bash
PYTHONPATH="$PWD" python scripts/observation_budget_evaluate.py
# 위 명령이 출력한 OUTPUT 경로를 RUN에 지정한다.
python -m pip install matplotlib
PYTHONPATH="$PWD" python scripts/report_observation_budget.py --run RUN
```

## 예측 적분 안정화 재현

공개된 [관측식 기반 적분](RESPONSE_QUADRATURE.md) 입력으로 과거 개인 실행 폴더 없이 재생한다. 같은 입력과 관측에서 적분만 비교하고, 모든 예측을 고정한 뒤 합성 정답을 재생성한다.

```bash
PYTHONPATH="$PWD" python scripts/response_quadrature_evaluate.py \
  --inputs docs/results/guided-quadrature-inputs.json
PYTHONPATH="$PWD" python scripts/response_quadrature_evaluate.py \
  --config config/guided-quadrature.json \
  --inputs docs/results/guided-quadrature-inputs.json
PYTHONPATH="$PWD" python scripts/report_response_quadrature.py --run GUIDED_RUN --coordinate-run INITIAL_RUN
```

그림 보고서는 앞서 설치한 matplotlib를 사용한다. 적분과 기존 스크립트 의존성에는 NumPy·SciPy·h5py가 필요하며 GPU는 사용하지 않는다. 환경과 환경 변수는 위 CPU 안내처럼 준비한다.

## 목표 미래를 고려한 입력 선택

[폐루프 pilot](TASK_AWARE_PILOT.md)은 설정에서 새 물리 상태와 잡음을 생성하므로 이전 개인 실행 데이터가 필요 없다. 준비 명령은 공개 이전 보고서 사본을 무결성 점검의 참조용으로 배치하며 기존 보고서는 유지한다.

```bash
python scripts/prepare_task_references.py
PYTHONPATH="$PWD" python scripts/task_aware_pilot.py
# 위 OUTPUT 경로를 RUN에 지정한다.
PYTHONPATH="$PWD" python scripts/report_task_aware_pilot.py --run RUN
```

관측 횟수는 2·4·8회의 중첩 prefix다. 모든 정책은 같은 누적 손상 한도와 연속 위험 검증기를 사용하고, 최종 예측은 같은 guided 적분기로 계산한다. 시뮬레이터가 새 관측을 제공하기 전에 결정과 증거를 고정한다. 실제 입력·관측·선택 점수·원본 증거는 실행 폴더에 보관하고, 공개 저장소에는 설정·조건별 수치·선택 정밀도 검사·그림을 게시한다.

## 단순 교대 규칙과 같은 데이터에서 비교

이전 pilot의 공개 입력·관측과 설정을 참조하므로 개인 실행 폴더가 필요 없다. [교대 기준선](ALTERNATING_FRONTIER.md)은 같은 상태·잡음을 생성하고 미래 점수 정책의 정확한 재생을 검사한다.

```bash
PYTHONPATH="$PWD" python scripts/alternating_frontier_evaluate.py
# 위 OUTPUT 경로를 RUN에 지정한다.
PYTHONPATH="$PWD" python scripts/report_alternating_frontier.py --run RUN
```

세 정책의 실행 위치를 조건마다 회전시키고, 모든 실행 증거를 새 cold 검사기로 대조한다. 무하중 우선이 주 비교이고 큰 입력 우선은 미리 지정한 순서 민감도 검사다. 이전 상태를 재사용하는 개발 비교이며 새 독립 일반화 시험은 아니다.

## 기존 환경 분리

| 환경 | 용도 | 검증한 버전 |
| --- | --- | --- |
| `gaussianfluent` | 물리·Gaussian·최신 파이프라인 | Python 3.9.25, PyTorch 2.0.0+cu118, SciPy 1.12.0, Warp 0.10.1 |
| `world-model` | 선택적 V-JEPA·작은 학습 모듈 | Python 3.11.12, PyTorch 2.7.1+cu128 |
| CPU 핵심 테스트 | GPU·가중치 없이 법칙/계약 검사 | Python 3.11, `requirements/test-core.txt` |

최신 `joint-damage-material`, `response-diversity`, `active-probe`에는 첫 환경만 필요하다. V-JEPA가 없어도 실행한다. 검증한 장치는 RTX 4090 24GB, 시스템 RAM 96GB이며 WSL에서 약 46GiB를 사용 가능했다. 다른 장치·배포판은 별도 검증이 필요하다.

## Ubuntu WSL2에서 시작

`uv`, CUDA Toolkit 11.8(`/usr/local/cuda-11.8`), C++ 빌드 도구, Git, `ffmpeg`를 준비한다.

```bash
git clone https://github.com/zzzrichard0101/gaussian-fracture-world-model.git
cd gaussian-fracture-world-model
bash scripts/wsl.sh bootstrap
bash scripts/wsl.sh doctor
bash scripts/wsl.sh test -q
```

환경·외부 코드·실행 결과의 기본 위치는 `$HOME/research/world-model-research`다. 다른 위치를 쓰려면 bootstrap 이전부터 모든 명령에 동일하게 설정한다.

```bash
export WM_RESEARCH_ROOT="$HOME/research/gaussian-fracture-world-model"
```

프로젝트 코드는 현재 저장소 폴더를 사용한다. CUDA 경로는 실행 프로세스에 설정한다. `requirements/*.lock.txt`는 원래 환경 기록으로 editable 설치의 원래 절대 경로도 들어 있다. 새 환경은 lock 파일을 직접 재설치하기보다 `bootstrap`으로 시작한다.

## 실험·새 충격·독립 감사

각 실행이 출력하는 `OUTPUT=...`를 아래 `RUN` 대신 사용한다. 개발 머신의 과거 실행 폴더는 공개 저장소에 포함하지 않는다.

```bash
bash scripts/wsl.sh cohesive-probe
bash scripts/wsl.sh joint-damage-material
bash scripts/wsl.sh response-diversity
bash scripts/wsl.sh active-probe

bash scripts/wsl.sh python scripts/active_probe_branch.py \
  --run RUN --impact 0.275 --impact-time 0.064

bash scripts/wsl.sh python scripts/audit_active_probe.py \
  --run RUN \
  --branch case-000-uniform49-response8-guarded-dv-0.275-t-0.064
```

`active-probe`의 순서는 생성 → 관측 후보 → 가상 충격 → 상태 압축 → 관측 정책 고정 → 새 관측 공개 → 후보 갱신 → 미래 계산 → 평가/영상이다. 중간 실패는 `<단계>.log`에 남는다. `--stage <단계> --run RUN`으로 조사할 수 있다. 일부 단계는 기존 출력 폴더를 덮어쓰지 않고 중단한다. 완전한 새 실행은 `active-probe`를 다시 호출해 새 폴더에 만든다.

과거 fixture와 파라미터가 겹치는지 확인하는 평가는 버전 관리된 설정에서 과거 값을 계산한다. 과거 머신의 실행 파일을 요구하지 않는다. 추가 관측을 제공하는 신뢰된 합성 환경만 숨은 상태를 읽는다. 정책과 갱신 단계는 그 정답을 직접 읽지 않는다. 깊이는 Gaussian 중심의 alpha 정규화 값으로 실물 표면 센서 모델과 다르다.

## 1차원 DIRECT 비교

두 fixture를 생성하고 각각 출력된 경로를 넘긴다.

```bash
bash scripts/wsl.sh history-refinement
bash scripts/wsl.sh history-refinement --config config/history-refinement-replication.json
bash scripts/wsl.sh direct-benchmark --sources FIRST_RUN SECOND_RUN
```

같은 시험 관측에 강한 기준선을 추가하는 비교이며 새 보류 일반화 시험이 아니다.

## 선택적 V-JEPA 실험

```bash
bash scripts/world-model.sh bootstrap
bash scripts/world-model.sh test -q
```

가중치는 약 5.1GB이며 별도로 내려받는다. 과거 `world-model`, `action-adapter`, `belief-state` 설정에는 선행 데이터의 원래 경로가 기록돼 있다. 선행 데이터를 생성하고 자신의 경로로 바꿔야 한다. 이 오래된 학습 실험의 원본 관측·모든 가중치를 배포하는 것은 아니다. 연결 순서는 [상세 실험 기록](EXPERIMENT_LOG.md)에 있다.

## 계산 예산 해석

- 49회는 관측 목적함수 평가 예산이다. 전체 물리 호출 수와 다르며 중복/cache DIRECT callback도 포함한다.
- 압축용 가상 충격은 후보 집합당 최대 147개 궤적을 추가한다. 이후 8개만 계산해도 앞단 비용은 남는다.
- 능동 관측은 7개 후보 충격을 비교한다. 전체/8개 조건을 분리해서 보고하며 평가에서는 여러 정책의 반사실적 결과도 계산한다.
- 공유 cache·batching·프로세스 시작·영상 생성 때문에 시간 범위를 명시한다. 전체 속도 향상은 입증하지 않았다.

구체적 시간·분기 수는 JSON의 `budget`, `pool_inference`, `inference`에 있다. 작은 장면의 GPU allocator 수치를 장치 전체 VRAM이나 일반 3D 요구량으로 해석하지 않는다.

## 공개 결과 갱신

[연속 prior·조건부 구간 판정](SPATIAL_CONTINUOUS.md)은 아래 명령으로 재현한다. QMC 재추정은 이전 공개 이력을 사용하고 새 행동을 실행하지 않는다. 구간 판정은 고정 증분 모형의 결과다.

```powershell
.\scripts\run-wsl.ps1 python scripts/spatial_continuous_evaluate.py
.\scripts\run-wsl.ps1 python scripts/report_spatial_continuous.py
.\scripts\run-wsl.ps1 python scripts/spatial_risk_evaluate.py
.\scripts\run-wsl.ps1 python scripts/spatial_risk_evaluate.py --followup-cells 31
.\scripts\run-wsl.ps1 python scripts/spatial_risk_evaluate.py --replay docs/results/spatial-risk-proofs.json
```

31개 상자 검사는 최초 7개 상자 결과를 본 뒤의 별도 예산 민감도 검사다. 넓은 영역은 두 경우 모두 보류했으며, 좁은 영역의 허용을 넓은 prior에서 얻은 성과로 바꾸지 않는다.

[공간 관측·유한 후보 추정](SPATIAL_OBSERVATION.md)과 사후 반례 진단은 아래 순서로 실행한다. 공개 결정 재생만 실행할 때는 원본 raw run이나 참 상태 파일이 필요 없다.

```powershell
.\scripts\run-wsl.ps1 python scripts/spatial_observation_evaluate.py
.\scripts\run-wsl.ps1 python scripts/spatial_policy_replay.py
.\scripts\run-wsl.ps1 python scripts/spatial_failure_search.py
.\scripts\run-wsl.ps1 test -q tests/test_spatial_observation.py
```

29개 후보의 균일 **이산 prior**이며 연속 prior의 격자 수렴 실험이 아니다. 목록 밖 stress·예측 기권·실제 손상 위반도 함께 저장한다. 반례 검색은 실패 조건을 보고 선택한 사후 분석이며 원래 실행을 바꾸지 않는다.

[세 구역 공간 순방향 모형](SPATIAL_FIXTURE.md)은 다음과 같이 실행한다. 다섯 설계 사례를 세 변위 증분에서 반복하며 준비 하중·평형·외부 일·이력 수렴을 검사한다. 합성 순방향 단계로 영상·추정·공간 위험 인증을 포함하지 않는다.

```powershell
.\scripts\run-wsl.ps1 python scripts/spatial_fixture_evaluate.py
.\scripts\run-wsl.ps1 test -q tests/test_spatial_fixture.py
```

외부 평가를 반영한 새 [식별 가능성 진단](PROBE_IDENTIFIABILITY.md)은 다음 명령으로 실행한다. 기존 WSL 의존성만 쓰며 GPU·외부 엔진·촬영 데이터가 필요 없다. 두 상태만 고려한 계산과 전체 가능 집합의 위험 검사를 구분한다.

```powershell
.\scripts\run-wsl.ps1 python scripts/probe_identifiability_evaluate.py
.\scripts\run-wsl.ps1 python scripts/probe_identifiability_evaluate.py --replay-proofs docs/results/probe-identifiability-proofs.json
.\scripts\run-wsl.ps1 test -q tests/test_probe_identifiability.py
```

공개 증거 재검사는 원본 raw run 없이 가능하다. Gaussian 이진 구분 점수는 별도의 확률 가정을 쓰는 oracle 진단이며 기존 정책 캠페인의 실측 성능이 아니다.

```bash
bash scripts/wsl.sh python scripts/publish_results.py
```

명시된 보고서·그림·영상만 `artifacts/`에서 `docs/`로 복사하고 표를 만든다. raw 관측·가중치·에셋·전체 실행 폴더는 복사하지 않는다. JSON에서 개발 머신의 연구 루트 문자열만 바꾸고 수치는 유지한다. 변환 전후 해시는 `docs/results/publication-manifest.json`에 있다.

같은 설정의 공개용 재실행은 재현 점검이며 새로운 독립 보류 시험으로 세지 않는다.
