# 관측으로 보정하는 Gaussian 파괴 디지털트윈

첫 충격의 관측으로 물리 상태를 보정하고, 그 상태에서 두 번째 충격의 결과를 예측하는 연구용 프로젝트다.

Windows에서는 이 폴더의 코드·설정을 관리하고, CUDA 계산과 환경은 기존 Ubuntu WSL2에서 실행한다. Python 환경과 외부 저장소는 `/home/richard/research/world-model-research` 아래에 격리한다. 기존 Python·CUDA 기본 설정은 변경하지 않는다.

```powershell
.\scripts\run-wsl.ps1 bootstrap
.\scripts\run-wsl.ps1 doctor
.\scripts\run-wsl.ps1 smoke
.\scripts\run-wsl.ps1 fetch-asset
.\scripts\run-wsl.ps1 asset-check
```

또는 WSL에서 이 프로젝트 폴더로 이동한 뒤 실행한다.

```bash
bash scripts/wsl.sh bootstrap
bash scripts/wsl.sh doctor
bash scripts/wsl.sh smoke
bash scripts/wsl.sh fetch-asset
bash scripts/wsl.sh asset-check
```

환경은 Python 3.9, PyTorch 2.0.0 CUDA 11.8, Warp 0.10.1, GaussianFluent의 고정된 커밋을 출발점으로 사용한다. `config/upstream.json`은 원본 커밋, `requirements/sim.txt`는 직접 의존성, `requirements/sim.lock.txt`는 실제 설치 결과를 기록한다.

GPU 검사 결과는 `artifacts/`에 저장한다. 대형 데이터·실행 출력·외부 코드는 Git에 추가하지 않는다. 원본 Gaussian Splatting 및 파생 코드·자산의 이용 조건은 각 원본 저장소를 따른다.

환경 구성, 첫 충격 실행기, 지원 범위 내 상태 저장·복원과 축약 접합면의 관측 기반 상태 추정 기준선을 구현했다. 현재 관측 → 상태 후보 → 새 충격 예측은 한 접합면·축방향 운동의 합성 장면에서 작동한다. V-JEPA 2의 사전학습 인코더·예측기를 실제 연결한 첫 실험도 완료했지만, 단순한 latent 후보 선택의 예측 개선은 확인하지 못했다. 일반 3D 균열·파편 분리의 정확도와 새로운 연구 기여는 아직 검증하지 않았다.

2026-10-27 검증 결과: PyTorch/Warp/Taichi 모두 RTX 4090 CUDA 실행에 성공했다. Gaussian 렌더러의 순전파·역전파와 이웃 검색, 216개 입자의 MPM 자유낙하 100스텝, Taichi CUDA 커널과 입자 채우기 모듈 import를 확인했다. 공개 수박 에셋의 1,381,100개 Gaussian을 512×512로 렌더링하고 이미지를 확인했다. 결과는 `artifacts/smoke.json`, `artifacts/asset-check.json`, `artifacts/watermelon-preview.png`에 있다.

`config/assets.json`은 공개 에셋의 Hugging Face revision을 고정한다. 다운로드는 수박 에셋 약 345MB만 수행하며, 파일별 SHA256은 `artifacts/asset-watermelon.json`에 기록한다. 대형 에셋·캐시는 WSL의 전용 연구 폴더에 둔다.

실행 스크립트는 CUDA 11.8과 WSL의 `/usr/lib/wsl/lib` 드라이버를 해당 프로세스에만 지정한다. 배포판에 설치된 Linux CUDA 드라이버와 구형 Warp의 `libcuda.so` 로딩 충돌을 피하기 위한 설정이다. simple-knn의 editable 설치 경로도 해당 프로세스의 PYTHONPATH에 추가한다. 기존 시스템 라이브러리·드라이버·WSL 메모리 설정은 수정하지 않았다.

원본 요구사항의 중복 OpenCV 패키지는 headless 패키지 하나로 통일했고, 소스에서 사용하는 누락 의존성 Open3D·scikit-learn 등을 직접 명시했다. 원본 저장소는 수정하지 않았다. `sim.lock.txt`는 현재 환경의 전체 버전 기록이며 editable 경로는 이 머신의 절대경로다. 환경을 새로 만드는 진입점은 `bootstrap`이다.

다음 구현 순서는 `IMPLEMENTATION.md`에 정리했다.

논문 기여 후보와 선행연구 중복 점검은 `RESEARCH_DIRECTION.md`에 있다. 현재 저장·복원 결과를 새로운 역학 발견이나 기존 방법 대비 성능 향상으로 해석하지 않는다. 새 방법 구현 전 판단 기준은 `config/research-protocol.json`에 기록했다.

첫 충격 실험은 다음 명령으로 실행한다. 기본 설정은 `config/first-impact.json`이다.

```powershell
.\scripts\run-wsl.ps1 test -q
.\scripts\run-wsl.ps1 first-impact
.\scripts\run-wsl.ps1 first-impact --beta 0.1
.\scripts\run-wsl.ps1 first-impact --beta 0.1 --impact-speed 10
.\scripts\run-wsl.ps1 first-impact --impact-speed 10
.\scripts\run-wsl.ps1 impact-report
```

`first-impact --frames 2`는 짧은 동작 확인용이다. 네 조건 비교 보고서는 위의 30프레임 실험을 모두 실행한 뒤 생성한다. CUDA 계산은 같은 GPU에서 순서대로 실행한다.

기본 실행기는 64³ 격자, 물리 입자 44,644개, 렌더링 Gaussian 1,379,914개를 사용한다. 전체 Gaussian의 점유 셀마다 최대 4개를 고정 seed로 선택하고, 각 셀의 부피를 선택된 물리 입자에 나눈다. 렌더링 Gaussian은 가장 가까운 물리 입자에 연결해 위치와 공분산을 갱신한다. 바닥 체크무늬는 시각적 참조용이며 물리 입자가 아니다.

2026-10-27의 네 조건은 각각 0.3초·3,000서브스텝을 완료했다. 전처리·렌더링·영상 저장을 포함한 측정 구간은 약 5.2초였으며 Python 프로세스 시작 시간은 제외된다. 장치 전체 GPU 메모리 표본 최댓값은 약 4,014MiB로, 다른 프로세스의 약 2,656MiB를 포함한다. PyTorch만의 최대 할당량은 약 680MiB이며 Warp 할당량을 포함하지 않는다. 이 수치를 원본 고해상도 실험이나 학습의 메모리 요구량으로 일반화하지 않는다.

출력은 다음과 같다.

- `artifacts/impact-comparison.mp4`: 네 조건을 함께 보는 영상, 약 6.67배 느린 재생.
- `artifacts/impact-comparison.png`: 질량중심·강체 정렬 후 형상 변화·내부 이력 변화 그래프.
- `artifacts/impact-comparison.json`: 네 조건의 수치와 원본 실행 폴더.
- `artifacts/first-impact-variants/<설정해시>/`: 조건별 영상·manifest·진단 수치. 같은 설정을 다시 실행하면 이 미리보기는 갱신된다.
- WSL `runs/<실행시각>-<설정해시>/`: 매 실행을 별도 보존. 프레임 PNG, HDF5 위치·속도·Jp 궤적, 초기 입자·부피·Gaussian 연결, 설정, 입력 PLY 및 로컬 코드 해시, 원본 커밋을 저장한다.

원본 데모와 차이가 있다. 기본 네 조건 비교는 APIC와 균일 beta를 사용했다. 원본 저장소는 수정하지 않았다. 배경 에셋·추가 조명은 제외했다. 색은 DC RGB로 고정하며, Jp에 따라 Gaussian을 숨기지 않는다. 원본 FLIP의 0/0 문제를 재현·보정한 `guarded-flip`과 색 기반 물성 분포를 선택적으로 사용하는 후속 실행기도 아래에 설명한다.

기본 APIC 비교에서 beta=1에서는 반발이, beta=0.1에서는 큰 눌림·퍼짐이 관찰된다. 이 결과를 정확한 취성 파괴라고 부르지 않는다. 가까운 물리 입자에 단일 연결하는 방식은 시각적 틈을 만들 수 있고, 기본 trial F는 소성 변형 이후의 기하 이력을 충분히 반영하지 못한다. 부피는 Gaussian 중심이 차지한 셀의 근사이며 닫힌 고체 내부 복원이 아니다. HDF5 궤적도 재개 가능한 전체 체크포인트가 아니다.

변형 이력을 누적하는 후속 파이프라인은 한 명령으로 실행한다.

```powershell
.\scripts\run-wsl.ps1 stateful-probe
```

`config/stateful-impact.json`으로 물리 실행 → 입자 연결 분석 → 동일 궤적의 세 방식 재생을 순서대로 수행한다. 결과는 `artifacts/stateful-probe/`에 모인다. `transfer-replay.mp4`와 `transfer-comparison.png`는 왼쪽부터 trial F, 누적 기하 변형, 물리 입자 중심이다. `appearance-vs-particles.mp4`, `connectivity.png`, `separation.json`은 렌더링 외형과 독립적으로 입자 분리를 확인한다. `report.json`은 원본 실행 폴더와 검증 결과를 기록한다.

이 프리셋은 96³ 격자, 물리 입자 71,627개, 약 138만 Gaussian, guarded FLIP, 색으로 선택한 영역의 beta=10, 나머지 beta=0.1을 사용한다. 어두운 영역 선택은 원본 코드의 색 기준을 옮긴 것으로, 실제 씨앗의 의미 분할 정답은 아니다. 원본의 beta=3e9를 적용한 별도 64³ 실험은 충돌 직후 frame 7의 Jp가 NaN이 되어 중단됐다. 프리셋의 beta=10은 안정성을 확인한 실험값이며 원본 수치의 재현이나 실측 물성 보정이 아니다.

`guarded-flip`은 격자 후처리에서 정확히 질량이 0인 노드에만 1e-30의 분모를 주어 원본 g2p의 0/0을 방지한다. 양의 질량·운동량·충돌 속도는 바꾸지 않으며, 다음 스텝의 zero_grid가 임시 값을 지운다. 정확히 0인 보간 가중치를 만드는 단일 입자 GPU 테스트에서 원본 NaN을 재현하고, 보정 후 중력 운동을 검증했다. 다른 불안정성까지 해결했다고 가정하지 않는다.

누적 기하 변형 G는 매 서브스텝에서 `A = F_trial(new) inverse(F_elastic(after return))`, `G(new) = A G(old)`로 갱신한다. 원본 g2p의 `F_trial(new) = (I + dt grad(v)) F_elastic` 관계에서 운동학적 증분을 얻는다. 소성 리턴으로 바뀌는 elastic/trial F를 원래 Gaussian의 전체 변형처럼 사용하는 문제를 줄이기 위한 로컬 근사다. 이는 새로운 파괴 법칙이나 검증된 균열 표현이 아니다. G와 trial F를 프레임별 HDF5에 기록하고, 같은 위치·속도 궤적에서 두 전달 방식만 달리해 재생한다. 초기 프레임 일치와 Gaussian-입자 연결 복원 일치를 검사한다.

입자 연결 분석은 반경 {1, 1.5, 2}×dx의 거리 그래프를 사용한다. 질량 1% 이상인 그룹을 입자 ID의 질량 IoU로 추적하고, 관측 시점에서 0.04초 이상 유지되며 상대 속력이 0.1을 넘는 두 그룹을 후보로 삼는다. 같은 프레임에서 두 반경 이상이 동의해야 최종 후보로 표시한다. 접촉 중인 파편의 균열은 이 방법으로 구별할 수 없으며, 공간적 분리 후보가 물리적으로 정확한 파괴의 증명도 아니다. 기존 APIC 눌림 조건과 새 프리셋에서는 지속적으로 분리된 큰 파편을 확인하지 못했다.

현재 프리셋의 실제 실행은 0.3초·약 3,139스텝을 완료했고, 기록·렌더링을 포함한 물리 실행 측정 구간은 약 9.1초였다. 장치 전체 메모리 표본 최댓값은 4,045MiB로 다른 프로세스를 포함한다. 기본 64³ 비교와는 입자 수·물성·전달법·부피 근사가 달라 해상도 수렴 실험으로 해석하지 않는다.

분석만 다시 실행할 때는 원본 실행 폴더를 지정한다. 이 명령은 물리 상태 재개 기능이 아니다.

```powershell
.\scripts\run-wsl.ps1 stateful-probe --run /home/richard/research/world-model-research/runs/20261027T111404773824Z-496d702c
```

각 단계는 `first-impact --config config/stateful-impact.json`, `analyze-separation --run <WSL실행폴더>`, `replay-transfer --run <WSL실행폴더>`로도 실행할 수 있다. 테스트에는 실제 분리를 지정한 두 블록 양성 대조군과 반경별 판정 시점이 다른 음성 대조군도 포함한다. 이 대조군은 분석기의 작동을 검증하며 파괴 물리를 검증하지 않는다.

체크포인트의 저장·재개 및 비교 검증은 다음 명령으로 실행한다.

```powershell
.\scripts\run-wsl.ps1 first-impact --config config/stateful-impact.json --checkpoint-frame 10
.\scripts\run-wsl.ps1 checkpoint-probe
```

첫 명령은 출력 폴더와 `CHECKPOINT=.../checkpoints/frame-00010.h5`를 표시한다. `--checkpoint-frame`을 여러 번 지정해 여러 프레임을 보존할 수 있다. 체크포인트는 완성된 프레임 경계에서 저장하며 기존 파일을 덮어쓰지 않는다. `trajectory.h5`는 분석용 궤적이고, 실제 재개용 파일은 `checkpoints/frame-*.h5`다.

현재 스키마 2 파일을 별도 프로세스로 재개하는 예시는 다음과 같다. 아래 파일은 `history-pilot`에서 생성한 관측 종료 상태이며 예정된 재충격도 이어 실행한다. `--until-frame`은 추가 프레임 수가 아닌 최종 절대 프레임 번호다.

```powershell
.\scripts\run-wsl.ps1 resume-impact --checkpoint /home/richard/research/world-model-research/runs/20261027T122702474949Z-history-pilot/history-00/observation-end.h5 --until-frame 40
```

재개 결과는 새 WSL 실행 폴더에 보존한다. `restore-audit.json`, 이어진 `trajectory.h5`, 프레임 PNG와 `resumed.mp4`, 다시 재개할 수 있는 `checkpoint-final.h5`가 생성된다. 외부 PLY나 Hugging Face 에셋은 다시 읽지 않는다. 실행 환경과 고정된 원본 코드는 설치되어 있어야 한다.

현재 HDF5 체크포인트 스키마 2는 다음 데이터를 포함한다. 초기 저장·복원 비교 수치는 스키마 1 당시의 기록이며 파일을 보존했다. 스키마 2는 하중 일정·적용 로그와 관련 코드 지문을 추가했고, 현재 로더는 과거 스키마 1 파일을 거부한다. 기존 파일을 덮어쓰거나 호환성 검사를 우회하지 않는다. 새 파일은 위의 실행 명령으로 생성한다.

- MPM 상태 배열 17개: 위치·속도, elastic/trial F, C, Jp, 응력, 공분산·회전, 부피·질량·밀도·선택 마스크, 격자 배열.
- 재료 배열 8개와 모든 모델 스칼라: E·nu·mu·lambda·kappa·beta·재료 종류·항복응력 등.
- 누적 기하 변형 G와 유효성 플래그, Gaussian 외형·공분산·색·불투명도·연결·초기 좌표와 선택 정보.
- 시간·프레임·전역 스텝, 경계 커널 순서와 모든 경계 파라미터, 설정·원본 에셋 provenance.
- Python·NumPy·Torch CPU/CUDA 난수 상태. 현재 지원하는 Warp 스텝에는 확률 커널이 없다.
- 선언형 충격 일정과 적용된 사건의 ID·프레임·충격량·운동에너지 변화 로그. 같은 충격을 복원 후 두 번 적용하거나 건너뛰지 않도록 프레임과 적용 로그를 검사한다.

파일 내부 데이터와 메타데이터의 SHA256을 검사하고, 물리 코드·주요 런타임 버전이 달라지거나 필드가 빠지면 복원을 거부한다. 불완전한 임시 파일을 완성된 체크포인트로 취급하지 않는다. GPU에 복사한 직후 38개 상태·표현 배열이 파일과 정확히 일치하는지 확인하고, 스칼라·경계·시간과 난수 복원도 검사한다. 기록된 데이터 파일은 이 설정에서 약 70.5MB였다.

현재 지원 범위는 단일 GPU의 정적 바닥·경계 상자, APIC 또는 guarded FLIP, 아래에 설명한 선언형 순간 충격량이다. 임의의 Python 콜백, 이동 경계, 원본 솔버의 미지원 외력·속도 수정자는 저장 단계에서 거부한다. 이를 조용히 누락한 채 완전한 체크포인트라고 부르지 않는다.

`checkpoint-probe`는 0.1초(frame 10)에서 저장한 파일을 두 개의 새 Python 프로세스에서 복원해 0.3초(frame 30)까지 실행한다. 비교 기준은 저장 후에도 계속 실행한 연속 궤적이다. 세 번째 새 프로세스는 위치·속도·물성을 유지하면서 내부 이력을 초기화하는 대조군이다. 허용 오차는 실행 전 `config/checkpoint-validation.json`에 정했다. 결과는 `artifacts/checkpoint-probe/`의 영상·그래프·`report.json`에 있다.

2026-10-27 검증 결과:

| 비교 | 검증 구간 최대 위치 RMS / dx | 최대 Jp 상대 RMS | 최종 영상 MAE (0–255) |
| --- | ---: | ---: | ---: |
| 전체 복원 A vs 연속 실행 | 0.02476 | 0.06217 | 0.03033 |
| 전체 복원 B vs 연속 실행 | 0.02427 | 0.07190 | 0.02355 |
| 복원 A vs B 반복 기준선 | 0.02114 | 0.05173 | 0.02026 |
| 내부 이력 초기화 vs 연속 실행 | 5.85014 | 0.88390 | 3.79359 |

복원 직후에는 상태가 정확히 일치했다. 이후의 비트 단위 궤적 동일성을 주장하지 않으며, GPU atomic 연산의 변동을 포함한 사전 허용 범위를 통과했다. 두 번의 복원 반복은 작은 변동 기준선이며 통계적 신뢰구간은 아니다. 내부 이력 초기화 대조군은 이 설정에서 이력 보존이 후속 동작에 영향을 준다는 증거이며 새로운 연구 기여나 실제 파괴 정확도의 검증은 아니다.

초기 저장·복원 단계에서 테스트 17개가 통과했다. APIC/guarded FLIP의 상태·난수 복원, 손상·누락·미지원 버전 파일 거부와 프레임 경계·콜백 제한을 포함한다.

과거 하중 → 관측 → 재충격 파일럿은 다음 명령으로 실행한다.

```powershell
.\scripts\run-wsl.ps1 history-pilot
```

`config/history-pilot.json`에 관측 프레임·하중 일정·후보·반복 조건·판정 임계값을 실행 전에 고정한다. 초기 상태는 동일한 체크포인트에서 시작한다. frame 2에서 현재 공간의 구형 영역에 서로 다른 세기의 국소 충격량을 주고, frames 12/14/16에서 관측한 뒤 frame 18에서 같은 하향 충격량을 준다. frame k 사건은 k의 관측·저장 이후, k→k+1 적분 전에 작용한다.

이 하중은 외부에서 지정한 순간 운동량 변화다. 위치·F·C·Jp·누적 G는 초기화하지 않는다. 각 사건의 실제 운동량·운동에너지 변화와 영향을 받은 입자 수를 기록한다. 충격체의 질량·접촉·운동을 풀어 얻은 하중이라고 해석하지 않는다.

관측용 `observations.h5`는 관측 구간의 RGB·합성 깊이·alpha·object alpha와 카메라를 포함한다. 내부 상태·전체 궤적·미래 반응은 별도 `evaluation.h5`에 두고, 관측 종료 체크포인트도 특권 정보로 취급한다. 깊이는 Gaussian 중심 camera-z의 alpha 정규화 평균이다. 바닥 가림을 같은 합성 과정으로 반영하지만, 실제 표면 RGB-D 센서나 보정된 잡음 모델을 구현한 것은 아니다.

2026-10-27 첫 실행: 8개 국소 충격 세기와 2개 반복, 총 10조건을 모두 완료했다. 가장 비슷하게 관측된 history-00/01의 미래 입자 위치 RMS 최댓값은 0.33791dx였다. 두 반복 조건에서 측정한 최대 변동은 0.33332dx였고, 사전에 정한 판정 규칙 `max(0.5dx, 3×반복 최댓값)`은 약 0.99996dx였다. 관측 유사성과 미래 구별 기준을 동시에 충족하는 쌍은 0개다. 이 한 차원 탐색에서는 Gate A를 입증하지 못했다. 10조건 모두 지속적 큰 파편 분리 후보도 없었으며 Gate B는 미검증이다. 새 상태 추정 방법의 효과·신규성에 대한 긍정 결과가 아니다.

결과는 `artifacts/history-pilot/report.json`, `pairwise.png`, `observation-pair.png`, `comparison.mp4`에 있다. 조건별 측정 구간은 약 9.3–9.6초, 장치 전체 메모리 표본 최댓값은 4,943MiB였으며 다른 앱 사용량을 포함한다. 전체 역추정의 시간·VRAM 측정값은 아니다. 첫 파일럿 단계에서 테스트 26개가 통과했다.

기존 파일럿의 분석만 재실행할 수 있다. 물리 재개 명령은 아니다.

```powershell
.\scripts\run-wsl.ps1 history-pilot --run /home/richard/research/world-model-research/runs/20261027T122702474949Z-history-pilot
```

파괴 순방향 진단과 실험용 경화 투영 보정의 실행 명령은 다음과 같다.

```powershell
.\scripts\run-wsl.ps1 fracture-probe
.\scripts\run-wsl.ps1 fracture-probe --config config/fracture-stable-probe.json
.\scripts\run-wsl.ps1 fracture-probe --config config/fracture-time-refinement.json
```

첫 명령은 물성·바닥·씨앗 beta·물리 입자 수·격자를 나눈 8조건 진단이다. 중앙 절반 bbox의 격자 점유율은 100%여서 현재 자산을 단순한 빈 껍질로 간주하지 않았다. 전체 1,379,914 Gaussian을 물리 입자로 쓰는 조건도 96³에서 실행했다. 이 조건은 조밀한 이웃 그래프의 메모리 부담 때문에 연결 분석을 하지 않았으며, 분리가 없었다고 판정하지 않는다. 96³/128³/192³의 분석한 완료 조건에서는 지속적 큰 파편 분리가 검출되지 않았다. 원본의 300³·전체 Gaussian·원본 시간 간격 조합을 충실히 재현한 실험은 아니다.

원본 물리 소스를 변경하지 않고 `wmfracture/experimental_nacc.py`에 실험용 보정을 분리했다. 극단적인 씨앗 beta=3e9에서 경화 레이 투영의 방향 정규화 불일치와 float32 이차식 overflow를 재현했다. 항복 분기와 경화 갱신을 한 계산 안에 두고, 스케일된 레이 해로 계산한다. 70자리 Decimal 기하 해와 두 개의 실제 실패 입력에 대조했다. 새로운 파괴 법칙이나 원본과 동일한 경화 동작을 주장하지 않는다.

보정 v2의 7조건은 반복·시간 간격 절반·128³/192³를 포함해 모두 0.6초까지 유한하게 완료됐다. 이후 시간 간격 1/4·1/8의 두 조건도 완료됐다. 추가 시간 세분화의 인접 궤적 차이는 최대 질량 가중 입자 RMS로 3.03175dx → 2.21543dx → 1.58294dx였다. 기준 조건의 반복 차이는 0.09361dx였으며, 미세 간격의 반복 분산은 별도 측정하지 않았다. 유한한 실행 성공이 시간 수렴이나 물리 정확도의 증명은 아니다. 모든 분석한 보정 조건에서도 지속적 큰 파편 분리는 검출되지 않아 Gate B는 미검증이다.

통합 보고서는 `artifacts/fracture-probe/audit-summary.json`, 시간 민감도 그림은 `time-refinement.png`와 `time-refinement-frames.png`다. 원본 진단·실패한 보정 v1·추가 회귀·최종 v2·시간 세분화까지 28개 실행을 원본 WSL 폴더에 보존했다. 조건별 결과는 `artifacts/fracture-probe/<실행ID>/`에 저장하며 `latest.json`이 최신 실행을 가리킨다. 최종 7조건의 비교는 `20261027T125253133605Z-fracture-probe/comparison.png`에 있다.

MPM 수치 회귀 단계에서 테스트 32개를 통과했다. 실험용 보정은 아직 정식 런타임·체크포인트에 통합하지 않았으며, 보정을 설치한 솔버의 체크포인트 저장은 명시적으로 거부한다. 기존 스키마 2 실행과 실험용 동역학을 혼용하지 않는다. 당시 관측 역추정은 미구현이었고, 아래 축약 접합면 기준선으로 진행했다. 새로운 연구 알고리즘은 아직 미구현이다.

작은 접합면의 손상·분리·재접촉 검증과 Gaussian 관측 생성은 다음 명령으로 실행한다.

```powershell
.\scripts\run-wsl.ps1 cohesive-probe
```

`wmfracture/cohesive.py`는 표준 bilinear cohesive law와 압축 penalty contact를 따로 구현한 검증 장치다. 두 강체 블록에 축방향 병진 자유도 하나씩, 처음부터 정해진 접합면 하나를 둔다. 손상 이력은 최대 벌어짐으로 저장하며, 완전 박리 후에도 압축 접촉은 작동하고 인장 결합은 복구되지 않는다. [Abaqus의 cohesive contact 문서](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-cohesivebehavior.htm)와 [MOOSE 구현 문서](https://mooseframework.inl.gov/docs/doxygen/modules/classBilinearMixedModeCohesiveZoneModel.html)를 참고한 독립 축약 구현이며, 해당 프로그램을 설치·실행한 재현은 아니다.

처방한 매끄러운 하중/제하로 서로 다른 손상 이력을 만들고 같은 위치·속도의 정지 상태로 돌아온다. 임의로 손상을 바꿔 비교하지 않는다. 관측 구간 후 같은 외부 충격량을 주며, 압축→반발→인장에 따른 접합면 박리를 계산한다. 별도 장면에서는 박리 후 닫히는 충격량을 주어 재접촉도 확인한다. 충격체 자체의 접촉 운동을 푼 모델은 아니다.

`config/cohesive-probe.json`은 세 장면·네 시간 간격·검증 허용값을 실행 전에 고정한다. 2026-10-27 최신 실행은 `20261027T132311352641Z-cohesive-probe`이며, 12개 계산과 9개 실행 판정 항목을 모두 통과했다. 원본 RGB·합성 깊이·물체 alpha는 사전 하중 유무 사이에 차이 0이었다. 같은 충격 뒤 초기 무손상 접합은 부분 손상 상태로 결합을 유지했고, 사전 손상 접합은 완전히 박리됐다. 0.18초의 틈은 각각 1.773mm와 23.900mm였다. 영상의 D·장면 이름은 검증용 정답 overlay이며 관측 HDF5에는 포함하지 않는다.

시간 간격 5µs와 2.5µs의 최대 위치 차이는 세 장면에서 각각 약 0.120µm, 0.123µm, 0.282µm였다. 네 시간 간격에서 박리 여부가 유지됐고, 가장 작은 두 간격에서 박리 시점 차이는 0이었다. 가장 작은 간격의 정규화된 에너지 잔차는 최대 약 2.25e-6이었다. 이는 이 축약 모델의 수치 비교이며 실제 재료 예측의 µm 정확도나 일반적인 시간/공간 수렴 증명은 아니다. 압축 접촉은 유한한 penalty 강성이므로 소량의 겹침을 허용한다.

물리 계산은 CPU float64, Gaussian 렌더링은 기존 CUDA 환경을 사용한다. 블록별 3,375개 Gaussian의 색·공분산·불투명도는 고정하고 물리 병진만 전달한다. `observations.h5`에는 관측 구간만, `evaluation.h5`와 사전 이력 파일에는 정답·미래 정보를 따로 둔다. 전용 JSON 상태 저장은 같은 코드에서의 자유 동역학 재개와 무결성을 검사했으며, MPM 체크포인트나 전체 외력 일정 재개를 대체하지 않는다.

전체 테스트는 42개 통과했다. 결과는 `artifacts/cohesive-probe/report.json`, `cohesive-contact.mp4`, `before-after.png`, `mechanics.png`다. 기존 물리 법칙을 확인하는 양성 대조군이며 새로운 연구 기여로 계산하지 않는다. 3D 변형체의 임의 균열 경로·균열면 마찰 접촉·기존 MPM 파괴의 Gate B는 여전히 미검증이다.

관측 → 하중 이력 후보 → 새 충격 예측의 기준선은 다음 명령으로 실행한다.

```powershell
.\scripts\run-wsl.ps1 cohesive-inverse
```

`config/cohesive-inverse.json`의 13개 고정 이력 후보를 실제 하중/제하로 재생하고, 짧은 RGB·합성 깊이·soft mask의 정합으로 후보를 거른다. 물성·형상·카메라·초기 정지 조건·관측 구간의 외력은 알려진 것으로 제한한다. 하나의 과거 하중 크기만 미지수다. 정답 이력 0.65mm와 1.85mm는 후보 격자에 포함하지 않았다. 후보를 하나의 정답으로 강제하지 않으며 정합 후보가 없으면 실패로 남긴다.

데이터 생성, 후보 선택, 미래 전진, 평가는 별도 프로세스로 실행한다. 선택 단계는 `public/manifest.json`, `public/observations.h5`, 공통 `fit-settings.json`만 읽는다. 입력 스키마는 내부 상태·미래 프레임·미래 외력을 거부한다. 후보 선택과 관측 종료 상태를 저장·해시로 고정한 뒤 `future-input.json`의 알려진 충격을 가해 전진한다. 정답 손상·미래 반응은 마지막 평가에서만 읽는다. 코드 계약과 파일 해시는 이 데이터 흐름을 검사하며, 확률 보정이나 새로운 추정 방법을 뜻하지 않는다.

최신 실행은 `20261027T140706166321Z-cohesive-inverse`다. 두 과거 이력 각각에 정지 관측/고정된 작은 충격 반응 관측을 주어 총 4문제를 만들었다. 각 쌍은 같은 잡음 실현을 사용하며, 작은 충격 이전의 noisy 프레임이 정확히 같은지도 검사한다. 관측은 0–0.028초, 알려진 작은 충격은 0.004초, 예측할 새 충격은 0.036초다. 정답 생성 dt=2.5µs, 추정/예측 dt=5µs를 사용했다. 잡음은 임의로 정한 합성 RGB·Gaussian 중심 깊이 잡음이며 실제 RGB-D 센서의 보정값이 아니다.

| 관측 조건 | 남은 후보 | 최저 관측 손실 후보의 미래 틈 RMSE | 기본 무손상 이력의 미래 틈 RMSE |
| --- | ---: | ---: | ---: |
| 0.65mm 이력, 정지 | 13/13 | 1.239mm | 1.239mm |
| 0.65mm 이력, 작은 충격 반응 | 2/13 | 0.819mm | 1.075mm |
| 1.85mm 이력, 정지 | 13/13 | 10.301mm | 10.301mm |
| 1.85mm 이력, 작은 충격 반응 | 4/13 | 0.054mm | 13.863mm |

정보가 늘어난 관측에서 단순 이력 탐색이 개선되는 예다. 같은 입력·예산의 강한 방법보다 우수하다는 비교나 새 알고리즘 기여는 아니다. 현재 평가값은 1차원 접합면 틈의 오차이며 전체 3D 형상·영상 정확도가 아니다. 작은 충격은 미리 정한 관측 절차이며 능동 탐색 정책을 학습하지 않았다.

중요한 한계: 남은 후보의 미래 곡선을 감싼 범위가 정답을 포함하는 평가 시점 비율은 작은 충격 조건에서 약 51.5%와 89.8%였다. 이는 한 궤적의 시간 표본 포함률이며 통계적 신뢰수준이 아니다. 후보가 줄어도 연속 이력 공간의 미탐색 상태를 포괄하지 못하므로 신뢰구간이라고 부르지 않는다. 정지 관측의 최저 손실 후보는 동률일 때 첫 후보를 고르는 임의 선택이고, 유일한 상태 식별을 뜻하지 않는다.

결과는 `artifacts/cohesive-inverse/report.json`, `forecast-envelopes.png`, `prediction-comparison.mp4`, `prediction-end.png`에 있다. 관측 정합 한 번은 13개 순방향 이력·104개 관측 렌더를 사용한다. 이 단계의 전체 테스트 49개가 통과했다. 학습된 모델의 연결 실험은 아래에 추가했으며, 새로운 추정 알고리즘·3D 파괴 백엔드의 검증은 남아 있다.

저장한 선택과 예측의 평가만 다시 실행할 수도 있다. 과거 하중 후보를 다시 선택하거나 정답으로 보정하지 않는다.

```powershell
.\scripts\run-wsl.ps1 cohesive-inverse --run /home/richard/research/world-model-research/runs/20261027T140706166321Z-cohesive-inverse
```

사전학습 월드 모델의 실제 연결 실험은 별도 환경에서 실행한다.

```powershell
.\scripts\run-world-model.ps1 bootstrap
.\scripts\run-world-model.ps1 test
.\scripts\run-world-model.ps1 run
# 저장한 선택을 유지한 평가만 재실행
.\scripts\run-world-model.ps1 evaluate --run /home/richard/research/world-model-research/runs/20261027T144148050017Z-world-model
```

`bootstrap`은 공개 V-JEPA 2 ViT-L 전체 체크포인트 약 5.13GB를 내려받고 고정 SHA256을 검사한다. 기존 Gaussian 환경과 분리한 Python 3.11.12 / Torch 2.7.1+cu128을 사용한다. 설정은 `config/world-model.json`, 설치 기록은 `requirements/world-model.lock.txt`다. [Meta 공식 구현](https://github.com/facebookresearch/vjepa2)의 인코더·학습된 masked latent 예측기·EMA target 인코더를 모두 로딩했다. 새 훈련은 하지 않았다.

앞의 네 합성 관측에서 첫 4프레임(0–12ms)만 입력하고 다음 4프레임(16–28ms)의 특징을 예측한다. 각 프레임을 두 번 반복하여 16프레임 텐서로 만든다. 공개 모델의 64프레임 masked-video 사전학습 설정과 다른 실험용 추론 프로토콜이다. 관측에 맞는 물리 후보의 Gaussian 영상과 예측 특징을 비교해 후보를 선택한 뒤, 36ms의 새 충격 반응은 물리 엔진으로 계산한다. 모델에 숫자 충격량을 직접 넣거나 모델만으로 새 충격 영상을 생성한 것은 아니다. 세 물리 예측 비교는 모두 같은 13개 후보를 공유한다.

결과는 `artifacts/world-model/report.json`, `pretrained-comparison.png`에 있다. 선택은 실제 관측의 후반부·정답 물리 상태를 읽기 전에 해시로 고정했다. 미지 미래 텐서의 채움값을 0에서 1로 바꿔도 네 입력 모두 예측 변화가 정확히 0이었다. 전체 테스트 53개, 새 환경의 입력 계약 테스트 4개가 통과했다.

| 관측 장면 | RGB-D만으로 선택한 미래 틈 RMSE | RGB-D 제약 + V-JEPA 선택 |
| --- | ---: | ---: |
| 낮은 사전 손상, 정지 | 1.239mm | 1.239mm |
| 낮은 사전 손상, 작은 충격 반응 | 0.819mm | 1.397mm |
| 높은 사전 손상, 정지 | 10.301mm | 10.301mm |
| 높은 사전 손상, 작은 충격 반응 | 0.058mm | 13.280mm |

현재의 단순 결합은 개선에 실패했다. latent ROI MAE도 예측기 0.551–0.555가 마지막 RGB 프레임을 반복한 강한 기준선 0.382–0.395보다 컸다(낮을수록 좋음). 마지막 context 특징만 반복하는 약한 기준선 0.647–0.654보다는 작았지만, 이를 미래 예측 능력 향상의 근거로 삼지 않는다. 강한 기준선은 최초 결과를 본 뒤 비교의 약점을 확인하기 위해 추가했고, 후보 선택은 바꾸지 않았다.

4090에서 모델 로딩·네 입력 예측·52개 후보 특징 비교를 포함한 순위 계산은 약 15.02초, Torch 최대 할당량 2,488MiB(약 2.43GiB)였다. 물리 후보 생성·Gaussian 렌더링·다운로드·별도 평가 시간과 다른 앱의 VRAM은 포함하지 않는다. 이 작은 추론 설정의 실행 가능성을 확인한 값이며 대형 모델 훈련이나 3D 파괴의 자원 요구량으로 일반화하지 않는다.

2026-10-28에는 **숫자 충격 입력을 받는 학습 모듈 → 물리 후보 정합 → Gaussian 영상**까지 연결했다.

```powershell
.\scripts\run-world-model.ps1 action-adapter
.\scripts\run-world-model.ps1 test -q
# 저장한 모델/예측을 바꾸지 않고 평가 또는 영상만 재생성
.\scripts\run-world-model.ps1 action-adapter --stage evaluate --run /home/richard/research/world-model-research/runs/20261027T153831097615Z-action-adapter
.\scripts\run-world-model.ps1 action-adapter --stage render --run /home/richard/research/world-model-research/runs/20261027T153831097615Z-action-adapter
```

`config/action-adapter.json` v2에 학습·검증·시험 조건을 고정했다. 서로 다른 44개 사전 하중 이력은 완전한 하중/제하로 생성하며 학습 24·검증 8·시험 12그룹으로 분리한다. 총 504문제는 학습 240·검증 48·시험 216개다. 하나의 관측을 여러 미래 충격에 재사용하므로 504개의 독립 물체/이력 사례는 아니다. 정지 관측은 모든 이력에서 정확히 같은 noisy prefix를 유지한다. 시험 충격은 학습에 쓴 크기, 학습 범위 안의 다른 크기, 범위 밖 0.42m/s로 나눠 보고한다. 범위 안의 다른 크기는 검증 단계의 epoch 선택에도 사용되며 완전히 미공개인 외력 분포는 아니다.

인코더·예측기를 동결한 V-JEPA 특징, 관측 RGB-D 요약과 13개 후보의 관측 잔차, 알려진 충격의 속도 변화량으로 작은 MLP를 학습한다. V-JEPA 결합 head는 29,606개 파라미터이며 37시점의 접합면 틈과 최종 박리 점수를 출력한다. 학습에는 합성 물리 정답을 사용한다. PCA/정규화는 학습 관측만, epoch 선택은 검증 응답만 사용하고 시험 예측을 모두 해시로 고정한 뒤 시험 정답을 읽는다. 충격량만·RGB-D+충격량·RGB-D+V-JEPA+충격량·미래 충격량 제외의 네 모듈을 각 3개 seed로 학습했다.

충격이 작용하기 직전 위치의 출력은 미래 충격량 입력을 구조적으로 제외한다. 초기 v1의 unrestricted MLP에서 이 위치가 최대 약 2.29mm 바뀌는 결함을 확인했고 v1 원본을 보존했다. v2는 다른 이력 값·분할 seed로 실행했으며 12개 head 모두 충격 직전 위치 변화가 0이었다. 성능 하이퍼파라미터를 v1 시험 결과에 맞춰 조정하지 않았다. 이는 고정된 충격 시각·표본 시각을 위한 경계 처리이며 임의 외력 일정의 인과 모델로 일반화하지 않는다.

| 반응 관측이 있는 시험 조건 | RGB-D + 물리 기준선 | V-JEPA + 충격 head | head + 물리 후보 정합 |
| --- | ---: | ---: | ---: |
| 새로운 이력, 학습한 충격 크기 | 0.951mm | 4.213mm | 4.637mm |
| 새로운 이력, 범위 안 다른 충격 | 1.314mm | 4.537mm | 4.755mm |
| 새로운 이력, 범위 밖 충격 | 0.868mm | 4.132mm | 3.515mm |

값은 시계열 틈 RMSE이며 head는 3개 학습 seed의 평균이다. 현재 결합은 정보가 있는 관측에서 물리 기준선을 이기지 못했다. 전체 시험의 직접 head 오차는 미래 충격 제외 11.434mm에서 충격 포함 3.841mm로 줄지만, 충격량만 쓰는 head도 3.752mm였다. V-JEPA 특징의 필요성·새로운 연구 기여를 확보했다고 해석하지 않는다. 정지 관측에서의 개선은 학습한 이력 분포의 평균 예측 효과이며 숨은 상태의 유일한 식별이 아니다.

출력은 `artifacts/action-adapter/report.json`, `action-comparison.png`, `prediction-comparison.mp4`, `prediction-end.png`다. 영상은 사전에 정한 첫 반응 관측 시험 이력·0.24m/s·seed 17의 예이며 성능으로 고른 장면이 아니다. 동일 Gaussian의 병진을 예측 틈과 알려진 질량중심 궤적으로 전달한다. 직접 head에는 접촉/손상 비가역성 보장이 없고, 물리 정합 출력은 관측에 맞는 기존 이력 후보 하나를 선택한다. 후보가 없으면 실패로 남긴다. 실제 3D 균열 경로·외형 학습·충격체 접촉을 구현한 것은 아니다.

v2의 데이터 생성 약 61.09초, frozen 특징 추출 15.58초, 12개 head 전체 학습 47.84초를 측정했다. 단계별 프로세스 시작·영상 생성 등은 별도다. 특징 추출의 Torch 최대 할당량은 약 2,500MiB, head 학습만의 최대 할당량은 약 65.89MiB였다. CUDA 문맥·다른 앱·기존 렌더러의 메모리는 이 head 수치에 포함하지 않는다. 기존 환경의 전체 테스트 61개, 새 환경의 관련 테스트 12개를 통과했다. 다음 설계에서는 직접 틈 회귀가 관측으로 얻은 물리 상태 정보를 잃는 문제를 우선 다룬다.

2026-10-28 추가: **관측만으로 상태 후보 가중치를 저장 → 저장한 후보별 물리를 재개 → 새로운 충격 조건별 Gaussian 영상**의 연결을 구현했다.

```powershell
.\scripts\run-world-model.ps1 belief-state
# 이 실행의 저장된 관측 상태를 그대로 두고 새 충격 크기/시각을 질의
.\scripts\run-world-model.ps1 belief-state --stage branch --run /home/richard/research/world-model-research/runs/20261028T044548684820Z-belief-state --prefix 129 --impact 0.245 --impact-time 0.044 --model learned_vjepa/seed-17
# 평가 또는 영상만 재생성
.\scripts\run-world-model.ps1 belief-state --stage evaluate --run /home/richard/research/world-model-research/runs/20261028T044548684820Z-belief-state
.\scripts\run-world-model.ps1 belief-state --stage render --run /home/richard/research/world-model-research/runs/20261028T044548684820Z-belief-state
```

`config/belief-state.json`은 앞선 두 action-head 데이터셋과 겹치지 않는 88개 하중 이력을 사용한다. train/validation/test는 48/16/24그룹, 외력별 사례는 480/96/432개다. 가중치 학습에는 합성 과거 하중의 격자 표현을 감독 신호로 쓰며 미래 응답 정답은 사용하지 않는다. 과거 하중은 13개 후보 사이의 barycentric 가중치로 표현한다. 이 인코딩 자체가 실제 연속 재료 상태의 사후분포라는 뜻은 아니다.

관측 잔차와 학습 이력 prior로 구성한 가중치에 작은 RGB-D/V-JEPA 수정 모듈을 추가한다. V-JEPA 수정 모듈은 4,877파라미터이며 관측 정보가 없는 정지 장면에서는 수정량을 정확히 0으로 둔다. 온도와 epoch는 validation의 상태 인코딩 손실로만 선택한다. 코드의 `calibrated_energy`는 이 온도 조절을 가리키며 확률 보정을 검증한 이름은 아니다. 균등 후보·최저 관측 잔차·온도 조절 가중치와 두 학습 모듈의 각 세 seed를 비교한다.

가중치 추론의 입력 계약은 관측 잔차·RGB-D 특징·관측 중 probe·prefix ID만 허용한다. 새 충격·미래 bank index·정답 필드는 거부한다. 전체 후보의 위치·속도·최대 벌어짐·물리 시계·누적 외부 일과 가중치를 체크섬으로 고정한 다음 알려진 새 충격을 가한다. 저장된 관측 시점 상태에서 재개한 234개 후보 응답은 처음부터 계속한 대응 계산과 틈 차이가 정확히 0이었고, 박리 사건도 같았다. 평균 상태를 새 물리 상태로 만들지 않는다. 이 재개는 시간 불변인 축약 법칙용 배치 계산이며 기존 MPM 체크포인트의 일반 외력/콜백 재개를 확장한 것은 아니다.

| 반응 관측이 있는 시험 조건 | 기존 최저 잔차 후보 | V-JEPA 가중치의 MAP 후보 | 가중 평균 틈의 통계 오차 |
| --- | ---: | ---: | ---: |
| 새로운 이력, 학습한 충격 크기 | 1.781mm | 1.919mm | 3.016mm |
| 새로운 이력, 범위 안 다른 충격 | 1.379mm | 1.548mm | 3.100mm |
| 새로운 이력, 범위 밖 충격 | 1.536mm | 1.572mm | 2.618mm |

값은 틈 RMSE이며 학습 모듈은 세 seed 평균이다. 가중 평균은 통계 요약으로만 평가하며 하나의 물리 궤적/상태로 렌더링하지 않는다. 이 실험은 앞선 head와 데이터 분할이 달라 수치를 그대로 비교해 개선율을 계산하면 안 된다. 반응 관측 시험에서 경로 energy score와 박리 Brier도 단일 후보보다 나빠졌다. V-JEPA의 추가 이득은 확인하지 못했고, 일부 seed에서는 검증이 학습 전의 수정량 0을 선택했다.

정지 관측에서는 가중치가 넓은 prior를 그대로 유지한다(유효 후보 약 11.63개). 반응 관측에서도 약 7.65개를 유지했다. 명목 90% 후보 밴드의 시점 포함률은 한 반응 관측 조건에서 약 77.83%였다. 이는 상관된 궤적 표본의 측정치이며 90% 신뢰구간이나 확률 보정의 증거가 아니다. 후보를 많이 유지하는 것만으로 예측이 좋아지지는 않았다.

`artifacts/belief-state/oracle-diagnostics.json`에는 평가용 특권 대조군도 기록했다. 같은 반응 관측 조건에서 정답 하중에 가장 가까운 후보는 1.705mm, 정답 상태 인코딩을 완벽히 준 혼합 평균도 1.670mm였다. 미래 정답을 보고 후보를 고르는 진단은 0.435mm였지만, 이 선택은 외력마다 달라질 수 있어 실제 방법으로 사용할 수 없다. 이 값들은 학습/온도 선택에 쓰지 않았다. 후보의 하중 거리와 미래 궤적 거리가 같은 것이 아니므로, 더 큰 수정 네트워크만으로 문제가 해결된다고 보지 않는다. 고정 후보 격자의 해상도와 상태 학습 목표도 후속 점검 대상이다.

`artifacts/belief-state/counterfactuals.mp4`는 같은 관측·상태 가중치로 0.15m/s@36ms, 0.42m/s@36ms, 0.325m/s@52ms를 새로 계산한 영상이다. 영상은 사전에 고정한 첫 반응 관측 시험 prefix와 seed 17의 물리 MAP 후보이며, 사용자 정의 분기의 정답 영상은 없다. `branch` 명령으로 0.245m/s@44ms도 실행해 입력 경로를 확인했다. 후보별 궤적과 박리 가중치도 WSL 실행 폴더의 `counterfactuals/`에 보존한다.

데이터 생성 101.89초, frozen 특징 23.02초, 여섯 수정 모듈 학습 5.36초였다. 특징의 Torch 최대 할당량은 약 2,500MiB, 수정 모듈 학습은 약 65.16MiB다. 별도 프로세스의 CUDA 문맥/렌더러/다른 앱은 포함하지 않는다. 전체 테스트 67개, 새 환경 관련 테스트 18개를 통과했다. 결과는 `artifacts/belief-state/report.json`, `belief-comparison.png`, `counterfactual-end.png`에 있다. 현재는 상태를 유지한 외력별 예측 기능을 확보한 단계이며, 예측 개선·novelty·일반 3D 파괴는 검증하지 못했다.

2026-10-28 추가: **관측 손실에 따라 하중 후보 구간을 세분화하는 탐색**을 구현하고 새 데이터 두 묶음에서 비교했다. 이번 단계는 물리/Gaussian 관측의 후보 탐색만 실행하며 V-JEPA나 학습 head는 사용하지 않는다.

```powershell
.\scripts\run-wsl.ps1 history-refinement
.\scripts\run-wsl.ps1 history-refinement --config config/history-refinement-replication.json
# 저장된 49개 후보 상태에 새 충격을 적용
.\scripts\run-wsl.ps1 history-refinement --stage branch --run /home/richard/research/world-model-research/runs/20261028T052334653812Z-history-refinement --case prefix-017 --method multiregion49 --impact 0.245 --impact-time 0.044
# 통합 비교 그림/보고서 재생성
.\scripts\run-wsl.ps1 python scripts/history_refinement_summary.py --runs /home/richard/research/world-model-research/runs/20261028T052334653812Z-history-refinement /home/richard/research/world-model-research/runs/20261028T053039987318Z-history-refinement
```

13개 고정 후보를 기준으로, 균일 49개·최저 관측 손실점 주변에 집중한 49개·여러 유망 구간과 미탐색 구간을 세분화한 49개를 비교한다. 적응 탐색은 처음 13개를 평가한 후 12라운드에 세 후보씩 추가한다. 여러 구간 방식은 라운드당 두 제안을 관측 손실의 local minimum 주변에, 한 제안을 가장 넓은 남은 구간에 배정한다. 후보는 매번 완전한 하중/제하와 관측 probe를 재생해 만들고 관측 시점의 전체 상태를 저장한다. 미래 외력·응답·정답 하중은 탐색 입력에 들어가지 않는다.

| 방법 | 첫 시험: 24개 반응 관측 이력 | 추가 시험: 12개 반응 관측 이력 |
| --- | ---: | ---: |
| 고정 13개, 작은 예산 기준선 | 1.231mm | 1.752mm |
| 균일 49개 | 0.612mm | 2.428mm |
| 단일 최저점 주변 49개 | 0.805mm | 1.838mm |
| 여러 구간 세분화 49개 | **0.336mm** | **0.322mm** |

각 값은 새 충격 세 크기(0.165/0.255/0.395m/s)에 대한 틈 RMSE다. 시험 이력은 이전 모든 데이터셋 및 두 묶음 사이에 겹치지 않는다. 첫 결과를 본 뒤 알고리즘·하이퍼파라미터를 바꾸지 않고 하중 값·관측 잡음 seed·사례 수만 바꿔 재확인했다. 여러 구간 방식은 이 두 시험의 전체 RMSE에서 더 좋았지만 모든 개별 사례에서 이긴 것은 아니다. 첫 시험에서는 네 방법의 박리 판정 정확도가 모두 71/72였고, 추가 시험에서는 단일 최저점/여러 구간 방식 36/36, 고정/균일 기준선 35/36이었다. 큰 오차 몇 사례가 RMSE에 영향을 주므로 중앙값·최댓값과 개별 결과도 보고서에 남겼다.

49개 방법은 사례당 49회의 서로 다른 관측 손실 평가와 196개 관측 렌더를 논리 예산으로 계산한다. 같은 물성·형상·관측 probe 조건의 순방향 캐시를 모든 방법이 공유한다. 실제 캐시 재사용·batch 크기가 달라 완전히 같은 실행 시간이나 속도 향상이라고 주장하지 않는다. 첫 실행의 전체 네 방법 탐색은 약 71.45초, 추가 실행은 38.64초였다. 실제 새 물리 이력 재생은 1,177/633회, Gaussian 관측 렌더는 5,048/2,632회였다. 원본 trace를 재생해 각 adaptive 제안이 그때까지 스스로 조회한 관측 손실만으로 재현되는지도 검사했다.

정지 관측은 모호함을 유지하고 고유한 point 예측을 거부한다. 주 정확도는 네 방법 모두 예측 가능한 반응 관측 시험의 공통 24/12개 이력을 사용한다. 선택한 상태 외에도 전체 후보와 관측 정합 집합을 저장한다. `branch`에서 정지 관측 `prefix-016`의 49개 대안도 전진시켜 고유 상태를 강제로 만들지 않는 입력 경로를 확인했다. 후보 집합을 신뢰구간으로 부르지 않는다.

물리 배치 계산에 속도·물리 시계·step·누적 외부 일 출력을 추가해 실제 관측 종료 상태를 재구성했다. 이를 기존 계산기와 대조하고 저장된 상태로 미래 외력을 계산한다. 내부 이력이나 시간을 초기화하거나 후보 상태를 평균하지 않는다. 전체 테스트 71개, 새 환경 관련 테스트 22개를 통과했다.

통합 결과는 `artifacts/history-refinement/comparison-summary.json`, `replication-comparison.png`, `integrity-checks.json`이다. 실행별 수치·그림·영상은 `artifacts/history-refinement/<실행ID>/`에 보존하며 루트의 `report.json`과 영상은 최신 실행을 가리킨다. 동일 예산에서 개선된 구성요소를 확보했지만, 이는 알려진 두 블록·한 미지 하중 파라미터의 합성 실험이다. 적응형 1D 탐색의 novelty·실제 재료 정확도·일반 3D 파괴는 입증하지 않았다.

2026-10-28 후속: **표준 DIRECT 비교에서 기존 탐색법의 우위는 확인하지 못했다.** 앞선 두 시험 관측을 재사용해 SciPy 1.12 DIRECT/DIRECT-L을 추가했다. 새로운 보류 데이터 실험으로 부르지 않는다. 실제 관측 목적함수 호출을 49회로 제한하며 cache/중복도 포함한다. `maxfun`은 엄격한 제한이 아니므로 callback에서 차단하고 실제 평가한 후보 중에서만 선택한다.

| 미래 틈 RMSE | 앞선 첫 시험 | 앞선 추가 시험 |
| --- | ---: | ---: |
| 여러 구간 49 | 0.336mm | 0.322mm |
| DIRECT 49 | **0.262mm** | **0.117mm** |
| DIRECT-L 49 | **0.262mm** | **0.117mm** |

균일/greedy 탐색 대비 개선이라는 앞선 결과는 유지하지만 표준 전역 최적화보다 좋은 새로운 탐색법이라는 근거는 없다. 보고서는 `artifacts/direct-benchmark/report.json`이다. 공유 cache 때문에 측정 시간 차이를 알고리즘 속도 차이로 해석하지 않는다.

같은 날짜에 **미지 과거 최대 하중과 미지 파괴 에너지의 2변수 파이프라인**도 구현했다. 공개 입력에서 정답 파괴 에너지를 제거하고 후보별 법칙으로 과거 하중 전체를 재생한다. 관측에 맞는 후보의 위치·속도·최대 벌어짐·일·물리 시계와 재료를 함께 저장한 뒤 새 충격에 분기한다. 최저 손실 후보는 대표 예측이며 유일한 식별 상태가 아니다. 이번 단계는 V-JEPA/신경망을 사용하지 않았다.

```powershell
.\scripts\run-wsl.ps1 direct-benchmark
.\scripts\run-wsl.ps1 joint-damage-material
# 기존 상태/재료로 다른 충격을 질의
.\scripts\run-wsl.ps1 joint-damage-material --stage branch --run /home/richard/research/world-model-research/runs/20261028T060646512016Z-joint-damage-material --case case-000 --method directL49 --impact 0.23 --impact-time 0.044
.\scripts\run-wsl.ps1 python scripts/audit_joint_damage_material.py --run /home/richard/research/world-model-research/runs/20261028T060646512016Z-joint-damage-material --branch case-000-directL49-dv-0.23-t-0.044
```

서로 다른 재료/과거 최대 벌어짐을 갖되 제하 강성이 같은 8쌍을 구성했다. 이 쌍들의 현재 손상도 D는 같을 수 있으며 최대 벌어짐 이력과 파괴 에너지가 다르다. 각 상태를 독립 계산·렌더링하고 쌍 안에 같은 잡음을 사용했다. 약한 관측 충격 0.04m/s에서는 **8/8쌍의 RGB·깊이·alpha가 정확히 같았지만**, 미래 충격 3개의 24개 쌍 비교 중 8개에서 최종 박리 사건이 달랐다. 같은 관측의 두 정답을 균등 가중한 이 구성에서 단일 결정론적 궤적의 통계적 RMSE 하한은 3.593mm다. 이 합성 대조군에만 해당하며 일반 물리 오차 하한은 아니다.

관측 충격을 0.12m/s로 올리면 2쌍이 구별됐지만 6쌍은 여전히 같았고 16상태 중 4개에 추가 손상이 생겼다. 자동으로 안전한 관측 하중을 선택하는 정책은 아직 없다. 약한/강한 관측의 대표 미래 RMSE는 uniform49 5.584/7.459mm, multiregion49 6.153/4.077mm, DIRECT49 5.164/5.156mm, DIRECT-L49 5.294/5.025mm다. 강한 조건의 일부 우위를 새 방법의 일반적 우수성으로 해석하지 않는다. 조건별 16관측×3미래 충격이며 96개의 독립 물체 실험은 아니다.

채택 후보 곡선 범위의 시점 포함률은 약 92–98%였지만 평균 폭이 13–24mm로 넓었다. 유한 후보 범위는 보정된 신뢰구간이나 정답 포함 보장이 아니다. `artifacts/joint-damage-material/`에 보고서·그래프·`matched-observation-futures.mp4`를 보존했다. 사전에 정한 첫 쌍/약한 관측/첫 미래 충격의 영상은 최종 틈 18.42mm와 2.19mm의 두 정답 및 하나의 대표 예측을 보여준다. 예측이 두 정답을 구별했다는 영상이 아니다.

관측 정합 탐색 약 112.79초, 고유 재료/이력 재생 755개·Gaussian 렌더 10,008프레임이었다. 데이터 생성·미래 분기·평가·영상 시간은 별도다. 49회 예산은 관측 목적함수 호출이며 전체 프로그램의 물리 호출 수와 다르다. 전체 테스트 **77개**, 새 환경 관련 **26개**를 통과했다. 새 분기의 세 후보를 독립 scalar 물리와 비교한 틈 차이는 0이고 원본 상태 packet 해시도 유지됐다. 여전히 두 강체·고정 접합면·1차원 운동의 진단이며 일반 3D 파괴나 새로운 연구 기여의 검증은 아니다.

2026-10-28 추가: **관측에 맞는 전체 상태 집합 → 가상 충격 반응 → 8개 상태 선택 → 새 충격의 Gaussian 영상**을 구현했다. `response-diversity`는 49개 관측 평가로 후보 집합을 생성하고 정합 여부와 전체 상태를 먼저 고정한다. 별도 가상 충격 0.16/0.29/0.43m/s에 대한 후보 자체 반응으로 표준 farthest-first k-center 선택을 수행한다. 관측 증거·원본 상태를 수정하지 않고 향후 계산할 부분집합만 선택한다. 평가 충격 0.195/0.335/0.465m/s와 정답은 선택 이후에만 읽는다.

```powershell
.\scripts\run-wsl.ps1 response-diversity
# 고정된 8개 상태만 실제로 전진시키는 새 질의
.\scripts\run-wsl.ps1 python scripts/response_diversity_branch.py --run /home/richard/research/world-model-research/runs/20261028T063241054478Z-response-diversity --impact 0.245 --impact-time 0.044
.\scripts\run-wsl.ps1 python scripts/audit_response_diversity.py --run /home/richard/research/world-model-research/runs/20261028T063241054478Z-response-diversity --branch case-000-uniform49-response8-dv-0.245-t-0.044
```

이전 2변수 실험과 정답 재료·하중 값이 겹치지 않는 새 6쌍/12상태×2관측 충격을 사용했다. 여전히 제하 강성을 맞춘 합성 대조군이며 일반적인 물체 분포가 아니다. 균일·DIRECT·세 고정 seed의 무작위 49개 집합 각각에서 관측 오차 순서 8개, 파라미터 다양성 8개, 미래 반응 다양성 8개, 무작위 8개(3 seed)를 비교한다. 전체 채택 후보는 더 큰 예산의 참고 집합이다. 모든 선택기는 동일 후보·동일 가상 반응을 제공받는다. 정답을 보고 가장 가까운 후보를 찾는 것은 평가용 진단이며 배포 가능한 점 예측이 아니다.

| 균일 후보 집합에서의 평가 | 관측 오차 순 8 | 파라미터 다양성 8 | 반응 다양성 8 |
| --- | ---: | ---: | ---: |
| 약한 관측: 전체 집합 반응의 표현 오차 | 5.533mm | 5.461mm | **3.470mm** |
| 강한 관측: 전체 집합 반응의 표현 오차 | 12.890mm | 7.499mm | **4.497mm** |
| 약한 관측: 정답과 가장 가까운 후보 RMSE | **0.713mm** | 2.141mm | 1.850mm |
| 강한 관측: 정답과 가장 가까운 후보 RMSE | **1.273mm** | 2.339mm | 1.789mm |

표현 오차는 각 문제/미래 충격에서 전체 채택 후보의 각 곡선을 가장 가까운 선택 곡선에 대조한 RMS 중 최댓값을 구하고 문제들에 평균한 값이다. 실제 정답과의 오차가 아니다. 5개 후보 집합×2관측 조건의 평균 표현 오차는 반응 다양성이 파라미터 다양성보다 9/10조건, 관측 오차 순서보다 10/10조건에서 작았다. 같은 물체를 재사용하므로 10개의 독립 반복 실험은 아니다. 무작위 후보 집합 하나의 약한 조건에서는 파라미터 다양성이 더 좋았다.

정답에 가까운 후보를 남기는 지표에서는 관측 오차 순서가 더 좋은 조건이 많았고, 반응 다양성의 범위 폭도 균일 집합에서 14.48/23.59mm로 넓었다. 폭+범위 밖 거리×20의 사전 고정 점수에서도 관측 오차 순서가 두 조건 모두 더 좋았다. 따라서 정확도·불확실성 보정·새 알고리즘의 우수성을 확보한 결과로 해석하지 않는다. 반응 다양성은 큰 집합의 가능한 미래를 소수의 실제 상태로 요약하는 데 유용한 초기 근거다.

49회 관측 평가 외에 후보 집합당 최대 147개의 가상 궤적 계산이 추가된다. 이 실험에서 가상 충격과 평가 참고 궤적은 각각 고유 물리 1,803개를 계산했다. 참고 평가에서는 전체 후보를 계산하므로 전체 실행 속도 개선은 측정하지 않았다. 별도 배포 명령은 실제로 8개만 계산했고, 8개의 모든 상태 필드를 독립 scalar 물리와 대조한 최대 차이는 약 1.101e-12였다. 관측 후보 생성은 약 74.76초였으며 다른 단계는 별도다.

추론·선택의 실험 파일 읽기를 제한하여 해당 실행의 비공개 정답/부모 설정/보류 충격 계획을 읽으면 중단하도록 했고, 실제 읽은 경로를 남겼다. 원본 후보와 선택의 SHA는 배포 후에도 유지됐다. 전체 테스트 **83개**, 새 환경 관련 **32개**가 통과했다. 결과는 `artifacts/response-diversity/report.json`, `compression-comparison.png`, `compressed-futures.mp4`, `integrity-checks.json`이다. 신경망·능동 관측 선택·일반 3D 파괴는 이번 단계에 포함하지 않았다.
