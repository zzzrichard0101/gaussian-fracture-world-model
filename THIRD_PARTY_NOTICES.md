# 외부 코드·모델·자산

이 저장소는 외부 모델 가중치, Gaussian 에셋, 외부 저장소 전체를 포함하지 않는다. 아래 구성요소는 실행 시 별도로 설치하거나 내려받으며 각각 원본의 이용 조건을 따른다. 이 프로젝트의 공개가 해당 구성요소의 라이선스를 변경하지 않는다.

| 구성요소 | 사용 목적 | 원본 |
| --- | --- | --- |
| GaussianFluent | 초기 MPM/Gaussian 실험 백엔드 | [공식 저장소](https://github.com/HB-pencil-zero/GaussianFluent) |
| 3D Gaussian Splatting 및 upstream rasterizer | Gaussian 렌더링과 CUDA 확장 | [공식 저장소](https://github.com/graphdeco-inria/gaussian-splatting) |
| simple-knn | 초기 에셋 이웃 검색 | GaussianFluent의 고정 submodule, `config/upstream.json` 참조 |
| V-JEPA 2 | 선택적 frozen encoder/predictor 실험 | [공식 저장소](https://github.com/facebookresearch/vjepa2) |
| PhysGaussian watermelon 에셋 | 초기 3D 렌더링·변형 실험 | `config/assets.json`의 원본 URL/revision 참조 |
| SciPy DIRECT | 표준 전역 최적화 기준선 | [SciPy](https://github.com/scipy/scipy) |

원본 커밋은 `config/upstream.json`, V-JEPA 출처와 가중치 해시는 `config/world-model.json`, 에셋 출처는 `config/assets.json`에 기록한다. 렌더러가 사용하는 Gaussian 표현과 축약 접합면의 물리 상태는 분리되어 있다. 공개 결과 그림·영상 중 두 블록 장면은 이 저장소의 합성 장면 생성기로 만든다.

별도 프로젝트 라이선스는 아직 지정하지 않았다. 공개 저장소라는 사실을 외부 의존성·모델·자산의 무제한 재사용 허가로 해석하지 않는다.
