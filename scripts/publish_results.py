"""Export a small, explicit set of public figures/tables; no raw runs or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        return value.replace("/home/richard/research/world-model-research/", "<WM_RESEARCH_ROOT>/")
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    project = parser.parse_args().project
    source = project/"artifacts"
    result = project/"docs/results"; result.mkdir(parents=True, exist_ok=True)
    media = project/"docs/media"; media.mkdir(parents=True, exist_ok=True)
    manifest = []
    stages = ["direct-benchmark", "joint-damage-material", "response-diversity", "active-probe", "world-model", "action-adapter", "belief-state", "cohesive-probe"]
    for stage in stages:
        path = source/stage/"report.json"
        value = portable(json.loads(path.read_text(encoding="utf-8")))
        target = result/(stage+".json")
        write_json(target, value)
        manifest.append({"source": f"artifacts/{stage}/report.json", "source_sha256": sha(path),
            "published": str(target.relative_to(project)).replace("\\", "/"), "published_sha256": sha(target),
            "transformation": "Machine-specific runtime root replaced in strings only; numeric results unchanged"})
    for stage, name in [("active-probe", "integrity-checks.json"), ("active-probe", "failure-diagnostics.json"),
                        ("response-diversity", "integrity-checks.json"), ("joint-damage-material", "integrity-checks.json")]:
        path = source/stage/name
        target = result/(stage+"-"+name)
        write_json(target, portable(json.loads(path.read_text(encoding="utf-8"))))
        manifest.append({"source": str(path.relative_to(project)).replace("\\", "/"), "source_sha256": sha(path),
            "published": str(target.relative_to(project)).replace("\\", "/"), "published_sha256": sha(target)})
    figures = {"direct-benchmark": ["direct-comparison.png"], "joint-damage-material": ["joint-comparison.png", "prediction-end.png", "matched-observation-futures.mp4"],
               "response-diversity": ["compression-comparison.png", "compressed-futures.mp4"], "active-probe": ["active-comparison.png", "prediction-end.png", "active-probe-futures.mp4"]}
    for stage, names in figures.items():
        for name in names:
            path = source/stage/name
            target = media/(stage+"-"+name)
            shutil.copy2(path, target)
            manifest.append({"source": str(path.relative_to(project)).replace("\\", "/"), "published": str(target.relative_to(project)).replace("\\", "/"), "sha256": sha(target)})
    direct = json.loads((result/"direct-benchmark.json").read_text(encoding="utf-8"))
    response = json.loads((result/"response-diversity.json").read_text(encoding="utf-8"))
    active = json.loads((result/"active-probe.json").read_text(encoding="utf-8"))
    lines = ["# 결과표와 해석", "", "모든 수치는 합성 두 블록·한 접합면·1차원 병진 문제의 결과다. 보류 이력/물성은 설정에서 분리했지만 독립적인 실물 물체나 일반 3D 균열 실험은 아니다.", "",
        "## 1. 강한 전역 최적화 기준선", "", "앞선 두 시험 관측을 재사용한 비교다. 새로운 보류 시험이 아니다. 오차는 미래 접합면 틈의 RMSE이며 낮을수록 좋다.", "",
        "| 방법 | 첫 시험 RMSE (mm) | 추가 시험 RMSE (mm) |", "| --- | ---: | ---: |"]
    for name in direct["folds"][0]["methods"]:
        values = [f["methods"][name]["gap_rmse_mm"] for f in direct["folds"]]
        lines.append(f"| {name} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ["", "DIRECT가 두 시험 모두 가장 작았다. 적응형 세분화의 새로운 최적화 알고리즘으로서의 우수성은 확보하지 못했다.", "", "![DIRECT 비교](media/direct-benchmark-direct-comparison.png)", "",
        "## 2. 전체 후보의 미래를 8개 상태로 요약", "", "전체 후보 집합을 가상 충격으로 비교해 일부 상태를 남긴다. `표현 오차`는 전체 후보의 각 곡선에서 가장 가까운 선택 곡선까지의 RMS 중 최댓값을 구하고 문제별로 평균한 값이다. 정답과의 오차가 아니다.", "",
        "`정답 최근접 후보 오차`는 평가 정답을 보고 집합에서 가장 가까운 곡선을 찾는 진단이다. 실제 예측 시 이 후보를 알아낼 수 없으므로 점 예측 성능으로 해석하지 않는다.", "",
        "| 관측 충격 | 균일 후보에서 선택법 | 표현 오차 (mm) | 정답 최근접 후보 RMSE (mm) | 범위 폭 (mm) | 시점 포함률 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for probe in ("0.04", "0.12"):
        for name in ("residual8", "parameter8", "response8", "random8-17", "random8-37", "random8-59", "full"):
            r = response["summaries"][probe]["uniform49"][name]
            lines.append(f"| {probe}m/s | {name} | {r['full_pool_response_max_nearest_rmse_mm']:.3f} | {r['nearest_candidate_rmse_mm']:.3f} | {r['envelope_width_mm']:.2f} | {100*r['envelope_time_coverage']:.1f}% |")
    lines += ["", "반응 다양성은 전체 미래 집합의 표현에 도움이 됐지만, 관측 잔차 순서의 단순 선택보다 정답 근처 후보를 잘 남긴다는 결과는 아니다. 넓은 범위는 보정된 신뢰구간이 아니다. 전체 집합 `full`은 더 큰 상태 수를 쓰는 참고값이다.", "", "![상태 압축](media/response-diversity-compression-comparison.png)", "",
        "## 3. 추가 관측 충격 선택과 실제 손상", "", "각 정책은 물체에 서로 다른 충격을 준다. 표의 관측 전·후는 **같은 정책이 선택한 충격을 받은 정답**과 비교한다. 정책 간 RMSE만으로 정보 획득의 인과 효과를 비교하면 안 된다. 여기서 MAP라는 원시 필드명은 최저 관측 잔차의 대표 후보를 가리키며 보정된 사후분포를 뜻하지 않는다.", "",
        "추가 손상은 추가 충격 없이 같은 시각까지 진행한 대조군에 대한 손상도 D 증가다. D는 0~1의 무차원 값이며 후보 제약값은 0.02다. 각 표는 실제 재료/이력 8개×미래 충격 3개를 재사용한다. 무작위 seed와 방법을 독립 물체 실험으로 세지 않는다."]
    for pool in ("uniform49", "direct49"):
        for state_set in ("full", "response8"):
            lines += ["", f"### {pool} / {state_set}", "", "| 정책 | 관측 전 RMSE (mm) | 관측 후 RMSE (mm) | 평균 추가 손상 D | 실제 제약 위반 | 평균 남은 후보 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for policy, r in active["summaries"][pool][state_set].items():
                before = r.get("before_map_matched_rmse_mm", float("nan")); after = r.get("after_map_matched_rmse_mm", float("nan"))
                violations = round(r["actual_damage_cap_exceeded"]*r["episodes"])
                lines.append(f"| {policy} | {before:.3f} | {after:.3f} | {r['actual_added_damage']:.4f} | {violations}/{r['episodes']} | {r['posterior_count']:.2f} |")
    lines += ["", "이번 실행에서는 모든 정책의 관측 후 예측이 가능했다. 다만 후보 수 감소가 정확도 개선을 보장하지 않았다.", "",
        "손상 제한 정책은 대부분 0 충격을 골랐고, DIRECT response8 조건에서만 8상태 중 2개에 0.20m/s를 골랐다. 제약 위반은 없었지만, 그 사실만으로 정보 획득·안전성·예측 개선이 입증되는 것은 아니다. 고정 0.20m/s는 실제 8상태 중 6개에서 손상 증가 0.02를 넘겼다.", "",
        "![능동 관측](media/active-probe-active-comparison.png)", "",
        "## 데이터와 검증", "", "[전체 원시 요약 JSON](results/active-probe.json) · [독립 물리 감사](results/active-probe-integrity-checks.json) · [사후 실패 진단](results/active-probe-failure-diagnostics.json)", "",
        "사후 진단은 방법/임계값을 바꾸는 데 쓰지 않았다. 관측 잔차가 낮다는 이유로 정답 미래에 가까운 후보를 제거하는 경우와, 8개 압축 집합이 전체 집합의 최악 손상을 과소평가하는 경우를 기록한다. 가설 집합 안의 손상 제약은 참 물체의 안전 보장이 아니다."]
    (project/"docs/RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # Keep the original publication intact and add explicit follow-up exports.
    followup_stages = ["compliant-pilot", "synthetic-filter-benchmark", "bias-aware-filter",
        "finite-damage-guard", "action-ceiling-diagnostic", "continuous-risk-probe",
        "continuous-risk-evaluation", "risk-reuse", "observation-budget", "response-quadrature", "guided-quadrature", "task-aware-pilot", "alternating-frontier", "probe-identifiability", "spatial-fixture", "spatial-observation", "spatial-failure-search", "spatial-continuous", "spatial-risk", "spatial-risk-refinement", "spatial-transport"]
    for stage in followup_stages:
        names = ["report.json"]
        if stage == "risk-reuse":
            names += ["inputs.json", "config.json", "code-sha256.json", "analysis-summary.json"]
        if stage == "continuous-risk-evaluation":
            names += ["evaluation-rows.json", "analysis-summary.json"]
        if stage == "observation-budget":
            names += ["evaluation-rows.json", "analysis-summary.json", "config.json", "analysis-plan.json", "code-sha256.json"]
        if stage == "response-quadrature":
            names += ["config.json", "refinement-checks.json", "adaptive-checks.json", "code-sha256.json"]
        if stage == "guided-quadrature":
            names += ["config.json", "inputs.json", "refinement-checks.json", "adaptive-checks.json",
                "analysis-summary.json", "code-sha256.json", "evaluation-rows.json"]
        if stage == "task-aware-pilot":
            names += ["config.json", "case-config.json", "analysis-plan.json", "score-audits.json", "tight-score-audits.json",
                "refinement-checks.json", "analysis-summary.json", "code-sha256.json", "evaluation-rows.json", "public-episodes.json"]
        if stage == "alternating-frontier":
            names += ["config.json", "comparison-config.json", "case-config.json", "reference-sha256.json", "control-replay.json",
                "refinement-checks.json", "analysis-summary.json", "code-sha256.json", "evaluation-rows.json"]
        if stage == "probe-identifiability":
            names += ["config.json", "code-sha256.json", "proofs.json"]
        if stage == "spatial-fixture":
            names += ["config.json", "code-sha256.json", "analysis-summary.json", "evaluation-rows.json", "preparations.json"]
        if stage == "spatial-observation":
            names += ["config.json", "catalog.json", "case-config.json", "code-sha256.json", "analysis-summary.json",
                "evaluation-rows.json", "public-episodes.json", "decisions.json", "reference-audit.json", "refinement-audit.json"]
        if stage == "spatial-failure-search":
            names += ["config.json", "code-sha256.json", "source-sha256.json", "inputs.json", "witnesses.json"]
        if stage == "spatial-continuous":
            names += ["config.json", "code-sha256.json", "source-sha256.json", "inputs.json", "predictions.json", "evaluation-rows.json"]
        if stage in ("spatial-risk", "spatial-risk-refinement"):
            names += ["config.json", "code-sha256.json", "inputs.json", "proofs.json", "analysis-summary.json"]
        if stage == "spatial-transport":
            names += ["config.json", "inputs.json", "code-sha256.json", "source-sha256.json", "report-source-sha256.json",
                "predictions.json", "adaptation.json", "evaluation-rows.json", "analysis-summary.json", "diagnostics.json"]
        for name in names:
            path = source/stage/name
            target = result/(stage+("" if name == "report.json" else "-"+name[:-5])+".json")
            write_json(target, portable(json.loads(path.read_text(encoding="utf-8"))))
            manifest.append({"source": str(path.relative_to(project)).replace("\\", "/"),
                "source_sha256": sha(path), "published": str(target.relative_to(project)).replace("\\", "/"),
                "published_sha256": sha(target), "transformation": "Runtime root replaced; numeric results unchanged"})
        figure = source/stage/"comparison.png"
        if figure.exists():
            target = media/(stage+"-comparison.png");shutil.copy2(figure,target)
            manifest.append({"source": str(figure.relative_to(project)).replace("\\", "/"),
                "published": str(target.relative_to(project)).replace("\\", "/"), "sha256": sha(target)})
    with (project/"docs/RESULTS.md").open("a",encoding="utf-8") as handle:
        handle.write("\n## 후속 스칼라 관측·위험 검증 실험\n\n"
            "[새 상태의 손상·예측 비교](CONTINUOUS_RISK_EVALUATION.md) · "
            "[동일 판정의 계산 재사용](RISK_REUSE.md) · "
            "[관측 횟수·입력 선택 비교](OBSERVATION_BUDGET.md) · "
            "[관측식 기반 적분 안정화](RESPONSE_QUADRATURE.md) · "
            "[목표 미래를 고려한 폐루프 관측](TASK_AWARE_PILOT.md) · "
            "[무하중/검증 입력 교대 기준선](ALTERNATING_FRONTIER.md) · "
            "[손상 한도·식별 가능성 진단](PROBE_IDENTIFIABILITY.md) · "
            "[조건별 수치](results/continuous-risk-evaluation-evaluation-rows.json)\n\n"
            "후속 단계는 합성 gap 값과 알려진 준정적 법칙을 사용한다. "
            "이 페이지의 앞선 Gaussian 영상 실험과 관측·역학 조건이 다르다.\n")
        handle.write("\n## 후속 공간 순방향 모형\n\n"
            "[세 구역 접합면·이동과 회전·준비 이력](SPATIAL_FIXTURE.md)을 구현했다. "
            "다섯 설계 사례에서 증분·평형·일과 소산을 검사한 순방향 단계이며 "
            "영상 관측·상태 추정·공간 위험 인증은 아직 포함하지 않는다.\n")
        handle.write("\n## 공간 관측과 유한 후보 추정의 연결\n\n"
            "[관측·추정·탐침 규칙·목록 밖 실패](SPATIAL_OBSERVATION.md)를 연결했다. "
            "29개 이산 후보 안의 조건부 결과와 네 목록 밖 상태의 stress를 분리한다. "
            "유한 검사로 놓친 위험의 공개 관측 기반 수치 반례도 함께 기록했다.\n")
        handle.write("\n## 연속 prior와 조건부 구간 판정\n\n"
            "[연속 추정·수렴 실패·구간 판정](SPATIAL_CONTINUOUS.md)을 구현했다. "
            "32개 고정 이력의 수렴 검사는 통과하지 못했다. 이전 위험점은 재확인했지만 "
            "넓은 파라미터 영역에서는 상한이 커서 보류했다. 좁은 대조 영역의 허용을 일반화하지 않는다.\n")
        handle.write("\n## 반응 기반 중요도 제안과 원래 prior 보정\n\n"
            "[국소 반응 제안·네 기준선·비용·실패](SPATIAL_TRANSPORT.md)를 추가했다. "
            "이미 본 8개 개발 이력에서 낮은 잡음 2개만 수렴 검사를 통과했다. "
            "높은 잡음은 모두 실패했고, 추가 적응 비용·정답 오차·전체 위험 상한의 문제는 남아 있다.\n")
    write_json(result/"publication-manifest.json", {"scope": "Selected reports, public replay inputs and media only. No weights, assets, observed clips or full execution data.", "files": manifest})
    print(f"Published {len(manifest)} selected artifacts to docs/")


if __name__ == "__main__":
    main()
