import numpy as np
import pytest

from wmfracture.hypothesis_selection import response_features, select_hypotheses


def example():
    return np.array([[0., 1.], [.2, 1.2], [.4, 1.4], [.6, 1.6]]), np.array([0., .01, .02, .03])


def test_response_selection_retains_rare_future_without_changing_accepted_states():
    parameters, scores = example()
    gaps = np.zeros((4, 2, 3)); gaps[3, 1] = .02
    failure = np.zeros((4, 2)); failure[3, 1] = 1.
    original = parameters.copy(); accepted = [0, 1, 3]
    selected = select_hypotheses(parameters, scores, accepted, 2, "response", [[0, 1], [1, 2]],
                                 response_features(gaps, failure, .005))
    assert selected == [0, 3] and accepted == [0, 1, 3]
    np.testing.assert_array_equal(parameters, original)
    assert select_hypotheses(parameters, scores, accepted, 2, "residual", [[0, 1], [1, 2]]) == [0, 1]


def test_diversity_uses_whole_trajectory_and_event_not_only_final_gap():
    gaps = np.zeros((3, 1, 4)); gaps[1, 0, 1] = .01
    failure = np.array([[0.], [0.], [1.]])
    features = response_features(gaps, failure, .005)
    assert np.linalg.norm(features[1]-features[0]) > 0
    assert np.linalg.norm(features[2]-features[0]) > 0


def test_all_rules_keep_best_and_budget_and_exclude_rejected_candidates():
    parameters, scores = example()
    for rule in ("all", "random", "residual", "parameter", "response"):
        selected = select_hypotheses(parameters, scores, [0, 2, 3], 2, rule, [[0, 1], [1, 2]], parameters, seed=17)
        assert selected[0] == 0 and set(selected) <= {0, 2, 3}
        assert len(set(selected)) == len(selected) == (3 if rule == "all" else 2)
    assert select_hypotheses(parameters, scores, [], 2, "residual", [[0, 1], [1, 2]]) == []


def test_invalid_candidate_features_fail_instead_of_silent_truth_or_nan_use():
    parameters, scores = example()
    with pytest.raises(ValueError):
        select_hypotheses(parameters, scores, [0, 4], 2, "residual", [[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        response_features(np.zeros((3, 4)), np.zeros((3, 4)), .005)
    with pytest.raises(ValueError):
        select_hypotheses(parameters, scores, [0, 1], 2, "response", [[0, 1], [1, 2]], np.full((4, 2), np.nan))
