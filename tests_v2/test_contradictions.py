"""The MRD Escalator — Raphael's rule: patients have the right to make
mistakes and to be poor historians."""
from v2.contradictions import Contradiction, evaluate_contradictions
from v2.models import Chart, Medication
from v2.scoring import load_config, score_chart

CFG = load_config()


def C(**kw):
    base = dict(statement_quote="I take three a day", chart_fact="chart: prescribed two per day",
                chart_source="MedicationRequest mr1", materiality="material",
                direction="favours_more_controlled", encounter_id="e1", confirmed=True)
    base.update(kw)
    return Contradiction(**base)


def test_single_contradiction_zero_points():
    out = evaluate_contradictions([C()])
    assert out["points"] == 0
    assert out["analysis"]["qualifying_for_scoring"] == 1


def test_single_contradiction_no_zone_change():
    meds = [Medication(name="Oxycodone", drug_class="opioid", daily_mg=10)]
    plain = score_chart(Chart(medications=meds, has_pdmp_data=True), CFG)
    with_c = score_chart(Chart(medications=meds, has_pdmp_data=True), CFG, contradictions=[C()])
    assert with_c["zone"] == plain["zone"] and with_c["total"] == plain["total"]
    assert len(with_c["contradictions"]["informational"]) == 1


def test_minor_contradiction_never_scores_even_repeated():
    out = evaluate_contradictions([C(materiality="minor", encounter_id="e1"),
                                   C(materiality="minor", encounter_id="e2"),
                                   C(materiality="minor", encounter_id="e3")])
    assert out["points"] == 0


def test_unconfirmed_never_scores():
    out = evaluate_contradictions([C(confirmed=False, encounter_id="e1"),
                                   C(confirmed=False, encounter_id="e2")])
    assert out["points"] == 0


def test_same_encounter_repetition_does_not_score():
    out = evaluate_contradictions([C(encounter_id="e1"), C(encounter_id="e1")])
    assert out["points"] == 0  # repetition must span separate encounters


def test_repeated_material_directional_scores():
    out = evaluate_contradictions([C(encounter_id="e1"), C(encounter_id="e2"), C(encounter_id="e3")])
    assert out["points"] == 8  # (3-1) repetitions x 4
    assert out["analysis"]["direction_pattern"].startswith("consistent_direction")


def test_random_direction_scores_far_less_than_directional():
    directional = evaluate_contradictions([C(encounter_id="e1"), C(encounter_id="e2"), C(encounter_id="e3")])
    random_dir = evaluate_contradictions([
        C(encounter_id="e1"), C(encounter_id="e2", direction="favours_less_controlled"),
        C(encounter_id="e3", direction="neutral")])
    assert random_dir["points"] == 2
    assert random_dir["points"] < directional["points"] / 2


def test_contradiction_cap_below_objective_layer_a():
    many = [C(encounter_id=f"e{i}") for i in range(10)]
    out = evaluate_contradictions(many)
    assert out["points"] == 12  # cap
    assert out["points"] < CFG["layer_a"]["opioid_benzo_overlap"]["points"]


def test_chart_error_targets_record_not_patient():
    out = evaluate_contradictions([C(chart_possibly_wrong=True, encounter_id="e1"),
                                   C(encounter_id="e2"), C(encounter_id="e3")])
    assert len(out["chart_corrections"]) == 1
    assert out["chart_corrections"][0]["targets"] == "the discrepancy, not the patient"
    assert out["points"] == 4  # chart-error entry excluded; the other two qualify (1 repetition x 4)
    assert out["informational"][0]["innocent_explanations"]


def test_chart_error_case_informational_in_score_output():
    meds = [Medication(name="Oxycodone", drug_class="opioid", daily_mg=10)]
    r = score_chart(Chart(medications=meds, has_pdmp_data=True), CFG,
                    contradictions=[C(chart_possibly_wrong=True)])
    assert r["contradictions"]["chart_corrections"]
    assert any("chart-correction" in s for s in r["support_actions"])
    assert r["subscores"]["layer_c_of_which_contradictions_mrd"] == 0


def test_interpreted_contradiction_held_informational():
    out = evaluate_contradictions([C(interpreted_encounter=True, encounter_id="e1"),
                                   C(interpreted_encounter=True, encounter_id="e2")])
    assert out["points"] == 0


def test_contradictions_inside_layer_c_cap_in_total():
    meds = [Medication(name="Oxycodone", drug_class="opioid", daily_mg=5)]
    feats = {"declines_uds_or_pdmp_consent": True, "declines_noncontrolled_alternatives": True,
             "travel_distance_anomaly": "x", "urgency_escalation_between_visits": True}
    r = score_chart(Chart(medications=meds, has_pdmp_data=True, encounter_features=feats), CFG,
                    contradictions=[C(encounter_id=f"e{i}") for i in range(6)])
    assert r["subscores"]["layer_c_expert_lived_experience_unvalidated"] <= CFG["layer_c"]["cap"]
    assert r["zone"] != "high"


def test_innocent_explanations_always_present():
    out = evaluate_contradictions([C()])
    expl = out["informational"][0]["innocent_explanations"]
    assert any("poor historians" in e for e in expl)
    assert any("chart may be wrong" in e for e in expl)
