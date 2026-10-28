"""v2 core: MME, Layer A rules, zones, caps, fairness, indeterminate."""
import copy

from v2.models import Chart, Medication
from v2.mme import mme_for_drug, methadone_factor
from v2.scoring import load_config, score_chart
from v2.rules import overlap_days

CFG = load_config()


def med(name, cls, mg=None, pres=None, pharm=None, start=None, end=None, disp=(), supply=None):
    return Medication(name=name, drug_class=cls, daily_mg=mg, prescriber_id=pres,
                      pharmacy_id=pharm, start_date=start, end_date=end,
                      dispense_dates=list(disp), days_supply=supply)


def chart(**kw):
    kw.setdefault("has_pdmp_data", True)
    return Chart(**kw)


# ---- MME ----
def test_mme_oxycodone():
    assert mme_for_drug("oxycodone", 10) == 15.0


def test_mme_hydrocodone():
    assert mme_for_drug("hydrocodone", 20) == 20.0


def test_mme_methadone_sliding():
    assert methadone_factor(10) == 4 and methadone_factor(30) == 8
    assert methadone_factor(50) == 10 and methadone_factor(70) == 12
    assert mme_for_drug("methadone", 30) == 240.0


def test_mme_buprenorphine_excluded():
    assert mme_for_drug("buprenorphine", 16) is None


def test_mme_unknown_drug_none():
    assert mme_for_drug("gabapentin", 300) is None


def test_mme_combination_product_name():
    assert mme_for_drug("Oxycodone-Acetaminophen", 10) == 15.0


# ---- Layer A rules ----
def test_overlap_days_math():
    a = med("Oxycodone", "opioid", start="2026-08-01", end="2026-08-31")
    b = med("Lorazepam", "benzodiazepine", start="2026-08-15", end="2026-09-15")
    assert overlap_days(a, b) == 17


def test_overlap_item_fires():
    r = score_chart(chart(medications=[
        med("Oxycodone", "opioid", 10, start="2026-08-01", end="2026-08-31"),
        med("Lorazepam", "benzodiazepine", 1, start="2026-08-10", end="2026-08-20")]), CFG)
    assert "opioid_benzo_overlap" in {i["id"] for i in r["items"]}


def test_no_overlap_after_taper():
    r = score_chart(chart(medications=[
        med("Oxycodone", "opioid", 10, start="2026-08-01", end="2026-08-31"),
        med("Lorazepam", "benzodiazepine", 1, start="2026-05-01", end="2026-06-01")]), CFG)
    assert "opioid_benzo_overlap" not in {i["id"] for i in r["items"]}


def test_prescriber_threshold():
    meds = [med("Oxycodone", "opioid", 5, pres=f"Dr {c}") for c in "ABC"]
    r = score_chart(chart(medications=meds), CFG)
    assert r["subscores"]["layer_a_published_guideline"] >= 10


def test_early_refill_rule():
    r = score_chart(chart(medications=[
        med("Oxycodone", "opioid", 10, disp=("2026-08-01", "2026-08-22"), supply=30)]), CFG)
    assert "early_refill" in {i["id"] for i in r["items"]}


def test_on_time_refill_no_flag():
    r = score_chart(chart(medications=[
        med("Oxycodone", "opioid", 10, disp=("2026-08-01", "2026-08-30"), supply=30)]), CFG)
    assert "early_refill" not in {i["id"] for i in r["items"]}


def test_mme_tiers():
    r50 = score_chart(chart(medications=[med("Oxycodone", "opioid", 40)]), CFG)
    assert "mme_ge_50" in {i["id"] for i in r50["items"]}
    r90 = score_chart(chart(medications=[med("Morphine", "opioid", 95)]), CFG)
    ids = {i["id"] for i in r90["items"]}
    assert "mme_ge_90" in ids and "mme_ge_50" not in ids


def test_cancer_context_suppresses_mme_scoring():
    r = score_chart(chart(conditions=["Cancer-related pain"],
                          medications=[med("Morphine", "opioid", 150)]), CFG)
    assert r["subscores"]["layer_a_published_guideline"] == 0
    assert "guideline_exempt_context" in r["details"]


def test_ort_and_comm_items():
    r = score_chart(chart(medications=[med("Oxycodone", "opioid", 5)], ort_total=9, comm_total=10), CFG)
    ids = {i["id"] for i in r["items"]}
    assert {"ort_high", "comm_elevated"} <= ids


def test_documented_prior_diversion_item():
    r = score_chart(chart(medications=[med("Oxycodone", "opioid", 5)],
                          documented_prior_diversion=True), CFG)
    assert "documented_prior_diversion" in {i["id"] for i in r["items"]}


def test_sud_history_scores_zero_but_routes_support():
    r = score_chart(chart(medications=[med("Buprenorphine", "moud")],
                          sud_history_documented=True), CFG)
    assert r["total"] == 0
    assert any("ZERO score points" in s for s in r["support_actions"])


# ---- zones / caps / indeterminate ----
def test_zone_high_combo():
    meds = [med("Oxycodone", "opioid", 80, pres="Dr A", start="2026-08-01", end="2026-09-30",
                disp=("2026-08-01", "2026-08-19"), supply=30),
            med("Oxycodone", "opioid", 80, pres="Dr B"), med("Oxycodone", "opioid", 80, pres="Dr C"),
            med("Diazepam", "benzodiazepine", 5, pres="Dr B", start="2026-08-10", end="2026-09-10")]
    r = score_chart(chart(medications=meds), CFG)
    assert r["zone"] == "high"


def test_indeterminate_without_structured_data():
    c = Chart(narrative="anything", encounter_features={"travel_distance_anomaly": "x"})
    r = score_chart(c, CFG)
    assert r["zone"] == "indeterminate"


def test_layer_c_cap_and_no_high_zone_alone():
    feats = {k: True for k in ("pressure_tactics_quoted", "story_inconsistency_across_visits",
                               "declines_noncontrolled_alternatives", "declines_uds_or_pdmp_consent",
                               "travel_distance_anomaly", "cash_pay_insistence_with_insurance",
                               "urgency_escalation_between_visits")}
    feats["has_insurance"] = True
    feats["pressure_tactics_quoted"] = "verbatim quote here"
    r = score_chart(chart(medications=[med("Oxycodone", "opioid", 5)], encounter_features=feats), CFG)
    assert r["subscores"]["layer_c_expert_lived_experience_unvalidated"] <= CFG["layer_c"]["cap"]
    assert r["zone"] != "high"


def test_layer_c_disabled_by_config():
    cfg2 = copy.deepcopy(CFG)
    cfg2["layer_c"]["enabled"] = False
    r = score_chart(chart(medications=[med("Oxycodone", "opioid", 5)],
                          encounter_features={"travel_distance_anomaly": "x"}), cfg2)
    assert r["subscores"]["layer_c_expert_lived_experience_unvalidated"] == 0


def test_unverified_layer_b_flag_never_scores():
    r = score_chart(chart(medications=[med("Oxycodone", "opioid", 5)]), CFG,
                    layer_b_flags=[{"signal": "lost_medication_report", "verbatim_quote": "x", "verified": False}])
    assert r["subscores"]["layer_b_ai_chart_review"] == 0


def test_output_pattern_language_and_unknown_sens_spec():
    r = score_chart(chart(medications=[med("Morphine", "opioid", 100)]), CFG)
    assert "UNKNOWN" in r["sensitivity_specificity"]
    assert r["human_confirmation_required"] is True
    assert "False positives" in r["risk_language"] and "False negatives" in r["risk_language"]
    assert "diverter" not in str(r).lower() and "manipulat" not in str(r).lower()


def test_naloxone_support_prompt_on_mme():
    r = score_chart(chart(medications=[med("Morphine", "opioid", 60)]), CFG)
    assert any("naloxone" in s.lower() for s in r["support_actions"])


# ---- fairness: protected attributes excluded ----
def test_protected_attributes_do_not_change_score():
    from v2.models import chart_from_dict
    payload = {"medications": [{"name": "Morphine", "drug_class": "opioid", "daily_mg": 100,
                                "prescriber_id": "Dr A"}],
               "has_pdmp_data": True, "patient_ref": "p1"}
    base = score_chart(chart_from_dict(copy.deepcopy(payload)), CFG)
    for key, val in [("race", "X"), ("ethnicity", "Y"), ("sex", "F"),
                     ("preferred_language", "Spanish"), ("disability_status", "disabled"),
                     ("insurance_type", "Medicaid"), ("gender", "female")]:
        p2 = copy.deepcopy(payload)
        p2[key] = val
        r = score_chart(chart_from_dict(p2), CFG)
        assert r["total"] == base["total"], key
        assert r["items"] == base["items"], key


def test_cash_pay_without_insurance_not_scored():
    r = score_chart(chart(medications=[med("Oxycodone", "opioid", 5)],
                          encounter_features={"cash_pay_insistence_with_insurance": True,
                                              "has_insurance": False}), CFG)
    assert "cash_pay_insistence_with_insurance" not in {i["id"] for i in r["items"]}
