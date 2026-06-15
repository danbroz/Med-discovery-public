from __future__ import annotations

"""
Nightmare Disorder siibra scaffold.

This script turns a short Nightmare Disorder chapter into an atlas-grounded,
mechanistic research scaffold using siibra. It is designed for exploratory
modeling, not diagnosis or treatment.

Chapter logic represented here:
- recurrent threat-laden dreams as a REM-linked parasomnia,
- bidirectional serotonergic dysregulation that can destabilize REM sleep,
- GABAergic/hypnotic disruption of sleep architecture,
- cholinergic REM drive that may intensify vivid dreaming,
- stress/HPA-axis sensitization and noradrenergic over-arousal,
- limbic-dominant REM circuitry involving amygdala, hippocampus, ACC, mPFC,
  brainstem nuclei, and relative deactivation of the dorsal executive network.

The scaffold is intentionally conservative. Neurochemistry and stress-endocrine
processes are modeled as latent biology rather than forced into specific
cytoarchitectonic parcels. The medial prefrontal cortex and brainstem REM
systems are represented with clearly labeled proxies because the chapter is
systems-level there and exact Julich labels may vary.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "SLC6A4",  # serotonin transporter
    "HTR1A",   # serotonergic receptor
    "HTR2A",   # serotonergic receptor / REM and dream phenomenology relevance
    "GABRA1",  # GABA-A alpha 1
    "GABRB2",  # GABA-A beta 2
    "CHRM2",   # muscarinic cholinergic receptor
    "CHRNA4",  # nicotinic cholinergic receptor
    "CRHR1",   # stress-axis signaling
    "FKBP5",   # glucocorticoid feedback / trauma sensitivity
    "NR3C1",   # glucocorticoid receptor
    "SLC6A2",  # norepinephrine transporter
    "ADRA2A",  # adrenergic autoreceptor / arousal regulation
    "BDNF",    # neuroplasticity and affective regulation
]


class NightmareDisorderModel:
    """
    Atlas-grounded research scaffold for Nightmare Disorder.

    Notes
    -----
    - This is a mechanistic interpretation of a chapter, not a validated disease model.
    - Direct molecular evidence for Nightmare Disorder remains limited, so the gene panel is exploratory.
    - Some nodes are explicit proxies when the chapter names a system but not a precise parcel.
    - Regional state scores reflect dysregulation burden rather than literal fMRI activation magnitudes.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        connectivity_subjects_to_average: int = 8,
    ) -> None:
        self.atlas = siibra.atlases.get(atlas_spec)
        self.parcellation_spec = parcellation_spec
        self.parcellation = (
            self.atlas.get_parcellation(parcellation_spec)
            if hasattr(self.atlas, "get_parcellation")
            else self.atlas.parcellations.get(parcellation_spec)
        )
        self.space_spec = space_spec
        self.space = (
            self.atlas.get_space(space_spec)
            if hasattr(self.atlas, "get_space")
            else self.atlas.spaces.get(space_spec)
        )
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.connectivity_subjects_to_average = max(1, int(connectivity_subjects_to_average))

        # Disorder worksheet distilled from the chapter.
        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Exploratory heritable liability for sleep-state regulation, emotional reactivity, and stress sensitivity."
            ),
            "trauma_history": (
                "Traumatic exposure that can precipitate recurrent nightmares and amplify threat imagery."
            ),
            "early_life_adversity": (
                "Early adversity that may leave epigenetic stress-sensitization signatures."
            ),
            "chronic_stress_load": (
                "Sustained stress burden increasing HPA activation and sleep-emotion dysregulation."
            ),
            "depression_anxiety_burden": (
                "Comorbid affective burden linked to REM dysregulation, emotional intensity, and nightmare risk."
            ),
            "serotonergic_perturbation": (
                "Altered serotonergic tone, including antidepressant-related REM instability or withdrawal effects."
            ),
            "gabaergic_hypnotic_exposure": (
                "Hypnotic/GABAergic sleep manipulation that may alter sleep architecture and parasomnia risk."
            ),
            "recovery_sleep_support": (
                "Protective sleep stabilization, trauma-informed care, and supportive recovery structure."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "stress_sensitized_sleep_liability": (
                "Trait-like stress and sleep vulnerability shaped by heritability and adversity."
            ),
            "serotonergic_dysregulation": (
                "Bidirectional 5-HT dysregulation affecting REM gating, mood, and anxiety."
            ),
            "sleep_architecture_disruption": (
                "Broad sleep-state disruption associated with hypnotics, stress, and REM instability."
            ),
            "cholinergic_rem_drive": (
                "REM-promoting cholinergic pressure that may intensify vivid dream generation."
            ),
            "hpa_axis_dysregulation": (
                "Stress-system dysregulation with altered cortisol feedback and trauma-linked sensitization."
            ),
            "noradrenergic_rem_arousal": (
                "REM-period over-arousal that increases awakening and threat-related autonomic intensity."
            ),
            "rem_sleep_instability": (
                "State instability of REM sleep enabling breakthrough of intense negative dream content."
            ),
            "executive_dream_regulation_failure": (
                "Reduced top-down regulation of dream affect and narrative control during REM."
            ),
            "frontolimbic_dream_dysregulation": (
                "Limbic-emotional predominance over regulatory prefrontal control during REM sleep."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "recurrent_threat_nightmares": (
                "Repeated disturbing dreams involving threat to survival, security, or physical integrity."
            ),
            "nightmare_emotional_intensity": (
                "High vividness and negative emotional load within dream content."
            ),
            "abrupt_awakening_alertness": (
                "Awakening from the nightmare with rapid orientation and heightened alertness."
            ),
            "rem_parasomnia_liability": (
                "Vulnerability to nightmare-related REM parasomnia expression and dream intrusion."
            ),
            "sleep_disruption": (
                "Sleep fragmentation and disturbed nocturnal continuity after nightmare episodes."
            ),
            "daytime_distress_impairment": (
                "Clinically meaningful distress or functional impairment resulting from nightmares and poor sleep."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus",
            ],
            "acc": [
                "Area p24ab (pACC) left",
                "Area p24c (pACC) left",
                "Area p32 (pACC) left",
                "Area 33 (ACC) left",
                "anterior cingulate",
                "p24",
            ],
            "dlpfc": [
                "Area 9/46d (DLPFC) left",
                "Area 9/46v (DLPFC) left",
                "dorsolateral prefrontal",
                "DLPFC",
                "9/46",
            ],
            "mpfc_proxy": [
                "Area Fp1 (FPole) left",
                "Area Fp2 (FPole) left",
                "medial prefrontal cortex",
                "medial prefrontal",
                "mPFC",
                "frontopolar",
            ],
            "brainstem_rem_proxy": [
                "brainstem",
                "pons",
                "midbrain",
                "reticular formation",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Threat and emotional salience node for intense negative dream affect."
            ),
            "hippocampus": (
                "Memory-context node for emotionally laden REM reactivation and dream narrative content."
            ),
            "acc": (
                "Anterior cingulate / ventral emotional-network node implicated in REM affective processing."
            ),
            "dlpfc": (
                "Dorsal executive network node modeled as relative REM deactivation / weak top-down control."
            ),
            "mpfc_proxy": (
                "Medial prefrontal regulatory proxy capturing systems-level REM emotion regulation claims."
            ),
            "brainstem_rem_proxy": (
                "Brainstem REM-generation proxy for cholinergic REM drive and awakening-related dyscontrol."
            ),
        }
        self.proxy_regions = {"mpfc_proxy", "brainstem_rem_proxy"}

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "stress_sensitized_sleep_liability",
                "relation": "contributes exploratory heritable vulnerability in sleep and emotion regulation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "trauma_history",
                "target": "stress_sensitized_sleep_liability",
                "relation": "traumatic exposure increases later threat-related dream vulnerability",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "early_life_adversity",
                "target": "stress_sensitized_sleep_liability",
                "relation": "early adversity may create long-lasting stress-reactive susceptibility",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "depression_anxiety_burden",
                "target": "stress_sensitized_sleep_liability",
                "relation": "affective burden amplifies vulnerability to REM-emotional dysregulation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "serotonergic_perturbation",
                "target": "serotonergic_dysregulation",
                "relation": "altered serotonergic tone can destabilize REM regulation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "depression_anxiety_burden",
                "target": "serotonergic_dysregulation",
                "relation": "mood and anxiety burden are linked to serotonergic sleep-wake dysregulation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "gabaergic_hypnotic_exposure",
                "target": "sleep_architecture_disruption",
                "relation": "hypnotic manipulation can alter REM and slow-wave sleep architecture",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "sleep_architecture_disruption",
                "relation": "stress destabilizes sleep continuity and regulation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "depression_anxiety_burden",
                "target": "cholinergic_rem_drive",
                "relation": "depression-linked REM pressure may reflect stronger cholinergic drive",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "cholinergic_rem_drive",
                "relation": "REM gating imbalance may allow stronger cholinergic dream-promoting influence",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "trauma_history",
                "target": "hpa_axis_dysregulation",
                "relation": "trauma perturbs stress-system regulation and cortisol dynamics",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "early_life_adversity",
                "target": "hpa_axis_dysregulation",
                "relation": "epigenetic stress sensitization may dysregulate the HPA axis",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hpa_axis_dysregulation",
                "relation": "ongoing stress maintains maladaptive HPA activation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "rem_sleep_instability",
                "relation": "altered serotonin signaling destabilizes REM sleep",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "sleep_architecture_disruption",
                "target": "rem_sleep_instability",
                "relation": "broader state instability makes REM more dysregulated",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "cholinergic_rem_drive",
                "target": "rem_sleep_instability",
                "relation": "heightened REM drive can intensify vivid REM phenomena",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "noradrenergic_rem_arousal",
                "relation": "stress-system dysregulation increases REM-period over-arousal",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "rem_sleep_instability",
                "relation": "stress physiology promotes unstable REM-emotional processing",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "rem_sleep_instability",
                "target": "executive_dream_regulation_failure",
                "relation": "unstable REM weakens top-down regulation of dream content",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "executive_dream_regulation_failure",
                "relation": "stress burden further weakens regulatory control during REM",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "rem_sleep_instability",
                "target": "frontolimbic_dream_dysregulation",
                "relation": "REM instability favors emotional-limbic dominance in dreams",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "stress_sensitized_sleep_liability",
                "target": "frontolimbic_dream_dysregulation",
                "relation": "trait vulnerability amplifies emotional-dream dysregulation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "executive_dream_regulation_failure",
                "target": "mpfc_proxy",
                "relation": "reduced medial prefrontal regulation is expressed as weaker REM control",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "executive_dream_regulation_failure",
                "target": "dlpfc",
                "relation": "dorsal executive network becomes relatively deactivated during REM",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "frontolimbic_dream_dysregulation",
                "target": "amygdala",
                "relation": "limbic-dominant REM processing heightens threat affect",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "frontolimbic_dream_dysregulation",
                "target": "hippocampus",
                "relation": "emotionally loaded memory/context reactivation shapes dream content",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "frontolimbic_dream_dysregulation",
                "target": "acc",
                "relation": "ventral emotional network engagement includes anterior cingulate involvement",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "rem_sleep_instability",
                "target": "brainstem_rem_proxy",
                "relation": "dysregulated REM engages brainstem REM-generative systems",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "cholinergic_rem_drive",
                "target": "brainstem_rem_proxy",
                "relation": "brainstem cholinergic REM promotion intensifies vivid dream generation",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "recurrent_threat_nightmares",
                "relation": "threat salience drives disturbing dream content",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "recurrent_threat_nightmares",
                "relation": "memory-context reactivation contributes to elaborate dream narratives",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "brainstem_rem_proxy",
                "target": "recurrent_threat_nightmares",
                "relation": "REM generative systems support recurrent dream expression",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "nightmare_emotional_intensity",
                "relation": "amygdala hyperreactivity intensifies negative dream affect",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "acc",
                "target": "nightmare_emotional_intensity",
                "relation": "ACC engagement contributes emotional salience and distress",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "noradrenergic_rem_arousal",
                "target": "abrupt_awakening_alertness",
                "relation": "REM over-arousal facilitates sudden awakening with alertness",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "brainstem_rem_proxy",
                "target": "abrupt_awakening_alertness",
                "relation": "REM-brainstem dyscontrol contributes to awakening from nightmares",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "rem_sleep_instability",
                "target": "rem_parasomnia_liability",
                "relation": "unstable REM increases nightmare parasomnia vulnerability",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "rem_parasomnia_liability",
                "relation": "5-HT dysregulation can facilitate REM parasomnia expression",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "recurrent_threat_nightmares",
                "target": "sleep_disruption",
                "relation": "nightmares fragment sleep continuity",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "abrupt_awakening_alertness",
                "target": "sleep_disruption",
                "relation": "sudden alert awakenings further disturb sleep maintenance",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "sleep_disruption",
                "target": "daytime_distress_impairment",
                "relation": "fragmented sleep contributes to daytime burden and dysfunction",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "nightmare_emotional_intensity",
                "target": "daytime_distress_impairment",
                "relation": "emotionally intense nightmares increase distress and impairment",
                "nightmare_disorder_change": "increased",
            },
            {
                "source": "recovery_sleep_support",
                "target": "hpa_axis_dysregulation",
                "relation": "supportive recovery conditions buffer stress-system dysregulation",
                "nightmare_disorder_change": "decreased",
            },
            {
                "source": "recovery_sleep_support",
                "target": "rem_sleep_instability",
                "relation": "sleep stabilization reduces REM instability",
                "nightmare_disorder_change": "decreased",
            },
            {
                "source": "recovery_sleep_support",
                "target": "executive_dream_regulation_failure",
                "relation": "supportive treatment and recovery structure improve top-down regulation",
                "nightmare_disorder_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _normalize_label_text(text: Any) -> str:
        s = str(text).lower().strip()
        for token in [
            " left",
            " right",
            " (pacc)",
            " (acc)",
            " (dlpfc)",
            " (amygdala)",
            " (hippocampus)",
            " (fpole)",
        ]:
            s = s.replace(token, "")
        s = s.replace("medial prefrontal cortex", "medial prefrontal")
        s = s.replace("dorsolateral prefrontal cortex", "dorsolateral prefrontal")
        return " ".join(s.split())

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        for modality in modalities:
            try:
                with siibra.QUIET:
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        matches: List[Any] = []
        try:
            matches = list(
                self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
            )
        except Exception:
            try:
                matches = list(self.parcellation.find(query))
            except Exception:
                matches = []

        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self._name_of(self.parcellation) == str(parc_name):
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "anterior cingulate",
            "medial prefrontal cortex",
            "dorsolateral prefrontal cortex",
            "brainstem",
            "pons",
        } else 0
        proxy_penalty = 1 if any(tok in name for tok in ["brainstem", "pons", "reticular"] ) else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass
            try:
                if hasattr(self.parcellation, "get_region"):
                    return self.parcellation.get_region(spec)
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if row in seen:
                continue
            seen.add(row)
            rows.append(
                {
                    "name": row[0],
                    "identifier": row[1],
                    "parcellation": row[2],
                }
            )
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows)

    def _spatial_props_list(self, region: Any) -> List[Any]:
        try:
            props = region.spatial_props(space=self.space)
        except Exception:
            return []
        if props is None:
            return []
        if isinstance(props, dict):
            return list(props.values())
        if isinstance(props, (list, tuple)):
            return list(props)
        if hasattr(props, "components"):
            return list(getattr(props, "components", []))
        return [props]

    def _point_to_tuple(self, point: Any) -> Optional[Tuple[float, float, float]]:
        if point is None:
            return None
        for attr in ("coordinate", "coordinates", "xyz"):
            value = getattr(point, attr, None)
            if value is not None:
                try:
                    return tuple(float(x) for x in value)  # type: ignore[arg-type]
                except Exception:
                    pass
        try:
            return tuple(float(x) for x in point)  # type: ignore[arg-type]
        except Exception:
            return None

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid_xyz = self._point_to_tuple(getattr(main, "centroid", None))
        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy()
                if isinstance(df, pd.Series):
                    df = df.to_frame(name="value")
                if not isinstance(df, pd.DataFrame):
                    continue
                df = df.reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
            except Exception:
                continue
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy()
            except Exception:
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            lower_cols = {str(c).lower(): c for c in df.columns}
            required = {"gene", "level", "zscore"}
            if required.issubset(lower_cols):
                gene_col = lower_cols["gene"]
                level_col = lower_cols["level"]
                zscore_col = lower_cols["zscore"]
                try:
                    return (
                        df.groupby(gene_col, dropna=False)
                        .agg(
                            level_mean=(level_col, "mean"),
                            level_std=(level_col, "std"),
                            probe_count=(level_col, "count"),
                            zscore_mean=(zscore_col, "mean"),
                            zscore_std=(zscore_col, "std"),
                        )
                        .reset_index()
                        .rename(columns={gene_col: "gene"})
                        .sort_values("gene")
                        .reset_index(drop=True)
                    )
                except Exception:
                    pass
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _first_dataframe(self, value: Any) -> Optional[pd.DataFrame]:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame()
        return None

    def _mean_connectivity_from_compound(self, compound: Any) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        try:
            direct = self._first_dataframe(getattr(compound, "data", None))
            if direct is not None:
                return direct
        except Exception:
            pass

        try:
            iterable: Iterable[Any] = compound[: self.connectivity_subjects_to_average]
        except Exception:
            iterable = []
            try:
                iterable = [compound[i] for i in range(self.connectivity_subjects_to_average)]
            except Exception:
                iterable = []

        for element in iterable:
            try:
                df = self._first_dataframe(getattr(element, "data", None))
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]

        try:
            total = frames[0].astype(float).copy()
            for df in frames[1:]:
                total = total.add(df.astype(float), fill_value=0.0)
            return total / float(len(frames))
        except Exception:
            return frames[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )
        self._connectivity_matrix = self._mean_connectivity_from_compound(compound)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        region_norm = self._normalize_label_text(region_name)

        for label in labels:
            if self._name_of(label) == region_name:
                return label

        exactish = []
        fuzzy = []
        for label in labels:
            label_name = self._name_of(label)
            label_norm = self._normalize_label_text(label_name)
            if label_norm == region_norm:
                exactish.append(label)
            elif region_norm in label_norm or label_norm in region_norm:
                fuzzy.append(label)

        pool = exactish if exactish else fuzzy
        if not pool:
            return None
        return sorted(pool, key=lambda x: self._region_rank(x))[0]

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)
        label = row_label if row_label is not None else col_label
        axis = "index" if row_label is not None else "columns"
        if label is None:
            return pd.DataFrame()

        try:
            series = matrix.loc[label] if axis == "index" else matrix[label]
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            df = series.dropna().sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        pairs: List[Tuple[str, Any]] = []
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                pairs.append((key, label))

        if not pairs:
            return pd.DataFrame()

        labels = [label for _, label in pairs]
        keys = [key for key, _ in pairs]
        try:
            sub = matrix.loc[labels, labels].copy()
            sub.index = keys
            sub.columns = keys
            return sub
        except Exception:
            return pd.DataFrame()

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

        for key, desc in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": desc,
                    "region_role": None,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            description = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
            region_role = "proxy" if key in self.proxy_regions else "atlas_backed"

            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": f"{description} (unresolved in this environment)",
                        "region_role": region_role,
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved",
                    }
                )
                continue

            self.region_objects[key] = region
            centroid_mni, volume_mm3 = self._main_component(region)
            receptor_df = self._receptor_table(region)
            gene_df = self._gene_table(region, gene_panel)
            conn_df = self._connectivity_profile(region, max_rows=connectivity_rows)
            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = conn_df

            nodes.append(
                {
                    "key": key,
                    "label": self._name_of(region),
                    "node_type": "region",
                    "description": description,
                    "region_role": region_role,
                    "atlas_region": self._name_of(region),
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not conn_df.empty else 'no'}"
                    ),
                }
            )

        for key, desc in self.latent_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "latent_biology",
                    "description": desc,
                    "region_role": None,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, desc in self.symptom_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "symptom",
                    "description": desc,
                    "region_role": None,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(self.edge_table)

        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(),
        }

    def simulate(
        self,
        genetic_vulnerability: float = 0.50,
        trauma_history: float = 0.50,
        early_life_adversity: float = 0.50,
        chronic_stress_load: float = 0.50,
        depression_anxiety_burden: float = 0.50,
        serotonergic_perturbation: float = 0.50,
        gabaergic_hypnotic_exposure: float = 0.30,
        recovery_sleep_support: float = 0.50,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator on normalized 0..1 inputs.

        The order is:
        inputs -> latent biology -> regional state -> symptoms -> phenotypes

        These are not probabilities, diagnoses, or treatment predictions.
        They are interpretable scaffold scores reflecting chapter logic.
        """
        c = self._clip01

        inputs = pd.Series(
            {
                "genetic_vulnerability": c(genetic_vulnerability),
                "trauma_history": c(trauma_history),
                "early_life_adversity": c(early_life_adversity),
                "chronic_stress_load": c(chronic_stress_load),
                "depression_anxiety_burden": c(depression_anxiety_burden),
                "serotonergic_perturbation": c(serotonergic_perturbation),
                "gabaergic_hypnotic_exposure": c(gabaergic_hypnotic_exposure),
                "recovery_sleep_support": c(recovery_sleep_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["stress_sensitized_sleep_liability"] = c(
            0.26 * inputs["genetic_vulnerability"]
            + 0.24 * inputs["trauma_history"]
            + 0.22 * inputs["early_life_adversity"]
            + 0.18 * inputs["depression_anxiety_burden"]
            + 0.08 * inputs["chronic_stress_load"]
            - 0.14 * inputs["recovery_sleep_support"]
        )
        latents["serotonergic_dysregulation"] = c(
            0.44 * inputs["serotonergic_perturbation"]
            + 0.26 * inputs["depression_anxiety_burden"]
            + 0.14 * inputs["chronic_stress_load"]
            - 0.10 * inputs["recovery_sleep_support"]
        )
        latents["sleep_architecture_disruption"] = c(
            0.48 * inputs["gabaergic_hypnotic_exposure"]
            + 0.18 * inputs["chronic_stress_load"]
            + 0.12 * inputs["depression_anxiety_burden"]
            - 0.08 * inputs["recovery_sleep_support"]
        )
        latents["cholinergic_rem_drive"] = c(
            0.34 * inputs["depression_anxiety_burden"]
            + 0.24 * latents["serotonergic_dysregulation"]
            + 0.16 * inputs["chronic_stress_load"]
            + 0.10 * latents["stress_sensitized_sleep_liability"]
        )
        latents["hpa_axis_dysregulation"] = c(
            0.30 * inputs["trauma_history"]
            + 0.30 * inputs["chronic_stress_load"]
            + 0.20 * inputs["early_life_adversity"]
            + 0.12 * inputs["depression_anxiety_burden"]
            - 0.18 * inputs["recovery_sleep_support"]
        )
        latents["noradrenergic_rem_arousal"] = c(
            0.40 * latents["hpa_axis_dysregulation"]
            + 0.20 * inputs["trauma_history"]
            + 0.16 * inputs["chronic_stress_load"]
            + 0.12 * latents["stress_sensitized_sleep_liability"]
            - 0.08 * inputs["recovery_sleep_support"]
        )
        latents["rem_sleep_instability"] = c(
            0.28 * latents["serotonergic_dysregulation"]
            + 0.24 * latents["sleep_architecture_disruption"]
            + 0.22 * latents["cholinergic_rem_drive"]
            + 0.16 * latents["hpa_axis_dysregulation"]
            + 0.08 * inputs["depression_anxiety_burden"]
            - 0.18 * inputs["recovery_sleep_support"]
        )
        latents["executive_dream_regulation_failure"] = c(
            0.36 * latents["rem_sleep_instability"]
            + 0.22 * latents["hpa_axis_dysregulation"]
            + 0.16 * inputs["depression_anxiety_burden"]
            + 0.10 * latents["stress_sensitized_sleep_liability"]
            - 0.18 * inputs["recovery_sleep_support"]
        )
        latents["frontolimbic_dream_dysregulation"] = c(
            0.34 * latents["rem_sleep_instability"]
            + 0.24 * latents["hpa_axis_dysregulation"]
            + 0.20 * latents["stress_sensitized_sleep_liability"]
            + 0.12 * latents["noradrenergic_rem_arousal"]
            - 0.16 * inputs["recovery_sleep_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["amygdala"] = c(
            0.46 * latents["frontolimbic_dream_dysregulation"]
            + 0.24 * latents["noradrenergic_rem_arousal"]
            + 0.14 * inputs["trauma_history"]
            + 0.08 * latents["rem_sleep_instability"]
        )
        regional_state["hippocampus"] = c(
            0.36 * latents["frontolimbic_dream_dysregulation"]
            + 0.24 * inputs["trauma_history"]
            + 0.18 * latents["rem_sleep_instability"]
            + 0.10 * latents["stress_sensitized_sleep_liability"]
        )
        regional_state["acc"] = c(
            0.38 * latents["frontolimbic_dream_dysregulation"]
            + 0.24 * latents["rem_sleep_instability"]
            + 0.16 * latents["hpa_axis_dysregulation"]
            + 0.08 * inputs["depression_anxiety_burden"]
        )
        regional_state["mpfc_proxy"] = c(
            0.44 * latents["executive_dream_regulation_failure"]
            + 0.22 * latents["hpa_axis_dysregulation"]
            + 0.14 * latents["frontolimbic_dream_dysregulation"]
            - 0.16 * inputs["recovery_sleep_support"]
        )
        regional_state["dlpfc"] = c(
            0.48 * latents["executive_dream_regulation_failure"]
            + 0.22 * latents["rem_sleep_instability"]
            + 0.12 * inputs["depression_anxiety_burden"]
            - 0.20 * inputs["recovery_sleep_support"]
        )
        regional_state["brainstem_rem_proxy"] = c(
            0.38 * latents["rem_sleep_instability"]
            + 0.28 * latents["cholinergic_rem_drive"]
            + 0.16 * latents["sleep_architecture_disruption"]
            + 0.12 * latents["noradrenergic_rem_arousal"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["recurrent_threat_nightmares"] = c(
            0.28 * regional_state["amygdala"]
            + 0.24 * regional_state["hippocampus"]
            + 0.24 * latents["rem_sleep_instability"]
            + 0.12 * regional_state["brainstem_rem_proxy"]
            + 0.08 * inputs["trauma_history"]
        )
        symptoms["nightmare_emotional_intensity"] = c(
            0.32 * regional_state["amygdala"]
            + 0.22 * regional_state["acc"]
            + 0.20 * latents["cholinergic_rem_drive"]
            + 0.16 * latents["hpa_axis_dysregulation"]
            + 0.08 * symptoms["recurrent_threat_nightmares"]
        )
        symptoms["abrupt_awakening_alertness"] = c(
            0.36 * latents["noradrenergic_rem_arousal"]
            + 0.30 * regional_state["brainstem_rem_proxy"]
            + 0.18 * latents["rem_sleep_instability"]
            + 0.06 * symptoms["nightmare_emotional_intensity"]
        )
        symptoms["rem_parasomnia_liability"] = c(
            0.34 * latents["rem_sleep_instability"]
            + 0.24 * latents["serotonergic_dysregulation"]
            + 0.22 * latents["sleep_architecture_disruption"]
            + 0.12 * regional_state["brainstem_rem_proxy"]
        )
        symptoms["sleep_disruption"] = c(
            0.28 * symptoms["recurrent_threat_nightmares"]
            + 0.26 * symptoms["abrupt_awakening_alertness"]
            + 0.22 * symptoms["rem_parasomnia_liability"]
            + 0.14 * latents["sleep_architecture_disruption"]
            - 0.10 * inputs["recovery_sleep_support"]
        )
        symptoms["daytime_distress_impairment"] = c(
            0.30 * symptoms["nightmare_emotional_intensity"]
            + 0.28 * symptoms["sleep_disruption"]
            + 0.18 * symptoms["recurrent_threat_nightmares"]
            + 0.14 * inputs["depression_anxiety_burden"]
            - 0.10 * inputs["recovery_sleep_support"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["trauma_stress_nightmare_profile"] = c(
            self._mean(
                [
                    latents["hpa_axis_dysregulation"],
                    regional_state["amygdala"],
                    symptoms["recurrent_threat_nightmares"],
                    symptoms["daytime_distress_impairment"],
                ]
            )
        )
        phenotypes["rem_instability_parasomnia_profile"] = c(
            self._mean(
                [
                    latents["rem_sleep_instability"],
                    regional_state["brainstem_rem_proxy"],
                    symptoms["abrupt_awakening_alertness"],
                    symptoms["rem_parasomnia_liability"],
                ]
            )
        )
        phenotypes["affective_nightmare_profile"] = c(
            self._mean(
                [
                    latents["serotonergic_dysregulation"],
                    symptoms["nightmare_emotional_intensity"],
                    symptoms["sleep_disruption"],
                    symptoms["daytime_distress_impairment"],
                ]
            )
        )
        phenotypes["executive_dream_regulation_profile"] = c(
            self._mean(
                [
                    latents["executive_dream_regulation_failure"],
                    regional_state["mpfc_proxy"],
                    regional_state["dlpfc"],
                    symptoms["recurrent_threat_nightmares"],
                ]
            )
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in getattr(assignments, "columns", []):
                return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Optional[Any]:
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(self.space)
            except Exception:
                return None


if __name__ == "__main__":
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 180)

    model = NightmareDisorderModel()
    built = model.build()

    print("\n=== Nodes ===")
    print(
        built["nodes"][
            [
                "key",
                "node_type",
                "region_role",
                "atlas_region",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== Edges ===")
    print(built["edges"].to_string(index=False))

    print("\n=== Region suggestions for 'amygdala' ===")
    print(model.suggest_regions("amygdala").head(10).to_string(index=False))

    for key in ["amygdala", "hippocampus", "acc", "dlpfc", "mpfc_proxy", "brainstem_rem_proxy"]:
        print(f"\n=== Region evidence: {key} ===")
        if key in built["regions"]:
            region = built["regions"][key]
            print("Resolved region:", getattr(region, "name", region))
        else:
            print("Resolved region: None")
        print("Receptors:")
        rec = built["receptors"].get(key, pd.DataFrame())
        print(rec.head(10).to_string(index=False) if not rec.empty else "<no receptor table>")
        print("Genes:")
        gene = built["genes"].get(key, pd.DataFrame())
        print(gene.head(10).to_string(index=False) if not gene.empty else "<no gene table>")
        print("Connectivity profile:")
        conn = built["connectivity_profiles"].get(key, pd.DataFrame())
        print(conn.head(10).to_string(index=False) if not conn.empty else "<no connectivity profile>")

    print("\n=== Circuit connectivity ===")
    cmat = built["circuit_connectivity"]
    print(cmat.to_string() if not cmat.empty else "<no circuit connectivity matrix>")

    sim = model.simulate(
        genetic_vulnerability=0.55,
        trauma_history=0.80,
        early_life_adversity=0.65,
        chronic_stress_load=0.70,
        depression_anxiety_burden=0.75,
        serotonergic_perturbation=0.60,
        gabaergic_hypnotic_exposure=0.35,
        recovery_sleep_support=0.30,
    )

    print("\n=== Simulation: inputs ===")
    print(sim["inputs"].to_string())
    print("\n=== Simulation: latents ===")
    print(sim["latents"].sort_values(ascending=False).to_string())
    print("\n=== Simulation: regional state ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())
    print("\n=== Simulation: symptoms ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())
    print("\n=== Simulation: phenotypes ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment if running in an environment with siibra installed:
    # print(model.assign_mni_point((0, -20, -12)).head())
