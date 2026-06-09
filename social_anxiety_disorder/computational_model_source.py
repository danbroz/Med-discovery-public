from __future__ import annotations

"""
Social Anxiety Disorder siibra scaffold.

This script turns a chapter-level biological summary of Social Anxiety
Disorder (SAD) into a conservative, atlas-grounded mechanistic scaffold using
siibra where available. It is intended for research prototyping and transparent
hypothesis exploration only. It is not a diagnostic, prognostic, or treatment
recommendation tool.

Core modeling choices from the chapter:
- SAD is modeled as a gene-environment-sensitive disorder in which early
  temperament, stress biology, and transmitter dysregulation interact with
  social novelty and evaluative threat.
- Serotonin, norepinephrine, GABA, epigenetic stress sensitization, and HPA
  axis burden are kept primarily as latent biology rather than over-forcing
  them into specific parcels.
- Region anchors are conservative: amygdala, hippocampus, a prefrontal-control
  proxy, and a locus-coeruleus proxy, because these are the structures named or
  strongly implied by the chapter.
- A central circuit claim from the chapter is preserved explicitly: weakened
  top-down regulation from PFC to amygdala permits fear responses to persist.
- The simulator is intentionally simple and acyclic:
  inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

The script degrades gracefully:
- If siibra is not installed, or if a feature/modality is unavailable, the
  atlas-backed parts stay empty instead of crashing.
- The simulator still runs even without atlas data.

Source chapter themes encoded here include:
- serotonergic, noradrenergic, and GABAergic contributions to anxiety,
- behavioral inhibition as an early temperamental risk trait,
- early adversity and epigenetic stress sensitization,
- HPA-axis hyperreactivity,
- weakened PFC-amygdala connectivity,
- amygdala, hippocampus, and PFC contributions to persistent social fear,
- visible-anxiety concern driven by autonomic arousal and negative evaluation.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
    _SIIBRA_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc


DEFAULT_GENE_PANEL = [
    # Serotonergic signaling
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    # Noradrenergic signaling / arousal
    "SLC6A2",
    "ADRA2A",
    # GABAergic inhibition
    "GAD1",
    "GABRA2",
    "GABRB2",
    # Stress responsivity / epigenetic vulnerability
    "CRHR1",
    "NR3C1",
    "FKBP5",
    # Plasticity / regulation / temperament-linked control
    "BDNF",
    "COMT",
    "MAOA",
]


class SocialAnxietyDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Social Anxiety Disorder.

    Notes
    -----
    - This is a research scaffold, not a validated disease model.
    - Regional values in `simulate()` quantify dysregulation / burden rather
      than healthy activation.
    - Proxies are used where the chapter stays systems-level or where a stable
      Julich label may vary across environments.
    - The model emphasizes social novelty, evaluative threat, autonomic arousal,
      and impaired top-down emotional regulation.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._atlas_available = False

        if siibra is None:
            warnings.warn(
                f"siibra could not be imported ({_SIIBRA_IMPORT_ERROR}). "
                "Atlas-backed methods will remain available only as graceful stubs; "
                "the simulator still works."
            )
        else:
            try:
                self.atlas = siibra.atlases.get(atlas_spec)
                self.parcellation = (
                    self.atlas.get_parcellation(parcellation_spec)
                    if hasattr(self.atlas, "get_parcellation")
                    else self.atlas.parcellations.get(parcellation_spec)
                )
                self.space = (
                    self.atlas.get_space(space_spec)
                    if hasattr(self.atlas, "get_space")
                    else self.atlas.spaces.get(space_spec)
                )
                self._atlas_available = (
                    self.atlas is not None and self.parcellation is not None and self.space is not None
                )
            except Exception as exc:
                warnings.warn(
                    f"Could not initialize siibra atlas resources: {exc}. "
                    "Atlas-backed helpers will degrade gracefully."
                )

        # Conservative region anchors from the chapter's explicitly named
        # circuitry. Proxies are preferred when the text stays broader than a
        # single stable cytoarchitectonic label.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "LA (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "CA left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 8Av left",
                "dorsolateral prefrontal cortex left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
            "locus_coeruleus_proxy": [
                "locus coeruleus left",
                "locus coeruleus",
                "brainstem",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Amygdala anchor for social threat tagging, fear reactivity, and "
                "persistent response to evaluative cues."
            ),
            "hippocampus": (
                "Hippocampal anchor for stress-sensitive contextual memory and "
                "anticipatory recall of socially threatening experiences."
            ),
            "pfc_control": (
                "Prefrontal-control proxy for top-down regulation, reappraisal, and "
                "inhibition of exaggerated social fear responses."
            ),
            "locus_coeruleus_proxy": (
                "Locus-coeruleus proxy for noradrenergic arousal, vigilance, and "
                "autonomic symptom amplification."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Polygenic liability influencing temperament, transmitter regulation, "
                "and stress responsivity."
            ),
            "behavioral_inhibition_trait": (
                "Early-emerging shy, cautious, withdrawn temperament that increases "
                "risk for social anxiety under novelty or social evaluation."
            ),
            "early_life_adversity": (
                "Adverse early experiences capable of biologically embedding stress "
                "sensitivity through lasting molecular changes."
            ),
            "current_social_stress": (
                "Current social-evaluative stress load that amplifies arousal, fear, "
                "and avoidance."
            ),
            "social_novelty_load": (
                "Exposure to unfamiliar social situations that engages the innate "
                "novelty-fear bias described in the chapter."
            ),
            "serotonergic_treatment_support": (
                "Protective serotonergic support representing SSRI-like stabilization of "
                "anxiety and negative emotional bias."
            ),
            "recovery_support": (
                "Protective coping, environmental support, and recovery-promoting "
                "conditions that buffer stress and improve regulation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": (
                "Altered serotonin signaling contributing to heightened anxiety and a "
                "negative emotional bias."
            ),
            "noradrenergic_hyperarousal": (
                "Excess norepinephrine-linked arousal contributing to sweating, tremor, "
                "tachycardia, and vigilance."
            ),
            "gabaergic_disinhibition": (
                "Reduced inhibitory buffering of fear-related circuitry, permitting "
                "neuronal hyperexcitability under social stress."
            ),
            "epigenetic_stress_sensitization": (
                "Stress-linked biological embedding of adversity that amplifies later "
                "emotional and stress reactivity."
            ),
            "hpa_axis_hyperreactivity": (
                "Hyper-reactive stress-axis burden that increases sensitivity to social "
                "threat and evaluative stress."
            ),
            "social_novelty_threat_bias": (
                "Temperament-linked bias to treat unfamiliar social situations as "
                "threatening and avoidant."
            ),
            "autonomic_self_monitoring_amplification": (
                "Escalating focus on bodily arousal as a visible sign of anxiety, which "
                "intensifies fear of negative evaluation."
            ),
            "frontolimbic_regulation_failure": (
                "Weak top-down regulation from prefrontal systems over limbic fear "
                "reactivity, especially the PFC-amygdala pathway."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "anticipatory_social_anxiety": (
                "Persistent anticipatory anxiety before social or performance situations."
            ),
            "fear_of_negative_evaluation": (
                "Fear of being judged negatively, embarrassed, or exposed as anxious."
            ),
            "physiological_hyperarousal": (
                "Autonomic anxiety symptoms such as sweating, tremor, flushing, and "
                "increased heart rate."
            ),
            "visible_anxiety_concern": (
                "Fear that bodily anxiety signs will be noticed and judged by others."
            ),
            "social_threat_hypervigilance": (
                "Excess monitoring for social danger cues, criticism, or signs of rejection."
            ),
            "social_avoidance": (
                "Avoidance or withdrawal from novel or evaluative social situations."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "social_novelty_threat_bias",
                "relation": "heritable liability shapes temperamental predisposition toward social fear and withdrawal",
                "sad_change": "increased",
            },
            {
                "source": "behavioral_inhibition_trait",
                "target": "social_novelty_threat_bias",
                "relation": "behavioral inhibition increases fear responses to unfamiliar social situations",
                "sad_change": "increased",
            },
            {
                "source": "early_life_adversity",
                "target": "epigenetic_stress_sensitization",
                "relation": "adverse early experience can induce lasting epigenetic changes in stress-regulation pathways",
                "sad_change": "increased",
            },
            {
                "source": "epigenetic_stress_sensitization",
                "target": "hpa_axis_hyperreactivity",
                "relation": "biological embedding of adversity sensitizes later stress responses",
                "sad_change": "increased",
            },
            {
                "source": "current_social_stress",
                "target": "hpa_axis_hyperreactivity",
                "relation": "ongoing social-evaluative stress activates stress-axis burden",
                "sad_change": "increased",
            },
            {
                "source": "social_novelty_load",
                "target": "social_novelty_threat_bias",
                "relation": "novel social situations engage the predisposed fear-to-unfamiliarity response",
                "sad_change": "increased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "serotonergic_dysregulation",
                "relation": "SSRI-like support partially recalibrates serotonin signaling",
                "sad_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "frontolimbic_regulation_failure",
                "relation": "protective support improves emotional regulation and buffering",
                "sad_change": "decreased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "genetic liability contributes to altered serotonin signaling",
                "sad_change": "increased",
            },
            {
                "source": "current_social_stress",
                "target": "noradrenergic_hyperarousal",
                "relation": "stress increases norepinephrine-linked arousal",
                "sad_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "amygdala",
                "relation": "reduced inhibitory tone permits fear-circuit hyperexcitability",
                "sad_change": "hyperreactive",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "amygdala",
                "relation": "serotonergic dysregulation contributes to negative emotional bias and heightened social fear",
                "sad_change": "hyperreactive",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "locus_coeruleus_proxy",
                "relation": "excess arousal burden centers on locus-coeruleus-linked vigilance circuitry",
                "sad_change": "hyperaroused",
            },
            {
                "source": "hpa_axis_hyperreactivity",
                "target": "hippocampus",
                "relation": "stress-axis burden compromises hippocampal contextual regulation",
                "sad_change": "dysregulated",
            },
            {
                "source": "hpa_axis_hyperreactivity",
                "target": "pfc_control",
                "relation": "stress burden weakens regulatory prefrontal control",
                "sad_change": "dysregulated",
            },
            {
                "source": "frontolimbic_regulation_failure",
                "target": "pfc_control",
                "relation": "regulatory failure reduces prefrontal inhibition of fear responses",
                "sad_change": "dysregulated",
            },
            {
                "source": "frontolimbic_regulation_failure",
                "target": "amygdala",
                "relation": "weakened top-down control allows amygdala-mediated social fear to persist unchecked",
                "sad_change": "hyperreactive",
            },
            {
                "source": "social_novelty_threat_bias",
                "target": "amygdala",
                "relation": "temperamental social-threat bias amplifies limbic salience to unfamiliar people or situations",
                "sad_change": "hyperreactive",
            },
            {
                "source": "autonomic_self_monitoring_amplification",
                "target": "fear_of_negative_evaluation",
                "relation": "attention to visible bodily arousal intensifies fear of being judged",
                "sad_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "physiological_hyperarousal",
                "relation": "noradrenergic arousal drives sweating, tremor, flushing, and increased heart rate",
                "sad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anticipatory_social_anxiety",
                "relation": "amygdala hyperreactivity sustains fear before social encounters",
                "sad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "social_threat_hypervigilance",
                "relation": "amygdala hyperreactivity increases monitoring for social threat cues",
                "sad_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "anticipatory_social_anxiety",
                "relation": "stress-sensitive contextual memory can amplify anticipatory recall of prior threat",
                "sad_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "fear_of_negative_evaluation",
                "relation": "weak prefrontal regulation fails to down-regulate socially evaluative fear",
                "sad_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "social_avoidance",
                "relation": "reduced top-down control promotes withdrawal rather than flexible engagement",
                "sad_change": "increased",
            },
            {
                "source": "physiological_hyperarousal",
                "target": "visible_anxiety_concern",
                "relation": "bodily symptoms become evidence of possible social embarrassment",
                "sad_change": "increased",
            },
            {
                "source": "fear_of_negative_evaluation",
                "target": "social_avoidance",
                "relation": "fear of judgment motivates avoidance of social exposure",
                "sad_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if siibra is None:
            return []

        cands: List[Any] = []
        try:
            if kind == "receptor":
                cands.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                cands.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                cands.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            cands.append("receptor density fingerprint")
        elif kind == "gene":
            cands.append("gene expressions")
        elif kind == "connectivity":
            cands.append("StreamlineCounts")
        return cands

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        if siibra is None or concept is None:
            return []
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
        if self.atlas is None:
            return []
        try:
            matches = self.atlas.find_regions(
                query,
                all_versions=False,
                filter_children=False,
                find_topmost=False,
            )
        except Exception:
            return []

        out = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        """
        Lower tuples rank better.
        Prefer left hemisphere, specific labels, and Julich labels over generic names.
        """
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "prefrontal cortex",
            "hippocampus",
            "amygdala",
            "brainstem",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None or self.parcellation is None:
            return None

        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.parcellation.get_region(spec)
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
        seen = set()
        matches = self._julich_matches(keyword)
        for region in sorted(matches, key=self._region_rank):
            row = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if row in seen:
                continue
            seen.add(row)
            rows.append(
                {"name": row[0], "identifier": row[1], "parcellation": row[2]}
            )
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows)

    def _spatial_props_list(self, region: Any) -> List[Any]:
        if region is None or self.space is None:
            return []
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

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = feats[0].data.copy().reset_index()
            if "index" in df.columns and "receptor" not in df.columns:
                df = df.rename(columns={"index": "receptor"})
            return df
        except Exception:
            return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        if not genes:
            return pd.DataFrame()

        feats = self._safe_features_any(
            region,
            self._modality_candidates("gene"),
            gene=list(genes),
        )
        if not feats:
            return pd.DataFrame()
        try:
            df = feats[0].data.copy()
        except Exception:
            return pd.DataFrame()

        lower_cols = {c.lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
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
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(
            self.parcellation,
            self._modality_candidates("connectivity"),
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            self._connectivity_matrix = compound[0].data.copy()
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None

        for x in labels:
            if x is region:
                return x

        exact = [x for x in labels if self._name_of(x) == self._name_of(region)]
        if exact:
            return exact[0]

        rn = self._name_of(region).lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        label = self._match_region_label(list(matrix.index), region)
        axis = "index"
        if label is None:
            label = self._match_region_label(list(matrix.columns), region)
            axis = "columns"
        if label is None:
            return pd.DataFrame()

        try:
            series = matrix.loc[label] if axis == "index" else matrix[label]
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a within-model connectivity submatrix for resolved region nodes,
        using fuzzy matching against the selected connectivity matrix.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        selected_labels: Dict[str, Any] = {}
        for node_key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None and col_label is not None:
                if row_label in matrix.index and col_label in matrix.columns:
                    selected_labels[node_key] = row_label

        if not selected_labels:
            return pd.DataFrame()

        common = [lbl for lbl in selected_labels.values() if lbl in matrix.index and lbl in matrix.columns]
        if not common:
            return pd.DataFrame()

        try:
            sub = matrix.loc[common, common].copy()
        except Exception:
            return pd.DataFrame()

        inverse = {v: k for k, v in selected_labels.items()}
        sub.index = [inverse.get(x, self._name_of(x)) for x in sub.index]
        sub.columns = [inverse.get(x, self._name_of(x)) for x in sub.columns]
        return sub

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

        nodes: List[Dict[str, Any]] = []

        for key, desc in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": desc,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
            if region is None:
                if self._atlas_available:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": f"{desc} (unresolved in this environment)",
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
                    "description": desc,
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

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 coordinate to Julich regions using a statistical map.

        Returns an empty dataframe if siibra or a statistical map is unavailable.
        """
        if siibra is None or self.parcellation is None:
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    try:
                        self._pmap = siibra.get_map(
                            parcellation=self.parcellation_spec,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
                    except Exception:
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
            except Exception:
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str, maptype: str = "labelled") -> Any:
        """
        Return a regional mask/volume object for a resolved region node, or None.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            if hasattr(region, "get_regional_mask"):
                return region.get_regional_mask(self.assignment_space, maptype=maptype)
        except Exception:
            pass
        try:
            if hasattr(region, "fetch_regional_map"):
                return region.fetch_regional_map(self.assignment_space, maptype=maptype)
        except Exception:
            pass
        return None

    def simulate(
        self,
        genetic_vulnerability: float = 0.45,
        behavioral_inhibition_trait: float = 0.65,
        early_life_adversity: float = 0.35,
        current_social_stress: float = 0.55,
        social_novelty_load: float = 0.70,
        serotonergic_treatment_support: float = 0.15,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Values are clipped to [0, 1]. Higher regional-state values mean greater
        dysregulation / burden in that circuit node.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "behavioral_inhibition_trait": self._clip01(behavioral_inhibition_trait),
                "early_life_adversity": self._clip01(early_life_adversity),
                "current_social_stress": self._clip01(current_social_stress),
                "social_novelty_load": self._clip01(social_novelty_load),
                "serotonergic_treatment_support": self._clip01(serotonergic_treatment_support),
                "recovery_support": self._clip01(recovery_support),
            },
            name="value",
        )

        # Inputs -> latent biology
        serotonergic_dysregulation = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["behavioral_inhibition_trait"]
            + 0.20 * inputs["current_social_stress"]
            + 0.10 * inputs["early_life_adversity"]
            + 0.10 * inputs["social_novelty_load"]
            - 0.30 * inputs["serotonergic_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        noradrenergic_hyperarousal = self._clip01(
            0.30 * inputs["current_social_stress"]
            + 0.20 * inputs["social_novelty_load"]
            + 0.15 * inputs["behavioral_inhibition_trait"]
            + 0.15 * inputs["early_life_adversity"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.10 * inputs["recovery_support"]
        )

        gabaergic_disinhibition = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["current_social_stress"]
            + 0.20 * inputs["early_life_adversity"]
            + 0.15 * inputs["behavioral_inhibition_trait"]
            + 0.10 * inputs["social_novelty_load"]
            - 0.10 * inputs["recovery_support"]
        )

        epigenetic_stress_sensitization = self._clip01(
            0.45 * inputs["early_life_adversity"]
            + 0.20 * inputs["genetic_vulnerability"]
            + 0.10 * inputs["behavioral_inhibition_trait"]
            + 0.10 * inputs["current_social_stress"]
            - 0.10 * inputs["recovery_support"]
        )

        hpa_axis_hyperreactivity = self._clip01(
            0.30 * epigenetic_stress_sensitization
            + 0.25 * inputs["current_social_stress"]
            + 0.15 * inputs["behavioral_inhibition_trait"]
            + 0.10 * inputs["social_novelty_load"]
            + 0.10 * inputs["early_life_adversity"]
            - 0.10 * inputs["recovery_support"]
        )

        social_novelty_threat_bias = self._clip01(
            0.35 * inputs["behavioral_inhibition_trait"]
            + 0.25 * inputs["social_novelty_load"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.10 * epigenetic_stress_sensitization
            + 0.05 * inputs["current_social_stress"]
            - 0.10 * inputs["recovery_support"]
        )

        autonomic_self_monitoring_amplification = self._clip01(
            0.35 * noradrenergic_hyperarousal
            + 0.20 * social_novelty_threat_bias
            + 0.15 * inputs["current_social_stress"]
            + 0.10 * inputs["behavioral_inhibition_trait"]
            + 0.05 * hpa_axis_hyperreactivity
            - 0.10 * inputs["recovery_support"]
        )

        frontolimbic_regulation_failure = self._clip01(
            0.25 * hpa_axis_hyperreactivity
            + 0.20 * serotonergic_dysregulation
            + 0.15 * gabaergic_disinhibition
            + 0.15 * social_novelty_threat_bias
            + 0.10 * autonomic_self_monitoring_amplification
            + 0.05 * noradrenergic_hyperarousal
            - 0.15 * inputs["recovery_support"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )

        latents = pd.Series(
            {
                "serotonergic_dysregulation": serotonergic_dysregulation,
                "noradrenergic_hyperarousal": noradrenergic_hyperarousal,
                "gabaergic_disinhibition": gabaergic_disinhibition,
                "epigenetic_stress_sensitization": epigenetic_stress_sensitization,
                "hpa_axis_hyperreactivity": hpa_axis_hyperreactivity,
                "social_novelty_threat_bias": social_novelty_threat_bias,
                "autonomic_self_monitoring_amplification": autonomic_self_monitoring_amplification,
                "frontolimbic_regulation_failure": frontolimbic_regulation_failure,
            },
            name="value",
        )

        # Latent biology -> regional-state burden
        amygdala = self._clip01(
            0.35 * social_novelty_threat_bias
            + 0.25 * frontolimbic_regulation_failure
            + 0.15 * gabaergic_disinhibition
            + 0.10 * noradrenergic_hyperarousal
            + 0.10 * serotonergic_dysregulation
            - 0.10 * inputs["recovery_support"]
        )

        hippocampus = self._clip01(
            0.30 * hpa_axis_hyperreactivity
            + 0.25 * epigenetic_stress_sensitization
            + 0.15 * social_novelty_threat_bias
            + 0.10 * serotonergic_dysregulation
            + 0.05 * inputs["current_social_stress"]
            - 0.10 * inputs["recovery_support"]
        )

        pfc_control = self._clip01(
            0.30 * frontolimbic_regulation_failure
            + 0.20 * hpa_axis_hyperreactivity
            + 0.15 * serotonergic_dysregulation
            + 0.10 * autonomic_self_monitoring_amplification
            + 0.10 * social_novelty_threat_bias
            - 0.10 * inputs["recovery_support"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )

        locus_coeruleus_proxy = self._clip01(
            0.45 * noradrenergic_hyperarousal
            + 0.20 * autonomic_self_monitoring_amplification
            + 0.10 * hpa_axis_hyperreactivity
            + 0.10 * inputs["current_social_stress"]
            + 0.05 * inputs["social_novelty_load"]
            - 0.10 * inputs["recovery_support"]
        )

        regional_state = pd.Series(
            {
                "amygdala": amygdala,
                "hippocampus": hippocampus,
                "pfc_control": pfc_control,
                "locus_coeruleus_proxy": locus_coeruleus_proxy,
            },
            name="value",
        )

        # Regional-state burden -> symptoms
        anticipatory_social_anxiety = self._clip01(
            0.30 * regional_state["amygdala"]
            + 0.20 * social_novelty_threat_bias
            + 0.20 * hpa_axis_hyperreactivity
            + 0.10 * regional_state["hippocampus"]
            + 0.10 * autonomic_self_monitoring_amplification
            + 0.10 * regional_state["pfc_control"]
        )

        fear_of_negative_evaluation = self._clip01(
            0.25 * regional_state["amygdala"]
            + 0.25 * autonomic_self_monitoring_amplification
            + 0.15 * regional_state["pfc_control"]
            + 0.15 * serotonergic_dysregulation
            + 0.10 * social_novelty_threat_bias
            + 0.05 * regional_state["hippocampus"]
        )

        physiological_hyperarousal = self._clip01(
            0.40 * regional_state["locus_coeruleus_proxy"]
            + 0.25 * noradrenergic_hyperarousal
            + 0.15 * hpa_axis_hyperreactivity
            + 0.10 * regional_state["amygdala"]
            + 0.05 * autonomic_self_monitoring_amplification
        )

        visible_anxiety_concern = self._clip01(
            0.35 * physiological_hyperarousal
            + 0.30 * fear_of_negative_evaluation
            + 0.15 * autonomic_self_monitoring_amplification
            + 0.10 * regional_state["pfc_control"]
            + 0.05 * regional_state["locus_coeruleus_proxy"]
        )

        social_threat_hypervigilance = self._clip01(
            0.35 * regional_state["amygdala"]
            + 0.25 * social_novelty_threat_bias
            + 0.15 * regional_state["locus_coeruleus_proxy"]
            + 0.10 * regional_state["hippocampus"]
            + 0.10 * serotonergic_dysregulation
        )

        social_avoidance = self._clip01(
            0.30 * anticipatory_social_anxiety
            + 0.25 * fear_of_negative_evaluation
            + 0.15 * visible_anxiety_concern
            + 0.15 * social_threat_hypervigilance
            + 0.10 * inputs["behavioral_inhibition_trait"]
            + 0.05 * regional_state["pfc_control"]
            - 0.05 * inputs["recovery_support"]
        )

        symptoms = pd.Series(
            {
                "anticipatory_social_anxiety": anticipatory_social_anxiety,
                "fear_of_negative_evaluation": fear_of_negative_evaluation,
                "physiological_hyperarousal": physiological_hyperarousal,
                "visible_anxiety_concern": visible_anxiety_concern,
                "social_threat_hypervigilance": social_threat_hypervigilance,
                "social_avoidance": social_avoidance,
            },
            name="value",
        )

        # Symptoms -> phenotype summaries
        phenotypes = pd.Series(
            {
                "social_threat_hyperreactivity_profile": self._clip01(
                    0.35 * symptoms["anticipatory_social_anxiety"]
                    + 0.35 * symptoms["fear_of_negative_evaluation"]
                    + 0.30 * symptoms["social_threat_hypervigilance"]
                ),
                "inhibited_avoidant_profile": self._clip01(
                    0.40 * symptoms["social_avoidance"]
                    + 0.20 * inputs["behavioral_inhibition_trait"]
                    + 0.20 * symptoms["anticipatory_social_anxiety"]
                    + 0.20 * social_novelty_threat_bias
                ),
                "visible_arousal_profile": self._clip01(
                    0.40 * symptoms["physiological_hyperarousal"]
                    + 0.35 * symptoms["visible_anxiety_concern"]
                    + 0.25 * regional_state["locus_coeruleus_proxy"]
                ),
                "global_sad_burden": self._clip01(float(symptoms.mean())),
            },
            name="value",
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = SocialAnxietyDisorderModel()
    built = model.build()

    print("\n=== Social Anxiety Disorder nodes ===")
    print(built["nodes"].to_string(index=False))

    print("\n=== Social Anxiety Disorder edges ===")
    print(built["edges"].to_string(index=False))

    if built["receptors"].get("amygdala") is not None and not built["receptors"]["amygdala"].empty:
        print("\n=== Example receptor table: amygdala ===")
        print(built["receptors"]["amygdala"].head().to_string(index=False))
    else:
        print("\nNo amygdala receptor table available in this environment.")

    if built["genes"].get("pfc_control") is not None and not built["genes"]["pfc_control"].empty:
        print("\n=== Example gene table: pfc_control ===")
        print(built["genes"]["pfc_control"].head().to_string(index=False))
    else:
        print("\nNo pfc_control gene table available in this environment.")

    if built["connectivity_profiles"].get("amygdala") is not None and not built["connectivity_profiles"]["amygdala"].empty:
        print("\n=== Example connectivity profile: amygdala ===")
        print(built["connectivity_profiles"]["amygdala"].head(10).to_string(index=False))
    else:
        print("\nNo amygdala connectivity profile available in this environment.")

    circuit = built.get("circuit_connectivity", pd.DataFrame())
    if isinstance(circuit, pd.DataFrame) and not circuit.empty:
        print("\n=== Within-model circuit connectivity ===")
        print(circuit.to_string())
    else:
        print("\nNo within-model circuit connectivity matrix available in this environment.")

    sim = model.simulate(
        genetic_vulnerability=0.50,
        behavioral_inhibition_trait=0.75,
        early_life_adversity=0.40,
        current_social_stress=0.60,
        social_novelty_load=0.80,
        serotonergic_treatment_support=0.20,
        recovery_support=0.25,
    )

    print("\n=== Example simulation: inputs ===")
    print(sim["inputs"].to_string())

    print("\n=== Example simulation: latent biology ===")
    print(sim["latents"].to_string())

    print("\n=== Example simulation: regional state ===")
    print(sim["regional_state"].to_string())

    print("\n=== Example simulation: symptoms ===")
    print(sim["symptoms"].to_string())

    print("\n=== Example simulation: phenotypes ===")
    print(sim["phenotypes"].to_string())

    # Optional coordinate assignment example in a siibra-enabled environment:
    # print(model.assign_mni_point((-6, -52, 18)).head())
    # Optional regional mask example:
    # mask = model.region_mask("amygdala")
