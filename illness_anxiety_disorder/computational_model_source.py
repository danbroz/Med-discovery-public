from __future__ import annotations

"""
Illness Anxiety Disorder siibra scaffold.

This script turns a chapter-level biological summary of Illness Anxiety Disorder (IAD)
into a transparent, atlas-grounded mechanistic scaffold using siibra. It is intended
for research prototyping, feature exploration, and educational modeling.

It is not a diagnostic, prognostic, or treatment tool.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_IAD_GENE_PANEL = [
    # Serotonergic signaling / transport
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    # GABAergic inhibition
    "GABRA2",
    "GABRB2",
    "SLC6A1",
    # Catecholamine / salience regulation
    "SLC6A2",
    "COMT",
    "DRD2",
    # Stress plasticity / resilience
    "BDNF",
    "FKBP5",
    "CRHR1",
]


class IllnessAnxietyDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Illness Anxiety Disorder.

    Conceptual flow
    ---------------
    inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

    Modeling choices
    ----------------
    - Keeps serotonin, GABA, and catecholamine systems primarily as latent biology.
    - Atlas-anchors regions named or strongly implied by the chapter:
      insula, amygdala, OFC, ACC, and a conservative PFC-control proxy.
    - Represents fronto-striatal involvement with a conservative striatal proxy,
      because the chapter stays systems-level rather than naming a specific nucleus.
    - Uses simple 0..1 normalization for interpretability.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        assignment_space_spec: str = "mni152",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:  # pragma: no cover - depends on runtime environment
            raise ImportError(
                "siibra is required to use IllnessAnxietyDisorderModel. "
                "Install siibra in your environment before running this scaffold."
            ) from _SIIBRA_IMPORT_ERROR

        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space_spec = assignment_space_spec
        self.connectivity_cohort = connectivity_cohort

        # Compatibility-first initialization.
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
        try:
            self.assignment_space = (
                self.atlas.get_space(assignment_space_spec)
                if hasattr(self.atlas, "get_space")
                else self.atlas.spaces.get(assignment_space_spec)
            )
        except Exception:
            self.assignment_space = assignment_space_spec

        self.region_candidates: Dict[str, List[str]] = {
            "insula": [
                "Area Ig2 (Insula) left",
                "Area Ig1 (Insula) left",
                "Area Id7 (Insula) left",
                "insula left",
                "insula",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo2 (OFC) left",
                "orbitofrontal cortex left",
                "orbitofrontal",
                "ofc",
            ],
            "pfc_control": [
                "Area Fp1 left",
                "Area Fp2 left",
                "medial prefrontal cortex left",
                "medial frontal cortex left",
                "prefrontal cortex",
            ],
            "acc": [
                "Area p24pr left",
                "Area p24ab left",
                "Area a24pr left",
                "anterior cingulate cortex left",
                "anterior cingulate",
                "acc",
            ],
            "striatum_proxy": [
                "caudate nucleus left",
                "putamen left",
                "striatum left",
                "caudate",
                "putamen",
                "striatum",
            ],
        }

        self.region_node_notes: Dict[str, str] = {
            "insula": "Interoceptive awareness and bodily salience node.",
            "amygdala": "Threat valuation and fear amplification node.",
            "ofc": "Orbitofrontal appraisal/reappraisal component of PFC control.",
            "pfc_control": "Conservative medial-prefrontal control proxy for top-down reappraisal.",
            "acc": "Conflict monitoring, rumination, and compulsive checking-related cingulate node.",
            "striatum_proxy": "Conservative fronto-striatal proxy for repetitive checking / habit pressure.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Broad heritable liability for anxiety, mood symptoms, and interoceptive/cognitive bias."
            ),
            "psychosocial_stress": (
                "Current psychosocial stressors that heighten bodily vigilance and distress."
            ),
            "bodily_sensation_load": (
                "Benign or ambiguous bodily sensations available for amplification and threat interpretation."
            ),
            "negative_interpretation_bias": (
                "Trait-like tendency to interpret ambiguous bodily information catastrophically."
            ),
            "medical_uncertainty": (
                "Ambiguous medical information or unresolved health uncertainty maintaining worry."
            ),
            "ssri_support": (
                "Serotonergic treatment support that can reduce obsessive health worry and improve control."
            ),
            "benzodiazepine_relief": (
                "Short-term GABAergic anxiolysis; modeled as temporary relief rather than durable correction."
            ),
            "recovery_support": (
                "Psychotherapy, coping structure, and reassurance-limiting support that improve reappraisal."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "somatization_bias": (
                "Psychological distress becoming expressed and tracked as bodily symptom burden."
            ),
            "interoceptive_amplification": (
                "Amplification and over-monitoring of normal bodily sensations."
            ),
            "serotonergic_dysregulation": (
                "Serotonergic imbalance linked to anxiety, obsessive worry, and mood burden."
            ),
            "gabaergic_disinhibition": (
                "Insufficient inhibitory braking on anxiety-related firing."
            ),
            "salience_arousal_bias": (
                "Catecholaminergic threat-salience and arousal bias encompassing norepinephrine/dopamine contributions."
            ),
            "catastrophic_health_appraisal": (
                "Appraising bodily sensations as threatening evidence of illness."
            ),
            "threat_circuit_hyperreactivity": (
                "Escalating fear/worry circuit reactivity to bodily cues."
            ),
            "prefrontal_reappraisal_failure": (
                "Weak top-down reappraisal and inhibition of fear responses."
            ),
            "frontostriatal_rumination_checking": (
                "Repetitive health-focused monitoring, checking, and compulsive reassurance loops."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "health_anxiety": "Persistent fear of serious illness despite limited objective evidence.",
            "somatic_preoccupation": "Preoccupation with bodily symptoms or sensations.",
            "persistent_worry_tension": "Chronic tension, worry, and anxious arousal.",
            "attentional_bias_to_body": "Preferential attention to bodily cues and internal sensations.",
            "reassurance_seeking_checking": "Checking and reassurance seeking driven by health threat beliefs.",
            "delusional_conviction_risk": "Rare severe end-state with poorly correctable illness conviction.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "loads broad anxiety/obsessional susceptibility",
                "iad_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "threat_circuit_hyperreactivity",
                "relation": "raises inherited threat sensitivity",
                "iad_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "somatization_bias",
                "relation": "channels distress into bodily symptom tracking",
                "iad_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "salience_arousal_bias",
                "relation": "amplifies bodily arousal and vigilance",
                "iad_change": "increased",
            },
            {
                "source": "bodily_sensation_load",
                "target": "interoceptive_amplification",
                "relation": "provides ambiguous bodily signals to over-monitor",
                "iad_change": "increased",
            },
            {
                "source": "negative_interpretation_bias",
                "target": "catastrophic_health_appraisal",
                "relation": "pushes ambiguous sensations toward threat meanings",
                "iad_change": "increased",
            },
            {
                "source": "medical_uncertainty",
                "target": "catastrophic_health_appraisal",
                "relation": "maintains illness-related ambiguity and fear",
                "iad_change": "increased",
            },
            {
                "source": "somatization_bias",
                "target": "interoceptive_amplification",
                "relation": "feeds the monitoring of bodily sensations",
                "iad_change": "increased",
            },
            {
                "source": "interoceptive_amplification",
                "target": "insula",
                "relation": "heightens interoceptive signal salience",
                "iad_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "threat_circuit_hyperreactivity",
                "relation": "reduces inhibitory control over anxiety circuitry",
                "iad_change": "increased",
            },
            {
                "source": "salience_arousal_bias",
                "target": "threat_circuit_hyperreactivity",
                "relation": "adds catecholaminergic threat urgency",
                "iad_change": "increased",
            },
            {
                "source": "catastrophic_health_appraisal",
                "target": "threat_circuit_hyperreactivity",
                "relation": "turns bodily uncertainty into escalating fear",
                "iad_change": "increased",
            },
            {
                "source": "threat_circuit_hyperreactivity",
                "target": "amygdala",
                "relation": "drives a fear-biased alarm response",
                "iad_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "prefrontal_reappraisal_failure",
                "relation": "weakens mood/anxiety regulation and cognitive control",
                "iad_change": "increased",
            },
            {
                "source": "prefrontal_reappraisal_failure",
                "target": "pfc_control",
                "relation": "reduces top-down threat reappraisal capacity",
                "iad_change": "decreased",
            },
            {
                "source": "prefrontal_reappraisal_failure",
                "target": "ofc",
                "relation": "weakens contextual appraisal and inhibitory evaluation",
                "iad_change": "decreased",
            },
            {
                "source": "frontostriatal_rumination_checking",
                "target": "acc",
                "relation": "loads conflict monitoring and perseverative attention",
                "iad_change": "increased",
            },
            {
                "source": "frontostriatal_rumination_checking",
                "target": "striatum_proxy",
                "relation": "promotes repetitive checking and habit-like reassurance loops",
                "iad_change": "increased",
            },
            {
                "source": "catastrophic_health_appraisal",
                "target": "frontostriatal_rumination_checking",
                "relation": "drives perseverative health monitoring",
                "iad_change": "increased",
            },
            {
                "source": "insula",
                "target": "attentional_bias_to_body",
                "relation": "focuses attention on internal bodily cues",
                "iad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "health_anxiety",
                "relation": "casts bodily signals as threatening",
                "iad_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "health_anxiety",
                "relation": "normally reappraises and suppresses excessive threat belief",
                "iad_change": "decreased",
            },
            {
                "source": "ofc",
                "target": "health_anxiety",
                "relation": "normally contextualizes illness-related beliefs",
                "iad_change": "decreased",
            },
            {
                "source": "acc",
                "target": "reassurance_seeking_checking",
                "relation": "supports repetitive monitoring and checking pressure",
                "iad_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "reassurance_seeking_checking",
                "relation": "adds habit-like checking and reassurance loops",
                "iad_change": "increased",
            },
            {
                "source": "health_anxiety",
                "target": "somatic_preoccupation",
                "relation": "keeps attention anchored to bodily symptoms",
                "iad_change": "increased",
            },
            {
                "source": "threat_circuit_hyperreactivity",
                "target": "persistent_worry_tension",
                "relation": "sustains chronic anxious arousal",
                "iad_change": "increased",
            },
            {
                "source": "catastrophic_health_appraisal",
                "target": "delusional_conviction_risk",
                "relation": "can escalate toward fixed illness conviction when severe",
                "iad_change": "increased",
            },
            {
                "source": "ssri_support",
                "target": "serotonergic_dysregulation",
                "relation": "restores serotonergic balance",
                "iad_change": "decreased",
            },
            {
                "source": "ssri_support",
                "target": "prefrontal_reappraisal_failure",
                "relation": "supports improved anxiety and obsessive-worry regulation",
                "iad_change": "decreased",
            },
            {
                "source": "benzodiazepine_relief",
                "target": "gabaergic_disinhibition",
                "relation": "temporarily boosts inhibitory tone",
                "iad_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "catastrophic_health_appraisal",
                "relation": "improves reappraisal and limits escalation of illness meanings",
                "iad_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "frontostriatal_rumination_checking",
                "relation": "reduces reassurance-seeking and checking loops",
                "iad_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap: Optional[Any] = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
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
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "insula"} else 0
        proxy_penalty = 1 if any(token in name for token in ["cortex", "brain", "lobe"]) and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
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
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            record = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if record in seen:
                continue
            seen.add(record)
            rows.append(
                {
                    "name": record[0],
                    "identifier": record[1],
                    "parcellation": record[2],
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

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = getattr(main, "volume", None)
        return centroid_xyz, (float(volume_mm3) if volume_mm3 is not None else None)

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy().reset_index()
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
            lower_cols = {str(c).lower(): c for c in df.columns}
            required = {"gene", "level", "zscore"}
            if required.issubset(lower_cols):
                gene_col = lower_cols["gene"]
                level_col = lower_cols["level"]
                zscore_col = lower_cols["zscore"]
                out = (
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
                )
                return out.reset_index(drop=True)
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (
                f
                for f in feats
                if str(getattr(f, "cohort", "")).lower() == self.connectivity_cohort.lower()
            ),
            feats[0],
        )

        # Some siibra versions expose a DataFrame directly on the compound feature.
        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        # More commonly, each compound element stores a subject-level matrix.
        try:
            first = compound[0]
            first_data = getattr(first, "data", None)
            if isinstance(first_data, pd.DataFrame):
                self._connectivity_matrix = first_data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        region_id = getattr(region, "identifier", None)

        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]

        if region_id is not None:
            id_matches = [x for x in labels if getattr(x, "identifier", None) == region_id]
            if id_matches:
                return id_matches[0]

        rn = region_name.lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        return None

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
            df = pd.DataFrame(
                {
                    "connected_region": [self._name_of(idx) for idx in series.index],
                    "value": pd.to_numeric(series.values, errors="coerce"),
                }
            )
            df = df[df["connected_region"] != region.name]
            df = df.dropna(subset=["value"]).sort_values("value", ascending=False)
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """Return a compact pairwise connectivity view for resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        keys = list(self.region_objects.keys())
        for i, src_key in enumerate(keys):
            src = self.region_objects[src_key]
            src_label = self._match_region_label(list(matrix.index), src)
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src)
            if src_label is None:
                continue

            for dst_key in keys[i + 1 :]:
                dst = self.region_objects[dst_key]
                dst_label = self._match_region_label(list(matrix.columns), dst)
                if dst_label is None:
                    dst_label = self._match_region_label(list(matrix.index), dst)
                if dst_label is None:
                    continue

                value = None
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix.loc[dst_label, src_label]
                    except Exception:
                        value = None
                if value is None:
                    continue
                try:
                    value = float(value)
                except Exception:
                    continue

                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": self._name_of(src),
                        "target_key": dst_key,
                        "target_region": self._name_of(dst),
                        "value": value,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_IAD_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
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
            note = self.region_node_notes.get(key, "Atlas-backed circuit node or conservative proxy.")
            region = self._resolve_region(candidates)
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
                        "description": f"{note} Unresolved in this runtime.",
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
                    "label": region.name,
                    "node_type": "region",
                    "description": note,
                    "atlas_region": region.name,
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

    def region_mask(self, node_key: str) -> Any:
        """Return a regional mask if the node resolved to an atlas region."""
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        space_candidates = [self.assignment_space, self.assignment_space_spec, self.space, self.space_spec]
        for space in space_candidates:
            try:
                return region.get_regional_mask(space)
            except Exception:
                continue
        return None

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI coordinate to Julich regions.

        Uses a statistical/probabilistic parcellation map when available.
        """
        if len(xyz) != 3:
            raise ValueError("xyz must contain exactly three coordinates.")

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space_spec,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        space=self.assignment_space,
                        parcellation=self.parcellation,
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space_spec)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        if isinstance(assignments, pd.DataFrame):
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
            return assignments.reset_index(drop=True)

        try:
            df = pd.DataFrame(assignments)
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in df.columns:
                    return df.sort_values(candidate, ascending=False).reset_index(drop=True)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def simulate(
        self,
        genetic_vulnerability: float = 0.50,
        psychosocial_stress: float = 0.55,
        bodily_sensation_load: float = 0.45,
        negative_interpretation_bias: float = 0.55,
        medical_uncertainty: float = 0.50,
        ssri_support: float = 0.00,
        benzodiazepine_relief: float = 0.00,
        recovery_support: float = 0.00,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent one-pass simulation on normalized 0..1 inputs.

        High values indicate more dysregulation / burden, except for pfc_control and ofc,
        where higher values indicate relatively preserved regulatory function.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "psychosocial_stress": self._clip01(psychosocial_stress),
                "bodily_sensation_load": self._clip01(bodily_sensation_load),
                "negative_interpretation_bias": self._clip01(negative_interpretation_bias),
                "medical_uncertainty": self._clip01(medical_uncertainty),
                "ssri_support": self._clip01(ssri_support),
                "benzodiazepine_relief": self._clip01(benzodiazepine_relief),
                "recovery_support": self._clip01(recovery_support),
            },
            name="value",
        )

        somatization_bias = self._clip01(
            0.40 * inputs["psychosocial_stress"]
            + 0.25 * inputs["negative_interpretation_bias"]
            + 0.20 * inputs["medical_uncertainty"]
            + 0.15 * inputs["genetic_vulnerability"]
            - 0.15 * inputs["recovery_support"]
        )

        interoceptive_amplification = self._clip01(
            0.40 * inputs["bodily_sensation_load"]
            + 0.35 * somatization_bias
            + 0.15 * inputs["psychosocial_stress"]
            + 0.15 * inputs["negative_interpretation_bias"]
            - 0.10 * inputs["recovery_support"]
        )

        serotonergic_dysregulation = self._clip01(
            0.40 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["psychosocial_stress"]
            + 0.15 * inputs["negative_interpretation_bias"]
            + 0.10 * inputs["medical_uncertainty"]
            - 0.35 * inputs["ssri_support"]
        )

        gabaergic_disinhibition = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.35 * inputs["psychosocial_stress"]
            + 0.15 * inputs["bodily_sensation_load"]
            - 0.30 * inputs["benzodiazepine_relief"]
            - 0.10 * inputs["recovery_support"]
        )

        salience_arousal_bias = self._clip01(
            0.30 * inputs["psychosocial_stress"]
            + 0.25 * inputs["medical_uncertainty"]
            + 0.20 * inputs["negative_interpretation_bias"]
            + 0.15 * interoceptive_amplification
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.10 * inputs["recovery_support"]
        )

        catastrophic_health_appraisal = self._clip01(
            0.35 * inputs["negative_interpretation_bias"]
            + 0.20 * inputs["medical_uncertainty"]
            + 0.20 * interoceptive_amplification
            + 0.15 * inputs["psychosocial_stress"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.15 * inputs["recovery_support"]
            - 0.10 * inputs["ssri_support"]
        )

        threat_circuit_hyperreactivity = self._clip01(
            0.30 * gabaergic_disinhibition
            + 0.25 * salience_arousal_bias
            + 0.20 * interoceptive_amplification
            + 0.15 * catastrophic_health_appraisal
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.15 * inputs["recovery_support"]
            - 0.10 * inputs["benzodiazepine_relief"]
        )

        prefrontal_reappraisal_failure = self._clip01(
            0.30 * serotonergic_dysregulation
            + 0.25 * inputs["psychosocial_stress"]
            + 0.20 * catastrophic_health_appraisal
            + 0.15 * salience_arousal_bias
            - 0.20 * inputs["recovery_support"]
            - 0.15 * inputs["ssri_support"]
        )

        frontostriatal_rumination_checking = self._clip01(
            0.30 * catastrophic_health_appraisal
            + 0.25 * serotonergic_dysregulation
            + 0.20 * salience_arousal_bias
            + 0.15 * inputs["medical_uncertainty"]
            + 0.10 * interoceptive_amplification
            - 0.15 * inputs["recovery_support"]
            - 0.05 * inputs["ssri_support"]
        )

        latents = pd.Series(
            {
                "somatization_bias": somatization_bias,
                "interoceptive_amplification": interoceptive_amplification,
                "serotonergic_dysregulation": serotonergic_dysregulation,
                "gabaergic_disinhibition": gabaergic_disinhibition,
                "salience_arousal_bias": salience_arousal_bias,
                "catastrophic_health_appraisal": catastrophic_health_appraisal,
                "threat_circuit_hyperreactivity": threat_circuit_hyperreactivity,
                "prefrontal_reappraisal_failure": prefrontal_reappraisal_failure,
                "frontostriatal_rumination_checking": frontostriatal_rumination_checking,
            },
            name="value",
        )

        pfc_control_load = self._clip01(
            0.60 * prefrontal_reappraisal_failure
            + 0.20 * serotonergic_dysregulation
            + 0.20 * inputs["psychosocial_stress"]
        )
        ofc_load = self._clip01(
            0.50 * prefrontal_reappraisal_failure
            + 0.20 * catastrophic_health_appraisal
            + 0.15 * serotonergic_dysregulation
            + 0.15 * inputs["psychosocial_stress"]
        )

        regional_state = pd.Series(
            {
                "insula": self._clip01(
                    0.65 * interoceptive_amplification
                    + 0.20 * somatization_bias
                    + 0.15 * salience_arousal_bias
                ),
                "amygdala": self._clip01(
                    0.60 * threat_circuit_hyperreactivity
                    + 0.20 * catastrophic_health_appraisal
                    + 0.10 * gabaergic_disinhibition
                    + 0.10 * inputs["medical_uncertainty"]
                ),
                "pfc_control": self._clip01(1.0 - pfc_control_load),
                "ofc": self._clip01(1.0 - ofc_load),
                "acc": self._clip01(
                    0.45 * frontostriatal_rumination_checking
                    + 0.25 * catastrophic_health_appraisal
                    + 0.20 * threat_circuit_hyperreactivity
                    + 0.10 * interoceptive_amplification
                ),
                "striatum_proxy": self._clip01(
                    0.55 * frontostriatal_rumination_checking
                    + 0.25 * salience_arousal_bias
                    + 0.20 * inputs["medical_uncertainty"]
                ),
            },
            name="value",
        )

        attentional_bias_to_body = self._clip01(
            0.55 * regional_state["insula"]
            + 0.25 * catastrophic_health_appraisal
            + 0.20 * inputs["medical_uncertainty"]
            - 0.10 * inputs["recovery_support"]
        )

        persistent_worry_tension = self._clip01(
            0.45 * regional_state["amygdala"]
            + 0.20 * gabaergic_disinhibition
            + 0.20 * serotonergic_dysregulation
            + 0.15 * salience_arousal_bias
        )

        health_anxiety = self._clip01(
            0.35 * regional_state["amygdala"]
            + 0.25 * catastrophic_health_appraisal
            + 0.15 * attentional_bias_to_body
            + 0.15 * (1.0 - regional_state["pfc_control"])
            + 0.10 * (1.0 - regional_state["ofc"])
        )

        somatic_preoccupation = self._clip01(
            0.35 * attentional_bias_to_body
            + 0.25 * health_anxiety
            + 0.20 * somatization_bias
            + 0.20 * inputs["medical_uncertainty"]
        )

        reassurance_seeking_checking = self._clip01(
            0.35 * frontostriatal_rumination_checking
            + 0.20 * regional_state["acc"]
            + 0.20 * health_anxiety
            + 0.15 * somatic_preoccupation
            + 0.10 * (1.0 - regional_state["pfc_control"])
        )

        delusional_conviction_risk = self._clip01(
            0.35 * catastrophic_health_appraisal
            + 0.25 * health_anxiety
            + 0.20 * (1.0 - regional_state["pfc_control"])
            + 0.10 * regional_state["amygdala"]
            + 0.10 * inputs["medical_uncertainty"]
        )

        symptoms = pd.Series(
            {
                "health_anxiety": health_anxiety,
                "somatic_preoccupation": somatic_preoccupation,
                "persistent_worry_tension": persistent_worry_tension,
                "attentional_bias_to_body": attentional_bias_to_body,
                "reassurance_seeking_checking": reassurance_seeking_checking,
                "delusional_conviction_risk": delusional_conviction_risk,
            },
            name="value",
        )

        phenotypes = pd.Series(
            {
                "somatic_hypervigilance_profile": self._clip01(
                    (attentional_bias_to_body + somatic_preoccupation + regional_state["insula"]) / 3.0
                ),
                "obsessive_health_worry_profile": self._clip01(
                    (health_anxiety + persistent_worry_tension + reassurance_seeking_checking) / 3.0
                ),
                "checking_loop_profile": self._clip01(
                    (
                        reassurance_seeking_checking
                        + regional_state["acc"]
                        + regional_state["striatum_proxy"]
                    )
                    / 3.0
                ),
                "conviction_escalation_profile": self._clip01(
                    (
                        delusional_conviction_risk
                        + catastrophic_health_appraisal
                        + (1.0 - regional_state["pfc_control"])
                    )
                    / 3.0
                ),
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


if __name__ == "__main__":  # pragma: no cover - example usage
    try:
        model = IllnessAnxietyDisorderModel()
    except ImportError as exc:
        print(exc)
        raise SystemExit(1)

    print("Building Illness Anxiety Disorder scaffold...\n")
    bundle = model.build(connectivity_rows=10)

    print("NODES")
    print(bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\nEDGES")
    print(bundle["edges"][["source", "target", "relation", "iad_change"]].to_string(index=False))

    if "amygdala" in bundle["receptors"] and not bundle["receptors"]["amygdala"].empty:
        print("\nAMYGDALA RECEPTOR FINGERPRINT (head)")
        print(bundle["receptors"]["amygdala"].head().to_string(index=False))

    if "insula" in bundle["genes"] and not bundle["genes"]["insula"].empty:
        print("\nINSULA GENE SUMMARY (head)")
        print(bundle["genes"]["insula"].head().to_string(index=False))

    if not bundle["circuit_connectivity"].empty:
        print("\nCIRCUIT CONNECTIVITY")
        print(bundle["circuit_connectivity"].head(20).to_string(index=False))

    print("\nSIMULATION EXAMPLE")
    sim = model.simulate(
        genetic_vulnerability=0.55,
        psychosocial_stress=0.75,
        bodily_sensation_load=0.65,
        negative_interpretation_bias=0.80,
        medical_uncertainty=0.70,
        ssri_support=0.25,
        benzodiazepine_relief=0.10,
        recovery_support=0.30,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-32, 20, 4)).head())
