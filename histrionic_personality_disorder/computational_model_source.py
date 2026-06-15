
from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Histrionic Personality Disorder (HPD).

This script converts a biologically oriented HPD chapter into a transparent,
research-facing mechanistic model. It is intended as a scaffold for exploration
and refinement, not as a validated disease model, diagnostic system, or
treatment recommendation engine.

Design choices:
- Atlas anchors are conservative and follow the chapter: amygdala, OFC, and
  vmPFC are central. ACC and insula are included as cautious proxies because
  they are mentioned in related structural hypotheses rather than as direct,
  settled HPD loci.
- Dopamine and serotonin are modeled primarily as latent biological processes,
  not forced into parcels.
- The simulator is explicit, one-pass, and normalized to 0..1 so each step can
  be inspected line-by-line.
- Atlas-backed feature retrieval degrades gracefully when siibra, receptors,
  genes, or connectivity data are unavailable.

Chapter-derived logic represented here:
inputs -> latent biology -> regional burden/control failure -> symptoms -> phenotype summaries
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - environment dependent
    import siibra  # type: ignore
    _SIIBRA_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc


DEFAULT_GENE_PANEL = [
    "DRD2",
    "DRD4",
    "SLC6A3",   # DAT1
    "COMT",
    "MAOA",
    "SLC6A4",
    "TPH2",
    "HTR1A",
    "HTR2A",
    "BDNF",
    "FKBP5",
    "OXTR",
]


class HistrionicPersonalityDisorderModel:
    """
    Research scaffold for chapter-driven HPD biology.

    The build() method resolves conservative atlas regions (or labeled proxies),
    retrieves receptor / gene / connectivity summaries where available, and
    returns node and edge tables.

    The simulate() method implements an interpretable one-pass mechanistic model:
    genetic + developmental + social reinforcement inputs feed dopaminergic
    attention reward drive, serotonergic disinhibition, and frontolimbic control
    failure, which then shape regional dysregulation burden and downstream HPD
    symptom expressions.

    Higher "regional_state" values indicate higher inferred dysregulation burden,
    not healthy activation.
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

        self.siibra_available = siibra is not None
        self.atlas = None
        self.parcellation = None
        self.space = None
        self._quiet = nullcontext()

        if self.siibra_available:
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
                self._quiet = getattr(siibra, "QUIET", nullcontext())
            except Exception as exc:
                warnings.warn(
                    f"siibra atlas initialization failed ({exc!r}). "
                    "Atlas-backed retrieval will be unavailable, but simulate() remains usable."
                )
                self.siibra_available = False
        else:
            warnings.warn(
                "siibra is not installed in this environment. "
                "build() will return empty atlas-backed feature tables, but simulate() still works."
            )

        # Region choices are conservative and left-lateralized where a single
        # representative region is needed. Proxies are explicit where atlas
        # labels vary or the chapter is less anatomically specific.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo4 left",
                "Area Fo3 left",
                "orbitofrontal cortex",
                "orbitofrontal",
                "OFC",
            ],
            "vmpfc_proxy": [
                "Area 14m left",
                "Area 14m",
                "Area 14 left",
                "medial orbitofrontal cortex",
                "ventromedial prefrontal cortex",
                "vmPFC",
            ],
            "acc_proxy": [
                "Area p24pr left",
                "Area p24 left",
                "Area a24pr left",
                "anterior cingulate cortex",
                "ACC",
            ],
            "insula_proxy": [
                "Area Id1 left",
                "Area Id2 left",
                "Area Ig2 left",
                "insula left",
                "insula",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Polygenic or familial liability affecting reward sensitivity, impulsivity, "
                "and emotional reactivity."
            ),
            "innate_emotional_reactivity": (
                "Temperamental baseline of high emotional intensity or fast arousal."
            ),
            "early_life_stress": (
                "Developmental and environmental adversity shaping later interpersonal and "
                "regulatory style."
            ),
            "social_attention_reinforcement": (
                "Repeated reward from being noticed, admired, or made central in social settings."
            ),
            "interpersonal_stress": (
                "Current relational strain that escalates affective volatility and impulsive behavior."
            ),
            "ssri_support": (
                "Protective serotonergic support proxy for off-label SSRI treatment response."
            ),
            "recovery_support": (
                "Psychotherapy, structure, feedback, and supportive relationships that stabilize behavior."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "dopaminergic_attention_reward_drive": (
                "Reward / novelty / social-attention salience pressure that reinforces dramatic "
                "or attention-seeking behavior."
            ),
            "serotonergic_disinhibition": (
                "Reduced serotonergic constraint on affective lability, aggression, and impulse control."
            ),
            "frontolimbic_disconnection": (
                "Weak structural or functional coupling between limbic reactivity and prefrontal control, "
                "consistent with amygdala-PFC disconnection hypotheses."
            ),
            "frontolimbic_dysregulation": (
                "Failure of top-down regulation over emotionally charged limbic responses."
            ),
            "prefrontal_social_decision_impairment": (
                "Reduced consequence evaluation, social norm calibration, and perspective taking."
            ),
            "attention_deprivation_empty_state": (
                "Boredom / emptiness state hypothesized to emerge when social-attention reward falls off."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "attention_seeking": "Persistent drive to become the focus of others' attention.",
            "novelty_sensation_seeking": "Pursuit of excitement, novelty, sensuality, or dramatic stimulation.",
            "short_lived_intense_engagement": (
                "Brief but intense enthusiasm in relationships or projects."
            ),
            "affective_lability": "Rapidly shifting emotional expression and mood volatility.",
            "impulsive_dramatic_outbursts": (
                "Sudden dramatic reactions, provocative actions, or rash behavioral displays."
            ),
            "anger_dyscontrol": "Frequent temper displays or weak control over anger.",
            "boredom_emptiness": "Chronic emptiness or boredom when not socially validated.",
            "provocative_boundary_blurring": (
                "Seductive, overly familiar, or socially inappropriate interpersonal behavior."
            ),
            "misperceived_intimacy": (
                "Overestimating closeness or misreading the true nature of relationships."
            ),
            "self_destructive_gestures": (
                "Occasional self-damaging gestures emerging from dysregulated affect and emptiness."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_attention_reward_drive",
                "relation": "shapes reward sensitivity and novelty-seeking liability",
                "hpd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_disinhibition",
                "relation": "raises baseline risk for affective and impulse-control instability",
                "hpd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "frontolimbic_disconnection",
                "relation": "contributes to heritable regulatory vulnerability",
                "hpd_change": "increased",
            },
            {
                "source": "innate_emotional_reactivity",
                "target": "frontolimbic_dysregulation",
                "relation": "creates a high-arousal substrate for dysregulated expression",
                "hpd_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "frontolimbic_disconnection",
                "relation": "interacts with biology to weaken later regulatory integration",
                "hpd_change": "increased",
            },
            {
                "source": "social_attention_reinforcement",
                "target": "dopaminergic_attention_reward_drive",
                "relation": "social centrality is reinforcing and sustains attention seeking",
                "hpd_change": "increased",
            },
            {
                "source": "interpersonal_stress",
                "target": "serotonergic_disinhibition",
                "relation": "stress worsens irritability and impulsive reactivity",
                "hpd_change": "increased",
            },
            {
                "source": "interpersonal_stress",
                "target": "frontolimbic_dysregulation",
                "relation": "relationship strain amplifies emotionally driven responding",
                "hpd_change": "increased",
            },
            {
                "source": "ssri_support",
                "target": "serotonergic_disinhibition",
                "relation": "serotonergic support may reduce affective instability and impulsivity",
                "hpd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "frontolimbic_dysregulation",
                "relation": "supportive containment improves reflective control",
                "hpd_change": "decreased",
            },
            {
                "source": "amygdala",
                "target": "frontolimbic_dysregulation",
                "relation": "limbic hyperreactivity drives unchecked emotional escalation",
                "hpd_change": "increased",
            },
            {
                "source": "ofc",
                "target": "prefrontal_social_decision_impairment",
                "relation": "OFC dysfunction weakens consequence evaluation and impulse control",
                "hpd_change": "increased",
            },
            {
                "source": "vmpfc_proxy",
                "target": "prefrontal_social_decision_impairment",
                "relation": "vmPFC dysfunction reduces social norm calibration and perspective taking",
                "hpd_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "frontolimbic_dysregulation",
                "relation": "weaker amygdala-PFC coupling limits top-down damping of affect",
                "hpd_change": "increased",
            },
            {
                "source": "dopaminergic_attention_reward_drive",
                "target": "attention_seeking",
                "relation": "drives repeated pursuit of centrality and admiration",
                "hpd_change": "increased",
            },
            {
                "source": "dopaminergic_attention_reward_drive",
                "target": "novelty_sensation_seeking",
                "relation": "drives excitement-seeking and intense stimulation seeking",
                "hpd_change": "increased",
            },
            {
                "source": "dopaminergic_attention_reward_drive",
                "target": "short_lived_intense_engagement",
                "relation": "supports rapidly intensified relationships and projects",
                "hpd_change": "increased",
            },
            {
                "source": "attention_deprivation_empty_state",
                "target": "boredom_emptiness",
                "relation": "low-attention reward states are experienced as emptiness or boredom",
                "hpd_change": "increased",
            },
            {
                "source": "serotonergic_disinhibition",
                "target": "affective_lability",
                "relation": "reduced serotonergic modulation destabilizes mood expression",
                "hpd_change": "increased",
            },
            {
                "source": "serotonergic_disinhibition",
                "target": "impulsive_dramatic_outbursts",
                "relation": "weak inhibitory control permits rash dramatic actions",
                "hpd_change": "increased",
            },
            {
                "source": "serotonergic_disinhibition",
                "target": "anger_dyscontrol",
                "relation": "lower serotonergic restraint increases temper dyscontrol",
                "hpd_change": "increased",
            },
            {
                "source": "serotonergic_disinhibition",
                "target": "self_destructive_gestures",
                "relation": "affective and impulse instability can spill into self-damaging gestures",
                "hpd_change": "increased",
            },
            {
                "source": "prefrontal_social_decision_impairment",
                "target": "provocative_boundary_blurring",
                "relation": "impaired norm tracking promotes seductive or overly familiar behavior",
                "hpd_change": "increased",
            },
            {
                "source": "prefrontal_social_decision_impairment",
                "target": "misperceived_intimacy",
                "relation": "reduced perspective taking distorts appraisal of relational closeness",
                "hpd_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        cands: List[Any] = []
        if siibra is not None:
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

    def _safe_features(self, concept: Any, modality: Any, **kwargs: Any) -> List[Any]:
        if not self.siibra_available or concept is None:
            return []
        try:
            with self._quiet:
                feats = siibra.features.get(concept, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        for modality in modalities:
            feats = self._safe_features(concept, modality, **kwargs)
            if feats:
                return feats
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        if not self.siibra_available or self.atlas is None:
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

        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        proxy_penalty = 1 if "proxy" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "insula",
            "anterior cingulate cortex",
            "prefrontal cortex",
            "orbitofrontal cortex",
        } else 0
        return (left_bonus, right_penalty, proxy_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available:
            return None
        for spec in candidates:
            try:
                if self.atlas is not None:
                    return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                if self.parcellation is not None:
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
                data = getattr(feat, "data", None)
                if isinstance(data, pd.DataFrame):
                    df = data.copy().reset_index()
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
                data = getattr(feat, "data", None)
                if not isinstance(data, pd.DataFrame):
                    continue
                df = data.copy()
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
                        .reset_index(drop=True)
                    )
                    return out
                return df.reset_index(drop=True)
            except Exception:
                continue
        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if not self.siibra_available or self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        preferred = None
        for feat in feats:
            cohort = getattr(feat, "cohort", None)
            if cohort == self.connectivity_cohort:
                preferred = feat
                break
        compound = preferred or feats[0]

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            sub = compound[0]
            data = getattr(sub, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
        if exact:
            return exact[0]

        rn = getattr(region, "name", "").lower()
        fuzzy = [
            x for x in labels
            if rn and (
                rn in self._name_of(x).lower()
                or self._name_of(x).lower() in rn
            )
        ]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or region is None:
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
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            region_name = getattr(region, "name", "")
            df = df[df["connected_region"] != region_name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _feature_summary(self, receptor_df: pd.DataFrame, gene_df: pd.DataFrame, conn_df: pd.DataFrame) -> str:
        return (
            f"receptors={'yes' if not receptor_df.empty else 'no'}; "
            f"genes={'yes' if not gene_df.empty else 'no'}; "
            f"connectivity={'yes' if not conn_df.empty else 'no'}"
        )

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a within-circuit connectivity submatrix for the resolved HPD nodes.

        Rows/columns are renamed to scaffold node keys to keep the output stable
        even when underlying region labels differ.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        matched_labels: Dict[str, Any] = {}
        for node_key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            axis = "index"
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
                axis = "columns"
            if label is None or axis != "index":
                continue
            matched_labels[node_key] = label

        if len(matched_labels) < 2:
            return pd.DataFrame()

        labels = list(matched_labels.values())
        try:
            sub = matrix.loc[labels, labels].copy()
            rename_map = {label: key for key, label in matched_labels.items()}
            sub = sub.rename(index=rename_map, columns=rename_map)
            return sub
        except Exception:
            return pd.DataFrame()

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        # Clear any prior state so repeated builds stay reproducible.
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
            region_is_proxy = key.endswith("_proxy")

            if region is None:
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if region_is_proxy else "region",
                        "description": (
                            "Explicit proxy node unresolved in this environment"
                            if region_is_proxy else
                            "Atlas-backed node unresolved in this environment"
                        ),
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
                    "label": getattr(region, "name", key),
                    "node_type": "region_proxy" if region_is_proxy else "region",
                    "description": (
                        "Atlas-backed proxy node for a chapter-level system"
                        if region_is_proxy else
                        "Atlas-backed circuit node"
                    ),
                    "atlas_region": getattr(region, "name", None),
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": self._feature_summary(receptor_df, gene_df, conn_df),
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

    def simulate(
        self,
        genetic_vulnerability: float = 0.50,
        innate_emotional_reactivity: float = 0.60,
        early_life_stress: float = 0.40,
        social_attention_reinforcement: float = 0.60,
        interpersonal_stress: float = 0.50,
        ssri_support: float = 0.00,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent 0..1 simulator.

        Inputs reflect risk loads or protective supports. Higher regional_state
        values mean higher inferred burden / dysregulation.

        Returns a dictionary of pandas Series:
        - inputs
        - latents
        - regional_state
        - symptoms
        - phenotypes
        """
        inputs = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "innate_emotional_reactivity": self._clip01(innate_emotional_reactivity),
            "early_life_stress": self._clip01(early_life_stress),
            "social_attention_reinforcement": self._clip01(social_attention_reinforcement),
            "interpersonal_stress": self._clip01(interpersonal_stress),
            "ssri_support": self._clip01(ssri_support),
            "recovery_support": self._clip01(recovery_support),
        }

        latents = {
            # Reward / novelty / social-centrality pressure.
            "dopaminergic_attention_reward_drive": self._clip01(
                0.34 * inputs["genetic_vulnerability"]
                + 0.26 * inputs["social_attention_reinforcement"]
                + 0.18 * inputs["innate_emotional_reactivity"]
                + 0.10 * inputs["interpersonal_stress"]
                - 0.08 * inputs["recovery_support"]
            ),
            # Serotonergic weakness in inhibitory control and affect modulation.
            "serotonergic_disinhibition": self._clip01(
                0.30 * inputs["genetic_vulnerability"]
                + 0.23 * inputs["early_life_stress"]
                + 0.20 * inputs["interpersonal_stress"]
                + 0.12 * inputs["innate_emotional_reactivity"]
                - 0.22 * inputs["ssri_support"]
                - 0.10 * inputs["recovery_support"]
            ),
            # Developmental / stress-linked weak integration of control and limbic systems.
            "frontolimbic_disconnection": self._clip01(
                0.30 * inputs["early_life_stress"]
                + 0.24 * inputs["genetic_vulnerability"]
                + 0.18 * inputs["interpersonal_stress"]
                + 0.10 * inputs["innate_emotional_reactivity"]
                - 0.12 * inputs["recovery_support"]
            ),
        }

        latents["frontolimbic_dysregulation"] = self._clip01(
            0.40 * latents["serotonergic_disinhibition"]
            + 0.30 * latents["frontolimbic_disconnection"]
            + 0.18 * inputs["innate_emotional_reactivity"]
            + 0.10 * inputs["interpersonal_stress"]
            - 0.12 * inputs["recovery_support"]
        )

        latents["prefrontal_social_decision_impairment"] = self._clip01(
            0.40 * latents["frontolimbic_disconnection"]
            + 0.24 * latents["serotonergic_disinhibition"]
            + 0.14 * inputs["interpersonal_stress"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.16 * inputs["recovery_support"]
            - 0.08 * inputs["ssri_support"]
        )

        latents["attention_deprivation_empty_state"] = self._clip01(
            0.36 * latents["dopaminergic_attention_reward_drive"]
            + 0.18 * latents["serotonergic_disinhibition"]
            + 0.16 * inputs["interpersonal_stress"]
            + 0.12 * inputs["social_attention_reinforcement"]
            - 0.18 * inputs["recovery_support"]
        )

        regional_state = {
            "amygdala": self._clip01(
                0.42 * inputs["innate_emotional_reactivity"]
                + 0.24 * inputs["interpersonal_stress"]
                + 0.20 * latents["frontolimbic_dysregulation"]
                + 0.10 * inputs["early_life_stress"]
            ),
            "ofc": self._clip01(
                0.46 * latents["prefrontal_social_decision_impairment"]
                + 0.20 * latents["serotonergic_disinhibition"]
                + 0.14 * latents["frontolimbic_disconnection"]
                + 0.08 * inputs["interpersonal_stress"]
            ),
            "vmpfc_proxy": self._clip01(
                0.48 * latents["prefrontal_social_decision_impairment"]
                + 0.22 * latents["frontolimbic_disconnection"]
                + 0.10 * inputs["interpersonal_stress"]
            ),
            "acc_proxy": self._clip01(
                0.32 * latents["frontolimbic_dysregulation"]
                + 0.22 * latents["serotonergic_disinhibition"]
                + 0.14 * inputs["interpersonal_stress"]
            ),
            "insula_proxy": self._clip01(
                0.30 * latents["attention_deprivation_empty_state"]
                + 0.18 * latents["frontolimbic_dysregulation"]
                + 0.14 * inputs["interpersonal_stress"]
            ),
        }

        symptoms = {
            "attention_seeking": self._clip01(
                0.44 * latents["dopaminergic_attention_reward_drive"]
                + 0.18 * latents["attention_deprivation_empty_state"]
                + 0.12 * inputs["social_attention_reinforcement"]
                + 0.08 * regional_state["vmpfc_proxy"]
                - 0.10 * inputs["recovery_support"]
            ),
            "novelty_sensation_seeking": self._clip01(
                0.42 * latents["dopaminergic_attention_reward_drive"]
                + 0.16 * regional_state["amygdala"]
                + 0.14 * latents["attention_deprivation_empty_state"]
                + 0.10 * inputs["genetic_vulnerability"]
                - 0.08 * inputs["recovery_support"]
            ),
            "short_lived_intense_engagement": self._clip01(
                0.34 * latents["dopaminergic_attention_reward_drive"]
                + 0.20 * latents["attention_deprivation_empty_state"]
                + 0.14 * inputs["interpersonal_stress"]
                + 0.08 * regional_state["amygdala"]
            ),
            "affective_lability": self._clip01(
                0.36 * latents["serotonergic_disinhibition"]
                + 0.28 * latents["frontolimbic_dysregulation"]
                + 0.12 * regional_state["amygdala"]
                - 0.14 * inputs["ssri_support"]
            ),
            "impulsive_dramatic_outbursts": self._clip01(
                0.30 * latents["serotonergic_disinhibition"]
                + 0.24 * latents["frontolimbic_dysregulation"]
                + 0.14 * regional_state["ofc"]
                + 0.10 * regional_state["amygdala"]
                - 0.10 * inputs["ssri_support"]
                - 0.06 * inputs["recovery_support"]
            ),
            "anger_dyscontrol": self._clip01(
                0.34 * latents["serotonergic_disinhibition"]
                + 0.22 * latents["frontolimbic_dysregulation"]
                + 0.12 * regional_state["amygdala"]
                - 0.12 * inputs["ssri_support"]
            ),
            "boredom_emptiness": self._clip01(
                0.46 * latents["attention_deprivation_empty_state"]
                + 0.16 * latents["serotonergic_disinhibition"]
                + 0.10 * inputs["interpersonal_stress"]
                - 0.10 * inputs["recovery_support"]
            ),
            "provocative_boundary_blurring": self._clip01(
                0.28 * latents["dopaminergic_attention_reward_drive"]
                + 0.28 * latents["prefrontal_social_decision_impairment"]
                + 0.12 * regional_state["vmpfc_proxy"]
                + 0.08 * inputs["social_attention_reinforcement"]
                - 0.10 * inputs["recovery_support"]
            ),
            "misperceived_intimacy": self._clip01(
                0.34 * latents["prefrontal_social_decision_impairment"]
                + 0.18 * latents["dopaminergic_attention_reward_drive"]
                + 0.12 * regional_state["vmpfc_proxy"]
                + 0.08 * inputs["interpersonal_stress"]
            ),
            "self_destructive_gestures": self._clip01(
                0.26 * latents["serotonergic_disinhibition"]
                + 0.22 * latents["attention_deprivation_empty_state"]
                + 0.10 * inputs["interpersonal_stress"]
                - 0.10 * inputs["ssri_support"]
                - 0.08 * inputs["recovery_support"]
            ),
        }

        phenotypes = {
            "attention_driven_dramatic_profile": self._clip01(
                (
                    symptoms["attention_seeking"]
                    + symptoms["novelty_sensation_seeking"]
                    + symptoms["short_lived_intense_engagement"]
                ) / 3.0
            ),
            "impulsive_affective_instability_profile": self._clip01(
                (
                    symptoms["affective_lability"]
                    + symptoms["impulsive_dramatic_outbursts"]
                    + symptoms["anger_dyscontrol"]
                    + symptoms["self_destructive_gestures"]
                ) / 4.0
            ),
            "interpersonal_boundary_instability_profile": self._clip01(
                (
                    symptoms["provocative_boundary_blurring"]
                    + symptoms["misperceived_intimacy"]
                    + symptoms["attention_seeking"]
                ) / 3.0
            ),
            "depleted_empty_rebound_profile": self._clip01(
                (
                    symptoms["boredom_emptiness"]
                    + symptoms["attention_seeking"]
                    + symptoms["novelty_sensation_seeking"]
                ) / 3.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, dtype=float),
            "latents": pd.Series(latents, dtype=float),
            "regional_state": pd.Series(regional_state, dtype=float),
            "symptoms": pd.Series(symptoms, dtype=float),
            "phenotypes": pd.Series(phenotypes, dtype=float),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 point to the statistical map of the selected parcellation.

        Returns an empty DataFrame if siibra is unavailable or if assignment fails.
        """
        if not self.siibra_available or siibra is None:
            return pd.DataFrame(
                [{"message": "siibra unavailable; coordinate assignment not possible in this environment."}]
            )

        if self._pmap is None:
            try:
                with self._quiet:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception as exc:
                return pd.DataFrame([{"message": f"Could not load statistical map: {exc!r}"}])

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with self._quiet:
                assignments = self._pmap.assign(point)
            if assignments is None:
                return pd.DataFrame()

            # Keep common scoring columns near the top when available.
            sort_candidates = [
                "map value",
                "correlation",
                "intersection over union",
                "score",
                "probability",
            ]
            for col in sort_candidates:
                if col in assignments.columns:
                    assignments = assignments.sort_values(col, ascending=False)
                    break
            return assignments.reset_index(drop=True)
        except Exception as exc:
            return pd.DataFrame([{"message": f"Point assignment failed: {exc!r}"}])

    def region_mask(self, node_key: str) -> Any:
        """
        Return a regional map / mask-like object when possible.

        Because siibra APIs vary across versions, this helper attempts a few
        compatible access patterns and otherwise returns None.
        """
        region = self.region_objects.get(node_key)
        if region is None or not self.siibra_available:
            return None

        if hasattr(region, "fetch_regional_map"):
            for kwargs in (
                {"space": self.space, "maptype": "labelled"},
                {"space": self.space, "maptype": "statistical"},
                {"space": self.space},
            ):
                try:
                    return region.fetch_regional_map(**kwargs)
                except Exception:
                    continue

        if self._pmap is None:
            try:
                with self._quiet:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.space_spec,
                        maptype="statistical",
                    )
            except Exception:
                self._pmap = None

        if self._pmap is not None:
            try:
                if hasattr(self._pmap, "fetch"):
                    return self._pmap.fetch(region)
            except Exception:
                pass

        return None


if __name__ == "__main__":
    model = HistrionicPersonalityDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODE TABLE ===")
    print(
        built["nodes"][
            ["key", "node_type", "atlas_region", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== EDGE TABLE (first 12 rows) ===")
    print(built["edges"].head(12).to_string(index=False))

    print("\n=== WITHIN-CIRCUIT CONNECTIVITY ===")
    circuit_df = built["circuit_connectivity"]
    if isinstance(circuit_df, pd.DataFrame) and not circuit_df.empty:
        print(circuit_df.to_string())
    else:
        print("No circuit connectivity matrix available in this environment.")

    print("\n=== AVAILABLE REGION-SPECIFIC TABLES ===")
    for key in ("amygdala", "ofc", "vmpfc_proxy"):
        receptor_rows = len(built["receptors"].get(key, pd.DataFrame()))
        gene_rows = len(built["genes"].get(key, pd.DataFrame()))
        conn_rows = len(built["connectivity_profiles"].get(key, pd.DataFrame()))
        print(f"{key}: receptors={receptor_rows}, genes={gene_rows}, connectivity_rows={conn_rows}")

    sim = model.simulate(
        genetic_vulnerability=0.65,
        innate_emotional_reactivity=0.75,
        early_life_stress=0.55,
        social_attention_reinforcement=0.80,
        interpersonal_stress=0.60,
        ssri_support=0.15,
        recovery_support=0.30,
    )

    print("\n=== INPUTS ===")
    print(sim["inputs"].to_string())

    print("\n=== LATENT BIOLOGY ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== REGIONAL STATE (BURDEN / DYSREGULATION) ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())

    print("\n=== SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Optional coordinate assignment example:
    # print(model.assign_mni_point((-22, -4, -18)).head())
