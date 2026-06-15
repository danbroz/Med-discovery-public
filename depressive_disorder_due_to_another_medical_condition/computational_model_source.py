
from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Depressive Disorder Due to Another Medical Condition.

This script turns a chapter-level biological summary into a transparent mechanistic
research scaffold. It is intended for hypothesis generation, educational use, and
iterative refinement against atlas-backed evidence. It is not a diagnostic or
treatment tool.

Chapter logic encoded here emphasizes:
- medical illness as a biological and psychosocial stressor,
- monoamine disruption (serotonin / norepinephrine / dopamine),
- neuroinflammation and neuroendocrine stress pathways,
- lesion-related and network-level circuit disruption,
- left DLPFC / basal ganglia / amygdala / subgenual ACC involvement,
- symptom emergence through distributed frontolimbic and frontostriatal dysregulation.

The scaffold is compatibility-first and degrades gracefully when:
- siibra is not installed,
- a requested Julich region cannot be resolved,
- receptor, gene-expression, or connectivity features are unavailable.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency in authoring environments
    import siibra  # type: ignore
except Exception:  # pragma: no cover
    siibra = None


DEFAULT_GENE_PANEL = [
    "SLC6A4",  # serotonin transporter
    "HTR1A",   # serotonin receptor
    "HTR2A",   # serotonin receptor
    "SLC6A2",  # norepinephrine transporter
    "DRD2",    # dopamine receptor
    "COMT",    # catecholamine metabolism
    "MAOA",    # monoamine metabolism
    "BDNF",    # neuroplasticity
    "CRHR1",   # stress-response signaling
    "NR3C1",   # glucocorticoid receptor
    "IL6",     # inflammatory signaling
    "TNF",     # inflammatory signaling
]


class DepressiveDisorderDueToAnotherMedicalConditionModel:
    """
    Mechanistic siibra scaffold for Depressive Disorder Due to Another Medical Condition.

    The model is anchored to the chapter's core ideas:
    1) medical illnesses and treatments can disturb monoamine systems,
    2) focal neurological injury can disconnect mood-relevant circuits,
    3) inflammation and stress biology can shift limbic / prefrontal activity,
    4) inherited vulnerability shapes the depressive response to medical insults.

    It uses normalized 0..1 arithmetic in a one-pass simulator:
    inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

    This is a research scaffold, not a validated disease model.
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
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

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
            except Exception as exc:
                warnings.warn(
                    f"siibra is installed but atlas initialization failed: {exc}. "
                    "Atlas-backed methods will return empty results."
                )
        else:
            warnings.warn(
                "siibra is not installed in this environment. "
                "Atlas-backed methods will return empty results, but simulate() still works."
            )

        # Region nodes: direct anchors where the chapter was anatomically specific,
        # plus clearly labeled proxies where precision would otherwise be overstated.
        self.region_candidates: Dict[str, List[str]] = {
            "left_dlpfc": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "dorsolateral prefrontal cortex left",
                "middle frontal gyrus left",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "sgacc_proxy": [
                "Area s24 left",
                "Area p32 left",
                "subgenual anterior cingulate left",
                "anterior cingulate cortex left",
                "cingulate cortex left",
            ],
            "left_basal_ganglia_proxy": [
                "caudate nucleus left",
                "putamen left",
                "striatum left",
                "basal ganglia left",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "left_dlpfc": (
                "Left dorsolateral prefrontal cortex anchor for cognitive control and "
                "executive dysfunction described in depression and post-stroke depression."
            ),
            "amygdala": (
                "Amygdala anchor for negative affect, threat bias, and inflammation-linked "
                "limbic hyperreactivity."
            ),
            "sgacc_proxy": (
                "Subgenual anterior cingulate proxy for mood-valence bias and rumination; "
                "kept as a proxy because exact Julich labels can vary across environments."
            ),
            "left_basal_ganglia_proxy": (
                "Left basal ganglia / striatal proxy for lesion-related and dopaminergic "
                "motivational disruption; intentionally conservative."
            ),
        }
        self.proxy_region_keys = {"sgacc_proxy", "left_basal_ganglia_proxy"}

        self.input_nodes: Dict[str, str] = {
            "medical_insult_burden": (
                "Overall physiological burden of the underlying medical condition."
            ),
            "focal_lesion_burden": (
                "Burden from focal brain injury, especially stroke or lesion load in mood-relevant pathways."
            ),
            "neurodegenerative_burden": (
                "Progressive neuronal loss that can diminish dopaminergic and broader neuromodulatory tone."
            ),
            "seizure_network_burden": (
                "Epilepsy-related network instability contributing to mood vulnerability."
            ),
            "medication_depressogenic_load": (
                "Depressogenic medication exposure such as interferon, methyldopa, or steroids."
            ),
            "inflammatory_burden": (
                "Systemic cytokine-driven inflammatory pressure affecting brain function."
            ),
            "endocrine_metabolic_burden": (
                "Endocrine or metabolic dysregulation that shifts neurochemical and stress systems."
            ),
            "genetic_vulnerability": (
                "Inherited liability shaping sensitivity of stress, monoamine, and mood-regulation systems."
            ),
            "psychosocial_stress_burden": (
                "Stress associated with illness, disability, uncertainty, or chronic disease management."
            ),
            "medical_condition_correction": (
                "Protective improvement from treating or stabilizing the underlying medical condition."
            ),
            "antidepressant_support": (
                "Protective support from antidepressant or other mood-targeted intervention."
            ),
            "rehabilitation_and_social_support": (
                "Protective cognitive, functional, and social recovery support."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "neuroinflammatory_signaling": (
                "Inflammatory signaling that alters limbic and prefrontal function."
            ),
            "hpa_axis_dysregulation": (
                "Stress-system dysregulation linking medical burden to depressive physiology."
            ),
            "serotonergic_loss": (
                "Reduced serotonergic signaling from illness, lesions, medications, or inflammatory effects."
            ),
            "noradrenergic_loss": (
                "Reduced noradrenergic signaling impairing arousal, drive, and cortical regulation."
            ),
            "dopaminergic_loss": (
                "Reduced dopaminergic signaling contributing to anhedonia and psychomotor slowing."
            ),
            "monoamine_disruption": (
                "Integrated monoaminergic dysfunction across serotonin, norepinephrine, and dopamine systems."
            ),
            "frontostriatal_circuit_disconnection": (
                "Disruption of cortico-striato-thalamo-cortical loops and connected control systems."
            ),
            "diaschisis_network_burden": (
                "Downstream network dysfunction in connected but structurally intact regions after focal injury."
            ),
            "limbic_prefrontal_imbalance": (
                "Mood network bias toward limbic negativity and away from top-down control."
            ),
            "motivational_anhedonia": (
                "Reduced reward valuation and initiation caused by dopaminergic and circuit burden."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "depressed_mood": "Low mood and emotional suffering attributable to medical-condition-driven biology.",
            "anhedonia": "Loss of interest or pleasure, especially under dopaminergic or striatal disruption.",
            "executive_dysfunction": "Cognitive slowing, planning difficulty, and reduced top-down regulation.",
            "psychomotor_slowing": "Reduced motor and behavioral initiation often linked to basal ganglia burden.",
            "fatigue_low_energy": "Low energy emerging from inflammatory, medical, and depressive mechanisms.",
            "rumination_negative_bias": "Persistent negative appraisal and repetitive negative thought.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "medical_insult_burden",
                "target": "monoamine_disruption",
                "relation": "broad physiological illness burden perturbs neuromodulatory homeostasis",
                "ddamc_change": "increased",
            },
            {
                "source": "medication_depressogenic_load",
                "target": "monoamine_disruption",
                "relation": "depressogenic medications can disrupt monoamine signaling",
                "ddamc_change": "increased",
            },
            {
                "source": "focal_lesion_burden",
                "target": "serotonergic_loss",
                "relation": "lesions can interrupt ascending serotonergic projections",
                "ddamc_change": "increased",
            },
            {
                "source": "focal_lesion_burden",
                "target": "noradrenergic_loss",
                "relation": "lesions can interrupt ascending noradrenergic projections",
                "ddamc_change": "increased",
            },
            {
                "source": "neurodegenerative_burden",
                "target": "dopaminergic_loss",
                "relation": "neurodegeneration can reduce dopaminergic tone",
                "ddamc_change": "increased",
            },
            {
                "source": "inflammatory_burden",
                "target": "neuroinflammatory_signaling",
                "relation": "systemic inflammation propagates cytokine-linked brain effects",
                "ddamc_change": "increased",
            },
            {
                "source": "psychosocial_stress_burden",
                "target": "hpa_axis_dysregulation",
                "relation": "illness-related stress sensitizes neuroendocrine stress systems",
                "ddamc_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "hpa_axis_dysregulation",
                "relation": "familial mood-disorder liability amplifies stress reactivity",
                "ddamc_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "limbic_prefrontal_imbalance",
                "relation": "genetic diathesis biases network responses to injury and stress",
                "ddamc_change": "increased",
            },
            {
                "source": "focal_lesion_burden",
                "target": "frontostriatal_circuit_disconnection",
                "relation": "stroke or focal injury disconnects mood-regulation circuits",
                "ddamc_change": "increased",
            },
            {
                "source": "seizure_network_burden",
                "target": "frontostriatal_circuit_disconnection",
                "relation": "epilepsy-related network dysfunction burdens mood circuits",
                "ddamc_change": "increased",
            },
            {
                "source": "focal_lesion_burden",
                "target": "diaschisis_network_burden",
                "relation": "focal injury reduces function in connected intact regions",
                "ddamc_change": "increased",
            },
            {
                "source": "serotonergic_loss",
                "target": "monoamine_disruption",
                "relation": "serotonergic deficiency contributes to the global monoamine burden",
                "ddamc_change": "increased",
            },
            {
                "source": "noradrenergic_loss",
                "target": "monoamine_disruption",
                "relation": "noradrenergic deficiency contributes to the global monoamine burden",
                "ddamc_change": "increased",
            },
            {
                "source": "dopaminergic_loss",
                "target": "monoamine_disruption",
                "relation": "dopaminergic deficiency contributes to the global monoamine burden",
                "ddamc_change": "increased",
            },
            {
                "source": "monoamine_disruption",
                "target": "limbic_prefrontal_imbalance",
                "relation": "monoamine loss destabilizes distributed mood-regulation networks",
                "ddamc_change": "increased",
            },
            {
                "source": "neuroinflammatory_signaling",
                "target": "limbic_prefrontal_imbalance",
                "relation": "inflammation biases prefrontal-limbic activity toward depression",
                "ddamc_change": "increased",
            },
            {
                "source": "frontostriatal_circuit_disconnection",
                "target": "left_dlpfc",
                "relation": "circuit disconnection weakens cognitive-control territory function",
                "ddamc_change": "decreased",
            },
            {
                "source": "neuroinflammatory_signaling",
                "target": "sgacc_proxy",
                "relation": "inflammation can enhance activity in negative-affect cingulate circuitry",
                "ddamc_change": "increased",
            },
            {
                "source": "neuroinflammatory_signaling",
                "target": "amygdala",
                "relation": "inflammation can enhance limbic threat and negative-affect reactivity",
                "ddamc_change": "increased",
            },
            {
                "source": "dopaminergic_loss",
                "target": "left_basal_ganglia_proxy",
                "relation": "dopaminergic loss diminishes striatal motivational throughput",
                "ddamc_change": "decreased",
            },
            {
                "source": "left_dlpfc",
                "target": "executive_dysfunction",
                "relation": "reduced DLPFC function impairs cognitive control and executive performance",
                "ddamc_change": "increased",
            },
            {
                "source": "sgacc_proxy",
                "target": "depressed_mood",
                "relation": "negative-valence cingulate activity promotes sustained low mood",
                "ddamc_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "rumination_negative_bias",
                "relation": "amygdala hyperreactivity reinforces negative affective appraisal",
                "ddamc_change": "increased",
            },
            {
                "source": "left_basal_ganglia_proxy",
                "target": "anhedonia",
                "relation": "striatal disruption reduces motivation and reward responsiveness",
                "ddamc_change": "increased",
            },
            {
                "source": "left_basal_ganglia_proxy",
                "target": "psychomotor_slowing",
                "relation": "basal ganglia dysfunction slows initiation and movement",
                "ddamc_change": "increased",
            },
            {
                "source": "medical_condition_correction",
                "target": "neuroinflammatory_signaling",
                "relation": "effective treatment of the underlying condition can reduce biological burden",
                "ddamc_change": "decreased",
            },
            {
                "source": "antidepressant_support",
                "target": "monoamine_disruption",
                "relation": "mood-targeted treatment can normalize monoaminergic function",
                "ddamc_change": "decreased",
            },
            {
                "source": "rehabilitation_and_social_support",
                "target": "frontostriatal_circuit_disconnection",
                "relation": "rehabilitation and support may improve network compensation and control",
                "ddamc_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if not self.siibra_available:
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
        if not self.siibra_available or concept is None:
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
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex"} else 0
        cyto_bonus = 0 if "area " in name or "(" in name else 1
        return (left_bonus, right_penalty, generic_penalty, cyto_bonus)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available or self.atlas is None or self.parcellation is None:
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
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row_key = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append(
                {
                    "name": row_key[0],
                    "identifier": row_key[1],
                    "parcellation": row_key[2],
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

    def _centroid_tuple(self, centroid: Any) -> Optional[Tuple[float, float, float]]:
        if centroid is None:
            return None
        try:
            vals = tuple(float(x) for x in centroid)
            if len(vals) >= 3:
                return (vals[0], vals[1], vals[2])
        except Exception:
            pass
        coords = []
        for attr in ("x", "y", "z"):
            if hasattr(centroid, attr):
                coords.append(float(getattr(centroid, attr)))
        return tuple(coords) if len(coords) == 3 else None

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid_xyz = self._centroid_tuple(getattr(main, "centroid", None))
        volume_raw = getattr(main, "volume", None)
        volume_mm3 = float(volume_raw) if volume_raw is not None else None
        return centroid_xyz, volume_mm3

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

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = compound[0]
            data = getattr(first, "data", None)
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
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile_from_matrix(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
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
            series = series.sort_values(ascending=False)
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != getattr(region, "name", "")]
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        # Prefer a matrix-based route because it also supports pairwise circuit extraction.
        df = self._connectivity_profile_from_matrix(region, max_rows=max_rows)
        if not df.empty:
            return df

        # Fallback: query region-level connectivity features directly if available.
        feats = self._safe_features_any(region, self._modality_candidates("connectivity"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                data = getattr(feat, "data", None)
                if isinstance(data, pd.DataFrame):
                    out = data.copy().reset_index(drop=False)
                    out.columns = [str(c) for c in out.columns]
                    return out.head(max_rows)
            except Exception:
                continue
        return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a pairwise connectivity table among resolved regional circuit nodes.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        labels: Dict[str, Any] = {}
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                labels[key] = label

        rows: List[Dict[str, Any]] = []
        keys = list(labels.keys())
        for i, source in enumerate(keys):
            for target in keys[i + 1 :]:
                s_label = labels[source]
                t_label = labels[target]
                value = None
                try:
                    value = matrix.loc[s_label, t_label]
                except Exception:
                    try:
                        value = matrix.loc[t_label, s_label]
                    except Exception:
                        try:
                            value = matrix[s_label][t_label]
                        except Exception:
                            value = None
                if value is not None:
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "value": float(value),
                        }
                    )
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["source", "target", "value"])

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        """
        Resolve atlas regions when possible and gather receptor / gene / connectivity summaries.
        """
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node.")
            if region is None:
                if self.siibra_available:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key in self.proxy_region_keys else "region",
                        "description": f"{desc} Unresolved in the current environment.",
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
                    "label": getattr(region, "name", key.replace("_", " ").title()),
                    "node_type": "region_proxy" if key in self.proxy_region_keys else "region",
                    "description": desc,
                    "atlas_region": getattr(region, "name", None),
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
        Probabilistically assign an MNI152 point to Julich regions using a statistical map.
        """
        if not self.siibra_available:
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(v) for v in xyz[:3]), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a region-specific mask / map representation when available.
        """
        if not self.siibra_available:
            return None
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        # Try a regional statistical map first.
        try:
            if hasattr(region, "get_regional_map"):
                regional_map = region.get_regional_map(self.assignment_space, "statistical")
                try:
                    return regional_map.fetch()
                except Exception:
                    return regional_map
        except Exception:
            pass

        # Fallback: use the global parcellation statistical map and fetch the region's fragment.
        try:
            pmap = siibra.get_map(
                parcellation=self.parcellation_spec,
                space=self.assignment_space,
                maptype="statistical",
            )
            try:
                return pmap.fetch(region=region)
            except Exception:
                return pmap
        except Exception:
            return None

    def simulate(
        self,
        medical_insult_burden: float = 0.50,
        focal_lesion_burden: float = 0.00,
        neurodegenerative_burden: float = 0.00,
        seizure_network_burden: float = 0.00,
        medication_depressogenic_load: float = 0.10,
        inflammatory_burden: float = 0.20,
        endocrine_metabolic_burden: float = 0.15,
        genetic_vulnerability: float = 0.25,
        psychosocial_stress_burden: float = 0.30,
        medical_condition_correction: float = 0.20,
        antidepressant_support: float = 0.20,
        rehabilitation_and_social_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Inputs are clipped to [0, 1]. Protective variables subtract from pathological
        latent states and downstream symptoms.
        """
        inputs = {
            "medical_insult_burden": self._clip01(medical_insult_burden),
            "focal_lesion_burden": self._clip01(focal_lesion_burden),
            "neurodegenerative_burden": self._clip01(neurodegenerative_burden),
            "seizure_network_burden": self._clip01(seizure_network_burden),
            "medication_depressogenic_load": self._clip01(medication_depressogenic_load),
            "inflammatory_burden": self._clip01(inflammatory_burden),
            "endocrine_metabolic_burden": self._clip01(endocrine_metabolic_burden),
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "psychosocial_stress_burden": self._clip01(psychosocial_stress_burden),
            "medical_condition_correction": self._clip01(medical_condition_correction),
            "antidepressant_support": self._clip01(antidepressant_support),
            "rehabilitation_and_social_support": self._clip01(rehabilitation_and_social_support),
        }

        support_mean = self._clip01(
            (
                inputs["medical_condition_correction"]
                + inputs["antidepressant_support"]
                + inputs["rehabilitation_and_social_support"]
            ) / 3.0
        )

        latents = {
            "neuroinflammatory_signaling": self._clip01(
                0.45 * inputs["medical_insult_burden"]
                + 0.45 * inputs["inflammatory_burden"]
                + 0.25 * inputs["endocrine_metabolic_burden"]
                + 0.15 * inputs["psychosocial_stress_burden"]
                - 0.25 * inputs["medical_condition_correction"]
            ),
            "hpa_axis_dysregulation": self._clip01(
                0.35 * inputs["psychosocial_stress_burden"]
                + 0.25 * inputs["medical_insult_burden"]
                + 0.25 * inputs["inflammatory_burden"]
                + 0.20 * inputs["genetic_vulnerability"]
                - 0.20 * inputs["rehabilitation_and_social_support"]
            ),
            "serotonergic_loss": self._clip01(
                0.25 * inputs["medical_insult_burden"]
                + 0.25 * inputs["medication_depressogenic_load"]
                + 0.20 * inputs["focal_lesion_burden"]
                + 0.20 * inputs["inflammatory_burden"]
                + 0.15 * inputs["genetic_vulnerability"]
                - 0.20 * inputs["antidepressant_support"]
                - 0.15 * inputs["medical_condition_correction"]
            ),
            "noradrenergic_loss": self._clip01(
                0.25 * inputs["medical_insult_burden"]
                + 0.25 * inputs["medication_depressogenic_load"]
                + 0.25 * inputs["focal_lesion_burden"]
                + 0.15 * inputs["psychosocial_stress_burden"]
                + 0.10 * inputs["genetic_vulnerability"]
                - 0.15 * inputs["antidepressant_support"]
                - 0.10 * inputs["rehabilitation_and_social_support"]
            ),
            "dopaminergic_loss": self._clip01(
                0.35 * inputs["neurodegenerative_burden"]
                + 0.20 * inputs["medical_insult_burden"]
                + 0.15 * inputs["medication_depressogenic_load"]
                + 0.10 * inputs["inflammatory_burden"]
                + 0.10 * inputs["genetic_vulnerability"]
                - 0.15 * inputs["medical_condition_correction"]
                - 0.10 * inputs["antidepressant_support"]
            ),
        }

        latents["monoamine_disruption"] = self._clip01(
            0.35 * latents["serotonergic_loss"]
            + 0.30 * latents["noradrenergic_loss"]
            + 0.35 * latents["dopaminergic_loss"]
        )

        latents["frontostriatal_circuit_disconnection"] = self._clip01(
            0.40 * inputs["focal_lesion_burden"]
            + 0.25 * inputs["neurodegenerative_burden"]
            + 0.20 * inputs["seizure_network_burden"]
            + 0.15 * latents["monoamine_disruption"]
            + 0.10 * latents["neuroinflammatory_signaling"]
            - 0.20 * inputs["rehabilitation_and_social_support"]
        )

        latents["diaschisis_network_burden"] = self._clip01(
            0.45 * inputs["focal_lesion_burden"]
            + 0.20 * inputs["seizure_network_burden"]
            + 0.20 * latents["frontostriatal_circuit_disconnection"]
            + 0.10 * inputs["medical_insult_burden"]
            - 0.15 * inputs["rehabilitation_and_social_support"]
        )

        latents["limbic_prefrontal_imbalance"] = self._clip01(
            0.30 * latents["monoamine_disruption"]
            + 0.25 * latents["neuroinflammatory_signaling"]
            + 0.20 * latents["hpa_axis_dysregulation"]
            + 0.15 * latents["frontostriatal_circuit_disconnection"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.20 * inputs["antidepressant_support"]
            - 0.10 * inputs["rehabilitation_and_social_support"]
        )

        latents["motivational_anhedonia"] = self._clip01(
            0.45 * latents["dopaminergic_loss"]
            + 0.25 * latents["frontostriatal_circuit_disconnection"]
            + 0.15 * latents["monoamine_disruption"]
            + 0.10 * inputs["medical_insult_burden"]
            - 0.10 * inputs["medical_condition_correction"]
        )

        regional_state = {
            "left_dlpfc_hypofunction": self._clip01(
                0.35 * latents["frontostriatal_circuit_disconnection"]
                + 0.25 * latents["monoamine_disruption"]
                + 0.20 * latents["hpa_axis_dysregulation"]
                + 0.15 * latents["diaschisis_network_burden"]
                - 0.20 * inputs["antidepressant_support"]
                - 0.15 * inputs["rehabilitation_and_social_support"]
            ),
            "sgacc_overdrive": self._clip01(
                0.35 * latents["limbic_prefrontal_imbalance"]
                + 0.25 * latents["neuroinflammatory_signaling"]
                + 0.15 * latents["hpa_axis_dysregulation"]
                + 0.10 * latents["monoamine_disruption"]
                - 0.15 * inputs["antidepressant_support"]
            ),
            "amygdala_hyperreactivity": self._clip01(
                0.35 * latents["neuroinflammatory_signaling"]
                + 0.25 * latents["hpa_axis_dysregulation"]
                + 0.20 * latents["limbic_prefrontal_imbalance"]
                + 0.10 * inputs["psychosocial_stress_burden"]
                - 0.15 * inputs["rehabilitation_and_social_support"]
            ),
            "left_basal_ganglia_disruption": self._clip01(
                0.35 * inputs["focal_lesion_burden"]
                + 0.30 * latents["dopaminergic_loss"]
                + 0.15 * latents["frontostriatal_circuit_disconnection"]
                + 0.10 * inputs["neurodegenerative_burden"]
                - 0.10 * inputs["medical_condition_correction"]
            ),
        }

        symptoms: Dict[str, float] = {}

        symptoms["depressed_mood"] = self._clip01(
            0.35 * regional_state["sgacc_overdrive"]
            + 0.25 * latents["monoamine_disruption"]
            + 0.20 * regional_state["amygdala_hyperreactivity"]
            + 0.10 * latents["hpa_axis_dysregulation"]
            - 0.15 * support_mean
        )

        symptoms["anhedonia"] = self._clip01(
            0.45 * latents["motivational_anhedonia"]
            + 0.25 * regional_state["left_basal_ganglia_disruption"]
            + 0.15 * latents["monoamine_disruption"]
            - 0.15 * support_mean
        )

        symptoms["executive_dysfunction"] = self._clip01(
            0.45 * regional_state["left_dlpfc_hypofunction"]
            + 0.25 * latents["frontostriatal_circuit_disconnection"]
            + 0.15 * latents["diaschisis_network_burden"]
            - 0.10 * inputs["rehabilitation_and_social_support"]
        )

        symptoms["psychomotor_slowing"] = self._clip01(
            0.40 * regional_state["left_basal_ganglia_disruption"]
            + 0.25 * latents["dopaminergic_loss"]
            + 0.15 * latents["diaschisis_network_burden"]
            - 0.10 * inputs["medical_condition_correction"]
        )

        symptoms["fatigue_low_energy"] = self._clip01(
            0.30 * inputs["medical_insult_burden"]
            + 0.25 * latents["neuroinflammatory_signaling"]
            + 0.20 * symptoms["depressed_mood"]
            + 0.10 * latents["monoamine_disruption"]
            - 0.10 * inputs["medical_condition_correction"]
        )

        symptoms["rumination_negative_bias"] = self._clip01(
            0.35 * regional_state["sgacc_overdrive"]
            + 0.25 * regional_state["amygdala_hyperreactivity"]
            + 0.20 * latents["limbic_prefrontal_imbalance"]
            - 0.10 * inputs["rehabilitation_and_social_support"]
        )

        phenotypes = {
            "vascular_depression_profile": self._clip01(
                0.35 * inputs["focal_lesion_burden"]
                + 0.20 * latents["frontostriatal_circuit_disconnection"]
                + 0.20 * regional_state["left_dlpfc_hypofunction"]
                + 0.15 * symptoms["executive_dysfunction"]
                + 0.10 * symptoms["psychomotor_slowing"]
            ),
            "inflammatory_depression_profile": self._clip01(
                0.30 * inputs["inflammatory_burden"]
                + 0.25 * latents["neuroinflammatory_signaling"]
                + 0.20 * symptoms["fatigue_low_energy"]
                + 0.15 * regional_state["amygdala_hyperreactivity"]
                + 0.10 * symptoms["depressed_mood"]
            ),
            "neurodegenerative_depression_profile": self._clip01(
                0.35 * inputs["neurodegenerative_burden"]
                + 0.25 * latents["dopaminergic_loss"]
                + 0.20 * symptoms["anhedonia"]
                + 0.10 * symptoms["psychomotor_slowing"]
            ),
            "network_injury_profile": self._clip01(
                0.30 * inputs["focal_lesion_burden"]
                + 0.20 * inputs["seizure_network_burden"]
                + 0.20 * latents["diaschisis_network_burden"]
                + 0.15 * symptoms["executive_dysfunction"]
                + 0.15 * symptoms["depressed_mood"]
            ),
            "overall_depressive_syndrome": self._clip01(
                0.22 * symptoms["depressed_mood"]
                + 0.22 * symptoms["anhedonia"]
                + 0.16 * symptoms["executive_dysfunction"]
                + 0.14 * symptoms["psychomotor_slowing"]
                + 0.14 * symptoms["fatigue_low_energy"]
                + 0.12 * symptoms["rumination_negative_bias"]
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="value").sort_index(),
            "latents": pd.Series(latents, name="value").sort_index(),
            "regional_state": pd.Series(regional_state, name="value").sort_index(),
            "symptoms": pd.Series(symptoms, name="value").sort_index(),
            "phenotypes": pd.Series(phenotypes, name="value").sort_index(),
        }


if __name__ == "__main__":
    model = DepressiveDisorderDueToAnotherMedicalConditionModel()
    scaffold = model.build()

    print("\n=== NODE TABLE (head) ===")
    print(scaffold["nodes"].head(12).to_string(index=False))

    print("\n=== EDGE TABLE (head) ===")
    print(scaffold["edges"].head(12).to_string(index=False))

    print("\n=== RESOLVED REGIONS ===")
    if scaffold["regions"]:
        for key, region in scaffold["regions"].items():
            print(f"- {key}: {getattr(region, 'name', region)}")
    else:
        print("No atlas regions resolved in this environment.")

    print("\n=== SAMPLE MULTIMODAL PROFILES ===")
    for key in ("left_dlpfc", "amygdala", "sgacc_proxy", "left_basal_ganglia_proxy"):
        receptor_df = scaffold["receptors"].get(key, pd.DataFrame())
        gene_df = scaffold["genes"].get(key, pd.DataFrame())
        conn_df = scaffold["connectivity_profiles"].get(key, pd.DataFrame())
        print(f"\n[{key}]")
        print(f"receptors: {len(receptor_df)} rows")
        print(f"genes: {len(gene_df)} rows")
        print(f"connectivity: {len(conn_df)} rows")
        if not gene_df.empty:
            print(gene_df.head(5).to_string(index=False))
        elif not conn_df.empty:
            print(conn_df.head(5).to_string(index=False))

    pairwise = scaffold["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    if isinstance(pairwise, pd.DataFrame) and not pairwise.empty:
        print(pairwise.head(10).to_string(index=False))
    else:
        print("Pairwise circuit connectivity unavailable in this environment.")

    print("\n=== SIMULATION EXAMPLE: POST-STROKE / INFLAMMATORY MIX ===")
    sim = model.simulate(
        medical_insult_burden=0.80,
        focal_lesion_burden=0.70,
        neurodegenerative_burden=0.10,
        seizure_network_burden=0.05,
        medication_depressogenic_load=0.15,
        inflammatory_burden=0.55,
        endocrine_metabolic_burden=0.20,
        genetic_vulnerability=0.35,
        psychosocial_stress_burden=0.55,
        medical_condition_correction=0.25,
        antidepressant_support=0.20,
        rehabilitation_and_social_support=0.30,
    )
    for block_name, series in sim.items():
        print(f"\n[{block_name}]")
        print(series.sort_values(ascending=False).to_string())

    # Optional heavy calls, commented out to avoid downloads during demonstration:
    # print(model.suggest_regions("dorsolateral prefrontal").head(10))
    # print(model.assign_mni_point((-38, 44, 26)).head(10))
    # mask_img = model.region_mask("left_dlpfc")
