from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Female Orgasmic Disorder.

This script turns a chapter-level biological summary into a transparent mechanistic
research scaffold. It is intended for hypothesis generation, teaching, and iterative
refinement against atlas-backed evidence. It is not a diagnostic or treatment tool.

The chapter indicates that Female Orgasmic Disorder (FOD) emerges from a complex,
multifactorial interaction among biological, psychological, and sociocultural factors,
but emphasizes several neurobiological themes that can be encoded mechanistically:

- serotonergic inhibition of orgasm, especially under SSRI or other serotonergic load,
- limbic and temporal-lobe dysregulation, particularly in epilepsy,
- hypothalamic / neuroendocrine disruption affecting sexual response,
- reward-circuit under-engagement and pleasure blunting,
- excessive inhibitory control and affective comorbidity burden.

Important modeling note:
Some chapter claims are directly evidenced in the text (for example, SSRI-linked
anorgasmia and temporal-lobe epilepsy associations), while others are explicitly
presented as plausible neuroimaging interpretations (for example, nucleus accumbens
hypoactivation or prefrontal over-inhibition). Those latter mechanisms are preserved
here, but implemented conservatively as proxies.

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
    "MAOA",    # monoamine metabolism
    "DRD2",    # dopamine receptor
    "SLC6A3",  # dopamine transporter
    "COMT",    # catecholamine metabolism
    "OPRM1",   # opioid reward / pleasure signaling
    "BDNF",    # plasticity and affective vulnerability
    "ESR1",    # estrogen signaling
    "ESR2",    # estrogen signaling
    "OXTR",    # oxytocin-related affiliative / sexual response biology
]


class FemaleOrgasmicDisorderModel:
    """
    Mechanistic siibra scaffold for Female Orgasmic Disorder.

    Main chapter-derived logic:
    1) serotonergic overdrive raises orgasmic inhibition and threshold,
    2) temporal-limbic dysfunction can disrupt pleasure processing and hypothalamic regulation,
    3) affective and genetic liabilities sensitize inhibitory and reward mechanisms,
    4) orgasmic failure emerges through a one-pass cascade:
       inputs -> latent biology -> regional dysfunction burden -> symptoms -> phenotypes.

    Regional-state values in this simulator represent *dysregulation burden*
    rather than raw neural firing. Higher values mean greater regional/pathway
    dysfunction relevant to the disorder.

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

        # Direct anchors are used where the chapter names structures explicitly.
        # Proxies are used where the chapter is systems-level or where Julich labels
        # may vary across environments.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "cingulate_gyrus_proxy": [
                "Area s24 left",
                "Area p32 left",
                "Area 24 left",
                "cingulate gyrus left",
                "cingulate cortex left",
            ],
            "right_temporal_limbic_proxy": [
                "entorhinal cortex right",
                "parahippocampal cortex right",
                "temporal pole right",
                "temporal lobe right",
            ],
            "hypothalamus_proxy": [
                "hypothalamus",
                "hypothalamic region",
            ],
            "nucleus_accumbens_proxy": [
                "nucleus accumbens left",
                "ventral striatum left",
                "accumbens left",
                "striatum left",
            ],
            "pfc_inhibitory_control_proxy": [
                "Area 10 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "prefrontal cortex left",
                "middle frontal gyrus left",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Amygdala anchor for limbic salience, affective arousal, and emotional gating "
                "of sexual response."
            ),
            "hippocampus": (
                "Hippocampal anchor for contextual memory, stress integration, and limbic-temporal "
                "network participation."
            ),
            "cingulate_gyrus_proxy": (
                "Cingulate proxy for motivational-affective integration and monitoring of internal "
                "state during sexual response."
            ),
            "right_temporal_limbic_proxy": (
                "Right temporal-limbic proxy reflecting the chapter's specific link between orgasmic "
                "physiology and right temporal epileptogenic burden."
            ),
            "hypothalamus_proxy": (
                "Hypothalamic proxy for neuroendocrine gating of sexual arousal and orgasmic response."
            ),
            "nucleus_accumbens_proxy": (
                "Reward-circuit proxy for orgasmic pleasure, motivation, and incentive engagement; "
                "kept conservative because the chapter frames this as plausible rather than directly imaged in FOD."
            ),
            "pfc_inhibitory_control_proxy": (
                "Prefrontal inhibitory-control proxy for excessive top-down suppression or performance-focused "
                "control that may inhibit orgasmic release."
            ),
        }
        self.proxy_region_keys = {
            "cingulate_gyrus_proxy",
            "right_temporal_limbic_proxy",
            "hypothalamus_proxy",
            "nucleus_accumbens_proxy",
            "pfc_inhibitory_control_proxy",
        }

        self.input_nodes: Dict[str, str] = {
            "serotonergic_medication_load": (
                "Burden of SSRI or other serotonergic medication exposure linked to orgasmic inhibition."
            ),
            "dose_related_serotonergic_pressure": (
                "Dose-sensitive serotonergic pressure increasing orgasm delay or anorgasmia risk."
            ),
            "temporal_lobe_epilepsy_burden": (
                "Epileptiform instability in temporal-limbic circuits associated with orgasmic dysfunction."
            ),
            "limbic_brain_injury_burden": (
                "Structural injury burden affecting limbic or temporal networks involved in sexual response."
            ),
            "neuroendocrine_instability": (
                "Hormonal or hypothalamic dysregulation that perturbs orgasmic physiology."
            ),
            "affective_comorbidity_load": (
                "Current depressive or anxiety burden that can suppress pleasure and orgasmic capacity."
            ),
            "genetic_vulnerability": (
                "Inherited liability to affective, anxiety, or reward-regulation dyscontrol."
            ),
            "psychosocial_inhibitory_context": (
                "Performance anxiety, trauma-related inhibition, relational strain, or restrictive conditioning."
            ),
            "medication_adjustment_support": (
                "Protective reduction in serotonergic side-effect burden through medication review or adjustment."
            ),
            "neurological_stabilization": (
                "Protective stabilization of seizures or neurological burden."
            ),
            "sexual_health_support": (
                "Protective psychoeducation, therapy, communication, and supportive sexual context."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_orgasmic_inhibition": (
                "Serotonin-linked inhibitory pressure that raises orgasmic threshold and delays climax."
            ),
            "limbic_temporal_dysregulation": (
                "Disordered temporal-limbic coordination affecting arousal, pleasure, and affective gating."
            ),
            "hypothalamic_neuroendocrine_disruption": (
                "Disturbed neuroendocrine regulation of sexual response and orgasmic physiology."
            ),
            "reward_circuit_hypoactivation": (
                "Blunted motivational and hedonic recruitment in reward circuits during erotic stimulation."
            ),
            "prefrontal_inhibitory_overcontrol": (
                "Excessive top-down inhibition, self-monitoring, or cognitive control that suppresses orgasm."
            ),
            "affective_comorbidity_sensitization": (
                "Mood- and anxiety-linked sensitization that amplifies inhibitory and pleasure-blunting processes."
            ),
            "orgasmic_threshold_elevation": (
                "Integrated latent barrier making orgasm harder to reach or sustain."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "delayed_orgasm": "Marked delay in reaching orgasm.",
            "anorgasmia": "Failure to achieve orgasm despite adequate stimulation or desire.",
            "reduced_orgasmic_intensity": "Diminished orgasmic pleasure or weakened climax.",
            "erotic_pleasure_blunting": "Reduced pleasure or reward value from erotic stimulation.",
            "sexual_distress_and_avoidance": "Distress, frustration, or avoidance secondary to orgasmic difficulty.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "serotonergic_medication_load",
                "target": "serotonergic_orgasmic_inhibition",
                "relation": "serotonergic medications can delay or inhibit orgasm",
                "fod_change": "increased",
            },
            {
                "source": "dose_related_serotonergic_pressure",
                "target": "serotonergic_orgasmic_inhibition",
                "relation": "greater serotonergic pressure can produce dose-related orgasmic dysfunction",
                "fod_change": "increased",
            },
            {
                "source": "temporal_lobe_epilepsy_burden",
                "target": "limbic_temporal_dysregulation",
                "relation": "temporal-limbic epileptiform activity disrupts sexual arousal and pleasure processing",
                "fod_change": "increased",
            },
            {
                "source": "limbic_brain_injury_burden",
                "target": "limbic_temporal_dysregulation",
                "relation": "limbic injury can alter sexual behavior and orgasmic processing",
                "fod_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "hypothalamic_neuroendocrine_disruption",
                "relation": "temporal-limbic dysfunction can disturb hypothalamic hormonal regulation",
                "fod_change": "increased",
            },
            {
                "source": "neuroendocrine_instability",
                "target": "hypothalamic_neuroendocrine_disruption",
                "relation": "endocrine instability burdens hypothalamic regulation of sexual response",
                "fod_change": "increased",
            },
            {
                "source": "affective_comorbidity_load",
                "target": "affective_comorbidity_sensitization",
                "relation": "depression and anxiety intensify orgasmic inhibition and pleasure blunting",
                "fod_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "affective_comorbidity_sensitization",
                "relation": "heritable affective vulnerability can indirectly increase FOD susceptibility",
                "fod_change": "increased",
            },
            {
                "source": "psychosocial_inhibitory_context",
                "target": "prefrontal_inhibitory_overcontrol",
                "relation": "performance anxiety and inhibitory context can heighten top-down suppression",
                "fod_change": "increased",
            },
            {
                "source": "serotonergic_orgasmic_inhibition",
                "target": "reward_circuit_hypoactivation",
                "relation": "serotonergic inhibition can blunt reward-linked orgasmic engagement",
                "fod_change": "increased",
            },
            {
                "source": "affective_comorbidity_sensitization",
                "target": "reward_circuit_hypoactivation",
                "relation": "affective burden reduces hedonic and motivational recruitment",
                "fod_change": "increased",
            },
            {
                "source": "affective_comorbidity_sensitization",
                "target": "prefrontal_inhibitory_overcontrol",
                "relation": "negative affect can amplify self-monitoring and inhibition",
                "fod_change": "increased",
            },
            {
                "source": "serotonergic_orgasmic_inhibition",
                "target": "orgasmic_threshold_elevation",
                "relation": "serotonergic inhibition raises the threshold for orgasmic release",
                "fod_change": "increased",
            },
            {
                "source": "reward_circuit_hypoactivation",
                "target": "orgasmic_threshold_elevation",
                "relation": "weaker reward engagement makes orgasm harder to achieve",
                "fod_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_overcontrol",
                "target": "orgasmic_threshold_elevation",
                "relation": "excessive cognitive control suppresses orgasmic release",
                "fod_change": "increased",
            },
            {
                "source": "hypothalamic_neuroendocrine_disruption",
                "target": "orgasmic_threshold_elevation",
                "relation": "neuroendocrine dysregulation impairs orgasmic physiology",
                "fod_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "right_temporal_limbic_proxy",
                "relation": "temporal-limbic dysfunction concentrates burden in right temporal orgasm-related circuitry",
                "fod_change": "increased",
            },
            {
                "source": "hypothalamic_neuroendocrine_disruption",
                "target": "hypothalamus_proxy",
                "relation": "latent neuroendocrine dysregulation burdens hypothalamic sexual-response gating",
                "fod_change": "increased",
            },
            {
                "source": "reward_circuit_hypoactivation",
                "target": "nucleus_accumbens_proxy",
                "relation": "reward-circuit under-engagement reduces orgasmic pleasure recruitment",
                "fod_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_overcontrol",
                "target": "pfc_inhibitory_control_proxy",
                "relation": "top-down inhibitory burden localizes to prefrontal control systems",
                "fod_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "amygdala",
                "relation": "limbic dysregulation heightens emotionally loaded gating of sexual response",
                "fod_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "hippocampus",
                "relation": "temporal-limbic disturbance affects contextual and memory-linked modulation of response",
                "fod_change": "increased",
            },
            {
                "source": "affective_comorbidity_sensitization",
                "target": "cingulate_gyrus_proxy",
                "relation": "affective burden dysregulates cingulate monitoring and motivational-affective integration",
                "fod_change": "increased",
            },
            {
                "source": "orgasmic_threshold_elevation",
                "target": "delayed_orgasm",
                "relation": "higher orgasmic threshold prolongs time to climax",
                "fod_change": "increased",
            },
            {
                "source": "orgasmic_threshold_elevation",
                "target": "anorgasmia",
                "relation": "severe threshold elevation can prevent orgasm entirely",
                "fod_change": "increased",
            },
            {
                "source": "nucleus_accumbens_proxy",
                "target": "reduced_orgasmic_intensity",
                "relation": "reward-circuit dysfunction weakens orgasmic pleasure intensity",
                "fod_change": "increased",
            },
            {
                "source": "reward_circuit_hypoactivation",
                "target": "erotic_pleasure_blunting",
                "relation": "reduced reward recruitment blunts erotic pleasure",
                "fod_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "sexual_distress_and_avoidance",
                "relation": "emotionally loaded limbic dysregulation contributes to distress and avoidance",
                "fod_change": "increased",
            },
            {
                "source": "medication_adjustment_support",
                "target": "serotonergic_orgasmic_inhibition",
                "relation": "reducing serotonergic side-effect burden can lower orgasmic inhibition",
                "fod_change": "decreased",
            },
            {
                "source": "neurological_stabilization",
                "target": "limbic_temporal_dysregulation",
                "relation": "better neurological control can reduce temporal-limbic instability",
                "fod_change": "decreased",
            },
            {
                "source": "sexual_health_support",
                "target": "prefrontal_inhibitory_overcontrol",
                "relation": "supportive therapy and communication can reduce performance-focused inhibition",
                "fod_change": "decreased",
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
        right_bonus = 0 if "right" in name else 1
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "striatum"} else 0
        cyto_bonus = 0 if "area " in name or "(" in name else 1
        return (left_bonus, right_bonus, generic_penalty, cyto_bonus)

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
        volume_mm3 = float(getattr(main, "volume", float("nan")))
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
        try:
            df = feats[0].data.copy()
        except Exception:
            return pd.DataFrame()
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
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
        if exact:
            return exact[0]
        rn = getattr(region, "name", "").lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

    @staticmethod
    def _connectivity_series(selection: Any, axis: str) -> pd.Series:
        """
        Normalize a connectivity slice to a 1D numeric series.

        Some siibra connectivity matrices expose duplicate labels or multi-level axes,
        so selecting a single label can return a DataFrame instead of a Series.
        When that happens, average across the duplicated axis to obtain one profile.
        """
        if isinstance(selection, pd.Series):
            return pd.to_numeric(selection, errors="coerce").dropna()
        if isinstance(selection, pd.DataFrame):
            numeric = selection.apply(pd.to_numeric, errors="coerce")
            reduced = numeric.mean(axis=0 if axis == "index" else 1, skipna=True)
            return pd.to_numeric(reduced, errors="coerce").dropna()
        return pd.Series(dtype=float)

    @staticmethod
    def _connectivity_scalar(selection: Any) -> Optional[float]:
        """
        Normalize a connectivity cell lookup to one numeric scalar.

        Duplicate row/column labels can make a pandas lookup return a Series or
        DataFrame. We reduce those cases by averaging the numeric values.
        """
        if selection is None:
            return None
        if isinstance(selection, pd.DataFrame):
            flat = pd.Series(selection.to_numpy().ravel())
            numeric = pd.to_numeric(flat, errors="coerce").dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(selection, pd.Series):
            numeric = pd.to_numeric(selection, errors="coerce").dropna()
            return float(numeric.mean()) if not numeric.empty else None
        try:
            return float(selection)
        except Exception:
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
            selection = matrix.loc[label] if axis == "index" else matrix[label]
            series = self._connectivity_series(selection, axis=axis)
            if not series.empty:
                df = series.sort_values(ascending=False).reset_index()
                df.columns = ["connected_region", "value"]
                df["connected_region"] = df["connected_region"].map(self._name_of)
                df = (
                    df.groupby("connected_region", as_index=False, dropna=False)["value"]
                    .mean()
                    .sort_values("value", ascending=False)
                )
                df = df[df["connected_region"] != getattr(region, "name", None)].head(max_rows)
                return df.reset_index(drop=True)
        except Exception:
            pass

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
                scalar = self._connectivity_scalar(value)
                if scalar is not None:
                    rows.append({"source": source, "target": target, "value": scalar})

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

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
        Return a region mask or map representation when available.
        """
        if not self.siibra_available:
            return None
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        try:
            if hasattr(region, "get_regional_mask"):
                mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
                try:
                    return mask.fetch()
                except Exception:
                    return mask
        except Exception:
            pass

        try:
            if hasattr(region, "get_regional_map"):
                regional_map = region.get_regional_map(self.assignment_space, "statistical")
                try:
                    return regional_map.fetch()
                except Exception:
                    return regional_map
        except Exception:
            pass

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
        serotonergic_medication_load: float = 0.40,
        dose_related_serotonergic_pressure: float = 0.30,
        temporal_lobe_epilepsy_burden: float = 0.00,
        limbic_brain_injury_burden: float = 0.00,
        neuroendocrine_instability: float = 0.25,
        affective_comorbidity_load: float = 0.30,
        genetic_vulnerability: float = 0.25,
        psychosocial_inhibitory_context: float = 0.30,
        medication_adjustment_support: float = 0.20,
        neurological_stabilization: float = 0.20,
        sexual_health_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Inputs are clipped to [0, 1]. Protective variables subtract from pathological
        latent states and downstream symptoms.
        """
        inputs = {
            "serotonergic_medication_load": self._clip01(serotonergic_medication_load),
            "dose_related_serotonergic_pressure": self._clip01(dose_related_serotonergic_pressure),
            "temporal_lobe_epilepsy_burden": self._clip01(temporal_lobe_epilepsy_burden),
            "limbic_brain_injury_burden": self._clip01(limbic_brain_injury_burden),
            "neuroendocrine_instability": self._clip01(neuroendocrine_instability),
            "affective_comorbidity_load": self._clip01(affective_comorbidity_load),
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "psychosocial_inhibitory_context": self._clip01(psychosocial_inhibitory_context),
            "medication_adjustment_support": self._clip01(medication_adjustment_support),
            "neurological_stabilization": self._clip01(neurological_stabilization),
            "sexual_health_support": self._clip01(sexual_health_support),
        }

        support_mean = self._clip01(
            (
                inputs["medication_adjustment_support"]
                + inputs["neurological_stabilization"]
                + inputs["sexual_health_support"]
            )
            / 3.0
        )

        latents: Dict[str, float] = {}
        latents["serotonergic_orgasmic_inhibition"] = self._clip01(
            0.55 * inputs["serotonergic_medication_load"]
            + 0.20 * inputs["dose_related_serotonergic_pressure"]
            + 0.10 * inputs["affective_comorbidity_load"]
            + 0.10 * inputs["genetic_vulnerability"]
            + 0.05 * inputs["psychosocial_inhibitory_context"]
            - 0.30 * inputs["medication_adjustment_support"]
        )
        latents["limbic_temporal_dysregulation"] = self._clip01(
            0.45 * inputs["temporal_lobe_epilepsy_burden"]
            + 0.30 * inputs["limbic_brain_injury_burden"]
            + 0.15 * inputs["affective_comorbidity_load"]
            + 0.15 * inputs["psychosocial_inhibitory_context"]
            - 0.25 * inputs["neurological_stabilization"]
        )
        latents["hypothalamic_neuroendocrine_disruption"] = self._clip01(
            0.40 * inputs["neuroendocrine_instability"]
            + 0.25 * latents["limbic_temporal_dysregulation"]
            + 0.15 * inputs["affective_comorbidity_load"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.20 * inputs["neurological_stabilization"]
        )
        latents["affective_comorbidity_sensitization"] = self._clip01(
            0.35 * inputs["affective_comorbidity_load"]
            + 0.25 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["psychosocial_inhibitory_context"]
            + 0.10 * latents["serotonergic_orgasmic_inhibition"]
            - 0.15 * inputs["sexual_health_support"]
        )
        latents["reward_circuit_hypoactivation"] = self._clip01(
            0.35 * latents["serotonergic_orgasmic_inhibition"]
            + 0.25 * latents["affective_comorbidity_sensitization"]
            + 0.20 * latents["hypothalamic_neuroendocrine_disruption"]
            + 0.15 * latents["limbic_temporal_dysregulation"]
            - 0.20 * inputs["sexual_health_support"]
        )
        latents["prefrontal_inhibitory_overcontrol"] = self._clip01(
            0.30 * inputs["psychosocial_inhibitory_context"]
            + 0.25 * latents["affective_comorbidity_sensitization"]
            + 0.20 * latents["serotonergic_orgasmic_inhibition"]
            + 0.15 * inputs["genetic_vulnerability"]
            - 0.15 * inputs["sexual_health_support"]
        )
        latents["orgasmic_threshold_elevation"] = self._clip01(
            0.35 * latents["serotonergic_orgasmic_inhibition"]
            + 0.25 * latents["reward_circuit_hypoactivation"]
            + 0.20 * latents["prefrontal_inhibitory_overcontrol"]
            + 0.20 * latents["hypothalamic_neuroendocrine_disruption"]
            + 0.10 * latents["limbic_temporal_dysregulation"]
            - 0.20 * support_mean
        )

        regional_state = {
            "amygdala": self._clip01(
                0.45 * latents["limbic_temporal_dysregulation"]
                + 0.25 * latents["affective_comorbidity_sensitization"]
                + 0.20 * inputs["psychosocial_inhibitory_context"]
                + 0.10 * latents["orgasmic_threshold_elevation"]
                - 0.10 * inputs["sexual_health_support"]
            ),
            "hippocampus": self._clip01(
                0.35 * latents["limbic_temporal_dysregulation"]
                + 0.25 * latents["affective_comorbidity_sensitization"]
                + 0.20 * inputs["psychosocial_inhibitory_context"]
                + 0.10 * inputs["limbic_brain_injury_burden"]
                - 0.10 * inputs["neurological_stabilization"]
            ),
            "cingulate_gyrus_proxy": self._clip01(
                0.30 * latents["affective_comorbidity_sensitization"]
                + 0.25 * latents["prefrontal_inhibitory_overcontrol"]
                + 0.20 * latents["orgasmic_threshold_elevation"]
                + 0.15 * latents["limbic_temporal_dysregulation"]
                - 0.10 * inputs["sexual_health_support"]
            ),
            "right_temporal_limbic_proxy": self._clip01(
                0.50 * inputs["temporal_lobe_epilepsy_burden"]
                + 0.25 * inputs["limbic_brain_injury_burden"]
                + 0.20 * latents["limbic_temporal_dysregulation"]
                - 0.20 * inputs["neurological_stabilization"]
            ),
            "hypothalamus_proxy": self._clip01(
                0.50 * latents["hypothalamic_neuroendocrine_disruption"]
                + 0.20 * latents["limbic_temporal_dysregulation"]
                + 0.10 * inputs["neuroendocrine_instability"]
                - 0.15 * inputs["neurological_stabilization"]
            ),
            "nucleus_accumbens_proxy": self._clip01(
                0.45 * latents["reward_circuit_hypoactivation"]
                + 0.25 * latents["serotonergic_orgasmic_inhibition"]
                + 0.15 * latents["affective_comorbidity_sensitization"]
                + 0.10 * latents["orgasmic_threshold_elevation"]
                - 0.15 * inputs["sexual_health_support"]
            ),
            "pfc_inhibitory_control_proxy": self._clip01(
                0.45 * latents["prefrontal_inhibitory_overcontrol"]
                + 0.20 * inputs["psychosocial_inhibitory_context"]
                + 0.15 * latents["affective_comorbidity_sensitization"]
                + 0.10 * latents["orgasmic_threshold_elevation"]
                - 0.10 * inputs["sexual_health_support"]
            ),
        }

        symptoms: Dict[str, float] = {}
        symptoms["delayed_orgasm"] = self._clip01(
            0.40 * latents["orgasmic_threshold_elevation"]
            + 0.20 * latents["serotonergic_orgasmic_inhibition"]
            + 0.15 * regional_state["pfc_inhibitory_control_proxy"]
            + 0.10 * regional_state["hypothalamus_proxy"]
            + 0.10 * regional_state["nucleus_accumbens_proxy"]
            - 0.15 * inputs["medication_adjustment_support"]
        )
        symptoms["anorgasmia"] = self._clip01(
            0.35 * latents["orgasmic_threshold_elevation"]
            + 0.20 * regional_state["nucleus_accumbens_proxy"]
            + 0.15 * regional_state["right_temporal_limbic_proxy"]
            + 0.15 * regional_state["hypothalamus_proxy"]
            + 0.10 * regional_state["pfc_inhibitory_control_proxy"]
            + 0.10 * latents["serotonergic_orgasmic_inhibition"]
            - 0.15 * support_mean
        )
        symptoms["reduced_orgasmic_intensity"] = self._clip01(
            0.35 * regional_state["nucleus_accumbens_proxy"]
            + 0.25 * latents["reward_circuit_hypoactivation"]
            + 0.15 * latents["serotonergic_orgasmic_inhibition"]
            + 0.10 * regional_state["hypothalamus_proxy"]
            - 0.15 * inputs["sexual_health_support"]
        )
        symptoms["erotic_pleasure_blunting"] = self._clip01(
            0.40 * latents["reward_circuit_hypoactivation"]
            + 0.20 * regional_state["nucleus_accumbens_proxy"]
            + 0.15 * latents["affective_comorbidity_sensitization"]
            + 0.10 * regional_state["amygdala"]
            - 0.15 * inputs["sexual_health_support"]
        )
        symptoms["sexual_distress_and_avoidance"] = self._clip01(
            0.30 * symptoms["anorgasmia"]
            + 0.20 * symptoms["delayed_orgasm"]
            + 0.20 * regional_state["amygdala"]
            + 0.15 * regional_state["pfc_inhibitory_control_proxy"]
            + 0.15 * latents["affective_comorbidity_sensitization"]
            - 0.15 * inputs["sexual_health_support"]
        )

        phenotypes = {
            "serotonergic_inhibition_profile": self._clip01(
                0.35 * symptoms["delayed_orgasm"]
                + 0.25 * symptoms["reduced_orgasmic_intensity"]
                + 0.25 * latents["serotonergic_orgasmic_inhibition"]
                + 0.15 * latents["orgasmic_threshold_elevation"]
            ),
            "temporal_limbic_neurological_profile": self._clip01(
                0.30 * symptoms["anorgasmia"]
                + 0.25 * regional_state["right_temporal_limbic_proxy"]
                + 0.20 * regional_state["hypothalamus_proxy"]
                + 0.25 * latents["limbic_temporal_dysregulation"]
            ),
            "affective_inhibitory_profile": self._clip01(
                0.30 * symptoms["sexual_distress_and_avoidance"]
                + 0.20 * regional_state["amygdala"]
                + 0.20 * regional_state["pfc_inhibitory_control_proxy"]
                + 0.30 * latents["affective_comorbidity_sensitization"]
            ),
            "global_female_orgasmic_disorder_profile": self._clip01(
                (
                    symptoms["delayed_orgasm"]
                    + symptoms["anorgasmia"]
                    + symptoms["reduced_orgasmic_intensity"]
                    + symptoms["erotic_pleasure_blunting"]
                    + symptoms["sexual_distress_and_avoidance"]
                )
                / 5.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="value"),
            "latents": pd.Series(latents, name="value"),
            "regional_state": pd.Series(regional_state, name="value"),
            "symptoms": pd.Series(symptoms, name="value"),
            "phenotypes": pd.Series(phenotypes, name="value"),
        }


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)

    model = FemaleOrgasmicDisorderModel()
    scaffold = model.build()

    print("\n=== Nodes ===")
    print(scaffold["nodes"].head(25).to_string(index=False))

    print("\n=== Edges ===")
    print(scaffold["edges"].head(20).to_string(index=False))

    print("\n=== Resolved regions ===")
    resolved = sorted(scaffold["regions"].keys())
    print(resolved if resolved else "No regions resolved in this environment.")

    for key in ["amygdala", "nucleus_accumbens_proxy", "pfc_inhibitory_control_proxy"]:
        receptor_df = scaffold["receptors"].get(key, pd.DataFrame())
        gene_df = scaffold["genes"].get(key, pd.DataFrame())
        conn_df = scaffold["connectivity_profiles"].get(key, pd.DataFrame())

        print(f"\n=== {key}: receptor summary ===")
        print(receptor_df.head(10).to_string(index=False) if not receptor_df.empty else "No receptor fingerprint available.")

        print(f"\n=== {key}: gene summary ===")
        print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "No gene-expression summary available.")

        print(f"\n=== {key}: connectivity profile ===")
        print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "No connectivity profile available.")

    print("\n=== Pairwise circuit connectivity ===")
    pairwise = scaffold["circuit_connectivity"]
    print(pairwise.head(20).to_string(index=False) if not pairwise.empty else "No circuit connectivity available.")

    sim = model.simulate(
        serotonergic_medication_load=0.85,
        dose_related_serotonergic_pressure=0.70,
        temporal_lobe_epilepsy_burden=0.25,
        limbic_brain_injury_burden=0.05,
        neuroendocrine_instability=0.35,
        affective_comorbidity_load=0.40,
        genetic_vulnerability=0.30,
        psychosocial_inhibitory_context=0.35,
        medication_adjustment_support=0.20,
        neurological_stabilization=0.55,
        sexual_health_support=0.40,
    )

    print("\n=== Simulation: inputs ===")
    print(sim["inputs"].to_string())

    print("\n=== Simulation: latents ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== Simulation: regional dysfunction burden ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())

    print("\n=== Simulation: symptoms ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== Simulation: phenotypes ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Optional coordinate examples for later exploration:
    # print(model.assign_mni_point((24, -8, -22)).head(10))  # right medial temporal neighborhood
    # mask_img = model.region_mask("amygdala")
