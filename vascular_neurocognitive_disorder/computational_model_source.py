
from __future__ import annotations

"""
Vascular Neurocognitive Disorder siibra scaffold.

This script turns a chapter-level biological summary of Major or Mild Vascular
Neurocognitive Disorder (VND) into a transparent, atlas-grounded mechanistic
scaffold using siibra.

It is intended for research prototyping, feature exploration, and educational
modeling. It is not a diagnostic, prognostic, or treatment tool.
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


DEFAULT_VND_GENE_PANEL = [
    # Hereditary small-vessel disease
    "NOTCH3",
    # Mixed vascular-degenerative overlap / lipid transport
    "APOE",
    "ABCA1",
    "LDLR",
    # Blood pressure / vascular tone
    "ACE",
    "AGT",
    "NOS3",
    # Endothelial / angiogenic signaling
    "VEGFA",
    # Inflammatory signaling
    "IL6",
    "TNF",
    # Additional vascular risk susceptibility
    "MTHFR",
]


class VascularNeurocognitiveDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Major or Mild Vascular
    Neurocognitive Disorder.

    Conceptual flow
    ---------------
    inputs -> lesion / vascular biology -> regional state -> symptoms -> phenotype summaries

    Modeling choices
    ----------------
    - Uses lesion and vascular-biology latents to reflect ischemic / hemorrhagic burden.
    - Treats VND as a disconnection syndrome centered on frontal-subcortical loops.
    - Atlas-anchors only regions named or strongly implied by the chapter:
      prefrontal control, basal ganglia, thalamus, pons, and internal capsule proxy.
    - Keeps mixed Alzheimer overlap as a conservative comorbidity load rather than
      pretending the chapter specifies a full Alzheimer disease model.
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
                "siibra is required to use VascularNeurocognitiveDisorderModel. "
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
            "pfc_control": [
                "Area 46 left",
                "Area 9 left",
                "middle frontal gyrus left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "putamen left",
                "globus pallidus left",
                "basal ganglia left",
                "basal ganglia",
                "striatum",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
            ],
            "internal_capsule_proxy": [
                "posterior limb of internal capsule left",
                "internal capsule left",
                "internal capsule",
            ],
            "pons_proxy": [
                "pons left",
                "pons",
                "brainstem",
            ],
        }

        self.region_node_notes: Dict[str, str] = {
            "pfc_control": (
                "Prefrontal executive-control representative node within frontal-subcortical loops; "
                "lower simulated values indicate disconnection-related loss of top-down control."
            ),
            "basal_ganglia_proxy": (
                "Conservative basal-ganglia proxy for lacunar burden and loop disruption."
            ),
            "thalamus_proxy": (
                "Conservative thalamic relay proxy in the frontal-subcortical disconnection syndrome."
            ),
            "internal_capsule_proxy": (
                "Deep white-matter conduit proxy reflecting lacunar / WMH-related disconnection burden."
            ),
            "pons_proxy": (
                "Conservative pons / brainstem proxy reflecting lacunar and gait-related burden."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_small_vessel_liability": (
                "Heritable vulnerability to small-vessel disease, including monogenic models such as NOTCH3/CADASIL."
            ),
            "vascular_risk_burden": (
                "Aggregate hypertension, diabetes, hyperlipidemia, and related cerebrovascular risk."
            ),
            "endothelial_inflammatory_load": (
                "Endothelial dysfunction, inflammatory tone, and vascular wall stress that worsen vessel injury."
            ),
            "ischemic_lesion_burden": (
                "Burden of ischemic injury including lacunar infarcts and chronic ischemic tissue damage."
            ),
            "hemorrhagic_lesion_burden": (
                "Burden of hemorrhagic injury and microhemorrhagic pathology."
            ),
            "mixed_ad_overlap": (
                "Concurrent Alzheimer-type neurodegenerative burden contributing to mixed dementia."
            ),
            "vascular_risk_control": (
                "Protective control of blood pressure and other vascular risks that can slow progression."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "small_vessel_disease_burden": (
                "Aggregate burden of cerebral small-vessel pathology."
            ),
            "chronic_hypoperfusion": (
                "Reduced sustained perfusion that promotes deep white-matter injury."
            ),
            "blood_brain_barrier_dysfunction": (
                "Barrier disruption contributing to tissue injury, gliosis, and white-matter change."
            ),
            "lacunar_burden": (
                "Accumulation of subcortical lacunar infarcts."
            ),
            "wmh_burden": (
                "White-matter hyperintensity / leukoaraiosis burden linked to disconnection."
            ),
            "microbleed_burden": (
                "Cerebral microbleed burden as a marker of advanced small-vessel disease."
            ),
            "neurotransmitter_pathway_disruption": (
                "Lesion-driven disruption of neurotransmitter pathways and distributed signaling."
            ),
            "frontal_subcortical_disconnection": (
                "Structural and functional disconnection of prefrontal, basal-ganglia, and thalamic loops."
            ),
            "executive_network_failure": (
                "Failure of executive-control circuitry supporting planning, sequencing, and flexibility."
            ),
            "neurovascular_degenerative_synergy": (
                "Amplifying interaction between vascular injury and Alzheimer-type pathology."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "dysexecutive_syndrome": (
                "Planning, sequencing, and organizational impairment typical of subcortical VND."
            ),
            "processing_speed_slowing": (
                "Slowed cognitive processing associated with white-matter lesion burden."
            ),
            "attentional_impairment": (
                "Reduced attentional control within frontal-subcortical loops."
            ),
            "cognitive_flexibility_loss": (
                "Reduced set-shifting and adaptive executive flexibility."
            ),
            "gait_disturbance": (
                "Subcortical gait impairment associated with lacunes and deep white-matter injury."
            ),
            "global_cognitive_decline": (
                "Overall cognitive decline reflecting vascular burden and mixed pathology."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_small_vessel_liability",
                "target": "small_vessel_disease_burden",
                "relation": "raises inherited vulnerability to cerebral small-vessel pathology",
                "vnd_change": "increased",
            },
            {
                "source": "vascular_risk_burden",
                "target": "small_vessel_disease_burden",
                "relation": "loads hypertension- and metabolic-linked vessel injury",
                "vnd_change": "increased",
            },
            {
                "source": "vascular_risk_burden",
                "target": "chronic_hypoperfusion",
                "relation": "promotes sustained perfusion deficits in deep tissue",
                "vnd_change": "increased",
            },
            {
                "source": "endothelial_inflammatory_load",
                "target": "blood_brain_barrier_dysfunction",
                "relation": "worsens endothelial integrity and barrier injury",
                "vnd_change": "increased",
            },
            {
                "source": "endothelial_inflammatory_load",
                "target": "small_vessel_disease_burden",
                "relation": "contributes to vessel-wall damage and vascular susceptibility",
                "vnd_change": "increased",
            },
            {
                "source": "ischemic_lesion_burden",
                "target": "lacunar_burden",
                "relation": "adds ischemic subcortical infarct load",
                "vnd_change": "increased",
            },
            {
                "source": "ischemic_lesion_burden",
                "target": "neurotransmitter_pathway_disruption",
                "relation": "compromises neurotransmitter pathways through tissue injury",
                "vnd_change": "increased",
            },
            {
                "source": "hemorrhagic_lesion_burden",
                "target": "microbleed_burden",
                "relation": "adds microhemorrhagic burden",
                "vnd_change": "increased",
            },
            {
                "source": "hemorrhagic_lesion_burden",
                "target": "neurotransmitter_pathway_disruption",
                "relation": "disrupts signaling pathways through hemorrhagic injury",
                "vnd_change": "increased",
            },
            {
                "source": "mixed_ad_overlap",
                "target": "neurovascular_degenerative_synergy",
                "relation": "adds mixed dementia pressure through vascular-degenerative overlap",
                "vnd_change": "increased",
            },
            {
                "source": "small_vessel_disease_burden",
                "target": "lacunar_burden",
                "relation": "increases risk of deep perforator infarcts",
                "vnd_change": "increased",
            },
            {
                "source": "small_vessel_disease_burden",
                "target": "wmh_burden",
                "relation": "promotes white-matter damage and leukoaraiosis",
                "vnd_change": "increased",
            },
            {
                "source": "small_vessel_disease_burden",
                "target": "microbleed_burden",
                "relation": "marks advanced small-vessel fragility",
                "vnd_change": "increased",
            },
            {
                "source": "chronic_hypoperfusion",
                "target": "wmh_burden",
                "relation": "drives demyelination, axonal loss, and gliosis",
                "vnd_change": "increased",
            },
            {
                "source": "blood_brain_barrier_dysfunction",
                "target": "wmh_burden",
                "relation": "contributes to white-matter lesion formation",
                "vnd_change": "increased",
            },
            {
                "source": "lacunar_burden",
                "target": "basal_ganglia_proxy",
                "relation": "adds lesion burden to basal-ganglia nodes",
                "vnd_change": "increased",
            },
            {
                "source": "lacunar_burden",
                "target": "thalamus_proxy",
                "relation": "adds lesion burden to thalamic relays",
                "vnd_change": "increased",
            },
            {
                "source": "lacunar_burden",
                "target": "internal_capsule_proxy",
                "relation": "disrupts deep white-matter conduits",
                "vnd_change": "increased",
            },
            {
                "source": "lacunar_burden",
                "target": "pons_proxy",
                "relation": "adds subcortical brainstem burden",
                "vnd_change": "increased",
            },
            {
                "source": "wmh_burden",
                "target": "internal_capsule_proxy",
                "relation": "reflects white-matter disconnection through deep pathways",
                "vnd_change": "increased",
            },
            {
                "source": "wmh_burden",
                "target": "frontal_subcortical_disconnection",
                "relation": "disconnects prefrontal and subcortical loops",
                "vnd_change": "increased",
            },
            {
                "source": "microbleed_burden",
                "target": "frontal_subcortical_disconnection",
                "relation": "adds diffuse advanced SVD burden to networks",
                "vnd_change": "increased",
            },
            {
                "source": "neurotransmitter_pathway_disruption",
                "target": "executive_network_failure",
                "relation": "weakens distributed circuit signaling needed for cognition",
                "vnd_change": "increased",
            },
            {
                "source": "frontal_subcortical_disconnection",
                "target": "pfc_control",
                "relation": "reduces intact top-down executive control",
                "vnd_change": "decreased",
            },
            {
                "source": "frontal_subcortical_disconnection",
                "target": "executive_network_failure",
                "relation": "degrades loop integrity for executive control",
                "vnd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "dysexecutive_syndrome",
                "relation": "contributes to subcortical executive syndrome",
                "vnd_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "attentional_impairment",
                "relation": "disrupts relay support for attentive control",
                "vnd_change": "increased",
            },
            {
                "source": "internal_capsule_proxy",
                "target": "processing_speed_slowing",
                "relation": "disconnects fast long-range information transfer",
                "vnd_change": "increased",
            },
            {
                "source": "pons_proxy",
                "target": "gait_disturbance",
                "relation": "supports gait burden when brainstem pathways are affected",
                "vnd_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "dysexecutive_syndrome",
                "relation": "normally supports planning and sequencing",
                "vnd_change": "decreased",
            },
            {
                "source": "executive_network_failure",
                "target": "dysexecutive_syndrome",
                "relation": "drives planning and sequencing impairment",
                "vnd_change": "increased",
            },
            {
                "source": "executive_network_failure",
                "target": "cognitive_flexibility_loss",
                "relation": "reduces set-shifting and flexible control",
                "vnd_change": "increased",
            },
            {
                "source": "wmh_burden",
                "target": "processing_speed_slowing",
                "relation": "correlates with slowed processing speed",
                "vnd_change": "increased",
            },
            {
                "source": "wmh_burden",
                "target": "global_cognitive_decline",
                "relation": "adds diffuse cognitive burden",
                "vnd_change": "increased",
            },
            {
                "source": "neurovascular_degenerative_synergy",
                "target": "global_cognitive_decline",
                "relation": "amplifies decline when vascular and degenerative processes co-occur",
                "vnd_change": "increased",
            },
            {
                "source": "vascular_risk_control",
                "target": "small_vessel_disease_burden",
                "relation": "reduces ongoing vascular injury",
                "vnd_change": "decreased",
            },
            {
                "source": "vascular_risk_control",
                "target": "chronic_hypoperfusion",
                "relation": "improves risk-factor control and perfusion pressure",
                "vnd_change": "decreased",
            },
            {
                "source": "vascular_risk_control",
                "target": "microbleed_burden",
                "relation": "reduces progression pressure from uncontrolled vascular disease",
                "vnd_change": "decreased",
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

    def _mean01(self, values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return self._clip01(sum(float(v) for v in values) / len(values))

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
        generic_penalty = 1 if name in {
            "prefrontal cortex",
            "thalamus",
            "pons",
            "basal ganglia",
            "internal capsule",
        } else 0
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
        gene_panel: Sequence[str] = DEFAULT_VND_GENE_PANEL,
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

        space_candidates = [
            self.assignment_space,
            self.assignment_space_spec,
            self.space,
            self.space_spec,
        ]
        for space in space_candidates:
            try:
                return region.get_regional_mask(space)
            except Exception:
                continue
        return None

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI coordinate to Julich regions.

        Uses a statistical / probabilistic parcellation map when available.
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
        return pd.DataFrame(assignments)

    def simulate(
        self,
        genetic_small_vessel_liability: float = 0.35,
        vascular_risk_burden: float = 0.60,
        endothelial_inflammatory_load: float = 0.45,
        ischemic_lesion_burden: float = 0.50,
        hemorrhagic_lesion_burden: float = 0.20,
        mixed_ad_overlap: float = 0.30,
        vascular_risk_control: float = 0.40,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized simulation of VND biology.

        Inputs are clipped to [0, 1]. Higher values indicate greater burden,
        except vascular_risk_control which is protective.
        """
        inputs = pd.Series(
            {
                "genetic_small_vessel_liability": self._clip01(genetic_small_vessel_liability),
                "vascular_risk_burden": self._clip01(vascular_risk_burden),
                "endothelial_inflammatory_load": self._clip01(endothelial_inflammatory_load),
                "ischemic_lesion_burden": self._clip01(ischemic_lesion_burden),
                "hemorrhagic_lesion_burden": self._clip01(hemorrhagic_lesion_burden),
                "mixed_ad_overlap": self._clip01(mixed_ad_overlap),
                "vascular_risk_control": self._clip01(vascular_risk_control),
            },
            name="value",
        )

        # Latent vascular and lesion biology
        small_vessel_disease_burden = self._clip01(
            0.35 * inputs["genetic_small_vessel_liability"]
            + 0.35 * inputs["vascular_risk_burden"]
            + 0.20 * inputs["endothelial_inflammatory_load"]
            + 0.10 * inputs["ischemic_lesion_burden"]
            - 0.25 * inputs["vascular_risk_control"]
        )

        chronic_hypoperfusion = self._clip01(
            0.45 * inputs["vascular_risk_burden"]
            + 0.25 * small_vessel_disease_burden
            + 0.20 * inputs["endothelial_inflammatory_load"]
            + 0.10 * inputs["mixed_ad_overlap"]
            - 0.25 * inputs["vascular_risk_control"]
        )

        blood_brain_barrier_dysfunction = self._clip01(
            0.45 * inputs["endothelial_inflammatory_load"]
            + 0.25 * inputs["vascular_risk_burden"]
            + 0.15 * inputs["hemorrhagic_lesion_burden"]
            + 0.15 * small_vessel_disease_burden
            - 0.15 * inputs["vascular_risk_control"]
        )

        lacunar_burden = self._clip01(
            0.45 * small_vessel_disease_burden
            + 0.35 * inputs["ischemic_lesion_burden"]
            + 0.15 * inputs["genetic_small_vessel_liability"]
            - 0.15 * inputs["vascular_risk_control"]
        )

        wmh_burden = self._clip01(
            0.35 * chronic_hypoperfusion
            + 0.30 * blood_brain_barrier_dysfunction
            + 0.25 * small_vessel_disease_burden
            + 0.15 * inputs["vascular_risk_burden"]
            - 0.20 * inputs["vascular_risk_control"]
        )

        microbleed_burden = self._clip01(
            0.35 * inputs["hemorrhagic_lesion_burden"]
            + 0.25 * small_vessel_disease_burden
            + 0.20 * blood_brain_barrier_dysfunction
            + 0.10 * inputs["vascular_risk_burden"]
            + 0.10 * inputs["ischemic_lesion_burden"]
            - 0.10 * inputs["vascular_risk_control"]
        )

        neurotransmitter_pathway_disruption = self._clip01(
            0.35 * inputs["ischemic_lesion_burden"]
            + 0.25 * inputs["hemorrhagic_lesion_burden"]
            + 0.20 * lacunar_burden
            + 0.20 * wmh_burden
        )

        frontal_subcortical_disconnection = self._clip01(
            0.40 * wmh_burden
            + 0.30 * lacunar_burden
            + 0.15 * microbleed_burden
            + 0.10 * chronic_hypoperfusion
            + 0.05 * inputs["mixed_ad_overlap"]
        )

        executive_network_failure = self._clip01(
            0.45 * frontal_subcortical_disconnection
            + 0.25 * neurotransmitter_pathway_disruption
            + 0.15 * inputs["mixed_ad_overlap"]
            + 0.15 * wmh_burden
        )

        neurovascular_degenerative_synergy = self._clip01(
            0.55 * inputs["mixed_ad_overlap"]
            + 0.20 * wmh_burden
            + 0.15 * lacunar_burden
            + 0.10 * microbleed_burden
        )

        latents = pd.Series(
            {
                "small_vessel_disease_burden": small_vessel_disease_burden,
                "chronic_hypoperfusion": chronic_hypoperfusion,
                "blood_brain_barrier_dysfunction": blood_brain_barrier_dysfunction,
                "lacunar_burden": lacunar_burden,
                "wmh_burden": wmh_burden,
                "microbleed_burden": microbleed_burden,
                "neurotransmitter_pathway_disruption": neurotransmitter_pathway_disruption,
                "frontal_subcortical_disconnection": frontal_subcortical_disconnection,
                "executive_network_failure": executive_network_failure,
                "neurovascular_degenerative_synergy": neurovascular_degenerative_synergy,
            },
            name="value",
        )

        # Regional state: higher values are more burdened for subcortical proxies,
        # whereas higher pfc_control indicates better preserved executive control.
        regional_state = pd.Series(
            {
                "pfc_control": self._clip01(
                    1.0
                    - (
                        0.55 * executive_network_failure
                        + 0.25 * frontal_subcortical_disconnection
                        + 0.15 * neurovascular_degenerative_synergy
                        + 0.10 * inputs["mixed_ad_overlap"]
                    )
                ),
                "basal_ganglia_proxy": self._clip01(
                    0.55 * lacunar_burden
                    + 0.25 * small_vessel_disease_burden
                    + 0.10 * neurotransmitter_pathway_disruption
                    + 0.10 * inputs["ischemic_lesion_burden"]
                ),
                "thalamus_proxy": self._clip01(
                    0.45 * lacunar_burden
                    + 0.25 * frontal_subcortical_disconnection
                    + 0.15 * small_vessel_disease_burden
                    + 0.15 * inputs["ischemic_lesion_burden"]
                ),
                "internal_capsule_proxy": self._clip01(
                    0.45 * wmh_burden
                    + 0.30 * lacunar_burden
                    + 0.15 * chronic_hypoperfusion
                    + 0.10 * small_vessel_disease_burden
                ),
                "pons_proxy": self._clip01(
                    0.40 * lacunar_burden
                    + 0.25 * microbleed_burden
                    + 0.20 * inputs["ischemic_lesion_burden"]
                    + 0.15 * inputs["hemorrhagic_lesion_burden"]
                ),
            },
            name="value",
        )

        dysexecutive_syndrome = self._clip01(
            0.40 * executive_network_failure
            + 0.20 * (1.0 - regional_state["pfc_control"])
            + 0.15 * regional_state["basal_ganglia_proxy"]
            + 0.15 * regional_state["thalamus_proxy"]
            + 0.10 * wmh_burden
        )

        processing_speed_slowing = self._clip01(
            0.45 * wmh_burden
            + 0.20 * regional_state["internal_capsule_proxy"]
            + 0.15 * frontal_subcortical_disconnection
            + 0.10 * microbleed_burden
            + 0.10 * regional_state["pons_proxy"]
        )

        attentional_impairment = self._clip01(
            0.40 * executive_network_failure
            + 0.20 * regional_state["thalamus_proxy"]
            + 0.20 * (1.0 - regional_state["pfc_control"])
            + 0.10 * wmh_burden
            + 0.10 * inputs["mixed_ad_overlap"]
        )

        cognitive_flexibility_loss = self._clip01(
            0.45 * executive_network_failure
            + 0.20 * regional_state["basal_ganglia_proxy"]
            + 0.15 * regional_state["thalamus_proxy"]
            + 0.10 * (1.0 - regional_state["pfc_control"])
            + 0.10 * wmh_burden
        )

        gait_disturbance = self._clip01(
            0.35 * lacunar_burden
            + 0.25 * regional_state["pons_proxy"]
            + 0.20 * regional_state["internal_capsule_proxy"]
            + 0.10 * regional_state["basal_ganglia_proxy"]
            + 0.10 * wmh_burden
        )

        global_cognitive_decline = self._clip01(
            0.35 * neurovascular_degenerative_synergy
            + 0.20 * dysexecutive_syndrome
            + 0.15 * processing_speed_slowing
            + 0.10 * attentional_impairment
            + 0.10 * microbleed_burden
            + 0.10 * inputs["mixed_ad_overlap"]
        )

        symptoms = pd.Series(
            {
                "dysexecutive_syndrome": dysexecutive_syndrome,
                "processing_speed_slowing": processing_speed_slowing,
                "attentional_impairment": attentional_impairment,
                "cognitive_flexibility_loss": cognitive_flexibility_loss,
                "gait_disturbance": gait_disturbance,
                "global_cognitive_decline": global_cognitive_decline,
            },
            name="value",
        )

        phenotypes = pd.Series(
            {
                "mild_vnd_expression": self._mean01(
                    [
                        dysexecutive_syndrome,
                        processing_speed_slowing,
                        attentional_impairment,
                    ]
                ),
                "major_vnd_expression": self._mean01(
                    [
                        global_cognitive_decline,
                        dysexecutive_syndrome,
                        gait_disturbance,
                        neurovascular_degenerative_synergy,
                    ]
                ),
                "mixed_dementia_profile": self._mean01(
                    [
                        global_cognitive_decline,
                        neurovascular_degenerative_synergy,
                        inputs["mixed_ad_overlap"],
                    ]
                ),
                "vascular_progression_pressure": self._mean01(
                    [
                        small_vessel_disease_burden,
                        wmh_burden,
                        microbleed_burden,
                        inputs["vascular_risk_burden"],
                    ]
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
        model = VascularNeurocognitiveDisorderModel()
    except ImportError as exc:
        print(exc)
        raise SystemExit(1)

    print("Building Vascular Neurocognitive Disorder scaffold...\n")
    bundle = model.build(connectivity_rows=10)

    print("NODES")
    print(bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\nEDGES")
    print(bundle["edges"][["source", "target", "relation", "vnd_change"]].to_string(index=False))

    if "pfc_control" in bundle["genes"] and not bundle["genes"]["pfc_control"].empty:
        print("\nPFC CONTROL GENE SUMMARY (head)")
        print(bundle["genes"]["pfc_control"].head().to_string(index=False))

    if "basal_ganglia_proxy" in bundle["connectivity_profiles"] and not bundle["connectivity_profiles"]["basal_ganglia_proxy"].empty:
        print("\nBASAL GANGLIA CONNECTIVITY PROFILE (head)")
        print(bundle["connectivity_profiles"]["basal_ganglia_proxy"].head().to_string(index=False))

    if not bundle["circuit_connectivity"].empty:
        print("\nCIRCUIT CONNECTIVITY")
        print(bundle["circuit_connectivity"].head(20).to_string(index=False))

    print("\nSIMULATION EXAMPLE")
    sim = model.simulate(
        genetic_small_vessel_liability=0.55,
        vascular_risk_burden=0.75,
        endothelial_inflammatory_load=0.60,
        ischemic_lesion_burden=0.65,
        hemorrhagic_lesion_burden=0.25,
        mixed_ad_overlap=0.35,
        vascular_risk_control=0.30,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-12, -16, 8)).head())
