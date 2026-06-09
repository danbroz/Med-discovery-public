from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc
else:
    _SIIBRA_IMPORT_ERROR = None


NARCOLEPSY_GENE_PANEL = [
    "HLA-DQB1",
    "HCRT",
    "HCRTR1",
    "HCRTR2",
    "GABBR1",
    "GABBR2",
]


class NarcolepsyModel:
    """
    Atlas-grounded siibra scaffold for narcolepsy.

    This script is a research scaffold, not a diagnostic, prognostic, or treatment
    tool. It translates a chapter-level description of narcolepsy into a transparent
    mechanistic model using conservative atlas anchoring.

    Chapter-driven design choices:
    - The primary biological center is kept as a hypothalamus proxy because the
      chapter localizes the key pathology to hypocretin-producing neurons in the
      hypothalamus, while exact Julich parcel support may vary across environments.
    - Brainstem / pontine involvement is modeled as a proxy node because the chapter
      discusses pontine and brainstem lesions in symptomatic narcolepsy-like states.
    - Thalamic and frontal cortical changes are included conservatively as proxy nodes
      because the chapter mentions subtle gray-matter changes in these regions.
    - Hippocampus is retained as an explicit anchor for secondary inflammatory /
      paraneoplastic cases involving mesial temporal and hippocampal structures.
    - Chemistry is kept mostly latent. GABAergic sleep consolidation and hypocretin
      deficiency are modeled as mechanistic processes rather than forced into overly
      precise parcels.

    Higher values returned by ``simulate()`` indicate greater dysfunction burden or
    symptom burden, not greater healthy activation.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:  # pragma: no cover - environment dependent
            raise ImportError(
                "siibra is required to use NarcolepsyModel. "
                "Install siibra-python in your runtime before building the scaffold."
            ) from _SIIBRA_IMPORT_ERROR

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

        self.region_candidates: Dict[str, List[str]] = {
            "hypothalamus_proxy": [
                "Hypothalamus left",
                "Hypothalamus",
                "hypothalamus",
            ],
            "thalamus_proxy": [
                "Thalamus left",
                "Thalamus",
                "thalamus",
            ],
            "frontal_cortex_proxy": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area IFJ left",
                "middle frontal gyrus",
                "frontal cortex",
                "prefrontal cortex",
            ],
            "brainstem_pontine_proxy": [
                "Pons left",
                "Pons",
                "pontine tegmentum",
                "brainstem",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "CA3 left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "hypothalamus_proxy": (
                "Proxy for hypothalamic hypocretin-producing systems central to narcolepsy biology"
            ),
            "thalamus_proxy": (
                "Proxy for thalamic participation in distributed sleep-wake state regulation"
            ),
            "frontal_cortex_proxy": (
                "Proxy for frontal cortical burden associated with chronic sleep state instability and daytime impairment"
            ),
            "brainstem_pontine_proxy": (
                "Proxy for pontine / brainstem REM-atonia and sleep-state gating circuitry"
            ),
            "hippocampus": (
                "Medial temporal anchor for secondary inflammatory / paraneoplastic narcolepsy-like presentations"
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "hla_autoimmune_risk": (
                "Immune-genetic susceptibility load, centered on strong HLA association"
            ),
            "infection_trigger_load": (
                "Post-infectious trigger burden that can precipitate autoimmune targeting in susceptible individuals"
            ),
            "autoimmune_encephalitis_trigger": (
                "Autoimmune encephalitic burden affecting hypothalamic or limbic sleep-regulation systems"
            ),
            "secondary_lesion_burden": (
                "Structural lesion burden affecting hypothalamus or brainstem, such as tumor, stroke, or inflammation"
            ),
            "rare_hypocretin_pathway_mutation": (
                "Rare direct hypocretin-system receptor or pathway defect load"
            ),
            "emotional_trigger_load": (
                "Emotional triggering pressure relevant to cataplexy susceptibility"
            ),
            "comorbid_sleep_disorder_burden": (
                "Additional burden from comorbid sleep pathology that can worsen sleep fragmentation and sleepiness"
            ),
            "sodium_oxybate_support": (
                "Protective nighttime GABA-B-linked stabilization and slow-wave sleep consolidation support"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "autoimmune_hypocretin_targeting": (
                "Selective immune attack against hypocretin-producing neurons"
            ),
            "hypocretin_deficiency": (
                "Loss of stabilizing hypocretin signaling due to neuronal loss or rare pathway defects"
            ),
            "state_boundary_dyscontrol": (
                "Instability of the boundaries between wakefulness, NREM sleep, and REM sleep"
            ),
            "rem_intrusion_liability": (
                "Propensity for REM-like phenomena to intrude into waking consciousness"
            ),
            "nocturnal_sleep_fragmentation": (
                "Poorly consolidated nocturnal sleep that worsens daytime instability"
            ),
            "pontine_rem_atonia_instability": (
                "Brainstem REM-atonia gating instability relevant to sleep paralysis and cataplexy-like states"
            ),
            "cataplexy_trigger_sensitivity": (
                "Emotion-linked susceptibility to sudden loss of muscle tone on a narcoleptic background"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "excessive_daytime_sleepiness": (
                "Pathological daytime sleepiness and irresistible sleep tendency"
            ),
            "cataplexy": (
                "Emotion-triggered episodes of muscle weakness or atonia"
            ),
            "hypnagogic_hypnopompic_hallucinations": (
                "Vivid dream-like perceptual experiences at sleep onset or awakening"
            ),
            "sleep_paralysis": (
                "Transient inability to move at the boundary of sleep and wake"
            ),
            "disrupted_nocturnal_sleep": (
                "Poorly consolidated nighttime sleep despite daytime sleepiness"
            ),
            "wake_rem_boundary_intrusions": (
                "General burden of REM-wake boundary intrusions across narcoleptic symptoms"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "hla_autoimmune_risk",
                "target": "autoimmune_hypocretin_targeting",
                "relation": "increases immune-mediated susceptibility to hypocretin-neuron loss",
                "narcolepsy_change": "increased",
            },
            {
                "source": "infection_trigger_load",
                "target": "autoimmune_hypocretin_targeting",
                "relation": "can trigger autoimmune cross-reactivity and molecular-mimicry pressure",
                "narcolepsy_change": "increased",
            },
            {
                "source": "autoimmune_encephalitis_trigger",
                "target": "autoimmune_hypocretin_targeting",
                "relation": "adds autoimmune inflammatory pressure on hypothalamic and limbic structures",
                "narcolepsy_change": "increased",
            },
            {
                "source": "rare_hypocretin_pathway_mutation",
                "target": "hypocretin_deficiency",
                "relation": "rare direct hypocretin-system pathway defects can destabilize state control",
                "narcolepsy_change": "increased",
            },
            {
                "source": "secondary_lesion_burden",
                "target": "state_boundary_dyscontrol",
                "relation": "secondary hypothalamic or brainstem lesions destabilize sleep-wake boundaries",
                "narcolepsy_change": "increased",
            },
            {
                "source": "secondary_lesion_burden",
                "target": "pontine_rem_atonia_instability",
                "relation": "brainstem and pontine lesions destabilize REM-atonia gating",
                "narcolepsy_change": "increased",
            },
            {
                "source": "comorbid_sleep_disorder_burden",
                "target": "nocturnal_sleep_fragmentation",
                "relation": "additional sleep pathology can worsen nocturnal sleep consolidation",
                "narcolepsy_change": "increased",
            },
            {
                "source": "sodium_oxybate_support",
                "target": "nocturnal_sleep_fragmentation",
                "relation": "enhances slow-wave sleep and consolidates nocturnal sleep",
                "narcolepsy_change": "decreased",
            },
            {
                "source": "sodium_oxybate_support",
                "target": "cataplexy_trigger_sensitivity",
                "relation": "reduces emotional triggering of cataplexy by stabilizing nighttime sleep architecture",
                "narcolepsy_change": "decreased",
            },
            {
                "source": "autoimmune_hypocretin_targeting",
                "target": "hypocretin_deficiency",
                "relation": "selective hypocretin-neuron loss reduces stabilizing wake-promotion signaling",
                "narcolepsy_change": "increased",
            },
            {
                "source": "hypocretin_deficiency",
                "target": "state_boundary_dyscontrol",
                "relation": "destabilizes boundaries between wakefulness, NREM, and REM sleep",
                "narcolepsy_change": "increased",
            },
            {
                "source": "state_boundary_dyscontrol",
                "target": "rem_intrusion_liability",
                "relation": "permits REM phenomena to intrude into waking consciousness",
                "narcolepsy_change": "increased",
            },
            {
                "source": "state_boundary_dyscontrol",
                "target": "nocturnal_sleep_fragmentation",
                "relation": "disrupts stable nighttime sleep architecture",
                "narcolepsy_change": "increased",
            },
            {
                "source": "state_boundary_dyscontrol",
                "target": "pontine_rem_atonia_instability",
                "relation": "destabilizes distributed REM-atonia control systems",
                "narcolepsy_change": "increased",
            },
            {
                "source": "emotional_trigger_load",
                "target": "cataplexy_trigger_sensitivity",
                "relation": "raises vulnerability to emotion-linked loss of tone",
                "narcolepsy_change": "increased",
            },
            {
                "source": "rem_intrusion_liability",
                "target": "cataplexy_trigger_sensitivity",
                "relation": "amplifies the likelihood that REM-like atonia intrudes during wakefulness",
                "narcolepsy_change": "increased",
            },
            {
                "source": "hypocretin_deficiency",
                "target": "hypothalamus_proxy",
                "relation": "maps the core narcoleptic lesion burden to hypothalamic systems",
                "narcolepsy_change": "increased",
            },
            {
                "source": "autoimmune_hypocretin_targeting",
                "target": "hypothalamus_proxy",
                "relation": "adds immune injury burden to hypothalamic sleep-wake regulation",
                "narcolepsy_change": "increased",
            },
            {
                "source": "state_boundary_dyscontrol",
                "target": "thalamus_proxy",
                "relation": "maps distributed state-instability burden onto thalamic gating systems",
                "narcolepsy_change": "increased",
            },
            {
                "source": "nocturnal_sleep_fragmentation",
                "target": "frontal_cortex_proxy",
                "relation": "chronic sleep instability burdens frontal cortical function",
                "narcolepsy_change": "increased",
            },
            {
                "source": "pontine_rem_atonia_instability",
                "target": "brainstem_pontine_proxy",
                "relation": "maps REM-atonia gating dysfunction to pontine / brainstem circuitry",
                "narcolepsy_change": "increased",
            },
            {
                "source": "autoimmune_encephalitis_trigger",
                "target": "hippocampus",
                "relation": "secondary autoimmune narcolepsy-like states can involve mesial temporal and hippocampal structures",
                "narcolepsy_change": "increased",
            },
            {
                "source": "secondary_lesion_burden",
                "target": "brainstem_pontine_proxy",
                "relation": "secondary structural pathology burdens pontine narcolepsy-related circuitry",
                "narcolepsy_change": "increased",
            },
            {
                "source": "hypothalamus_proxy",
                "target": "excessive_daytime_sleepiness",
                "relation": "core hypothalamic hypocretin-system burden promotes pathological sleepiness",
                "narcolepsy_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "excessive_daytime_sleepiness",
                "relation": "distributed state-regulation burden worsens vigilance and wake maintenance",
                "narcolepsy_change": "increased",
            },
            {
                "source": "frontal_cortex_proxy",
                "target": "excessive_daytime_sleepiness",
                "relation": "frontal cortical burden contributes to daytime performance collapse and sleepiness",
                "narcolepsy_change": "increased",
            },
            {
                "source": "cataplexy_trigger_sensitivity",
                "target": "cataplexy",
                "relation": "raises the probability of emotion-triggered atonia episodes",
                "narcolepsy_change": "increased",
            },
            {
                "source": "rem_intrusion_liability",
                "target": "hypnagogic_hypnopompic_hallucinations",
                "relation": "produces vivid dream-like imagery at sleep-wake transitions",
                "narcolepsy_change": "increased",
            },
            {
                "source": "rem_intrusion_liability",
                "target": "sleep_paralysis",
                "relation": "permits REM atonia to persist into wakefulness",
                "narcolepsy_change": "increased",
            },
            {
                "source": "brainstem_pontine_proxy",
                "target": "sleep_paralysis",
                "relation": "brainstem REM-atonia gating dysfunction worsens paralysis episodes",
                "narcolepsy_change": "increased",
            },
            {
                "source": "brainstem_pontine_proxy",
                "target": "hypnagogic_hypnopompic_hallucinations",
                "relation": "brainstem REM-state instability intensifies dream-like intrusion burden",
                "narcolepsy_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "hypnagogic_hypnopompic_hallucinations",
                "relation": "secondary limbic involvement can accompany narcolepsy-like dream intrusion states",
                "narcolepsy_change": "increased",
            },
            {
                "source": "nocturnal_sleep_fragmentation",
                "target": "disrupted_nocturnal_sleep",
                "relation": "directly drives nighttime sleep disruption and poor consolidation",
                "narcolepsy_change": "increased",
            },
            {
                "source": "rem_intrusion_liability",
                "target": "wake_rem_boundary_intrusions",
                "relation": "summarizes the global burden of REM-wake overlap",
                "narcolepsy_change": "increased",
            },
            {
                "source": "state_boundary_dyscontrol",
                "target": "wake_rem_boundary_intrusions",
                "relation": "unstable state transitions increase boundary-intrusion burden",
                "narcolepsy_change": "increased",
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
        matches: List[Any] = []
        try:
            matches.extend(
                list(
                    self.atlas.find_regions(
                        query,
                        all_versions=False,
                        filter_children=False,
                        find_topmost=False,
                    )
                )
            )
        except Exception:
            pass
        try:
            if hasattr(self.parcellation, "find"):
                found = self.parcellation.find(query, filter_children=False)
                if found:
                    matches.extend(list(found))
        except Exception:
            pass

        out: List[Any] = []
        seen = set()
        for region in matches:
            key = (self._name_of(region), getattr(region, "identifier", None))
            if key in seen:
                continue
            seen.add(key)
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self.parcellation.name in str(parc_name):
                out.append(region)
                continue
            if not parc_name:
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_terms = {
            "hypothalamus",
            "thalamus",
            "frontal cortex",
            "prefrontal cortex",
            "brainstem",
            "pons",
            "hippocampus",
        }
        generic_penalty = 1 if name in generic_terms else 0
        parent_penalty = 1 if name.count("(") == 0 and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, parent_penalty)

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
            rows.append({"name": row[0], "identifier": row[1], "parcellation": row[2]})
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
            self._connectivity_matrix = compound[0].data.copy()
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]
        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
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
            series = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        matched: Dict[str, Any] = {}
        all_labels = list(dict.fromkeys(list(matrix.index) + list(matrix.columns)))
        for key, region in self.region_objects.items():
            label = self._match_region_label(all_labels, region)
            if label is not None:
                matched[key] = label

        if not matched:
            return pd.DataFrame()

        rows: List[pd.Series] = []
        for src_key, src_label in matched.items():
            row: Dict[str, float] = {}
            for dst_key, dst_label in matched.items():
                value = pd.NA
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix[dst_label].loc[src_label]
                    except Exception:
                        value = pd.NA
                row[dst_key] = value
            rows.append(pd.Series(row, name=src_key))
        return pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    def build(
        self,
        gene_panel: Sequence[str] = NARCOLEPSY_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        nodes = []
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
                        "description": self.region_node_descriptions.get(
                            key, "Atlas-backed region or proxy node (unresolved in this environment)"
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
                    "label": region.name,
                    "node_type": "region",
                    "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node"),
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
            "metadata": {
                "model_name": self.__class__.__name__,
                "gene_panel": list(gene_panel),
                "connectivity_cohort": self.connectivity_cohort,
                "space": self.space_spec,
                "parcellation": self.parcellation_spec,
                "notes": (
                    "Research scaffold only. The model emphasizes hypothalamic hypocretin loss, "
                    "sleep-wake boundary dyscontrol, REM intrusion, and secondary structural / autoimmune pathways."
                ),
            },
        }

    def simulate(
        self,
        hla_autoimmune_risk: float = 0.8,
        infection_trigger_load: float = 0.3,
        autoimmune_encephalitis_trigger: float = 0.05,
        secondary_lesion_burden: float = 0.05,
        rare_hypocretin_pathway_mutation: float = 0.02,
        emotional_trigger_load: float = 0.45,
        comorbid_sleep_disorder_burden: float = 0.15,
        sodium_oxybate_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass simulation.

        All inputs are expected on a 0..1 scale. Higher support values are protective;
        all other inputs represent higher burden or risk. Returned regional-state values
        represent dysfunction burden in the anchored system.
        """

        inputs = {
            "hla_autoimmune_risk": self._clip01(hla_autoimmune_risk),
            "infection_trigger_load": self._clip01(infection_trigger_load),
            "autoimmune_encephalitis_trigger": self._clip01(autoimmune_encephalitis_trigger),
            "secondary_lesion_burden": self._clip01(secondary_lesion_burden),
            "rare_hypocretin_pathway_mutation": self._clip01(rare_hypocretin_pathway_mutation),
            "emotional_trigger_load": self._clip01(emotional_trigger_load),
            "comorbid_sleep_disorder_burden": self._clip01(comorbid_sleep_disorder_burden),
            "sodium_oxybate_support": self._clip01(sodium_oxybate_support),
        }

        latents = {
            "autoimmune_hypocretin_targeting": self._clip01(
                0.46 * inputs["hla_autoimmune_risk"]
                + 0.24 * inputs["infection_trigger_load"]
                + 0.22 * inputs["autoimmune_encephalitis_trigger"]
            ),
            "hypocretin_deficiency": 0.0,
            "state_boundary_dyscontrol": 0.0,
            "rem_intrusion_liability": 0.0,
            "nocturnal_sleep_fragmentation": 0.0,
            "pontine_rem_atonia_instability": 0.0,
            "cataplexy_trigger_sensitivity": 0.0,
        }

        latents["hypocretin_deficiency"] = self._clip01(
            0.58 * latents["autoimmune_hypocretin_targeting"]
            + 0.20 * inputs["rare_hypocretin_pathway_mutation"]
            + 0.12 * inputs["secondary_lesion_burden"]
        )

        latents["state_boundary_dyscontrol"] = self._clip01(
            0.54 * latents["hypocretin_deficiency"]
            + 0.16 * inputs["secondary_lesion_burden"]
            + 0.10 * inputs["autoimmune_encephalitis_trigger"]
            + 0.08 * inputs["comorbid_sleep_disorder_burden"]
            - 0.20 * inputs["sodium_oxybate_support"]
        )

        latents["rem_intrusion_liability"] = self._clip01(
            0.46 * latents["state_boundary_dyscontrol"]
            + 0.18 * inputs["secondary_lesion_burden"]
            + 0.14 * inputs["comorbid_sleep_disorder_burden"]
            - 0.16 * inputs["sodium_oxybate_support"]
        )

        latents["nocturnal_sleep_fragmentation"] = self._clip01(
            0.38 * latents["state_boundary_dyscontrol"]
            + 0.24 * inputs["comorbid_sleep_disorder_burden"]
            + 0.10 * inputs["secondary_lesion_burden"]
            - 0.28 * inputs["sodium_oxybate_support"]
        )

        latents["pontine_rem_atonia_instability"] = self._clip01(
            0.36 * latents["state_boundary_dyscontrol"]
            + 0.30 * inputs["secondary_lesion_burden"]
            + 0.14 * inputs["autoimmune_encephalitis_trigger"]
            + 0.08 * latents["rem_intrusion_liability"]
            - 0.10 * inputs["sodium_oxybate_support"]
        )

        latents["cataplexy_trigger_sensitivity"] = self._clip01(
            0.36 * latents["hypocretin_deficiency"]
            + 0.24 * latents["rem_intrusion_liability"]
            + 0.22 * inputs["emotional_trigger_load"]
            - 0.18 * inputs["sodium_oxybate_support"]
        )

        regional_state = {
            "hypothalamus_proxy": self._clip01(
                0.58 * latents["hypocretin_deficiency"]
                + 0.22 * latents["autoimmune_hypocretin_targeting"]
                + 0.12 * inputs["secondary_lesion_burden"]
            ),
            "thalamus_proxy": self._clip01(
                0.42 * latents["state_boundary_dyscontrol"]
                + 0.20 * latents["nocturnal_sleep_fragmentation"]
                + 0.12 * inputs["comorbid_sleep_disorder_burden"]
            ),
            "frontal_cortex_proxy": 0.0,
            "brainstem_pontine_proxy": self._clip01(
                0.54 * latents["pontine_rem_atonia_instability"]
                + 0.20 * inputs["secondary_lesion_burden"]
                + 0.10 * inputs["autoimmune_encephalitis_trigger"]
            ),
            "hippocampus": self._clip01(
                0.42 * inputs["autoimmune_encephalitis_trigger"]
                + 0.16 * inputs["secondary_lesion_burden"]
                + 0.12 * latents["nocturnal_sleep_fragmentation"]
            ),
        }

        regional_state["frontal_cortex_proxy"] = self._clip01(
            0.34 * latents["state_boundary_dyscontrol"]
            + 0.28 * latents["nocturnal_sleep_fragmentation"]
            + 0.12 * regional_state["thalamus_proxy"]
            + 0.12 * inputs["comorbid_sleep_disorder_burden"]
        )

        symptoms = {
            "excessive_daytime_sleepiness": self._clip01(
                0.34 * latents["state_boundary_dyscontrol"]
                + 0.24 * latents["nocturnal_sleep_fragmentation"]
                + 0.20 * regional_state["hypothalamus_proxy"]
                + 0.12 * regional_state["thalamus_proxy"]
                + 0.08 * regional_state["frontal_cortex_proxy"]
            ),
            "cataplexy": self._clip01(
                0.46 * latents["cataplexy_trigger_sensitivity"]
                + 0.20 * latents["rem_intrusion_liability"]
                + 0.16 * regional_state["brainstem_pontine_proxy"]
                + 0.12 * inputs["emotional_trigger_load"]
                - 0.12 * inputs["sodium_oxybate_support"]
            ),
            "hypnagogic_hypnopompic_hallucinations": self._clip01(
                0.42 * latents["rem_intrusion_liability"]
                + 0.20 * regional_state["brainstem_pontine_proxy"]
                + 0.12 * regional_state["hippocampus"]
                + 0.10 * latents["state_boundary_dyscontrol"]
            ),
            "sleep_paralysis": self._clip01(
                0.44 * latents["rem_intrusion_liability"]
                + 0.26 * regional_state["brainstem_pontine_proxy"]
                + 0.12 * latents["nocturnal_sleep_fragmentation"]
            ),
            "disrupted_nocturnal_sleep": self._clip01(
                0.58 * latents["nocturnal_sleep_fragmentation"]
                + 0.16 * latents["state_boundary_dyscontrol"]
                + 0.10 * regional_state["brainstem_pontine_proxy"]
                - 0.12 * inputs["sodium_oxybate_support"]
            ),
            "wake_rem_boundary_intrusions": self._clip01(
                0.50 * latents["rem_intrusion_liability"]
                + 0.20 * latents["state_boundary_dyscontrol"]
                + 0.10 * regional_state["brainstem_pontine_proxy"]
            ),
        }

        mean_core_symptoms = (
            symptoms["excessive_daytime_sleepiness"]
            + symptoms["cataplexy"]
            + symptoms["hypnagogic_hypnopompic_hallucinations"]
            + symptoms["sleep_paralysis"]
            + symptoms["disrupted_nocturnal_sleep"]
        ) / 5.0

        phenotypes = {
            "sleep_wake_boundary_dyscontrol_profile": self._clip01(
                (
                    symptoms["excessive_daytime_sleepiness"]
                    + symptoms["disrupted_nocturnal_sleep"]
                    + symptoms["wake_rem_boundary_intrusions"]
                )
                / 3.0
            ),
            "narcolepsy_type1_profile": self._clip01(
                (
                    symptoms["excessive_daytime_sleepiness"]
                    + symptoms["cataplexy"]
                    + symptoms["hypnagogic_hypnopompic_hallucinations"]
                    + symptoms["sleep_paralysis"]
                )
                / 4.0
            ),
            "rem_intrusion_profile": self._clip01(
                (
                    symptoms["wake_rem_boundary_intrusions"]
                    + symptoms["hypnagogic_hypnopompic_hallucinations"]
                    + symptoms["sleep_paralysis"]
                    + symptoms["cataplexy"]
                )
                / 4.0
            ),
            "secondary_narcolepsy_like_profile": self._clip01(
                0.35 * mean_core_symptoms
                + 0.20 * inputs["secondary_lesion_burden"]
                + 0.18 * inputs["autoimmune_encephalitis_trigger"]
                + 0.12 * regional_state["brainstem_pontine_proxy"]
                + 0.10 * regional_state["hippocampus"]
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="value"),
            "latents": pd.Series(latents, name="value"),
            "regional_state": pd.Series(regional_state, name="value"),
            "symptoms": pd.Series(symptoms, name="value"),
            "phenotypes": pd.Series(phenotypes, name="value"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        out = assignments.copy()
        if "region" in out.columns:
            out["region"] = out["region"].map(self._name_of)
        for candidate in (
            "map value",
            "correlation",
            "intersection over union",
            "contained",
            "contains",
        ):
            if candidate in out.columns:
                out = out.sort_values(candidate, ascending=False)
                break
        return out.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None and node_key in self.region_candidates:
            region = self._resolve_region(self.region_candidates[node_key])
        if region is None:
            return None
        try:
            with siibra.QUIET:
                return region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            return None


if __name__ == "__main__":
    if siibra is None:  # pragma: no cover - environment dependent
        print("siibra is not installed in this runtime. Install siibra-python to build and query the scaffold.")
        raise SystemExit(0)

    model = NarcolepsyModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(
        built["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "region_identifier",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(built["edges"].to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY ===")
    circuit_df = built["circuit_connectivity"]
    if circuit_df.empty:
        print("No circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.round(3).to_string())

    for region_key in (
        "hypothalamus_proxy",
        "thalamus_proxy",
        "brainstem_pontine_proxy",
        "hippocampus",
    ):
        receptor_df = built["receptors"].get(region_key, pd.DataFrame())
        gene_df = built["genes"].get(region_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(region_key, pd.DataFrame())

        print(f"\n=== RECEPTOR TABLE: {region_key} ===")
        if receptor_df.empty:
            print("No receptor fingerprint available.")
        else:
            print(receptor_df.head(10).to_string(index=False))

        print(f"\n=== GENE TABLE: {region_key} ===")
        if gene_df.empty:
            print("No gene-expression summary available.")
        else:
            print(gene_df.head(10).to_string(index=False))

        print(f"\n=== CONNECTIVITY PROFILE: {region_key} ===")
        if conn_df.empty:
            print("No connectivity profile available.")
        else:
            print(conn_df.to_string(index=False))

    print("\n=== SIMULATION EXAMPLE ===")
    simulated = model.simulate(
        hla_autoimmune_risk=0.9,
        infection_trigger_load=0.55,
        autoimmune_encephalitis_trigger=0.08,
        secondary_lesion_burden=0.04,
        rare_hypocretin_pathway_mutation=0.02,
        emotional_trigger_load=0.65,
        comorbid_sleep_disorder_burden=0.18,
        sodium_oxybate_support=0.45,
    )
    for name, series in simulated.items():
        print(f"\n--- {name.upper()} ---")
        print(series.sort_values(ascending=False).round(3).to_string())

    # Example coordinate assignments in MNI152:
    # print(model.assign_mni_point((-6, -10, -8)).head())   # hypothalamic vicinity
    # print(model.assign_mni_point((4, -28, -30)).head())   # pontine vicinity
