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


PANIC_DISORDER_GENE_PANEL = [
    "GABRA1",
    "GABRA2",
    "GABRB2",
    "GABRG2",
    "CCK",
    "CCKBR",
    "BDNF",
    "CRHR1",
    "NR3C1",
]


class PanicDisorderModel:
    """
    Atlas-grounded siibra scaffold for Panic Disorder.

    This script is a research scaffold, not a diagnostic, prognostic, or treatment
    tool. It translates a chapter-level description of panic disorder into a
    transparent mechanistic graph and a simple normalized simulator.

    Chapter-driven design choices:
    - GABAergic dysfunction is modeled as a latent inhibitory-control deficit rather
      than being forced into a single anatomical parcel. Benzodiazepine support is
      represented as an acute protective input because the chapter highlights rapid
      symptom relief via GABA-A receptor agonism.
    - The amygdala, insula, anterior cingulate cortex (ACC), prefrontal control
      cortex, and a brainstem alarm proxy are used as conservative atlas anchors
      because the chapter directly names these structures in the panic network.
    - Familial vulnerability, CO2 panicogenic sensitivity, and early-life trauma /
      sensitization are modeled as distinct inputs because the chapter emphasizes all
      three as biologically relevant routes into panic vulnerability.
    - Cholecystokinin (CCK) is retained as a latent modulatory pressure on the GABA
      system instead of being over-localized.

    Higher regional-state values produced by ``simulate()`` represent greater
    dysfunction burden in that node, not healthier activation.
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
                "siibra is required to use PanicDisorderModel. "
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
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "insula": [
                "Area Id1 left",
                "Area Id2 left",
                "Area Ig2 left",
                "insula left",
                "insula",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area s24 (pACC) left",
                "Area p24ab left",
                "anterior cingulate cortex",
                "anterior cingulate",
                "cingulate",
            ],
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area IFJ left",
                "Area 10 left",
                "prefrontal cortex",
                "dorsolateral prefrontal",
            ],
            "brainstem_alarm_proxy": [
                "Pons left",
                "Pons",
                "brainstem",
                "pontine tegmentum",
                "midbrain",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": "Threat-detection and fear-salience node expected to be hyperreactive during panic.",
            "insula": "Interoceptive awareness node contributing to alarm interpretations of bodily sensations.",
            "acc": "Anterior cingulate control and conflict-monitoring node integrating affective and cognitive burden.",
            "pfc_control": "Prefrontal control proxy for appraisal, inhibition, decision-making, and top-down regulation.",
            "brainstem_alarm_proxy": "Proxy for brainstem autonomic alarm and arousal systems active during panic attacks.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Familial or inherited biological diathesis increasing panic vulnerability"
            ),
            "co2_panic_sensitivity": (
                "Trait-like panicogenic sensitivity to respiratory / suffocation-like interoceptive challenge"
            ),
            "early_life_trauma": (
                "Childhood trauma or abuse contributing to long-term stress sensitization and epigenetic vulnerability"
            ),
            "current_stress_load": (
                "Current stress burden capable of precipitating or amplifying panic circuitry activation"
            ),
            "bodily_sensation_trigger_load": (
                "Intensity of interoceptive cues such as chest tightness, palpitations, dyspnea, or dizziness"
            ),
            "benzodiazepine_support": (
                "Acute GABA-A-enhancing anxiolytic support that can rapidly reduce panic symptoms"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "fear_stress_sensitization": (
                "Kindling-like sensitization of fear and stress systems driven by genetic risk and adverse experience"
            ),
            "cck_gaba_modulatory_pressure": (
                "CCK-linked modulation of inhibitory control that can intensify anxiety regulation problems"
            ),
            "gabaergic_disinhibition": (
                "Reduced inhibitory buffering and increased neuronal hyperexcitability within the panic network"
            ),
            "interoceptive_alarm_bias": (
                "Hyper-reactive interpretation of bodily sensations as danger signals"
            ),
            "frontolimbic_regulatory_failure": (
                "Failure of top-down PFC and ACC regulation over hyperactive limbic threat responses"
            ),
            "autonomic_brainstem_alarm_activation": (
                "Rapid physiological alarm-state activation contributing to the somatic surge of panic"
            ),
            "catastrophic_bodily_misinterpretation": (
                "Escalating belief that bodily sensations signify catastrophe such as heart attack or stroke"
            ),
            "anticipatory_anxiety_circuit": (
                "Between-attack worry and expectancy-driven vigilance that maintains panic vulnerability"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "panic_attack_intensity": (
                "Acute intensity of the panic episode itself"
            ),
            "somatic_alarm_symptoms": (
                "Prominent physical symptoms that often drive emergency medical presentation"
            ),
            "catastrophic_medical_misinterpretation": (
                "Conviction that panic sensations indicate a serious medical emergency"
            ),
            "anticipatory_anxiety": (
                "Persistent worry and dread between attacks"
            ),
            "attention_memory_executive_difficulty": (
                "Cognitive inefficiency in attention, memory, mental flexibility, and executive control"
            ),
            "avoidance_safety_behavior": (
                "Avoidance, reassurance-seeking, or safety behavior aimed at preventing another attack"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "fear_stress_sensitization",
                "relation": "increases inherited biological diathesis for panic-network reactivity",
                "panic_change": "increased",
            },
            {
                "source": "early_life_trauma",
                "target": "fear_stress_sensitization",
                "relation": "kindles and sensitizes fear-stress circuitry across development",
                "panic_change": "increased",
            },
            {
                "source": "current_stress_load",
                "target": "fear_stress_sensitization",
                "relation": "adds current load to already sensitized stress systems",
                "panic_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "cck_gaba_modulatory_pressure",
                "relation": "can contribute to inherited vulnerability in inhibitory-neuropeptide regulation",
                "panic_change": "increased",
            },
            {
                "source": "current_stress_load",
                "target": "cck_gaba_modulatory_pressure",
                "relation": "stress increases modulatory pressure on anxiety-related inhibitory regulation",
                "panic_change": "increased",
            },
            {
                "source": "cck_gaba_modulatory_pressure",
                "target": "gabaergic_disinhibition",
                "relation": "can weaken inhibitory buffering within the anxiety network",
                "panic_change": "increased",
            },
            {
                "source": "benzodiazepine_support",
                "target": "gabaergic_disinhibition",
                "relation": "acutely restores inhibitory tone through GABA-A receptor agonism",
                "panic_change": "decreased",
            },
            {
                "source": "co2_panic_sensitivity",
                "target": "interoceptive_alarm_bias",
                "relation": "raises vulnerability to panicogenic respiratory and suffocation-like cues",
                "panic_change": "increased",
            },
            {
                "source": "bodily_sensation_trigger_load",
                "target": "interoceptive_alarm_bias",
                "relation": "provides the bodily cues that are read as imminent danger",
                "panic_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "interoceptive_alarm_bias",
                "relation": "reduces damping of alarm responses to internal bodily sensations",
                "panic_change": "increased",
            },
            {
                "source": "fear_stress_sensitization",
                "target": "frontolimbic_regulatory_failure",
                "relation": "weakens top-down control over hyperreactive fear circuits",
                "panic_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "frontolimbic_regulatory_failure",
                "relation": "heightens neuronal excitability and destabilizes cortical regulation",
                "panic_change": "increased",
            },
            {
                "source": "interoceptive_alarm_bias",
                "target": "autonomic_brainstem_alarm_activation",
                "relation": "rapidly recruits physiological alarm-state systems during panic",
                "panic_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "autonomic_brainstem_alarm_activation",
                "relation": "permits stronger autonomic alarm surges",
                "panic_change": "increased",
            },
            {
                "source": "interoceptive_alarm_bias",
                "target": "catastrophic_bodily_misinterpretation",
                "relation": "turns bodily sensations into perceived evidence of catastrophe",
                "panic_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "catastrophic_bodily_misinterpretation",
                "relation": "impairs cognitive reappraisal of bodily danger signals",
                "panic_change": "increased",
            },
            {
                "source": "fear_stress_sensitization",
                "target": "anticipatory_anxiety_circuit",
                "relation": "maintains a persistent expectation of future panic",
                "panic_change": "increased",
            },
            {
                "source": "catastrophic_bodily_misinterpretation",
                "target": "anticipatory_anxiety_circuit",
                "relation": "feeds between-attack worry and hypervigilance",
                "panic_change": "increased",
            },
            {
                "source": "fear_stress_sensitization",
                "target": "amygdala",
                "relation": "maps sensitized threat bias onto amygdala hyperreactivity",
                "panic_change": "increased",
            },
            {
                "source": "interoceptive_alarm_bias",
                "target": "insula",
                "relation": "maps interoceptive danger monitoring onto insular hyperactivity",
                "panic_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "acc",
                "relation": "burdens ACC conflict and affect-integration functions",
                "panic_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "pfc_control",
                "relation": "burdens prefrontal appraisal, inhibition, and decision-making systems",
                "panic_change": "increased",
            },
            {
                "source": "autonomic_brainstem_alarm_activation",
                "target": "brainstem_alarm_proxy",
                "relation": "maps panic-state physiological alarm onto brainstem arousal systems",
                "panic_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "panic_attack_intensity",
                "relation": "drives fear intensity during the panic episode",
                "panic_change": "increased",
            },
            {
                "source": "insula",
                "target": "panic_attack_intensity",
                "relation": "intensifies conscious experience of bodily alarm during panic",
                "panic_change": "increased",
            },
            {
                "source": "brainstem_alarm_proxy",
                "target": "somatic_alarm_symptoms",
                "relation": "drives the prominent physiological surge of panic",
                "panic_change": "increased",
            },
            {
                "source": "catastrophic_bodily_misinterpretation",
                "target": "catastrophic_medical_misinterpretation",
                "relation": "creates conviction that panic symptoms are life-threatening medical events",
                "panic_change": "increased",
            },
            {
                "source": "anticipatory_anxiety_circuit",
                "target": "anticipatory_anxiety",
                "relation": "maintains between-attack worry and expectancy",
                "panic_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "attention_memory_executive_difficulty",
                "relation": "prefrontal dysfunction reduces working memory, flexibility, and strategy use",
                "panic_change": "increased",
            },
            {
                "source": "acc",
                "target": "attention_memory_executive_difficulty",
                "relation": "cingulate dysfunction burdens conflict monitoring and attention",
                "panic_change": "increased",
            },
            {
                "source": "anticipatory_anxiety_circuit",
                "target": "avoidance_safety_behavior",
                "relation": "supports avoidance and safety strategies aimed at preventing future attacks",
                "panic_change": "increased",
            },
            {
                "source": "catastrophic_bodily_misinterpretation",
                "target": "avoidance_safety_behavior",
                "relation": "encourages reassurance seeking and avoidance of feared sensations or contexts",
                "panic_change": "increased",
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
            "amygdala",
            "insula",
            "anterior cingulate cortex",
            "anterior cingulate",
            "prefrontal cortex",
            "brainstem",
            "pons",
            "cingulate",
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
        gene_panel: Sequence[str] = PANIC_DISORDER_GENE_PANEL,
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
                    "Research scaffold only. The model emphasizes GABAergic disinhibition, interoceptive alarm bias, "
                    "frontolimbic control failure, and somatic-autonomic panic expression."
                ),
            },
        }

    def simulate(
        self,
        genetic_vulnerability: float = 0.7,
        co2_panic_sensitivity: float = 0.6,
        early_life_trauma: float = 0.35,
        current_stress_load: float = 0.55,
        bodily_sensation_trigger_load: float = 0.8,
        benzodiazepine_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass panic-disorder simulation.

        All inputs are expected on a 0..1 scale. Higher support values are protective;
        all other inputs represent higher burden or risk. Returned regional-state
        values reflect dysfunction burden in the anchored system.
        """

        inputs = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "co2_panic_sensitivity": self._clip01(co2_panic_sensitivity),
            "early_life_trauma": self._clip01(early_life_trauma),
            "current_stress_load": self._clip01(current_stress_load),
            "bodily_sensation_trigger_load": self._clip01(bodily_sensation_trigger_load),
            "benzodiazepine_support": self._clip01(benzodiazepine_support),
        }

        latents = {
            "fear_stress_sensitization": self._clip01(
                0.34 * inputs["genetic_vulnerability"]
                + 0.36 * inputs["early_life_trauma"]
                + 0.18 * inputs["current_stress_load"]
            ),
            "cck_gaba_modulatory_pressure": 0.0,
            "gabaergic_disinhibition": 0.0,
            "interoceptive_alarm_bias": 0.0,
            "frontolimbic_regulatory_failure": 0.0,
            "autonomic_brainstem_alarm_activation": 0.0,
            "catastrophic_bodily_misinterpretation": 0.0,
            "anticipatory_anxiety_circuit": 0.0,
        }

        latents["cck_gaba_modulatory_pressure"] = self._clip01(
            0.26 * inputs["genetic_vulnerability"]
            + 0.22 * inputs["current_stress_load"]
            + 0.14 * latents["fear_stress_sensitization"]
        )

        latents["gabaergic_disinhibition"] = self._clip01(
            0.30 * latents["cck_gaba_modulatory_pressure"]
            + 0.24 * latents["fear_stress_sensitization"]
            + 0.14 * inputs["co2_panic_sensitivity"]
            + 0.10 * inputs["current_stress_load"]
            - 0.34 * inputs["benzodiazepine_support"]
        )

        latents["interoceptive_alarm_bias"] = self._clip01(
            0.32 * inputs["co2_panic_sensitivity"]
            + 0.26 * inputs["bodily_sensation_trigger_load"]
            + 0.18 * latents["gabaergic_disinhibition"]
            + 0.14 * latents["fear_stress_sensitization"]
            - 0.10 * inputs["benzodiazepine_support"]
        )

        latents["frontolimbic_regulatory_failure"] = self._clip01(
            0.34 * latents["fear_stress_sensitization"]
            + 0.28 * latents["gabaergic_disinhibition"]
            + 0.18 * inputs["current_stress_load"]
            - 0.12 * inputs["benzodiazepine_support"]
        )

        latents["autonomic_brainstem_alarm_activation"] = self._clip01(
            0.38 * latents["interoceptive_alarm_bias"]
            + 0.28 * latents["gabaergic_disinhibition"]
            + 0.12 * inputs["co2_panic_sensitivity"]
            + 0.10 * inputs["bodily_sensation_trigger_load"]
            - 0.12 * inputs["benzodiazepine_support"]
        )

        latents["catastrophic_bodily_misinterpretation"] = self._clip01(
            0.40 * latents["interoceptive_alarm_bias"]
            + 0.26 * latents["frontolimbic_regulatory_failure"]
            + 0.16 * inputs["bodily_sensation_trigger_load"]
            + 0.08 * inputs["current_stress_load"]
        )

        latents["anticipatory_anxiety_circuit"] = self._clip01(
            0.34 * latents["catastrophic_bodily_misinterpretation"]
            + 0.26 * latents["fear_stress_sensitization"]
            + 0.18 * latents["frontolimbic_regulatory_failure"]
            + 0.10 * inputs["current_stress_load"]
        )

        regional_state = {
            "amygdala": self._clip01(
                0.50 * latents["fear_stress_sensitization"]
                + 0.18 * latents["frontolimbic_regulatory_failure"]
                + 0.12 * latents["interoceptive_alarm_bias"]
            ),
            "insula": self._clip01(
                0.48 * latents["interoceptive_alarm_bias"]
                + 0.16 * latents["catastrophic_bodily_misinterpretation"]
                + 0.14 * inputs["bodily_sensation_trigger_load"]
            ),
            "acc": self._clip01(
                0.44 * latents["frontolimbic_regulatory_failure"]
                + 0.18 * latents["anticipatory_anxiety_circuit"]
                + 0.10 * inputs["current_stress_load"]
            ),
            "pfc_control": self._clip01(
                0.46 * latents["frontolimbic_regulatory_failure"]
                + 0.18 * latents["catastrophic_bodily_misinterpretation"]
                + 0.14 * latents["anticipatory_anxiety_circuit"]
            ),
            "brainstem_alarm_proxy": self._clip01(
                0.56 * latents["autonomic_brainstem_alarm_activation"]
                + 0.12 * inputs["co2_panic_sensitivity"]
                + 0.10 * inputs["current_stress_load"]
            ),
        }

        symptoms = {
            "panic_attack_intensity": self._clip01(
                0.26 * regional_state["amygdala"]
                + 0.22 * regional_state["insula"]
                + 0.18 * regional_state["brainstem_alarm_proxy"]
                + 0.16 * latents["catastrophic_bodily_misinterpretation"]
                + 0.10 * inputs["bodily_sensation_trigger_load"]
                - 0.12 * inputs["benzodiazepine_support"]
            ),
            "somatic_alarm_symptoms": self._clip01(
                0.42 * latents["autonomic_brainstem_alarm_activation"]
                + 0.20 * regional_state["brainstem_alarm_proxy"]
                + 0.16 * regional_state["insula"]
                + 0.12 * inputs["bodily_sensation_trigger_load"]
                - 0.10 * inputs["benzodiazepine_support"]
            ),
            "catastrophic_medical_misinterpretation": self._clip01(
                0.58 * latents["catastrophic_bodily_misinterpretation"]
                + 0.16 * regional_state["insula"]
                + 0.12 * regional_state["pfc_control"]
            ),
            "anticipatory_anxiety": self._clip01(
                0.52 * latents["anticipatory_anxiety_circuit"]
                + 0.16 * regional_state["pfc_control"]
                + 0.12 * regional_state["acc"]
                + 0.10 * regional_state["amygdala"]
            ),
            "attention_memory_executive_difficulty": self._clip01(
                0.34 * regional_state["pfc_control"]
                + 0.24 * regional_state["acc"]
                + 0.12 * latents["anticipatory_anxiety_circuit"]
                + 0.10 * inputs["current_stress_load"]
            ),
            "avoidance_safety_behavior": self._clip01(
                0.34 * latents["anticipatory_anxiety_circuit"]
                + 0.24 * latents["catastrophic_bodily_misinterpretation"]
                + 0.14 * regional_state["amygdala"]
                + 0.10 * inputs["bodily_sensation_trigger_load"]
            ),
        }

        phenotypes = {
            "acute_panic_episode_profile": self._clip01(
                (
                    symptoms["panic_attack_intensity"]
                    + symptoms["somatic_alarm_symptoms"]
                    + symptoms["catastrophic_medical_misinterpretation"]
                )
                / 3.0
            ),
            "interoceptive_fear_profile": self._clip01(
                (
                    latents["interoceptive_alarm_bias"]
                    + regional_state["insula"]
                    + symptoms["catastrophic_medical_misinterpretation"]
                )
                / 3.0
            ),
            "anticipatory_panic_profile": self._clip01(
                (
                    symptoms["anticipatory_anxiety"]
                    + symptoms["avoidance_safety_behavior"]
                    + latents["anticipatory_anxiety_circuit"]
                )
                / 3.0
            ),
            "emergency_presentation_pressure": self._clip01(
                0.34 * symptoms["somatic_alarm_symptoms"]
                + 0.30 * symptoms["catastrophic_medical_misinterpretation"]
                + 0.22 * symptoms["panic_attack_intensity"]
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

    model = PanicDisorderModel()
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
        "amygdala",
        "insula",
        "acc",
        "pfc_control",
        "brainstem_alarm_proxy",
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
        genetic_vulnerability=0.78,
        co2_panic_sensitivity=0.72,
        early_life_trauma=0.42,
        current_stress_load=0.65,
        bodily_sensation_trigger_load=0.88,
        benzodiazepine_support=0.22,
    )
    for name, series in simulated.items():
        print(f"\n--- {name.upper()} ---")
        print(series.sort_values(ascending=False).round(3).to_string())

    # Example coordinate assignments in MNI152:
    # print(model.assign_mni_point((-22, -4, -16)).head())   # amygdala vicinity
    # print(model.assign_mni_point((34, 20, 0)).head())      # insular / opercular vicinity
    # print(model.assign_mni_point((2, 34, 20)).head())      # anterior cingulate / medial PFC vicinity
