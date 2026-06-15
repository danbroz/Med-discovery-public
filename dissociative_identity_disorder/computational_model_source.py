from __future__ import annotations

"""
Dissociative Identity Disorder siibra scaffold.

This script translates a DID chapter into a transparent, atlas-grounded research
scaffold using siibra. It is a mechanistic interpretation of chapter logic, not a
validated disease model, diagnostic tool, or treatment recommender.

The chapter emphasizes DID as a severe trauma-related disorder with disruptions in
identity, memory, consciousness, affect regulation, and sensory-motor functioning.
Because direct DID biomarker studies remain limited, several latent biology terms in
this scaffold intentionally represent trauma-informed inferences from adjacent
literatures (for example PTSD, depression, and chronic stress neurobiology) rather
than DID-specific causal proof.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - depends on user environment
    siibra = None
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on user environment
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_DID_GENE_PANEL = [
    "SLC6A4",  # serotonin transporter
    "SLC6A2",  # norepinephrine transporter
    "BDNF",    # plasticity / stress sensitivity
    "FKBP5",   # glucocorticoid signaling
    "NR3C1",   # glucocorticoid receptor
    "CRHR1",   # stress-axis signaling
    "COMT",    # catecholamine degradation
    "DRD2",    # dopaminergic salience / control
    "MAOA",    # monoamine degradation / impulsive aggression links
    "OXTR",    # social-affiliative modulation / stress buffering proxy
]


class DissociativeIdentityDisorderModel:
    """
    Atlas-grounded research scaffold for Dissociative Identity Disorder (DID).

    The model follows a transparent one-pass order:
        inputs -> latent biology -> regional dysregulation burden -> symptoms -> phenotypes

    Important caveat:
    - Higher values in `regional_state` indicate *burden / dysregulation* rather than
      healthy activation.
    - Several latent processes are chapter-faithful abstractions of trauma-related
      neurobiology and are not validated DID biomarkers.
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
                "siibra is required for this scaffold. Install siibra-python in the "
                "target environment before instantiating the model."
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
                "IF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "HC-Transsubiculum (Hippocampus) left",
                "hippocampus left",
                "hippocampus",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area 8d2 (SFG) left",
                "middle frontal gyrus left",
                "prefrontal cortex",
            ],
            "parietal_self_processing": [
                "Area PFm (IPL) left",
                "Area PGa (IPL) left",
                "Area PFcm (IPL) left",
                "Area 7A (SPL) left",
                "inferior parietal lobule left",
                "parietal cortex",
            ],
            "thalamus_proxy": [
                "CGL (Metathalamus) left",
                "CGM (Metathalamus) left",
                "thalamus left",
                "thalamus",
            ],
            "basal_ganglia_proxy": [
                "nucleus accumbens left",
                "caudate left",
                "putamen left",
                "basal ganglia left",
                "basal ganglia",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": "Emotional salience and arousal hub implicated in traumatic affect tagging.",
            "hippocampus": "Mnemonic context-binding system involved in episodic memory consolidation and fragmentation under stress.",
            "pfc_control": "Prefrontal executive-control proxy for top-down inhibition, integration, and self-regulation.",
            "parietal_self_processing": "Parietal self/agency integration proxy relevant to self-awareness and body-state continuity.",
            "thalamus_proxy": "Thalamic gating proxy for conscious-state regulation and distributed information integration.",
            "basal_ganglia_proxy": "Basal-ganglia proxy for action-selection and sensorimotor state shifts mentioned in the chapter.",
        }

        self.input_nodes: Dict[str, str] = {
            "early_life_trauma": "Severe developmental trauma exposure, the chapter's primary etiologic driver.",
            "chronic_stress_load": "Ongoing allostatic burden and stress-system activation over time.",
            "trauma_cue_exposure": "Current or recent cue-driven triggering of trauma-linked states.",
            "genetic_vulnerability": "Inherited liability affecting stress reactivity, monoamine regulation, and plasticity.",
            "depressive_comorbidity": "Depressive burden used by the chapter to motivate monoaminergic dysfunction.",
            "ptsd_hyperarousal_liability": "Trauma-linked hyperarousal tendency inferred from PTSD overlap.",
            "grounding_support": "External stabilization / containment support; optional protective scaffold input.",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_sensitization": "Stress-axis sensitization and chronic trauma-related endocrine dysregulation.",
            "monoaminergic_dysregulation": "Serotonergic, noradrenergic, and dopaminergic imbalance affecting mood, cognition, and impulse control.",
            "noradrenergic_hyperarousal": "High arousal / vigilance state linked to norepinephrine-mediated threat readiness.",
            "trauma_memory_compartmentalization": "Poor integration of factual and emotional trauma memory traces.",
            "frontolimbic_disconnection": "Weak top-down integration between prefrontal control systems and limbic reactivity.",
            "self_state_compartmentalization": "Segregation of self-states and agency representations across contexts.",
            "sensorimotor_dissociation": "Disrupted integration of sensorimotor experience, bodily state, and agency.",
            "temporal_lobe_dysrhythmia": "Temporal-lobe instability proxy reflecting seizure-like alterations in memory, affect, and consciousness described as phenomenologic analogies.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "identity_discontinuity": "Marked discontinuity in self and agency across dissociative states.",
            "dissociative_amnesia": "Memory gaps beyond ordinary forgetting.",
            "altered_consciousness": "Fluctuations in awareness, perception, and state continuity.",
            "affective_lability": "Rapid changes in affective tone and emotion regulation.",
            "hypervigilance": "Persistent threat monitoring and autonomic alarm.",
            "sensory_motor_shifts": "Sensory-motor changes across dissociative states.",
            "impulsivity_aggression": "Secondary impulsive/aggressive expression related to poor serotonergic inhibition.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "early_life_trauma",
                "target": "hpa_axis_sensitization",
                "relation": "recurrent developmental trauma sensitizes the stress axis",
                "did_change": "increased",
            },
            {
                "source": "early_life_trauma",
                "target": "trauma_memory_compartmentalization",
                "relation": "overwhelming trauma fragments autobiographical encoding",
                "did_change": "increased",
            },
            {
                "source": "early_life_trauma",
                "target": "frontolimbic_disconnection",
                "relation": "repeated trauma weakens integrated self-regulation circuitry",
                "did_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hpa_axis_sensitization",
                "relation": "sustained stress keeps the organism in a high-reactivity state",
                "did_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "monoaminergic_dysregulation",
                "relation": "prolonged stress perturbs serotonergic, noradrenergic, and dopaminergic tone",
                "did_change": "increased",
            },
            {
                "source": "trauma_cue_exposure",
                "target": "noradrenergic_hyperarousal",
                "relation": "trauma reminders reactivate high-arousal autonomic responding",
                "did_change": "increased",
            },
            {
                "source": "trauma_cue_exposure",
                "target": "self_state_compartmentalization",
                "relation": "cue-triggered switching pressure increases segregation of self-states",
                "did_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "monoaminergic_dysregulation",
                "relation": "stress-related polymorphisms can bias monoamine regulation",
                "did_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "hpa_axis_sensitization",
                "relation": "genetic liability can heighten the impact of early stress on stress biology",
                "did_change": "increased",
            },
            {
                "source": "depressive_comorbidity",
                "target": "monoaminergic_dysregulation",
                "relation": "depressive comorbidity reinforces monoaminergic disturbance",
                "did_change": "increased",
            },
            {
                "source": "ptsd_hyperarousal_liability",
                "target": "noradrenergic_hyperarousal",
                "relation": "PTSD-like trauma physiology amplifies autonomic alarm",
                "did_change": "increased",
            },
            {
                "source": "grounding_support",
                "target": "noradrenergic_hyperarousal",
                "relation": "stabilization support can buffer runaway arousal",
                "did_change": "decreased",
            },
            {
                "source": "grounding_support",
                "target": "frontolimbic_disconnection",
                "relation": "containment and regulation support reduce system fragmentation pressure",
                "did_change": "decreased",
            },
            {
                "source": "hpa_axis_sensitization",
                "target": "hippocampus",
                "relation": "chronic stress burden impairs hippocampal memory integrity",
                "did_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "amygdala",
                "relation": "high norepinephrine sustains limbic threat bias",
                "did_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "pfc_control",
                "relation": "poor monoaminergic regulation weakens top-down inhibitory control",
                "did_change": "increased",
            },
            {
                "source": "trauma_memory_compartmentalization",
                "target": "hippocampus",
                "relation": "segregated trauma encoding burdens mnemonic integration systems",
                "did_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "pfc_control",
                "relation": "frontolimbic disconnection manifests as impaired control integration",
                "did_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "self_state_compartmentalization",
                "relation": "loss of integrated control fosters separated self-state organization",
                "did_change": "increased",
            },
            {
                "source": "self_state_compartmentalization",
                "target": "parietal_self_processing",
                "relation": "state segregation disturbs agency and self-referential integration",
                "did_change": "increased",
            },
            {
                "source": "sensorimotor_dissociation",
                "target": "thalamus_proxy",
                "relation": "gating and state-integration burden can propagate through thalamic relays",
                "did_change": "increased",
            },
            {
                "source": "sensorimotor_dissociation",
                "target": "basal_ganglia_proxy",
                "relation": "action-selection and motor-set systems carry dissociative state shifts",
                "did_change": "increased",
            },
            {
                "source": "temporal_lobe_dysrhythmia",
                "target": "altered_consciousness",
                "relation": "temporal-lobe instability can resemble dissociative changes in awareness and perception",
                "did_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "dissociative_amnesia",
                "relation": "mnestic integration failure promotes episodic memory gaps",
                "did_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "affective_lability",
                "relation": "limbic hyperreactivity destabilizes emotion processing",
                "did_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "hypervigilance",
                "relation": "elevated norepinephrine sustains vigilance and alarm",
                "did_change": "increased",
            },
            {
                "source": "self_state_compartmentalization",
                "target": "identity_discontinuity",
                "relation": "segregated self-state organization produces discontinuity in self and agency",
                "did_change": "increased",
            },
            {
                "source": "parietal_self_processing",
                "target": "identity_discontinuity",
                "relation": "self-referential integration burden destabilizes continuity of agency",
                "did_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "altered_consciousness",
                "relation": "thalamic gating burden disrupts continuity of conscious state",
                "did_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "sensory_motor_shifts",
                "relation": "basal-ganglia burden contributes to state-linked motor shifts",
                "did_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "impulsivity_aggression",
                "relation": "low serotonergic inhibitory tone can increase impulsive or aggressive expression",
                "did_change": "increased",
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
            cands.extend(["receptor density fingerprint", "ReceptorDensityFingerprint"])
        elif kind == "gene":
            cands.extend(["gene expressions", "GeneExpressions"])
        elif kind == "connectivity":
            cands.extend(["StreamlineCounts", "streamline counts"])
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
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "thalamus",
            "prefrontal cortex",
            "parietal cortex",
            "basal ganglia",
        } else 0
        proxy_penalty = 1 if "gapmap" in name else 0
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
                return self.parcellation.get_region(spec)
            except Exception:
                pass
            try:
                return siibra.get_region(self.parcellation_spec, spec)
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
        for space_arg in (self.space, self.space_spec):
            try:
                props = region.spatial_props(space=space_arg)
            except Exception:
                continue
            if props is None:
                continue
            if isinstance(props, dict):
                return list(props.values())
            if isinstance(props, (list, tuple)):
                return list(props)
            if hasattr(props, "components"):
                return list(getattr(props, "components", []))
            return [props]
        return []

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid.coordinate)
                except Exception:
                    centroid_xyz = None
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
                    return df.reset_index(drop=True)
            return df.reset_index(drop=True)
        return pd.DataFrame()

    @staticmethod
    def _cohort_of(feature: Any) -> str:
        return str(getattr(feature, "cohort", "")).upper().replace(" ", "")

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        wanted = str(self.connectivity_cohort).upper().replace(" ", "")
        compound = next((f for f in feats if self._cohort_of(f) == wanted), feats[0])

        # Connectivity queries often return CompoundFeatures with element-wise DataFrames.
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
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]

        rn = region.name.lower()
        fuzzy = [
            x for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        # fall back to candidate-name matching for proxies
        for candidate in self.region_candidates.get(next((k for k, v in self.region_objects.items() if v == region), ""), []):
            candidate_lower = candidate.lower()
            fuzzy = [x for x in labels if candidate_lower in self._name_of(x).lower()]
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
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """Return pairwise structural connectivity among the resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        rows: List[Dict[str, Any]] = []
        for src_key, src_region in self.region_objects.items():
            src_label = self._match_region_label(list(matrix.index), src_region)
            axis = "index"
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src_region)
                axis = "columns"
            if src_label is None:
                continue

            try:
                src_series = matrix.loc[src_label] if axis == "index" else matrix[src_label]
            except Exception:
                continue

            for dst_key, dst_region in self.region_objects.items():
                if src_key == dst_key:
                    continue
                dst_label = self._match_region_label(list(src_series.index), dst_region)
                if dst_label is None:
                    continue
                try:
                    value = float(src_series.loc[dst_label])
                except Exception:
                    continue
                rows.append(
                    {
                        "source": src_key,
                        "target": dst_key,
                        "value": value,
                    }
                )
        return pd.DataFrame(rows).sort_values(["source", "value"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame(columns=["source", "target", "value"])

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_DID_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

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
                        "description": self.region_descriptions.get(key, "Atlas-backed region or proxy node."),
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
                    "description": self.region_descriptions.get(key, "Atlas-backed region or proxy node."),
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
        circuit_conn = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_conn,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """Probabilistically assign an MNI coordinate to Julich regions."""
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        parcellation=self.parcellation,
                        space=self.assignment_space,
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        """Return a best-effort statistical regional map / mask for a resolved node."""
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        # Compatible with examples using region.get_regional_map("mni152", "statistical")
        for space_arg in (self.assignment_space, self.space, self.space_spec):
            try:
                return region.get_regional_map(space_arg, "statistical")
            except Exception:
                pass
            try:
                return region.get_regional_map(space=space_arg, maptype="statistical")
            except Exception:
                pass
            try:
                return region.get_regional_mask(space_arg)
            except Exception:
                pass
        return None

    def simulate(
        self,
        early_life_trauma: float = 0.80,
        chronic_stress_load: float = 0.70,
        trauma_cue_exposure: float = 0.60,
        genetic_vulnerability: float = 0.35,
        depressive_comorbidity: float = 0.45,
        ptsd_hyperarousal_liability: float = 0.60,
        grounding_support: float = 0.15,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized DID simulation.

        Parameters are 0..1 normalized knobs. Higher symptom values indicate greater
        modeled burden.
        """
        inputs = pd.Series(
            {
                "early_life_trauma": self._clip01(early_life_trauma),
                "chronic_stress_load": self._clip01(chronic_stress_load),
                "trauma_cue_exposure": self._clip01(trauma_cue_exposure),
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "depressive_comorbidity": self._clip01(depressive_comorbidity),
                "ptsd_hyperarousal_liability": self._clip01(ptsd_hyperarousal_liability),
                "grounding_support": self._clip01(grounding_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["hpa_axis_sensitization"] = self._clip01(
            0.42 * inputs["early_life_trauma"]
            + 0.26 * inputs["chronic_stress_load"]
            + 0.14 * inputs["genetic_vulnerability"]
            + 0.12 * inputs["ptsd_hyperarousal_liability"]
            - 0.20 * inputs["grounding_support"]
        )
        latents["monoaminergic_dysregulation"] = self._clip01(
            0.24 * inputs["chronic_stress_load"]
            + 0.20 * inputs["depressive_comorbidity"]
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.16 * latents["hpa_axis_sensitization"]
            + 0.10 * inputs["trauma_cue_exposure"]
            - 0.14 * inputs["grounding_support"]
        )
        latents["noradrenergic_hyperarousal"] = self._clip01(
            0.30 * inputs["trauma_cue_exposure"]
            + 0.26 * inputs["ptsd_hyperarousal_liability"]
            + 0.18 * inputs["chronic_stress_load"]
            + 0.14 * latents["hpa_axis_sensitization"]
            - 0.18 * inputs["grounding_support"]
        )
        latents["trauma_memory_compartmentalization"] = self._clip01(
            0.34 * inputs["early_life_trauma"]
            + 0.20 * latents["hpa_axis_sensitization"]
            + 0.16 * latents["noradrenergic_hyperarousal"]
            + 0.14 * latents["monoaminergic_dysregulation"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.12 * inputs["grounding_support"]
        )
        latents["frontolimbic_disconnection"] = self._clip01(
            0.28 * inputs["early_life_trauma"]
            + 0.20 * latents["monoaminergic_dysregulation"]
            + 0.18 * latents["hpa_axis_sensitization"]
            + 0.16 * latents["noradrenergic_hyperarousal"]
            + 0.10 * inputs["trauma_cue_exposure"]
            - 0.22 * inputs["grounding_support"]
        )
        latents["self_state_compartmentalization"] = self._clip01(
            0.34 * latents["trauma_memory_compartmentalization"]
            + 0.28 * latents["frontolimbic_disconnection"]
            + 0.16 * inputs["trauma_cue_exposure"]
            + 0.08 * inputs["genetic_vulnerability"]
            - 0.12 * inputs["grounding_support"]
        )
        latents["sensorimotor_dissociation"] = self._clip01(
            0.28 * latents["self_state_compartmentalization"]
            + 0.24 * latents["noradrenergic_hyperarousal"]
            + 0.18 * latents["frontolimbic_disconnection"]
            + 0.12 * inputs["trauma_cue_exposure"]
            - 0.10 * inputs["grounding_support"]
        )
        latents["temporal_lobe_dysrhythmia"] = self._clip01(
            0.24 * latents["trauma_memory_compartmentalization"]
            + 0.18 * latents["noradrenergic_hyperarousal"]
            + 0.12 * inputs["trauma_cue_exposure"]
            + 0.10 * inputs["genetic_vulnerability"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["amygdala"] = self._clip01(
            0.52 * latents["noradrenergic_hyperarousal"]
            + 0.20 * inputs["trauma_cue_exposure"]
            + 0.14 * latents["monoaminergic_dysregulation"]
            - 0.16 * inputs["grounding_support"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.42 * latents["trauma_memory_compartmentalization"]
            + 0.22 * latents["hpa_axis_sensitization"]
            + 0.12 * latents["temporal_lobe_dysrhythmia"]
            - 0.10 * inputs["grounding_support"]
        )
        regional_state["pfc_control"] = self._clip01(
            0.40 * latents["frontolimbic_disconnection"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.14 * latents["hpa_axis_sensitization"]
            + 0.08 * latents["noradrenergic_hyperarousal"]
            - 0.24 * inputs["grounding_support"]
        )
        regional_state["parietal_self_processing"] = self._clip01(
            0.34 * latents["self_state_compartmentalization"]
            + 0.18 * latents["frontolimbic_disconnection"]
            + 0.16 * latents["sensorimotor_dissociation"]
            - 0.10 * inputs["grounding_support"]
        )
        regional_state["thalamus_proxy"] = self._clip01(
            0.24 * latents["sensorimotor_dissociation"]
            + 0.22 * latents["self_state_compartmentalization"]
            + 0.16 * latents["noradrenergic_hyperarousal"]
            - 0.08 * inputs["grounding_support"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.24 * latents["sensorimotor_dissociation"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.12 * latents["temporal_lobe_dysrhythmia"]
            - 0.06 * inputs["grounding_support"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["identity_discontinuity"] = self._clip01(
            0.38 * latents["self_state_compartmentalization"]
            + 0.20 * regional_state["pfc_control"]
            + 0.18 * regional_state["parietal_self_processing"]
            + 0.10 * regional_state["amygdala"]
        )
        symptoms["dissociative_amnesia"] = self._clip01(
            0.40 * regional_state["hippocampus"]
            + 0.26 * latents["trauma_memory_compartmentalization"]
            + 0.10 * latents["temporal_lobe_dysrhythmia"]
            + 0.08 * regional_state["thalamus_proxy"]
        )
        symptoms["altered_consciousness"] = self._clip01(
            0.28 * regional_state["thalamus_proxy"]
            + 0.24 * latents["temporal_lobe_dysrhythmia"]
            + 0.18 * regional_state["parietal_self_processing"]
            + 0.14 * regional_state["pfc_control"]
        )
        symptoms["affective_lability"] = self._clip01(
            0.34 * regional_state["amygdala"]
            + 0.20 * regional_state["pfc_control"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.14 * latents["noradrenergic_hyperarousal"]
        )
        symptoms["hypervigilance"] = self._clip01(
            0.42 * latents["noradrenergic_hyperarousal"]
            + 0.20 * regional_state["amygdala"]
            + 0.16 * latents["hpa_axis_sensitization"]
            + 0.08 * inputs["trauma_cue_exposure"]
        )
        symptoms["sensory_motor_shifts"] = self._clip01(
            0.30 * latents["sensorimotor_dissociation"]
            + 0.18 * regional_state["basal_ganglia_proxy"]
            + 0.16 * regional_state["thalamus_proxy"]
            + 0.10 * regional_state["parietal_self_processing"]
        )
        symptoms["impulsivity_aggression"] = self._clip01(
            0.28 * latents["monoaminergic_dysregulation"]
            + 0.18 * regional_state["pfc_control"]
            + 0.12 * regional_state["amygdala"]
            + 0.08 * latents["noradrenergic_hyperarousal"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["trauma_dissociation_profile"] = self._clip01(
            float(symptoms[["identity_discontinuity", "dissociative_amnesia", "altered_consciousness"]].mean())
        )
        phenotypes["mnestic_fragmentation_profile"] = self._clip01(
            float(pd.Series([
                symptoms["dissociative_amnesia"],
                regional_state["hippocampus"],
                latents["trauma_memory_compartmentalization"],
            ]).mean())
        )
        phenotypes["hyperarousal_switching_profile"] = self._clip01(
            float(pd.Series([
                symptoms["hypervigilance"],
                symptoms["identity_discontinuity"],
                regional_state["amygdala"],
            ]).mean())
        )
        phenotypes["sensorimotor_state_shift_profile"] = self._clip01(
            float(pd.Series([
                symptoms["sensory_motor_shifts"],
                regional_state["basal_ganglia_proxy"],
                regional_state["thalamus_proxy"],
            ]).mean())
        )

        return {
            "inputs": inputs,
            "latents": latents.round(4),
            "regional_state": regional_state.round(4),
            "symptoms": symptoms.round(4),
            "phenotypes": phenotypes.round(4),
        }


if __name__ == "__main__":  # pragma: no cover - example usage
    try:
        model = DissociativeIdentityDisorderModel()
        scaffold = model.build()

        print("\n=== Nodes ===")
        print(scaffold["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

        print("\n=== Edges ===")
        print(scaffold["edges"][["source", "target", "relation", "did_change"]].to_string(index=False))

        print("\n=== Resolved regions ===")
        for key, region in scaffold["regions"].items():
            print(f"- {key}: {region.name}")

        print("\n=== Example receptor table: hippocampus ===")
        hippocampus_receptors = scaffold["receptors"].get("hippocampus", pd.DataFrame())
        if hippocampus_receptors.empty:
            print("No receptor fingerprint available for hippocampus in this environment.")
        else:
            print(hippocampus_receptors.head().to_string(index=False))

        print("\n=== Example gene table: amygdala ===")
        amygdala_genes = scaffold["genes"].get("amygdala", pd.DataFrame())
        if amygdala_genes.empty:
            print("No gene-expression table available for amygdala in this environment.")
        else:
            print(amygdala_genes.head().to_string(index=False))

        print("\n=== Example connectivity profile: pfc_control ===")
        pfc_conn = scaffold["connectivity_profiles"].get("pfc_control", pd.DataFrame())
        if pfc_conn.empty:
            print("No connectivity profile available for pfc_control in this environment.")
        else:
            print(pfc_conn.head(10).to_string(index=False))

        print("\n=== Example circuit connectivity ===")
        if scaffold["circuit_connectivity"].empty:
            print("No pairwise circuit connectivity values available in this environment.")
        else:
            print(scaffold["circuit_connectivity"].head(20).to_string(index=False))

        sim = model.simulate(
            early_life_trauma=0.90,
            chronic_stress_load=0.75,
            trauma_cue_exposure=0.65,
            genetic_vulnerability=0.40,
            depressive_comorbidity=0.55,
            ptsd_hyperarousal_liability=0.70,
            grounding_support=0.20,
        )

        print("\n=== Simulation: inputs ===")
        print(sim["inputs"].to_string())
        print("\n=== Simulation: latents ===")
        print(sim["latents"].to_string())
        print("\n=== Simulation: regional_state ===")
        print(sim["regional_state"].to_string())
        print("\n=== Simulation: symptoms ===")
        print(sim["symptoms"].to_string())
        print("\n=== Simulation: phenotypes ===")
        print(sim["phenotypes"].to_string())

        # Example optional coordinate assignment:
        # assignments = model.assign_mni_point((-24, -12, -20))
        # print(assignments.head())

    except ImportError as exc:
        print("This script requires siibra-python to run the atlas-backed portions of the scaffold.")
        print(exc)
