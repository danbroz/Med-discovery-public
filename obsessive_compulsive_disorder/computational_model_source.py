from __future__ import annotations

"""
Obsessive-Compulsive Disorder siibra scaffold.

This script turns a chapter-level biological summary of Obsessive-Compulsive
Disorder (OCD) into a conservative, atlas-grounded mechanistic scaffold using
siibra where available. It is intended for research prototyping and transparent
hypothesis exploration only. It is not a diagnostic, prognostic, or treatment
recommendation tool.

Core modeling choices from the chapter:
- OCD is modeled as an interaction between heritable vulnerability,
  temperament-linked anxiety/emotional reactivity, and monoaminergic plus
  stress-sensitive circuit dysregulation.
- Serotonergic and noradrenergic dysregulation are treated as primary latent
  biology because the chapter emphasizes SSRI/clomipramine response and shared
  monoamine substrates with anxiety and depressive states.
- Amygdala-driven threat tagging is treated as a key affective amplifier that
  loads orbitofrontal and anterior cingulate components of the CSTC loop.
- Executive control network weakness, centered here on a conservative DLPFC
  anchor, captures the chapter's emphasis on cognitive rigidity and weak
  top-down suppression of compulsive responses.
- The simulator is intentionally simple and acyclic:
  inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

The script degrades gracefully:
- If siibra is not installed, or if a feature/modality is unavailable, the
  atlas-backed parts stay empty instead of crashing.
- The simulator still runs even without atlas data.
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
    # Noradrenergic signaling
    "SLC6A2",
    "ADRA2A",
    # Broad monoamine modulation / trait-control biology
    "COMT",
    "MAOA",
    # Plasticity and stress responsivity
    "BDNF",
    "FKBP5",
    "NR3C1",
]


class ObsessiveCompulsiveDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Obsessive-Compulsive Disorder.

    Notes
    -----
    - This is a research scaffold, not a validated disease model.
    - Regional values in `simulate()` quantify dysregulation / burden rather
      than healthy activation.
    - Proxies are used where the chapter stays systems-level or where a stable
      Julich label may vary across environments.
    - The chapter emphasizes CSTC and executive-control dysfunction without
      naming a single unique striatal or thalamic parcel, so these remain
      explicit proxies.
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

        # Conservative region anchors based on the chapter's explicit anatomy
        # and strongly implied CSTC loop structure.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "LA (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4 left",
                "Fo3 left",
                "orbitofrontal cortex left",
                "orbitofrontal cortex",
                "orbitofrontal",
            ],
            "acc": [
                "Area p32 left",
                "Area a24pr left",
                "Area a24 left",
                "anterior cingulate cortex left",
                "anterior cingulate cortex",
                "cingulate cortex",
                "ACC",
            ],
            "dlpfc": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 8Av left",
                "dorsolateral prefrontal cortex left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "caudate left",
                "putamen left",
                "striatum left",
                "basal ganglia left",
                "basal ganglia",
                "striatum",
            ],
            "thalamus_proxy": [
                "mediodorsal thalamus left",
                "thalamus left",
                "thalamus",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Amygdala anchor for obsession-linked fear, anxiety, dread, and emotional threat tagging."
            ),
            "ofc": (
                "Orbitofrontal anchor for threat valuation, urgency signaling, and cortical CSTC hyperactivity."
            ),
            "acc": (
                "Anterior cingulate anchor for conflict monitoring, distress amplification, and compulsive drive."
            ),
            "dlpfc": (
                "Dorsolateral prefrontal / executive-control anchor for cognitive flexibility and top-down inhibition."
            ),
            "basal_ganglia_proxy": (
                "Basal-ganglia / striatal proxy capturing the strongly implied striatal arm of CSTC loop dysfunction."
            ),
            "thalamus_proxy": (
                "Thalamic proxy capturing the relay/integration arm of the strongly implied CSTC loop."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "polygenic_liability": (
                "Distributed heritable vulnerability affecting monoamine regulation, emotional reactivity, and control circuitry."
            ),
            "developmental_trait_vulnerability": (
                "Broad developmental vulnerability contributing to enduring OCD-related control and temperament risk."
            ),
            "anxious_temperament": (
                "Trait propensity toward fearfulness, anxiety, and exaggerated threat sensitivity."
            ),
            "irritability_emotional_reactivity": (
                "Quick-tempered or volatile affective style that lowers the threshold for distress under stress."
            ),
            "social_avoidant_trait_load": (
                "Trait social avoidance and withdrawal that may share anxiety-linked vulnerability with OCD."
            ),
            "environmental_stress_load": (
                "Stress burden that amplifies emotional reactivity and obsession-linked threat processing."
            ),
            "monoamine_treatment_support": (
                "Protective monoamine-targeted treatment support representing SSRI or clomipramine-like stabilization."
            ),
            "recovery_support": (
                "Protective structure, support, and emotion-regulation resources that buffer escalation of the OCD cycle."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": (
                "Maladaptive serotonin signaling consistent with the chapter's pharmacologic emphasis in OCD."
            ),
            "noradrenergic_dysregulation": (
                "Altered norepinephrine tone contributing to anxious arousal, urgency, and distress."
            ),
            "stress_sensitized_threat_response": (
                "Stress-amplified threat responsivity that increases the emotional force of obsessions."
            ),
            "amygdala_threat_bias": (
                "Pathological threat tagging of obsessional content by amygdala-centered fear circuitry."
            ),
            "top_down_control_failure": (
                "Weak prefrontal inhibition and impaired emotion regulation over intrusive urges and fear responses."
            ),
            "executive_control_network_weakening": (
                "Reduced executive control, set-shifting, and goal-directed suppression of prepotent responses."
            ),
            "cstc_loop_dysregulation": (
                "Dysregulation across cortico-striato-thalamo-cortical loops sustaining obsessions and compulsions."
            ),
            "obsessional_threat_tagging": (
                "Neutral thoughts or situations become marked as dangerous, urgent, or morally salient."
            ),
            "compulsive_neutralization_drive": (
                "Urgent drive to ritualize or neutralize distress generated by obsessions and fear circuitry."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "intrusive_obsessions": (
                "Persistent intrusive thoughts, images, or impulses with exaggerated threat or moral salience."
            ),
            "compulsive_ritualizing": (
                "Repetitive compulsive acts or mental rituals aimed at reducing distress or preventing harm."
            ),
            "anxiety_dread": (
                "Acute fear, dread, and anxious arousal linked to obsessional content."
            ),
            "cognitive_rigidity": (
                "Difficulty shifting sets, loosening rules, or disengaging from repetitive cognition."
            ),
            "inhibitory_control_failure": (
                "Failure to suppress prepotent compulsive responses despite awareness of irrationality."
            ),
            "affective_distress": (
                "Broader emotional suffering including anxiety, mood instability, and depressive burden secondary to the cycle."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "polygenic_liability",
                "target": "serotonergic_dysregulation",
                "relation": "distributed heritable vulnerability increases monoamine instability",
                "ocd_change": "increased",
            },
            {
                "source": "polygenic_liability",
                "target": "noradrenergic_dysregulation",
                "relation": "distributed heritable vulnerability increases anxious-arousal instability",
                "ocd_change": "increased",
            },
            {
                "source": "developmental_trait_vulnerability",
                "target": "top_down_control_failure",
                "relation": "developmental vulnerability weakens stable inhibitory control",
                "ocd_change": "increased",
            },
            {
                "source": "anxious_temperament",
                "target": "stress_sensitized_threat_response",
                "relation": "trait anxiety lowers threshold for exaggerated threat responses",
                "ocd_change": "increased",
            },
            {
                "source": "irritability_emotional_reactivity",
                "target": "stress_sensitized_threat_response",
                "relation": "emotional volatility amplifies distress reactivity",
                "ocd_change": "increased",
            },
            {
                "source": "social_avoidant_trait_load",
                "target": "obsessional_threat_tagging",
                "relation": "anxiety-linked social avoidance contributes to maladaptive threat interpretation",
                "ocd_change": "increased",
            },
            {
                "source": "environmental_stress_load",
                "target": "stress_sensitized_threat_response",
                "relation": "stress magnifies the affective force of obsessional content",
                "ocd_change": "increased",
            },
            {
                "source": "monoamine_treatment_support",
                "target": "serotonergic_dysregulation",
                "relation": "SSRI/clomipramine-like support partially stabilizes serotonin signaling",
                "ocd_change": "decreased",
            },
            {
                "source": "monoamine_treatment_support",
                "target": "noradrenergic_dysregulation",
                "relation": "monoamine-targeted treatment dampens anxious arousal burden",
                "ocd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "top_down_control_failure",
                "relation": "supportive regulation resources improve control over intrusive urges",
                "ocd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "stress_sensitized_threat_response",
                "relation": "support buffers stress-amplified threat escalation",
                "ocd_change": "decreased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "cstc_loop_dysregulation",
                "relation": "monoamine dysregulation destabilizes obsessive-compulsive circuit balance",
                "ocd_change": "increased",
            },
            {
                "source": "noradrenergic_dysregulation",
                "target": "cstc_loop_dysregulation",
                "relation": "anxious arousal biases loop activity toward urgency and repetitive action",
                "ocd_change": "increased",
            },
            {
                "source": "stress_sensitized_threat_response",
                "target": "amygdala_threat_bias",
                "relation": "stress-sensitive fear processing heightens amygdala threat tagging",
                "ocd_change": "increased",
            },
            {
                "source": "amygdala_threat_bias",
                "target": "obsessional_threat_tagging",
                "relation": "fear circuitry tags neutral thoughts and situations as dangerous and urgent",
                "ocd_change": "increased",
            },
            {
                "source": "amygdala_threat_bias",
                "target": "cstc_loop_dysregulation",
                "relation": "amygdala hyperreactivity loads orbitofrontal and cingulate components of the loop",
                "ocd_change": "increased",
            },
            {
                "source": "top_down_control_failure",
                "target": "executive_control_network_weakening",
                "relation": "weak prefrontal regulation impairs executive control and set shifting",
                "ocd_change": "increased",
            },
            {
                "source": "top_down_control_failure",
                "target": "compulsive_neutralization_drive",
                "relation": "insufficient inhibition allows compulsive neutralization urges to dominate behavior",
                "ocd_change": "increased",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "ofc",
                "relation": "loop dysregulation increases orbitofrontal urgency and valuation burden",
                "ocd_change": "hyperactive",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "acc",
                "relation": "loop dysregulation heightens cingulate conflict and distress processing",
                "ocd_change": "hyperactive",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "loop dysregulation burdens the striatal action-selection arm of compulsive responding",
                "ocd_change": "dysregulated",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "thalamus_proxy",
                "relation": "loop dysregulation burdens thalamic relay and re-entry dynamics",
                "ocd_change": "dysregulated",
            },
            {
                "source": "amygdala_threat_bias",
                "target": "amygdala",
                "relation": "fear bias heightens amygdala reactivity to obsessional content",
                "ocd_change": "hyperreactive",
            },
            {
                "source": "executive_control_network_weakening",
                "target": "dlpfc",
                "relation": "executive-network weakening impairs DLPFC-mediated cognitive flexibility",
                "ocd_change": "dysregulated",
            },
            {
                "source": "obsessional_threat_tagging",
                "target": "intrusive_obsessions",
                "relation": "threat-tagged intrusions become persistent obsessions",
                "ocd_change": "increased",
            },
            {
                "source": "compulsive_neutralization_drive",
                "target": "compulsive_ritualizing",
                "relation": "neutralization urgency drives repetitive rituals and checking-like behavior",
                "ocd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anxiety_dread",
                "relation": "amygdala hyperreactivity produces fear, dread, and urgent anxiety",
                "ocd_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "cognitive_rigidity",
                "relation": "executive-control weakness contributes to poor set shifting and rigidity",
                "ocd_change": "increased",
            },
            {
                "source": "ofc",
                "target": "intrusive_obsessions",
                "relation": "orbitofrontal overvaluation amplifies obsessional salience and urgency",
                "ocd_change": "increased",
            },
            {
                "source": "acc",
                "target": "compulsive_ritualizing",
                "relation": "conflict-monitoring and distress signals increase ritual pressure",
                "ocd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "compulsive_ritualizing",
                "relation": "striatal action-selection dysregulation supports repetitive enactment of rituals",
                "ocd_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "inhibitory_control_failure",
                "relation": "weak executive control impairs suppression of compulsive responses",
                "ocd_change": "increased",
            },
            {
                "source": "intrusive_obsessions",
                "target": "affective_distress",
                "relation": "persistent intrusions produce broad emotional suffering",
                "ocd_change": "increased",
            },
            {
                "source": "compulsive_ritualizing",
                "target": "affective_distress",
                "relation": "repetitive rituals maintain distress, exhaustion, and mood burden",
                "ocd_change": "increased",
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
            "amygdala",
            "orbitofrontal cortex",
            "anterior cingulate cortex",
            "prefrontal cortex",
            "thalamus",
            "basal ganglia",
            "striatum",
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
        polygenic_liability: float = 0.50,
        developmental_trait_vulnerability: float = 0.35,
        anxious_temperament: float = 0.65,
        irritability_emotional_reactivity: float = 0.45,
        social_avoidant_trait_load: float = 0.35,
        environmental_stress_load: float = 0.55,
        monoamine_treatment_support: float = 0.20,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Values are clipped to [0, 1]. Higher regional-state values mean greater
        dysregulation / burden in that circuit node.
        """
        inputs = pd.Series(
            {
                "polygenic_liability": self._clip01(polygenic_liability),
                "developmental_trait_vulnerability": self._clip01(developmental_trait_vulnerability),
                "anxious_temperament": self._clip01(anxious_temperament),
                "irritability_emotional_reactivity": self._clip01(irritability_emotional_reactivity),
                "social_avoidant_trait_load": self._clip01(social_avoidant_trait_load),
                "environmental_stress_load": self._clip01(environmental_stress_load),
                "monoamine_treatment_support": self._clip01(monoamine_treatment_support),
                "recovery_support": self._clip01(recovery_support),
            },
            name="value",
        )

        # Inputs -> latent biology
        serotonergic_dysregulation = self._clip01(
            0.35 * inputs["polygenic_liability"]
            + 0.20 * inputs["anxious_temperament"]
            + 0.15 * inputs["environmental_stress_load"]
            + 0.10 * inputs["developmental_trait_vulnerability"]
            + 0.10 * inputs["social_avoidant_trait_load"]
            - 0.30 * inputs["monoamine_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        noradrenergic_dysregulation = self._clip01(
            0.30 * inputs["anxious_temperament"]
            + 0.20 * inputs["environmental_stress_load"]
            + 0.15 * inputs["polygenic_liability"]
            + 0.10 * inputs["irritability_emotional_reactivity"]
            + 0.10 * inputs["social_avoidant_trait_load"]
            - 0.20 * inputs["monoamine_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        stress_sensitized_threat_response = self._clip01(
            0.30 * inputs["environmental_stress_load"]
            + 0.25 * inputs["anxious_temperament"]
            + 0.20 * inputs["irritability_emotional_reactivity"]
            + 0.10 * inputs["social_avoidant_trait_load"]
            + 0.10 * inputs["polygenic_liability"]
            - 0.10 * inputs["recovery_support"]
        )

        amygdala_threat_bias = self._clip01(
            0.35 * stress_sensitized_threat_response
            + 0.20 * inputs["anxious_temperament"]
            + 0.15 * inputs["irritability_emotional_reactivity"]
            + 0.10 * serotonergic_dysregulation
            + 0.10 * noradrenergic_dysregulation
            - 0.05 * inputs["recovery_support"]
        )

        top_down_control_failure = self._clip01(
            0.25 * inputs["developmental_trait_vulnerability"]
            + 0.20 * inputs["polygenic_liability"]
            + 0.20 * stress_sensitized_threat_response
            + 0.15 * serotonergic_dysregulation
            + 0.10 * noradrenergic_dysregulation
            + 0.05 * inputs["irritability_emotional_reactivity"]
            - 0.15 * inputs["monoamine_treatment_support"]
            - 0.15 * inputs["recovery_support"]
        )

        executive_control_network_weakening = self._clip01(
            0.35 * top_down_control_failure
            + 0.20 * inputs["developmental_trait_vulnerability"]
            + 0.15 * stress_sensitized_threat_response
            + 0.10 * inputs["social_avoidant_trait_load"]
            + 0.10 * inputs["anxious_temperament"]
            - 0.10 * inputs["recovery_support"]
        )

        cstc_loop_dysregulation = self._clip01(
            0.25 * serotonergic_dysregulation
            + 0.20 * noradrenergic_dysregulation
            + 0.20 * amygdala_threat_bias
            + 0.20 * top_down_control_failure
            + 0.10 * stress_sensitized_threat_response
            - 0.10 * inputs["monoamine_treatment_support"]
        )

        obsessional_threat_tagging = self._clip01(
            0.30 * amygdala_threat_bias
            + 0.25 * cstc_loop_dysregulation
            + 0.15 * inputs["anxious_temperament"]
            + 0.10 * inputs["social_avoidant_trait_load"]
            + 0.10 * stress_sensitized_threat_response
        )

        compulsive_neutralization_drive = self._clip01(
            0.35 * cstc_loop_dysregulation
            + 0.25 * top_down_control_failure
            + 0.15 * obsessional_threat_tagging
            + 0.10 * executive_control_network_weakening
            + 0.05 * amygdala_threat_bias
        )

        latents = pd.Series(
            {
                "serotonergic_dysregulation": serotonergic_dysregulation,
                "noradrenergic_dysregulation": noradrenergic_dysregulation,
                "stress_sensitized_threat_response": stress_sensitized_threat_response,
                "amygdala_threat_bias": amygdala_threat_bias,
                "top_down_control_failure": top_down_control_failure,
                "executive_control_network_weakening": executive_control_network_weakening,
                "cstc_loop_dysregulation": cstc_loop_dysregulation,
                "obsessional_threat_tagging": obsessional_threat_tagging,
                "compulsive_neutralization_drive": compulsive_neutralization_drive,
            },
            name="value",
        )

        # Latent biology -> regional dysregulation burden
        regional_state = pd.Series(
            {
                "amygdala": self._clip01(
                    0.40 * amygdala_threat_bias
                    + 0.25 * stress_sensitized_threat_response
                    + 0.15 * obsessional_threat_tagging
                    + 0.10 * inputs["anxious_temperament"]
                ),
                "ofc": self._clip01(
                    0.35 * cstc_loop_dysregulation
                    + 0.20 * obsessional_threat_tagging
                    + 0.15 * amygdala_threat_bias
                    + 0.10 * serotonergic_dysregulation
                    + 0.05 * compulsive_neutralization_drive
                    - 0.10 * inputs["monoamine_treatment_support"]
                ),
                "acc": self._clip01(
                    0.30 * cstc_loop_dysregulation
                    + 0.20 * amygdala_threat_bias
                    + 0.20 * compulsive_neutralization_drive
                    + 0.10 * stress_sensitized_threat_response
                    + 0.10 * top_down_control_failure
                ),
                "dlpfc": self._clip01(
                    0.35 * executive_control_network_weakening
                    + 0.25 * top_down_control_failure
                    + 0.10 * cstc_loop_dysregulation
                    + 0.10 * stress_sensitized_threat_response
                    - 0.15 * inputs["recovery_support"]
                    - 0.10 * inputs["monoamine_treatment_support"]
                ),
                "basal_ganglia_proxy": self._clip01(
                    0.35 * cstc_loop_dysregulation
                    + 0.25 * compulsive_neutralization_drive
                    + 0.15 * serotonergic_dysregulation
                    + 0.10 * noradrenergic_dysregulation
                    - 0.10 * inputs["monoamine_treatment_support"]
                ),
                "thalamus_proxy": self._clip01(
                    0.30 * cstc_loop_dysregulation
                    + 0.20 * compulsive_neutralization_drive
                    + 0.15 * amygdala_threat_bias
                    + 0.10 * noradrenergic_dysregulation
                    + 0.10 * stress_sensitized_threat_response
                ),
            },
            name="value",
        )

        # Regional dysregulation -> symptoms
        intrusive_obsessions = self._clip01(
            0.35 * obsessional_threat_tagging
            + 0.20 * regional_state["ofc"]
            + 0.20 * regional_state["amygdala"]
            + 0.10 * regional_state["acc"]
            + 0.10 * inputs["anxious_temperament"]
        )

        compulsive_ritualizing = self._clip01(
            0.35 * compulsive_neutralization_drive
            + 0.20 * regional_state["basal_ganglia_proxy"]
            + 0.15 * regional_state["acc"]
            + 0.10 * regional_state["ofc"]
            + 0.10 * regional_state["dlpfc"]
        )

        anxiety_dread = self._clip01(
            0.35 * regional_state["amygdala"]
            + 0.20 * stress_sensitized_threat_response
            + 0.20 * obsessional_threat_tagging
            + 0.10 * regional_state["acc"]
            + 0.05 * noradrenergic_dysregulation
        )

        cognitive_rigidity = self._clip01(
            0.35 * executive_control_network_weakening
            + 0.20 * regional_state["dlpfc"]
            + 0.15 * cstc_loop_dysregulation
            + 0.10 * regional_state["ofc"]
            + 0.10 * top_down_control_failure
        )

        inhibitory_control_failure = self._clip01(
            0.35 * top_down_control_failure
            + 0.20 * regional_state["dlpfc"]
            + 0.20 * regional_state["basal_ganglia_proxy"]
            + 0.10 * regional_state["acc"]
            + 0.05 * compulsive_neutralization_drive
            - 0.05 * inputs["recovery_support"]
        )

        affective_distress = self._clip01(
            0.25 * anxiety_dread
            + 0.20 * intrusive_obsessions
            + 0.20 * compulsive_ritualizing
            + 0.10 * regional_state["amygdala"]
            + 0.10 * serotonergic_dysregulation
            + 0.05 * inputs["social_avoidant_trait_load"]
        )

        symptoms = pd.Series(
            {
                "intrusive_obsessions": intrusive_obsessions,
                "compulsive_ritualizing": compulsive_ritualizing,
                "anxiety_dread": anxiety_dread,
                "cognitive_rigidity": cognitive_rigidity,
                "inhibitory_control_failure": inhibitory_control_failure,
                "affective_distress": affective_distress,
            },
            name="value",
        )

        # Symptom bundles / phenotype summaries
        anxious_obsessional_profile = self._clip01(
            0.35 * intrusive_obsessions
            + 0.30 * anxiety_dread
            + 0.15 * regional_state["amygdala"]
            + 0.10 * regional_state["ofc"]
        )

        compulsive_neutralization_profile = self._clip01(
            0.35 * compulsive_ritualizing
            + 0.25 * compulsive_neutralization_drive
            + 0.15 * regional_state["basal_ganglia_proxy"]
            + 0.10 * regional_state["acc"]
        )

        rigid_control_failure_profile = self._clip01(
            0.30 * cognitive_rigidity
            + 0.30 * inhibitory_control_failure
            + 0.15 * regional_state["dlpfc"]
            + 0.10 * top_down_control_failure
        )

        emotion_tagged_ocd_profile = self._clip01(
            0.30 * amygdala_threat_bias
            + 0.25 * intrusive_obsessions
            + 0.20 * affective_distress
            + 0.10 * regional_state["amygdala"]
            + 0.10 * regional_state["acc"]
        )

        global_ocd_burden = self._clip01(
            0.20 * intrusive_obsessions
            + 0.20 * compulsive_ritualizing
            + 0.18 * anxiety_dread
            + 0.15 * cognitive_rigidity
            + 0.15 * inhibitory_control_failure
            + 0.12 * affective_distress
        )

        phenotypes = pd.Series(
            {
                "anxious_obsessional_profile": anxious_obsessional_profile,
                "compulsive_neutralization_profile": compulsive_neutralization_profile,
                "rigid_control_failure_profile": rigid_control_failure_profile,
                "emotion_tagged_ocd_profile": emotion_tagged_ocd_profile,
                "global_ocd_burden": global_ocd_burden,
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
    model = ObsessiveCompulsiveDisorderModel()
    built = model.build()

    print("\n=== OCD NODES (head) ===")
    print(built["nodes"].head(15).to_string(index=False))

    print("\n=== OCD EDGES (head) ===")
    print(built["edges"].head(15).to_string(index=False))

    print("\n=== REGION RESOLUTION SUMMARY ===")
    region_cols = [
        "key",
        "label",
        "atlas_region",
        "region_identifier",
        "centroid_mni",
        "volume_mm3",
        "feature_summary",
    ]
    print(
        built["nodes"]
        .loc[built["nodes"]["node_type"] == "region", region_cols]
        .to_string(index=False)
    )

    if built["receptors"].get("ofc") is not None and not built["receptors"]["ofc"].empty:
        print("\n=== OFC RECEPTOR PROFILE (head) ===")
        print(built["receptors"]["ofc"].head().to_string(index=False))
    else:
        print("\nNo OFC receptor fingerprint available in this environment.")

    if built["genes"].get("acc") is not None and not built["genes"]["acc"].empty:
        print("\n=== ACC GENE PROFILE ===")
        print(built["genes"]["acc"].to_string(index=False))
    else:
        print("\nNo ACC gene-expression profile available in this environment.")

    if built["connectivity_profiles"].get("dlpfc") is not None and not built["connectivity_profiles"]["dlpfc"].empty:
        print("\n=== DLPFC CONNECTIVITY PROFILE (head) ===")
        print(built["connectivity_profiles"]["dlpfc"].head(10).to_string(index=False))
    else:
        print("\nNo DLPFC connectivity profile available in this environment.")

    if not built["circuit_connectivity"].empty:
        print("\n=== WITHIN-MODEL CIRCUIT CONNECTIVITY ===")
        print(built["circuit_connectivity"].to_string())
    else:
        print("\nNo within-model circuit connectivity matrix available in this environment.")

    sim = model.simulate(
        polygenic_liability=0.62,
        developmental_trait_vulnerability=0.45,
        anxious_temperament=0.78,
        irritability_emotional_reactivity=0.52,
        social_avoidant_trait_load=0.40,
        environmental_stress_load=0.68,
        monoamine_treatment_support=0.25,
        recovery_support=0.20,
    )

    print("\n=== SIMULATION: INPUTS ===")
    print(sim["inputs"].to_string())
    print("\n=== SIMULATION: LATENT BIOLOGY ===")
    print(sim["latents"].sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: REGIONAL STATE ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example optional atlas utilities for siibra-enabled environments:
    # print(model.suggest_regions("orbitofrontal"))
    # print(model.assign_mni_point((-8, 42, -12)).head())
    # mask = model.region_mask("ofc")
