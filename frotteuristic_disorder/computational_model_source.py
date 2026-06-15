
from __future__ import annotations

"""
Frotteuristic Disorder atlas-grounded siibra scaffold.

This script translates a chapter-level biological discussion of Frotteuristic
Disorder into a transparent, research-oriented mechanistic graph. The chapter
does not provide direct disorder-specific neurobiological experiments, so the
model is intentionally inferential and conservative. It focuses on the chapter's
main biological themes:

- serotonergic and dopaminergic influences on impulsivity and reward seeking,
- stress and HPA-axis-related lowering of inhibitory thresholds,
- prefrontal hypo-function affecting orbitofrontal and dorsolateral control,
- amygdala-centered salience/arousal hyper-reactivity,
- ventral-striatal / reward-circuit involvement,
- cue-primed readiness for repeated transgressive behavior in public settings.

Important:
- This is a research scaffold, not a diagnostic or treatment tool.
- The model describes pathology and dysregulation. It should not be read as
  normalizing or excusing nonconsensual sexual behavior.
- Higher simulated values indicate greater dysregulation burden or symptom
  pressure unless explicitly noted otherwise.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_FROTTEURISTIC_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor 1A
    "HTR2A",    # serotonin receptor 2A
    "SLC6A3",   # dopamine transporter
    "DRD2",     # dopamine receptor D2
    "DRD4",     # dopamine receptor D4
    "COMT",     # catecholamine metabolism / executive control
    "MAOA",     # monoamine degradation / impulsive aggression literature
    "CRHR1",    # stress-response signaling
    "FKBP5",    # stress sensitivity / trauma-related regulation
    "NR3C1",    # glucocorticoid receptor
    "BDNF",     # plasticity and stress-response adaptation
]


class FrotteuristicDisorderModel:
    """
    Atlas-grounded research scaffold for Frotteuristic Disorder.

    The motivating chapter frames the disorder as an interaction among
    dysregulated sexual arousal, reward learning, impulse-control failure,
    stress-linked threshold lowering, and abnormal salience/cue processing.
    Because the chapter is systems-level rather than based on a settled lesion or
    parcel model, this scaffold deliberately combines atlas-backed regions with
    explicit proxy nodes.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:
            raise ImportError(
                "siibra is required to use this scaffold. Install it in your Python "
                "environment before running the model."
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

        # Conservative atlas anchors. OFC and dlPFC are named explicitly.
        # Amygdala is directly named and central to the chapter. The nucleus
        # accumbens / reward circuitry is kept as a ventral-striatal proxy.
        self.region_candidates: Dict[str, List[str]] = {
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
                "orbitofrontal",
            ],
            "dlpfc": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9 left",
                "Area 45 left",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens left",
                "caudate left",
                "putamen left",
                "striatum",
                "basal ganglia",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "ofc": (
                "Orbitofrontal control node representing valuation, inhibition of socially inappropriate acts, "
                "and consequence-sensitive behavioral stopping."
            ),
            "dlpfc": (
                "Dorsolateral prefrontal control node representing executive control, working regulation, "
                "and response inhibition."
            ),
            "amygdala": (
                "Amygdala node representing salience detection, emotional arousal, and chapter-linked hyper-reactivity."
            ),
            "ventral_striatum_proxy": (
                "Proxy for nucleus accumbens / ventral-striatal reward circuitry involved in incentive salience "
                "and pathological reinforcement."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_impulsivity_reward_liability": (
                "Inherited vulnerability affecting serotonergic impulse control and dopaminergic reward seeking."
            ),
            "early_life_stress_trauma": (
                "Early adverse experiences that may shape later stress responsivity and gene-environment vulnerability."
            ),
            "chronic_stress_tension": (
                "Sustained stress or emotional tension that lowers inhibitory thresholds and raises action pressure."
            ),
            "cue_rich_crowded_context_exposure": (
                "Exposure to public, crowded, or otherwise trigger-rich situations associated with the behavior."
            ),
            "pathological_reinforcement_history": (
                "Prior reinforcement in which the act or fantasy transiently reduces tension or provides reward, "
                "thereby sensitizing repetition."
            ),
            "executive_control_support": (
                "Protective regulatory support from structure, monitoring, treatment engagement, or other inhibitory scaffolding."
            ),
            "stress_regulation_support": (
                "Protective stress-reduction support that lowers arousal and helps keep thresholds from collapsing."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_stress_sensitization": (
                "Stress-hormone burden that biases the system toward arousal, impulsivity, and weaker control."
            ),
            "serotonergic_impulse_control_deficit": (
                "Serotonergic dysregulation plausibly reducing inhibitory control and increasing impulsive responding."
            ),
            "dopaminergic_reward_salience_bias": (
                "Reward/salience bias that increases the motivational pull of cues and pathological reinforcement."
            ),
            "frontolimbic_disconnection": (
                "Weakened functional coupling between prefrontal control systems and limbic arousal systems."
            ),
            "prefrontal_inhibitory_failure": (
                "Failure of orbitofrontal and dorsolateral executive inhibition over inappropriate action tendencies."
            ),
            "amygdala_salience_hyperreactivity": (
                "Exaggerated salience and emotional arousal response to internal stress states or external trigger cues."
            ),
            "cue_primed_action_readiness": (
                "A lowered threshold state in which cues more easily precipitate transgressive behavioral sequences."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "recurrent_frotteuristic_urges": (
                "Recurrent, intense urges or fantasies linked to nonconsensual rubbing/touching behavior."
            ),
            "cue_triggered_urge_escalation": (
                "Rapid intensification of urges in trigger-rich or stress-loaded contexts."
            ),
            "boundary_violating_contact_propensity": (
                "Increased likelihood of acting on urges in ways that violate consent and social boundaries."
            ),
            "compulsive_repetition": (
                "Repetition pressure supported by reinforcement, salience bias, and failed inhibitory control."
            ),
            "secrecy_escape_orientation": (
                "Preference for covert/public conditions with perceived opportunity to avoid detection."
            ),
            "distress_impairment": (
                "Clinically significant distress, preoccupation, or functional impairment tied to the syndrome."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_impulsivity_reward_liability",
                "target": "serotonergic_impulse_control_deficit",
                "relation": "raises inherited vulnerability in impulse-control systems",
                "frotteuristic_change": "increased",
            },
            {
                "source": "genetic_impulsivity_reward_liability",
                "target": "dopaminergic_reward_salience_bias",
                "relation": "raises inherited vulnerability in reward-seeking and salience systems",
                "frotteuristic_change": "increased",
            },
            {
                "source": "early_life_stress_trauma",
                "target": "hpa_axis_stress_sensitization",
                "relation": "can increase long-term stress responsivity and threshold lowering",
                "frotteuristic_change": "increased",
            },
            {
                "source": "early_life_stress_trauma",
                "target": "frontolimbic_disconnection",
                "relation": "can weaken prefrontal-limbic coupling through developmental stress effects",
                "frotteuristic_change": "increased",
            },
            {
                "source": "chronic_stress_tension",
                "target": "hpa_axis_stress_sensitization",
                "relation": "elevates stress-hormone burden and arousal pressure",
                "frotteuristic_change": "increased",
            },
            {
                "source": "cue_rich_crowded_context_exposure",
                "target": "amygdala_salience_hyperreactivity",
                "relation": "provides salient external triggers for emotional and motivational activation",
                "frotteuristic_change": "increased",
            },
            {
                "source": "cue_rich_crowded_context_exposure",
                "target": "cue_primed_action_readiness",
                "relation": "potentiates trigger-linked readiness for action",
                "frotteuristic_change": "increased",
            },
            {
                "source": "pathological_reinforcement_history",
                "target": "dopaminergic_reward_salience_bias",
                "relation": "strengthens incentive salience through prior reward or tension relief",
                "frotteuristic_change": "increased",
            },
            {
                "source": "pathological_reinforcement_history",
                "target": "cue_primed_action_readiness",
                "relation": "sensitizes the action sequence through repetition and conditioned expectancy",
                "frotteuristic_change": "increased",
            },
            {
                "source": "executive_control_support",
                "target": "prefrontal_inhibitory_failure",
                "relation": "supports stronger inhibitory gating and reduces discontrol",
                "frotteuristic_change": "decreased",
            },
            {
                "source": "stress_regulation_support",
                "target": "hpa_axis_stress_sensitization",
                "relation": "buffers chronic arousal and lowers stress-linked threshold collapse",
                "frotteuristic_change": "decreased",
            },
            {
                "source": "hpa_axis_stress_sensitization",
                "target": "prefrontal_inhibitory_failure",
                "relation": "impairs executive control under stress",
                "frotteuristic_change": "increased",
            },
            {
                "source": "hpa_axis_stress_sensitization",
                "target": "amygdala_salience_hyperreactivity",
                "relation": "raises emotional and motivational responsiveness",
                "frotteuristic_change": "increased",
            },
            {
                "source": "serotonergic_impulse_control_deficit",
                "target": "prefrontal_inhibitory_failure",
                "relation": "reduces the capacity to suppress inappropriate action tendencies",
                "frotteuristic_change": "increased",
            },
            {
                "source": "dopaminergic_reward_salience_bias",
                "target": "cue_primed_action_readiness",
                "relation": "amplifies the pull of cues and anticipated reward",
                "frotteuristic_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "prefrontal_inhibitory_failure",
                "relation": "reduces top-down regulation of arousal and impulses",
                "frotteuristic_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "amygdala_salience_hyperreactivity",
                "relation": "permits greater unmodulated limbic reactivity",
                "frotteuristic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "ofc",
                "relation": "increases dysfunction burden in orbitofrontal inhibitory valuation circuitry",
                "frotteuristic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "dlpfc",
                "relation": "increases dysfunction burden in dorsolateral executive-control circuitry",
                "frotteuristic_change": "increased",
            },
            {
                "source": "amygdala_salience_hyperreactivity",
                "target": "amygdala",
                "relation": "raises limbic arousal burden in trigger and stress states",
                "frotteuristic_change": "increased",
            },
            {
                "source": "dopaminergic_reward_salience_bias",
                "target": "ventral_striatum_proxy",
                "relation": "increases dysregulation burden in reward and reinforcement circuitry",
                "frotteuristic_change": "increased",
            },
            {
                "source": "cue_primed_action_readiness",
                "target": "recurrent_frotteuristic_urges",
                "relation": "supports rapid emergence and strengthening of urges",
                "frotteuristic_change": "increased",
            },
            {
                "source": "amygdala_salience_hyperreactivity",
                "target": "cue_triggered_urge_escalation",
                "relation": "accelerates urge intensity in trigger-rich conditions",
                "frotteuristic_change": "increased",
            },
            {
                "source": "cue_primed_action_readiness",
                "target": "cue_triggered_urge_escalation",
                "relation": "lowers the provocation needed for escalation",
                "frotteuristic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "boundary_violating_contact_propensity",
                "relation": "weakens inhibition of behavior that breaches consent and social boundaries",
                "frotteuristic_change": "increased",
            },
            {
                "source": "recurrent_frotteuristic_urges",
                "target": "boundary_violating_contact_propensity",
                "relation": "provides motivational pressure toward acting on urges",
                "frotteuristic_change": "increased",
            },
            {
                "source": "dopaminergic_reward_salience_bias",
                "target": "compulsive_repetition",
                "relation": "supports repetition through wanting and reinforcement pressure",
                "frotteuristic_change": "increased",
            },
            {
                "source": "pathological_reinforcement_history",
                "target": "compulsive_repetition",
                "relation": "maintains repetition through learned tension relief and reward expectancy",
                "frotteuristic_change": "increased",
            },
            {
                "source": "cue_rich_crowded_context_exposure",
                "target": "secrecy_escape_orientation",
                "relation": "favors environments allowing covert action and perceived escape",
                "frotteuristic_change": "increased",
            },
            {
                "source": "boundary_violating_contact_propensity",
                "target": "distress_impairment",
                "relation": "increases the overall burden and consequences of the syndrome",
                "frotteuristic_change": "increased",
            },
            {
                "source": "compulsive_repetition",
                "target": "distress_impairment",
                "relation": "increases burden through repetition, preoccupation, and functional impact",
                "frotteuristic_change": "increased",
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
            "prefrontal cortex",
            "orbitofrontal cortex",
            "striatum",
            "basal ganglia",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                region = self.atlas.get_region(spec, parcellation=self.parcellation)
                if region is not None:
                    return region
            except Exception:
                pass
            try:
                region = self.parcellation.get_region(spec)
                if region is not None:
                    return region
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
        centroid_xyz = tuple(float(v) for v in centroid) if centroid is not None else None
        volume = getattr(main, "volume", None)
        volume_mm3 = float(volume) if volume is not None else None
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

    @staticmethod
    def _numeric_scalar(value: Any) -> Optional[float]:
        if isinstance(value, pd.DataFrame):
            numeric = value.apply(pd.to_numeric, errors="coerce").stack().dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").dropna()
            return float(numeric.mean()) if not numeric.empty else None
        try:
            return float(value)
        except Exception:
            return None

    def _collapse_connectivity_selection(self, selected: Any, axis: str) -> Optional[pd.Series]:
        if isinstance(selected, pd.DataFrame):
            numeric = selected.apply(pd.to_numeric, errors="coerce")
            collapsed = numeric.mean(axis=0 if axis == "index" else 1)
        elif isinstance(selected, pd.Series):
            collapsed = pd.to_numeric(selected, errors="coerce")
        else:
            return None

        collapsed = collapsed.dropna()
        if collapsed.empty:
            return None
        return collapsed

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
            selected = matrix.loc[label] if axis == "index" else matrix[label]
            series = self._collapse_connectivity_selection(selected, axis=axis)
            if series is None or series.empty:
                return pd.DataFrame()
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = (
                df[df["connected_region"] != region.name]
                .groupby("connected_region", as_index=False)["value"]
                .mean()
                .sort_values("value", ascending=False)
                .head(max_rows)
            )
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return pairwise connectivity values among the scaffold's resolved circuit nodes.
        Missing or unresolved regions are skipped.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        matched_rows: Dict[str, Any] = {}
        matched_cols: Dict[str, Any] = {}

        for key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None:
                matched_rows[key] = row_label
            if col_label is not None:
                matched_cols[key] = col_label

        keys = list(self.region_objects.keys())
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                row_label = matched_rows.get(src)
                col_label = matched_cols.get(dst)
                if row_label is None or col_label is None:
                    continue
                try:
                    value = matrix.loc[row_label, col_label]
                except Exception:
                    continue

                scalar = self._numeric_scalar(value)
                if scalar is None:
                    continue

                rows.append(
                    {
                        "source_key": src,
                        "source_region": self.region_objects[src].name,
                        "target_key": dst,
                        "target_region": self.region_objects[dst].name,
                        "value": scalar,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_FROTTEURISTIC_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Build the atlas-grounded graph and collect multimodal summaries.
        """
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
            desc = self.region_descriptions.get(key, "Atlas-backed circuit node")

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
                        "description": f"{desc} Unresolved in this siibra environment; keep as an explicit proxy.",
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
                    "description": desc,
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

    def simulate(
        self,
        genetic_impulsivity_reward_liability: float = 0.55,
        early_life_stress_trauma: float = 0.35,
        chronic_stress_tension: float = 0.50,
        cue_rich_crowded_context_exposure: float = 0.50,
        pathological_reinforcement_history: float = 0.40,
        executive_control_support: float = 0.30,
        stress_regulation_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator for the scaffold.

        Inputs are clipped to 0..1. Protective variables subtract from
        dysregulation. The calculation is acyclic:
            inputs -> latent biology -> regional burden -> symptoms -> phenotypes
        """
        inputs = {
            "genetic_impulsivity_reward_liability": self._clip01(genetic_impulsivity_reward_liability),
            "early_life_stress_trauma": self._clip01(early_life_stress_trauma),
            "chronic_stress_tension": self._clip01(chronic_stress_tension),
            "cue_rich_crowded_context_exposure": self._clip01(cue_rich_crowded_context_exposure),
            "pathological_reinforcement_history": self._clip01(pathological_reinforcement_history),
            "executive_control_support": self._clip01(executive_control_support),
            "stress_regulation_support": self._clip01(stress_regulation_support),
        }

        latents = {
            "hpa_axis_stress_sensitization": self._clip01(
                0.40 * inputs["early_life_stress_trauma"]
                + 0.38 * inputs["chronic_stress_tension"]
                + 0.08 * inputs["cue_rich_crowded_context_exposure"]
                - 0.24 * inputs["stress_regulation_support"]
            ),
            "serotonergic_impulse_control_deficit": self._clip01(
                0.42 * inputs["genetic_impulsivity_reward_liability"]
                + 0.20 * inputs["early_life_stress_trauma"]
                + 0.14 * inputs["chronic_stress_tension"]
                - 0.16 * inputs["executive_control_support"]
            ),
            "dopaminergic_reward_salience_bias": self._clip01(
                0.36 * inputs["genetic_impulsivity_reward_liability"]
                + 0.30 * inputs["pathological_reinforcement_history"]
                + 0.18 * inputs["cue_rich_crowded_context_exposure"]
                + 0.08 * inputs["chronic_stress_tension"]
                - 0.14 * inputs["executive_control_support"]
            ),
        }

        latents["frontolimbic_disconnection"] = self._clip01(
            0.40 * inputs["early_life_stress_trauma"]
            + 0.24 * latents["hpa_axis_stress_sensitization"]
            + 0.16 * inputs["genetic_impulsivity_reward_liability"]
            - 0.18 * inputs["executive_control_support"]
            - 0.12 * inputs["stress_regulation_support"]
        )
        latents["prefrontal_inhibitory_failure"] = self._clip01(
            0.34 * latents["serotonergic_impulse_control_deficit"]
            + 0.28 * latents["hpa_axis_stress_sensitization"]
            + 0.20 * latents["frontolimbic_disconnection"]
            + 0.12 * latents["dopaminergic_reward_salience_bias"]
            + 0.08 * inputs["chronic_stress_tension"]
            - 0.28 * inputs["executive_control_support"]
        )
        latents["amygdala_salience_hyperreactivity"] = self._clip01(
            0.34 * latents["hpa_axis_stress_sensitization"]
            + 0.24 * latents["frontolimbic_disconnection"]
            + 0.18 * inputs["cue_rich_crowded_context_exposure"]
            + 0.14 * inputs["chronic_stress_tension"]
            + 0.06 * inputs["pathological_reinforcement_history"]
            - 0.14 * inputs["stress_regulation_support"]
        )
        latents["cue_primed_action_readiness"] = self._clip01(
            0.30 * latents["dopaminergic_reward_salience_bias"]
            + 0.26 * inputs["cue_rich_crowded_context_exposure"]
            + 0.18 * inputs["pathological_reinforcement_history"]
            + 0.16 * latents["amygdala_salience_hyperreactivity"]
            + 0.08 * inputs["chronic_stress_tension"]
            - 0.18 * inputs["executive_control_support"]
        )

        regional_state = {
            "ofc": self._clip01(
                0.50 * latents["prefrontal_inhibitory_failure"]
                + 0.20 * latents["frontolimbic_disconnection"]
                + 0.12 * latents["hpa_axis_stress_sensitization"]
                + 0.08 * inputs["chronic_stress_tension"]
                - 0.20 * inputs["executive_control_support"]
            ),
            "dlpfc": self._clip01(
                0.48 * latents["prefrontal_inhibitory_failure"]
                + 0.22 * latents["serotonergic_impulse_control_deficit"]
                + 0.12 * latents["frontolimbic_disconnection"]
                + 0.08 * inputs["chronic_stress_tension"]
                - 0.22 * inputs["executive_control_support"]
            ),
            "amygdala": self._clip01(
                0.52 * latents["amygdala_salience_hyperreactivity"]
                + 0.22 * latents["frontolimbic_disconnection"]
                + 0.10 * inputs["cue_rich_crowded_context_exposure"]
                - 0.10 * inputs["stress_regulation_support"]
            ),
            "ventral_striatum_proxy": self._clip01(
                0.48 * latents["dopaminergic_reward_salience_bias"]
                + 0.30 * latents["cue_primed_action_readiness"]
                + 0.10 * inputs["pathological_reinforcement_history"]
                - 0.12 * inputs["executive_control_support"]
            ),
        }

        symptoms = {
            "recurrent_frotteuristic_urges": self._clip01(
                0.38 * latents["cue_primed_action_readiness"]
                + 0.24 * regional_state["ventral_striatum_proxy"]
                + 0.18 * latents["amygdala_salience_hyperreactivity"]
                + 0.10 * inputs["cue_rich_crowded_context_exposure"]
                + 0.06 * inputs["pathological_reinforcement_history"]
            ),
            "cue_triggered_urge_escalation": self._clip01(
                0.34 * latents["amygdala_salience_hyperreactivity"]
                + 0.30 * latents["cue_primed_action_readiness"]
                + 0.18 * inputs["chronic_stress_tension"]
                + 0.10 * inputs["cue_rich_crowded_context_exposure"]
            ),
        }

        # Complete the symptoms dictionary in a visible, acyclic sequence.
        symptoms["boundary_violating_contact_propensity"] = self._clip01(
            0.34 * latents["prefrontal_inhibitory_failure"]
            + 0.26 * symptoms["cue_triggered_urge_escalation"]
            + 0.20 * symptoms["recurrent_frotteuristic_urges"]
            + 0.10 * regional_state["ventral_striatum_proxy"]
            - 0.12 * inputs["executive_control_support"]
        )
        symptoms["compulsive_repetition"] = self._clip01(
            0.32 * symptoms["recurrent_frotteuristic_urges"]
            + 0.30 * latents["dopaminergic_reward_salience_bias"]
            + 0.18 * inputs["pathological_reinforcement_history"]
            + 0.10 * latents["cue_primed_action_readiness"]
        )
        symptoms["secrecy_escape_orientation"] = self._clip01(
            0.38 * inputs["cue_rich_crowded_context_exposure"]
            + 0.24 * symptoms["boundary_violating_contact_propensity"]
            + 0.18 * latents["cue_primed_action_readiness"]
            + 0.10 * latents["prefrontal_inhibitory_failure"]
        )
        symptoms["distress_impairment"] = self._clip01(
            0.28 * symptoms["recurrent_frotteuristic_urges"]
            + 0.28 * symptoms["compulsive_repetition"]
            + 0.22 * symptoms["boundary_violating_contact_propensity"]
            + 0.12 * symptoms["secrecy_escape_orientation"]
            + 0.06 * inputs["chronic_stress_tension"]
        )

        phenotypes = {
            "frotteuristic_disorder_core_profile": self._clip01(
                (
                    symptoms["recurrent_frotteuristic_urges"]
                    + symptoms["boundary_violating_contact_propensity"]
                    + symptoms["compulsive_repetition"]
                    + symptoms["distress_impairment"]
                ) / 4.0
            ),
            "stress_primed_impulse_profile": self._clip01(
                (
                    latents["hpa_axis_stress_sensitization"]
                    + latents["amygdala_salience_hyperreactivity"]
                    + symptoms["cue_triggered_urge_escalation"]
                    + inputs["chronic_stress_tension"]
                ) / 4.0
            ),
            "boundary_violation_risk_profile": self._clip01(
                (
                    symptoms["boundary_violating_contact_propensity"]
                    + latents["prefrontal_inhibitory_failure"]
                    + symptoms["recurrent_frotteuristic_urges"]
                    + latents["cue_primed_action_readiness"]
                ) / 4.0
            ),
            "compulsive_repetition_profile": self._clip01(
                (
                    symptoms["compulsive_repetition"]
                    + latents["dopaminergic_reward_salience_bias"]
                    + inputs["pathological_reinforcement_history"]
                ) / 3.0
            ),
            "covert_public_trigger_profile": self._clip01(
                (
                    inputs["cue_rich_crowded_context_exposure"]
                    + symptoms["secrecy_escape_orientation"]
                    + latents["cue_primed_action_readiness"]
                ) / 3.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="inputs"),
            "latents": pd.Series(latents, name="latents"),
            "regional_state": pd.Series(regional_state, name="regional_state"),
            "symptoms": pd.Series(symptoms, name="symptoms"),
            "phenotypes": pd.Series(phenotypes, name="phenotypes"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI coordinate to Julich regions using a statistical map.
        """
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
                        space=self.atlas.get_space(self.assignment_space),
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break

        if "region" in assignments.columns:
            assignments = assignments.copy()
            assignments["region"] = assignments["region"].map(self._name_of)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a regional mask object for a resolved region node.

        Call `.fetch()` on the returned object to obtain the underlying image.
        Returns None for unresolved nodes.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype="labelled")
            except Exception:
                return None


if __name__ == "__main__":
    if siibra is None:
        print(
            "siibra is not installed in this environment. Install siibra, then rerun "
            "this script to build the atlas-grounded Frotteuristic Disorder scaffold."
        )
        raise SystemExit(0)

    model = FrotteuristicDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(built["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== EDGES ===")
    print(built["edges"][["source", "target", "relation", "frotteuristic_change"]].to_string(index=False))

    print("\n=== REGION RESOLUTION ===")
    if model.region_objects:
        for key, region in model.region_objects.items():
            print(f"{key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for node_key in ["amygdala", "ofc", "dlpfc", "ventral_striatum_proxy"]:
        receptor_df = built["receptors"].get(node_key, pd.DataFrame())
        gene_df = built["genes"].get(node_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(node_key, pd.DataFrame())

        print(f"\n=== FEATURES: {node_key} ===")
        print("Receptors:")
        print(receptor_df.head().to_string(index=False) if not receptor_df.empty else "<none>")
        print("Genes:")
        print(gene_df.head().to_string(index=False) if not gene_df.empty else "<none>")
        print("Connectivity:")
        print(conn_df.head().to_string(index=False) if not conn_df.empty else "<none>")

    circuit_df = built["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(12).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        genetic_impulsivity_reward_liability=0.65,
        early_life_stress_trauma=0.40,
        chronic_stress_tension=0.70,
        cue_rich_crowded_context_exposure=0.75,
        pathological_reinforcement_history=0.55,
        executive_control_support=0.30,
        stress_regulation_support=0.25,
    )

    print("\n=== SIMULATION: INPUTS ===")
    print(sim["inputs"].round(3).to_string())
    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: REGIONAL STATE ===")
    print(sim["regional_state"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].round(3).sort_values(ascending=False).to_string())

    # Example coordinate assignment for future use:
    # print(model.assign_mni_point((-22, -4, -18)).head())
    # mask = model.region_mask("amygdala")
    # if mask is not None:
    #     nii = mask.fetch()
