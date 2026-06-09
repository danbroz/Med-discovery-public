
from __future__ import annotations

"""
Premenstrual Dysphoric Disorder siibra scaffold.

This script turns a chapter-level biological summary of Premenstrual Dysphoric
Disorder (PMDD) into a conservative, atlas-grounded mechanistic scaffold using
siibra where available. It is intended for research prototyping and transparent
hypothesis exploration only. It is not a diagnostic, prognostic, or treatment
recommendation tool.

Core modeling choices from the chapter:
- PMDD is modeled primarily as an abnormal central nervous system sensitivity to
  otherwise normal cyclical fluctuations of ovarian steroid hormones rather than
  as a simple estrogen or progesterone deficiency/excess state.
- Estrogen/progesterone-linked modulation of serotonin, norepinephrine,
  dopamine, and GABA is represented as interacting latent biology rather than
  over-forcing each transmitter into a specific parcel.
- Region anchors are conservative: amygdala, hippocampus, hypothalamus proxy,
  orbitofrontal cortex proxy, and dorsolateral prefrontal cortex proxy, because
  the chapter names these structures or subregions directly.
- Stress-linked HPO/HPA coupling and luteal self-regulatory burden are modeled
  as important intermediates that help explain anxiety, irritability,
  concentration difficulty, and emotional lability.
- The simulator is intentionally simple and acyclic:
  inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

The script degrades gracefully:
- If siibra is not installed, or if a feature/modality is unavailable, the
  atlas-backed parts stay empty instead of crashing.
- The simulator still runs even without atlas data.

Source chapter themes encoded here include:
- trait vulnerability to ovarian steroid fluctuations,
- hormone-linked modulation of serotonin, dopamine, norepinephrine, and GABA,
- interaction of HPO and HPA axis processes,
- amygdala, hippocampus, hypothalamus, OFC, and DLPFC circuitry,
- luteal-phase impairment of top-down emotional and cognitive regulation.
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
    # Ovarian steroid sensitivity / receptor signaling
    "ESR1",
    "ESR2",
    "PGR",
    # Serotonergic regulation
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    # Noradrenergic regulation
    "SLC6A2",
    "ADRA2A",
    # Dopaminergic modulation
    "DRD2",
    "COMT",
    # GABAergic tone / neurosteroid sensitivity
    "GAD1",
    "GABRA2",
    "GABRD",
    # Stress responsivity / plasticity
    "NR3C1",
    "CRHR1",
    "BDNF",
]


class PremenstrualDysphoricDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Premenstrual Dysphoric Disorder.

    Notes
    -----
    - This is a research scaffold, not a validated disease model.
    - PMDD is represented as abnormal CNS sensitivity to normal ovarian steroid
      fluctuations, not as a simple absolute hormone-level abnormality.
    - Regional values in `simulate()` quantify dysregulation / burden rather
      than healthy activation.
    - Proxies are used where the chapter stays systems-level or where a stable
      Julich label may vary across environments.
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

        # Conservative region anchors from the chapter's explicitly named
        # circuitry. Proxies are preferred when the text stays systems-level.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "LA (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "CA left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "hypothalamus_proxy": [
                "hypothalamus left",
                "hypothalamus",
                "hypothalamic",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo4 left",
                "Area Fo3 left",
                "Fo4 left",
                "Fo3 left",
                "orbitofrontal cortex left",
                "orbitofrontal left",
                "orbitofrontal cortex",
                "orbitofrontal",
            ],
            "dlpfc": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 8Av left",
                "dorsolateral prefrontal cortex left",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Amygdala anchor for anxiety, tension, irritability, and emotionally "
                "tagged salience amplified during symptomatic phases."
            ),
            "hippocampus": (
                "Hippocampal anchor for memory, concentration-linked burden, and "
                "stress-sensitive regulation."
            ),
            "hypothalamus_proxy": (
                "Hypothalamic proxy for HPO/HPA orchestration of cyclical hormone "
                "signaling and neuroendocrine stress coupling."
            ),
            "ofc": (
                "Orbitofrontal control anchor for emotional valuation, impulse control, "
                "and regulatory failure during the luteal phase."
            ),
            "dlpfc": (
                "Dorsolateral prefrontal anchor for executive control, attention, and "
                "self-regulatory capacity."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Heritable liability influencing sensitivity of neuroendocrine and "
                "neurotransmitter systems to ovarian steroid fluctuations."
            ),
            "baseline_vulnerability_traits": (
                "Stable vulnerability traits present throughout the cycle that increase "
                "susceptibility to symptomatic luteal decompensation."
            ),
            "ovarian_steroid_flux": (
                "Normal cyclical rise and fall of estrogen and progesterone acting as the "
                "primary physiological trigger in vulnerable individuals."
            ),
            "environmental_stress_load": (
                "Environmental or psychological stress that can worsen symptom severity "
                "through stress-endocrine interactions."
            ),
            "luteal_metabolic_demand": (
                "Luteal-phase self-regulatory or metabolic burden that can reduce "
                "prefrontal control capacity."
            ),
            "monoamine_support": (
                "Protective serotonergic/noradrenergic support representing SSRI/SNRI-like "
                "stabilization of monoamine signaling."
            ),
            "cycle_stabilization_support": (
                "Protective support representing interventions that stabilize ovarian "
                "steroid fluctuations or dampen downstream CNS sensitivity."
            ),
            "recovery_support": (
                "Protective behavioral, medical, and environmental support that buffers "
                "stress amplification and improves regulation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "ovarian_steroid_response_sensitivity": (
                "Abnormally heightened CNS sensitivity to normal cyclical estrogen and "
                "progesterone changes."
            ),
            "serotonergic_destabilization": (
                "Hormone-sensitive perturbation of serotonin signaling contributing to "
                "dysphoria and emotional dysregulation."
            ),
            "gabaergic_destabilization": (
                "Hormone-sensitive disruption of inhibitory tone contributing to tension, "
                "lability, and reduced emotional buffering."
            ),
            "dopaminergic_modulation_shift": (
                "Hormone-linked changes in dopaminergic reward/motivation modulation that "
                "can alter irritability, salience, and behavioral control."
            ),
            "noradrenergic_arousal_dysregulation": (
                "Altered norepinephrine-linked arousal and vigilance contributing to "
                "tension and hyperreactivity."
            ),
            "hpo_hpa_coupling_dysregulation": (
                "Dysregulated interaction between reproductive and stress endocrine systems."
            ),
            "prefrontal_self_regulation_burden": (
                "Reduced prefrontal self-regulatory capacity, including luteal-phase energy "
                "or control depletion."
            ),
            "frontolimbic_dysregulation": (
                "Weak top-down control of limbic reactivity by prefrontal systems."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "anxiety_tension": (
                "Marked anxiety, tension, and hyperarousal linked to amygdala and "
                "noradrenergic burden."
            ),
            "irritability": (
                "Heightened irritability or anger-proneness emerging from frontolimbic and "
                "arousal dysregulation."
            ),
            "emotional_lability": (
                "Rapid shifts in affect and reduced emotional stability across the luteal phase."
            ),
            "dysphoric_mood": (
                "Dysphoric, low, or depressed mood emerging from hormone-sensitive "
                "monoamine and limbic dysregulation."
            ),
            "concentration_difficulty": (
                "Attention and executive inefficiency linked to DLPFC and hippocampal burden."
            ),
            "forgetfulness": (
                "Memory inefficiency or subjective forgetfulness linked to hippocampal and "
                "stress-related burden."
            ),
            "impulse_control_difficulty": (
                "Reduced capacity to inhibit impulses or regulate emotionally driven actions."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "ovarian_steroid_response_sensitivity",
                "relation": "heritable liability increases CNS sensitivity to cyclical ovarian steroid shifts",
                "pmdd_change": "increased",
            },
            {
                "source": "baseline_vulnerability_traits",
                "target": "ovarian_steroid_response_sensitivity",
                "relation": "stable vulnerability traits raise the likelihood of symptomatic hormone responsiveness",
                "pmdd_change": "increased",
            },
            {
                "source": "ovarian_steroid_flux",
                "target": "ovarian_steroid_response_sensitivity",
                "relation": "normal cyclical estrogen/progesterone fluctuations act as the physiological trigger",
                "pmdd_change": "increased",
            },
            {
                "source": "environmental_stress_load",
                "target": "hpo_hpa_coupling_dysregulation",
                "relation": "stress intensifies reproductive-stress endocrine coupling burden",
                "pmdd_change": "increased",
            },
            {
                "source": "luteal_metabolic_demand",
                "target": "prefrontal_self_regulation_burden",
                "relation": "luteal self-regulatory demand reduces prefrontal control resources",
                "pmdd_change": "increased",
            },
            {
                "source": "monoamine_support",
                "target": "serotonergic_destabilization",
                "relation": "monoamine-targeted treatment partially stabilizes serotonin signaling",
                "pmdd_change": "decreased",
            },
            {
                "source": "monoamine_support",
                "target": "noradrenergic_arousal_dysregulation",
                "relation": "monoamine-targeted treatment partially stabilizes norepinephrine-linked arousal",
                "pmdd_change": "decreased",
            },
            {
                "source": "cycle_stabilization_support",
                "target": "ovarian_steroid_response_sensitivity",
                "relation": "cycle stabilization reduces the impact of ovarian steroid fluctuations",
                "pmdd_change": "decreased",
            },
            {
                "source": "cycle_stabilization_support",
                "target": "hpo_hpa_coupling_dysregulation",
                "relation": "cycle stabilization dampens endocrine trigger pressure",
                "pmdd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "frontolimbic_dysregulation",
                "relation": "protective supports improve regulatory buffering",
                "pmdd_change": "decreased",
            },
            {
                "source": "ovarian_steroid_response_sensitivity",
                "target": "serotonergic_destabilization",
                "relation": "hormone sensitivity destabilizes serotonin-linked affect regulation",
                "pmdd_change": "increased",
            },
            {
                "source": "ovarian_steroid_response_sensitivity",
                "target": "gabaergic_destabilization",
                "relation": "hormone sensitivity perturbs inhibitory GABA-linked emotional buffering",
                "pmdd_change": "increased",
            },
            {
                "source": "ovarian_steroid_response_sensitivity",
                "target": "dopaminergic_modulation_shift",
                "relation": "hormone sensitivity perturbs dopamine-linked motivational and salience signaling",
                "pmdd_change": "increased",
            },
            {
                "source": "ovarian_steroid_response_sensitivity",
                "target": "noradrenergic_arousal_dysregulation",
                "relation": "hormone sensitivity perturbs norepinephrine-linked arousal systems",
                "pmdd_change": "increased",
            },
            {
                "source": "ovarian_steroid_flux",
                "target": "hpo_hpa_coupling_dysregulation",
                "relation": "cyclical ovarian hormone changes engage hypothalamic reproductive-stress coupling",
                "pmdd_change": "increased",
            },
            {
                "source": "hpo_hpa_coupling_dysregulation",
                "target": "hypothalamus_proxy",
                "relation": "endocrine coupling burden centers on hypothalamic control systems",
                "pmdd_change": "dysregulated",
            },
            {
                "source": "hpo_hpa_coupling_dysregulation",
                "target": "hippocampus",
                "relation": "stress-endocrine burden compromises hippocampal cognitive regulation",
                "pmdd_change": "dysregulated",
            },
            {
                "source": "prefrontal_self_regulation_burden",
                "target": "ofc",
                "relation": "reduced control resources compromise orbitofrontal regulation",
                "pmdd_change": "dysregulated",
            },
            {
                "source": "prefrontal_self_regulation_burden",
                "target": "dlpfc",
                "relation": "reduced control resources compromise executive control systems",
                "pmdd_change": "dysregulated",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "amygdala",
                "relation": "weak top-down control permits exaggerated limbic reactivity",
                "pmdd_change": "hyperreactive",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "ofc",
                "relation": "frontolimbic dysregulation destabilizes emotional valuation and control",
                "pmdd_change": "dysregulated",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "dlpfc",
                "relation": "frontolimbic dysregulation weakens executive governance of emotion",
                "pmdd_change": "dysregulated",
            },
            {
                "source": "amygdala",
                "target": "anxiety_tension",
                "relation": "amygdala hyperreactivity amplifies anxiety and tension",
                "pmdd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "irritability",
                "relation": "limbic hyperreactivity contributes to irritability and anger-proneness",
                "pmdd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotional_lability",
                "relation": "heightened emotional salience destabilizes affect across the luteal phase",
                "pmdd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "concentration_difficulty",
                "relation": "hippocampal burden contributes to cognitive inefficiency",
                "pmdd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "forgetfulness",
                "relation": "hippocampal burden contributes to memory inefficiency and forgetfulness",
                "pmdd_change": "increased",
            },
            {
                "source": "ofc",
                "target": "impulse_control_difficulty",
                "relation": "orbitofrontal dysregulation weakens inhibitory control and emotional judgment",
                "pmdd_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "concentration_difficulty",
                "relation": "executive control disruption worsens attention and concentration",
                "pmdd_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "impulse_control_difficulty",
                "relation": "weakened top-down control impairs suppression of emotionally driven responses",
                "pmdd_change": "increased",
            },
            {
                "source": "serotonergic_destabilization",
                "target": "dysphoric_mood",
                "relation": "serotonergic destabilization contributes to dysphoria and low mood",
                "pmdd_change": "increased",
            },
            {
                "source": "noradrenergic_arousal_dysregulation",
                "target": "anxiety_tension",
                "relation": "arousal dysregulation increases vigilance and tension",
                "pmdd_change": "increased",
            },
            {
                "source": "gabaergic_destabilization",
                "target": "emotional_lability",
                "relation": "reduced inhibitory buffering destabilizes mood reactivity",
                "pmdd_change": "increased",
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
            "prefrontal cortex",
            "hippocampus",
            "amygdala",
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
        genetic_vulnerability: float = 0.45,
        baseline_vulnerability_traits: float = 0.50,
        ovarian_steroid_flux: float = 0.75,
        environmental_stress_load: float = 0.40,
        luteal_metabolic_demand: float = 0.55,
        monoamine_support: float = 0.15,
        cycle_stabilization_support: float = 0.05,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Values are clipped to [0, 1]. Higher regional-state values mean greater
        dysregulation / burden in that circuit node.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "baseline_vulnerability_traits": self._clip01(baseline_vulnerability_traits),
                "ovarian_steroid_flux": self._clip01(ovarian_steroid_flux),
                "environmental_stress_load": self._clip01(environmental_stress_load),
                "luteal_metabolic_demand": self._clip01(luteal_metabolic_demand),
                "monoamine_support": self._clip01(monoamine_support),
                "cycle_stabilization_support": self._clip01(cycle_stabilization_support),
                "recovery_support": self._clip01(recovery_support),
            },
            name="value",
        )

        # Inputs -> latent biology
        ovarian_steroid_response_sensitivity = self._clip01(
            0.30 * inputs["genetic_vulnerability"]
            + 0.30 * inputs["baseline_vulnerability_traits"]
            + 0.25 * inputs["ovarian_steroid_flux"]
            + 0.05 * inputs["environmental_stress_load"]
            - 0.25 * inputs["cycle_stabilization_support"]
            - 0.10 * inputs["recovery_support"]
        )

        serotonergic_destabilization = self._clip01(
            0.30 * ovarian_steroid_response_sensitivity
            + 0.20 * inputs["ovarian_steroid_flux"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.10 * inputs["environmental_stress_load"]
            - 0.25 * inputs["monoamine_support"]
            - 0.15 * inputs["cycle_stabilization_support"]
            - 0.05 * inputs["recovery_support"]
        )

        gabaergic_destabilization = self._clip01(
            0.35 * ovarian_steroid_response_sensitivity
            + 0.25 * inputs["ovarian_steroid_flux"]
            + 0.10 * inputs["environmental_stress_load"]
            + 0.05 * inputs["baseline_vulnerability_traits"]
            - 0.20 * inputs["cycle_stabilization_support"]
            - 0.05 * inputs["recovery_support"]
        )

        dopaminergic_modulation_shift = self._clip01(
            0.20 * ovarian_steroid_response_sensitivity
            + 0.20 * inputs["ovarian_steroid_flux"]
            + 0.15 * inputs["luteal_metabolic_demand"]
            + 0.10 * inputs["baseline_vulnerability_traits"]
            + 0.05 * inputs["environmental_stress_load"]
            - 0.10 * inputs["cycle_stabilization_support"]
            - 0.05 * inputs["recovery_support"]
        )

        noradrenergic_arousal_dysregulation = self._clip01(
            0.25 * inputs["environmental_stress_load"]
            + 0.20 * inputs["ovarian_steroid_flux"]
            + 0.20 * ovarian_steroid_response_sensitivity
            + 0.10 * inputs["baseline_vulnerability_traits"]
            - 0.25 * inputs["monoamine_support"]
            - 0.05 * inputs["recovery_support"]
        )

        hpo_hpa_coupling_dysregulation = self._clip01(
            0.30 * inputs["ovarian_steroid_flux"]
            + 0.25 * inputs["environmental_stress_load"]
            + 0.20 * ovarian_steroid_response_sensitivity
            + 0.10 * inputs["genetic_vulnerability"]
            + 0.10 * inputs["baseline_vulnerability_traits"]
            - 0.20 * inputs["cycle_stabilization_support"]
            - 0.10 * inputs["recovery_support"]
        )

        prefrontal_self_regulation_burden = self._clip01(
            0.35 * inputs["luteal_metabolic_demand"]
            + 0.20 * noradrenergic_arousal_dysregulation
            + 0.15 * serotonergic_destabilization
            + 0.10 * hpo_hpa_coupling_dysregulation
            + 0.10 * inputs["environmental_stress_load"]
            - 0.10 * inputs["recovery_support"]
        )

        frontolimbic_dysregulation = self._clip01(
            0.25 * prefrontal_self_regulation_burden
            + 0.20 * serotonergic_destabilization
            + 0.15 * gabaergic_destabilization
            + 0.15 * noradrenergic_arousal_dysregulation
            + 0.10 * hpo_hpa_coupling_dysregulation
            + 0.10 * ovarian_steroid_response_sensitivity
            - 0.10 * inputs["recovery_support"]
        )

        latents = pd.Series(
            {
                "ovarian_steroid_response_sensitivity": ovarian_steroid_response_sensitivity,
                "serotonergic_destabilization": serotonergic_destabilization,
                "gabaergic_destabilization": gabaergic_destabilization,
                "dopaminergic_modulation_shift": dopaminergic_modulation_shift,
                "noradrenergic_arousal_dysregulation": noradrenergic_arousal_dysregulation,
                "hpo_hpa_coupling_dysregulation": hpo_hpa_coupling_dysregulation,
                "prefrontal_self_regulation_burden": prefrontal_self_regulation_burden,
                "frontolimbic_dysregulation": frontolimbic_dysregulation,
            },
            name="value",
        )

        # Latent biology -> regional-state burden
        hypothalamus_proxy = self._clip01(
            0.45 * hpo_hpa_coupling_dysregulation
            + 0.25 * ovarian_steroid_response_sensitivity
            + 0.10 * inputs["ovarian_steroid_flux"]
            + 0.05 * inputs["environmental_stress_load"]
            - 0.15 * inputs["cycle_stabilization_support"]
        )

        amygdala = self._clip01(
            0.30 * frontolimbic_dysregulation
            + 0.25 * noradrenergic_arousal_dysregulation
            + 0.20 * ovarian_steroid_response_sensitivity
            + 0.10 * hpo_hpa_coupling_dysregulation
            - 0.10 * inputs["recovery_support"]
        )

        hippocampus = self._clip01(
            0.30 * hpo_hpa_coupling_dysregulation
            + 0.20 * serotonergic_destabilization
            + 0.20 * frontolimbic_dysregulation
            + 0.10 * inputs["environmental_stress_load"]
            - 0.10 * inputs["recovery_support"]
        )

        ofc = self._clip01(
            0.35 * prefrontal_self_regulation_burden
            + 0.25 * frontolimbic_dysregulation
            + 0.10 * dopaminergic_modulation_shift
            + 0.10 * serotonergic_destabilization
            - 0.10 * inputs["recovery_support"]
        )

        dlpfc = self._clip01(
            0.40 * prefrontal_self_regulation_burden
            + 0.20 * frontolimbic_dysregulation
            + 0.15 * hpo_hpa_coupling_dysregulation
            + 0.10 * serotonergic_destabilization
            - 0.10 * inputs["recovery_support"]
        )

        regional_state = pd.Series(
            {
                "amygdala": amygdala,
                "hippocampus": hippocampus,
                "hypothalamus_proxy": hypothalamus_proxy,
                "ofc": ofc,
                "dlpfc": dlpfc,
            },
            name="value",
        )

        # Regional-state / latent burden -> symptoms
        anxiety_tension = self._clip01(
            0.40 * amygdala
            + 0.25 * noradrenergic_arousal_dysregulation
            + 0.15 * hypothalamus_proxy
            + 0.10 * frontolimbic_dysregulation
        )

        irritability = self._clip01(
            0.30 * amygdala
            + 0.20 * ofc
            + 0.15 * dopaminergic_modulation_shift
            + 0.15 * noradrenergic_arousal_dysregulation
            + 0.10 * ovarian_steroid_response_sensitivity
        )

        emotional_lability = self._clip01(
            0.30 * frontolimbic_dysregulation
            + 0.25 * amygdala
            + 0.20 * ovarian_steroid_response_sensitivity
            + 0.10 * ofc
            + 0.10 * gabaergic_destabilization
        )

        dysphoric_mood = self._clip01(
            0.30 * serotonergic_destabilization
            + 0.20 * hippocampus
            + 0.20 * frontolimbic_dysregulation
            + 0.10 * hpo_hpa_coupling_dysregulation
        )

        concentration_difficulty = self._clip01(
            0.35 * dlpfc
            + 0.20 * prefrontal_self_regulation_burden
            + 0.20 * hippocampus
            + 0.10 * hpo_hpa_coupling_dysregulation
        )

        forgetfulness = self._clip01(
            0.35 * hippocampus
            + 0.25 * concentration_difficulty
            + 0.15 * hpo_hpa_coupling_dysregulation
        )

        impulse_control_difficulty = self._clip01(
            0.25 * ofc
            + 0.25 * dlpfc
            + 0.15 * dopaminergic_modulation_shift
            + 0.15 * amygdala
            + 0.10 * prefrontal_self_regulation_burden
        )

        symptoms = pd.Series(
            {
                "anxiety_tension": anxiety_tension,
                "irritability": irritability,
                "emotional_lability": emotional_lability,
                "dysphoric_mood": dysphoric_mood,
                "concentration_difficulty": concentration_difficulty,
                "forgetfulness": forgetfulness,
                "impulse_control_difficulty": impulse_control_difficulty,
            },
            name="value",
        )

        # Symptom composites / phenotype summaries
        luteal_anxious_irritable_profile = self._clip01(
            (anxiety_tension + irritability + emotional_lability) / 3.0
        )

        luteal_dysphoric_profile = self._clip01(
            (dysphoric_mood + emotional_lability + anxiety_tension) / 3.0
        )

        cognitive_pmdd_profile = self._clip01(
            (concentration_difficulty + forgetfulness + dlpfc + hippocampus) / 4.0
        )

        hormone_sensitive_frontolimbic_profile = self._clip01(
            (
                ovarian_steroid_response_sensitivity
                + frontolimbic_dysregulation
                + amygdala
                + ofc
            ) / 4.0
        )

        control_depletion_profile = self._clip01(
            (
                impulse_control_difficulty
                + prefrontal_self_regulation_burden
                + ofc
                + dlpfc
            ) / 4.0
        )

        global_pmdd_burden = self._clip01(float(symptoms.mean()))

        phenotypes = pd.Series(
            {
                "luteal_anxious_irritable_profile": luteal_anxious_irritable_profile,
                "luteal_dysphoric_profile": luteal_dysphoric_profile,
                "cognitive_pmdd_profile": cognitive_pmdd_profile,
                "hormone_sensitive_frontolimbic_profile": hormone_sensitive_frontolimbic_profile,
                "control_depletion_profile": control_depletion_profile,
                "global_pmdd_burden": global_pmdd_burden,
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
    model = PremenstrualDysphoricDisorderModel()

    print("\n=== Building atlas-backed scaffold ===")
    scaffold = model.build(connectivity_rows=10)

    node_cols = ["key", "node_type", "atlas_region", "feature_summary"]
    print("\nNodes:")
    print(scaffold["nodes"][node_cols].to_string(index=False))

    print("\nEdges (first 20):")
    print(scaffold["edges"].head(20).to_string(index=False))

    if "amygdala" in scaffold["receptors"] and not scaffold["receptors"]["amygdala"].empty:
        print("\nAmygdala receptor fingerprint:")
        print(scaffold["receptors"]["amygdala"].head().to_string(index=False))
    else:
        print("\nNo amygdala receptor table available in this environment.")

    if "dlpfc" in scaffold["genes"] and not scaffold["genes"]["dlpfc"].empty:
        print("\nDLPFC gene summary:")
        print(scaffold["genes"]["dlpfc"].head().to_string(index=False))
    else:
        print("\nNo DLPFC gene table available in this environment.")

    if (
        "hippocampus" in scaffold["connectivity_profiles"]
        and not scaffold["connectivity_profiles"]["hippocampus"].empty
    ):
        print("\nHippocampus connectivity profile:")
        print(scaffold["connectivity_profiles"]["hippocampus"].head().to_string(index=False))
    else:
        print("\nNo hippocampus connectivity profile available in this environment.")

    if not scaffold["circuit_connectivity"].empty:
        print("\nWithin-model circuit connectivity:")
        print(scaffold["circuit_connectivity"].round(3).to_string())
    else:
        print("\nNo within-model circuit connectivity matrix available in this environment.")

    print("\n=== Simulation example: hormone-sensitive luteal dysregulation ===")
    sim = model.simulate(
        genetic_vulnerability=0.50,
        baseline_vulnerability_traits=0.60,
        ovarian_steroid_flux=0.80,
        environmental_stress_load=0.45,
        luteal_metabolic_demand=0.65,
        monoamine_support=0.15,
        cycle_stabilization_support=0.05,
        recovery_support=0.25,
    )
    for name, series in sim.items():
        print(f"\n{name}:")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-8, -14, -10)).head())

    # Example regional mask retrieval:
    # mask = model.region_mask("amygdala")
    # if mask is not None:
    #     print(mask)
