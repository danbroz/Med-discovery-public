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


VOYEURISTIC_DISORDER_GENE_PANEL = [
    "DRD2",
    "DRD3",
    "SLC6A3",
    "COMT",
    "SLC6A4",
    "HTR2A",
    "HTR1B",
    "MAOA",
    "BDNF",
]


class VoyeuristicDisorderModel:
    """
    Atlas-grounded siibra scaffold for Voyeuristic Disorder.

    This script is a research scaffold, not a diagnostic, forensic, prognostic,
    or treatment tool. It translates a chapter-level description of Voyeuristic
    Disorder into a transparent mechanistic graph and a simple normalized
    simulator.

    Chapter-driven design choices:
    - The core mechanism is modeled as an interaction between conditioned visual
      incentive salience, mesolimbic reward capture, and impaired top-down
      inhibitory control.
    - Dopamine is kept mostly latent and expressed anatomically through a
      ventral-striatum / nucleus-accumbens proxy, because the chapter centers the
      VTA-NAc reward system but does not justify precise parcel-level claims for
      every component.
    - Serotonergic dysfunction is modeled as a latent control deficit affecting
      impulsivity and affect regulation, with SSRI support represented as a
      protective input.
    - The amygdala is treated as a risk-arousal node whose fear-of-being-caught
      signal can become coupled to reward, while the insula and ACC capture the
      salience-network emphasis described in the chapter.
    - OFC and DLPFC are included as separate prefrontal-control nodes because the
      chapter specifically predicts hypoactivation in both regions.

    Because direct disorder-specific imaging and genetics remain limited, this
    scaffold is intentionally conservative and partly hypothesis-driven. Higher
    values in ``simulate()`` represent greater disorder-linked burden or circuit
    dysregulation, not healthier activation.
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
                "siibra is required to use VoyeuristicDisorderModel. "
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
            "visual_cortex": [
                "Area hOc1 [V1] left",
                "Area hOc2 [V2] left",
                "hOc1",
                "hOc2",
                "visual cortex",
                "occipital cortex",
            ],
            "ventral_striatum_proxy": [
                "Nucleus accumbens left",
                "Accumbens left",
                "nucleus accumbens",
                "ventral striatum",
                "caudate nucleus",
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
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
                "orbitofrontal cortex",
                "orbitofrontal",
            ],
            "dlpfc": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area IFJ left",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "visual_cortex": (
                "Visual-cue processing node predicted to show heightened response to conditioned voyeuristic cues."
            ),
            "ventral_striatum_proxy": (
                "Proxy for nucleus accumbens / ventral-striatal reward reinforcement and incentive salience."
            ),
            "insula": (
                "Interoceptive and salience node contributing to arousal, bodily awareness, and cue significance."
            ),
            "acc": (
                "Anterior cingulate salience and conflict-monitoring node integrating reward, risk, and control burden."
            ),
            "amygdala": (
                "Fear- and emotional-salience node whose detection-risk signal may become coupled to excitement."
            ),
            "ofc": (
                "Orbitofrontal control node supporting inhibition, social rule evaluation, and value-based restraint."
            ),
            "dlpfc": (
                "Dorsolateral prefrontal control node supporting executive inhibition and top-down behavioral regulation."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Shared heritable liability for impulsivity, sensation-seeking, emotional dysregulation, and related psychopathology"
            ),
            "impulsivity_trait_load": (
                "Trait-level impulsivity increasing difficulty resisting reinforced urges"
            ),
            "sensation_seeking_drive": (
                "Drive toward novel, intense, and risky stimulation that can amplify cue pursuit"
            ),
            "emotional_dysregulation_load": (
                "Irritability, anxiety, dysphoria, and poor affect regulation that can intensify compulsive cycles"
            ),
            "conditioned_cue_exposure": (
                "Strength of learned cue associations that make voyeuristic stimuli highly salient and rewarding"
            ),
            "risk_of_detection_context": (
                "Contextual intensity of secrecy, fear of being caught, and detection-related arousal"
            ),
            "ssri_support": (
                "Protective serotonergic treatment support that may reduce impulsive-compulsive pressure"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_control_deficit": (
                "Reduced serotonergic modulation of impulsivity, affect regulation, and behavioral restraint"
            ),
            "conditioned_visual_incentive_salience": (
                "Voyeuristic cues becoming conditioned high-value reinforcers that capture attention"
            ),
            "mesolimbic_reward_capture": (
                "Dopamine-linked reward capture and approach motivation centered on mesolimbic reinforcement"
            ),
            "amygdala_risk_reward_coupling": (
                "Aberrant coupling in which fear of detection amplifies rather than suppresses excitement"
            ),
            "salience_network_bias": (
                "Hyperallocation of visual, interoceptive, and cingulate attention to disorder-relevant cues"
            ),
            "top_down_inhibitory_failure": (
                "Reduced OFC and DLPFC control over urges, reward seeking, and social-moral constraints"
            ),
            "compulsive_tension_relief_cycle": (
                "Rising pre-act tension followed by reinforcing gratification or relief that stamps in repetition"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "compulsive_voyeuristic_urge": (
                "Strong repetitive urge to seek disorder-relevant voyeuristic observation cues"
            ),
            "pre_act_tension_arousal": (
                "Escalating tension and arousal before the act"
            ),
            "post_act_gratification_relief": (
                "Short-lived gratification or relief after the act that reinforces repetition"
            ),
            "repetitive_boundary_violating_behavior": (
                "Persistence of the behavior despite social, moral, and rational constraints"
            ),
            "social_moral_constraint_override": (
                "Failure of inhibitory control and social rule evaluation to block the behavior"
            ),
            "risk_amplified_excitation": (
                "Excitement intensified by secrecy, threat, or fear of being caught"
            ),
            "affective_instability": (
                "Irritability, anxiety, dysphoria, or emotional lability linked to poor impulse regulation"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_control_deficit",
                "relation": "contributes shared heritable liability for impulsivity and poor emotional regulation",
                "voyeuristic_change": "increased",
            },
            {
                "source": "emotional_dysregulation_load",
                "target": "serotonergic_control_deficit",
                "relation": "raises affective dysregulation burden associated with poor impulse control",
                "voyeuristic_change": "increased",
            },
            {
                "source": "ssri_support",
                "target": "serotonergic_control_deficit",
                "relation": "can improve serotonergic modulation and reduce impulsive-compulsive pressure",
                "voyeuristic_change": "decreased",
            },
            {
                "source": "conditioned_cue_exposure",
                "target": "conditioned_visual_incentive_salience",
                "relation": "strengthens learned cue value so relevant visual stimuli become powerful reinforcers",
                "voyeuristic_change": "increased",
            },
            {
                "source": "sensation_seeking_drive",
                "target": "conditioned_visual_incentive_salience",
                "relation": "increases attraction to novel and intense stimulation",
                "voyeuristic_change": "increased",
            },
            {
                "source": "conditioned_visual_incentive_salience",
                "target": "mesolimbic_reward_capture",
                "relation": "drives reward-system capture by learned high-value cues",
                "voyeuristic_change": "increased",
            },
            {
                "source": "sensation_seeking_drive",
                "target": "mesolimbic_reward_capture",
                "relation": "amplifies reward pursuit and approach behavior",
                "voyeuristic_change": "increased",
            },
            {
                "source": "risk_of_detection_context",
                "target": "amygdala_risk_reward_coupling",
                "relation": "activates fear-of-being-caught arousal that can become fused with excitement",
                "voyeuristic_change": "increased",
            },
            {
                "source": "mesolimbic_reward_capture",
                "target": "amygdala_risk_reward_coupling",
                "relation": "permits risk-related arousal to be co-opted by reward processes",
                "voyeuristic_change": "increased",
            },
            {
                "source": "conditioned_visual_incentive_salience",
                "target": "salience_network_bias",
                "relation": "biases attention toward disorder-relevant cues",
                "voyeuristic_change": "increased",
            },
            {
                "source": "amygdala_risk_reward_coupling",
                "target": "salience_network_bias",
                "relation": "adds emotional intensity and threat salience to the cue state",
                "voyeuristic_change": "increased",
            },
            {
                "source": "serotonergic_control_deficit",
                "target": "top_down_inhibitory_failure",
                "relation": "weakens behavioral restraint and affect regulation",
                "voyeuristic_change": "increased",
            },
            {
                "source": "impulsivity_trait_load",
                "target": "top_down_inhibitory_failure",
                "relation": "raises baseline risk of failing to resist the reinforced urge",
                "voyeuristic_change": "increased",
            },
            {
                "source": "mesolimbic_reward_capture",
                "target": "top_down_inhibitory_failure",
                "relation": "reward pressure overwhelms rational and inhibitory control systems",
                "voyeuristic_change": "increased",
            },
            {
                "source": "mesolimbic_reward_capture",
                "target": "compulsive_tension_relief_cycle",
                "relation": "provides the gratification and pleasure that stamp in repetition",
                "voyeuristic_change": "increased",
            },
            {
                "source": "amygdala_risk_reward_coupling",
                "target": "compulsive_tension_relief_cycle",
                "relation": "adds risk-linked arousal to the pre-act tension and post-act relief loop",
                "voyeuristic_change": "increased",
            },
            {
                "source": "top_down_inhibitory_failure",
                "target": "compulsive_tension_relief_cycle",
                "relation": "permits the urge-tension-relief loop to continue unchecked",
                "voyeuristic_change": "increased",
            },
            {
                "source": "conditioned_visual_incentive_salience",
                "target": "visual_cortex",
                "relation": "maps learned cue salience onto visual-cortical hyperresponsivity",
                "voyeuristic_change": "increased",
            },
            {
                "source": "mesolimbic_reward_capture",
                "target": "ventral_striatum_proxy",
                "relation": "maps reward capture onto ventral-striatal reinforcement circuitry",
                "voyeuristic_change": "increased",
            },
            {
                "source": "salience_network_bias",
                "target": "insula",
                "relation": "maps salience and arousal burden onto insular processing",
                "voyeuristic_change": "increased",
            },
            {
                "source": "salience_network_bias",
                "target": "acc",
                "relation": "burdens cingulate conflict and salience-integration functions",
                "voyeuristic_change": "increased",
            },
            {
                "source": "amygdala_risk_reward_coupling",
                "target": "amygdala",
                "relation": "maps risk-fear-excitement coupling onto amygdala reactivity",
                "voyeuristic_change": "increased",
            },
            {
                "source": "top_down_inhibitory_failure",
                "target": "ofc",
                "relation": "maps reduced social-rule and value-based restraint onto orbitofrontal dysfunction",
                "voyeuristic_change": "increased",
            },
            {
                "source": "top_down_inhibitory_failure",
                "target": "dlpfc",
                "relation": "maps reduced executive inhibition onto dorsolateral prefrontal dysfunction",
                "voyeuristic_change": "increased",
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "compulsive_voyeuristic_urge",
                "relation": "drives incentive salience and urge strength",
                "voyeuristic_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "pre_act_tension_arousal",
                "relation": "contributes fear-linked arousal before the act",
                "voyeuristic_change": "increased",
            },
            {
                "source": "insula",
                "target": "pre_act_tension_arousal",
                "relation": "contributes conscious arousal and bodily tension",
                "voyeuristic_change": "increased",
            },
            {
                "source": "compulsive_tension_relief_cycle",
                "target": "post_act_gratification_relief",
                "relation": "drives the relief and gratification that reinforce future repetition",
                "voyeuristic_change": "increased",
            },
            {
                "source": "ofc",
                "target": "social_moral_constraint_override",
                "relation": "orbitofrontal dysfunction weakens social and moral constraint evaluation",
                "voyeuristic_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "social_moral_constraint_override",
                "relation": "dorsolateral control failure weakens executive inhibition",
                "voyeuristic_change": "increased",
            },
            {
                "source": "compulsive_voyeuristic_urge",
                "target": "repetitive_boundary_violating_behavior",
                "relation": "strong urges increase repetition of the behavior",
                "voyeuristic_change": "increased",
            },
            {
                "source": "post_act_gratification_relief",
                "target": "repetitive_boundary_violating_behavior",
                "relation": "reward and relief stamp in repetition",
                "voyeuristic_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "risk_amplified_excitation",
                "relation": "fear of detection contributes to the excitement state",
                "voyeuristic_change": "increased",
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "risk_amplified_excitation",
                "relation": "reward reinforcement magnifies the excitement attached to risk",
                "voyeuristic_change": "increased",
            },
            {
                "source": "serotonergic_control_deficit",
                "target": "affective_instability",
                "relation": "links poor serotonergic regulation to irritability, anxiety, and dysphoria",
                "voyeuristic_change": "increased",
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
            "visual cortex",
            "occipital cortex",
            "nucleus accumbens",
            "ventral striatum",
            "insula",
            "anterior cingulate cortex",
            "anterior cingulate",
            "amygdala",
            "orbitofrontal cortex",
            "prefrontal cortex",
            "dorsolateral prefrontal cortex",
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
        gene_panel: Sequence[str] = VOYEURISTIC_DISORDER_GENE_PANEL,
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
                    "Research scaffold only. The model emphasizes conditioned visual incentive salience, "
                    "mesolimbic reward capture, salience-network bias, risk-reward coupling, and prefrontal "
                    "inhibitory failure."
                ),
            },
        }

    def simulate(
        self,
        genetic_vulnerability: float = 0.55,
        impulsivity_trait_load: float = 0.65,
        sensation_seeking_drive: float = 0.6,
        emotional_dysregulation_load: float = 0.5,
        conditioned_cue_exposure: float = 0.65,
        risk_of_detection_context: float = 0.5,
        ssri_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass Voyeuristic Disorder simulation.

        All inputs are expected on a 0..1 scale. Higher support values are
        protective; all other inputs represent higher burden or vulnerability.
        Returned regional-state values reflect disorder-linked burden in the
        anchored system.
        """

        inputs = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "impulsivity_trait_load": self._clip01(impulsivity_trait_load),
            "sensation_seeking_drive": self._clip01(sensation_seeking_drive),
            "emotional_dysregulation_load": self._clip01(emotional_dysregulation_load),
            "conditioned_cue_exposure": self._clip01(conditioned_cue_exposure),
            "risk_of_detection_context": self._clip01(risk_of_detection_context),
            "ssri_support": self._clip01(ssri_support),
        }

        latents = {
            "serotonergic_control_deficit": 0.0,
            "conditioned_visual_incentive_salience": 0.0,
            "mesolimbic_reward_capture": 0.0,
            "amygdala_risk_reward_coupling": 0.0,
            "salience_network_bias": 0.0,
            "top_down_inhibitory_failure": 0.0,
            "compulsive_tension_relief_cycle": 0.0,
        }

        latents["serotonergic_control_deficit"] = self._clip01(
            0.32 * inputs["genetic_vulnerability"]
            + 0.28 * inputs["emotional_dysregulation_load"]
            + 0.18 * inputs["impulsivity_trait_load"]
            + 0.10 * inputs["sensation_seeking_drive"]
            - 0.34 * inputs["ssri_support"]
        )

        latents["conditioned_visual_incentive_salience"] = self._clip01(
            0.44 * inputs["conditioned_cue_exposure"]
            + 0.18 * inputs["sensation_seeking_drive"]
            + 0.10 * inputs["genetic_vulnerability"]
        )

        latents["mesolimbic_reward_capture"] = self._clip01(
            0.34 * latents["conditioned_visual_incentive_salience"]
            + 0.26 * inputs["sensation_seeking_drive"]
            + 0.16 * inputs["impulsivity_trait_load"]
            + 0.10 * inputs["risk_of_detection_context"]
            + 0.08 * inputs["genetic_vulnerability"]
        )

        latents["amygdala_risk_reward_coupling"] = self._clip01(
            0.34 * inputs["risk_of_detection_context"]
            + 0.28 * latents["mesolimbic_reward_capture"]
            + 0.14 * inputs["emotional_dysregulation_load"]
        )

        latents["salience_network_bias"] = self._clip01(
            0.32 * latents["conditioned_visual_incentive_salience"]
            + 0.24 * latents["amygdala_risk_reward_coupling"]
            + 0.18 * inputs["emotional_dysregulation_load"]
            + 0.10 * latents["mesolimbic_reward_capture"]
        )

        latents["top_down_inhibitory_failure"] = self._clip01(
            0.34 * latents["serotonergic_control_deficit"]
            + 0.26 * inputs["impulsivity_trait_load"]
            + 0.18 * inputs["emotional_dysregulation_load"]
            + 0.10 * latents["mesolimbic_reward_capture"]
            - 0.12 * inputs["ssri_support"]
        )

        latents["compulsive_tension_relief_cycle"] = self._clip01(
            0.30 * latents["mesolimbic_reward_capture"]
            + 0.24 * latents["top_down_inhibitory_failure"]
            + 0.20 * latents["amygdala_risk_reward_coupling"]
            + 0.10 * latents["salience_network_bias"]
            - 0.10 * inputs["ssri_support"]
        )

        regional_state = {
            "visual_cortex": self._clip01(
                0.54 * latents["conditioned_visual_incentive_salience"]
                + 0.18 * latents["salience_network_bias"]
            ),
            "ventral_striatum_proxy": self._clip01(
                0.56 * latents["mesolimbic_reward_capture"]
                + 0.16 * latents["conditioned_visual_incentive_salience"]
            ),
            "insula": self._clip01(
                0.42 * latents["salience_network_bias"]
                + 0.18 * latents["amygdala_risk_reward_coupling"]
                + 0.12 * inputs["emotional_dysregulation_load"]
            ),
            "acc": self._clip01(
                0.40 * latents["salience_network_bias"]
                + 0.24 * latents["top_down_inhibitory_failure"]
                + 0.14 * latents["amygdala_risk_reward_coupling"]
            ),
            "amygdala": self._clip01(
                0.50 * latents["amygdala_risk_reward_coupling"]
                + 0.18 * latents["salience_network_bias"]
            ),
            "ofc": self._clip01(
                0.48 * latents["top_down_inhibitory_failure"]
                + 0.14 * latents["compulsive_tension_relief_cycle"]
                + 0.10 * latents["mesolimbic_reward_capture"]
            ),
            "dlpfc": self._clip01(
                0.52 * latents["top_down_inhibitory_failure"]
                + 0.12 * inputs["emotional_dysregulation_load"]
                + 0.10 * latents["salience_network_bias"]
            ),
        }

        symptoms = {
            "compulsive_voyeuristic_urge": self._clip01(
                0.34 * regional_state["ventral_striatum_proxy"]
                + 0.20 * latents["top_down_inhibitory_failure"]
                + 0.16 * regional_state["visual_cortex"]
                + 0.10 * latents["salience_network_bias"]
            ),
            "pre_act_tension_arousal": self._clip01(
                0.30 * regional_state["amygdala"]
                + 0.24 * regional_state["insula"]
                + 0.18 * latents["compulsive_tension_relief_cycle"]
                + 0.10 * inputs["risk_of_detection_context"]
            ),
            "post_act_gratification_relief": self._clip01(
                0.42 * regional_state["ventral_striatum_proxy"]
                + 0.24 * latents["compulsive_tension_relief_cycle"]
                + 0.10 * latents["amygdala_risk_reward_coupling"]
            ),
            "repetitive_boundary_violating_behavior": 0.0,
            "social_moral_constraint_override": 0.0,
            "risk_amplified_excitation": 0.0,
            "affective_instability": 0.0,
        }

        symptoms["repetitive_boundary_violating_behavior"] = self._clip01(
            0.30 * symptoms["compulsive_voyeuristic_urge"]
            + 0.24 * latents["top_down_inhibitory_failure"]
            + 0.20 * symptoms["post_act_gratification_relief"]
            + 0.08 * inputs["conditioned_cue_exposure"]
        )

        symptoms["social_moral_constraint_override"] = self._clip01(
            0.30 * regional_state["ofc"]
            + 0.24 * regional_state["dlpfc"]
            + 0.18 * regional_state["ventral_striatum_proxy"]
            + 0.10 * inputs["impulsivity_trait_load"]
        )

        symptoms["risk_amplified_excitation"] = self._clip01(
            0.34 * regional_state["amygdala"]
            + 0.24 * regional_state["ventral_striatum_proxy"]
            + 0.18 * inputs["risk_of_detection_context"]
            + 0.10 * regional_state["insula"]
        )

        symptoms["affective_instability"] = self._clip01(
            0.34 * latents["serotonergic_control_deficit"]
            + 0.22 * regional_state["insula"]
            + 0.16 * regional_state["amygdala"]
            + 0.10 * inputs["emotional_dysregulation_load"]
            - 0.10 * inputs["ssri_support"]
        )

        phenotypes = {
            "compulsive_reward_profile": self._clip01(
                (
                    symptoms["compulsive_voyeuristic_urge"]
                    + symptoms["post_act_gratification_relief"]
                    + symptoms["repetitive_boundary_violating_behavior"]
                )
                / 3.0
            ),
            "risk_coupled_arousal_profile": self._clip01(
                (
                    symptoms["pre_act_tension_arousal"]
                    + symptoms["risk_amplified_excitation"]
                    + latents["amygdala_risk_reward_coupling"]
                )
                / 3.0
            ),
            "poor_control_profile": self._clip01(
                (
                    symptoms["social_moral_constraint_override"]
                    + latents["top_down_inhibitory_failure"]
                    + symptoms["repetitive_boundary_violating_behavior"]
                )
                / 3.0
            ),
            "serotonergic_dyscontrol_profile": self._clip01(
                (
                    symptoms["affective_instability"]
                    + latents["serotonergic_control_deficit"]
                    + symptoms["pre_act_tension_arousal"]
                )
                / 3.0
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

    model = VoyeuristicDisorderModel()
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
        "visual_cortex",
        "ventral_striatum_proxy",
        "insula",
        "acc",
        "amygdala",
        "ofc",
        "dlpfc",
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
        genetic_vulnerability=0.62,
        impulsivity_trait_load=0.74,
        sensation_seeking_drive=0.68,
        emotional_dysregulation_load=0.58,
        conditioned_cue_exposure=0.82,
        risk_of_detection_context=0.66,
        ssri_support=0.2,
    )
    for name, series in simulated.items():
        print(f"\n--- {name.upper()} ---")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment (requires a siibra-enabled runtime):
    # print(model.assign_mni_point((10, 12, -8)).head())
