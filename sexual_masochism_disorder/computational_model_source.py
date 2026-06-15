from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Sexual Masochism Disorder (SMD).

This script translates the supplied chapter into a transparent mechanistic
research scaffold. It is intended for exploratory modeling and atlas-backed
inspection, not diagnosis, treatment, or behavioral instruction.

Modeling interpretation of the chapter:
- Treat Sexual Masochism Disorder as a distributed alteration across stress,
  pain, reward, interoception, and cognitive reappraisal systems rather than a
  single neurotransmitter deficit.
- Keep stress chemistry, opioid reinforcement, dopaminergic salience, and
  pain-reward coupling primarily as latent biology nodes unless the chapter
  explicitly localizes them.
- Anchor only regions the chapter names or strongly implies: insula, anterior
  cingulate cortex, amygdala, ventral striatum / nucleus-accumbens-type reward
  circuitry, and prefrontal control/reappraisal cortex.
- Use a heuristic gene panel spanning opioid, dopamine, serotonin, stress, and
  plasticity systems because the chapter explicitly says no definitive SMD gene
  has been identified.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


SEXUAL_MASOCHISM_DISORDER_GENE_PANEL = [
    "OPRM1",
    "OPRK1",
    "DRD2",
    "DRD4",
    "SLC6A3",
    "SLC6A4",
    "COMT",
    "MAOA",
    "BDNF",
    "NR3C1",
    "CRHR1",
    "OXTR",
    "FAAH",
]


class SexualMasochismDisorderModel:
    """
    Mechanistic siibra scaffold for Sexual Masochism Disorder.

    Conceptual flow:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

    Important limitation:
    The chapter is largely theoretical and extrapolative. It names a small set
    of relevant regions and systems but does not provide a definitive imaging or
    molecular biomarker set. This scaffold therefore uses conservative region
    anchors, proxy nodes where appropriate, and hypothesis-driven latent
    processes.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        aggregate_connectivity_subjects: int = 6,
        debug: bool = False,
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.aggregate_connectivity_subjects = max(1, int(aggregate_connectivity_subjects))
        self.debug = bool(debug)

        # Compatibility-first resource lookup.
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

        self.disorder_name = "Sexual Masochism Disorder"
        self.disorder_key = "sexual_masochism_disorder"

        # The chapter explicitly names these regions or very close systems.
        self.region_candidates: Dict[str, List[str]] = {
            "insula": [
                "Area Id1 (Insula) left",
                "Area Id1 left",
                "Area Id2 (Insula) left",
                "Area Id2 left",
                "Area Ig1 (Insula) left",
                "Area Ig1 left",
                "insula left",
                "insular cortex left",
                "insula",
            ],
            "acc": [
                "Area p24ab (pACC) left",
                "Area p24ab left",
                "Area a24pr left",
                "Area p32 left",
                "anterior cingulate cortex left",
                "cingulate gyrus left",
                "anterior cingulate cortex",
                "cingulate gyrus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "ventral_striatum_proxy": [
                "FuCd (Ventral Striatum, Fundus of Caudate Nucleus) left",
                "ventral striatum left",
                "nucleus accumbens left",
                "ventral striatum",
                "nucleus accumbens",
                "striatum left",
                "striatum",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v1 left",
                "Area 8v2 (MFG) left",
                "Area 8v2 left",
                "Area 9/46d left",
                "Area 46 left",
                "middle frontal gyrus left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "early_life_adversity": (
                "Early trauma or adversity hypothesized to shape later pain, stress, and submission-related learning"
            ),
            "chronic_stress_load": (
                "Ongoing stress burden capable of producing durable physiological and psychological change"
            ),
            "sensation_seeking_trait": (
                "Temperamental drive toward intense experiences and risk-taking, framed as a vulnerability factor"
            ),
            "affective_instability_trait": (
                "Trait-level emotional volatility and anger-control difficulty that may amplify intense interpersonal scripts"
            ),
            "ritualized_submission_learning": (
                "Learned sexual-script association linking submission, restraint, and lack of control with arousal"
            ),
            "humiliation_association_learning": (
                "Learned association linking humiliation or degradation with arousal and relief"
            ),
            "physical_pain_salience": (
                "Salience of painful physical stimulation as a trigger for arousal, anticipation, and conditioning"
            ),
            "aversive_appraisal_reserve": (
                "Protective reserve preserving ordinary aversive interpretation of pain and humiliation"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "stress_arousal_sensitization": (
                "Stress-response sensitization producing stronger physiological arousal under threatening or painful cues"
            ),
            "endogenous_opioid_reinforcement": (
                "Opioid-mediated reinforcement of pain, relief, or endurance-like states"
            ),
            "dopaminergic_incentive_salience": (
                "Reward-circuit wanting and motivational salience applied to painful, submissive, or humiliating cues"
            ),
            "pain_reward_coupling": (
                "Core reversal whereby painful stimulation becomes linked to reward rather than aversion"
            ),
            "social_pain_eroticization": (
                "Overlap of social pain and erotic arousal, especially humiliation and degradation"
            ),
            "interoceptive_amplification": (
                "Amplified awareness of bodily arousal, pain, restraint, and visceral tension"
            ),
            "prefrontal_reappraisal_bias": (
                "Top-down reinterpretation bias transforming painful or humiliating signals into desirable meaning"
            ),
            "frontolimbic_regulatory_imbalance": (
                "Imbalance across prefrontal, cingulate, amygdala, and insular systems affecting emotion and control"
            ),
            "submission_relief_loop": (
                "Relief and arousal loop in which surrender of control becomes reinforcing"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "paradoxical_stress_arousal": (
                "Stressful or painful cues evoke excitement rather than ordinary aversion"
            ),
            "pain_desire_coupling": (
                "Pain becomes desirable or sexually salient rather than merely unpleasant"
            ),
            "humiliation_arousal_coupling": (
                "Humiliation or degradation becomes linked to arousal through social-pain overlap"
            ),
            "submission_seeking": (
                "Motivational pull toward submission, restraint, or loss-of-control scenarios"
            ),
            "ritualized_enactment_pressure": (
                "Pressure toward ritualized reenactment of suffering-linked arousal scripts"
            ),
            "distress_impairment": (
                "Overall burden from the trait constellation and circuit pattern described by the chapter"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "early_life_adversity",
                "target": "stress_arousal_sensitization",
                "relation": "can prime long-lasting stress responsivity and vulnerability",
                "smd_change": "increased",
            },
            {
                "source": "early_life_adversity",
                "target": "frontolimbic_regulatory_imbalance",
                "relation": "can bias emotional regulation circuitry over development",
                "smd_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "stress_arousal_sensitization",
                "relation": "deepens stress physiology and persistent arousal coupling",
                "smd_change": "increased",
            },
            {
                "source": "sensation_seeking_trait",
                "target": "dopaminergic_incentive_salience",
                "relation": "raises motivational pull toward intense experiences",
                "smd_change": "increased",
            },
            {
                "source": "affective_instability_trait",
                "target": "frontolimbic_regulatory_imbalance",
                "relation": "adds volatility to limbic and control-system balance",
                "smd_change": "increased",
            },
            {
                "source": "ritualized_submission_learning",
                "target": "submission_relief_loop",
                "relation": "conditions surrender and restraint as reinforcing",
                "smd_change": "increased",
            },
            {
                "source": "ritualized_submission_learning",
                "target": "prefrontal_reappraisal_bias",
                "relation": "shapes meaning-making around submission and perceived lack of control",
                "smd_change": "increased",
            },
            {
                "source": "humiliation_association_learning",
                "target": "social_pain_eroticization",
                "relation": "conditions humiliation and degradation as arousing signals",
                "smd_change": "increased",
            },
            {
                "source": "physical_pain_salience",
                "target": "pain_reward_coupling",
                "relation": "makes painful stimulation more available for conditioning into reward",
                "smd_change": "increased",
            },
            {
                "source": "physical_pain_salience",
                "target": "interoceptive_amplification",
                "relation": "intensifies bodily awareness of pain and arousal",
                "smd_change": "increased",
            },
            {
                "source": "stress_arousal_sensitization",
                "target": "endogenous_opioid_reinforcement",
                "relation": "prolonged or intense stress may recruit relief-like opioid reinforcement",
                "smd_change": "increased",
            },
            {
                "source": "endogenous_opioid_reinforcement",
                "target": "pain_reward_coupling",
                "relation": "supports pleasurable or relieving response to painful stimulation",
                "smd_change": "increased",
            },
            {
                "source": "dopaminergic_incentive_salience",
                "target": "pain_reward_coupling",
                "relation": "adds wanting and reward prediction to pain-associated cues",
                "smd_change": "increased",
            },
            {
                "source": "pain_reward_coupling",
                "target": "ventral_striatum_proxy",
                "relation": "links painful stimulation to accumbens-type reward processing",
                "smd_change": "increased",
            },
            {
                "source": "social_pain_eroticization",
                "target": "acc",
                "relation": "engages cingulate reinterpretation of social and physical pain",
                "smd_change": "increased",
            },
            {
                "source": "interoceptive_amplification",
                "target": "insula",
                "relation": "maps bodily arousal and pain awareness onto insular processing",
                "smd_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_imbalance",
                "target": "amygdala",
                "relation": "heightens affective salience and arousal/fear coupling",
                "smd_change": "increased",
            },
            {
                "source": "prefrontal_reappraisal_bias",
                "target": "pfc_control",
                "relation": "recruits prefrontal meaning-making and top-down reinterpretation",
                "smd_change": "increased",
            },
            {
                "source": "prefrontal_reappraisal_bias",
                "target": "acc",
                "relation": "biases cingulate appraisal from aversion toward desirability",
                "smd_change": "increased",
            },
            {
                "source": "pain_reward_coupling",
                "target": "paradoxical_stress_arousal",
                "relation": "supports excitement under painful or stressful cues",
                "smd_change": "increased",
            },
            {
                "source": "pain_reward_coupling",
                "target": "pain_desire_coupling",
                "relation": "directly makes pain motivationally desirable",
                "smd_change": "increased",
            },
            {
                "source": "social_pain_eroticization",
                "target": "humiliation_arousal_coupling",
                "relation": "makes humiliation and degradation sexually salient",
                "smd_change": "increased",
            },
            {
                "source": "submission_relief_loop",
                "target": "submission_seeking",
                "relation": "turns surrender and loss of control into motivational targets",
                "smd_change": "increased",
            },
            {
                "source": "submission_seeking",
                "target": "ritualized_enactment_pressure",
                "relation": "drives repeated ritualized enactment of the learned script",
                "smd_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_imbalance",
                "target": "distress_impairment",
                "relation": "adds broader behavioral and emotional burden",
                "smd_change": "increased",
            },
            {
                "source": "aversive_appraisal_reserve",
                "target": "pain_reward_coupling",
                "relation": "preserves ordinary aversive interpretation of pain",
                "smd_change": "decreased",
            },
            {
                "source": "aversive_appraisal_reserve",
                "target": "social_pain_eroticization",
                "relation": "preserves ordinary aversive interpretation of humiliation and rejection-like signals",
                "smd_change": "decreased",
            },
            {
                "source": "aversive_appraisal_reserve",
                "target": "paradoxical_stress_arousal",
                "relation": "buffers conversion of threat into excitement",
                "smd_change": "decreased",
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
        self._build_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _mean_clip(values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        if not vals:
            return 0.0
        return max(0.0, min(1.0, round(sum(vals) / len(vals), 4)))

    @staticmethod
    def _series_from(mapping: Dict[str, float], name: str) -> pd.Series:
        return pd.Series({k: round(float(v), 4) for k, v in mapping.items()}, name=name)

    def _debug(self, msg: str) -> None:
        if self.debug:
            print(f"[SMD DEBUG] {msg}")

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        for modality in modalities:
            try:
                with siibra.QUIET:
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception as exc:
                self._debug(f"Feature lookup failed for modality={modality!r}: {exc}")
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
        except Exception as exc:
            self._debug(f"Region search failed for query={query!r}: {exc}")
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
            "insula",
            "amygdala",
            "striatum",
            "ventral striatum",
            "nucleus accumbens",
            "anterior cingulate cortex",
            "cingulate gyrus",
            "prefrontal cortex",
        } else 0
        subregion_bonus = 0 if any(
            token in name
            for token in (
                "area ",
                "lb",
                "cm",
                "sf",
                "24",
                "32",
                "8v",
                "46",
                "id1",
                "id2",
                "ig1",
                "fucd",
            )
        ) else 1
        return (left_bonus, right_penalty, generic_penalty, subregion_bonus)

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
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            item = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if item in seen:
                continue
            seen.add(item)
            rows.append(
                {
                    "name": item[0],
                    "identifier": item[1],
                    "parcellation": item[2],
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
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _extract_tabular_data(self, feature: Any) -> pd.DataFrame:
        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if isinstance(data, pd.Series):
                return data.to_frame().reset_index(drop=False)
            if isinstance(data, dict):
                return pd.DataFrame(data)
        except Exception:
            pass
        return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._extract_tabular_data(feats[0]).reset_index(drop=False)
            if "index" in df.columns and "receptor" not in df.columns:
                df = df.rename(columns={"index": "receptor"})
            return df
        except Exception:
            return pd.DataFrame()

    def _gene_symbols_for_query(self, gene: str) -> List[str]:
        aliases = {
            "SLC6A4": ["SLC6A4", "SERT", "5HTT"],
            "SLC6A3": ["SLC6A3", "DAT1", "DAT"],
            "NR3C1": ["NR3C1", "GR"],
            "OPRM1": ["OPRM1"],
            "OPRK1": ["OPRK1"],
            "DRD2": ["DRD2"],
            "DRD4": ["DRD4"],
            "COMT": ["COMT"],
            "MAOA": ["MAOA"],
            "BDNF": ["BDNF"],
            "CRHR1": ["CRHR1"],
            "OXTR": ["OXTR"],
            "FAAH": ["FAAH"],
        }
        return aliases.get(gene, [gene])

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        modalities = self._modality_candidates("gene")

        for gene in genes:
            retrieved = False
            for alias in self._gene_symbols_for_query(str(gene)):
                feats = self._safe_features_any(region, modalities, gene=[alias])
                if not feats:
                    continue
                df = self._extract_tabular_data(feats[0])
                if df.empty:
                    continue
                if "gene" not in {str(c).lower() for c in df.columns}:
                    df = df.copy()
                    df["gene"] = str(gene)
                frames.append(df)
                retrieved = True
                break
            if not retrieved:
                self._debug(f"No gene-expression rows found for gene={gene!r} in region={self._name_of(region)!r}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True, sort=False)
        lower_cols = {str(c).lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            try:
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
            except Exception:
                return df.reset_index(drop=True)
        return df.reset_index(drop=True)

    def _candidate_connectivity_feature(self) -> Any:
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            return None
        cohort_match = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), None)
        return cohort_match if cohort_match is not None else feats[0]

    def _matrix_like(self, obj: Any) -> pd.DataFrame:
        try:
            data = getattr(obj, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
        except Exception:
            pass
        return pd.DataFrame()

    def _average_matrices(self, matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
        usable = [m for m in matrices if isinstance(m, pd.DataFrame) and not m.empty]
        if not usable:
            return pd.DataFrame()
        if len(usable) == 1:
            return usable[0].copy()
        try:
            running = usable[0].copy()
            for m in usable[1:]:
                running = running.add(m, fill_value=0.0)
            return running / float(len(usable))
        except Exception:
            return usable[0].copy()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feat = self._candidate_connectivity_feature()
        if feat is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        matrices: List[pd.DataFrame] = []
        try:
            for idx, element in enumerate(feat):
                mat = self._matrix_like(element)
                if not mat.empty:
                    matrices.append(mat)
                if idx + 1 >= self.aggregate_connectivity_subjects:
                    break
        except Exception:
            pass

        if not matrices:
            direct = self._matrix_like(feat)
            if not direct.empty:
                matrices = [direct]

        self._connectivity_matrix = self._average_matrices(matrices)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = getattr(region, "name", self._name_of(region))
        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]

        rn = region_name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        region_tokens = {
            t for t in rn.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(t) > 2
        }
        best = None
        best_score = 0
        for label in labels:
            ln = self._name_of(label).lower()
            label_tokens = {
                t for t in ln.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(t) > 2
            }
            score = len(region_tokens & label_tokens)
            if score > best_score:
                best = label
                best_score = score
        return best if best_score > 0 else None

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
            df = df[df["connected_region"] != getattr(region, "name", self._name_of(region))]
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public atlas helpers
    # ------------------------------------------------------------------
    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )
        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        if isinstance(assignments, pd.DataFrame):
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            raise KeyError(f"No resolved region object for node '{node_key}'")

        for method_name in ("get_regional_mask", "fetch_regional_map"):
            method = getattr(region, method_name, None)
            if method is None:
                continue
            try:
                return method(space=self.assignment_space, maptype="labelled")
            except TypeError:
                try:
                    return method(self.assignment_space)
                except Exception:
                    continue
            except Exception:
                continue
        raise RuntimeError(f"Could not fetch a region mask for node '{node_key}'")

    # ------------------------------------------------------------------
    # Build and circuit connectivity summaries
    # ------------------------------------------------------------------
    def circuit_connectivity(self, max_rows_per_seed: int = 10) -> pd.DataFrame:
        if not self.region_objects:
            self.build(connectivity_rows=max_rows_per_seed)

        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        labels_index = list(matrix.index)
        labels_columns = list(matrix.columns)

        for seed_key, seed_region in self.region_objects.items():
            seed_label = self._match_region_label(labels_index, seed_region)
            axis = "index"
            if seed_label is None:
                seed_label = self._match_region_label(labels_columns, seed_region)
                axis = "columns"
            if seed_label is None:
                continue

            try:
                series = matrix.loc[seed_label] if axis == "index" else matrix[seed_label]
                series = series.sort_values(ascending=False)
            except Exception:
                continue

            taken = 0
            for target_label, value in series.items():
                target_name = self._name_of(target_label)
                if target_name == getattr(seed_region, "name", self._name_of(seed_region)):
                    continue
                target_key = None
                for key, region in self.region_objects.items():
                    if self._name_of(region) == target_name:
                        target_key = key
                        break
                rows.append(
                    {
                        "seed_key": seed_key,
                        "seed_region": getattr(seed_region, "name", self._name_of(seed_region)),
                        "target_key": target_key,
                        "target_region": target_name,
                        "value": float(value),
                    }
                )
                taken += 1
                if taken >= max_rows_per_seed:
                    break

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["seed_key", "value"], ascending=[True, False]).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = SEXUAL_MASOCHISM_DISORDER_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        # Reset build-side caches in case build() is run repeatedly.
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

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
            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": "Atlas-backed node (unresolved in this environment)",
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
                    "label": getattr(region, "name", self._name_of(region)),
                    "node_type": "region",
                    "description": "Atlas-backed circuit node",
                    "atlas_region": getattr(region, "name", self._name_of(region)),
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
        circuit_df = self.circuit_connectivity(max_rows_per_seed=connectivity_rows)

        self._build_cache = {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
        }
        return self._build_cache

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulate(
        self,
        early_life_adversity: float = 0.35,
        chronic_stress_load: float = 0.40,
        sensation_seeking_trait: float = 0.45,
        affective_instability_trait: float = 0.30,
        ritualized_submission_learning: float = 0.55,
        humiliation_association_learning: float = 0.45,
        physical_pain_salience: float = 0.50,
        aversive_appraisal_reserve: float = 0.20,
    ) -> Dict[str, pd.Series]:
        inp = {
            "early_life_adversity": self._clip01(early_life_adversity),
            "chronic_stress_load": self._clip01(chronic_stress_load),
            "sensation_seeking_trait": self._clip01(sensation_seeking_trait),
            "affective_instability_trait": self._clip01(affective_instability_trait),
            "ritualized_submission_learning": self._clip01(ritualized_submission_learning),
            "humiliation_association_learning": self._clip01(humiliation_association_learning),
            "physical_pain_salience": self._clip01(physical_pain_salience),
            "aversive_appraisal_reserve": self._clip01(aversive_appraisal_reserve),
        }

        lat: Dict[str, float] = {}
        lat["stress_arousal_sensitization"] = self._clip01(
            0.34 * inp["early_life_adversity"]
            + 0.34 * inp["chronic_stress_load"]
            + 0.12 * inp["affective_instability_trait"]
            + 0.08 * inp["physical_pain_salience"]
            - 0.18 * inp["aversive_appraisal_reserve"]
        )
        lat["endogenous_opioid_reinforcement"] = self._clip01(
            0.28 * lat["stress_arousal_sensitization"]
            + 0.24 * inp["physical_pain_salience"]
            + 0.18 * inp["ritualized_submission_learning"]
            + 0.12 * inp["humiliation_association_learning"]
        )
        lat["dopaminergic_incentive_salience"] = self._clip01(
            0.30 * inp["sensation_seeking_trait"]
            + 0.22 * lat["endogenous_opioid_reinforcement"]
            + 0.20 * inp["ritualized_submission_learning"]
            + 0.14 * lat["stress_arousal_sensitization"]
        )
        lat["pain_reward_coupling"] = self._clip01(
            0.28 * inp["physical_pain_salience"]
            + 0.26 * lat["endogenous_opioid_reinforcement"]
            + 0.22 * lat["dopaminergic_incentive_salience"]
            + 0.10 * lat["stress_arousal_sensitization"]
            - 0.22 * inp["aversive_appraisal_reserve"]
        )
        lat["social_pain_eroticization"] = self._clip01(
            0.30 * inp["humiliation_association_learning"]
            + 0.22 * inp["ritualized_submission_learning"]
            + 0.20 * lat["pain_reward_coupling"]
            + 0.12 * inp["affective_instability_trait"]
            - 0.20 * inp["aversive_appraisal_reserve"]
        )
        lat["interoceptive_amplification"] = self._clip01(
            0.32 * lat["stress_arousal_sensitization"]
            + 0.28 * inp["physical_pain_salience"]
            + 0.20 * lat["pain_reward_coupling"]
            + 0.08 * inp["humiliation_association_learning"]
        )
        lat["prefrontal_reappraisal_bias"] = self._clip01(
            0.30 * inp["ritualized_submission_learning"]
            + 0.24 * inp["humiliation_association_learning"]
            + 0.16 * lat["stress_arousal_sensitization"]
            + 0.16 * lat["social_pain_eroticization"]
        )
        lat["frontolimbic_regulatory_imbalance"] = self._clip01(
            0.26 * inp["early_life_adversity"]
            + 0.22 * inp["affective_instability_trait"]
            + 0.18 * inp["chronic_stress_load"]
            + 0.14 * lat["social_pain_eroticization"]
            + 0.12 * lat["prefrontal_reappraisal_bias"]
            - 0.12 * inp["aversive_appraisal_reserve"]
        )
        lat["submission_relief_loop"] = self._clip01(
            0.30 * inp["ritualized_submission_learning"]
            + 0.22 * lat["pain_reward_coupling"]
            + 0.18 * lat["social_pain_eroticization"]
            + 0.16 * lat["dopaminergic_incentive_salience"]
            + 0.08 * lat["stress_arousal_sensitization"]
        )

        regional = {
            "insula": self._clip01(
                0.38 * lat["interoceptive_amplification"]
                + 0.22 * lat["pain_reward_coupling"]
                + 0.14 * lat["stress_arousal_sensitization"]
                + 0.08 * lat["social_pain_eroticization"]
            ),
            "acc": self._clip01(
                0.28 * lat["prefrontal_reappraisal_bias"]
                + 0.24 * lat["social_pain_eroticization"]
                + 0.18 * lat["pain_reward_coupling"]
                + 0.12 * lat["frontolimbic_regulatory_imbalance"]
                - 0.16 * inp["aversive_appraisal_reserve"]
            ),
            "amygdala": self._clip01(
                0.34 * lat["stress_arousal_sensitization"]
                + 0.24 * lat["frontolimbic_regulatory_imbalance"]
                + 0.18 * lat["social_pain_eroticization"]
            ),
            "ventral_striatum_proxy": self._clip01(
                0.36 * lat["dopaminergic_incentive_salience"]
                + 0.26 * lat["endogenous_opioid_reinforcement"]
                + 0.18 * lat["pain_reward_coupling"]
                + 0.10 * lat["submission_relief_loop"]
            ),
            "pfc_control": self._clip01(
                0.30 * lat["prefrontal_reappraisal_bias"]
                + 0.24 * lat["frontolimbic_regulatory_imbalance"]
                + 0.14 * lat["submission_relief_loop"]
                + 0.10 * lat["social_pain_eroticization"]
            ),
        }

        symptoms: Dict[str, float] = {}
        symptoms["paradoxical_stress_arousal"] = self._clip01(
            0.30 * lat["stress_arousal_sensitization"]
            + 0.22 * lat["interoceptive_amplification"]
            + 0.18 * lat["pain_reward_coupling"]
            + 0.12 * regional["insula"]
            - 0.22 * inp["aversive_appraisal_reserve"]
        )
        symptoms["pain_desire_coupling"] = self._clip01(
            0.36 * lat["pain_reward_coupling"]
            + 0.22 * regional["ventral_striatum_proxy"]
            + 0.18 * lat["endogenous_opioid_reinforcement"]
            + 0.10 * regional["insula"]
        )
        symptoms["humiliation_arousal_coupling"] = self._clip01(
            0.34 * lat["social_pain_eroticization"]
            + 0.22 * regional["acc"]
            + 0.16 * regional["amygdala"]
            + 0.12 * lat["prefrontal_reappraisal_bias"]
        )
        symptoms["submission_seeking"] = self._clip01(
            0.30 * lat["submission_relief_loop"]
            + 0.22 * symptoms["humiliation_arousal_coupling"]
            + 0.20 * regional["ventral_striatum_proxy"]
            + 0.10 * regional["pfc_control"]
        )
        symptoms["ritualized_enactment_pressure"] = self._clip01(
            0.26 * symptoms["submission_seeking"]
            + 0.24 * symptoms["pain_desire_coupling"]
            + 0.18 * symptoms["humiliation_arousal_coupling"]
            + 0.12 * lat["dopaminergic_incentive_salience"]
        )
        symptoms["distress_impairment"] = self._clip01(
            0.26 * lat["frontolimbic_regulatory_imbalance"]
            + 0.20 * symptoms["paradoxical_stress_arousal"]
            + 0.18 * symptoms["ritualized_enactment_pressure"]
            + 0.16 * inp["affective_instability_trait"]
            + 0.10 * inp["chronic_stress_load"]
        )

        phenotypes = {
            "pain_reward_dominant_profile": self._mean_clip(
                [
                    lat["pain_reward_coupling"],
                    regional["ventral_striatum_proxy"],
                    symptoms["pain_desire_coupling"],
                    symptoms["ritualized_enactment_pressure"],
                ]
            ),
            "humiliation_submission_profile": self._mean_clip(
                [
                    lat["social_pain_eroticization"],
                    regional["acc"],
                    symptoms["humiliation_arousal_coupling"],
                    symptoms["submission_seeking"],
                ]
            ),
            "stress_arousal_conversion_profile": self._mean_clip(
                [
                    lat["stress_arousal_sensitization"],
                    regional["insula"],
                    regional["amygdala"],
                    symptoms["paradoxical_stress_arousal"],
                ]
            ),
            "frontolimbic_reappraisal_profile": self._mean_clip(
                [
                    lat["prefrontal_reappraisal_bias"],
                    lat["frontolimbic_regulatory_imbalance"],
                    regional["pfc_control"],
                    symptoms["distress_impairment"],
                ]
            ),
        }

        return {
            "inputs": self._series_from(inp, "inputs"),
            "latents": self._series_from(lat, "latents"),
            "regional_state": self._series_from(regional, "regional_state"),
            "symptoms": self._series_from(symptoms, "symptoms"),
            "phenotypes": self._series_from(phenotypes, "phenotypes"),
        }


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    model = SexualMasochismDisorderModel()
    bundle = model.build()

    print("\n=== NODE TABLE (key fields) ===")
    print(
        bundle["nodes"][["key", "node_type", "atlas_region", "region_identifier", "feature_summary"]]
        .fillna("")
        .to_string(index=False)
    )

    print("\n=== EDGE TABLE ===")
    print(bundle["edges"].to_string(index=False))

    for region_key in ["insula", "acc", "amygdala", "ventral_striatum_proxy", "pfc_control"]:
        receptor_df = bundle["receptors"].get(region_key, pd.DataFrame())
        gene_df = bundle["genes"].get(region_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(region_key, pd.DataFrame())

        print(f"\n=== REGION: {region_key} ===")
        print("Receptors:")
        print(receptor_df.head(10).to_string(index=False) if not receptor_df.empty else "<no receptor fingerprint available>")
        print("\nGenes:")
        print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "<no gene-expression table available>")
        print("\nConnectivity:")
        print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "<no connectivity profile available>")

    print("\n=== CIRCUIT CONNECTIVITY ===")
    circuit_df = bundle["circuit_connectivity"]
    print(circuit_df.head(25).to_string(index=False) if not circuit_df.empty else "<no circuit connectivity available>")

    example = model.simulate(
        early_life_adversity=0.50,
        chronic_stress_load=0.55,
        sensation_seeking_trait=0.50,
        affective_instability_trait=0.35,
        ritualized_submission_learning=0.70,
        humiliation_association_learning=0.60,
        physical_pain_salience=0.55,
        aversive_appraisal_reserve=0.15,
    )

    print("\n=== SIMULATION: example pain-reward / submission-linked SMD profile ===")
    for name, series in example.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Example coordinate usage:
    # assignments = model.assign_mni_point((-8, 18, 36))
    # print(assignments.head(10).to_string(index=False))
