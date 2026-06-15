from __future__ import annotations

"""
Exhibitionistic Disorder siibra scaffold.

This script turns a chapter-level biological summary of Exhibitionistic Disorder
into a conservative, atlas-grounded mechanistic scaffold using siibra where
available. It is intended for research prototyping and transparent hypothesis
exploration only. It is not a diagnostic, prognostic, forensic, or treatment
tool.

Core modeling choices from the chapter:
- The chapter frames exhibitionistic behaviour as an imbalance between an
  overactive reward / limbic "go" system and underpowered prefrontal "stop"
  control.
- Dopamine-mediated incentive salience, cue conditioning, developmental
  androgen-pathway influences, and trait-like disinhibition are treated as the
  main latent biological drivers.
- Region anchors are conservative and hypothesis-driven, because the chapter
  explicitly notes that disorder-specific structural or functional neuroimaging
  evidence is limited.
- The simulator is intentionally simple and acyclic:
  inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

The script degrades gracefully:
- If siibra is not installed, or if a feature/modality is unavailable, the
  atlas-backed parts stay empty instead of crashing.
- The simulator still runs even without atlas data.

Source chapter themes encoded here include:
- dopamine-linked reward seeking and conditioned sexual cues,
- nucleus-accumbens / ventral-striatal incentive salience,
- amygdala and limbic reward reactivity,
- prefrontal and anterior-cingulate control failure,
- developmental androgen signaling and sex-differentiation pathways,
- genetic and family-environment contributions to disinhibition,
- repetitive urges and enactment despite adverse consequences.
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
    # Dopamine / reward salience
    "DRD2",
    "DRD3",
    "SLC6A3",
    "COMT",
    "MAOA",
    # General impulse control / affect regulation
    "SLC6A4",
    "BDNF",
    # Developmental androgen signaling / sex differentiation
    "AR",
    "SRD5A2",
    "HSD17B3",
    "CYP19A1",
    "ESR1",
]


class ExhibitionisticDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Exhibitionistic Disorder.

    Notes
    -----
    - This is a research scaffold, not a validated disorder model.
    - Regional values in `simulate()` quantify dysregulation / functional burden,
      not healthy activation.
    - Because the chapter explicitly reports limited disorder-specific imaging,
      region nodes should be interpreted as conservative circuit hypotheses.
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

        # Region anchors are deliberately conservative because the chapter does
        # not offer disorder-specific imaging findings, only testable circuit
        # hypotheses.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "LA (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens left",
                "accumbens left",
                "ventral striatum left",
                "ventral striatum",
                "striatum left",
                "striatum",
            ],
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "dorsolateral prefrontal cortex left",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex",
            ],
            "acc": [
                "Area p32 left",
                "Area p24ab left",
                "Area a24pr left",
                "Area 24 left",
                "Area 32 left",
                "anterior cingulate cortex left",
                "anterior cingulate cortex",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Amygdala anchor for salience tagging, affective relevance, and "
                "cue-triggered limbic reactivity hypothesized in the chapter."
            ),
            "ventral_striatum_proxy": (
                "Ventral striatal / nucleus-accumbens proxy for dopamine-mediated "
                "reward anticipation and conditioned incentive salience."
            ),
            "pfc_control": (
                "Prefrontal top-down control proxy centered on dorsolateral PFC-like "
                "executive inhibition and judgment."
            ),
            "acc": (
                "Anterior cingulate control / conflict-monitoring anchor relevant to "
                "urge-control competition and response inhibition."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_liability": (
                "Polygenic or familial liability affecting reward salience, disinhibition, "
                "and sexual-behaviour regulation."
            ),
            "developmental_androgen_variation": (
                "Variation in androgen-pathway or sex-differentiation influences shaping "
                "sexual-behaviour circuitry during development."
            ),
            "disinhibitory_trait_load": (
                "Trait impulsivity, sensation seeking, and poor baseline self-regulation."
            ),
            "family_adversity_history": (
                "Abuse history or dysfunctional family environment that can amplify later "
                "vulnerability and maladaptive learning."
            ),
            "cue_conditioning_strength": (
                "Strength of learned associations linking public exposure contexts and "
                "sexual arousal."
            ),
            "dopamine_agonist_exposure": (
                "Pharmacologic dopaminergic stimulation that can increase compulsive sexual urges."
            ),
            "acute_self_regulatory_depletion": (
                "Temporary reduction in restraint capacity under stress, depletion, or "
                "other contexts that weaken self-control."
            ),
            "inhibitory_support": (
                "Protective external structure, supervision, or treatment support that "
                "reduces urge enactment opportunities."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "trait_disinhibition": (
                "Stable tendency toward weak restraint and high sensation-seeking that "
                "amplifies risk for compulsive enactment."
            ),
            "androgenic_developmental_bias": (
                "Developmental organization of sexual-behaviour circuitry influenced by "
                "androgen signaling pathways."
            ),
            "dopaminergic_incentive_salience": (
                "Reward-system capture in which exposure-related cues gain excessive wanting value."
            ),
            "conditioned_cue_reactivity": (
                "Learned cue-triggered arousal and craving when relevant social contexts are encountered."
            ),
            "limbic_reward_hyperreactivity": (
                "Overactive limbic and reward responses to paraphilic cues."
            ),
            "frontostriatal_control_failure": (
                "Weak top-down regulation of reward urges by prefrontal control systems."
            ),
            "compulsive_urge_loop": (
                "Self-reinforcing repetition of urges and enactment despite negative consequences."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "intrusive_exposure_urge": (
                "Intense repetitive urge to expose oneself in a cue-linked context."
            ),
            "cue_triggered_craving": (
                "Craving-like increase in arousal and anticipation when conditioned cues are present."
            ),
            "inhibitory_control_failure": (
                "Failure to restrain the impulse despite awareness that the behaviour is inappropriate."
            ),
            "public_exposure_enactment": (
                "Execution of exhibitionistic behaviour in real-world public or stranger contexts."
            ),
            "repetition_despite_consequences": (
                "Continued repetitive behaviour despite adverse legal, social, or personal outcomes."
            ),
            "ego_dystonic_conflict": (
                "Conflict between urges and explicit judgment that the behaviour is inappropriate."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_liability",
                "target": "trait_disinhibition",
                "relation": "familial and heritable influences increase baseline disinhibition risk",
                "exhibitionism_change": "increased",
            },
            {
                "source": "genetic_liability",
                "target": "androgenic_developmental_bias",
                "relation": "genetic variation shapes developmental sexual-behaviour circuitry",
                "exhibitionism_change": "increased",
            },
            {
                "source": "developmental_androgen_variation",
                "target": "androgenic_developmental_bias",
                "relation": "androgen-pathway variation alters organization of sexual preference circuitry",
                "exhibitionism_change": "increased",
            },
            {
                "source": "disinhibitory_trait_load",
                "target": "trait_disinhibition",
                "relation": "impulsivity and sensation-seeking increase weak-restraint bias",
                "exhibitionism_change": "increased",
            },
            {
                "source": "family_adversity_history",
                "target": "trait_disinhibition",
                "relation": "adverse family environments may amplify later behavioural dysregulation",
                "exhibitionism_change": "increased",
            },
            {
                "source": "cue_conditioning_strength",
                "target": "conditioned_cue_reactivity",
                "relation": "learned public-exposure cues become strong triggers of arousal and wanting",
                "exhibitionism_change": "increased",
            },
            {
                "source": "dopamine_agonist_exposure",
                "target": "dopaminergic_incentive_salience",
                "relation": "dopaminergic stimulation intensifies compulsive sexual urges",
                "exhibitionism_change": "increased",
            },
            {
                "source": "acute_self_regulatory_depletion",
                "target": "frontostriatal_control_failure",
                "relation": "temporary depletion weakens the ability to inhibit sexual impulses",
                "exhibitionism_change": "increased",
            },
            {
                "source": "trait_disinhibition",
                "target": "frontostriatal_control_failure",
                "relation": "stable disinhibition undermines top-down control",
                "exhibitionism_change": "increased",
            },
            {
                "source": "androgenic_developmental_bias",
                "target": "dopaminergic_incentive_salience",
                "relation": "developmental sexual-circuit bias can potentiate reward salience",
                "exhibitionism_change": "increased",
            },
            {
                "source": "dopaminergic_incentive_salience",
                "target": "conditioned_cue_reactivity",
                "relation": "reward salience makes conditioned cues more motivationally potent",
                "exhibitionism_change": "increased",
            },
            {
                "source": "conditioned_cue_reactivity",
                "target": "limbic_reward_hyperreactivity",
                "relation": "conditioned cues heighten limbic and reward-system responses",
                "exhibitionism_change": "increased",
            },
            {
                "source": "dopaminergic_incentive_salience",
                "target": "ventral_striatum_proxy",
                "relation": "dopamine-mediated wanting engages nucleus-accumbens-like reward circuitry",
                "exhibitionism_change": "hyperactive",
            },
            {
                "source": "conditioned_cue_reactivity",
                "target": "amygdala",
                "relation": "conditioned sexual cues recruit affective salience processing",
                "exhibitionism_change": "hyperactive",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "pfc_control",
                "relation": "weak top-down regulation compromises prefrontal inhibitory control",
                "exhibitionism_change": "dysregulated",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "acc",
                "relation": "response-conflict monitoring and inhibitory engagement become inefficient",
                "exhibitionism_change": "dysregulated",
            },
            {
                "source": "limbic_reward_hyperreactivity",
                "target": "intrusive_exposure_urge",
                "relation": "overactive reward responses intensify repetitive urges",
                "exhibitionism_change": "increased",
            },
            {
                "source": "conditioned_cue_reactivity",
                "target": "cue_triggered_craving",
                "relation": "conditioned contexts trigger craving-like arousal",
                "exhibitionism_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "inhibitory_control_failure",
                "relation": "prefrontal dysregulation weakens restraint over impulses",
                "exhibitionism_change": "increased",
            },
            {
                "source": "acc",
                "target": "inhibitory_control_failure",
                "relation": "impaired conflict monitoring reduces successful inhibition",
                "exhibitionism_change": "increased",
            },
            {
                "source": "intrusive_exposure_urge",
                "target": "public_exposure_enactment",
                "relation": "strong urges increase the probability of acting out in public contexts",
                "exhibitionism_change": "increased",
            },
            {
                "source": "cue_triggered_craving",
                "target": "public_exposure_enactment",
                "relation": "cue-linked craving raises enactment pressure",
                "exhibitionism_change": "increased",
            },
            {
                "source": "inhibitory_control_failure",
                "target": "public_exposure_enactment",
                "relation": "failed restraint allows the behaviour to be executed",
                "exhibitionism_change": "increased",
            },
            {
                "source": "public_exposure_enactment",
                "target": "repetition_despite_consequences",
                "relation": "repeated acting out persists despite adverse consequences",
                "exhibitionism_change": "increased",
            },
            {
                "source": "intrusive_exposure_urge",
                "target": "ego_dystonic_conflict",
                "relation": "awareness of inappropriate urges generates internal conflict",
                "exhibitionism_change": "increased",
            },
            {
                "source": "inhibitory_support",
                "target": "frontostriatal_control_failure",
                "relation": "external support reduces enactment pressure and improves inhibition",
                "exhibitionism_change": "decreased",
            },
            {
                "source": "inhibitory_support",
                "target": "public_exposure_enactment",
                "relation": "protective structure reduces opportunities for behavioural execution",
                "exhibitionism_change": "decreased",
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
            "striatum",
            "ventral striatum",
            "prefrontal cortex",
            "anterior cingulate cortex",
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
        genetic_liability: float = 0.40,
        developmental_androgen_variation: float = 0.45,
        disinhibitory_trait_load: float = 0.60,
        family_adversity_history: float = 0.30,
        cue_conditioning_strength: float = 0.70,
        dopamine_agonist_exposure: float = 0.00,
        acute_self_regulatory_depletion: float = 0.40,
        inhibitory_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Values are clipped to [0, 1]. Higher regional-state values mean greater
        dysregulation / control failure in that circuit node.
        """
        inputs = pd.Series(
            {
                "genetic_liability": self._clip01(genetic_liability),
                "developmental_androgen_variation": self._clip01(developmental_androgen_variation),
                "disinhibitory_trait_load": self._clip01(disinhibitory_trait_load),
                "family_adversity_history": self._clip01(family_adversity_history),
                "cue_conditioning_strength": self._clip01(cue_conditioning_strength),
                "dopamine_agonist_exposure": self._clip01(dopamine_agonist_exposure),
                "acute_self_regulatory_depletion": self._clip01(acute_self_regulatory_depletion),
                "inhibitory_support": self._clip01(inhibitory_support),
            },
            name="value",
        )

        # Inputs -> latent biology
        trait_disinhibition = self._clip01(
            0.35 * inputs["disinhibitory_trait_load"]
            + 0.25 * inputs["genetic_liability"]
            + 0.20 * inputs["family_adversity_history"]
            + 0.10 * inputs["acute_self_regulatory_depletion"]
            - 0.15 * inputs["inhibitory_support"]
        )

        androgenic_developmental_bias = self._clip01(
            0.60 * inputs["developmental_androgen_variation"]
            + 0.25 * inputs["genetic_liability"]
            + 0.05 * inputs["family_adversity_history"]
        )

        dopaminergic_incentive_salience = self._clip01(
            0.30 * inputs["cue_conditioning_strength"]
            + 0.25 * inputs["dopamine_agonist_exposure"]
            + 0.20 * inputs["genetic_liability"]
            + 0.15 * androgenic_developmental_bias
            + 0.05 * inputs["acute_self_regulatory_depletion"]
        )

        conditioned_cue_reactivity = self._clip01(
            0.45 * inputs["cue_conditioning_strength"]
            + 0.20 * dopaminergic_incentive_salience
            + 0.15 * inputs["family_adversity_history"]
            + 0.10 * androgenic_developmental_bias
            + 0.05 * inputs["acute_self_regulatory_depletion"]
        )

        limbic_reward_hyperreactivity = self._clip01(
            0.35 * dopaminergic_incentive_salience
            + 0.30 * conditioned_cue_reactivity
            + 0.20 * androgenic_developmental_bias
            + 0.10 * trait_disinhibition
        )

        frontostriatal_control_failure = self._clip01(
            0.35 * trait_disinhibition
            + 0.25 * inputs["acute_self_regulatory_depletion"]
            + 0.15 * conditioned_cue_reactivity
            + 0.10 * inputs["dopamine_agonist_exposure"]
            + 0.05 * androgenic_developmental_bias
            - 0.25 * inputs["inhibitory_support"]
        )

        compulsive_urge_loop = self._clip01(
            0.30 * conditioned_cue_reactivity
            + 0.25 * limbic_reward_hyperreactivity
            + 0.20 * frontostriatal_control_failure
            + 0.15 * dopaminergic_incentive_salience
            - 0.10 * inputs["inhibitory_support"]
        )

        latents = pd.Series(
            {
                "trait_disinhibition": trait_disinhibition,
                "androgenic_developmental_bias": androgenic_developmental_bias,
                "dopaminergic_incentive_salience": dopaminergic_incentive_salience,
                "conditioned_cue_reactivity": conditioned_cue_reactivity,
                "limbic_reward_hyperreactivity": limbic_reward_hyperreactivity,
                "frontostriatal_control_failure": frontostriatal_control_failure,
                "compulsive_urge_loop": compulsive_urge_loop,
            },
            name="value",
        )

        # Latent biology -> regional dysregulation burden
        regional_state = pd.Series(
            {
                "amygdala": self._clip01(
                    0.40 * conditioned_cue_reactivity
                    + 0.25 * limbic_reward_hyperreactivity
                    + 0.15 * inputs["family_adversity_history"]
                    + 0.10 * inputs["acute_self_regulatory_depletion"]
                ),
                "ventral_striatum_proxy": self._clip01(
                    0.40 * dopaminergic_incentive_salience
                    + 0.25 * conditioned_cue_reactivity
                    + 0.20 * limbic_reward_hyperreactivity
                    + 0.10 * inputs["dopamine_agonist_exposure"]
                ),
                "pfc_control": self._clip01(
                    0.40 * frontostriatal_control_failure
                    + 0.20 * trait_disinhibition
                    + 0.15 * inputs["acute_self_regulatory_depletion"]
                    + 0.10 * compulsive_urge_loop
                    - 0.20 * inputs["inhibitory_support"]
                ),
                "acc": self._clip01(
                    0.35 * frontostriatal_control_failure
                    + 0.25 * conditioned_cue_reactivity
                    + 0.15 * inputs["acute_self_regulatory_depletion"]
                    + 0.10 * compulsive_urge_loop
                    - 0.15 * inputs["inhibitory_support"]
                ),
            },
            name="value",
        )

        # Regional dysregulation -> symptoms
        intrusive_exposure_urge = self._clip01(
            0.35 * compulsive_urge_loop
            + 0.25 * regional_state["ventral_striatum_proxy"]
            + 0.20 * regional_state["amygdala"]
            + 0.10 * androgenic_developmental_bias
        )

        cue_triggered_craving = self._clip01(
            0.40 * conditioned_cue_reactivity
            + 0.30 * regional_state["ventral_striatum_proxy"]
            + 0.15 * regional_state["amygdala"]
            + 0.10 * dopaminergic_incentive_salience
        )

        inhibitory_control_failure = self._clip01(
            0.35 * regional_state["pfc_control"]
            + 0.25 * regional_state["acc"]
            + 0.20 * frontostriatal_control_failure
            + 0.10 * inputs["acute_self_regulatory_depletion"]
            - 0.15 * inputs["inhibitory_support"]
        )

        public_exposure_enactment = self._clip01(
            0.30 * intrusive_exposure_urge
            + 0.25 * inhibitory_control_failure
            + 0.20 * cue_triggered_craving
            + 0.10 * inputs["acute_self_regulatory_depletion"]
            + 0.05 * regional_state["amygdala"]
            - 0.15 * inputs["inhibitory_support"]
        )

        repetition_despite_consequences = self._clip01(
            0.35 * public_exposure_enactment
            + 0.25 * compulsive_urge_loop
            + 0.20 * regional_state["ventral_striatum_proxy"]
            + 0.10 * inhibitory_control_failure
        )

        ego_dystonic_conflict = self._clip01(
            0.35 * intrusive_exposure_urge
            + 0.25 * inhibitory_control_failure
            + 0.20 * regional_state["acc"]
            + 0.10 * repetition_despite_consequences
        )

        symptoms = pd.Series(
            {
                "intrusive_exposure_urge": intrusive_exposure_urge,
                "cue_triggered_craving": cue_triggered_craving,
                "inhibitory_control_failure": inhibitory_control_failure,
                "public_exposure_enactment": public_exposure_enactment,
                "repetition_despite_consequences": repetition_despite_consequences,
                "ego_dystonic_conflict": ego_dystonic_conflict,
            },
            name="value",
        )

        # Symptom bundles / phenotype summaries
        compulsive_exhibitionism_profile = self._clip01(
            0.30 * intrusive_exposure_urge
            + 0.25 * public_exposure_enactment
            + 0.20 * repetition_despite_consequences
            + 0.15 * inhibitory_control_failure
            + 0.05 * compulsive_urge_loop
        )

        cue_reactive_exhibitionism_profile = self._clip01(
            0.35 * cue_triggered_craving
            + 0.25 * intrusive_exposure_urge
            + 0.20 * regional_state["amygdala"]
            + 0.15 * regional_state["ventral_striatum_proxy"]
        )

        disinhibited_public_risk_profile = self._clip01(
            0.35 * public_exposure_enactment
            + 0.30 * inhibitory_control_failure
            + 0.15 * inputs["acute_self_regulatory_depletion"]
            + 0.10 * regional_state["pfc_control"]
        )

        reward_control_imbalance = self._clip01(
            0.35 * regional_state["ventral_striatum_proxy"]
            + 0.25 * regional_state["pfc_control"]
            + 0.20 * regional_state["acc"]
            + 0.10 * dopaminergic_incentive_salience
        )

        global_exhibitionism_burden = self._clip01(
            0.25 * intrusive_exposure_urge
            + 0.20 * cue_triggered_craving
            + 0.20 * inhibitory_control_failure
            + 0.20 * public_exposure_enactment
            + 0.10 * repetition_despite_consequences
        )

        phenotypes = pd.Series(
            {
                "compulsive_exhibitionism_profile": compulsive_exhibitionism_profile,
                "cue_reactive_exhibitionism_profile": cue_reactive_exhibitionism_profile,
                "disinhibited_public_risk_profile": disinhibited_public_risk_profile,
                "reward_control_imbalance": reward_control_imbalance,
                "global_exhibitionism_burden": global_exhibitionism_burden,
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
    model = ExhibitionisticDisorderModel()

    print("\n=== Building atlas-backed scaffold ===")
    scaffold = model.build(connectivity_rows=10)

    node_cols = ["key", "node_type", "atlas_region", "feature_summary"]
    print("\nNodes:")
    print(scaffold["nodes"][node_cols].to_string(index=False))

    print("\nEdges (first 18):")
    print(scaffold["edges"].head(18).to_string(index=False))

    if "amygdala" in scaffold["receptors"] and not scaffold["receptors"]["amygdala"].empty:
        print("\nAmygdala receptor fingerprint:")
        print(scaffold["receptors"]["amygdala"].head().to_string(index=False))
    else:
        print("\nNo amygdala receptor table available in this environment.")

    if "pfc_control" in scaffold["genes"] and not scaffold["genes"]["pfc_control"].empty:
        print("\nPFC-control gene summary:")
        print(scaffold["genes"]["pfc_control"].head().to_string(index=False))
    else:
        print("\nNo PFC-control gene table available in this environment.")

    if (
        "ventral_striatum_proxy" in scaffold["connectivity_profiles"]
        and not scaffold["connectivity_profiles"]["ventral_striatum_proxy"].empty
    ):
        print("\nVentral striatum connectivity profile:")
        print(scaffold["connectivity_profiles"]["ventral_striatum_proxy"].head().to_string(index=False))
    else:
        print("\nNo ventral striatum connectivity profile available in this environment.")

    if not scaffold["circuit_connectivity"].empty:
        print("\nWithin-model circuit connectivity:")
        print(scaffold["circuit_connectivity"].round(3).to_string())
    else:
        print("\nNo within-model circuit connectivity matrix available in this environment.")

    print("\n=== Simulation example: strong cue conditioning with weak control ===")
    sim = model.simulate(
        genetic_liability=0.50,
        developmental_androgen_variation=0.55,
        disinhibitory_trait_load=0.75,
        family_adversity_history=0.35,
        cue_conditioning_strength=0.85,
        dopamine_agonist_exposure=0.05,
        acute_self_regulatory_depletion=0.55,
        inhibitory_support=0.20,
    )
    for name, series in sim.items():
        print(f"\n{name}:")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-18, 10, -10)).head())

    # Example regional mask retrieval:
    # mask = model.region_mask("amygdala")
    # if mask is not None:
    #     print(mask)
