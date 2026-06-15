from __future__ import annotations

"""
Rapid Eye Movement Sleep Behavior Disorder siibra scaffold.

This script translates a Rapid Eye Movement Sleep Behavior Disorder (RBD)
chapter into a transparent, atlas-grounded research scaffold using siibra.
It is a mechanistic interpretation of chapter logic, not a validated disease
model, diagnostic instrument, or treatment recommender.

The source chapter emphasizes several linked biological themes:
- state dissociation in which complex motor behavior intrudes into REM sleep,
- dysfunction or degeneration of pontine REM-atonia circuitry, including
  SLD-like glutamatergic drivers of motor inhibition,
- strong prodromal coupling to alpha-synucleinopathies, especially pathways
  involving alpha-synuclein and GBA1,
- hypothalamic hypocretin/orexin-linked sleep-wake state instability in
  narcolepsy-associated cases,
- limbic recruitment, especially the amygdala, contributing to vivid fearful
  dream content,
- basal-ganglia involvement in Parkinson-related RBD and REM motor expression,
- secondary or lesional RBD arising from focal brainstem insults, plus early
  microstructural degeneration detectable with advanced imaging.

To stay conservative, the scaffold anchors only regions named or strongly
implied by the chapter and uses proxy nodes where exact Julich labels may vary
or where detailed human brainstem nuclei are not robustly exposed in a given
siibra environment.
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


DEFAULT_RAPID_EYE_MOVEMENT_SLEEP_BEHAVIOR_DISORDER_GENE_PANEL = [
    "SNCA",      # alpha-synuclein / synucleinopathy biology
    "GBA1",      # glucocerebrosidase pathway risk
    "HLA-DQB1",  # narcolepsy-linked state instability signal
    "HCRT",      # hypocretin / orexin precursor
    "HCRTR2",    # orexin receptor 2
    "SLC17A6",   # vesicular glutamate transporter / excitatory pontine output
    "GRIN2B",    # glutamatergic signaling
    "GAD1",      # GABA synthesis
    "GAD2",      # GABA synthesis
    "GLRA1",     # glycinergic motor inhibition
    "SLC6A3",    # dopamine transporter / basal-ganglia relevance
    "DRD2",      # dopamine receptor / motor circuit relevance
]


class RapidEyeMovementSleepBehaviorDisorderModel:
    """
    Atlas-grounded research scaffold for Rapid Eye Movement Sleep Behavior Disorder.

    The model follows a one-pass transparent order:
        inputs -> latent biology -> regional dysfunction burden -> symptoms -> phenotypes

    Important caveats
    -----------------
    - Higher `regional_state` values indicate modeled dysfunction burden or
      maladaptive circuit recruitment, not healthy activation.
    - Several key nodes are modeled as proxies because the chapter is more
      specific at the systems level than at the cytoarchitectonic-parcel level,
      especially for pontine and hypothalamic nuclei.
    - This is a chapter-faithful research scaffold, not a clinically validated
      causal model.
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
            "pontine_rem_atonia_circuit_proxy": [
                "sublaterodorsal",
                "subcoeruleus",
                "pontine tegmentum",
                "pons",
                "brainstem",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "basal_ganglia_proxy": [
                "Putamen left",
                "Caudate nucleus left",
                "striatum left",
                "putamen",
                "caudate",
                "basal ganglia",
                "striatum",
            ],
            "hypothalamus_orexin_proxy": [
                "lateral hypothalamus",
                "hypothalamus left",
                "hypothalamus",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "pontine_rem_atonia_circuit_proxy": (
                "Pontine REM-atonia generator proxy representing SLD/subcoeruleus-like "
                "brainstem circuitry that normally suppresses motor output during REM sleep."
            ),
            "amygdala": (
                "Amygdala node representing limbic recruitment associated with intense "
                "emotional dream content and fearful/aggressive enactment."
            ),
            "basal_ganglia_proxy": (
                "Basal-ganglia proxy capturing Parkinson-linked motor-circuit involvement "
                "and the paradoxical escape of motor behavior during REM dream enactment."
            ),
            "hypothalamus_orexin_proxy": (
                "Hypothalamic orexin/hypocretin proxy representing sleep-wake state "
                "stabilization relevant to narcolepsy-associated RBD."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "alpha_synucleinopathy_liability": (
                "Baseline molecular liability toward alpha-synuclein misfolding and early "
                "synucleinopathic change."
            ),
            "gba1_pathway_genetic_risk": (
                "GBA1-pathway genetic risk that strengthens the overlap between RBD and "
                "Parkinson-spectrum synucleinopathy."
            ),
            "neurodegenerative_progression": (
                "Slowly progressive neurodegenerative burden affecting REM-atonia circuits "
                "over long prodromal intervals."
            ),
            "environmental_toxin_exposure": (
                "Environmental toxin burden, such as pesticide exposure, linked to both RBD "
                "and Parkinsonian neurodegeneration."
            ),
            "glutamatergic_dysregulation_load": (
                "Excitatory glutamatergic dysregulation burden within REM-atonia circuitry."
            ),
            "rem_state_instability": (
                "General instability between REM sleep and wakefulness that allows state "
                "components to mix."
            ),
            "narcolepsy_orexin_instability": (
                "Hypocretin/orexin-linked state instability relevant to narcolepsy-associated RBD."
            ),
            "brainstem_lesion_burden": (
                "Secondary or lesional burden from focal ischemic, demyelinating, tumoral, "
                "or cavernomatous insults in the brainstem."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "alpha_synuclein_accumulation": (
                "Misfolding and aggregation pressure related to alpha-synucleinopathy biology."
            ),
            "pontine_rem_atonia_circuit_degeneration": (
                "Degeneration or dysfunction of pontine REM-atonia circuitry that normally "
                "prevents complex movement in REM sleep."
            ),
            "glutamatergic_instability": (
                "Abnormal excitatory signaling in SLD-like REM-atonia generators."
            ),
            "orexin_state_instability": (
                "Hypocretin/orexin instability weakening normal stabilization of sleep-wake states."
            ),
            "rem_state_dissociation": (
                "Core state-dissociation process in which REM sleep is mixed with waking-like "
                "motor output."
            ),
            "limbic_dream_enactment_drive": (
                "Limbic amplification of emotionally intense dream content likely to be enacted."
            ),
            "basal_ganglia_motor_bypass": (
                "REM motor-expression process that can bypass or escape typical basal-ganglia "
                "constraints in Parkinsonian states."
            ),
            "brainstem_microstructural_degeneration": (
                "Early microstructural degeneration in brainstem-associated white matter and "
                "related circuits detectable with advanced imaging."
            ),
            "prodromal_synucleinopathy_risk": (
                "Latent burden representing RBD as a prodromal state of Parkinson disease, "
                "dementia with Lewy bodies, or multiple system atrophy."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "rem_without_atonia": (
                "Loss of normal REM motor atonia during dream sleep."
            ),
            "dream_enactment_behaviors": (
                "Observable complex motor behaviors that enact dream content during REM sleep."
            ),
            "violent_fearful_dreaming": (
                "Emotionally intense, fearful, aggressive, or violent dream content commonly "
                "reported in RBD."
            ),
            "sleep_related_injury_risk": (
                "Risk of patient or bed-partner injury caused by REM dream enactment."
            ),
            "synucleinopathy_conversion_risk": (
                "Likelihood that apparently isolated RBD reflects prodromal synucleinopathy."
            ),
            "secondary_lesional_rbd_burden": (
                "Burden of secondary RBD arising from focal or structural brainstem pathology."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "alpha_synucleinopathy_liability",
                "target": "alpha_synuclein_accumulation",
                "relation": "baseline synucleinopathy liability increases alpha-synuclein burden",
                "rbd_change": "increased",
            },
            {
                "source": "gba1_pathway_genetic_risk",
                "target": "alpha_synuclein_accumulation",
                "relation": "GBA1-pathway risk strengthens synucleinopathy-related vulnerability",
                "rbd_change": "increased",
            },
            {
                "source": "environmental_toxin_exposure",
                "target": "alpha_synuclein_accumulation",
                "relation": "toxin exposure may increase Parkinson-linked molecular vulnerability",
                "rbd_change": "increased",
            },
            {
                "source": "neurodegenerative_progression",
                "target": "alpha_synuclein_accumulation",
                "relation": "slow neurodegeneration amplifies synucleinopathic burden over time",
                "rbd_change": "increased",
            },
            {
                "source": "neurodegenerative_progression",
                "target": "brainstem_microstructural_degeneration",
                "relation": "progressive degeneration produces early microstructural change in brainstem-linked tracts",
                "rbd_change": "increased",
            },
            {
                "source": "brainstem_lesion_burden",
                "target": "pontine_rem_atonia_circuit_degeneration",
                "relation": "focal brainstem insults can directly disrupt REM-atonia circuitry",
                "rbd_change": "increased",
            },
            {
                "source": "brainstem_lesion_burden",
                "target": "brainstem_microstructural_degeneration",
                "relation": "structural insults increase microstructural disruption in REM-related pathways",
                "rbd_change": "increased",
            },
            {
                "source": "brainstem_lesion_burden",
                "target": "secondary_lesional_rbd_burden",
                "relation": "secondary RBD may arise from focal pontine or tegmental lesions",
                "rbd_change": "increased",
            },
            {
                "source": "glutamatergic_dysregulation_load",
                "target": "glutamatergic_instability",
                "relation": "altered glutamate signaling destabilizes REM-atonia generators",
                "rbd_change": "increased",
            },
            {
                "source": "rem_state_instability",
                "target": "rem_state_dissociation",
                "relation": "state instability promotes admixture of REM sleep and waking-like motor output",
                "rbd_change": "increased",
            },
            {
                "source": "narcolepsy_orexin_instability",
                "target": "orexin_state_instability",
                "relation": "orexin deficiency or instability weakens sleep-wake stabilization",
                "rbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_accumulation",
                "target": "pontine_rem_atonia_circuit_degeneration",
                "relation": "synucleinopathic change damages the brainstem circuitry of REM atonia",
                "rbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_accumulation",
                "target": "basal_ganglia_motor_bypass",
                "relation": "Parkinson-spectrum pathology alters basal-ganglia participation in REM motor expression",
                "rbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_accumulation",
                "target": "prodromal_synucleinopathy_risk",
                "relation": "alpha-synuclein pathology raises the chance that RBD is prodromal PD/DLB/MSA",
                "rbd_change": "increased",
            },
            {
                "source": "brainstem_microstructural_degeneration",
                "target": "pontine_rem_atonia_circuit_degeneration",
                "relation": "microstructural degeneration further weakens REM-atonia pathways",
                "rbd_change": "increased",
            },
            {
                "source": "glutamatergic_instability",
                "target": "pontine_rem_atonia_circuit_proxy",
                "relation": "abnormal excitatory signaling loads pontine REM-atonia circuitry",
                "rbd_change": "increased",
            },
            {
                "source": "pontine_rem_atonia_circuit_degeneration",
                "target": "pontine_rem_atonia_circuit_proxy",
                "relation": "pontine dysfunction is reflected in REM-atonia generator burden",
                "rbd_change": "increased",
            },
            {
                "source": "pontine_rem_atonia_circuit_degeneration",
                "target": "rem_state_dissociation",
                "relation": "failure of REM-atonia generators allows motor activation to intrude into REM sleep",
                "rbd_change": "increased",
            },
            {
                "source": "glutamatergic_instability",
                "target": "rem_state_dissociation",
                "relation": "glutamatergic imbalance destabilizes the REM atonia cascade",
                "rbd_change": "increased",
            },
            {
                "source": "orexin_state_instability",
                "target": "hypothalamus_orexin_proxy",
                "relation": "state-stabilization dysfunction is reflected in hypothalamic orexin-system burden",
                "rbd_change": "increased",
            },
            {
                "source": "orexin_state_instability",
                "target": "rem_state_dissociation",
                "relation": "orexin instability predisposes to intrusion between sleep-wake states",
                "rbd_change": "increased",
            },
            {
                "source": "rem_state_dissociation",
                "target": "rem_without_atonia",
                "relation": "state dissociation produces REM sleep without normal atonia",
                "rbd_change": "increased",
            },
            {
                "source": "rem_state_dissociation",
                "target": "limbic_dream_enactment_drive",
                "relation": "mixed REM-wake states amplify the chance that dream content will be enacted",
                "rbd_change": "increased",
            },
            {
                "source": "rem_state_dissociation",
                "target": "dream_enactment_behaviors",
                "relation": "motor intrusion into REM sleep produces overt dream enactment",
                "rbd_change": "increased",
            },
            {
                "source": "limbic_dream_enactment_drive",
                "target": "amygdala",
                "relation": "emotionally intense dream content recruits amygdalar circuitry",
                "rbd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "violent_fearful_dreaming",
                "relation": "amygdalar limbic activation supports fearful and aggressive dream content",
                "rbd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "dream_enactment_behaviors",
                "relation": "limbic overactivation increases emotionally charged motor enactment",
                "rbd_change": "increased",
            },
            {
                "source": "basal_ganglia_motor_bypass",
                "target": "basal_ganglia_proxy",
                "relation": "REM motor expression engages altered basal-ganglia related motor circuitry",
                "rbd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "dream_enactment_behaviors",
                "relation": "basal-ganglia involvement contributes to complex REM motor behavior",
                "rbd_change": "increased",
            },
            {
                "source": "hypothalamus_orexin_proxy",
                "target": "dream_enactment_behaviors",
                "relation": "state-stabilization failure favors intrusion of motor activity into REM sleep",
                "rbd_change": "increased",
            },
            {
                "source": "pontine_rem_atonia_circuit_proxy",
                "target": "rem_without_atonia",
                "relation": "pontine REM-atonia circuit failure manifests as loss of REM motor inhibition",
                "rbd_change": "increased",
            },
            {
                "source": "rem_without_atonia",
                "target": "sleep_related_injury_risk",
                "relation": "loss of motor atonia raises the chance of injury during sleep",
                "rbd_change": "increased",
            },
            {
                "source": "dream_enactment_behaviors",
                "target": "sleep_related_injury_risk",
                "relation": "complex dream enactment increases injury risk to the patient or bed partner",
                "rbd_change": "increased",
            },
            {
                "source": "violent_fearful_dreaming",
                "target": "sleep_related_injury_risk",
                "relation": "violent or fearful dream content increases dangerous enactment risk",
                "rbd_change": "increased",
            },
            {
                "source": "prodromal_synucleinopathy_risk",
                "target": "synucleinopathy_conversion_risk",
                "relation": "prodromal synucleinopathy burden predicts later PD/DLB/MSA emergence",
                "rbd_change": "increased",
            },
            {
                "source": "brainstem_microstructural_degeneration",
                "target": "synucleinopathy_conversion_risk",
                "relation": "microstructural degeneration supports an evolving prodromal neurodegenerative process",
                "rbd_change": "increased",
            },
            {
                "source": "neurodegenerative_progression",
                "target": "prodromal_synucleinopathy_risk",
                "relation": "progressive degeneration increases the chance of later clinically defined synucleinopathy",
                "rbd_change": "increased",
            },
            {
                "source": "environmental_toxin_exposure",
                "target": "prodromal_synucleinopathy_risk",
                "relation": "environmental toxin burden may increase prodromal neurodegenerative risk",
                "rbd_change": "increased",
            },
            {
                "source": "pontine_rem_atonia_circuit_degeneration",
                "target": "secondary_lesional_rbd_burden",
                "relation": "brainstem REM-atonia circuit damage contributes to secondary RBD burden",
                "rbd_change": "increased",
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
            tabular = getattr(siibra.features, "tabular", None)
            if tabular is not None:
                if kind == "receptor" and hasattr(tabular, "ReceptorDensityFingerprint"):
                    cands.append(getattr(tabular, "ReceptorDensityFingerprint"))
                elif kind == "gene" and hasattr(tabular, "GeneExpressions"):
                    cands.append(getattr(tabular, "GeneExpressions"))
        except Exception:
            pass

        try:
            molecular = getattr(siibra.features, "molecular", None)
            if molecular is not None:
                if kind == "receptor" and hasattr(molecular, "ReceptorDensityFingerprint"):
                    cands.append(getattr(molecular, "ReceptorDensityFingerprint"))
                elif kind == "gene" and hasattr(molecular, "GeneExpressions"):
                    cands.append(getattr(molecular, "GeneExpressions"))
        except Exception:
            pass

        try:
            connectivity = getattr(siibra.features, "connectivity", None)
            if kind == "connectivity" and connectivity is not None and hasattr(connectivity, "StreamlineCounts"):
                cands.append(getattr(connectivity, "StreamlineCounts"))
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
            "brainstem",
            "pons",
            "pontine tegmentum",
            "amygdala",
            "basal ganglia",
            "striatum",
            "putamen",
            "caudate",
            "hypothalamus",
        } else 0
        gapmap_penalty = 1 if "gapmap" in name else 0
        return (left_bonus, right_penalty, generic_penalty, gapmap_penalty)

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

        region_key = next((k for k, v in self.region_objects.items() if v == region), "")
        for candidate in self.region_candidates.get(region_key, []):
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
        """Return pairwise structural connectivity among resolved circuit nodes."""
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
                rows.append({"source": src_key, "target": dst_key, "value": value})

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return (
            pd.DataFrame(rows)
            .sort_values(["source", "value"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_RAPID_EYE_MOVEMENT_SLEEP_BEHAVIOR_DISORDER_GENE_PANEL,
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
        """Return a best-effort statistical regional map or mask for a resolved node."""
        region = self.region_objects.get(node_key)
        if region is None:
            return None

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
        alpha_synucleinopathy_liability: float = 0.45,
        gba1_pathway_genetic_risk: float = 0.40,
        neurodegenerative_progression: float = 0.55,
        environmental_toxin_exposure: float = 0.20,
        glutamatergic_dysregulation_load: float = 0.45,
        rem_state_instability: float = 0.55,
        narcolepsy_orexin_instability: float = 0.15,
        brainstem_lesion_burden: float = 0.10,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized RBD simulation.

        Parameters are 0..1 normalized knobs. Higher symptom values indicate greater
        modeled burden.
        """
        inputs = pd.Series(
            {
                "alpha_synucleinopathy_liability": self._clip01(alpha_synucleinopathy_liability),
                "gba1_pathway_genetic_risk": self._clip01(gba1_pathway_genetic_risk),
                "neurodegenerative_progression": self._clip01(neurodegenerative_progression),
                "environmental_toxin_exposure": self._clip01(environmental_toxin_exposure),
                "glutamatergic_dysregulation_load": self._clip01(glutamatergic_dysregulation_load),
                "rem_state_instability": self._clip01(rem_state_instability),
                "narcolepsy_orexin_instability": self._clip01(narcolepsy_orexin_instability),
                "brainstem_lesion_burden": self._clip01(brainstem_lesion_burden),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["alpha_synuclein_accumulation"] = self._clip01(
            0.35 * inputs["alpha_synucleinopathy_liability"]
            + 0.25 * inputs["gba1_pathway_genetic_risk"]
            + 0.25 * inputs["neurodegenerative_progression"]
            + 0.15 * inputs["environmental_toxin_exposure"]
        )
        latents["pontine_rem_atonia_circuit_degeneration"] = self._clip01(
            0.40 * latents["alpha_synuclein_accumulation"]
            + 0.30 * inputs["brainstem_lesion_burden"]
            + 0.20 * inputs["neurodegenerative_progression"]
            + 0.10 * inputs["glutamatergic_dysregulation_load"]
        )
        latents["glutamatergic_instability"] = self._clip01(
            0.60 * inputs["glutamatergic_dysregulation_load"]
            + 0.20 * latents["pontine_rem_atonia_circuit_degeneration"]
            + 0.20 * inputs["neurodegenerative_progression"]
        )
        latents["orexin_state_instability"] = self._clip01(
            0.55 * inputs["narcolepsy_orexin_instability"]
            + 0.30 * inputs["rem_state_instability"]
            + 0.15 * inputs["neurodegenerative_progression"]
        )
        latents["rem_state_dissociation"] = self._clip01(
            0.35 * latents["pontine_rem_atonia_circuit_degeneration"]
            + 0.25 * latents["glutamatergic_instability"]
            + 0.25 * latents["orexin_state_instability"]
            + 0.15 * inputs["rem_state_instability"]
        )
        latents["limbic_dream_enactment_drive"] = self._clip01(
            0.45 * latents["rem_state_dissociation"]
            + 0.25 * latents["glutamatergic_instability"]
            + 0.20 * latents["orexin_state_instability"]
            + 0.10 * inputs["neurodegenerative_progression"]
        )
        latents["basal_ganglia_motor_bypass"] = self._clip01(
            0.40 * latents["alpha_synuclein_accumulation"]
            + 0.25 * latents["rem_state_dissociation"]
            + 0.20 * inputs["neurodegenerative_progression"]
            + 0.15 * latents["glutamatergic_instability"]
        )
        latents["brainstem_microstructural_degeneration"] = self._clip01(
            0.45 * inputs["neurodegenerative_progression"]
            + 0.30 * latents["pontine_rem_atonia_circuit_degeneration"]
            + 0.25 * inputs["brainstem_lesion_burden"]
        )
        latents["prodromal_synucleinopathy_risk"] = self._clip01(
            0.45 * latents["alpha_synuclein_accumulation"]
            + 0.20 * inputs["gba1_pathway_genetic_risk"]
            + 0.20 * inputs["neurodegenerative_progression"]
            + 0.15 * latents["brainstem_microstructural_degeneration"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["pontine_rem_atonia_circuit_proxy"] = self._clip01(
            0.55 * latents["pontine_rem_atonia_circuit_degeneration"]
            + 0.25 * latents["glutamatergic_instability"]
            + 0.20 * latents["rem_state_dissociation"]
        )
        regional_state["amygdala"] = self._clip01(
            0.60 * latents["limbic_dream_enactment_drive"]
            + 0.25 * latents["rem_state_dissociation"]
            + 0.15 * latents["orexin_state_instability"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.60 * latents["basal_ganglia_motor_bypass"]
            + 0.25 * latents["alpha_synuclein_accumulation"]
            + 0.15 * latents["brainstem_microstructural_degeneration"]
        )
        regional_state["hypothalamus_orexin_proxy"] = self._clip01(
            0.60 * latents["orexin_state_instability"]
            + 0.25 * latents["rem_state_dissociation"]
            + 0.15 * inputs["neurodegenerative_progression"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["rem_without_atonia"] = self._clip01(
            0.40 * latents["rem_state_dissociation"]
            + 0.35 * regional_state["pontine_rem_atonia_circuit_proxy"]
            + 0.25 * latents["glutamatergic_instability"]
        )
        symptoms["dream_enactment_behaviors"] = self._clip01(
            0.30 * symptoms["rem_without_atonia"]
            + 0.25 * regional_state["amygdala"]
            + 0.25 * regional_state["basal_ganglia_proxy"]
            + 0.20 * regional_state["hypothalamus_orexin_proxy"]
        )
        symptoms["violent_fearful_dreaming"] = self._clip01(
            0.50 * regional_state["amygdala"]
            + 0.25 * latents["limbic_dream_enactment_drive"]
            + 0.15 * latents["rem_state_dissociation"]
            + 0.10 * regional_state["hypothalamus_orexin_proxy"]
        )
        symptoms["sleep_related_injury_risk"] = self._clip01(
            0.45 * symptoms["dream_enactment_behaviors"]
            + 0.30 * symptoms["violent_fearful_dreaming"]
            + 0.25 * symptoms["rem_without_atonia"]
        )
        symptoms["synucleinopathy_conversion_risk"] = self._clip01(
            0.45 * latents["prodromal_synucleinopathy_risk"]
            + 0.25 * latents["alpha_synuclein_accumulation"]
            + 0.15 * latents["brainstem_microstructural_degeneration"]
            + 0.15 * latents["pontine_rem_atonia_circuit_degeneration"]
        )
        symptoms["secondary_lesional_rbd_burden"] = self._clip01(
            0.40 * inputs["brainstem_lesion_burden"]
            + 0.35 * latents["pontine_rem_atonia_circuit_degeneration"]
            + 0.25 * latents["brainstem_microstructural_degeneration"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["isolated_rbd_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["rem_without_atonia"],
                        symptoms["dream_enactment_behaviors"],
                        regional_state["pontine_rem_atonia_circuit_proxy"],
                    ]
                ).mean()
            )
        )
        phenotypes["violent_dream_enactment_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["violent_fearful_dreaming"],
                        regional_state["amygdala"],
                        symptoms["dream_enactment_behaviors"],
                    ]
                ).mean()
            )
        )
        phenotypes["narcolepsy_associated_state_instability_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        latents["orexin_state_instability"],
                        regional_state["hypothalamus_orexin_proxy"],
                        latents["rem_state_dissociation"],
                    ]
                ).mean()
            )
        )
        phenotypes["prodromal_synucleinopathy_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        latents["prodromal_synucleinopathy_risk"],
                        symptoms["synucleinopathy_conversion_risk"],
                        latents["alpha_synuclein_accumulation"],
                    ]
                ).mean()
            )
        )
        phenotypes["secondary_lesional_rbd_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        inputs["brainstem_lesion_burden"],
                        symptoms["secondary_lesional_rbd_burden"],
                        latents["brainstem_microstructural_degeneration"],
                    ]
                ).mean()
            )
        )
        phenotypes["severe_sleep_injury_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["sleep_related_injury_risk"],
                        symptoms["violent_fearful_dreaming"],
                        symptoms["dream_enactment_behaviors"],
                    ]
                ).mean()
            )
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
        model = RapidEyeMovementSleepBehaviorDisorderModel()
        scaffold = model.build()

        print("\n=== Nodes ===")
        print(scaffold["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

        print("\n=== Edges (first 15) ===")
        print(scaffold["edges"].head(15).to_string(index=False))

        print("\n=== Circuit connectivity ===")
        circuit_conn = scaffold["circuit_connectivity"]
        if circuit_conn.empty:
            print("No circuit connectivity matrix available in this environment.")
        else:
            print(circuit_conn.head(15).to_string(index=False))

        for node_key in ("amygdala", "pontine_rem_atonia_circuit_proxy", "hypothalamus_orexin_proxy"):
            print(f"\n=== {node_key} receptor fingerprint ===")
            rec = scaffold["receptors"].get(node_key, pd.DataFrame())
            print(rec.head(10).to_string(index=False) if not rec.empty else "No receptor data available.")

            print(f"\n=== {node_key} gene summary ===")
            genes = scaffold["genes"].get(node_key, pd.DataFrame())
            print(genes.head(10).to_string(index=False) if not genes.empty else "No gene-expression data available.")

            print(f"\n=== {node_key} connectivity profile ===")
            conn = scaffold["connectivity_profiles"].get(node_key, pd.DataFrame())
            print(conn.head(10).to_string(index=False) if not conn.empty else "No connectivity data available.")

        print("\n=== Example simulation ===")
        sim = model.simulate(
            alpha_synucleinopathy_liability=0.50,
            gba1_pathway_genetic_risk=0.45,
            neurodegenerative_progression=0.60,
            environmental_toxin_exposure=0.25,
            glutamatergic_dysregulation_load=0.50,
            rem_state_instability=0.65,
            narcolepsy_orexin_instability=0.20,
            brainstem_lesion_burden=0.10,
        )
        for name, series in sim.items():
            print(f"\n{name.upper()}")
            print(series.to_string())

        # Example optional coordinate assignment:
        # print(model.assign_mni_point((0, -30, -20)).head())

    except Exception as exc:
        print(
            "RapidEyeMovementSleepBehaviorDisorderModel example could not complete in this "
            f"environment: {exc}"
        )
