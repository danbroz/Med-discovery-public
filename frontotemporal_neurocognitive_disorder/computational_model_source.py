
from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Major or Mild Frontotemporal Neurocognitive Disorder.

This script converts a chapter-level biological summary into a transparent mechanistic
research scaffold. It is intended for hypothesis generation, teaching, and iterative
refinement against atlas-backed evidence. It is not a diagnostic or treatment tool.

Chapter-matched biological themes encoded here:
- FTNCD is treated as a spectrum of frontotemporal lobar degeneration (FTLD) pathologies,
  especially tau, TDP-43, and FUS proteinopathies.
- Clinical phenotype is modeled as emerging from shared frontotemporal network failure
  rather than from a one-to-one mapping between a symptom pattern and a single molecular subtype.
- Apathy and inertia are linked to medial frontal-subcortical dopaminergic failure.
- Disinhibition and repetitive behaviors are linked to orbitofrontal-striatal dyscontrol.
- Variant-specific language phenotypes are linked to left-lateralized language-network burden.
- Structural atrophy and FDG-PET/SPECT hypometabolism are modeled as coupled signatures of
  frontotemporal circuit degeneration.

Important modeling note:
The chapter explicitly describes substantial heterogeneity. Therefore, tau, TDP-43, and FUS
loads are routed through shared latent degeneration processes instead of being mapped directly
to unique clinical syndromes. That is deliberate and preserves the chapter's claim that
phenotype does not reliably predict molecular pathology.

The scaffold is compatibility-first and degrades gracefully when:
- siibra is not installed,
- a requested Julich region cannot be resolved,
- receptor, gene-expression, or connectivity features are unavailable.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency in authoring environments
    import siibra  # type: ignore
except Exception:  # pragma: no cover
    siibra = None


DEFAULT_GENE_PANEL = [
    "MAPT",      # tau
    "TARDBP",    # TDP-43
    "FUS",       # FUS proteinopathy
    "GRN",       # common FTLD genetic driver
    "C9orf72",   # common familial FTLD driver
    "TMEM106B",  # FTLD risk modifier
    "VCP",       # multisystem proteinopathy / FTD linkage
    "SQSTM1",    # protein handling / FTLD linkage
    "CHMP2B",    # rare familial FTD gene
    "DCTN1",     # parkinsonism / FTD overlap
    "DRD2",      # dopamine signaling
    "SLC6A3",    # dopamine transporter
    "COMT",      # catecholamine metabolism
    "BDNF",      # plasticity / network vulnerability
]


class MajorOrMildFrontotemporalNeurocognitiveDisorderModel:
    """
    Mechanistic siibra scaffold for Major or Mild Frontotemporal Neurocognitive Disorder.

    Main chapter-derived logic:
    1) heterogeneous FTLD proteinopathies drive a shared burden of frontal and temporal degeneration,
    2) frontotemporal degeneration produces coupled structural and metabolic dysfunction,
    3) medial frontal-subcortical dopamine failure contributes to apathy and inertia,
    4) orbitofrontal-striatal dyscontrol contributes to disinhibition and stereotyped behavior,
    5) left-lateralized language-network burden contributes to PNFA-, semantic-, and logopenic-like phenotypes.

    Regional-state values in this simulator represent dysregulation burden rather than raw neural firing.
    Higher values indicate greater disease-relevant dysfunction.

    This is a research scaffold, not a validated disease model.
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
        self.siibra_available = siibra is not None

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

        if self.siibra_available:
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
            except Exception as exc:
                warnings.warn(
                    f"siibra is installed but atlas initialization failed: {exc}. "
                    "Atlas-backed methods will return empty results."
                )
        else:
            warnings.warn(
                "siibra is not installed in this environment. "
                "Atlas-backed methods will return empty results, but simulate() still works."
            )

        # Direct anchors are used where the chapter names structures or network loci explicitly.
        # Proxies are used when the chapter is systems-level or the exact Julich label may vary.
        self.region_candidates: Dict[str, List[str]] = {
            "right_orbitofrontal_proxy": [
                "Area Fo4 (OFC) right",
                "Area Fo3 (OFC) right",
                "orbitofrontal cortex right",
                "orbitofrontal right",
            ],
            "right_insula_proxy": [
                "insula right",
                "anterior insula right",
                "insular cortex right",
            ],
            "anterior_cingulate_proxy": [
                "Area s24 right",
                "Area p32 right",
                "anterior cingulate cortex right",
                "cingulate cortex right",
            ],
            "right_anterior_temporal_proxy": [
                "temporal pole right",
                "anterior temporal lobe right",
                "entorhinal cortex right",
                "parahippocampal cortex right",
                "temporal cortex right",
            ],
            "striatum_proxy": [
                "nucleus accumbens right",
                "caudate nucleus right",
                "putamen right",
                "striatum right",
                "basal ganglia right",
            ],
            "left_posterior_frontoinsular_proxy": [
                "Area 44 (IFG) left",
                "Area 45 (IFG) left",
                "insula left",
                "frontal operculum left",
                "inferior frontal gyrus left",
            ],
            "left_anterior_temporal_language_proxy": [
                "temporal pole left",
                "anterior temporal lobe left",
                "entorhinal cortex left",
                "parahippocampal cortex left",
                "temporal cortex left",
            ],
            "left_posterior_temporal_inferior_parietal_proxy": [
                "Area TPJ (STG, SMG) left",
                "Area TPJ (STG/SMG) left",
                "Area PGp (IPL) left",
                "Area PGa (IPL) left",
                "Area PFm (IPL) left",
                "Area TE 3 (STG) left",
                "angular gyrus left",
                "supramarginal gyrus left",
                "inferior parietal lobule left",
                "posterior temporal lobe left",
                "middle temporal gyrus left",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "right_orbitofrontal_proxy": (
                "Right-lateralized orbitofrontal proxy for the behavioral-variant pattern of "
                "disinhibition, reward-control disturbance, and erratic behavioral regulation."
            ),
            "right_insula_proxy": (
                "Right insular proxy for salience, interoceptive-social integration, and the "
                "bvFTNCD pattern of socioemotional dysfunction."
            ),
            "anterior_cingulate_proxy": (
                "Anterior cingulate proxy for motivational drive, monitoring, and the apathy-prone "
                "medial frontal network burden described in FTNCD."
            ),
            "right_anterior_temporal_proxy": (
                "Right anterior temporal proxy for anterior temporal degeneration associated with "
                "socioemotional and behavioral change."
            ),
            "striatum_proxy": (
                "Striatal / basal-ganglia proxy for medial frontal-subcortical dopamine circuits, "
                "repetitive behaviors, and parkinsonian overlap."
            ),
            "left_posterior_frontoinsular_proxy": (
                "Left posterior fronto-insular proxy reflecting the chapter's PNFA-associated "
                "atrophy pattern."
            ),
            "left_anterior_temporal_language_proxy": (
                "Left anterior temporal proxy reflecting the chapter's semantic-dementia language-network burden."
            ),
            "left_posterior_temporal_inferior_parietal_proxy": (
                "Left posterior temporal / inferior parietal proxy reflecting the chapter's "
                "logopenic-like language-network pattern."
            ),
        }
        self.proxy_region_keys = set(self.region_candidates.keys())

        self.input_nodes: Dict[str, str] = {
            "disease_stage": (
                "Overall neurodegenerative progression burden increasing frontotemporal circuit failure."
            ),
            "familial_genetic_load": (
                "Familial or inherited vulnerability reflecting the substantial heritable component of FTNCD."
            ),
            "mapt_tauopathy_load": (
                "Tau-related molecular burden linked to MAPT-associated FTLD-tau."
            ),
            "tdp43_proteinopathy_load": (
                "TDP-43-related molecular burden contributing to FTLD heterogeneity."
            ),
            "fus_proteinopathy_load": (
                "FUS-related molecular burden contributing to FTLD heterogeneity."
            ),
            "right_frontoinsular_predominance": (
                "Bias toward the right frontal / insular / anterior cingulate / anterior temporal pattern typical of bvFTNCD."
            ),
            "left_language_network_predominance": (
                "Bias toward left-lateralized language-network vulnerability across primary progressive aphasia variants."
            ),
            "anterior_temporal_predominance": (
                "Bias toward anterior temporal involvement, especially relevant to semantic-predominant patterns."
            ),
            "nigrostriatal_involvement": (
                "Additional nigrostriatal dopamine-system burden linked to parkinsonism and motivational slowing."
            ),
            "cognitive_reserve": (
                "Protective reserve buffering symptom expression despite biological burden."
            ),
            "structured_environment_support": (
                "Protective caregiver structure and external routines that can reduce behavioral dyscontrol."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "ftld_proteinopathy_burden": (
                "Shared molecular pathology burden integrating tau, TDP-43, FUS, and inherited susceptibility."
            ),
            "frontotemporal_degeneration": (
                "Progressive frontal and temporal lobe degeneration that disrupts executive, social, and language circuits."
            ),
            "frontotemporal_hypometabolism": (
                "Frontal and anterior temporal hypometabolism / hypoperfusion paralleling structural degeneration."
            ),
            "medial_frontal_subcortical_dopamine_failure": (
                "Dopaminergic hypofunction across medial frontal-subcortical circuits linked to apathy and inertia."
            ),
            "orbitofrontal_striatal_discontrol": (
                "Orbitofrontal-striatal dysregulation contributing to impulsivity, disinhibition, and stereotyped behavior."
            ),
            "salience_social_circuit_breakdown": (
                "Failure of right fronto-insular-anterior cingulate-anterior temporal systems supporting social-emotional regulation."
            ),
            "language_network_breakdown": (
                "Failure of left-lateralized language systems underlying nonfluent, semantic, and logopenic presentations."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "apathy_inertia": "Reduced drive, initiative, and behavioral activation.",
            "disinhibition_impulsivity": "Behavioral disinhibition, impulsivity, or poor social restraint.",
            "repetitive_stereotyped_behavior": "Repetitive, stereotyped, or compulsive-like actions.",
            "executive_dysfunction": "Impaired planning, organization, abstraction, or cognitive control.",
            "social_emotional_blunting": "Loss of empathy, emotional attunement, or socioemotional responsiveness.",
            "nonfluent_aphasia": "Effortful, halting, or nonfluent speech-production impairment.",
            "semantic_loss": "Loss of conceptual / word meaning knowledge typical of semantic-predominant syndromes.",
            "logopenic_language_impairment": "Word-finding and repetition difficulties with posterior language-network burden.",
            "parkinsonism": "Motor slowing or parkinsonian features associated with nigrostriatal involvement.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "mapt_tauopathy_load",
                "target": "ftld_proteinopathy_burden",
                "relation": "tau-linked molecular burden contributes to FTLD pathology",
                "ftncd_change": "increased",
            },
            {
                "source": "tdp43_proteinopathy_load",
                "target": "ftld_proteinopathy_burden",
                "relation": "TDP-43 pathology contributes to FTLD molecular burden",
                "ftncd_change": "increased",
            },
            {
                "source": "fus_proteinopathy_load",
                "target": "ftld_proteinopathy_burden",
                "relation": "FUS pathology contributes to FTLD molecular burden",
                "ftncd_change": "increased",
            },
            {
                "source": "familial_genetic_load",
                "target": "ftld_proteinopathy_burden",
                "relation": "familial genetic liability increases risk of frontotemporal lobar degeneration",
                "ftncd_change": "increased",
            },
            {
                "source": "disease_stage",
                "target": "frontotemporal_degeneration",
                "relation": "progressive disease burden drives frontal and temporal lobe degeneration",
                "ftncd_change": "increased",
            },
            {
                "source": "ftld_proteinopathy_burden",
                "target": "frontotemporal_degeneration",
                "relation": "shared proteinopathy burden drives selective frontotemporal vulnerability",
                "ftncd_change": "increased",
            },
            {
                "source": "frontotemporal_degeneration",
                "target": "frontotemporal_hypometabolism",
                "relation": "degeneration produces the frontal and anterior temporal metabolic signature seen on PET/SPECT",
                "ftncd_change": "increased",
            },
            {
                "source": "frontotemporal_degeneration",
                "target": "medial_frontal_subcortical_dopamine_failure",
                "relation": "degeneration disrupts medial frontal-subcortical motivational circuitry",
                "ftncd_change": "increased",
            },
            {
                "source": "nigrostriatal_involvement",
                "target": "medial_frontal_subcortical_dopamine_failure",
                "relation": "additional nigrostriatal burden worsens dopamine-linked motivational failure",
                "ftncd_change": "increased",
            },
            {
                "source": "frontotemporal_degeneration",
                "target": "orbitofrontal_striatal_discontrol",
                "relation": "frontotemporal injury destabilizes orbitofrontal-striatal behavioral control",
                "ftncd_change": "increased",
            },
            {
                "source": "right_frontoinsular_predominance",
                "target": "salience_social_circuit_breakdown",
                "relation": "right fronto-insular emphasis amplifies the behavioral-variant socioemotional syndrome",
                "ftncd_change": "increased",
            },
            {
                "source": "anterior_temporal_predominance",
                "target": "salience_social_circuit_breakdown",
                "relation": "anterior temporal emphasis worsens social-emotional and semantic network breakdown",
                "ftncd_change": "increased",
            },
            {
                "source": "left_language_network_predominance",
                "target": "language_network_breakdown",
                "relation": "left-lateralized vulnerability drives aphasic and language-network presentations",
                "ftncd_change": "increased",
            },
            {
                "source": "frontotemporal_hypometabolism",
                "target": "right_orbitofrontal_proxy",
                "relation": "behavioral-variant imaging burden includes orbitofrontal hypometabolism",
                "ftncd_change": "increased",
            },
            {
                "source": "frontotemporal_hypometabolism",
                "target": "right_anterior_temporal_proxy",
                "relation": "frontotemporal metabolic dysfunction involves anterior temporal systems",
                "ftncd_change": "increased",
            },
            {
                "source": "salience_social_circuit_breakdown",
                "target": "right_insula_proxy",
                "relation": "salience-network failure burdens the insula in bvFTNCD",
                "ftncd_change": "increased",
            },
            {
                "source": "medial_frontal_subcortical_dopamine_failure",
                "target": "anterior_cingulate_proxy",
                "relation": "dopamine-linked motivational failure burdens anterior cingulate systems",
                "ftncd_change": "increased",
            },
            {
                "source": "orbitofrontal_striatal_discontrol",
                "target": "striatum_proxy",
                "relation": "behavioral dyscontrol engages striatal / basal-ganglia circuitry",
                "ftncd_change": "increased",
            },
            {
                "source": "language_network_breakdown",
                "target": "left_posterior_frontoinsular_proxy",
                "relation": "language-network failure burdens the PNFA-associated posterior fronto-insular region",
                "ftncd_change": "increased",
            },
            {
                "source": "language_network_breakdown",
                "target": "left_anterior_temporal_language_proxy",
                "relation": "language-network failure burdens the semantic-dementia-associated anterior temporal region",
                "ftncd_change": "increased",
            },
            {
                "source": "language_network_breakdown",
                "target": "left_posterior_temporal_inferior_parietal_proxy",
                "relation": "language-network failure burdens the posterior temporal / inferior parietal language system",
                "ftncd_change": "increased",
            },
            {
                "source": "medial_frontal_subcortical_dopamine_failure",
                "target": "apathy_inertia",
                "relation": "dopaminergic hypofunction contributes to apathy and inertia",
                "ftncd_change": "increased",
            },
            {
                "source": "orbitofrontal_striatal_discontrol",
                "target": "disinhibition_impulsivity",
                "relation": "orbitofrontal-striatal dyscontrol contributes to disinhibition and impulsivity",
                "ftncd_change": "increased",
            },
            {
                "source": "orbitofrontal_striatal_discontrol",
                "target": "repetitive_stereotyped_behavior",
                "relation": "orbitofrontal-striatal dyscontrol contributes to repetitive or stereotyped acts",
                "ftncd_change": "increased",
            },
            {
                "source": "salience_social_circuit_breakdown",
                "target": "social_emotional_blunting",
                "relation": "salience and socioemotional circuit failure produces emotional blunting and empathy loss",
                "ftncd_change": "increased",
            },
            {
                "source": "frontotemporal_degeneration",
                "target": "executive_dysfunction",
                "relation": "frontal degeneration disrupts executive function",
                "ftncd_change": "increased",
            },
            {
                "source": "left_posterior_frontoinsular_proxy",
                "target": "nonfluent_aphasia",
                "relation": "posterior fronto-insular burden produces nonfluent language impairment",
                "ftncd_change": "increased",
            },
            {
                "source": "left_anterior_temporal_language_proxy",
                "target": "semantic_loss",
                "relation": "anterior temporal burden produces semantic impairment",
                "ftncd_change": "increased",
            },
            {
                "source": "left_posterior_temporal_inferior_parietal_proxy",
                "target": "logopenic_language_impairment",
                "relation": "posterior temporal / inferior parietal burden produces logopenic-like language difficulty",
                "ftncd_change": "increased",
            },
            {
                "source": "nigrostriatal_involvement",
                "target": "parkinsonism",
                "relation": "nigrostriatal involvement can produce parkinsonian features",
                "ftncd_change": "increased",
            },
            {
                "source": "cognitive_reserve",
                "target": "executive_dysfunction",
                "relation": "reserve can buffer symptom expression relative to biological burden",
                "ftncd_change": "decreased",
            },
            {
                "source": "cognitive_reserve",
                "target": "nonfluent_aphasia",
                "relation": "reserve can partially buffer language symptom expression",
                "ftncd_change": "decreased",
            },
            {
                "source": "structured_environment_support",
                "target": "disinhibition_impulsivity",
                "relation": "external structure can reduce behavioral dyscontrol expression",
                "ftncd_change": "decreased",
            },
            {
                "source": "structured_environment_support",
                "target": "repetitive_stereotyped_behavior",
                "relation": "care structure can reduce repetitive behavioral escalation",
                "ftncd_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _reduce_numeric(value: Any) -> Optional[float]:
        """
        Convert a scalar-like, Series-like, or DataFrame-like connectivity lookup to one float.

        siibra connectivity matrices can occasionally yield duplicated labels after averaging
        compound features, so `.loc[row, col]` may return a scalar, Series, or DataFrame.
        This helper collapses any numeric result to a mean value and returns None when
        no numeric data are available.
        """
        if value is None:
            return None
        try:
            if isinstance(value, pd.DataFrame):
                numeric = value.apply(pd.to_numeric, errors="coerce")
                arr = numeric.to_numpy().ravel()
            elif isinstance(value, pd.Series):
                arr = pd.to_numeric(value, errors="coerce").to_numpy().ravel()
            else:
                try:
                    return float(value)
                except Exception:
                    arr = pd.to_numeric(pd.Series(list(value)), errors="coerce").to_numpy().ravel()
        except Exception:
            return None

        arr = [float(x) for x in arr if pd.notna(x)]
        if not arr:
            return None
        return float(sum(arr) / len(arr))

    def _canonicalize_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize connectivity-matrix labels to region names and average duplicates.

        siibra connectivity matrices use region objects as row/column labels, and averaged
        compound features may still expose repeated region names. Converting labels to names
        and collapsing duplicates keeps downstream profile extraction and pairwise lookups scalar.
        """
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            return pd.DataFrame()

        df = matrix.copy()
        try:
            df.index = [self._name_of(x) for x in df.index]
        except Exception:
            df.index = [str(x) for x in df.index]
        try:
            df.columns = [self._name_of(x) for x in df.columns]
        except Exception:
            df.columns = [str(x) for x in df.columns]

        try:
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            pass

        if not df.index.is_unique:
            df = df.groupby(level=0).mean(numeric_only=True)
        if not df.columns.is_unique:
            df = df.T.groupby(level=0).mean(numeric_only=True).T
        return df

    @staticmethod
    def _preferred_hemisphere(spec: str) -> Optional[str]:
        low = str(spec).lower()
        if "right" in low:
            return "right"
        if "left" in low:
            return "left"
        return None

    def _modality_candidates(self, kind: str) -> List[Any]:
        if not self.siibra_available:
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
        if not self.siibra_available or concept is None:
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
        if not self.siibra_available or self.atlas is None:
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

    def _region_rank(self, region: Any, prefer_hemisphere: Optional[str] = None) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        if prefer_hemisphere == "right":
            hemi_penalty = 0 if "right" in name else 1
        elif prefer_hemisphere == "left":
            hemi_penalty = 0 if "left" in name else 1
        else:
            hemi_penalty = 0 if "left" in name else 1

        generic_penalty = 1 if name in {
            "insula",
            "orbitofrontal cortex",
            "cingulate cortex",
            "temporal cortex",
            "striatum",
        } else 0
        cyto_penalty = 0 if "area " in name or "(" in name else 1
        length_penalty = len(name)
        return (hemi_penalty, generic_penalty, cyto_penalty, length_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available or self.atlas is None or self.parcellation is None:
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
                prefer = self._preferred_hemisphere(spec)
                return sorted(matches, key=lambda r: self._region_rank(r, prefer))[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
        seen = set()
        matches = self._julich_matches(keyword)
        for region in sorted(matches, key=lambda r: self._region_rank(r, None)):
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

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = None
        if centroid is not None:
            try:
                coord = getattr(centroid, "coordinate", centroid)
                centroid_xyz = tuple(float(x) for x in coord[:3])
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid)[:3]
                except Exception:
                    centroid_xyz = None
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
                pass
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if not self.siibra_available or self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = self._canonicalize_connectivity_matrix(data)
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            self._connectivity_matrix = self._canonicalize_connectivity_matrix(compound[0].data.copy())
            return self._connectivity_matrix
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = str(getattr(region, "name", region)).strip().lower()
        exact = [x for x in labels if self._name_of(x).strip().lower() == region_name]
        if exact:
            return exact[0]

        fuzzy = [
            x for x in labels
            if region_name in self._name_of(x).lower() or self._name_of(x).lower() in region_name
        ]
        return fuzzy[0] if fuzzy else None

    def _extract_connectivity_series(
        self,
        matrix: pd.DataFrame,
        label: Any,
        axis: str = "index",
    ) -> pd.Series:
        try:
            obj = matrix.loc[label] if axis == "index" else matrix[label]
        except Exception:
            return pd.Series(dtype=float)

        if isinstance(obj, pd.DataFrame):
            try:
                numeric = obj.apply(pd.to_numeric, errors="coerce")
                obj = numeric.mean(axis=0 if axis == "index" else 1)
            except Exception:
                return pd.Series(dtype=float)
        elif isinstance(obj, pd.Series):
            try:
                obj = pd.to_numeric(obj, errors="coerce")
            except Exception:
                return pd.Series(dtype=float)
        else:
            value = self._reduce_numeric(obj)
            if value is None:
                return pd.Series(dtype=float)
            obj = pd.Series({self._name_of(label): value}, dtype=float)

        return obj.dropna().astype(float)

    def _pairwise_connectivity_value(
        self,
        matrix: pd.DataFrame,
        source_label: Any,
        target_label: Any,
    ) -> Optional[float]:
        getters = [
            lambda: matrix.at[source_label, target_label],
            lambda: matrix.at[target_label, source_label],
            lambda: matrix.loc[source_label, target_label],
            lambda: matrix.loc[target_label, source_label],
            lambda: self._extract_connectivity_series(matrix, source_label, axis="index").get(target_label),
            lambda: self._extract_connectivity_series(matrix, source_label, axis="columns").get(target_label),
            lambda: self._extract_connectivity_series(matrix, target_label, axis="index").get(source_label),
            lambda: self._extract_connectivity_series(matrix, target_label, axis="columns").get(source_label),
        ]
        for getter in getters:
            try:
                value = getter()
            except Exception:
                continue
            scalar = self._reduce_numeric(value)
            if scalar is not None:
                return scalar
        return None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if not matrix.empty:
            label = self._match_region_label(list(matrix.index), region)
            axis = "index"
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
                axis = "columns"
            if label is not None:
                series = self._extract_connectivity_series(matrix, label, axis=axis)
                if not series.empty:
                    df = series.sort_values(ascending=False).reset_index()
                    df.columns = ["connected_region", "value"]
                    df["connected_region"] = df["connected_region"].map(self._name_of)
                    region_name = str(getattr(region, "name", region))
                    df = df[df["connected_region"] != region_name].head(max_rows)
                    return df.reset_index(drop=True)

        feats = self._safe_features_any(region, self._modality_candidates("connectivity"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                data = getattr(feat, "data", None)
                if isinstance(data, pd.DataFrame):
                    out = data.copy().reset_index(drop=False)
                    out.columns = [str(c) for c in out.columns]
                    return out.head(max_rows)
            except Exception:
                continue
        return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a pairwise connectivity table among resolved regional circuit nodes.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        labels: Dict[str, Any] = {}
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                labels[key] = label

        rows: List[Dict[str, Any]] = []
        keys = list(labels.keys())
        for i, source in enumerate(keys):
            for target in keys[i + 1:]:
                s_label = labels[source]
                t_label = labels[target]
                value = self._pairwise_connectivity_value(matrix, s_label, t_label)
                scalar = self._reduce_numeric(value)
                if scalar is not None:
                    rows.append({"source": source, "target": target, "value": scalar})

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        """
        Resolve atlas regions when possible and gather receptor / gene / connectivity summaries.
        """
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node.")
            if region is None:
                if self.siibra_available:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key in self.proxy_region_keys else "region",
                        "description": f"{desc} Unresolved in the current environment.",
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
                    "label": getattr(region, "name", key.replace("_", " ").title()),
                    "node_type": "region_proxy" if key in self.proxy_region_keys else "region",
                    "description": desc,
                    "atlas_region": getattr(region, "name", None),
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
        Probabilistically assign an MNI152 point to Julich regions using a statistical map.
        """
        if not self.siibra_available:
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(v) for v in xyz[:3]), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a region mask or map representation when available.
        """
        if not self.siibra_available:
            return None
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        try:
            if hasattr(region, "get_regional_mask"):
                mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
                try:
                    return mask.fetch()
                except Exception:
                    return mask
        except Exception:
            pass

        try:
            if hasattr(region, "get_regional_map"):
                regional_map = region.get_regional_map(self.assignment_space, "statistical")
                try:
                    return regional_map.fetch()
                except Exception:
                    return regional_map
        except Exception:
            pass

        try:
            pmap = siibra.get_map(
                parcellation=self.parcellation_spec,
                space=self.assignment_space,
                maptype="statistical",
            )
            try:
                return pmap.fetch(region=region)
            except Exception:
                return pmap
        except Exception:
            return None

    def simulate(
        self,
        disease_stage: float = 0.55,
        familial_genetic_load: float = 0.40,
        mapt_tauopathy_load: float = 0.35,
        tdp43_proteinopathy_load: float = 0.35,
        fus_proteinopathy_load: float = 0.10,
        right_frontoinsular_predominance: float = 0.55,
        left_language_network_predominance: float = 0.35,
        anterior_temporal_predominance: float = 0.45,
        nigrostriatal_involvement: float = 0.20,
        cognitive_reserve: float = 0.25,
        structured_environment_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Inputs are clipped to [0, 1]. Protective variables buffer symptom expression
        more than core molecular pathology, reflecting the neurodegenerative nature
        of the disorder.
        """
        inputs = {
            "disease_stage": self._clip01(disease_stage),
            "familial_genetic_load": self._clip01(familial_genetic_load),
            "mapt_tauopathy_load": self._clip01(mapt_tauopathy_load),
            "tdp43_proteinopathy_load": self._clip01(tdp43_proteinopathy_load),
            "fus_proteinopathy_load": self._clip01(fus_proteinopathy_load),
            "right_frontoinsular_predominance": self._clip01(right_frontoinsular_predominance),
            "left_language_network_predominance": self._clip01(left_language_network_predominance),
            "anterior_temporal_predominance": self._clip01(anterior_temporal_predominance),
            "nigrostriatal_involvement": self._clip01(nigrostriatal_involvement),
            "cognitive_reserve": self._clip01(cognitive_reserve),
            "structured_environment_support": self._clip01(structured_environment_support),
        }

        support_mean = self._clip01(
            (inputs["cognitive_reserve"] + inputs["structured_environment_support"]) / 2.0
        )

        variant_biases: Dict[str, float] = {}
        variant_biases["behavioral_variant_bias"] = self._clip01(
            0.65 * inputs["right_frontoinsular_predominance"]
            + 0.20 * inputs["anterior_temporal_predominance"]
            + 0.15 * inputs["disease_stage"]
        )
        variant_biases["language_variant_bias"] = self._clip01(
            0.70 * inputs["left_language_network_predominance"]
            + 0.20 * inputs["disease_stage"]
            + 0.10 * inputs["familial_genetic_load"]
        )
        variant_biases["semantic_variant_bias"] = self._clip01(
            0.60 * inputs["anterior_temporal_predominance"]
            + 0.40 * inputs["left_language_network_predominance"]
        )
        variant_biases["posterior_language_bias"] = self._clip01(
            0.65 * inputs["left_language_network_predominance"]
            + 0.35 * (1.0 - inputs["anterior_temporal_predominance"])
        )

        latents: Dict[str, float] = {}
        latents["ftld_proteinopathy_burden"] = self._clip01(
            0.25 * inputs["mapt_tauopathy_load"]
            + 0.25 * inputs["tdp43_proteinopathy_load"]
            + 0.15 * inputs["fus_proteinopathy_load"]
            + 0.15 * inputs["familial_genetic_load"]
            + 0.20 * inputs["disease_stage"]
        )
        latents["frontotemporal_degeneration"] = self._clip01(
            0.45 * inputs["disease_stage"]
            + 0.35 * latents["ftld_proteinopathy_burden"]
            + 0.10 * variant_biases["behavioral_variant_bias"]
            + 0.10 * variant_biases["language_variant_bias"]
        )
        latents["frontotemporal_hypometabolism"] = self._clip01(
            0.45 * latents["frontotemporal_degeneration"]
            + 0.20 * variant_biases["behavioral_variant_bias"]
            + 0.20 * variant_biases["language_variant_bias"]
            + 0.15 * latents["ftld_proteinopathy_burden"]
        )
        latents["medial_frontal_subcortical_dopamine_failure"] = self._clip01(
            0.35 * latents["frontotemporal_degeneration"]
            + 0.25 * inputs["nigrostriatal_involvement"]
            + 0.20 * variant_biases["behavioral_variant_bias"]
            + 0.20 * latents["frontotemporal_hypometabolism"]
        )
        latents["orbitofrontal_striatal_discontrol"] = self._clip01(
            0.35 * latents["frontotemporal_degeneration"]
            + 0.30 * variant_biases["behavioral_variant_bias"]
            + 0.20 * latents["frontotemporal_hypometabolism"]
            + 0.15 * inputs["nigrostriatal_involvement"]
        )
        latents["salience_social_circuit_breakdown"] = self._clip01(
            0.35 * latents["frontotemporal_degeneration"]
            + 0.30 * variant_biases["behavioral_variant_bias"]
            + 0.20 * inputs["anterior_temporal_predominance"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
        )
        latents["language_network_breakdown"] = self._clip01(
            0.35 * latents["frontotemporal_degeneration"]
            + 0.30 * variant_biases["language_variant_bias"]
            + 0.20 * latents["frontotemporal_hypometabolism"]
            + 0.15 * latents["ftld_proteinopathy_burden"]
        )

        regional_state: Dict[str, float] = {}
        regional_state["right_orbitofrontal_proxy"] = self._clip01(
            0.45 * latents["orbitofrontal_striatal_discontrol"]
            + 0.25 * latents["frontotemporal_degeneration"]
            + 0.20 * variant_biases["behavioral_variant_bias"]
            + 0.10 * latents["frontotemporal_hypometabolism"]
            - 0.10 * inputs["structured_environment_support"]
        )
        regional_state["right_insula_proxy"] = self._clip01(
            0.45 * latents["salience_social_circuit_breakdown"]
            + 0.25 * latents["frontotemporal_degeneration"]
            + 0.20 * variant_biases["behavioral_variant_bias"]
            + 0.10 * latents["frontotemporal_hypometabolism"]
            - 0.10 * inputs["structured_environment_support"]
        )
        regional_state["anterior_cingulate_proxy"] = self._clip01(
            0.35 * latents["medial_frontal_subcortical_dopamine_failure"]
            + 0.30 * latents["salience_social_circuit_breakdown"]
            + 0.20 * latents["frontotemporal_degeneration"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            - 0.10 * inputs["cognitive_reserve"]
        )
        regional_state["right_anterior_temporal_proxy"] = self._clip01(
            0.35 * latents["salience_social_circuit_breakdown"]
            + 0.25 * latents["frontotemporal_degeneration"]
            + 0.25 * inputs["anterior_temporal_predominance"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
        )
        regional_state["striatum_proxy"] = self._clip01(
            0.40 * latents["medial_frontal_subcortical_dopamine_failure"]
            + 0.30 * latents["orbitofrontal_striatal_discontrol"]
            + 0.20 * inputs["nigrostriatal_involvement"]
            + 0.10 * latents["frontotemporal_hypometabolism"]
        )
        regional_state["left_posterior_frontoinsular_proxy"] = self._clip01(
            0.40 * latents["language_network_breakdown"]
            + 0.25 * variant_biases["language_variant_bias"]
            + 0.20 * latents["frontotemporal_degeneration"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            - 0.10 * inputs["cognitive_reserve"]
        )
        regional_state["left_anterior_temporal_language_proxy"] = self._clip01(
            0.35 * latents["language_network_breakdown"]
            + 0.30 * variant_biases["semantic_variant_bias"]
            + 0.20 * latents["frontotemporal_degeneration"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            - 0.10 * inputs["cognitive_reserve"]
        )
        regional_state["left_posterior_temporal_inferior_parietal_proxy"] = self._clip01(
            0.35 * latents["language_network_breakdown"]
            + 0.30 * variant_biases["posterior_language_bias"]
            + 0.20 * latents["frontotemporal_degeneration"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            - 0.10 * inputs["cognitive_reserve"]
        )

        symptoms: Dict[str, float] = {}
        symptoms["apathy_inertia"] = self._clip01(
            0.40 * latents["medial_frontal_subcortical_dopamine_failure"]
            + 0.25 * regional_state["anterior_cingulate_proxy"]
            + 0.20 * regional_state["striatum_proxy"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            - 0.20 * inputs["structured_environment_support"]
            - 0.15 * inputs["cognitive_reserve"]
        )
        symptoms["disinhibition_impulsivity"] = self._clip01(
            0.40 * latents["orbitofrontal_striatal_discontrol"]
            + 0.25 * regional_state["right_orbitofrontal_proxy"]
            + 0.20 * regional_state["right_insula_proxy"]
            + 0.15 * variant_biases["behavioral_variant_bias"]
            - 0.25 * inputs["structured_environment_support"]
            - 0.10 * inputs["cognitive_reserve"]
        )
        symptoms["repetitive_stereotyped_behavior"] = self._clip01(
            0.35 * latents["orbitofrontal_striatal_discontrol"]
            + 0.30 * regional_state["striatum_proxy"]
            + 0.20 * regional_state["right_orbitofrontal_proxy"]
            + 0.15 * latents["frontotemporal_degeneration"]
            - 0.25 * inputs["structured_environment_support"]
        )
        symptoms["executive_dysfunction"] = self._clip01(
            0.40 * latents["frontotemporal_degeneration"]
            + 0.20 * regional_state["anterior_cingulate_proxy"]
            + 0.15 * regional_state["right_orbitofrontal_proxy"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            + 0.10 * latents["language_network_breakdown"]
            - 0.25 * inputs["cognitive_reserve"]
        )
        symptoms["social_emotional_blunting"] = self._clip01(
            0.35 * latents["salience_social_circuit_breakdown"]
            + 0.25 * regional_state["right_insula_proxy"]
            + 0.20 * regional_state["right_anterior_temporal_proxy"]
            + 0.20 * regional_state["anterior_cingulate_proxy"]
            - 0.15 * inputs["structured_environment_support"]
        )
        symptoms["nonfluent_aphasia"] = self._clip01(
            0.40 * latents["language_network_breakdown"]
            + 0.30 * regional_state["left_posterior_frontoinsular_proxy"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            + 0.15 * variant_biases["language_variant_bias"]
            - 0.20 * inputs["cognitive_reserve"]
        )
        symptoms["semantic_loss"] = self._clip01(
            0.35 * latents["language_network_breakdown"]
            + 0.35 * regional_state["left_anterior_temporal_language_proxy"]
            + 0.15 * variant_biases["semantic_variant_bias"]
            + 0.15 * regional_state["right_anterior_temporal_proxy"]
            - 0.15 * inputs["cognitive_reserve"]
        )
        symptoms["logopenic_language_impairment"] = self._clip01(
            0.35 * latents["language_network_breakdown"]
            + 0.35 * regional_state["left_posterior_temporal_inferior_parietal_proxy"]
            + 0.15 * variant_biases["posterior_language_bias"]
            + 0.15 * latents["frontotemporal_hypometabolism"]
            - 0.15 * inputs["cognitive_reserve"]
        )
        symptoms["parkinsonism"] = self._clip01(
            0.45 * inputs["nigrostriatal_involvement"]
            + 0.25 * latents["medial_frontal_subcortical_dopamine_failure"]
            + 0.20 * regional_state["striatum_proxy"]
            + 0.10 * latents["ftld_proteinopathy_burden"]
            - 0.10 * inputs["structured_environment_support"]
        )

        phenotypes: Dict[str, float] = {}
        phenotypes["behavioral_variant_ftncd_profile"] = self._clip01(
            (
                symptoms["apathy_inertia"]
                + symptoms["disinhibition_impulsivity"]
                + symptoms["repetitive_stereotyped_behavior"]
                + symptoms["social_emotional_blunting"]
                + symptoms["executive_dysfunction"]
            ) / 5.0
        )
        phenotypes["pnfa_profile"] = self._clip01(
            (
                symptoms["nonfluent_aphasia"]
                + symptoms["executive_dysfunction"]
                + regional_state["left_posterior_frontoinsular_proxy"]
            ) / 3.0
        )
        phenotypes["semantic_dementia_profile"] = self._clip01(
            (
                symptoms["semantic_loss"]
                + symptoms["social_emotional_blunting"]
                + regional_state["left_anterior_temporal_language_proxy"]
            ) / 3.0
        )
        phenotypes["lvppa_profile"] = self._clip01(
            (
                symptoms["logopenic_language_impairment"]
                + latents["language_network_breakdown"]
                + regional_state["left_posterior_temporal_inferior_parietal_proxy"]
            ) / 3.0
        )
        phenotypes["motor_parkinsonian_profile"] = self._clip01(
            (
                symptoms["parkinsonism"]
                + symptoms["apathy_inertia"]
                + regional_state["striatum_proxy"]
            ) / 3.0
        )
        phenotypes["language_variant_mean"] = self._clip01(
            (
                phenotypes["pnfa_profile"]
                + phenotypes["semantic_dementia_profile"]
                + phenotypes["lvppa_profile"]
            ) / 3.0
        )
        phenotypes["frontotemporal_imaging_signature"] = self._clip01(
            (
                latents["frontotemporal_degeneration"]
                + latents["frontotemporal_hypometabolism"]
                + regional_state["right_orbitofrontal_proxy"]
                + regional_state["right_anterior_temporal_proxy"]
            ) / 4.0
        )
        phenotypes["overall_ftncd_burden"] = self._clip01(
            0.35 * latents["frontotemporal_degeneration"]
            + 0.20 * latents["frontotemporal_hypometabolism"]
            + 0.25 * phenotypes["behavioral_variant_ftncd_profile"]
            + 0.20 * phenotypes["language_variant_mean"]
            - 0.10 * support_mean
        )

        return {
            "inputs": pd.Series(inputs, dtype=float),
            "variant_biases": pd.Series(variant_biases, dtype=float),
            "latents": pd.Series(latents, dtype=float),
            "regional_state": pd.Series(regional_state, dtype=float),
            "symptoms": pd.Series(symptoms, dtype=float),
            "phenotypes": pd.Series(phenotypes, dtype=float),
        }


if __name__ == "__main__":
    model = MajorOrMildFrontotemporalNeurocognitiveDisorderModel()

    built = model.build()
    print("\n=== Nodes (first 12) ===")
    print(built["nodes"].head(12).to_string(index=False))

    print("\n=== Edge count ===")
    print(len(built["edges"]))

    print("\n=== Resolved regions ===")
    if built["regions"]:
        for key, region in built["regions"].items():
            print(f"- {key}: {getattr(region, 'name', region)}")
    else:
        print("No atlas-backed regions resolved in this environment.")

    print("\n=== Connectivity among resolved circuit nodes ===")
    print(built["circuit_connectivity"].head(10).to_string(index=False))

    print("\n=== Example receptor / gene / connectivity tables ===")
    for key in [
        "right_orbitofrontal_proxy",
        "anterior_cingulate_proxy",
        "left_posterior_frontoinsular_proxy",
    ]:
        print(f"\nNode: {key}")
        print(f"  receptors rows: {len(built['receptors'].get(key, pd.DataFrame()))}")
        print(f"  genes rows: {len(built['genes'].get(key, pd.DataFrame()))}")
        print(f"  connectivity rows: {len(built['connectivity_profiles'].get(key, pd.DataFrame()))}")

    example = model.simulate(
        disease_stage=0.68,
        familial_genetic_load=0.45,
        mapt_tauopathy_load=0.50,
        tdp43_proteinopathy_load=0.30,
        fus_proteinopathy_load=0.10,
        right_frontoinsular_predominance=0.72,
        left_language_network_predominance=0.28,
        anterior_temporal_predominance=0.58,
        nigrostriatal_involvement=0.30,
        cognitive_reserve=0.20,
        structured_environment_support=0.30,
    )

    print("\n=== Variant biases ===")
    print(example["variant_biases"].round(3).to_string())

    print("\n=== Latent biology ===")
    print(example["latents"].round(3).to_string())

    print("\n=== Regional state ===")
    print(example["regional_state"].round(3).to_string())

    print("\n=== Symptoms ===")
    print(example["symptoms"].round(3).to_string())

    print("\n=== Phenotypes ===")
    print(example["phenotypes"].round(3).to_string())

    # Example optional anatomical assignment when siibra is available:
    # print(model.assign_mni_point((40, 24, -12)).head())
