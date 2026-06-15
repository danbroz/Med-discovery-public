from __future__ import annotations

"""
Sexual Sadism Disorder atlas-grounded siibra scaffold.

This script translates a chapter-level biological discussion of Sexual Sadism Disorder
into a transparent, atlas-grounded mechanistic scaffold using siibra.
Where the chapter is system-level or the atlas labels may vary across siibra
versions, this model uses explicit proxy nodes rather than inventing false
parcel precision.

Design principles
-----------------
- Use atlas-backed anchors only where the chapter is reasonably specific.
- Keep chemistry as latent biology unless the text clearly localizes it.
- Represent systems-level claims with explicit proxies instead of inventing
  false parcel precision.
- Keep the simulator acyclic and normalized to 0..1 for interpretability.

Important
---------
This scaffold is a research and teaching aid. It is not a validated disease
model and must not be used for diagnosis or treatment decisions.

Chapter-derived mechanistic emphasis
------------------------------------
    - reward-circuit misattribution causing others' suffering to acquire appetitive salience
    - dopaminergic reinforcement, glutamate/GABA imbalance, and prefrontal inhibitory failure as core mechanisms
    - heritable aggression/impulsivity traits plus trauma-related epigenetic sensitization as upstream risk architecture
    - dysfunctional interaction among ventral striatal reward systems, amygdala salience coding, empathy networks, and prefrontal control
"""

import re
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


DEFAULT_SEXUAL_SADISM_DISORDER_GENE_PANEL: List[str] = ['SLC6A4',
 'MAOA',
 'COMT',
 'DRD2',
 'DRD4',
 'BDNF',
 'NR3C1',
 'FKBP5',
 'GABRA2',
 'GRIN2B',
 'OXTR',
 'SLC6A2']


class SexualSadismDisorderModel:
    """
    Atlas-grounded research scaffold for Sexual Sadism Disorder.

    The chapter frames Sexual Sadism Disorder as involving:
    - reward-circuit misattribution causing others' suffering to acquire appetitive salience
    - dopaminergic reinforcement, glutamate/GABA imbalance, and prefrontal inhibitory failure as core mechanisms
    - heritable aggression/impulsivity traits plus trauma-related epigenetic sensitization as upstream risk architecture
    - dysfunctional interaction among ventral striatal reward systems, amygdala salience coding, empathy networks, and prefrontal control

    Several nodes therefore remain explicit proxies because the chapter is more
    systems-level than parcel-precise.
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
                "siibra is required to use this scaffold. Install siibra in your "
                "Python environment before running the model."
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

        self.region_candidates: Dict[str, List[str]] = {'acc': ['Area p24ab left', 'Area a24 left', 'anterior cingulate cortex', 'ACC', 'cingulate cortex'],
 'amygdala': ['LA (Amygdala) left', 'LB (Amygdala) left', 'CM (Amygdala) left', 'amygdala'],
 'dlpfc': ['Area 46 left',
           'Area 9/46d left',
           'Area 9 left',
           'dorsolateral prefrontal cortex',
           'middle frontal gyrus'],
 'insula': ['Area Id7 left', 'Area Ig2 left', 'insula', 'insular cortex'],
 'ofc': ['Area Fo4 left', 'Area Fo3 left', 'orbitofrontal cortex', 'OFC'],
 'ventral_striatum_proxy': ['nucleus accumbens', 'ventral striatum', 'striatum']}
        self.input_nodes: Dict[str, str] = {'compulsive_fantasy_rehearsal': 'Repetitive reinforcing rehearsal of sadistic fantasies and scripts.',
 'empathy_deficit_load': 'Reduced empathic responsiveness or callousness burden.',
 'impulsivity_disinhibition': 'Baseline inhibitory-control weakness and behavioral disinhibition.',
 'inhibitory_control_support': 'Protective support for top-down inhibition and external containment.',
 'trait_aggression_vulnerability': 'Heritable aggression-related trait burden.',
 'trauma_stress_embedding': 'Early trauma/stress burden with lasting biological embedding.',
 'treatment_engagement': 'Protective treatment engagement that can reduce symptom expression.'}
        self.latent_nodes: Dict[str, str] = {'amygdala_salience_reversal': 'Threat or distress cues taking on aberrant appetitive salience.',
 'dissociative_glutamatergic_instability': 'Altered excitatory/inhibitory balance facilitating dissociative '
                                           'or disinhibited states.',
 'mesolimbic_reward_misattribution': 'Aberrant reward coding of suffering-related cues.',
 'prefrontal_inhibitory_failure': 'Insufficient orbitofrontal/dorsolateral control over aggressive sexual '
                                  'impulses.',
 'trauma_epigenetic_sensitization': 'Trauma-linked molecular sensitization of stress and aggression systems.'}
        self.symptom_nodes: Dict[str, str] = {'aggressive_sexual_urge_disinhibition': 'Failure to inhibit aggressive sexual urges.',
 'coercive_sadistic_fantasy_intensity': 'Intensity and dominance of sadistic fantasy content.',
 'compulsive_preoccupation': 'Persistent preoccupation with sadistic fantasies or urges.',
 'empathy_suppression': "Reduced affective resonance with others' distress.",
 'violence_risk_expression': 'Expression-level risk signal for coercive or violent enactment.'}
        self.edge_table: List[Dict[str, str]] = [{'relation': 'rehearsal reinforces reward value assigned to sadistic cues',
  'source': 'compulsive_fantasy_rehearsal',
  'ssd_change': 'increased',
  'target': 'mesolimbic_reward_misattribution'},
 {'relation': 'baseline disinhibition weakens top-down control',
  'source': 'impulsivity_disinhibition',
  'ssd_change': 'increased',
  'target': 'prefrontal_inhibitory_failure'},
 {'relation': 'early stress can biologically sensitize aggression and arousal systems',
  'source': 'trauma_stress_embedding',
  'ssd_change': 'increased',
  'target': 'trauma_epigenetic_sensitization'},
 {'relation': 'trauma-linked sensitization can distort salience processing',
  'source': 'trauma_epigenetic_sensitization',
  'ssd_change': 'increased',
  'target': 'amygdala_salience_reversal'},
 {'relation': 'aberrant reward coding strengthens fantasy intensity',
  'source': 'mesolimbic_reward_misattribution',
  'ssd_change': 'increased',
  'target': 'coercive_sadistic_fantasy_intensity'},
 {'relation': 'poor top-down control permits aggressive urge escalation',
  'source': 'prefrontal_inhibitory_failure',
  'ssd_change': 'increased',
  'target': 'aggressive_sexual_urge_disinhibition'},
 {'relation': 'callous-empathy deficit reduces affective resonance with suffering',
  'source': 'empathy_deficit_load',
  'ssd_change': 'increased',
  'target': 'empathy_suppression'},
 {'relation': 'high-intensity fantasies feed persistent preoccupation',
  'source': 'coercive_sadistic_fantasy_intensity',
  'ssd_change': 'increased',
  'target': 'compulsive_preoccupation'},
 {'relation': 'disinhibited aggressive urges elevate enactment risk',
  'source': 'aggressive_sexual_urge_disinhibition',
  'ssd_change': 'increased',
  'target': 'violence_risk_expression'},
 {'relation': 'reduced empathy weakens inhibitory social-emotional brakes',
  'source': 'empathy_suppression',
  'ssd_change': 'increased',
  'target': 'violence_risk_expression'},
 {'relation': 'treatment and containment can reduce expressed risk',
  'source': 'treatment_engagement',
  'ssd_change': 'decreased',
  'target': 'violence_risk_expression'},
 {'relation': 'protective support can buffer control failure',
  'source': 'inhibitory_control_support',
  'ssd_change': 'decreased',
  'target': 'prefrontal_inhibitory_failure'}]

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

    @staticmethod
    def _normalize_string(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text).replace("_", " ").replace("-", " ")).strip().lower()

    @staticmethod
    def _preferred_hemisphere(spec: str) -> Optional[str]:
        low = str(spec).lower()
        if "left" in low:
            return "left"
        if "right" in low:
            return "right"
        return None

    def _modality_candidates(self, kind: str) -> List[Any]:
        cands: List[Any] = []
        try:  # pragma: no cover - depends on siibra version
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

    def _safe_features_any(
        self,
        concept: Any,
        modalities: Sequence[Any],
        **kwargs: Any,
    ) -> List[Any]:
        for modality in modalities:
            try:
                with siibra.QUIET:
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _try_find_regions(self, query: str) -> List[Any]:
        finders = [
            getattr(self.atlas, "find_regions", None),
            getattr(self.parcellation, "find", None),
        ]
        for finder in finders:
            if finder is None:
                continue
            attempts = [
                {
                    "query": query,
                    "all_versions": False,
                    "filter_children": False,
                    "find_topmost": False,
                },
                {
                    "regionspec": query,
                    "filter_children": False,
                    "find_topmost": False,
                },
                {"regionspec": query},
                {"query": query},
                {},
            ]
            for params in attempts:
                try:
                    found = finder(**params) if params else finder(query)
                    if found:
                        return list(found)
                except Exception:
                    continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        matches = self._try_find_regions(query)
        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if not parc_name or "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._normalize_string(self._name_of(region))
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "insula",
            "anterior cingulate cortex",
            "orbitofrontal cortex",
            "dorsolateral prefrontal cortex",
            "prefrontal cortex",
            "basal ganglia",
            "thalamus",
            "brainstem",
            "temporal pole",
        } else 0
        lobar_penalty = 1 if any(k in name for k in ["frontal", "temporal", "parietal", "cingulate"]) and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, lobar_penalty)

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
                pref = self._preferred_hemisphere(spec)
                if pref:
                    matches = sorted(
                        matches,
                        key=lambda r: (self._normalize_string(self._name_of(r)).find(pref) == -1, self._region_rank(r)),
                    )
                else:
                    matches = sorted(matches, key=self._region_rank)
                return matches[0]
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
            rows.append({
                "name": row[0],
                "identifier": row[1],
                "parcellation": row[2],
            })
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
        exact = [x for x in labels if self._normalize_string(self._name_of(x)) == self._normalize_string(region.name)]
        if exact:
            return exact[0]
        rn = self._normalize_string(region.name)
        fuzzy = [x for x in labels if rn in self._normalize_string(self._name_of(x)) or self._normalize_string(self._name_of(x)) in rn]
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
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_SEXUAL_SADISM_DISORDER_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []

        for key, desc in self.input_nodes.items():
            nodes.append({
                "key": key,
                "label": key.replace("_", " ").title(),
                "node_type": "input",
                "description": desc,
                "atlas_region": None,
                "region_identifier": None,
                "centroid_mni": None,
                "volume_mm3": None,
                "feature_summary": None,
            })

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            if region is None:
                warnings.warn(
                    f"Could not resolve a Julich region for '{key}'. Keeping it as a proxy node."
                )
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append({
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "region_proxy",
                    "description": "Systems-level proxy node retained because atlas resolution is unavailable or intentionally broad.",
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": "proxy/unresolved",
                })
                continue

            self.region_objects[key] = region
            centroid_mni, volume_mm3 = self._main_component(region)
            receptor_df = self._receptor_table(region)
            gene_df = self._gene_table(region, gene_panel)
            conn_df = self._connectivity_profile(region, max_rows=connectivity_rows)
            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = conn_df

            nodes.append({
                "key": key,
                "label": region.name,
                "node_type": "region",
                "description": "Atlas-backed circuit node",
                "atlas_region": region.name,
                "region_identifier": getattr(region, "identifier", None),
                "centroid_mni": centroid_mni,
                "volume_mm3": volume_mm3,
                "feature_summary": (
                    f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                    f"genes={'yes' if not gene_df.empty else 'no'}; "
                    f"connectivity={'yes' if not conn_df.empty else 'no'}"
                ),
            })

        for key, desc in self.latent_nodes.items():
            nodes.append({
                "key": key,
                "label": key.replace("_", " ").title(),
                "node_type": "latent_biology",
                "description": desc,
                "atlas_region": None,
                "region_identifier": None,
                "centroid_mni": None,
                "volume_mm3": None,
                "feature_summary": None,
            })

        for key, desc in self.symptom_nodes.items():
            nodes.append({
                "key": key,
                "label": key.replace("_", " ").title(),
                "node_type": "symptom",
                "description": desc,
                "atlas_region": None,
                "region_identifier": None,
                "centroid_mni": None,
                "volume_mm3": None,
                "feature_summary": None,
            })

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

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        rows: List[Dict[str, Any]] = []
        region_items = list(self.region_objects.items())
        for src_key, src_region in region_items:
            src_label = self._match_region_label(list(matrix.index), src_region)
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src_region)
            if src_label is None:
                continue

            for dst_key, dst_region in region_items:
                if src_key == dst_key:
                    continue
                dst_label = self._match_region_label(list(matrix.columns), dst_region)
                if dst_label is None:
                    dst_label = self._match_region_label(list(matrix.index), dst_region)
                if dst_label is None:
                    continue

                value = None
                try:
                    value = float(matrix.loc[src_label, dst_label])
                except Exception:
                    try:
                        value = float(matrix.loc[dst_label, src_label])
                    except Exception:
                        continue

                rows.append({
                    "source": src_key,
                    "target": dst_key,
                    "value": value,
                })

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def region_mask(self, node_key: str, maptype: str = "labelled") -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype=maptype)
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype=maptype)
            except Exception:
                return None

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
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

    def simulate(
        self,
        trait_aggression_vulnerability: float = 0.55,
        impulsivity_disinhibition: float = 0.6,
        trauma_stress_embedding: float = 0.45,
        empathy_deficit_load: float = 0.55,
        compulsive_fantasy_rehearsal: float = 0.6,
        inhibitory_control_support: float = 0.25,
        treatment_engagement: float = 0.2,
    ) -> Dict[str, pd.Series]:
        clip = self._clip01

        inputs = pd.Series(
            {
                "trait_aggression_vulnerability": self._clip01(trait_aggression_vulnerability),
            "impulsivity_disinhibition": self._clip01(impulsivity_disinhibition),
            "trauma_stress_embedding": self._clip01(trauma_stress_embedding),
            "empathy_deficit_load": self._clip01(empathy_deficit_load),
            "compulsive_fantasy_rehearsal": self._clip01(compulsive_fantasy_rehearsal),
            "inhibitory_control_support": self._clip01(inhibitory_control_support),
            "treatment_engagement": self._clip01(treatment_engagement)
            },
            name="input_value",
        )
        i = inputs.to_dict()

        latents: Dict[str, float] = {}
        latents["trauma_epigenetic_sensitization"] = clip(0.45*i["trauma_stress_embedding"] + 0.15*i["trait_aggression_vulnerability"] + 0.10*i["empathy_deficit_load"] - 0.10*i["treatment_engagement"])
        latents["mesolimbic_reward_misattribution"] = clip(0.30*i["compulsive_fantasy_rehearsal"] + 0.20*i["trait_aggression_vulnerability"] + 0.20*i["empathy_deficit_load"] + 0.10*latents["trauma_epigenetic_sensitization"] - 0.10*i["treatment_engagement"])
        latents["prefrontal_inhibitory_failure"] = clip(0.40*i["impulsivity_disinhibition"] + 0.20*i["trauma_stress_embedding"] + 0.15*i["compulsive_fantasy_rehearsal"] - 0.20*i["inhibitory_control_support"] - 0.10*i["treatment_engagement"])
        latents["amygdala_salience_reversal"] = clip(0.30*latents["trauma_epigenetic_sensitization"] + 0.25*latents["mesolimbic_reward_misattribution"] + 0.15*i["empathy_deficit_load"])
        latents["dissociative_glutamatergic_instability"] = clip(0.30*i["trauma_stress_embedding"] + 0.25*i["impulsivity_disinhibition"] + 0.15*i["compulsive_fantasy_rehearsal"] - 0.10*i["treatment_engagement"])

        regional_state: Dict[str, float] = {}
        regional_state["amygdala"] = clip(0.45*latents["amygdala_salience_reversal"] + 0.15*latents["trauma_epigenetic_sensitization"])
        regional_state["ofc"] = clip(0.45*latents["prefrontal_inhibitory_failure"] + 0.15*i["impulsivity_disinhibition"])
        regional_state["dlpfc"] = clip(0.35*latents["prefrontal_inhibitory_failure"] + 0.15*latents["dissociative_glutamatergic_instability"])
        regional_state["acc"] = clip(0.25*latents["amygdala_salience_reversal"] + 0.20*i["empathy_deficit_load"])
        regional_state["insula"] = clip(0.30*i["empathy_deficit_load"] + 0.20*latents["amygdala_salience_reversal"])
        regional_state["ventral_striatum_proxy"] = clip(0.55*latents["mesolimbic_reward_misattribution"] + 0.10*i["compulsive_fantasy_rehearsal"])

        symptoms: Dict[str, float] = {}
        symptoms["coercive_sadistic_fantasy_intensity"] = clip(0.45*latents["mesolimbic_reward_misattribution"] + 0.20*latents["amygdala_salience_reversal"] + 0.15*i["compulsive_fantasy_rehearsal"])
        symptoms["aggressive_sexual_urge_disinhibition"] = clip(0.40*latents["prefrontal_inhibitory_failure"] + 0.25*latents["mesolimbic_reward_misattribution"] + 0.15*latents["dissociative_glutamatergic_instability"] - 0.15*i["inhibitory_control_support"])
        symptoms["empathy_suppression"] = clip(0.45*i["empathy_deficit_load"] + 0.20*regional_state["insula"] + 0.10*regional_state["acc"])
        symptoms["compulsive_preoccupation"] = clip(0.40*symptoms["coercive_sadistic_fantasy_intensity"] + 0.20*latents["mesolimbic_reward_misattribution"] + 0.10*i["compulsive_fantasy_rehearsal"])
        symptoms["violence_risk_expression"] = clip(0.35*symptoms["aggressive_sexual_urge_disinhibition"] + 0.25*symptoms["empathy_suppression"] + 0.20*symptoms["compulsive_preoccupation"] - 0.15*i["treatment_engagement"] - 0.10*i["inhibitory_control_support"])

        phenotypes: Dict[str, float] = {}
        phenotypes["reward_dominant_sadistic_profile"] = clip((latents["mesolimbic_reward_misattribution"] + symptoms["coercive_sadistic_fantasy_intensity"] + regional_state["ventral_striatum_proxy"]) / 3.0)
        phenotypes["disinhibited_aggressive_profile"] = clip((latents["prefrontal_inhibitory_failure"] + symptoms["aggressive_sexual_urge_disinhibition"] + regional_state["ofc"]) / 3.0)
        phenotypes["empathy_collapse_risk_profile"] = clip((symptoms["empathy_suppression"] + symptoms["violence_risk_expression"] + regional_state["amygdala"]) / 3.0)

        return {
            "inputs": inputs,
            "latents": pd.Series(latents, name="latent_value"),
            "regional_state": pd.Series(regional_state, name="regional_value"),
            "symptoms": pd.Series(symptoms, name="symptom_value"),
            "phenotypes": pd.Series(phenotypes, name="phenotype_value"),
        }


if __name__ == "__main__":
    model = SexualSadismDisorderModel()
    bundle = model.build()

    print("\n=== Nodes (head) ===")
    print(bundle["nodes"].head(12).to_string(index=False))

    print("\n=== Edges (head) ===")
    print(bundle["edges"].head(12).to_string(index=False))

    for region_key in ['amygdala', 'ventral_striatum_proxy']:
        if region_key in bundle["genes"] and not bundle["genes"][region_key].empty:
            print(f"\n=== Gene summary: {region_key} ===")
            print(bundle["genes"][region_key].head(10).to_string(index=False))
        if region_key in bundle["connectivity_profiles"] and not bundle["connectivity_profiles"][region_key].empty:
            print(f"\n=== Connectivity profile: {region_key} ===")
            print(bundle["connectivity_profiles"][region_key].head(10).to_string(index=False))

    sim = model.simulate()
    print("\n=== Inputs ===")
    print(sim["inputs"].to_string())
    print("\n=== Latents ===")
    print(sim["latents"].sort_values(ascending=False).to_string())
    print("\n=== Regional state ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())
    print("\n=== Symptoms ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())
    print("\n=== Phenotypes ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((0, -24, 10)).head())
