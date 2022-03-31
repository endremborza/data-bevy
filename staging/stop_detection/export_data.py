import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from colassigner import get_all_cols
from tqdm.notebook import tqdm

from stops import PingFeatures, StopFeatures


def dump_data(takeout_zip):

    with TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(takeout_zip) as zfp:
            zfp.extractall(tempdir)

        lhdir = ("Takeout", "Location History")
        dump_raw_dfs(Path(tempdir, *lhdir, "Records.json"))
        dump_semantic(Path(tempdir, *lhdir, "Semantic Location History"))


def dump_raw_dfs(js_path):
    raw_recs = json.loads(js_path.read_text())
    pings = []
    acts = []
    infers = []
    scans = []
    for i, rec in enumerate(tqdm(raw_recs["locations"])):
        _extend(infers, rec.pop("inferredLocation", []), i)
        _extend(
            acts,
            [
                {**a, "timestamp": d["timestamp"]}
                for d in rec.pop("activity", [])
                for a in d["activity"]
            ],
            i,
        )
        _extend(scans, rec.pop("activeWifiScan", {}).get("accessPoints", []), i)
        rec.pop("locationMetadata", {})  # TODO
        pings.append(rec)

    pd.DataFrame(pings).to_parquet("pings.parquet")
    pd.DataFrame(acts).pipe(
        lambda df: pd.concat(
            [
                df.drop("extra", axis=1),
                df["extra"]
                .dropna()
                .pipe(lambda s: pd.DataFrame(s.tolist(), index=s.index))
                .rename(lambda s: f"extra_{s}"),
            ]
        )
    )  # TODO .to_parquet("acts.parquet")
    pd.DataFrame(infers)  # TODO.to_parquet("infers.parquet")
    pd.DataFrame(scans)  # TODO .to_parquet("scans.parquet")


def dump_semantic(sem_path):
    semantic_recs = []
    for jsp in sem_path.glob("**/*.json"):
        semantic_dic = json.loads(jsp.read_text())
        for tl_obj in semantic_dic["timelineObjects"]:
            pv = tl_obj.get("placeVisit")
            if pv:
                _ = pv.pop("otherCandidateLocations", [])  # TODO
                semantic_recs.append(
                    {
                        **{
                            f"semantic_{k}": v
                            for k, v in pv.pop("location", {}).items()
                        },
                        **pv.pop("duration", {}),
                        **pv,
                    }
                )

    pd.DataFrame(semantic_recs).to_parquet("semantic.parquet")


def parse_ping_df(df):
    return df.assign(
        **{
            PingFeatures.loc.lat: df["latitudeE7"] / 10**7,
            PingFeatures.loc.lon: df["longitudeE7"] / 10**7,
            PingFeatures.datetime: pd.to_datetime(df["timestamp"]),
            PingFeatures.device_id: df["deviceTag"],
        }
    ).loc[:, get_all_cols(PingFeatures)]


def parse_sem_df(df):
    return df.assign(
        **{
            StopFeatures.center.lat: df["semantic_latitudeE7"] / 10**7,
            StopFeatures.center.lon: df["semantic_longitudeE7"] / 10**7,
            StopFeatures.interval.start: df["startTimestamp"].pipe(pd.to_datetime),
            StopFeatures.interval.end: df["endTimestamp"].pipe(pd.to_datetime),
            StopFeatures.device_id: "N/A",
            StopFeatures.destination_label: df["semantic_placeId"],
            StopFeatures.is_home: df["semantic_semanticType"] == "TYPE_HOME",
            StopFeatures.is_work: df["semantic_semanticType"] == "TYPE_WORK",
            StopFeatures.info: df["semantic_address"],
            StopFeatures.n_events: -1,
            StopFeatures.stay_number: range(df.shape[0]),
        }
    ).loc[:, get_all_cols(StopFeatures)]


def _extend(olist, recl, i):
    olist += [{"ind": i, **d} for d in recl]
