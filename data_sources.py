"""
Data sources for the Norway Energy Analyzer.

Every function returns a pandas object and is cached by Streamlit, so each API is
called at most once per hour (prices, weather) or per day (history).
All calls run on the server, so keys stay secret and browsers can't block them.
"""

from __future__ import annotations

import datetime as dt
import os
import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

OSLO = "Europe/Oslo"
UA = {"User-Agent": "NorwayEnergyAnalyzer/1.0 (student thesis project)"}

AREAS = {
    "NO1": {"name": "Øst-Norge (Oslo)", "eic": "10YNO-1--------2", "station": "SN18700", "lat": 59.9423, "lon": 10.72, "alt": 94},
    "NO2": {"name": "Sør-Norge (Kristiansand)", "eic": "10YNO-2--------T", "station": "SN39040", "lat": 58.2042, "lon": 8.0767, "alt": 12},
    "NO3": {"name": "Midt-Norge (Trondheim)", "eic": "10YNO-3--------J", "station": "SN69100", "lat": 63.4581, "lon": 10.9242, "alt": 12},
    "NO4": {"name": "Nord-Norge (Tromsø)", "eic": "10YNO-4--------9", "station": "SN90450", "lat": 69.6537, "lon": 18.9368, "alt": 100},
    "NO5": {"name": "Vest-Norge (Bergen)", "eic": "10Y1001A1001A48H", "station": "SN50540", "lat": 60.3829, "lon": 5.3327, "alt": 12},
}
NEIGHBOURS = {"NO1": ["NO2", "NO3", "NO5", "SE3"], "NO2": ["NO1", "NO5", "DK1", "NL", "DE-LU", "GB"],
              "NO3": ["NO1", "NO4", "NO5", "SE2"], "NO4": ["NO3", "SE1", "SE2", "FI"], "NO5": ["NO1", "NO2", "NO3"]}
EIC = {k: v["eic"] for k, v in AREAS.items()} | {
    "SE1": "10Y1001A1001A44P", "SE2": "10Y1001A1001A45N", "SE3": "10Y1001A1001A46L", "FI": "10YFI-1--------U",
    "DK1": "10YDK-1--------W", "NL": "10YNL----------L", "DE-LU": "10Y1001A1001A82H", "GB": "10YGB----------A"}
PSR = {"B12": ("Hydro reservoir", "#2b6cb0"), "B11": ("Hydro run-of-river", "#63b3ed"), "B10": ("Hydro pumped storage", "#90cdf4"),
       "B19": ("Wind onshore", "#38a169"), "B18": ("Wind offshore", "#68d391"), "B16": ("Solar", "#ecc94b"),
       "B04": ("Gas", "#dd6b20"), "B17": ("Waste", "#a0aec0"), "B01": ("Biomass", "#975a16"),
       "B15": ("Other renewable", "#81e6d9"), "B20": ("Other", "#718096")}
WEATHER = {  # Frost element id: (label, unit)
    "mean(air_temperature P1D)": ("Air temperature", "°C"),
    "sum(precipitation_amount P1D)": ("Precipitation", "mm"),
    "mean(wind_speed P1D)": ("Wind speed", "m/s"),
    "max(wind_speed_of_gust P1D)": ("Wind gust (max)", "m/s"),
    "mean(air_pressure_at_sea_level P1D)": ("Air pressure", "hPa"),
    "mean(cloud_area_fraction P1D)": ("Cloud cover", "octas"),
    "surface_snow_thickness": ("Snow depth", "cm"),
}


def secret(name: str) -> str:
    """Key from Streamlit secrets (cloud) or environment / keys file (local)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def _get(url, params=None, headers=None, timeout=90):
    r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
    return r


# ---------------------------------------------------------------- ENTSO-E
def _entsoe(params: dict) -> list[tuple[dict, pd.Series]]:
    token = secret("ENTSOE_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_TOKEN missing in secrets")
    r = _get("https://web-api.tp.entsoe.eu/api", {**params, "securityToken": token}, timeout=120)
    root = ET.fromstring(r.content)
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    if "Acknowledgement" in root.tag:
        txt = " ".join(x.text or "" for x in root.iter(ns + "text"))
        if "No matching data" in txt:
            return []
        raise RuntimeError("ENTSO-E: " + txt[:200])
    out = []
    for ts in root.iter(ns + "TimeSeries"):
        def g(tag):
            el = ts.find(f".//{ns}{tag}")
            return el.text if el is not None else None
        attrs = {"psr": g("psrType"), "in": g("inBiddingZone_Domain.mRID"), "out": g("outBiddingZone_Domain.mRID"), "curve": g("curveType")}
        times, vals = [], []
        for p in ts.iter(ns + "Period"):
            t0 = pd.Timestamp(p.find(f"{ns}timeInterval/{ns}start").text)
            t1 = pd.Timestamp(p.find(f"{ns}timeInterval/{ns}end").text)
            step = pd.Timedelta(minutes={"PT15M": 15, "PT30M": 30, "PT60M": 60, "P1D": 1440, "P7D": 10080}.get(p.find(f"{ns}resolution").text, 60))
            pos = {}
            for pt in p.iter(ns + "Point"):
                q = pt.find(f"{ns}quantity")
                if q is None:
                    q = pt.find(f"{ns}price.amount")
                pos[int(pt.find(f"{ns}position").text)] = float(q.text)
            last = None
            for i in range(1, int((t1 - t0) / step) + 1):
                if i in pos:
                    last = pos[i]
                elif attrs["curve"] != "A03":
                    continue
                if last is not None:
                    times.append(t0 + (i - 1) * step)
                    vals.append(last)
        out.append((attrs, pd.Series(vals, index=pd.DatetimeIndex(times), dtype=float)))
    return out


def _entsoe_range(params: dict, start: dt.date, end: dt.date):
    """ENTSO-E allows one year per request; split longer ranges."""
    res, a = [], pd.Timestamp(start, tz=OSLO)
    stop = pd.Timestamp(end + dt.timedelta(days=1), tz=OSLO)
    while a < stop:
        b = min(a + pd.Timedelta(days=364), stop)
        f = lambda t: t.tz_convert("UTC").strftime("%Y%m%d%H%M")
        res += _entsoe({**params, "periodStart": f(a), "periodEnd": f(b)})
        a = b
    return res


def _hourly(series_list) -> pd.Series:
    if not series_list:
        return pd.Series(dtype=float)
    s = pd.concat(series_list).sort_index()
    s = s[~s.index.duplicated()]
    return s.tz_convert(OSLO).resample("h").mean().dropna()


@st.cache_data(ttl=3600, show_spinner="Downloading day-ahead prices from ENTSO-E…")
def prices(area: str, start: dt.date, end: dt.date) -> pd.Series:
    """Hourly day-ahead price, EUR/MWh (15-min prices since Oct 2025 averaged to hourly)."""
    z = EIC[area]
    return _hourly([s for _, s in _entsoe_range({"documentType": "A44", "in_Domain": z, "out_Domain": z}, start, end)])


@st.cache_data(ttl=3600, show_spinner="Downloading consumption from ENTSO-E…")
def load(area: str, start: dt.date, end: dt.date, kind: str = "actual") -> pd.Series:
    proc = {"actual": "A16", "dayahead": "A01", "weekahead": "A31"}[kind]
    return _hourly([s for _, s in _entsoe_range({"documentType": "A65", "processType": proc, "outBiddingZone_Domain": EIC[area]}, start, end)])


@st.cache_data(ttl=3600, show_spinner="Downloading production by type from ENTSO-E…")
def generation(area: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    by = {}
    for attrs, s in _entsoe_range({"documentType": "A75", "processType": "A16", "in_Domain": EIC[area]}, start, end):
        if attrs["out"] and not attrs["in"]:
            continue  # pumping consumption
        by.setdefault(attrs["psr"] or "B20", []).append(s)
    return pd.DataFrame({k: _hourly(v) for k, v in by.items()})


@st.cache_data(ttl=3600, show_spinner="Downloading cross-border flows from ENTSO-E…")
def flows(area: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Net flow per neighbour, MW: positive = into the area."""
    cols = {}
    for n in NEIGHBOURS[area]:
        try:
            imp = _hourly([s for _, s in _entsoe_range({"documentType": "A11", "in_Domain": EIC[area], "out_Domain": EIC[n]}, start, end)])
            exp = _hourly([s for _, s in _entsoe_range({"documentType": "A11", "in_Domain": EIC[n], "out_Domain": EIC[area]}, start, end)])
            if len(imp) or len(exp):
                cols[n] = imp.reindex(imp.index.union(exp.index)).fillna(0) - exp.reindex(imp.index.union(exp.index)).fillna(0)
        except Exception:
            pass
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- free price API (fallback, no key)
@st.cache_data(ttl=900, show_spinner="Downloading prices from hvakosterstrommen.no…")
def prices_hks(area: str, start: dt.date, end: dt.date) -> pd.Series:
    rows = []
    d = start
    while d <= end:
        r = _get(f"https://www.hvakosterstrommen.no/api/v1/prices/{d:%Y}/{d:%m-%d}_{area}.json", timeout=20)
        if r.status_code == 200:
            rows += r.json()
        d += dt.timedelta(days=1)
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    s = pd.Series(df["EUR_per_kWh"].values * 1000, index=pd.to_datetime(df["time_start"], utc=True).dt.tz_convert(OSLO))
    return s.resample("h").mean().dropna()


def prices_any(area, start, end) -> tuple[pd.Series, str]:
    """ENTSO-E if a token exists, else the free API. Recent days (incl. tomorrow) are topped up from the free API."""
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    if secret("ENTSOE_TOKEN"):
        try:
            s = prices(area, start, min(end, tomorrow))
            last = s.index.max().date() if len(s) else start - dt.timedelta(days=1)
            if last < min(end, tomorrow):
                extra = prices_hks(area, last + dt.timedelta(days=1), min(end, tomorrow))
                s = pd.concat([s, extra]).sort_index()
            return s[~s.index.duplicated()], "ENTSO-E"
        except Exception as e:
            st.toast(f"ENTSO-E failed ({e}); using hvakosterstrommen.no")
    return prices_hks(area, start, min(end, tomorrow)), "hvakosterstrommen.no"


# ---------------------------------------------------------------- NVE reservoirs
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading reservoir levels from NVE…")
def reservoirs(area_nr: int) -> pd.DataFrame:
    base = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/"
    data = pd.DataFrame(_get(base + "HentOffentligData").json())
    stats = pd.DataFrame(_get(base + "HentOffentligDataMinMaxMedian").json())
    data = data[(data.omrType == "EL") & (data.omrnr == area_nr)].copy()
    stats = stats[(stats.omrType == "EL") & (stats.omrnr == area_nr)].copy()
    med_col = "medianFyllingsGrad" if "medianFyllingsGrad" in stats else "medianFyllingsgrad"
    stats = stats.rename(columns={med_col: "median", "minFyllingsgrad": "min", "maxFyllingsgrad": "max"})[["iso_uke", "min", "median", "max"]]
    df = data.merge(stats, on="iso_uke", how="left")
    df["date"] = pd.to_datetime(df.iso_aar.astype(str) + "-" + df.iso_uke.astype(str) + "-1", format="%G-%V-%u")
    for c in ["fyllingsgrad", "min", "median", "max"]:
        df[c] = df[c] * 100
    df["deviation"] = df.fyllingsgrad - df["median"]
    return df.sort_values("date").set_index("date")


# ---------------------------------------------------------------- MET Norway
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading weather history from MET Frost…")
def weather_history(station: str, elements: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    cid = secret("FROST_ID")
    if not cid:
        raise RuntimeError("FROST_ID missing in secrets")
    cols = {}
    for el in elements:
        r = requests.get("https://frost.met.no/observations/v0.jsonld", auth=(cid, ""), headers=UA, timeout=90, params={
            "sources": station, "elements": el, "referencetime": f"{start}/{end + dt.timedelta(days=1)}",
            "timeoffsets": "default", "levels": "default", "qualities": "0,1,2,3,4"})
        if r.status_code in (404, 412):
            continue  # not measured at this station
        r.raise_for_status()
        vals = {}
        for o in r.json().get("data", []):
            vals.setdefault(pd.Timestamp(o["referenceTime"][:10]), o["observations"][0]["value"])
        cols[el] = pd.Series(vals, dtype=float)
    return pd.DataFrame(cols).sort_index()


@st.cache_data(ttl=24 * 3600)
def station_info(station: str) -> dict:
    cid = secret("FROST_ID")
    try:
        r = requests.get("https://frost.met.no/sources/v0.jsonld", params={"ids": station}, auth=(cid, ""), headers=UA, timeout=30)
        d = r.json()["data"][0]
        lon, lat = d.get("geometry", {}).get("coordinates", [None, None])[:2]
        return {"name": d.get("name", station).title(), "masl": d.get("masl"), "lat": lat, "lon": lon, "municipality": d.get("municipality", "")}
    except Exception:
        return {"name": station}


@st.cache_data(ttl=1800, show_spinner="Downloading weather forecast from MET…")
def weather_forecast(lat: float, lon: float, alt: int) -> pd.DataFrame:
    """Daily mean temperature, mean wind, total precipitation for the next ~9 days."""
    j = _get("https://api.met.no/weatherapi/locationforecast/2.0/compact", {"lat": lat, "lon": lon, "altitude": alt}).json()
    rows = []
    for ts in j["properties"]["timeseries"]:
        d = ts["data"]
        p = (d.get("next_1_hours") or {}).get("details", {}).get("precipitation_amount")
        if p is None and "next_1_hours" not in d:
            p = (d.get("next_6_hours") or {}).get("details", {}).get("precipitation_amount", 0)
        rows.append({"t": pd.Timestamp(ts["time"]).tz_convert(OSLO), "temp": d["instant"]["details"]["air_temperature"],
                     "wind": d["instant"]["details"]["wind_speed"], "precip": p or 0})
    df = pd.DataFrame(rows)
    df["day"] = df.t.dt.tz_localize(None).dt.normalize()
    g = df.groupby("day").agg(temp=("temp", "mean"), wind=("wind", "mean"), precip=("precip", "sum"), n=("temp", "size"))
    return g[g.n >= 3].drop(columns="n")


# ---------------------------------------------------------------- NVE HydAPI
@st.cache_data(ttl=24 * 3600, show_spinner="Finding river stations (NVE HydAPI)…")
def hyd_stations(counties: tuple[str, ...]) -> pd.DataFrame:
    key = secret("HYDAPI_KEY")
    if not key:
        return pd.DataFrame()
    r = _get("https://hydapi.nve.no/api/v1/Stations", {"Active": "OnlyActive"}, {"X-API-Key": key})
    st_ = pd.DataFrame(r.json()["data"])
    st_ = st_[st_.countyName.isin(counties) & st_.seriesList.apply(lambda l: any(x.get("parameter") == 1001 for x in (l or [])))]
    return st_.sort_values("drainageBasinArea", ascending=False)[["stationId", "stationName", "riverName", "drainageBasinArea"]].head(25)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading river discharge (NVE HydAPI)…")
def discharge(station_ids: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    key = secret("HYDAPI_KEY")
    r = _get("https://hydapi.nve.no/api/v1/Observations", {"StationId": ",".join(station_ids), "Parameter": "1001",
             "ResolutionTime": "1440", "ReferenceTime": f"{start}/{end}"}, {"X-API-Key": key})
    cols = {}
    for s in r.json().get("data", []):
        cols[f"{s['stationName']} ({s['stationId']})"] = pd.Series(
            {pd.Timestamp(o["time"][:10]): o["value"] for o in s.get("observations", []) if o.get("value") is not None}, dtype=float)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------- Norges Bank exchange rate
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading EUR/NOK rates from Norges Bank…")
def eur_nok(start: dt.date, end: dt.date) -> pd.Series:
    """Official daily EUR→NOK rate (business days), forward-filled to every calendar day."""
    r = _get("https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP", {
        "format": "csv", "startPeriod": str(start - dt.timedelta(days=10)), "endPeriod": str(end), "locale": "en"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), sep=None, engine="python")
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values, index=pd.to_datetime(df["TIME_PERIOD"])).dropna()
    days = pd.date_range(s.index.min(), pd.Timestamp(end) + pd.Timedelta(days=10), freq="D")
    return s.reindex(days).ffill()


def daily(s: pd.Series | pd.DataFrame):
    """Hourly (Oslo time) → daily mean with a naive date index."""
    if s is None or len(s) == 0:
        return s
    out = s.resample("D").mean()
    out.index = out.index.tz_localize(None)
    return out
