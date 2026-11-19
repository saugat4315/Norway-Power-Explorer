"""
⚡ Norway Power Explorer
Day-ahead power price (bold line) compared with weather, hydrology, demand/supply and
cross-border flows. All settings live in the left sidebar.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (see README.md)
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_sources as ds

st.set_page_config(page_title="Norway Power Explorer", page_icon="⚡", layout="wide")

PRICE_COLOR = "#2b6cb0"
COMPARE_COLORS = ["#e47aa0", "#38a169", "#dd6b20", "#805ad5", "#319795", "#d69e2e", "#718096"]
TODAY = pd.Timestamp.now(tz=ds.OSLO).date()
TOMORROW = TODAY + dt.timedelta(days=1)

STATIONS = {
    "SN18700": "Oslo - Blindern (Oslo)", "SN4780": "Gardermoen (Ullensaker)", "SN17150": "Rygge (Moss)",
    "SN39040": "Kristiansand - Kjevik", "SN69100": "Trondheim - Værnes", "SN90450": "Tromsø",
    "SN50540": "Bergen - Florida",
}
AREA_STATION = {"NO1": "SN18700", "NO2": "SN39040", "NO3": "SN69100", "NO4": "SN90450", "NO5": "SN50540"}

# ====================================================================== sidebar
with st.sidebar:
    st.title("⚡ Norway Power Explorer")

    area = st.selectbox("Bidding zone", list(ds.AREAS), format_func=lambda a: f"{a} · {ds.AREAS[a]['name']}")
    station = st.selectbox("Weather station", list(STATIONS), index=list(STATIONS).index(AREA_STATION[area]),
                           format_func=STATIONS.get, help="MET Norway observation station used for the weather series")
    st.caption(f"{STATIONS[station].split(' (')[0]} · measures temperature, precipitation, wind and more")

    rng = st.date_input("Date range", (dt.date(TODAY.year, 1, 1), TOMORROW), min_value=dt.date(2015, 1, 5),
                        max_value=TOMORROW, format="YYYY/MM/DD")
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (rng[0], rng[0]))

    resolution = st.radio("Resolution", ["Hourly", "Daily", "Weekly", "Monthly"], index=1, horizontal=True)

    st.subheader("Price unit")
    unit_choice = st.radio("Show prices in", ["EUR/MWh", "NOK/kWh"], horizontal=True, label_visibility="collapsed")
    vat = st.checkbox("Include 25 % VAT", False, help="Not charged in NO4. Grid tariffs are never included.")

    st.subheader("Compare price with")
    st.caption("Day-ahead power price is always shown (bold line).")
    has_frost, has_entsoe, has_hyd = bool(ds.secret("FROST_ID")), bool(ds.secret("ENTSOE_TOKEN")), bool(ds.secret("HYDAPI_KEY"))

    st.markdown("**Weather (MET)**")
    wx_sel = st.multiselect("Weather", list(ds.WEATHER), default=["mean(air_temperature P1D)"] if has_frost else [],
                            format_func=lambda e: f"{ds.WEATHER[e][0]} ({ds.WEATHER[e][1]})", disabled=not has_frost,
                            label_visibility="collapsed")
    st.markdown("**Hydro (NVE)**")
    use_res = st.checkbox("Reservoir filling (%)", True)
    use_resdev = st.checkbox("Reservoir vs 20-yr median (pp)", False)
    hyd_ids = []
    if has_hyd and st.checkbox("River discharge (m³/s)", False):
        try:
            hs = ds.hyd_stations(("Oslo", "Akershus", "Østfold", "Buskerud", "Innlandet", "Viken")).set_index("stationId")
            hyd_ids = st.multiselect("River stations", hs.index.tolist(), default=hs.index.tolist()[:1],
                                     format_func=lambda i: f"{hs.at[i, 'stationName']} – {hs.at[i, 'riverName']}")
        except Exception as e:
            st.warning(f"HydAPI: {e}")
    st.markdown("**Demand & supply (ENTSO-E)**")
    use_load = st.checkbox("Consumption (MW)", False, disabled=not has_entsoe)
    use_gen = st.checkbox("Production (MW)", False, disabled=not has_entsoe)
    use_hydro = st.checkbox("Hydro production (MW)", False, disabled=not has_entsoe)
    use_wind = st.checkbox("Wind production (MW)", False, disabled=not has_entsoe)
    st.markdown("**Transmission (ENTSO-E)**")
    use_net = st.checkbox(f"Net import into {area} (MW)", False, disabled=not has_entsoe)

    if not has_entsoe or not has_frost:
        st.caption("Greyed-out options need a key in the app's Secrets.")
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# ====================================================================== units
NOK = unit_choice == "NOK/kWh"
VAT = 1.25 if (vat and area != "NO4") else 1.0
U = unit_choice + (" incl. VAT" if VAT > 1 else "")
DP = ".3f" if NOK else ".1f"
RULE = {"Hourly": None, "Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[resolution]


def conv(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0 or not NOK:
        return s * VAT if s is not None else s
    days = s.index.tz_localize(None).normalize()
    try:
        r = ds.eur_nok(days.min().date(), days.max().date()).reindex(days).ffill().bfill().values
    except Exception:
        r = 11.7
    return s * r / 1000 * VAT


def to_res(s, how="mean"):
    """Hourly (tz-aware) or daily (naive) series → chosen resolution, naive index."""
    if s is None or len(s) == 0:
        return s
    if getattr(s.index, "tz", None) is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
        s = s[~s.index.duplicated()]   # the repeated hour when clocks go back in October
    if RULE is None:
        return s
    kw = {"label": "left", "closed": "left"} if RULE.startswith("W") else {}
    return getattr(s.resample(RULE, **kw), how)().dropna()


# ====================================================================== data
st.title("Day-ahead price vs. drivers")
st.caption(f"**{area}** · {ds.AREAS[area]['name']} · {start:%d.%m.%Y} – {end:%d.%m.%Y} · {resolution.lower()} · prices in {U}")

if start > end:
    st.error("Start date is after end date.")
    st.stop()

price_eur, src = ds.prices_any(area, start, end)
if price_eur.empty:
    st.error("No price data for this period.")
    st.stop()
P = conv(price_eur)
Pr = to_res(P)

series = {}  # label -> (series at chosen resolution, unit)
data_end = min(end, TODAY)
if wx_sel:
    try:
        W = ds.weather_history(station, tuple(wx_sel), start, data_end)
        for el in W:
            lab, u = ds.WEATHER[el]
            series[lab] = (to_res(W[el], "sum" if "precip" in el else "mean"), u)
    except Exception as e:
        st.warning(f"Weather (Frost): {e}")
if use_res or use_resdev:
    try:
        R = ds.reservoirs(int(area[-1]))
        R = R[(R.index >= pd.Timestamp(start) - pd.Timedelta(days=7)) & (R.index <= pd.Timestamp(end))]
        daily_idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(data_end), freq="D")
        for on, col, lab, u in [(use_res, "fyllingsgrad", "Reservoir filling", "%"), (use_resdev, "deviation", "Reservoir vs median", "pp")]:
            if on:
                # weekly NVE value carried over each day of its week, then grouped like the price
                s = R[col].reindex(R.index.union(daily_idx)).ffill().reindex(daily_idx)
                series[lab] = (s if RULE is None else to_res(s), u)
    except Exception as e:
        st.warning(f"NVE reservoirs: {e}")
if hyd_ids:
    try:
        H = ds.discharge(tuple(hyd_ids), start, data_end)
        for c in H:
            series[c.split(" (")[0] + " discharge"] = (to_res(H[c]), "m³/s")
    except Exception as e:
        st.warning(f"HydAPI: {e}")
if use_load:
    try:
        series["Consumption"] = (to_res(ds.load(area, start, data_end)), "MW")
    except Exception as e:
        st.warning(f"ENTSO-E consumption: {e}")
if use_gen or use_hydro or use_wind:
    try:
        G = ds.generation(area, start, data_end)
        if use_gen:
            series["Production"] = (to_res(G.sum(axis=1)), "MW")
        if use_hydro:
            series["Hydro production"] = (to_res(G[[c for c in G if c in ("B10", "B11", "B12")]].sum(axis=1)), "MW")
        if use_wind:
            series["Wind production"] = (to_res(G[[c for c in G if c in ("B18", "B19")]].sum(axis=1)), "MW")
    except Exception as e:
        st.warning(f"ENTSO-E production: {e}")
if use_net:
    try:
        F = ds.flows(area, start, data_end)
        series[f"Net import into {area}"] = (to_res(F.sum(axis=1)), "MW")
    except Exception as e:
        st.warning(f"ENTSO-E flows: {e}")

# ====================================================================== KPIs
k = st.columns(5)
k[0].metric("Average price", f"{P.mean():{DP}} {U}")
k[1].metric("Lowest hour", f"{P.min():{DP}}", P.idxmin().strftime("%d.%m %H:00"), delta_color="off")
k[2].metric("Highest hour", f"{P.max():{DP}}", P.idxmax().strftime("%d.%m %H:00"), delta_color="off")
k[3].metric("Volatility (std)", f"{P.std():{DP}}")
tm = P[P.index.date == TOMORROW]
k[4].metric("Tomorrow average", f"{tm.mean():{DP}}" if len(tm) else "after ~13:00")


# ====================================================================== charts
def twin_chart(label, s, unit, color):
    fig = go.Figure()
    step = "hv" if label.startswith("Reservoir") else "linear"
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{label} ({unit})", yaxis="y2",
                             line=dict(color=color, width=1.6, shape=step)))
    fig.add_trace(go.Scatter(x=Pr.index, y=Pr.values, name=f"Day-ahead price ({U})",
                             line=dict(color=PRICE_COLOR, width=3.5)))
    fig.update_layout(
        template="plotly_white", height=460, hovermode="x unified", margin=dict(t=40, r=70, b=40, l=70),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13)),
        yaxis=dict(title=dict(text=f"Day-ahead price ({U})", font=dict(color=PRICE_COLOR)), gridcolor="#eef1f5"),
        yaxis2=dict(title=dict(text=f"{label} ({unit})", font=dict(color=color)), overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color=color)),
        xaxis=dict(gridcolor="#eef1f5"))
    return fig


def corr(s):
    j = pd.concat([Pr.rename("p"), s.rename("x")], axis=1).dropna()
    if len(j) < 5 or j.x.std() == 0:
        return np.nan, np.nan, len(j)
    r = j.p.corr(j.x)
    return r, r * j.p.std() / j.x.std(), len(j)


if not series:
    fig = go.Figure(go.Scatter(x=Pr.index, y=Pr.values, name=f"Day-ahead price ({U})", line=dict(color=PRICE_COLOR, width=3)))
    fig.update_layout(template="plotly_white", height=460, yaxis_title=U, margin=dict(t=30, r=30, b=40, l=70))
    st.plotly_chart(fig, width="stretch")
    st.info("Tick a parameter under **Compare price with** in the sidebar to see it next to the price.")
else:
    rows = []
    for i, (label, (s, unit)) in enumerate(series.items()):
        if s is None or len(s) == 0:
            st.warning(f"No data for {label} in this period.")
            continue
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        st.plotly_chart(twin_chart(label, s, unit, color), width="stretch")
        r, slope, n = corr(s)
        if not np.isnan(r):
            strength = "strong" if abs(r) >= .5 else "moderate" if abs(r) >= .3 else "weak" if abs(r) >= .1 else "no"
            st.caption(f"Correlation with price: **{r:+.2f}** ({strength}) · price changes about {slope:+{DP[:-1]}f} {U} "
                       f"per 1 {unit} · {n} {resolution.lower()} points")
            rows.append({"Parameter": label, "Correlation": r, "Price change per unit": f"{slope:+{DP[:-1]}f} {U} per {unit}", "Points": n})
    if len(rows) > 1:
        st.subheader("Which parameter moves most with the price?")
        st.dataframe(pd.DataFrame(rows).sort_values("Correlation", key=lambda c: -c.abs()), hide_index=True, width="stretch",
                     column_config={"Correlation": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.2f")})
        st.caption("Correlation shows values moving together, not cause and effect.")

with st.expander("📄 Data table & download"):
    table = pd.concat([Pr.rename(f"Price ({U})")] + [s.rename(f"{l} ({u})") for l, (s, u) in series.items() if s is not None and len(s)], axis=1)
    st.dataframe(table, width="stretch")
    st.download_button("Download CSV", table.to_csv().encode(), f"{area}_{start}_{end}.csv", "text/csv")

st.caption(f"Prices: {src} (excl. VAT unless ticked; 15-min prices averaged to hourly) · Weather: MET Norway Frost · "
           "Reservoirs: NVE · Rivers: NVE HydAPI · Load, production, flows: ENTSO-E · EUR→NOK: Norges Bank")
