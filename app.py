import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from geopy.geocoders import Nominatim

from pipeline import FEATURES, LOCATIONS, NDVI_THR, get_location, load_all

st.set_page_config(page_title="Crop Stress Early Warning", page_icon="🌾", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.card {border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 0.8rem; border: 1px solid #e5e7eb;}
.high {background:#fdecea; border-color:#e74c3c;}
.medium {background:#fff7e6; border-color:#f39c12;}
.low {background:#e8f8ef; border-color:#27ae60;}
.badge {display:inline-block; padding:4px 12px; border-radius:999px; font-weight:600; color:white;}
.step {background:#f4f7f5; border-radius:12px; padding:0.9rem; height:100%;}
</style>""", unsafe_allow_html=True)

# Plain-language explanations for farmers
DRIVERS = {
    'ndvi_deviation': ("Crop is less green than normal",
                       "Check leaves for yellowing and look for pests or nutrient shortage."),
    'ndvi_change_16d': ("Crop greenness dropped in the last 2 weeks",
                        "Inspect for pests, disease or waterlogging."),
    'consecutive_dry_days': ("Long dry spell",
                             "Plan irrigation now."),
    'rain_7d': ("Little rain this week", "Irrigate if the soil is dry."),
    'rain_14d': ("Little rain in the last 2 weeks", "Irrigate if the soil is dry."),
    'rain_30d': ("Little rain this month", "Review the irrigation schedule."),
    'temp_max': ("Very hot today", "Irrigate early morning or evening; mulch if possible."),
    'temp_7d': ("Hot week overall", "Keep soil moist and avoid midday field work."),
    'heat_days_7d': ("Several days above 35°C", "Keep soil moist; watch for wilting in afternoons."),
    'doy_sin': ("Crop is in a sensitive growth stage", "Check the crop more often this season."),
    'doy_cos': ("Crop is in a sensitive growth stage", "Check the crop more often this season."),
}

ACTIONS = {
    'High': ("Act now", "Irrigate within 3 days unless rain is forecast. Walk the field and check for wilting, yellow leaves and pests.", 'high', '#e74c3c'),
    'Medium': ("Watch closely", "Dig 10 cm into the soil. If it is dry, plan irrigation. Check again in 2 days.", 'medium', '#f39c12'),
    'Low': ("Normal", "No action needed. Continue your regular field care.", 'low', '#27ae60'),
}

CHECKLIST = {
    'High': ["Walk the field today and inspect at least 10 plants",
             "Irrigate within 3 days unless rain is forecast",
             "Look for pests and yellowing leaves",
             "Check the risk again tomorrow"],
    'Medium': ["Dig 10 cm into the soil and feel the moisture",
               "Check the rain forecast for the next 48 hours",
               "Check the risk again in 2 days"],
    'Low': ["No action needed", "Continue normal field care"],
}


@st.cache_resource
def load_model():
    return joblib.load('stress_model.pkl')['model']


model = load_model()


@st.cache_data
def load_scored():
    d = load_all().dropna(subset=FEATURES).copy()
    d['risk'] = model.predict_proba(d[FEATURES])[:, 1]
    return d


def level_of(risk):
    return 'High' if risk >= 0.5 else 'Medium' if risk >= 0.2 else 'Low'


def reasons_for(row):
    X = pd.DataFrame([row[FEATURES].astype(float).values], columns=FEATURES)
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    out, seen = [], set()
    for i in np.argsort(-contrib):
        if contrib[i] <= 0 or len(out) == 3:
            break
        label, tip = DRIVERS[FEATURES[i]]
        if label not in seen:
            seen.add(label)
            out.append((label, tip))
    return out


data = load_scored()
latest = data.sort_values('date').groupby('location').tail(1).copy()
latest['level'] = latest['risk'].apply(level_of)
reason_map = {r.location: reasons_for(r) for _, r in latest.iterrows()}
latest['main_reason'] = latest['location'].map(
    lambda l: reason_map[l][0][0] if reason_map[l] else 'No major driver')
latest['size'] = 18

# ---------- Sidebar: search any place ----------
with st.sidebar:
    st.header("📍 Search an area")
    st.caption("Type any town or village in Tamil Nadu.")
    query = st.text_input("Place name", placeholder="e.g. Pollachi")
    added = False
    if query and st.button("Analyse this area"):
        try:
            geo = Nominatim(user_agent="crop_stress_app").geocode(f"{query}, Tamil Nadu, India")
        except Exception:
            geo = None
        if geo is None:
            st.error("Place not found. Try a nearby town.")
        else:
            with st.spinner("Fetching satellite data (about 1-2 minutes)..."):
                try:
                    get_location(query.strip().title(), geo.latitude, geo.longitude)
                    added = True
                except Exception as e:
                    st.error(f"Satellite data unavailable right now: {e}")
    if added:
        load_scored.clear()
        st.rerun()
    st.divider()
    st.caption("Places in the built-in list are instant. Searched places use an experimental model trained on Kongu belt data.")

# ---------- Header ----------
st.title("🌾 Crop Stress Early Warning")
st.caption(f"Satellite and weather data for Tamil Nadu farm areas. Latest data: {data['date'].max():%d %b %Y}")

c1, c2, c3 = st.columns(3)
for col, (title, text) in zip((c1, c2, c3), [
    ("1. Satellite checks the crop", "Greenness (NDVI) from NASA MODIS, compared with what is normal for this place and date."),
    ("2. Weather is checked", "Rainfall, dry spells and heat from CHIRPS and ERA5 data."),
    ("3. Risk is estimated", "The model estimates the chance of crop stress in the next 10 days and explains the main reasons."),
]):
    col.markdown(f'<div class="step"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🗺️ Farm map", "🔎 Field details", "📊 Model & data"])

# ---------- Tab 1: overview ----------
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Areas monitored", len(latest))
    k2.metric("🔴 High risk", int((latest['level'] == 'High').sum()))
    k3.metric("🟠 Medium risk", int((latest['level'] == 'Medium').sum()))
    k4.metric("🟢 Low risk", int((latest['level'] == 'Low').sum()))

    fig = px.scatter_map(latest, lat='lat', lon='lon', color='risk', size='size',
                         hover_name='location', size_max=18,
                         color_continuous_scale=[(0, '#27ae60'), (0.2, '#f39c12'), (1, '#e74c3c')],
                         range_color=(0, 1), zoom=6.6, center=dict(lat=11.0, lon=77.4),
                         map_style='open-street-map', height=430)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title='Risk'))
    st.plotly_chart(fig)

    table = latest.sort_values('risk', ascending=False).assign(
        Risk=lambda d: d['risk'] * 100,
        Action=lambda d: d['level'].map(lambda l: ACTIONS[l][0]))
    table = table[['location', 'Risk', 'level', 'main_reason', 'Action']].rename(
        columns={'location': 'Area', 'level': 'Level', 'main_reason': 'Main reason'})
    st.dataframe(table, hide_index=True, column_config={
        'Risk': st.column_config.ProgressColumn('Risk', min_value=0, max_value=100, format='%d%%')})

# ---------- Tab 2: field details ----------
with tab2:
    loc = st.selectbox("Area", sorted(data['location'].unique()))
    g = data[data['location'] == loc].sort_values('date').reset_index(drop=True)
    dates = g['date'].dt.date.tolist()
    day = st.select_slider("Date", options=dates, value=dates[-1],
                           format_func=lambda d: d.strftime('%d %b %Y'))
    row = g.iloc[dates.index(day)]
    lvl = level_of(row['risk'])
    title, tip, css, color = ACTIONS[lvl]

    left, right = st.columns([1, 1.4])
    with left:
        gauge = go.Figure(go.Indicator(
            mode='gauge+number', value=row['risk'] * 100, number={'suffix': '%'},
            title={'text': 'Stress risk, next 10 days'},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#2c3e50'},
                   'steps': [{'range': [0, 20], 'color': '#c8e6c9'},
                             {'range': [20, 50], 'color': '#ffe0b2'},
                             {'range': [50, 100], 'color': '#ffcdd2'}]}))
        gauge.update_layout(height=260, margin=dict(t=60, b=0, l=20, r=20))
        st.plotly_chart(gauge)
    with right:
        st.markdown(f'''<div class="card {css}">
            <span class="badge" style="background:{color}">{lvl} risk</span>
            <h3 style="margin:.5rem 0 .2rem">{title}</h3><p>{tip}</p></div>''',
            unsafe_allow_html=True)
        st.markdown("**Why this level**")
        reasons = reasons_for(row)
        if reasons:
            for label, advice in reasons:
                st.markdown(f"- **{label}**: {advice}")
        else:
            st.markdown("- No major stress driver right now.")
        st.markdown("**This week's checklist**")
        for item in CHECKLIST[lvl]:
            st.checkbox(item, key=f'{loc}-{day}-{item}')

    w = g[g['date'] <= pd.Timestamp(day)].tail(120)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Crop greenness vs normal** (below the red line = stress)")
        f = go.Figure()
        f.add_trace(go.Scatter(x=w['date'], y=w['expected_ndvi'], name='Normal for this date',
                               line=dict(dash='dash', color='#7f8c8d')))
        f.add_trace(go.Scatter(x=w['date'], y=w['expected_ndvi'] * (1 + NDVI_THR), name='Stress line',
                               line=dict(color='#e74c3c'), fill='tonexty',
                               fillcolor='rgba(231,76,60,0.12)'))
        f.add_trace(go.Scatter(x=w['date'], y=w['ndvi'], name='Observed',
                               line=dict(color='#27ae60', width=3)))
        f.update_layout(height=320, yaxis_title='NDVI', legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(f)
    with c2:
        st.markdown("**Daily rainfall (mm)**")
        st.plotly_chart(px.bar(w, x='date', y='rainfall', height=320,
                               labels={'rainfall': 'mm', 'date': ''}))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Daily maximum temperature (°C)**")
        t = px.line(w, x='date', y='temp_max', height=320, labels={'temp_max': '°C', 'date': ''})
        t.add_hline(y=35, line_dash='dot', line_color='#e74c3c', annotation_text='Heat stress 35°C')
        st.plotly_chart(t)
    with c4:
        st.markdown("**Risk over the last 6 months**")
        h = g[g['date'] <= pd.Timestamp(day)].tail(180)
        r = px.line(h, x='date', y='risk', height=320, labels={'risk': 'Risk', 'date': ''})
        r.add_hline(y=0.2, line_dash='dot', line_color='#f39c12', annotation_text='Medium')
        r.add_hline(y=0.5, line_dash='dot', line_color='#e74c3c', annotation_text='High')
        r.update_yaxes(range=[0, 1], tickformat='.0%')
        st.plotly_chart(r)

# ---------- Tab 3: model & data ----------
with tab3:
    with open('metrics.json') as f:
        m = json.load(f)
    st.subheader("Did the early warning work?")
    st.caption(f"Tested on {m['test_rows']} days from a period the model never trained on, "
               f"including {m['onset_days']} days where the crop was not yet stressed.")
    a, b, c = st.columns(3)
    a.metric("Upcoming stress caught", f"{m['onset_recall']:.0%}",
             help="Share of future stress events that the model warned about in advance.")
    b.metric("Warnings that were right", f"{m['onset_precision']:.0%}",
             help="Share of warnings that were followed by real stress within 10 days.")
    c.metric("Simple 'stressed today' rule", "0% warned",
             help="A rule that only says 'stressed now' cannot warn before stress starts.")

    imp = pd.DataFrame({'Factor': [DRIVERS[f][0] for f in FEATURES],
                        'Importance': model.feature_importances_})
    imp = imp.groupby('Factor', as_index=False)['Importance'].sum().sort_values('Importance')
    st.plotly_chart(px.bar(imp, x='Importance', y='Factor', orientation='h', height=380,
                           title='What the model pays most attention to'))

    with st.expander("How to read this app"):
        st.markdown("""
- **Risk** is the chance of crop stress within the next 10 days: Low below 20%, Medium 20 to 50%, High above 50%.
- **Greenness (NDVI)** measures how green the crop is from satellite images. Lower than normal can mean stress.
- **Stress** is defined here by a rule: greenness well below normal and little rain over two weeks. It is a proxy, not a field inspection.
""")
    with st.expander("Limitations"):
        st.markdown(f"""
- Stress labels are rule-based proxies, not verified field observations.
- NDVI is at 250 m resolution, so results describe the neighbourhood of a point, not a single field.
- Satellite greenness is available about every 16 days and is interpolated in between.
- The model was trained on {len(m['locations'])} areas over about 1.5 years, so it is an early prototype.
- Searched places use the same model, which was not validated outside the Kongu belt.
""")
    st.caption("Data: CHIRPS (rainfall), ERA5-Land (temperature), MODIS MOD13Q1 (NDVI), via Google Earth Engine.")