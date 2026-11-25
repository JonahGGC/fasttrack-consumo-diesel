# Dashboard del Proyecto final 1: FastTrack Logistics
# Ejecutar en la terminal, desde la carpeta principal del proyecto:
#     streamlit run app_dashboard.py
# Lee los datos de dataset/ y los logos de reporte/

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="FastTrack Logistics: consumo de diésel", layout="wide")

alpha = 0.05
routes = ["Urbana", "Autopista", "Montaña"]
colors = {"Urbana": "#4C72B0", "Autopista": "#55A868", "Montaña": "#C44E52"}
markers = {"Urbana": "o", "Autopista": "s", "Montaña": "^"}
sns.set_theme(style="whitegrid")

# ---------- Datos y modelos ----------
base_dir = Path(__file__).parent
data = pd.read_csv(base_dir / "dataset" / "datos_logistica_transporte.csv", encoding="utf-8")
data["Tipo_Ruta"] = pd.Categorical(data["Tipo_Ruta"], categories=routes)

groups = [data.loc[data["Tipo_Ruta"] == r, "Consumo_Litros"] for r in routes]
f_stat, p_anova = stats.f_oneway(*groups)
f_crit = stats.f.ppf(1 - alpha, 2, len(data) - 3)
anova_table = anova_lm(smf.ols("Consumo_Litros ~ C(Tipo_Ruta)", data=data).fit())
eta2 = anova_table["sum_sq"].iloc[0] / anova_table["sum_sq"].sum()

model_1 = smf.ols("Consumo_Litros ~ Peso_Carga_Ton + Velocidad_Promedio_kmh", data=data).fit()
model_2 = smf.ols("Consumo_Litros ~ Peso_Carga_Ton + Velocidad_Promedio_kmh + C(Tipo_Ruta, Treatment('Autopista'))", data=data).fit()
model_3 = smf.ols("Consumo_Litros ~ Velocidad_Promedio_kmh + Peso_Carga_Ton * C(Tipo_Ruta, Treatment('Autopista'))", data=data).fit()
b1 = model_1.params
b2 = model_2.params
p3 = model_3.params
slope_route = {"Autopista": p3["Peso_Carga_Ton"],
               "Urbana": p3["Peso_Carga_Ton"] + p3["Peso_Carga_Ton:C(Tipo_Ruta, Treatment('Autopista'))[T.Urbana]"],
               "Montaña": p3["Peso_Carga_Ton"] + p3["Peso_Carga_Ton:C(Tipo_Ruta, Treatment('Autopista'))[T.Montaña]"]}
mean_route = data.groupby("Tipo_Ruta", observed=True)["Consumo_Litros"].mean()

# ---------- Encabezado e indicadores ----------
st.title("FastTrack Logistics: ¿qué hace que un camión gaste más diésel?")
st.caption("Proyecto final 1 · Estadística Aplicada a la Investigación · 50 viajes de la flota")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Viajes analizados", len(data))
col2.metric("Consumo promedio", f"{data['Consumo_Litros'].mean():.1f} L")
col3.metric("Ruta más cara", f"Montaña ({mean_route['Montaña']:.1f} L)",
            f"+{mean_route['Montaña'] - mean_route[['Urbana', 'Autopista']].mean():.1f} L vs las otras", delta_color="inverse")
col4.metric("Precisión del modelo (R²)", f"{model_2.rsquared:.3f}")

tab_intro, tab_desc, tab_anova, tab_corr, tab_reg, tab_pred, tab_dec = st.tabs(
    ["Presentación", "Descriptivos", "ANOVA", "Correlación", "Regresión", "Pronóstico", "Decisiones"])

# ---------- Presentación (portada) ----------
with tab_intro:
    logo_left, header_center, logo_right = st.columns([1, 4, 1])
    logo_left.image(str(base_dir / "reporte" / "logo_uach.png"), width=120)
    header_center.markdown(
        "<div style='text-align:center'>"
        "<div style='font-size:1.9rem; font-weight:700'>Universidad Autónoma de Chihuahua</div>"
        "<div style='font-size:1.4rem; font-weight:700'>Facultad de Ingeniería</div>"
        "<div style='font-size:1.1rem; font-weight:600'>Maestría en Ingeniería en Computación</div>"
        "</div>", unsafe_allow_html=True)
    logo_right.image(str(base_dir / "reporte" / "logo_fing.png"), width=120)
    st.markdown(
        "<div style='text-align:center; margin-top:2.5rem'>"
        "<div style='font-size:2.2rem; font-weight:700; line-height:1.25'>Proyecto Final 1:<br>"
        "Optimización de Rutas y Eficiencia de Combustible</div>"
        "<div style='font-size:1.3rem; font-style:italic; margin-top:1.5rem'>Estadística Aplicada a la Investigación</div>"
        "<div style='font-size:1.1rem'><b><i>Catedrático:</i></b> <i>Dr. José Saul De Lira Miramontes</i></div>"
        "</div>"
        "<div style='text-align:right; font-size:1.3rem; margin-top:2.5rem'>"
        "<b><i>Alumno:</i></b><br><b>345491</b> González Cardoza Jonahtan</div>"
        "<div style='text-align:center; font-size:1.2rem; margin-top:2rem'>25 de noviembre de 2026</div>",
        unsafe_allow_html=True)
    st.divider()
    st.subheader("El caso: FastTrack Logistics")
    st.markdown("""
FastTrack Logistics es una empresa mediana de transporte de carga terrestre que opera en el centro del país
y se especializa en entregas *Just-in-Time* para el sector industrial. En el último trimestre fiscal sus costos
operativos subieron **18 %**, impulsados casi por completo por el gasto en diésel.

**Objetivos**
- Saber si el consumo cambia según el tipo de ruta (Urbana, Autopista, Montaña).
- Medir la relación entre velocidad, peso de la carga y consumo.
- Construir un modelo que pronostique el consumo y apoyar las decisiones de la dirección.

**Cómo leer este dashboard:** cada pestaña es una fase del análisis (descriptivos, ANOVA, correlación y regresión),
seguida de un simulador de consumo y las respuestas a las preguntas de negocio.
""")

# ---------- Descriptivos ----------
with tab_desc:
    selected = st.multiselect("Rutas a mostrar", routes, default=routes)
    subset = data[data["Tipo_Ruta"].isin(selected)]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Consumo por ruta")
        table = subset.groupby("Tipo_Ruta", observed=True)["Consumo_Litros"].agg(["count", "mean", "std", "min", "max"])
        table.columns = ["Viajes", "Media (L)", "Desv. est. (L)", "Mínimo", "Máximo"]
        st.dataframe(table.round(2), use_container_width=True)
        st.subheader("Promedios por ruta")
        means = subset.groupby("Tipo_Ruta", observed=True)[["Peso_Carga_Ton", "Velocidad_Promedio_kmh", "Consumo_Litros"]].mean()
        means.columns = ["Peso (t)", "Velocidad (km/h)", "Consumo (L)"]
        st.dataframe(means.round(2), use_container_width=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=subset, x="Tipo_Ruta", y="Consumo_Litros", order=selected, hue="Tipo_Ruta", hue_order=selected, palette=colors, legend=False, width=0.5, showfliers=False, ax=ax)
        sns.stripplot(data=subset, x="Tipo_Ruta", y="Consumo_Litros", order=selected, color="black", size=4, jitter=0.15, ax=ax)
        ax.set_xlabel("Tipo de ruta")
        ax.set_ylabel("Consumo (litros)")
        st.pyplot(fig)
    with st.expander("Ver los datos completos"):
        st.dataframe(subset, use_container_width=True)

# ---------- ANOVA ----------
with tab_anova:
    st.subheader("¿El consumo medio es igual en las tres rutas?")
    st.markdown("**H₀:** μ Urbana = μ Autopista = μ Montaña  \n**H₁:** al menos una media es diferente  \nNivel de significancia α = 0.05")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F calculada", f"{f_stat:.2f}")
    c2.metric("F crítica (2, 47)", f"{f_crit:.2f}")
    c3.metric("Valor p", f"{p_anova:.1e}")
    c4.metric("η² (efecto de la ruta)", f"{eta2 * 100:.1f} %")
    st.success("Se rechaza H₀: el consumo medio no es igual en todas las rutas. Si las tres consumieran lo mismo, "
               "la probabilidad de ver diferencias así sería prácticamente cero.")
    table = anova_table.copy()
    table.index = ["Entre rutas", "Dentro (error)"]
    table.columns = ["gl", "SC", "CM", "F", "valor p"]
    st.dataframe(table[["gl", "SC", "CM", "F", "valor p"]].style.format(
        {"gl": "{:.0f}", "SC": "{:.2f}", "CM": "{:.2f}",
         "F": lambda v: "" if pd.isna(v) else f"{v:.3f}", "valor p": lambda v: "" if pd.isna(v) else f"{v:.2e}"}), use_container_width=True)
    st.subheader("¿Cuáles rutas son diferentes? (Tukey HSD)")
    tukey = pairwise_tukeyhsd(data["Consumo_Litros"], data["Tipo_Ruta"].astype(str), alpha=alpha)
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.columns = ["Grupo 1", "Grupo 2", "Diferencia (L)", "p ajustado", "Límite inf.", "Límite sup.", "¿Diferentes?"]
    tukey_df["¿Diferentes?"] = tukey_df["¿Diferentes?"].map({True: "Sí", False: "No"})
    st.dataframe(tukey_df, use_container_width=True, hide_index=True)
    st.info("Montaña gasta unos 33 L más que Urbana y Autopista. Urbana y Autopista no son diferentes entre sí.")

# ---------- Correlación ----------
with tab_corr:
    labels = {"Peso_Carga_Ton": "Peso (t)", "Velocidad_Promedio_kmh": "Velocidad (km/h)", "Consumo_Litros": "Consumo (L)"}
    corr = data[list(labels)].corr().rename(index=labels, columns=labels)
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Matriz de Pearson")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        st.markdown(f"La variable más relacionada con el consumo es el **peso** (r = {corr.iloc[0, 2]:.3f}, fuerte y positiva). "
                    f"La velocidad casi no se relaciona (r = {corr.iloc[1, 2]:.3f}, no significativa).")
    with right:
        x_var = st.radio("Variable en el eje X", ["Peso_Carga_Ton", "Velocidad_Promedio_kmh"], format_func=lambda v: labels[v], horizontal=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for r in routes:
            sub = data[data["Tipo_Ruta"] == r]
            ax.scatter(sub[x_var], sub["Consumo_Litros"], color=colors[r], marker=markers[r], edgecolor="black", s=50, label=r)
        ax.set_xlabel(labels[x_var])
        ax.set_ylabel("Consumo (litros)")
        ax.legend()
        st.pyplot(fig)
        st.caption("La velocidad depende de la ruta: autopista 80–105 km/h, montaña 42–60 km/h, ciudad 20–43 km/h.")

# ---------- Regresión ----------
with tab_reg:
    st.subheader("Modelos de regresión lineal múltiple")
    st.markdown(f"**Modelo 1:** Consumo = {b1['Intercept']:.2f} + {b1['Peso_Carga_Ton']:.3f}·Peso {b1['Velocidad_Promedio_kmh']:+.3f}·Velocidad")
    st.markdown(f"**Modelo 2:** Consumo = {b2['Intercept']:.2f} + {b2['Peso_Carga_Ton']:.3f}·Peso + {b2['Velocidad_Promedio_kmh']:.3f}·Velocidad "
                f"+ {b2.iloc[1]:.2f}·Urbana + {b2.iloc[2]:.2f}·Montaña  (Autopista es la referencia)")
    rows = []
    for name, model in [("Modelo 1: Peso + Velocidad", model_1), ("Modelo 2: + Ruta", model_2), ("Modelo 3: + Peso × Ruta", model_3)]:
        rows.append({"Modelo": name, "R²": model.rsquared, "R² ajustado": model.rsquared_adj,
                     "RMSE (L)": np.sqrt(model.mse_resid * model.df_resid / model.nobs),
                     "MAE (L)": np.mean(np.abs(model.resid)), "AIC": model.aic})
    st.dataframe(pd.DataFrame(rows).set_index("Modelo").round(3), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Coeficientes del Modelo 2**")
        coef = pd.DataFrame({"Coeficiente": model_2.params, "Valor p": model_2.pvalues})
        coef.index = ["Intercepto", "Ruta Urbana", "Ruta Montaña", "Peso (por t)", "Velocidad (por km/h)"]
        coef["¿Significativo?"] = np.where(coef["Valor p"] < alpha, "Sí", "No")
        st.dataframe(coef.style.format({"Coeficiente": "{:.4f}", "Valor p": "{:.2e}"}), use_container_width=True)
    with right:
        model_choice = st.selectbox("Real vs predicho", ["Modelo 1", "Modelo 2"], index=1)
        model = model_1 if model_choice == "Modelo 1" else model_2
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for r in routes:
            mask = data["Tipo_Ruta"] == r
            ax.scatter(data.loc[mask, "Consumo_Litros"], model.fittedvalues[mask], color=colors[r], marker=markers[r], edgecolor="black", s=45, label=r)
        ax.plot([25, 95], [25, 95], color="black")
        ax.set_xlabel("Consumo real (L)")
        ax.set_ylabel("Consumo predicho (L)")
        ax.set_title(f"{model_choice}: R² = {model.rsquared:.3f}")
        ax.legend()
        st.pyplot(fig)
    st.warning("En el Modelo 1 la velocidad sale negativa: parece que correr ahorra diésel. Es un engaño: los camiones rápidos son los de "
               "autopista, que es la ruta que menos gasta. Al agregar la ruta (Modelo 2), la velocidad sale positiva y no significativa.")

# ---------- Pronóstico ----------
with tab_pred:
    st.subheader("Simulador de consumo")
    c1, c2, c3 = st.columns(3)
    weight = c1.slider("Peso de la carga (t)", 4.0, 28.0, 15.0, 0.5)
    speed = c2.slider("Velocidad promedio (km/h)", 20, 105, 80, 1)
    route = c3.selectbox("Tipo de ruta", routes, index=1)
    new_truck = pd.DataFrame({"Peso_Carga_Ton": [weight], "Velocidad_Promedio_kmh": [speed], "Tipo_Ruta": [route]})
    pred_1 = model_1.get_prediction(new_truck).summary_frame(alpha=alpha).iloc[0]
    pred_2 = model_2.get_prediction(new_truck).summary_frame(alpha=alpha).iloc[0]
    m1, m2 = st.columns(2)
    m1.metric("Modelo 1 (Peso + Velocidad)", f"{pred_1['mean']:.1f} L")
    m1.caption(f"Intervalo de predicción 95 %: {pred_1['obs_ci_lower']:.1f} a {pred_1['obs_ci_upper']:.1f} L")
    m2.metric(f"Modelo 2 (con ruta {route})", f"{pred_2['mean']:.1f} L")
    m2.caption(f"Intervalo de predicción 95 %: {pred_2['obs_ci_lower']:.1f} a {pred_2['obs_ci_upper']:.1f} L")
    sub = data[data["Tipo_Ruta"] == route]
    low, high = sub["Velocidad_Promedio_kmh"].min(), sub["Velocidad_Promedio_kmh"].max()
    if speed < low - 1 or speed > high + 1:
        st.warning(f"Cuidado: en ruta {route} los camiones observados van de {low} a {high} km/h; esta predicción es una extrapolación.")
    st.markdown(f"**Caso del proyecto (15 t a 80 km/h):** Modelo 1 = "
                f"{model_1.predict(pd.DataFrame({'Peso_Carga_Ton': [15], 'Velocidad_Promedio_kmh': [80]}))[0]:.2f} L; "
                f"Modelo 2 en autopista = "
                f"{model_2.predict(pd.DataFrame({'Peso_Carga_Ton': [15], 'Velocidad_Promedio_kmh': [80], 'Tipo_Ruta': ['Autopista']}))[0]:.2f} L.")

# ---------- Decisiones ----------
with tab_dec:
    st.subheader("Respuestas para la dirección")
    st.markdown(f"""
**1. ¿Evitar ciertas rutas para cargas pesadas?** Sí, la **montaña**. Cada tonelada extra cuesta
{slope_route['Montaña']:.2f} L en montaña, {slope_route['Urbana']:.2f} L en ciudad y {slope_route['Autopista']:.2f} L en autopista.
Las cargas pesadas conviene mandarlas por autopista cuando se pueda, o dividirlas.

**2. ¿Dónde se gasta más y es significativo?** En **montaña**: {mean_route['Montaña']:.1f} L promedio contra
{mean_route['Urbana']:.1f} L (urbana) y {mean_route['Autopista']:.1f} L (autopista). El ANOVA da F = {f_stat:.2f} con p = {p_anova:.1e},
así que la diferencia sí es significativa. Urbana y autopista no son diferentes entre sí.

**3. ¿Controlar la velocidad o el peso?** El **peso**. Bajar 5 t ahorra {5 * b2['Peso_Carga_Ton']:.1f} L por viaje;
bajar 10 km/h solo {10 * b2['Velocidad_Promedio_kmh']:.1f} L, y el efecto de la velocidad no es significativo al tomar en cuenta la ruta.

**4. Pronóstico para 15 t a 80 km/h:** {model_1.predict(pd.DataFrame({'Peso_Carga_Ton': [15], 'Velocidad_Promedio_kmh': [80]}))[0]:.1f} L con el Modelo 1,
o {model_2.predict(pd.DataFrame({'Peso_Carga_Ton': [15], 'Velocidad_Promedio_kmh': [80], 'Tipo_Ruta': ['Autopista']}))[0]:.1f} L con el Modelo 2 si el viaje es por autopista.
""")
