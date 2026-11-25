# Proyecto Final 1: Optimización de Rutas y Eficiencia de Combustible

Estadística Aplicada a la Investigación · Maestría en Ingeniería en Computación, Facultad de Ingeniería, UACH
Alumno: 345491 González Cardoza Jonahtan · Catedrático: Dr. José Saul De Lira Miramontes

## Estructura

```
app_dashboard.py               Dashboard en Streamlit
Proyecto_final_analisis.ipynb  Análisis completo en Python (Jupyter)
requirements.txt               Librerías necesarias
dataset/                       Datos originales y datos limpios (UTF-8)
reporte/                       Reporte en PDF, código LaTeX, figuras y logos
```

## Cómo ejecutar

Desde esta carpeta:

```
pip install -r requirements.txt
streamlit run app_dashboard.py
```

El notebook también se ejecuta desde esta carpeta; lee `dataset/datos_logistica_transporte1.csv` y vuelve a generar
`dataset/datos_logistica_transporte.csv` y las figuras de `reporte/figuras/`.

El reporte se compila con XeLaTeX (o pdfLaTeX) dentro de la carpeta `reporte/`.
