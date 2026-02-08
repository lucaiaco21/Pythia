# -*- coding: utf-8 -*-
""" 
☕ Cafe Sales Prediction Dashboard - Multi-Scenario Forecasting
TFM UCM - Page 4 
Updated: 08/02/2026
""" 

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np
import re
import io

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Cafe Sales Prediction Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .stApp { background-color: #FAFAFA; }
    .main-header {
        background: linear-gradient(135deg, #6B3410 0%, #8B4513 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { color: white !important; font-size: 2.5rem !important; font-weight: 700 !important; margin: 0 !important; }
    .header-subtitle { color: #E0E0E0 !important; font-size: 1.1rem !important; margin-top: 0.5rem !important; }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #8B4513;
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #2C3E50; }
    .metric-label { font-size: 0.9rem; color: #7F8C8D; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ==================== PIPELINE CONSTANTS & MAPPING ====================

# Static Weather File (Updated to 'data/' folder)
AVG_WEATHER_PATH = "data/2026-01-31 Average Weather 2022 to 2026.xlsx"

# Full Product Category Mapping
PRODUCT_TO_CATEGORY = {
    '* Leche de Almendra': 'SUPLEMENTOS',
    '* Leche de Soja*': 'SUPLEMENTOS',
    '* Leche de Soja* -': 'SUPLEMENTOS',
    '*Leche de Avena*': 'SUPLEMENTOS',
    '*Leche de Avena* -': 'SUPLEMENTOS',
    '.AQUARIUS LIMON': 'BEBIDAS_SIN_ALCOHOL',
    '.AQUARIUS NARANJA': 'BEBIDAS_SIN_ALCOHOL',
    '.NESTEA': 'BEBIDAS_SIN_ALCOHOL',
    '1/3 SIN ALCOHOL': 'BEBIDAS_SIN_ALCOHOL',
    'ABSOLUT': 'BEBIDAS_CON_ALCOHOL',
    'AGUA CON GAS': 'BEBIDAS_SIN_ALCOHOL',
    'AGUA CON GAS.': 'BEBIDAS_SIN_ALCOHOL',
    'AGUA DE COCO': 'BEBIDAS_SIN_ALCOHOL',
    'AGUA MINERAL': 'BEBIDAS_SIN_ALCOHOL',
    'AGUA MINERAL BOTELL': 'BEBIDAS_SIN_ALCOHOL',
    'ALMENDRA': 'SUPLEMENTOS',
    'AMERICANO': 'CAFES_CLASICOS',
    'AMERICANO DOBLE': 'CAFES_CLASICOS',
    'ANTIOXIDANTE': 'FRAPPES_Y_SMOOTHIES',
    'AQUARIUS LIMON.': 'BEBIDAS_SIN_ALCOHOL',
    'AVENA': 'SUPLEMENTOS',
    'BALLANTINES': 'BEBIDAS_CON_ALCOHOL',
    'BAYLES ORGULLO': 'BEBIDAS_CON_ALCOHOL',
    'BEEFEATER': 'BEBIDAS_CON_ALCOHOL',
    'BERLINA CHOCOLATE': 'DULCES_Y_REPOSTERIA',
    'BERLINA CHOCOLATE -': 'DULCES_Y_REPOSTERIA',
    'BIZCOCHO LIMON': 'DULCES_Y_REPOSTERIA',
    'BIZCOCHO LIMÓN': 'DULCES_Y_REPOSTERIA',
    'BIZCOCHO MANZANA': 'DULCES_Y_REPOSTERIA',
    'BOC CASTELLANO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC JAMON SERRANO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC JAMON Y QUESO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC JAMON YORK E QU': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC JAMON YORK QUES': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC JAMÓN COCIDO Y': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC QUESO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC QUESO MANCHEGO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC VEGANO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC. GRANJERO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC. ITALIANO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOC. NORUEGO': 'DESAYUNOS_TOSTADAS_PANES',
    'BOCADILHOS TOSTAS V': 'DESAYUNOS_TOSTADAS_PANES',
    'BOMBAY': 'BEBIDAS_CON_ALCOHOL',
    'BOMBOM': 'CAFES_CON_PERSONALIDAD',
    'BROWNIE': 'DULCES_Y_REPOSTERIA',
    'C ZERO ZERO': 'BEBIDAS_SIN_ALCOHOL',
    'C. CAPPUCCINO': 'CAFES_CLASICOS',
    'C. LECHE': 'CAFES_CLASICOS',
    'C. LECHE DOBLE': 'CAFES_CLASICOS',
    'CAFE AMERICANO': 'CAFES_CLASICOS',
    'CAFE AMERICANO DOBL': 'CAFES_CLASICOS',
    'CAFE BOMBON': 'CAFES_CON_PERSONALIDAD',
    'CAFE CAPPUCCINO': 'CAFES_CLASICOS',
    'CAFE CAPPUCCINO DOB': 'CAFES_CLASICOS',
    'CAFE CARAJILLO': 'CAFE_CON_ALCOHOL',
    'CAFE CARAMELO': 'CAFES_CON_PERSONALIDAD',
    'CAFE CARAMELO BIG': 'CAFES_CON_PERSONALIDAD',
    'CAFE CON LECHE': 'CAFES_CLASICOS',
    'CAFE CON LECHE DOBL': 'CAFES_CLASICOS',
    'CAFE CON LECHE RENT': 'CAFES_CLASICOS',
    'CAFE CORTADO': 'CAFES_CLASICOS',
    'CAFE EL PATIO VERTI': 'CAFES_CON_PERSONALIDAD',
    'CAFE ESPRESSO': 'CAFES_CLASICOS',
    'CAFE ICED BAILEYS': 'CAFE_CON_ALCOHOL',
    'CAFE IRLANDES': 'CAFE_CON_ALCOHOL',
    'CAFE LATE CAPPUCCIN': 'CAFES_CLASICOS',
    'CAFE LATTE BIG': 'CAFES_CLASICOS',
    'CAFE LATTE NORMAL': 'CAFES_CLASICOS',
    'CAFE LECHE LECHE': 'CAFES_CLASICOS',
    'CAFE MOKKA BLANCO': 'CAFES_CON_PERSONALIDAD',
    'CAFE MOKKA BLANCO B': 'CAFES_CON_PERSONALIDAD',
    'CAFE MOKKA NEGRO': 'CAFES_CON_PERSONALIDAD',
    'CAFE MOKKA NEGRO BI': 'CAFES_CON_PERSONALIDAD',
    'CAFE RUSO': 'CAFES_CLASICOS',
    'CAFE SOLO': 'CAFES_CLASICOS',
    'CAFE VAINILLA': 'CAFES_CON_PERSONALIDAD',
    'CAFE VAINILLA BIG': 'CAFES_CON_PERSONALIDAD',
    'CAFE VIENES': 'CAFES_CON_PERSONALIDAD',
    'CAFÉ CUBANO': 'CAFES_CON_PERSONALIDAD',
    'CAMBIO CROISSANT': 'DULCES_Y_REPOSTERIA',
    'CAMBIO TOSTADA MIXT': 'DESAYUNOS_TOSTADAS_PANES',
    'CANELA ROLL': 'DULCES_Y_REPOSTERIA',
    'CAPPUCCINO': 'CAFES_CLASICOS',
    'CAPPUCCINO DOBLE': 'CAFES_CLASICOS',
    'CAPPUCCINO RENTALIS': 'CAFES_CLASICOS',
    'CAPRESE': 'ENSALADAS',
    'CAPUCCINO': 'CAFES_CLASICOS',
    'CAPUCCINO DOBLE': 'CAFES_CLASICOS',
    'CARAJILLO': 'CAFE_CON_ALCOHOL',
    'CAÑA': 'CERVEZAS',
    'CAÑA 400 ML': 'CERVEZAS',
    'CAÑA MAHOU': 'CERVEZAS',
    'CERVEZA LATA': 'CERVEZAS',
    'CERVEZA LATA.': 'CERVEZAS',
    'CERVEZA SIN ALCOOL': 'CERVEZAS',
    'CESAR': 'ENSALADAS',
    'CHEESECAKE CON ARAN': 'DULCES_Y_REPOSTERIA',
    'CHOCOL CHIP CHEESE': 'DULCES_Y_REPOSTERIA',
    'CHOCOLATE': 'CAFES_CON_PERSONALIDAD',
    'CHOCOLATE BATIDO': 'CAFES_CON_PERSONALIDAD',
    'CHOCOLATE TAZA': 'CAFES_CON_PERSONALIDAD',
    'CIBELES': 'ENSALADAS',
    'COCA': 'BEBIDAS_SIN_ALCOHOL',
    'COCA COLA LIGHT ORG': 'BEBIDAS_SIN_ALCOHOL',
    'COCA COLA ZERO ORGU': 'BEBIDAS_SIN_ALCOHOL',
    'COCA COLA.': 'BEBIDAS_SIN_ALCOHOL',
    'COCA LATA': 'BEBIDAS_SIN_ALCOHOL',
    'COCA LIGHT': 'BEBIDAS_SIN_ALCOHOL',
    'COCA ZERO': 'BEBIDAS_SIN_ALCOHOL',
    'COCA ZERO LATA': 'BEBIDAS_SIN_ALCOHOL',
    'COLA CAO RENTALIS': 'CAFES_CLASICOS',
    'COLISEO PINCHO': 'PINCHOS_Y_RACIONES',
    'COOKIE CHOCO': 'DULCES_Y_REPOSTERIA',
    'COPA BLANCO RUEDA': 'BEBIDAS_CON_ALCOHOL',
    'COPA TINTO CRIANZA': 'BEBIDAS_CON_ALCOHOL',
    'CORTADO': 'CAFES_CLASICOS',
    'CORTADO DOBLE': 'CAFES_CLASICOS',
    'CREMA BOLETUS': 'CREMA',
    'CREMA CALABAZA ZANA': 'CREMA',
    'CREMA DE CALABACIN': 'CREMA',
    'CREMA DE VERDURAS': 'CREMA',
    'CROISSANT': 'DULCES_Y_REPOSTERIA',
    'CROISSANT ARTESANO': 'DULCES_Y_REPOSTERIA',
    'CROISSANT CHOCOLATE': 'DULCES_Y_REPOSTERIA',
    'CROISSANT JAMÓN Y Q': 'DULCES_Y_REPOSTERIA',
    'CROISSANT MIXTO': 'DULCES_Y_REPOSTERIA',
    'CROISSANT NUTELLA': 'DULCES_Y_REPOSTERIA',
    'DETOX': 'FRAPPES_Y_SMOOTHIES',
    'DOBLE ESPRESSO': 'CAFES_CLASICOS',
    'DOBLE EXPRESO': 'CAFES_CLASICOS',
    'DOBLE MAHOU': 'CERVEZAS',
    'DONUT CHOCOLATE': 'DULCES_Y_REPOSTERIA',
    'DONUTS': 'DULCES_Y_REPOSTERIA',
    'DONUTS CHOCOLATE': 'DULCES_Y_REPOSTERIA',
    'DOUGALL´S 942': 'CERVEZAS',
    'DOUGALL´S IPA4': 'CERVEZAS',
    'ENERGETICO': 'FRAPPES_Y_SMOOTHIES',
    'EXPRESO': 'CAFES_CLASICOS',
    'EXPRESO -': 'CAFES_CLASICOS',
    'Extra Café Grande': 'CAFES_CLASICOS',
    'FANTA LIMON.': 'BEBIDAS_SIN_ALCOHOL',
    'FANTA NARANJA': 'BEBIDAS_SIN_ALCOHOL',
    'FANTA NARANJA LATA': 'BEBIDAS_SIN_ALCOHOL',
    'FANTA NARANJA.': 'BEBIDAS_SIN_ALCOHOL',
    'FLAT WHITE': 'CAFES_CLASICOS',
    'FRAPPE CAFE LATTE B': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE CAFE LATTE N': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE MATCHA': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE MOCCA CHIP': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE MOCCA CHIP B': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE TE CHAI LATT': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE VAINILLA EXP': 'FRAPPES_Y_SMOOTHIES',
    'FRAPPE WHITE CHOCOL': 'FRAPPES_Y_SMOOTHIES',
    'FUZE TEA': 'BEBIDAS_SIN_ALCOHOL',
    'GALLETA AVENA Y CHO': 'DULCES_Y_REPOSTERIA',
    'GALLETA AVENA Y FRU': 'DULCES_Y_REPOSTERIA',
    'GINEBRAS': 'BEBIDAS_CON_ALCOHOL',
    'HENDRICKS': 'BEBIDAS_CON_ALCOHOL',
    'HOP FICTION': 'CERVEZAS',
    'INF JENGIBRE LIMÓN': 'INFUSIONES_Y_TES',
    'INF. CANELA': 'INFUSIONES_Y_TES',
    'INF. FRUTOS ROJOS': 'INFUSIONES_Y_TES',
    'INF. MANZANILLA': 'INFUSIONES_Y_TES',
    'INF. POLEO MENTA': 'INFUSIONES_Y_TES',
    'INF. TILA': 'INFUSIONES_Y_TES',
    'INFUSIONES RENTALIS': 'INFUSIONES_Y_TES',
    'KEY LIME CHEESE CAK': 'DULCES_Y_REPOSTERIA',
    'LA FUERZA VERDE': 'FRAPPES_Y_SMOOTHIES',
    'LASAGNA': 'PASTAS_Y_ARROCES',
    'LATA C': 'BEBIDAS_SIN_ALCOHOL',
    'LECHE': 'SUPLEMENTOS',
    'LECHE LECHE': 'SUPLEMENTOS',
    'LICORES': 'BEBIDAS_CON_ALCOHOL',
    'LIMÓN MERENGUE': 'DULCES_Y_REPOSTERIA',
    'LLIPA': 'CERVEZAS',
    'MACEDONIA': 'DULCES_Y_REPOSTERIA',
    'MARTINI': 'BEBIDAS_CON_ALCOHOL',
    'MARTINI ROJO (ROSSO': 'BEBIDAS_CON_ALCOHOL',
    'MASCARPONE Y FRUTOS': 'DULCES_Y_REPOSTERIA',
    'MENU RENTALIS': 'NO_MODELAR',
    'MILHOJAS': 'DULCES_Y_REPOSTERIA',
    'MINI CERVEZA': 'CERVEZAS',
    'MINI SANGRIA': 'BEBIDAS_CON_ALCOHOL',
    'MINI TINTO DE V': 'BEBIDAS_CON_ALCOHOL',
    'MUFFIN CHOCOLATE': 'DULCES_Y_REPOSTERIA',
    'MUFFIN MANZANA Y CA': 'DULCES_Y_REPOSTERIA',
    'NESTEA.': 'BEBIDAS_SIN_ALCOHOL',
    'NÓRDICA': 'ENSALADAS',
    'PAN SIN GLUTEN': 'DESAYUNOS_TOSTADAS_PANES',
    'PAN SIN GLUTEN -': 'DESAYUNOS_TOSTADAS_PANES',
    'PENNE A LA ARRABIAT': 'PASTAS_Y_ARROCES',
    'PILSNER URQUELL': 'CERVEZAS',
    'PLAZA MAYOR PINCHO': 'PINCHOS_Y_RACIONES',
    'PROTECTOR': 'FRAPPES_Y_SMOOTHIES',
    'PROTECTOR açai': 'FRAPPES_Y_SMOOTHIES',
    'PUERTO DE INDIAS': 'BEBIDAS_CON_ALCOHOL',
    'PULGUITA DE JAMON': 'DESAYUNOS_TOSTADAS_PANES',
    'PULGUITA DE MANCHEG': 'DESAYUNOS_TOSTADAS_PANES',
    'PULGUITA DE SERRANO': 'DESAYUNOS_TOSTADAS_PANES',
    'PULGUITA JAMÖN Y QU': 'DESAYUNOS_TOSTADAS_PANES',
    'RACION JAMON': 'PINCHOS_Y_RACIONES',
    'RACION QUESO MANCHE': 'PINCHOS_Y_RACIONES',
    'RED BULL': 'BEBIDAS_SIN_ALCOHOL',
    'RED BULL.': 'BEBIDAS_SIN_ALCOHOL',
    'RED LABEL': 'BEBIDAS_CON_ALCOHOL',
    'REGENERADOR': 'FRAPPES_Y_SMOOTHIES',
    'RETIRO PINCHO': 'PINCHOS_Y_RACIONES',
    'REVITALIZANTE': 'FRAPPES_Y_SMOOTHIES',
    'REVITALIZANTE pitay': 'FRAPPES_Y_SMOOTHIES',
    'RICCOTA Y PISTA CCH': 'DULCES_Y_REPOSTERIA',
    'RICOTA Y CHOCOLATE': 'DULCES_Y_REPOSTERIA',
    'RON BARCELO': 'BEBIDAS_CON_ALCOHOL',
    'RON BRUGAL': 'BEBIDAS_CON_ALCOHOL',
    'S/GLUT CHOCOLATE': 'SUPLEMENTOS',
    'S/GLUT COCOLA': 'SUPLEMENTOS',
    'S/GLUT LIMON': 'SUPLEMENTOS',
    'SANFRUTOS LARGER': 'CERVEZAS',
    'SANFRUTOS ORO NEGRO': 'CERVEZAS',
    'SANGRIA': 'BEBIDAS_CON_ALCOHOL',
    'SANGRIA 400': 'BEBIDAS_CON_ALCOHOL',
    'SOJA': 'SUPLEMENTOS',
    'SOKA': 'SUPLEMENTOS',
    'SOLO': 'CAFES_CLASICOS',
    'SPAGHETTI CARBO NAR': 'PASTAS_Y_ARROCES',
    'SPRITE ORGULLO': 'BEBIDAS_SIN_ALCOHOL',
    'SPRITE.': 'BEBIDAS_SIN_ALCOHOL',
    'STRUDEL': 'DULCES_Y_REPOSTERIA',
    'SUPL LECHE': 'SUPLEMENTOS',
    'SUPL LECHE.': 'SUPLEMENTOS',
    'SUPL QUESO MANCHEGO': 'SUPLEMENTOS',
    'SUPLEM LECHE': 'SUPLEMENTOS',
    'SUPLEMENTO NATA-': 'SUPLEMENTOS',
    'SUPLEMENTO YOGUR': 'SUPLEMENTOS',
    'Suplem Sin Lactosa': 'SUPLEMENTOS',
    'T. ALBARICAQUE Y RO': 'DESAYUNOS_TOSTADAS_PANES',
    'T. IBÉRICO BRIE': 'DESAYUNOS_TOSTADAS_PANES',
    'T. JAMÓN Y AGUACATE': 'DESAYUNOS_TOSTADAS_PANES',
    'T. SALMÓN Y AGUACAT': 'DESAYUNOS_TOSTADAS_PANES',
    'TAKE AWAY': 'TAKE_AWAY',
    'TAKE AWAY COMIDA': 'TAKE_AWAY',
    'TANQUERAY': 'BEBIDAS_CON_ALCOHOL',
    'TARTA 2': 'DULCES_Y_REPOSTERIA',
    'TARTA 3': 'DULCES_Y_REPOSTERIA',
    'TARTA ALBARICOQUE': 'DULCES_Y_REPOSTERIA',
    'TARTA DE CALABAZA': 'DULCES_Y_REPOSTERIA',
    'TARTA DE MANZANA': 'DULCES_Y_REPOSTERIA',
    'TARTA DE ZANAHORIA': 'DULCES_Y_REPOSTERIA',
    'TARTA LIMON Y MEREN': 'DULCES_Y_REPOSTERIA',
    'TARTA QUESO Y FRESA': 'DULCES_Y_REPOSTERIA',
    'TARTA RED VELVET': 'DULCES_Y_REPOSTERIA',
    'TE AMERICANO': 'INFUSIONES_Y_TES',
    'TE AZUL': 'INFUSIONES_Y_TES',
    'TE CHAI LATTE CALIE': 'INFUSIONES_Y_TES',
    'TE MATCHA': 'INFUSIONES_Y_TES',
    'TERCIO LA VIRGEN': 'CERVEZAS',
    'TERCIO MAHOU': 'CERVEZAS',
    'TINTO V 400': 'BEBIDAS_CON_ALCOHOL',
    'TIRAMISSU SIN GLUTE': 'DULCES_Y_REPOSTERIA',
    'TONICA': 'BEBIDAS_SIN_ALCOHOL',
    'TONICA.': 'BEBIDAS_SIN_ALCOHOL',
    'TOST AGUACATE HUEVO': 'DESAYUNOS_TOSTADAS_PANES',
    'TOST PLAZA MAYOR': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA AGUACATE': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA CREMA DE QU': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA CREMA RENTA': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA JAMON IBERI': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA JAMON SERRA': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA MANT E MARM': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA MIXTA': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA MM RENTALIS': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA NUTELA': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA TOMATE': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA TOMATE -': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA TOMATE RENT': 'DESAYUNOS_TOSTADAS_PANES',
    'TOSTADA TRUFA': 'DESAYUNOS_TOSTADAS_PANES',
    'TÉ EARL GREY': 'INFUSIONES_Y_TES',
    'TÉ ENGLISH BREAKFAS': 'INFUSIONES_Y_TES',
    'TÉ GREEN': 'INFUSIONES_Y_TES',
    'TÉ MOORISH': 'INFUSIONES_Y_TES',
    'TÉ NEGRO CHAI': 'INFUSIONES_Y_TES',
    'TÉ PAI MU TAN': 'INFUSIONES_Y_TES',
    'TÉ PU ERH': 'INFUSIONES_Y_TES',
    'TÉ ROOIBOS': 'INFUSIONES_Y_TES',
    'TÉ VERDE CON GENGIB': 'INFUSIONES_Y_TES',
    'TÉS VILLA PIVA': 'INFUSIONES_Y_TES',
    'VARIOS': 'NO_MODELAR',
    'VARIOS CAFES ESP': 'CAFES_CLASICOS',
    'VARIOS DESAYUNOS': 'DESAYUNOS_TOSTADAS_PANES',
    'VARIOS DRINKS': 'BEBIDAS_SIN_ALCOHOL',
    'VARIOS FRAPPES': 'FRAPPES_Y_SMOOTHIES',
    'VARIOS NON CAFES': 'CAFES_CLASICOS',
    'VARIOS PASTA': 'PASTAS_Y_ARROCES',
    'VARIOS REFRESCOS': 'BEBIDAS_SIN_ALCOHOL',
    'VARIOS SMOOTHIES': 'FRAPPES_Y_SMOOTHIES',
    'VEGANA': 'ENSALADAS',
    'VEGANO PINCHO': 'PINCHOS_Y_RACIONES',
    'VERTICAL PINCHO': 'PINCHOS_Y_RACIONES',
    'VIENES': 'CAFES_CON_PERSONALIDAD',
    'VODKA RON OTROS': 'BEBIDAS_CON_ALCOHOL',
    'Vaso de Agua': 'BEBIDAS_SIN_ALCOHOL',
    'WHISKY': 'BEBIDAS_CON_ALCOHOL',
    'YOGUR CASERO': 'DESAYUNOS_TOSTADAS_PANES',
    'Z MELOCOTON RENTALI': 'ZUMOS',
    'Z NARANJA RENTALIS': 'ZUMOS',
    'ZUMO MELOCOTON': 'ZUMOS',
    'ZUMO MOSTO': 'ZUMOS',
    'ZUMO NARANJA BIG': 'ZUMOS',
    'ZUMO NARANJA CASERO': 'ZUMOS',
    'ZUMO NARANJA CASERO -': 'ZUMOS',
    'ZUMO PIÑA': 'ZUMOS',
    'ZUMO VILLA PIVA': 'ZUMOS',
    'adicional tomate': 'SUPLEMENTOS',
    'cafe big caixa': 'CAFES_CLASICOS',
    'estra tomate': 'SUPLEMENTOS',
    'extra frutas': 'SUPLEMENTOS',
    'extra tomate': 'SUPLEMENTOS',
    'mostaza, lechuga y': 'SUPLEMENTOS',
    'suplem ALMENDRA': 'SUPLEMENTOS',
    'suplem ALMENDRA -': 'SUPLEMENTOS',
    'suplem AVENA': 'SUPLEMENTOS',
    'suplem SOJA': 'SUPLEMENTOS',
    'suplem SOJA -': 'SUPLEMENTOS',
}

# ==================== HELPER FUNCTIONS ====================

DASH_LINE = re.compile(r"^-{10,}\s*$")
EQUAL_LINE = re.compile(r"^={10,}\s*$")
ITEM_LINE = re.compile(r"^\s*(\d+,\d{2})\s+(.+?)\s+(\d+,\d{2})\s*$")

def _clean_line(s: str) -> str:
    s = s.replace("€", "").replace("", "")
    s = re.sub(r"\s{2,}", " ", s)
    return s.rstrip()

def _parse_invoices(content: str):
    """
    Parses the raw text content of the invoice file.
    Adapted to accept string content directly.
    """
    raw_invoices = re.split(r"FACTURA(?: SIMPLIFICADA(?:\s+ABONO)?)?", content)
    invoices = []

    for block in raw_invoices[1:]:
        lines = [_clean_line(ln) for ln in block.splitlines()]
        invoice = {"items": []}

        m_num = re.search(r"N.? Op\.:\s+([A-Z0-9\-]+)", block)
        if m_num:
            invoice["invoice_number"] = m_num.group(1).strip()

        m_related = re.search(r">>\s*Doc\. Relacionado:\s*(T-\d+)", block)
        if m_related:
            invoice["related_invoice"] = m_related.group(1)

        m_date = re.search(r"(\d{2}/\d{2}/\d{4})", block)
        if m_date:
            invoice["date"] = m_date.group(1)

        m_loc = re.search(r"N.? Op\.:\s+[A-Z0-9\-]+\s+([A-Za-z0-9/]+)", block)
        if m_loc:
            invoice["location"] = m_loc.group(1).strip()

        # Totals
        for ln in lines:
            m_total = re.search(r"Total\s+\(Impuestos Incl\.\)\s+(-?\d+,\d{2})", ln)
            if m_total:
                invoice["grand_total"] = float(m_total.group(1).replace(",", "."))
                break
        
        # Payment
        for ln in lines:
            m_pay = re.search(r"\b(Tarjeta|Efectivo)\b\s+(\d+,\d{2})", ln)
            if m_pay:
                invoice["payment_method"] = m_pay.group(1)
                invoice["amount_paid"] = float(m_pay.group(2).replace(",", "."))
                break

        # Items section detection
        items_start = None
        for i, ln in enumerate(lines):
            if ln.strip().startswith("Uds.") and "Producto" in ln and "Importe" in ln:
                items_start = i + 1
                if items_start < len(lines) and DASH_LINE.match(lines[items_start]):
                    items_start += 1
                break

        items_end = None
        if items_start is not None:
            for j in range(items_start, len(lines)):
                if DASH_LINE.match(lines[j]):
                    items_end = j
                    break

        # Parse item lines
        if items_start is not None and items_end is not None:
            for ln in lines[items_start:items_end]:
                if not ln.strip() or DASH_LINE.match(ln) or EQUAL_LINE.match(ln):
                    continue
                
                m_item = ITEM_LINE.match(ln)
                if not m_item:
                    # Fallback for complex lines
                    qty_m = re.search(r"^\s*(\d+,\d{2})", ln)
                    prices = re.findall(r"(\d+,\d{2})", ln)
                    if qty_m and prices:
                        qty_str = qty_m.group(1)
                        price_str = prices[-1]
                        qty = float(qty_str.replace(",", "."))
                        price = float(price_str.replace(",", "."))
                        qty_end = qty_m.end()
                        price_start = ln.rfind(price_str)
                        product = ln[qty_end:price_start].strip()
                        if product:
                            invoice["items"].append({"product": product, "quantity": qty, "price": price})
                    continue

                qty = float(m_item.group(1).replace(",", "."))
                product = m_item.group(2).strip()
                price = float(m_item.group(3).replace(",", "."))
                invoice["items"].append({"product": product, "quantity": qty, "price": price})

        invoices.append(invoice)
    return invoices

def _invoices_json_to_df(records):
    rows = []
    for inv in records:
        inv_number = inv.get("invoice_number")
        invoice_type = None
        if isinstance(inv_number, str):
            if inv_number.startswith("TD-"): invoice_type = "TD"
            elif inv_number.startswith("T-"): invoice_type = "T"
            elif inv_number.startswith("F-"): invoice_type = "F"

        base = {
            "invoice_number": inv_number,
            "invoice_type": invoice_type,
            "related_invoice": inv.get("related_invoice"),
            "date": inv.get("date"),
            "location": inv.get("location"),
            "grand_total": inv.get("grand_total"),
            "payment_method": inv.get("payment_method"),
            "amount_paid": inv.get("amount_paid"),
        }
        items = inv.get("items") or []
        if items:
            for item in items:
                row = base.copy()
                row.update({
                    "product": item.get("product"),
                    "quantity": item.get("quantity"),
                    "price": item.get("price"),
                })
                rows.append(row)
        else:
            row = base.copy()
            row.update({"product": None, "quantity": None, "price": None})
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    return df

def build_prophet_prediction_inputs(
    invoice_content: str,
    avg_weather_xlsx_path: str,
    horizon_days: int = 14,
    invoices_lag_days: int = 14,
    visitantes_baseline: int = 1934,
    visitantes_low: int = 1419,
    visitantes_high: int = 2739,
    # CHANGE: Default empty tuple so we don't drop weather cols by default
    drop_weather_cols: tuple = (), 
    drop_calendar_cols: tuple = ("day", "month"),
    verbose: bool = True,
):
    # ---- A) Parse txt -> json -> df_invoices
    input_json = _parse_invoices(invoice_content)
    df_invoices = _invoices_json_to_df(input_json)
    
    if verbose:
        st.write(f"Processed {len(input_json)} invoices.")
    
    # ---- B) Remove related invoices + TD invoices
    related_invoices = (
        df_invoices["related_invoice"]
        .dropna().astype(str).str.strip().unique().tolist()
    )
    df_invoices = df_invoices.loc[~df_invoices["invoice_number"].isin(related_invoices)].copy()
    df_invoices = df_invoices.loc[df_invoices["invoice_type"].astype(str).str.strip().ne("TD")].copy()

    # ---- C) Map categories
    df_invoices["product_category"] = df_invoices["product"].map(PRODUCT_TO_CATEGORY)

    # ---- D) Daily summary: revenue + invoice count
    invoice_level = (
        df_invoices.groupby("invoice_number", as_index=False)
        .agg(date=("date", "first"), grand_total=("grand_total", "first"))
    )
    invoice_level["date"] = pd.to_datetime(invoice_level["date"], errors="coerce")
    invoice_level["date_day"] = invoice_level["date"].dt.normalize()
    
    daily_summary = (
        invoice_level.groupby("date_day", as_index=False)
        .agg(total_revenue=("grand_total", "sum"), count_invoices=("invoice_number", "nunique"))
        .sort_values("date_day")
    )

    # ---- E) Units per category
    df_invoices["date_day"] = df_invoices["date"].dt.normalize()
    daily_units_pivot = (
        df_invoices.groupby(["date_day", "product_category"], as_index=False)["quantity"]
        .sum()
        .pivot(index="date_day", columns="product_category", values="quantity")
        .fillna(0)
    )
    daily_units_pivot.columns = [f"units_{c}" for c in daily_units_pivot.columns]
    
    daily_summary_enriched = (
        daily_summary.merge(daily_units_pivot.reset_index(), on="date_day", how="left")
        .sort_values("date_day")
    )
    unit_cols = [c for c in daily_summary_enriched.columns if c.startswith("units_")]
    daily_summary_enriched[unit_cols] = daily_summary_enriched[unit_cols].fillna(0)

    # ---- F) Build future frame
    future_df = daily_summary_enriched.copy()
    future_df["date_day"] = pd.to_datetime(future_df["date_day"], errors="coerce")
    future_df = future_df.sort_values("date_day").set_index("date_day").asfreq("D")
    
    future_df["day"] = future_df.index.day
    future_df["month"] = future_df.index.month
    future_df = future_df.reset_index()
    
    last_hist_date = future_df["date_day"].max()
    future_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    future_rows = pd.DataFrame({"date_day": future_dates})
    future_rows["day"] = future_rows["date_day"].dt.day
    future_rows["month"] = future_rows["date_day"].dt.month
    future_rows["count_invoices"] = np.nan
    future_rows["total_revenue"] = np.nan
    for c in unit_cols: future_rows[c] = np.nan
    
    future_df = pd.concat([future_df, future_rows], ignore_index=True)
    future_df = future_df.sort_values("date_day").set_index("date_day").asfreq("D")

    # ---- G) Lag feature
    lag_col = f"invoices_lag_{invoices_lag_days}"
    future_df[lag_col] = future_df["count_invoices"].shift(invoices_lag_days)
    future_df = future_df.reset_index()
    future_df = future_df.dropna(subset=[lag_col])
    
    max_hist_day = daily_summary_enriched["date_day"].max()
    future_df = future_df.loc[future_df["date_day"] > max_hist_day].copy()
    future_df = future_df.drop(columns=["count_invoices", "total_revenue"], errors="ignore")

    # ---- H) Join average weather
    try:
        avg_weather_df = pd.read_excel(avg_weather_xlsx_path)
        future_df = future_df.merge(avg_weather_df, on=["month", "day"], how="left")
        
        # Calculate derived features but KEEP originals
        if "tmax" in future_df.columns and "tmin" in future_df.columns:
            future_df["temp_range"] = future_df["tmax"] - future_df["tmin"]
        if "prec" in future_df.columns:
            future_df["is_rain"] = (future_df["prec"] > 0).astype(int)
            future_df["is_heavy_rain"] = (future_df["prec"] >= 10).astype(int)
        if "tmed" in future_df.columns:
            future_df["is_hot"] = (future_df["tmed"] >= 30).astype(int)
            future_df["is_cold"] = (future_df["tmed"] <= 6).astype(int)
    except Exception as e:
        st.error(f"Error loading weather file: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    future_df = future_df.rename(columns={"date_day": "ds"})

    # ---- I) Build three scenarios (RENAMED TO MATCH SOURCE)
    future_1_baseline_df = future_df.copy()
    future_2_lowcxf_df = future_df.copy()
    future_3_highcxf_df = future_df.copy()
    
    future_1_baseline_df["visitantes_cxf"] = visitantes_baseline
    future_2_lowcxf_df["visitantes_cxf"] = visitantes_low
    future_3_highcxf_df["visitantes_cxf"] = visitantes_high
    
    cols_to_drop = list(drop_calendar_cols) + list(drop_weather_cols)
    extra_drop = [c for c in future_1_baseline_df.columns if c.startswith("units_")] 
    
    # CHANGE: Added 'tmax', 'tmin', 'prec' to final_cols so they are passed to the model
    final_cols = [
        "ds", "invoices_lag_14", "tmed", "temp_range", 
        "is_hot", "is_cold", "visitantes_cxf",
        "tmax", "tmin", "prec", "is_rain", "is_heavy_rain"
    ]
    
    def clean_cols(df):
        # Only drop columns that are explicitly in the drop lists
        df = df.drop(columns=cols_to_drop+extra_drop, errors="ignore")
        # Filter to keep available columns that match final_cols
        available_cols = [c for c in final_cols if c in df.columns]
        return df[available_cols]

    return clean_cols(future_1_baseline_df), clean_cols(future_2_lowcxf_df), clean_cols(future_3_highcxf_df)


# ==================== MAIN UI ====================

st.markdown("""
<div class="main-header">
    <h1 class="header-title">Cafe Sales Prediction Dashboard</h1>
    <p class="header-subtitle">Multi-Scenario Forecasting | 📚 TFM UCM | ☕ Powered by Prophet ML</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Data Input")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Invoice Layout (.txt)", type=["txt"])

st.sidebar.info(f"Using static weather file:\n**{AVG_WEATHER_PATH}**")

# Load Model
model_path = None
# Search for a .joblib model in the current directory or models folder
possible_models = list(Path(".").glob("*.joblib"))
if not possible_models:
    # Try models folder
    possible_models = list(Path("models").glob("*.joblib"))

if possible_models:
    model_path = possible_models[0]

loaded_model = None
if model_path:
    try:
        loaded_model = joblib.load(model_path)
        # Handle dict format if applicable
        if isinstance(loaded_model, dict):
            for key in ['model', 'prophet', 'estimator']:
                if key in loaded_model:
                    loaded_model = loaded_model[key]
                    break
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
else:
    st.sidebar.warning("No Prophet model (.joblib) found in directory.")

# Main Execution
if uploaded_file is not None and loaded_model is not None:
    try:
        content = uploaded_file.read().decode("latin-1")
        
        with st.spinner("Processing invoices and generating scenarios..."):
            # Calling the function which now returns the named DFs
            f1_base, f2_low, f3_high = build_prophet_prediction_inputs(content, AVG_WEATHER_PATH)
            
        if f1_base.empty:
            st.error("Failed to generate input data. Check logs/files.")
        else:
            # Define Scenarios
            scenarios = {
                "Baseline": {
                    "data": f1_base,
                    "description": "Expected normal conditions",
                    "icon": "📊",
                    "color": "#3498DB"
                },
                "Low (Pessimistic)": {
                    "data": f2_low,
                    "description": "Conservative estimate",
                    "icon": "📉",
                    "color": "#E74C3C"
                },
                "High (Optimistic)": {
                    "data": f3_high,
                    "description": "Best case scenario",
                    "icon": "📈",
                    "color": "#2ECC71"
                }
            }

            st.success("Data processed successfully! Generating predictions...")

            # Iterate and Predict
            results = []
            
            # Create tabs for results
            tab_names = list(scenarios.keys())
            tabs = st.tabs(tab_names)

            for i, (name, config) in enumerate(scenarios.items()):
                with tabs[i]:
                    st.markdown(f"### {config['icon']} {name} Scenario")
                    st.markdown(f"*{config['description']}*")
                    
                    df_input = config['data']
                    
                    # PREDICT
                    try:
                        forecast = loaded_model.predict(df_input)
                        
                        # Show key metrics
                        total_sales = forecast['yhat'].sum()
                        avg_sales = forecast['yhat'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Forecasted Sales</div>
                                <div class="metric-value">{total_sales:,.2f} €</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Daily Average</div>
                                <div class="metric-value">{avg_sales:,.2f} €</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'], 
                            y=forecast['yhat'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color=config['color'], width=3)
                        ))
                        # Add intervals if available
                        if 'yhat_lower' in forecast.columns:
                            fig.add_trace(go.Scatter(
                                x=forecast['ds'], y=forecast['yhat_upper'],
                                mode='lines', marker=dict(color="#444"),
                                line=dict(width=0), showlegend=False, hoverinfo='skip'
                            ))
                            fig.add_trace(go.Scatter(
                                x=forecast['ds'], y=forecast['yhat_lower'],
                                mode='lines', marker=dict(color="#444"),
                                line=dict(width=0), fill='tonexty',
                                fillcolor='rgba(68, 68, 68, 0.1)',
                                showlegend=False, hoverinfo='skip'
                            ))

                        fig.update_layout(
                            title=f"Sales Forecast - {name}",
                            xaxis_title="Date",
                            yaxis_title="Sales (€)",
                            template="plotly_white",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View Forecast Data"):
                            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(14))
                            
                    except Exception as e:
                        st.error(f"Prediction failed for {name}: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

elif uploaded_file is None:
    st.info("👋 Please upload an Invoice Layout text file in the sidebar to begin.")

elif loaded_model is None:
    st.error("Model not loaded. Please ensure the .joblib file is in the application directory.")
