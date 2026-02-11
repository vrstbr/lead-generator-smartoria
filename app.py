import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json
import os
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO

# Környezeti változók betöltése
load_dotenv()

# Streamlit oldal beállítása
st.set_page_config(
    page_title="Smartoria Lead Generator",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, div { color: #000000 !important; }
    .stButton > button { background-color: #D32F2F !important; color: #FFFFFF !important; border: none; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #B71C1C !important; }
    .stTextInput > div > div > input { color: #000000 !important; background-color: #FFFFFF !important; border: 1px solid #000000 !important; }
    .stProgress > div > div > div { background-color: #D32F2F !important; }
    .stAlert { border-left: 4px solid #D32F2F !important; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# API Kulcsok
try:
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Sidebar
st.sidebar.header("⚙️ Beállítások")
if not SERPER_API_KEY: st.sidebar.warning("⚠️ SERPER_API_KEY hiányzik!")
else: st.sidebar.success("✅ Serper API aktív")

if not GEMINI_API_KEY: st.sidebar.warning("⚠️ GEMINI_API_KEY hiányzik!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    st.sidebar.success("✅ Gemini API aktív")

# UI
st.title("🏗️ Smartoria Lead Generator")
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    search_query = st.text_input("Keresési kulcsszó", value="Generálkivitelező Budapest Kft")
with col2:
    num_results = st.slider("Találatok száma", 1, 50, 10)
st.markdown("---")
start_button = st.button("🔍 Keresés és Feldolgozás Indítása", type="primary", use_container_width=True)

# --- FÜGGVÉNYEK ---

def search_with_serper(query, num_results):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results, "gl": "hu", "hl": "hu"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            results = response.json().get("organic", [])
            return [r.get("link") for r in results if r.get("link")]
    except: pass
    return []

def is_company_website(url):
    blacklist = ["facebook", "instagram", "linkedin", "youtube", "google", "jofogas", "joszaki", "szaki", "cylex", "aranyoldalak", "profession", "allas", "ingatlan", "wikipedia", "amazon", "emag", "pinterest", "tiktok"]
    try:
        domain = urlparse(url).netloc.lower()
        if any(x in domain for x in blacklist): return False
        return True
    except: return False

def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for tag in soup(["script", "style", "nav", "footer", "iframe", "svg"]): tag.decompose()
        
        # Regex keresés (Mankó)
        text = soup.get_text(separator=" ")
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        phones = re.findall(r"(?:\+36|06)[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{3,4}", text)
        
        contacts = {
            "emails": list(set(emails))[:3],
            "phones": list(set(phones))[:3]
        }
        return text[:20000], contacts
    except:
        return "", None

def extract_with_gemini(text, detected_contacts):
    system_instruction = """
    Extract: Company Name, Email, Phone Number, Address, Website URL, Contact Person, Role.
    Prioritize decision makers (Ügyvezető, Tulajdonos).
    Return ONLY raw JSON. Use 'N/A' if not found.
    """
    hints = f"Hints: {json.dumps(detected_contacts)}" if detected_contacts else ""
    prompt = f"{system_instruction}\n{hints}\n\nTEXT:\n{text}"
    
    # 1. VÁLTOZTATÁS: Stabilabb modell használata
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    defaults = {"Company Name": "N/A", "Email": "N/A", "Phone Number": "N/A", "Address": "N/A", "Contact Person": "N/A", "Role": "N/A", "Website URL": "N/A"}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response.text:
                clean_text = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_text)
                for k, v in defaults.items(): data.setdefault(k, v)
                return data
        except Exception as e:
            if "429" in str(e):
                # 2. VÁLTOZTATÁS: 60 másodperces várakozás
                wait = 60 
                st.warning(f"⚠️ Kvóta limit (429). Hűtés {wait} másodpercig... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                st.error(f"Hiba: {e}")
                break
    return defaults

# --- FŐ PROGRAM ---

if start_button:
    st.session_state.processing = True
    st.session_state.results_df = None
    
    with st.spinner("Keresés..."):
        links = search_with_serper(search_query, num_results)
    
    company_links = [l for l in links if is_company_website(l)]
    
    if not company_links:
        st.error("Nem találtunk megfelelő céges weboldalt.")
    else:
        st.success(f"{len(company_links)} weboldal feldolgozása...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, link in enumerate(company_links):
            status_text.text(f"Feldolgozás ({i+1}/{len(company_links)}): {link}")
            progress_bar.progress((i+1)/len(company_links))
            
            text, contacts = scrape_website(link)
            
            if text:
                data = extract_with_gemini(text, contacts)
                data["Source URL"] = link
                results.append(data)
            
            # 3. VÁLTOZTATÁS: Lassabb tempó a weboldalak között (20 mp)
            time.sleep(20)
            
        st.session_state.processing = False
        
        if results:
            df = pd.DataFrame(results)
            # Email szűrés
            if "Email" in df.columns:
                 df = df[df["Email"].astype(str).str.contains("@") & (df["Email"] != "N/A")]
            
            if not df.empty:
                st.session_state.results_df = df
                st.success(f"✅ {len(df)} lead sikeresen kinyerve!")
                st.dataframe(df)
                
                # Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                st.download_button("📥 Excel Letöltése", output.getvalue(), "leads.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("Nem találtunk érvényes email címet.")
        else:
            st.warning("Nem sikerült adatot kinyerni.")
