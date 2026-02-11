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

# Custom CSS for Red, Black, and White theme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Main text */
    .stMarkdown, p, div {
        color: #000000 !important;
    }
    
    /* Buttons - Red with White text */
    .stButton > button {
        background-color: #D32F2F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #B71C1C !important;
        color: #FFFFFF !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }
    
    /* Slider */
    .stSlider {
        color: #D32F2F !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #FFFFFF !important;
    }
    
    /* Dataframe */
    .dataframe {
        color: #000000 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #D32F2F !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #D32F2F !important;
        color: #FFFFFF !important;
    }
    
    /* Warning boxes */
    .stAlert {
        border-left: 4px solid #D32F2F !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# --- API Kulcsok kezelése (Helyi .env vagy Streamlit Secrets) ---
try:
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Sidebar - API Key Verification
st.sidebar.header("⚙️ Beállítások")
st.sidebar.markdown("---")

if not SERPER_API_KEY:
    st.sidebar.warning("⚠️ SERPER_API_KEY hiányzik!")
else:
    st.sidebar.success("✅ Serper.dev API kulcs aktív")

if not GEMINI_API_KEY:
    st.sidebar.warning("⚠️ GEMINI_API_KEY hiányzik!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    st.sidebar.success("✅ Gemini API kulcs aktív")

# Main title
st.title("🏗️ Smartoria Lead Generator")
st.markdown("---")

# Input Section
st.header("📋 Keresési Paraméterek")

col1, col2 = st.columns([2, 1])

with col1:
    search_query = st.text_input(
        "Keresési kulcsszó",
        value="Generálkivitelező Budapest Kft",
        help="Adja meg a keresési kifejezést"
    )

with col2:
    num_results = st.slider(
        "Találatok száma",
        min_value=1,
        max_value=50,
        value=10,
        help="Válassza ki a keresett találatok számát"
    )

st.markdown("---")

# Start button
start_button = st.button(
    "🔍 Keresés és Feldolgozás Indítása",
    type="primary",
    use_container_width=True
)

# --- FÜGGVÉNYEK ---

def search_with_serper(query: str, num_results: int) -> list:
    """Search using Serper.dev API and return organic links"""
    if not SERPER_API_KEY:
        st.error("Serper.dev API kulcs hiányzik!")
        return []
    
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results,
        "gl": "hu",
        "hl": "hu"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract organic results
        organic_results = data.get("organic", [])
        links = [result.get("link", "") for result in organic_results if result.get("link")]
        return links
    except Exception as e:
        st.error(f"Hiba a keresés során: {str(e)}")
        return []

def is_company_website(url: str) -> bool:
    """Heuristic filter to skip directories, social media and aggregators."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return False

    if not domain:
        return False

    # Domains we generally want to skip
    blacklist_substrings = [
        "facebook.com", "instagram.com", "linkedin.com", "youtube.com",
        "twitter.com", "x.com", "maps.google.", "google.com", "bing.com",
        "yahoo.com", "cylex", "aranyoldalak", "telefonkonyv", "ceginfo",
        "ceguzlet", "cegkereso", "cegkatalogus", "profession.hu", "cvonline",
        "allasportal", "allasok.", "ingatlan.com", "ingatlanbazar",
        "jofogas.hu", "tripadvisor.", "booking.com", "airbnb.", "joszaki",
        "nemzeticegtar", "opten", "bisnode"
    ]

    if any(bad in domain for bad in blacklist_substrings):
        return False

    return True

def scrape_website(url: str):
    """Scrape a website, clean HTML and return visible text plus detected contacts."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove non-content / noisy elements
        for tag in soup(["script", "style", "meta", "link", "nav", "header", "footer", "aside", "noscript", "iframe", "svg", "form"]):
            tag.decompose()

        # --- Hybrid contact extraction (regex + HTML analysis) ---
        emails = set()
        phones = set()

        # From mailto: links
        for a in soup.select('a[href^="mailto:"]'):
            href = a.get("href", "")
            try:
                addr = href.split(":", 1)[1].split("?", 1)[0].strip()
                if addr: emails.add(addr)
            except Exception: continue

        # From tel: links
        for a in soup.select('a[href^="tel:"]'):
            href = a.get("href", "")
            try:
                tel = href.split(":", 1)[1].strip()
                if tel: phones.add(tel)
            except Exception: continue

        # Text for regex scanning
        raw_text = soup.get_text(separator=" ")

        # Email regex
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        emails.update(email_pattern.findall(raw_text))

        # Hungarian phone patterns
        phone_pattern = re.compile(r"(?:\+36|36|06)\s*(?:1|20|30|31|70|[2-9]\d)\s*[/\-]?\s*\d{3}\s*[/\-]?\s*\d{3,4}")
        phones.update(phone_pattern.findall(raw_text))

        detected_contacts = {
            "emails": sorted({e.strip() for e in emails if e.strip()})[:3],
            "phones": sorted({p.strip() for p in phones if p.strip()})[:3],
        }

        # Clean text
        text = " ".join(raw_text.split())
        text = text[:15000] # Limit text length

        return text, detected_contacts
        
    except Exception as e:
        # st.warning(f"⚠️ Nem sikerült letölteni: {url}")
        return "", {"emails": [], "phones": []}

def extract_with_gemini(text, detected_contacts=None):
    """
    Kinyeri az adatokat a szövegből a Gemini segítségével, újrapróbálkozással.
    """
    # 1. Prompt összeállítása
    system_instruction = """
    You are a data extraction assistant. Extract: Company Name, Email, Phone Number, Address, Website URL, Contact Person, Role.
    Prioritize decision makers (Ügyvezető, Tulajdonos).
    Return ONLY raw JSON. Use 'N/A' if not found.
    """
    
    hints = ""
    if detected_contacts:
        hints = f"\nDetected Contacts via Regex: {json.dumps(detected_contacts)}"

    prompt = f"{system_instruction}\n{hints}\n\nTEXT CONTENT:\n{text[:20000]}"

    # Alapértelmezett értékek hiba esetére
    defaults = {
        "Company Name": "N/A", "Email": "N/A", "Phone Number": "N/A",
        "Address": "N/A", "Website URL": "N/A", "Contact Person": "N/A", "Role": "N/A"
    }

    # Modell inicializálása
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Retry Logic (Újrapróbálkozás)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            
            if response and response.text:
                text_res = response.text.strip()
                # Markdown tisztítás
                if text_res.startswith("```json"): text_res = text_res[7:]
                if text_res.startswith("```"): text_res = text_res[3:]
                if text_res.endswith("```"): text_res = text_res[:-3]
                
                data = json.loads(text_res.strip())
                
                # Hiányzó kulcsok pótlása
                for key, val in defaults.items():
                    data.setdefault(key, val)
                    
                return data
                
        except Exception as e:
            if "429" in str(e):
                wait = 30
                st.warning(f"Kvóta limit (429). Várakozás {wait} mp... ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                st.error(f"Gemini hiba: {e}")
                if attempt == max_retries - 1:
                    break
    
    return defaults.copy()

# --- FŐ PROGRAM ---

if start_button and not st.session_state.processing:
    if not SERPER_API_KEY or not GEMINI_API_KEY:
        st.error("❌ Kérjük, állítsa be az API kulcsokat!")
    else:
        st.session_state.processing = True
        st.session_state.results_df = None
        
        # 1. lépés: Keresés
        st.header("🔍 1. lépés: Keresés")
        with st.spinner("Keresés folyamatban..."):
            links = search_with_serper(search_query, num_results)
        
        if not links:
            st.error("Nem található eredmény.")
            st.session_state.processing = False
        else:
            # Szűrés
            company_links = [link for link in links if is_company_website(link)]

            if not company_links:
                st.warning("⚠️ Nem találtunk közvetlen cég honlapot (minden oldal kiszűrve).")
                st.session_state.processing = False
            else:
                st.success(f"✅ {len(company_links)} közvetlen cég oldal felhasználva a(z) {len(links)} találatból")
                
                # 2-3. lépés: Adat kinyerés
                st.header("📊 2-3. lépés: Adatgyűjtés (Email & Döntéshozó fókusz)")
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total = len(company_links)
                for idx, link in enumerate(company_links):
                    status_text.text(f"Feldolgozás: {idx + 1}/{total} - {link[:50]}...")
                    progress_bar.progress((idx + 1) / total)
                    
                    # Scrape
                    scraped_text, detected_contacts = scrape_website(link)

                    if scraped_text:
                        # Extract
                        extracted_data = extract_with_gemini(scraped_text, detected_contacts)
                        extracted_data["Source URL"] = link
                        results.append(extracted_data)

                    # Udvariassági várakozás
                    time.sleep(10)
                
                progress_bar.empty()
                status_text.empty()
                st.session_state.processing = False
                
                if results:
                    df = pd.DataFrame(results)

                    # Szűrés: Csak ahol van email
                    if "Email" in df.columns:
                        df = df[
                            df["Email"].notna()
                            & df["Email"].astype(str).str.strip().ne("")
                            & df["Email"].astype(str).str.upper().ne("N/A")
                        ]

                    if df.empty:
                        st.warning("⚠️ Nem találtunk érvényes email címmel rendelkező leadet.")
                    else:
                        # Oszloprendezés
                        cols = ["Company Name", "Contact Person", "Role", "Email", "Phone Number", "Address", "Website URL", "Source URL"]
                        df = df[[c for c in cols if c in df.columns]]
                        st.session_state.results_df = df
                        st.success(f"✅ {len(df)} minőségi lead sikeresen feldolgozva!")
                else:
                    st.warning("⚠️ Nem sikerült adatokat kinyerni.")

# Eredmények megjelenítése
if st.session_state.results_df is not None:
    st.markdown("---")
    st.header("📋 Eredmények")
    
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    # Excel Letöltés
    st.markdown("---")
    
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Leads')
        output.seek(0)
        return output.getvalue()
    
    excel_data = to_excel(st.session_state.results_df)
    st.download_button(
        label="📥 Letöltés Excelben",
        data=excel_data,
        file_name=f"smartoria_leads_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )