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

# Load environment variables
load_dotenv()

# Configure Streamlit page
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

# Sidebar - API Key Verification
st.sidebar.header("⚙️ Beállítások")
st.sidebar.markdown("---")

# Check for API keys
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not SERPER_API_KEY:
    st.sidebar.warning("⚠️ SERPER_API_KEY hiányzik a .env fájlból!")
else:
    st.sidebar.success("✅ Serper.dev API kulcs beállítva")

if not GEMINI_API_KEY:
    st.sidebar.warning("⚠️ GEMINI_API_KEY hiányzik a .env fájlból!")
else:
    st.sidebar.success("✅ Gemini API kulcs beállítva")

# Main title
st.title("🏗️ Smartoria Lead Generator")
st.markdown("---")

# Input Section
st.header("📋 Keresési Paraméterek")

col1, col2 = st.columns([2, 1])

with col1:
    search_query = st.text_input(
        "Keresési kulcsszó",
        value="Szeged Építőipari Központ",
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

# Core Functions
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
        "num": num_results
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
    """Heuristic filter to skip directories, social media and aggregators and keep likely direct company sites."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return False

    if not domain:
        return False

    # Domains we generally want to skip (not direct company homepages)
    blacklist_substrings = [
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "youtube.com",
        "twitter.com",
        "x.com",
        "maps.google.",
        "google.com",
        "bing.com",
        "yahoo.com",
        "cylex",
        "aranyoldalak",
        "telefonkonyv",
        "ceginfo",
        "ceguzlet",
        "cegkereso",
        "cegkatalogus",
        "profession.hu",
        "cvonline",
        "allasportal",
        "allasok.",
        "ingatlan.com",
        "ingatlanbazar",
        "jofogas.hu",
        "tripadvisor.",
        "booking.com",
        "airbnb.",
    ]

    if any(bad in domain for bad in blacklist_substrings):
        return False

    # If it passes blacklist, consider it a company site candidate
    return True

def scrape_website(url: str):
    """Scrape a website, clean HTML and return visible text plus detected contacts (emails + phones)."""
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
                if addr:
                    emails.add(addr)
            except Exception:
                continue

        # From tel: links
        for a in soup.select('a[href^="tel:"]'):
            href = a.get("href", "")
            try:
                tel = href.split(":", 1)[1].strip()
                if tel:
                    phones.add(tel)
            except Exception:
                continue

        # Text for regex scanning
        raw_text = soup.get_text(separator=" ")

        # Email regex (general)
        email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        emails.update(email_pattern.findall(raw_text))

        # Hungarian phone patterns: +36, 06, local mobile prefixes, etc.
        phone_pattern = re.compile(
            r"(?:\+36|36|06)\s*(?:1|20|30|31|70|[2-9]\d)\s*[/\-]?\s*\d{3}\s*[/\-]?\s*\d{3,4}"
        )
        phones.update(phone_pattern.findall(raw_text))

        detected_contacts = {
            "emails": sorted({e.strip() for e in emails if e.strip()}),
            "phones": sorted({p.strip() for p in phones if p.strip()}),
        }

        # Get cleaned text from page
        text = raw_text
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        # Limit text length to avoid token limits
        text = text[:5000] if len(text) > 5000 else text

        return text, detected_contacts
    except Exception as e:
        st.warning(f"⚠️ Nem sikerült letölteni: {url} - {str(e)}")
        return "", {"emails": [], "phones": []}

def extract_with_gemini(text: str, detected_contacts: dict | None = None) -> dict:
    """Extract company and leader information using Gemini AI, with strong focus on email quality."""
    if not GEMINI_API_KEY:
        st.error("Gemini API kulcs hiányzik!")
        return {}
    
    try:
        # Configure Gemini / PaLM client (v1beta)
        genai.configure(api_key=GEMINI_API_KEY)

        detected_contacts = detected_contacts or {"emails": [], "phones": []}

        system_prompt = """You are a highly precise data extraction assistant for B2B lead generation in the Hungarian construction industry.

Your PRIMARY goal is to identify the single BEST email address to contact the decision-maker at a company.

Use BOTH:
- the full website text
- a list of regex-detected contact candidates called `detected_contacts` (emails and phone numbers)

Extraction rules (email + leader):
1. Identify specific people with leadership roles such as (Hungarian examples):
   - "Ügyvezető", "Ügyvezető igazgató"
   - "Tulajdonos", "Társtulajdonos"
   - "Cégvezető"
   - "Műszaki vezető"
   - "Építésvezető"
   - "Projektvezető"
2. If you find a specific person AND at least one direct email clearly associated with them, use THAT person's email as the primary 'Email' field.
3. If multiple emails exist, apply this PRIORITY order:
   - Personal named email (e.g. 'vezeto.nev@ceg.hu') > role-based (e.g. 'ertekesites@') > generic ('info@', 'iroda@').
4. Only use email addresses that clearly appear in the website text or in `detected_contacts["emails"]`. Never invent or guess an email address.
5. If no good leader email is found, fall back to the best general company email.
6. If absolutely no email address can be found, set "Email" to "N/A" and still fill in the other fields if possible.
7. For phone numbers, you may still use the best available phone (mobile preferred over landline), but email quality is more important than phone.

You MUST return ONLY raw JSON in this exact format:
{
  "Company Name": "...",
  "Contact Person": "...",   // leader's name if found, otherwise "General"
  "Role": "...",             // e.g. "Ügyvezető", "Tulajdonos", "Office", etc.
  "Email": "...",
  "Phone Number": "...",
  "Address": "...",
  "Website URL": "..."
}

Important:
- Never add extra keys.
- Never add comments or explanations.
- Use the Hungarian text as-is for names and roles when possible.
"""

        prompt = (
            f"{system_prompt}\n\n"
            f"Full website text:\n{text}\n\n"
            f"detected_contacts (regex candidates) as JSON:\n"
            f"{json.dumps(detected_contacts, ensure_ascii=False)}"
        )

        # Default values for all expected keys
        defaults = {
            "Company Name": "N/A",
            "Contact Person": "General",
            "Role": "N/A",
            "Email": "N/A",
            "Phone Number": "N/A",
            "Address": "N/A",
            "Website URL": "N/A",
        }

        # Try several possible model names to be compatible with different library / API versions
        model_candidates = [
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash",
            "models/gemini-pro",
            "models/gemini-2.5-flash",
        ]

        response = None
        last_error = None

        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                import time
from google.api_core import exceptions # Ezt írd a fájl legtetejére az importokhoz!

# ... a cikluson belül ...

max_retries = 3
for attempt in range(max_retries):
    try:
        # Próbáljuk meg hívni a Geminit
        response = model.generate_content(prompt_text)
        
        # Ha sikerült, lépjünk ki a próbálkozós ciklusból
        break 
        
    except Exception as e:
        if "429" in str(e):
            wait_time = 30 # Most már 30 másodpercet várunk büntetés esetén
            st.warning(f"Túl sok kérés (429). Várakozás {wait_time} másodpercig...")
            time.sleep(wait_time)
            # És újrapróbálja a ciklus miatt...
        else:
            # Ha más hiba van (nem 429), akkor azt tényleg dobja el
            st.error(f"Hiba történt: {e}")
            response = None
            break

   # ... innen folytatódik a kód, ha megvan a response ...

        if response is None:
            st.warning(f"⚠️ Gemini extraction hiba (nincs elérhető modell): {last_error}")
            return defaults.copy()

        response_text = response.text.strip()

        # Clean the response (remove markdown code blocks if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON
        data = json.loads(response_text)

        # Ensure all expected keys exist
        for key, val in defaults.items():
            data.setdefault(key, val)

        return data
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ JSON parsing hiba: {str(e)}")
        return {
            "Company Name": "N/A",
            "Contact Person": "General",
            "Role": "N/A",
            "Email": "N/A",
            "Phone Number": "N/A",
            "Address": "N/A",
            "Website URL": "N/A",
        }
    except Exception as e:
        st.warning(f"⚠️ Gemini extraction hiba: {str(e)}")
        return {
            "Company Name": "N/A",
            "Contact Person": "General",
            "Role": "N/A",
            "Email": "N/A",
            "Phone Number": "N/A",
            "Address": "N/A",
            "Website URL": "N/A",
        }

# Main Processing Logic
if start_button and not st.session_state.processing:
    if not SERPER_API_KEY or not GEMINI_API_KEY:
        st.error("❌ Kérjük, állítsa be az API kulcsokat a .env fájlban!")
    else:
        st.session_state.processing = True
        st.session_state.results_df = None
        
        # Step 1: Search
        st.header("🔍 1. lépés: Keresés")
        with st.spinner("Keresés folyamatban..."):
            links = search_with_serper(search_query, num_results)
        
        if not links:
            st.error("Nem található eredmény vagy hiba történt a keresés során.")
            st.session_state.processing = False
        else:
            # Filter to likely direct company sites
            company_links = [link for link in links if is_company_website(link)]

            if not company_links:
                st.warning("⚠️ Nem találtunk közvetlen cég honlapot a találatok között (minden oldal kiszűrve lett).")
                st.session_state.processing = False
            else:
                st.success(
                    f"✅ {len(company_links)} közvetlen cég oldal felhasználva a(z) {len(links)} találatból"
                )
                st.session_state.processing = False
                
                # Step 2 & 3: Scrape and Extract
                st.header("📊 2-3. lépés: Weboldal feldolgozás és adat kinyerés (email fókusz)")
                
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
                        # Extract with Gemini (leader prioritization + hybrid contacts, email-focused)
                        extracted_data = extract_with_gemini(scraped_text, detected_contacts)
                        extracted_data["Source URL"] = link
                        results.append(extracted_data)

                    # Add delay to be polite (15 seconds between requests)
                    time.sleep(15)
                
                progress_bar.empty()
                status_text.empty()
                
                if results:
                    # Create DataFrame
                    df = pd.DataFrame(results)

                    # Focus on rows where we actually have an email
                    if "Email" in df.columns:
                        df = df[
                            df["Email"].notna()
                            & df["Email"].astype(str).str.strip().ne("")
                            & df["Email"].astype(str).str.upper().ne("N/A")
                        ]

                    if df.empty:
                        st.warning("⚠️ Nem találtunk érvényes email címmel rendelkező leadet.")
                    else:
                        # Reorder columns
                        column_order = [
                            "Company Name",
                            "Contact Person",
                            "Role",
                            "Email",
                            "Phone Number",
                            "Address",
                            "Website URL",
                            "Source URL",
                        ]
                        df = df[[col for col in column_order if col in df.columns]]
                        st.session_state.results_df = df
                        st.success(f"✅ {len(df)} emailes lead sikeresen feldolgozva!")
                else:
                    st.warning("⚠️ Nem sikerült adatokat kinyerni az oldalakról.")

# Display  
if st.session_state.results_df is not None:
    st.markdown("---")
    st.header("📋 Eredmények")
    
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    # Excel Download
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
        file_name=f"construction_leads_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
