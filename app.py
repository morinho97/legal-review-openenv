"""app.py - LexScan AI Legal Review System v2.0 with OCR"""
from __future__ import annotations
import base64,hashlib,io,json,random,re,sys
from datetime import datetime
import numpy as np
import streamlit as st
from PIL import Image,ImageEnhance,ImageFilter

try:
    import pdfplumber
    PDFPLUMBER_OK=True
except ImportError:
    PDFPLUMBER_OK=False

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_OK=True
except ImportError:
    OCR_OK=False

sys.path.insert(0,".")
from legal_env import(LegalReviewEnv,ACTION_LABELS,CLAUSE_TYPES,DIFFICULTY_PRESETS,grade_episode,rule_based_action)

st.set_page_config(page_title="LexScan",page_icon="scales",layout="wide",initial_sidebar_state="expanded")

CSS="""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');
*{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;background:#0e1117;color:#e8e4d9}
.main .block-container{background:#0e1117;padding:0 2rem 4rem;max-width:1400px}
.mhead{background:linear-gradient(135deg,#0a1628,#111d35,#0d1f3c);border-bottom:1px solid #c9a84c40;margin:0 -2rem 2rem;position:relative}
.mhead::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#c9a84c,#e8c97a,#c9a84c)}
.mhead-inner{display:flex;align-items:center;justify-content:space-between;padding:20px 40px}
.mhead-brand{display:flex;align-items:center;gap:16px}
.mhead-logo{width:48px;height:48px;background:#c9a84c;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:24px}
.mhead-title{font-family:'DM Serif Display',serif;font-size:26px;color:#e8e4d9;margin:0}
.mhead-sub{font-size:11px;color:#c9a84c;letter-spacing:.18em;text-transform:uppercase;margin:2px 0 0}
.mbadges{display:flex;gap:8px;align-items:center}
.mbadge{font-size:10px;padding:4px 12px;border:1px solid #c9a84c40;border-radius:20px;color:#c9a84c;font-family:'DM Mono',monospace}
.sdot{width:8px;height:8px;border-radius:50%;background:#2ecc71;display:inline-block;margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.sh{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#c9a84c;border-bottom:1px solid #c9a84c30;padding-bottom:8px;margin:24px 0 16px}
.card{background:#141b2d;border:1px solid #1e2d4a;border-radius:10px;padding:20px 24px;margin-bottom:16px}
.card-a{border-top:2px solid #c9a84c}
.sbar{display:flex;gap:0;margin:24px 0;background:#0a1628;border-radius:8px;overflow:hidden;border:1px solid #1e2d4a}
.si{flex:1;padding:12px 16px;text-align:center;font-size:11px;color:#8896b0;border-right:1px solid #1e2d4a}
.si:last-child{border-right:none}
.si.active{background:#c9a84c15;color:#c9a84c}
.si.done{background:#2ecc7115;color:#2ecc71}
.sn{display:block;font-family:'DM Mono',monospace;font-size:18px;font-weight:500;margin-bottom:4px}
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600;font-family:'DM Mono',monospace;letter-spacing:.06em;text-transform:uppercase}
.p-critical{background:#e74c3c22;color:#e74c3c;border:1px solid #e74c3c44}
.p-high{background:#e67e2222;color:#e67e22;border:1px solid #e67e2244}
.p-medium{background:#f1c40f22;color:#f1c40f;border:1px solid #f1c40f44}
.p-low{background:#3498db22;color:#3498db;border:1px solid #3498db44}
.p-none{background:#2ecc7122;color:#2ecc71;border:1px solid #2ecc7144}
.p-approve{background:#2ecc7122;color:#2ecc71;border:1px solid #2ecc7144}
.p-flag{background:#e74c3c22;color:#e74c3c;border:1px solid #e74c3c44}
.p-redline{background:#e67e2222;color:#e67e22;border:1px solid #e67e2244}
.p-clarify{background:#3498db22;color:#3498db;border:1px solid #3498db44}
.p-escalate{background:#9b59b622;color:#9b59b6;border:1px solid #9b59b644}
.ctp{font-family:'DM Serif Display',serif;font-size:13px;line-height:1.7;color:#c8d0e0;padding:12px 16px;background:#0a1020;border-radius:6px;border-left:3px solid #c9a84c50;margin-top:8px;font-style:italic}
.term{background:#060d1a;border:1px solid #1e2d4a;border-radius:8px;padding:16px;font-family:'DM Mono',monospace;font-size:11px;line-height:1.8;max-height:280px;overflow-y:auto;color:#2ecc71}
.enc{background:#060d1a;border:1px solid #c9a84c30;border-radius:8px;padding:14px 18px;font-family:'DM Mono',monospace;font-size:11px;color:#c9a84c;word-break:break-all;line-height:1.7}
.scanline{height:2px;background:linear-gradient(90deg,transparent,#c9a84c,transparent);animation:scan 2s linear infinite;margin:8px 0}
@keyframes scan{0%{opacity:0}50%{opacity:1}100%{opacity:0}}
section[data-testid="stSidebar"]{background:#080e1c !important;border-right:1px solid #1e2d4a}
section[data-testid="stSidebar"] *{color:#e8e4d9 !important}
.stButton>button{background:#c9a84c !important;color:#0a1628 !important;border:none !important;border-radius:6px !important;font-family:'DM Mono',monospace !important;font-size:12px !important;font-weight:500 !important;padding:10px 20px !important}
.stButton>button:hover{background:#e8c97a !important}
.stTabs [data-baseweb="tab-list"]{background:#141b2d;border-radius:8px;padding:4px;gap:4px;border:1px solid #1e2d4a}
.stTabs [data-baseweb="tab"]{background:transparent;color:#8896b0;border-radius:6px;font-family:'DM Mono',monospace;font-size:12px;padding:8px 20px}
.stTabs [aria-selected="true"]{background:#c9a84c !important;color:#0a1628 !important}
.stProgress>div>div{background:#c9a84c !important}
[data-testid="stFileUploader"]{background:#0a1628;border:2px dashed #c9a84c50;border-radius:12px;padding:12px}
.stMetric{background:#141b2d;border-radius:8px;padding:12px;border:1px solid #1e2d4a}
.stMetric label{color:#8896b0 !important;font-size:11px !important}
.stMetric [data-testid="stMetricValue"]{color:#c9a84c !important;font-family:'DM Mono',monospace !important}
div[data-testid="stExpander"]{background:#141b2d;border:1px solid #1e2d4a;border-radius:8px;margin-bottom:8px}
div[data-testid="stExpander"] summary{color:#e8e4d9 !important}
textarea,input{background:#141b2d !important;color:#e8e4d9 !important;border-color:#1e2d4a !important}
.stSelectbox>div>div{background:#141b2d !important;color:#e8e4d9 !important}
.stMultiSelect>div{background:#141b2d !important}
</style>
"""
st.markdown(CSS,unsafe_allow_html=True)
st.markdown('<div class="mhead"><div class="mhead-inner"><div class="mhead-brand"><div class="mhead-logo">AL</div><div><div class="mhead-title">LexScan</div><div class="mhead-sub">AI Legal Document Review System</div></div></div><div class="mbadges"><span class="mbadge"><span class="sdot"></span>SYSTEM ONLINE</span><span class="mbadge">OCR ENABLED</span><span class="mbadge">v2.0.0</span></div></div></div>',unsafe_allow_html=True)

def _init():
    D={"env":None,"difficulty":"easy","done":False,"step_num":0,"log_lines":[],"current_clause":None,"last_action":None,"last_reward":None,"metrics_snapshot":None,"pdf_results":[],"pdf_filename":"","pdf_text":"","scan_method":"","ocr_pages":0}
    for k,v in D.items():
        if k not in st.session_state:st.session_state[k]=v
_init()

def rp(r):
    icons={"CRITICAL":"[C]","HIGH":"[H]","MEDIUM":"[M]","LOW":"[L]","NONE":"[N]"}
    cls=r.lower();return f'<span class="pill p-{cls}">{icons.get(r,"")} {r}</span>'
def ap(a):
    lab={"approve":"APPROVE","flag_clause":"FLAG","redline":"REDLINE","request_clarify":"CLARIFY","escalate_counsel":"ESCALATE"}
    kind={"approve":"approve","flag_clause":"flag","redline":"redline","request_clarify":"clarify","escalate_counsel":"escalate"}.get(a,"approve")
    return f'<span class="pill p-{kind}">{lab.get(a,a.upper())}</span>'
def enc(t):return{"sha256":hashlib.sha256(t.encode()).hexdigest(),"base64":base64.b64encode(t.encode()).decode(),"md5":hashlib.md5(t.encode()).hexdigest()}
def start_env(d):
    e=LegalReviewEnv(difficulty=d);obs=e.reset()
    st.session_state.update({"env":e,"difficulty":d,"done":False,"step_num":0,"log_lines":[],"last_action":None,"last_reward":None,"metrics_snapshot":None})
    st.session_state.current_clause=e.state().get("current_clause")
    st.session_state.log_lines.append(f'<span style="color:#3498db">[START] Task: {d} | {DIFFICULTY_PRESETS[d]["contract_type"]}</span>')

def preproc(img):
    img=img.convert("L");img=ImageEnhance.Contrast(img).enhance(2.0);img=img.filter(ImageFilter.SHARPEN)
    w,h=img.size
    if w<1200:s=1200/w;img=img.resize((int(w*s),int(h*s)),Image.LANCZOS)
    return img

def ext_digital(b):
    if not PDFPLUMBER_OK:return"",0
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:return"\n\n".join(p.extract_text() or "" for p in pdf.pages),len(pdf.pages)
    except:return"",0

def ext_ocr_pdf(b):
    if not OCR_OK:return"",0
    try:
        pages=convert_from_bytes(b,dpi=300)
        cfg=r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        texts=[]
        for i,p in enumerate(pages):
            t=pytesseract.image_to_string(preproc(p),config=cfg)
            if t.strip():texts.append(f"[Page {i+1}]\n{t}")
        return"\n\n".join(texts),len(pages)
    except Exception as e:return f"OCR error: {e}",0

def ext_image(b):
    if not OCR_OK:return""
    try:return pytesseract.image_to_string(preproc(Image.open(io.BytesIO(b))),config=r'--oem 3 --psm 6')
    except Exception as e:return f"OCR error: {e}"

def smart_ext(f):
    n=f.name.lower();raw=f.read()
    if n.endswith((".txt",".md")):return raw.decode("utf-8",errors="replace"),"plaintext",1
    if n.endswith(".pdf"):
        t,p=ext_digital(raw)
        if t.strip() and len(t.strip())>100:return t,"digital_pdf",p
        t,p=ext_ocr_pdf(raw);return t,"ocr_pdf",p
    if n.endswith((".jpg",".jpeg",".png",".bmp",".tiff",".webp")):return ext_image(raw),"image_ocr",1
    try:return raw.decode("utf-8",errors="replace"),"plaintext",1
    except:return"","unknown",0

def split_cls(text):
    n=re.split(r'\n(?=\s*(?:clause|section|article|para(?:graph)?)?\s*\.?\s*\d+[\.\d]*\s*[:\.]?\s*[A-Z])',text,flags=re.IGNORECASE)
    if len(n)>=3:
        c=[x.strip() for x in n if len(x.strip())>50]
        if len(c)>=2:return c
    caps=re.split(r'\n(?=[A-Z][A-Z\s\-]{4,}\.?\n)',text)
    if len(caps)>=3:
        c=[x.strip() for x in caps if len(x.strip())>50]
        if len(c)>=2:return c
    paras=[p.strip() for p in re.split(r'\n\s*\n',text) if len(p.strip())>60]
    if len(paras)>=2:return paras
    sents=re.split(r'(?<=[.!?])\s+',text);chunks=[];cur=""
    for s in sents:
        cur+=" "+s
        if len(cur)>250:chunks.append(cur.strip());cur=""
    if cur.strip():chunks.append(cur.strip())
    return chunks if chunks else[text[:1000]]if text.strip() else[]

def cls_type(text):
    t=text.lower()
    rules=[(["indemnif","hold harmless","defend and hold"],"indemnity"),(["limitation of liability","in no event","not be liable","consequential"],"limitation_of_liability"),(["terminat","cancel this","end this agreement"],"termination"),(["intellectual property","ip ownership","patent","copyright"],"ip_ownership"),(["confidential","non-disclosure","nda","trade secret"],"confidentiality"),(["payment","invoice","fee","price","billable"],"payment_terms"),(["arbitration","dispute","mediation"],"dispute_resolution"),(["governing law","applicable law"],"governing_law"),(["warrants","represents","warranty","disclaimer"],"warranty"),(["data protection","gdpr","privacy","personal data"],"data_protection")]
    for kws,ct in rules:
        if any(k in t for k in kws):return ct
    return random.choice(CLAUSE_TYPES)

def cls_risk(text,ctype):
    H={"indemnity","limitation_of_liability","termination","ip_ownership"};M={"payment_terms","data_protection","dispute_resolution"}
    t=text.lower()
    if any(k in t for k in["unlimited liability","any and all claims","irrevocable","in perpetuity","all damages"]):return"CRITICAL"
    if ctype in H:return"HIGH"
    if ctype in M:return"MEDIUM"
    if any(w in t for w in["shall","must","required","obligation"]):return"LOW"
    return"NONE"

def review(text,idx):
    ct=cls_type(text);risk=cls_risk(text,ct)
    ctx={"clause_type":ct,"has_nested_ref":bool(re.search(r'see clause|as defined in|schedule|exhibit',text.lower())),"prior_redlines":0,"time_remaining":999}
    a=rule_based_action(ctx);an=ACTION_LABELS[a]
    HR={"indemnity","limitation_of_liability","termination","ip_ownership"}
    cl=ct in HR or risk in("HIGH","CRITICAL")
    if an=="approve" and cl:rw=-5.0
    elif an in("flag_clause","escalate_counsel") and cl:rw=2.0
    elif an=="approve":rw=1.0
    elif an=="redline":rw=-0.3
    else:rw=-0.2
    return{"clause_id":idx,"text":text[:400]+("..." if len(text)>400 else""),"full_text":text,"clause_type":ct,"risk":risk,"action":an,"reward":round(rw,2),"contains_liability":cl,"nested_ref":ctx["has_nested_ref"],"word_count":len(text.split())}

# SIDEBAR
with st.sidebar:
    st.markdown("## LexScan AI");st.markdown("---")
    st.markdown("**Capabilities**")
    for lbl,ok in[("Digital PDF",PDFPLUMBER_OK),("Scanned PDF (OCR)",OCR_OK),("Image OCR",OCR_OK),("Plain Text",True),("AI Analysis",True)]:
        st.markdown(f"{'OK' if ok else 'X'} {lbl}")
    if not OCR_OK:st.warning("OCR not installed.\npip install pytesseract pdf2image\n\nAlso install Tesseract:\nWindows: UB-Mannheim website\nLinux: apt install tesseract-ocr\nMac: brew install tesseract")
    st.markdown("---");st.markdown("**Simulation**")
    diff=st.selectbox("Difficulty",["easy","medium","hard"])
    if st.button("Init Simulation"):start_env(diff);st.success("Ready")
    if st.session_state.env:
        p=DIFFICULTY_PRESETS[st.session_state.difficulty]
        st.markdown(f"Contract: {p['contract_type']}");st.markdown(f"Step: {st.session_state.step_num}")
    st.markdown("---");st.markdown("**Actions**")
    for c,n,col in[("0","APPROVE","#2ecc71"),("1","FLAG","#e74c3c"),("2","REDLINE","#e67e22"),("3","CLARIFY","#3498db"),("4","ESCALATE","#9b59b6")]:
        st.markdown(f'<span style="color:{col};font-family:DM Mono,monospace;font-size:12px">{c} = {n}</span>',unsafe_allow_html=True)

tab_scan,tab_sim,tab_enc_tab=st.tabs(["Document Scanner","AI Simulation","Encryption Lab"])

with tab_scan:
    re_exist=bool(st.session_state.pdf_results)
    d="done";a="active";e=""
    s1=d if re_exist else a;s2=d if re_exist else e;s3=d if re_exist else e;s4=d if re_exist else e
    st.markdown(f'<div class="sbar"><div class="si {s1}"><span class="sn">01</span>Upload</div><div class="si {s2}"><span class="sn">02</span>Extract</div><div class="si {s3}"><span class="sn">03</span>AI Review</div><div class="si {s4}"><span class="sn">04</span>Results</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="sh">Upload Contract Document</div>',unsafe_allow_html=True)
    cu,ci=st.columns([3,2],gap="large")
    with cu:
        st.markdown('<div style="background:#0a1628;border:2px dashed #c9a84c50;border-radius:12px;padding:32px 24px;text-align:center;margin-bottom:16px"><div style="font-size:36px;margin-bottom:8px">PDF</div><div style="font-family:DM Serif Display,serif;font-size:18px;color:#e8e4d9;margin-bottom:6px">Drop your contract here</div><div style="font-size:12px;color:#8896b0">PDF (digital or scanned) - JPG - PNG - TXT</div><div style="font-size:11px;color:#c9a84c;margin-top:6px">Scanned documents processed with OCR automatically</div></div>',unsafe_allow_html=True)
        uploaded=st.file_uploader("Choose file",type=["pdf","txt","jpg","jpeg","png","bmp","tiff"],label_visibility="collapsed")
        st.markdown("**Or paste contract text:**")
        pasted=st.text_area("",height=130,placeholder="Paste contract clauses here...\n\nINDEMNIFICATION\nParty A shall indemnify Party B from all claims...\n\nTERMINATION\nEither party may terminate upon 30 days notice.",label_visibility="collapsed",key="paste_in")
    with ci:
        st.markdown('<div class="card card-a"><div style="font-family:DM Mono,monospace;font-size:10px;letter-spacing:.15em;color:#c9a84c;margin-bottom:12px">HOW IT WORKS</div><div style="margin-bottom:12px"><div style="font-size:13px;font-weight:600;color:#e8e4d9;margin-bottom:4px">Digital PDF</div><div style="font-size:12px;color:#8896b0;line-height:1.6">Text extracted directly - fast, 100% accuracy.</div></div><div style="margin-bottom:12px"><div style="font-size:13px;font-weight:600;color:#e8e4d9;margin-bottom:4px">Scanned PDF or Photo</div><div style="font-size:12px;color:#8896b0;line-height:1.6">Pages converted to images, enhanced (contrast + sharpness), then Tesseract OCR reads every word.</div></div><div style="margin-bottom:12px"><div style="font-size:13px;font-weight:600;color:#e8e4d9;margin-bottom:4px">AI Review</div><div style="font-size:12px;color:#8896b0;line-height:1.6">Each clause classified by type and risk. Agent decides: approve, flag, redline, clarify or escalate.</div></div><div style="border-top:1px solid #1e2d4a;padding-top:10px;font-family:DM Mono,monospace;font-size:11px;color:#c9a84c">PDF - JPG - PNG - BMP - TIFF - TXT</div></div>',unsafe_allow_html=True)
    b1,b2,_=st.columns([1,1,3])
    with b1:go=st.button("Scan and Analyse")
    with b2:
        if st.button("Clear Results"):
            st.session_state.pdf_results=[];st.session_state.pdf_text="";st.session_state.pdf_filename="";st.session_state.scan_method="";st.rerun()
    if go:
        raw,method,pages,fname="","",0,"input"
        if uploaded:
            fname=uploaded.name
            with st.spinner("Extracting text from document..."):raw,method,pages=smart_ext(uploaded)
            mlab={"digital_pdf":"Digital PDF - text extracted directly","ocr_pdf":"Scanned PDF - OCR applied to all pages","image_ocr":"Image - Tesseract OCR applied","plaintext":"Plain text file read directly"}
            if raw.strip() and len(raw.strip())>20:st.success(f"{mlab.get(method,'Extracted')} | {pages} page(s) | {len(raw):,} characters")
            else:st.error("Could not extract text. For scanned PDFs ensure Tesseract is installed. Try pasting text manually.")
        elif pasted.strip():raw,method,pages,fname=pasted.strip(),"plaintext",1,"pasted_text"
        else:st.warning("Upload a file or paste text first.")
        if raw.strip() and len(raw.strip())>20:
            with st.spinner("AI agent reviewing clauses..."):
                clauses=split_cls(raw)
                if not clauses:st.error("No clauses identified in extracted text.")
                else:
                    results=[review(c,i) for i,c in enumerate(clauses)]
                    st.session_state.pdf_results=results;st.session_state.pdf_text=raw
                    st.session_state.pdf_filename=fname;st.session_state.scan_method=method;st.session_state.ocr_pages=pages
            if st.session_state.pdf_results:st.success(f"Analysis complete - {len(st.session_state.pdf_results)} clauses reviewed")
    if st.session_state.pdf_results:
        results=st.session_state.pdf_results;method=st.session_state.scan_method;fname=st.session_state.pdf_filename
        st.markdown('<div class="sh">Analysis Results</div>',unsafe_allow_html=True)
        minfo={"digital_pdf":"Digital PDF - text extracted directly","ocr_pdf":"Scanned PDF - OCR applied to all pages","image_ocr":"Image file - OCR applied","plaintext":"Plain text - read directly"}
        st.markdown(f'<div class="card" style="display:flex;align-items:center;gap:16px;padding:14px 20px;margin-bottom:16px"><div style="font-size:24px;background:#c9a84c22;border:1px solid #c9a84c44;border-radius:6px;width:48px;height:48px;display:flex;align-items:center;justify-content:center;font-family:DM Mono,monospace;font-weight:600;color:#c9a84c">PDF</div><div><div style="font-family:DM Mono,monospace;font-size:11px;color:#c9a84c;letter-spacing:.1em">{method.upper().replace("_"," ")}</div><div style="font-size:13px;color:#e8e4d9;margin-top:2px">{fname}</div><div style="font-size:11px;color:#8896b0;margin-top:2px">{minfo.get(method,"")} - {len(results)} clauses identified</div></div></div>',unsafe_allow_html=True)
        nc=sum(1 for r in results if r["risk"]=="CRITICAL");nh=sum(1 for r in results if r["risk"]=="HIGH")
        nf=sum(1 for r in results if r["action"] in("flag_clause","escalate_counsel","redline"))
        na=sum(1 for r in results if r["action"]=="approve");nl=sum(1 for r in results if r["contains_liability"])
        m1,m2,m3,m4,m5,m6=st.columns(6)
        m1.metric("Clauses",len(results));m2.metric("Critical",nc,delta=f"-{nc}"if nc else None,delta_color="inverse")
        m3.metric("High Risk",nh,delta=f"-{nh}"if nh else None,delta_color="inverse");m4.metric("Flagged",nf);m5.metric("Approved",na);m6.metric("Liability",nl)
        st.markdown("<br>",unsafe_allow_html=True)
        fc1,fc2=st.columns([2,3])
        with fc1:filter_risk=st.multiselect("Filter risk",["CRITICAL","HIGH","MEDIUM","LOW","NONE"],default=["CRITICAL","HIGH","MEDIUM"])
        with fc2:sort_by=st.selectbox("Sort",["Risk (highest first)","Clause order","Reward (lowest first)"])
        filtered=[r for r in results if r["risk"] in filter_risk]if filter_risk else results
        if sort_by=="Risk (highest first)":ro={"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3,"NONE":4};filtered=sorted(filtered,key=lambda r:ro.get(r["risk"],5))
        elif sort_by=="Reward (lowest first)":filtered=sorted(filtered,key=lambda r:r["reward"])
        st.markdown(f'<div style="font-size:12px;color:#8896b0;margin-bottom:12px">Showing <strong style="color:#c9a84c">{len(filtered)}</strong> of <strong style="color:#c9a84c">{len(results)}</strong> clauses</div>',unsafe_allow_html=True)
        rbg={"CRITICAL":"#3d0000","HIGH":"#2d1500","MEDIUM":"#2d2900","LOW":"#001530","NONE":"#001a0a"}
        rbd={"CRITICAL":"#e74c3c","HIGH":"#e67e22","MEDIUM":"#f1c40f","LOW":"#3498db","NONE":"#2ecc71"}
        for r in filtered:
            bg=rbg.get(r["risk"],"#141b2d");bd=rbd.get(r["risk"],"#1e2d4a")
            with st.expander(f"Clause {r['clause_id']+1} | {r['clause_type'].upper().replace('_',' ')} | {r['risk']} | {r['action'].upper()} | Reward: {r['reward']:+.1f}",expanded=r["risk"] in("CRITICAL","HIGH")):
                c1,c2,c3,c4,c5=st.columns(5)
                c1.markdown(f"**Type**<br>`{r['clause_type'].replace('_',' ')}`",unsafe_allow_html=True)
                c2.markdown(f"**Risk**<br>{rp(r['risk'])}",unsafe_allow_html=True)
                c3.markdown(f"**Action**<br>{ap(r['action'])}",unsafe_allow_html=True)
                c4.markdown(f"**Reward**<br>`{r['reward']:+.2f}`",unsafe_allow_html=True)
                c5.markdown(f"**Words**<br>`{r['word_count']}`",unsafe_allow_html=True)
                st.markdown(f'<div style="background:{bg};border-left:3px solid {bd};border-radius:6px;padding:14px 18px;margin-top:12px;font-family:Georgia,serif;font-size:13px;line-height:1.8;color:#d0d8e8;font-style:italic">{r["full_text"][:700]}{"..." if len(r["full_text"])>700 else ""}</div>',unsafe_allow_html=True)
                flags=[]
                if r["contains_liability"]:flags.append("WARNING: Contains liability exposure")
                if r["nested_ref"]:flags.append("INFO: References another clause")
                if r["risk"]=="CRITICAL":flags.append("CRITICAL: Requires senior counsel immediately")
                if r["risk"]=="HIGH":flags.append("HIGH RISK: Do not sign without legal review")
                if flags:
                    st.markdown("<br>",unsafe_allow_html=True)
                    for flag in flags:st.markdown(f'<div style="font-size:12px;color:#e8e4d9;padding:3px 0">{flag}</div>',unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        dl1,dl2=st.columns(2)
        with dl1:st.download_button("Download Full Report JSON",json.dumps(results,indent=2),file_name=f"lexscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",mime="application/json")
        with dl2:
            csv=["clause_id,type,risk,action,reward,liability,words"]+[f"{r['clause_id']},{r['clause_type']},{r['risk']},{r['action']},{r['reward']},{r['contains_liability']},{r['word_count']}" for r in results]
            st.download_button("Download Summary CSV","\n".join(csv),file_name=f"lexscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",mime="text/csv")
        with st.expander("View extracted text"):
            if method in("ocr_pdf","image_ocr"):st.info("Extracted via OCR - minor character errors from scan quality are normal.")
            st.markdown('<div class="scanline"></div>',unsafe_allow_html=True)
            st.text_area("",value=st.session_state.pdf_text,height=300,key="tv",label_visibility="collapsed")

with tab_sim:
    env=st.session_state.env;cl,cr=st.columns([3,2],gap="large")
    with cl:
        st.markdown('<div class="sh">Dashboard</div>',unsafe_allow_html=True)
        tr,rev,rem=0.0,0,0
        if env:info=env._get_info();tr=info["total_reward"];rev=info["reviewed"];rem=info["remaining"]
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Reviewed",rev);d2.metric("Remaining",rem);d3.metric("Total Reward",f"{tr:.3f}");d4.metric("Status","DONE"if st.session_state.done else"ACTIVE")
        st.markdown('<div class="sh">Current Clause</div>',unsafe_allow_html=True)
        clause=st.session_state.current_clause
        if clause:
            cm1,cm2,cm3,cm4=st.columns(4)
            cm1.markdown(f"**ID** `{clause['clause_id']}`");cm2.markdown(f"**Type** `{clause['clause_type']}`")
            cm3.markdown(f"**Jurisdiction** `{clause['jurisdiction']}`")
            cm4.markdown(f"**Risk** {rp(clause.get('true_risk','NONE'))}",unsafe_allow_html=True)
            st.markdown(f'<div class="card card-a" style="margin-top:10px"><div class="ctp">{clause.get("text","...")}</div><div style="font-size:11px;color:#8896b0;margin-top:8px;font-family:DM Mono,monospace">NESTED: {"YES"if clause.get("has_nested_ref")else"no"} | REDLINES: {clause.get("prior_redlines",0)} | LIABILITY: {"YES"if clause.get("contains_liability")else"no"}</div></div>',unsafe_allow_html=True)
        elif st.session_state.done:st.info("Episode complete. Reinitialise from sidebar.")
        else:st.info("Click Init Simulation in sidebar to begin.")
        if st.session_state.last_action is not None:
            st.markdown('<div class="sh">Last Decision</div>',unsafe_allow_html=True)
            la1,la2=st.columns(2)
            la1.markdown(f"**Action** {ap(ACTION_LABELS[st.session_state.last_action])}",unsafe_allow_html=True)
            la2.markdown(f"**Reward** `{st.session_state.last_reward:+.4f}`")
        st.markdown('<div class="sh">Performance Metrics</div>',unsafe_allow_html=True)
        sm=st.session_state.metrics_snapshot
        pm1,pm2,pm3,pm4=st.columns(4)
        pm1.metric("F1",f"{sm['f1']:.3f}"if sm else"--");pm2.metric("Precision",f"{sm['precision']:.3f}"if sm else"--")
        pm3.metric("Recall",f"{sm['recall']:.3f}"if sm else"--");pm4.metric("Score",f"{grade_episode(env):.3f}"if(env and env._cursor>0)else"--")
        st.markdown('<div class="sh">Controls</div>',unsafe_allow_html=True)
        sb1,sb2,sb3=st.columns(3)
        with sb1:
            if st.button("Next Step"):
                if env and not st.session_state.done:
                    s=env.state();ctx=s.get("current_clause") or{};ctx["contract_type"]=s.get("contract_type","");ctx["time_remaining"]=s.get("time_remaining",999)
                    action=rule_based_action(ctx);obs,reward,done,info=env.step(action)
                    st.session_state.step_num+=1;st.session_state.done=done;st.session_state.last_action=action;st.session_state.last_reward=reward
                    st.session_state.metrics_snapshot=env.episode_metrics();st.session_state.current_clause=env.state().get("current_clause")
                    st.session_state.log_lines.append(f'<span style="color:#2ecc71">[STEP] step={st.session_state.step_num}, reward={reward:.4f}</span>')
                    if done:score=grade_episode(env);st.session_state.log_lines.append(f'<span style="color:#c9a84c">[END] Score: {score:.4f}</span>')
                else:st.warning("Initialise from sidebar.")
        with sb2:
            if st.button("Run Full Episode"):
                if env and not st.session_state.done:
                    prog=st.progress(0);nt=env.preset["n_clauses"]
                    while not st.session_state.done:
                        s=env.state();ctx=s.get("current_clause") or{};ctx["contract_type"]=s.get("contract_type","");ctx["time_remaining"]=s.get("time_remaining",999)
                        action=rule_based_action(ctx);obs,reward,done,info=env.step(action)
                        st.session_state.step_num+=1;st.session_state.done=done;st.session_state.last_action=action;st.session_state.last_reward=reward
                        st.session_state.log_lines.append(f'<span style="color:#2ecc71">[STEP] step={st.session_state.step_num}, reward={reward:.4f}</span>')
                        prog.progress(min(st.session_state.step_num/nt,1.0))
                        if done:break
                    score=grade_episode(env);st.session_state.metrics_snapshot=env.episode_metrics();st.session_state.current_clause=env.state().get("current_clause")
                    st.session_state.log_lines.append(f'<span style="color:#c9a84c">[END] Score: {score:.4f}</span>')
                    prog.progress(1.0);st.success(f"Done! Score: {score:.4f}")
                else:st.warning("Initialise from sidebar.")
        with sb3:
            if st.button("Reset Episode"):start_env(st.session_state.difficulty);st.rerun()
        if env and env._decisions:
            st.markdown('<div class="sh">Decision Log</div>',unsafe_allow_html=True)
            rows="".join(f"<tr style='font-size:11px'><td style='padding:6px 10px;color:#8896b0;font-family:DM Mono,monospace'>{d['clause_id']}</td><td style='padding:6px 10px;color:#e8e4d9'>{d['clause_type'].replace('_',' ')}</td><td style='padding:6px 10px;color:#8896b0'>{d['jurisdiction']}</td><td style='padding:6px 10px'>{d['action'].upper()}</td><td style='padding:6px 10px'>{d['true_risk']}</td><td style='padding:6px 10px;color:{'#2ecc71'if d['reward']>0 else'#e74c3c'}'>{d['reward']:+.2f}</td></tr>" for d in reversed(env._decisions[-12:]))
            st.markdown(f'<table style="width:100%;border-collapse:collapse;background:#141b2d;border-radius:8px;overflow:hidden"><thead><tr style="background:#0a1628"><th style="padding:8px 10px;text-align:left;font-size:10px;color:#c9a84c;font-family:DM Mono,monospace">ID</th><th style="padding:8px 10px;text-align:left;font-size:10px;color:#c9a84c;font-family:DM Mono,monospace">TYPE</th><th style="padding:8px 10px;text-align:left;font-size:10px;color:#c9a84c;font-family:DM Mono,monospace">JUR</th><th style="padding:8px 10px;text-align:left;font-size:10px;color:#c9a84c;font-family:DM Mono,monospace">ACTION</th><th style="padding:8px 10px;text-align:left;font-size:10px;color:#c9a84c;font-family:DM Mono,monospace">RISK</th><th style="padding:8px 10px;text-align:left;font-size:10px;color:#c9a84c;font-family:DM Mono,monospace">REWARD</th></tr></thead><tbody>{rows}</tbody></table>',unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="sh">Execution Terminal</div>',unsafe_allow_html=True)
        log_html="<br>".join(st.session_state.log_lines)if st.session_state.log_lines else'<span style="color:#1e2d4a">Awaiting initialisation...</span>'
        st.markdown(f'<div class="term">{log_html}</div>',unsafe_allow_html=True)

with tab_enc_tab:
    st.markdown('<div class="sh">Document Encryption Simulation</div>',unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-bottom:20px"><div style="font-size:13px;color:#8896b0;line-height:1.8"><strong style="color:#e8e4d9">SHA-256</strong> - irreversible fingerprint, detects tampering. <strong style="color:#e8e4d9">Base64</strong> - reversible encoding for safe transmission. <strong style="color:#e8e4d9">MD5</strong> - fast uniqueness check.</div></div>',unsafe_allow_html=True)
    enc_in=st.text_area("Document to encrypt","Party A shall indemnify and hold harmless Party B from any and all claims and damages arising from breach of this Agreement.",height=90,key="enc_in")
    if st.button("Encrypt Document"):
        if enc_in.strip():
            ev=enc(enc_in.strip())
            e1,e2=st.columns(2)
            with e1:st.metric("Input chars",len(enc_in));st.metric("Base64 size",len(ev["base64"]))
            with e2:st.metric("Algorithm","SHA-256 + Base64 + MD5");st.metric("Hash length","64 hex chars")
            st.markdown("**SHA-256 Hash:**");st.markdown(f'<div class="enc">{ev["sha256"]}</div>',unsafe_allow_html=True)
            st.markdown("**MD5 Checksum:**");st.markdown(f'<div class="enc">{ev["md5"]}</div>',unsafe_allow_html=True)
            st.markdown("**Base64 Encoding:**");st.markdown(f'<div class="enc">{ev["base64"]}</div>',unsafe_allow_html=True)
    if st.session_state.pdf_results and st.session_state.pdf_text:
        st.markdown("---")
        st.markdown(f'<div class="sh">Encrypt Uploaded Contract - {st.session_state.pdf_filename}</div>',unsafe_allow_html=True)
        if st.button("Encrypt Full Contract"):
            ev=enc(st.session_state.pdf_text)
            st.markdown("**SHA-256:**");st.markdown(f'<div class="enc">{ev["sha256"]}</div>',unsafe_allow_html=True)
            st.markdown("**Base64 preview:**");st.markdown(f'<div class="enc">{ev["base64"][:300]}...</div>',unsafe_allow_html=True)
            st.success(f"Fingerprint: {ev['sha256'][:20]}...")

st.markdown('<div style="border-top:1px solid #1e2d4a;margin-top:40px;padding-top:16px;text-align:center;font-family:DM Mono,monospace;font-size:10px;color:#3a4a6a;letter-spacing:.12em">LEXSCAN AI LEGAL REVIEW - META PYTORCH OPENENV HACKATHON - v2.0.0 - NOT LEGAL ADVICE</div>',unsafe_allow_html=True)
