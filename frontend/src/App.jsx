import { useState } from "react";

const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000").replace(/\/$/, "");
const HISTORY_KEY = "truthlens-history-v1";

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [gemini, setGemini] = useState(null);
  const [geminiBusy, setGeminiBusy] = useState(false);
  const [saveHistory, setSaveHistory] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
  });

  async function analyze() {
    setBusy(true); setError(""); setResult(null); setGemini(null);
    try {
      let response;
      if (file) {
        const body = new FormData(); body.append("file", file);
        response = await fetch(API + "/api/analyze-pdf", { method: "POST", body });
      } else {
        response = await fetch(API + "/api/analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Analysis request failed.");
      setResult(data);
      if (saveHistory) {
        const entry = { id: Date.now(), date: new Date().toISOString(), title: file?.name || text.trim().slice(0, 90), prediction: data.prediction, excerpt: file ? "" : text.trim().slice(0, 200) };
        const next = [entry, ...history].slice(0, 10);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next)); setHistory(next);
      }
    } catch (e) { setError(e.message + " Make sure the Python API is running at " + API + "."); }
    finally { setBusy(false); }
  }

  async function analyzeWithGemini() {
    setGeminiBusy(true); setError("");
    try {
      const response = await fetch(API + "/api/gemini-analysis", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: file ? "" : text })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Gemini analysis failed.");
      setGemini(data);
    } catch (e) { setError(e.message); }
    finally { setGeminiBusy(false); }
  }

  const reset = () => { setText(""); setFile(null); setResult(null); setGemini(null); setError(""); };
  const label = result?.prediction ? String(result.prediction).toUpperCase() : "MODEL NOT TRAINED";

  return <div className="shell">
    <header className="topbar">
      <a className="brand" href="#"><span className="logo">◉</span><span>truth<span>lens</span><small>NEWS INTELLIGENCE</small></span></a>
      <button className="outline" onClick={() => setHistoryOpen(!historyOpen)}>◷ History</button>
    </header>
    {historyOpen && <section className="history panel">
      <div className="row"><h3>Recent analyses</h3><button className="link" onClick={() => {localStorage.removeItem(HISTORY_KEY);setHistory([]);}}>Clear history</button></div>
      <p className="muted">Stored only in this browser, and only when you opt in.</p>
      {history.length ? history.map(item => <button className="history-item" key={item.id} onClick={() => {setText(item.excerpt);setFile(null);setResult(null);setHistoryOpen(false);}}><span>{item.title}<small>{new Date(item.date).toLocaleString()}</small></span><b>{item.prediction || "NO MODEL"}</b></button>) : <p className="muted">Nothing saved yet.</p>}
    </section>}

    <section className="hero">
      <div><div className="eyebrow">INDEPENDENT THINKING STARTS HERE</div>
        <h1>Look closer.<br/><em>Know more.</em></h1>
        <p>Inspect news text, discover suspicious wording, and build a clearer picture before you share.</p>
        <div className="proof"><span>✓ Explainable signals</span><span>✓ Evidence-first approach</span></div>
      </div>
      <div className="hero-graphic" aria-hidden="true"><div className="orbit"></div><div className="lens">⌕<small>VERIFY</small></div><div className="float-tag">● Pattern analysis <b>ACTIVE</b></div></div>
    </section>

    <main>
      <div className="section-title"><div><div className="eyebrow">01 / INPUT</div><h2>Put a story under the lens</h2></div><span className="muted">Files are processed in memory</span></div>
      <div className="grid">
        <section className="panel input-panel">
          <div className="tabs"><span className={!file ? "selected" : ""}>▤ Article text</span><span className={file ? "selected" : ""}>▧ PDF document</span></div>
          {!file ? <>
            <label htmlFor="article">ARTICLE OR CLAIM</label>
            <textarea id="article" value={text} maxLength={200000} onChange={e => {setText(e.target.value);setResult(null);setGemini(null);}} placeholder="Paste the full article or a specific claim you want to examine..."/>
            <div className="meta"><span>{text.length.toLocaleString()} / 200,000 characters</span><button className="link" onClick={reset}>Clear text</button></div>
            <div className="divider"><span>OR UPLOAD A DOCUMENT</span></div>
            <label className="upload" htmlFor="pdf"><strong>＋ Choose a PDF to analyze</strong><small>Selectable text · Up to 8 MB · 40 pages</small></label>
            <input id="pdf" type="file" accept=".pdf,application/pdf" hidden onChange={e => {const f=e.target.files?.[0];if(!f)return;if(!f.name.toLowerCase().endsWith(".pdf")){setError("Choose a PDF file.");return;}if(f.size>8*1024*1024){setError("PDF must be 8 MB or smaller.");return;}setFile(f);setText("");setResult(null);setError("");}}/>
          </> : <div className="chosen-file"><span>▧</span><div><strong>{file.name}</strong><small>{(file.size/1048576).toFixed(2)} MB PDF</small></div><button className="link" onClick={() => {setFile(null);setResult(null);}}>Remove</button></div>}
          <div className="actions"><label className="check"><input type="checkbox" checked={saveHistory} onChange={e => setSaveHistory(e.target.checked)}/> Save result in local browser history</label><button className="primary" disabled={busy || (!file && text.trim().length < 20)} onClick={analyze}>{busy ? "◌ Analyzing..." : "✦ Analyze story"}</button></div>
        </section>
        <aside className="panel how-panel"><div className="spark">✦</div><div className="eyebrow">HOW IT WORKS</div><h3>Signals, not snap judgments.</h3><p>TruthLens combines a text-classification baseline with transparent writing-style signals. Evidence retrieval is a planned next step.</p>
          <div className="step"><b>01</b><span><strong>Text patterns</strong><small>ML prediction, when trained</small></span></div>
          <div className="step"><b>02</b><span><strong>Language signals</strong><small>Flagged phrases and context</small></span></div>
          <div className="step"><b>03</b><span><strong>Independent evidence</strong><small>Human verification required</small></span></div>
          <div className="warning">⚠ A model score is not the probability that a claim is true.</div>
        </aside>
      </div>
      {error && <div className="error" role="alert">{error}<button onClick={() => setError("")}>×</button></div>}

      {result && <section className="results">
        <div className="section-title"><div><div className="eyebrow">02 / RESULTS</div><h2>What the analysis found</h2></div><span className="muted">Completed just now</span></div>
        <div className="result-grid">
          <article className="panel result-card"><div className="eyebrow">MODEL PREDICTION</div><h3 className={label === "REAL" ? "real" : label === "FAKE" ? "fake" : ""}>{label}</h3>
            {result.model_available ? <p className="muted">Learned label from the current dataset. Not a verified fact-check.</p> : <p className="muted">No trained model is loaded yet. Language signals are available below.</p>}
            {result.scores && Object.entries(result.scores).map(([key,value]) => <div className="score" key={key}><div><span>{key}</span><b>{(value*100).toFixed(1)}%</b></div><div className="bar"><i style={{width:Math.max(0,Math.min(100,value*100))+"%"}}/></div></div>)}
            {result.pdf && <p className="muted">PDF: {result.pdf.page_count} pages · {result.pdf.character_count.toLocaleString()} characters extracted</p>}
          </article>
          <article className="panel result-card"><div className="eyebrow">LANGUAGE SIGNALS · {(result.suspicious_language || []).length}</div><h3>{result.suspicious_language?.length ? "Potentially sensational phrasing" : "No listed phrases found"}</h3>
            <p className="muted">A phrase match is not proof of misinformation; this is a small rule-based list.</p>
            {(result.suspicious_language || []).map((flag,i) => <div className="signal" key={flag.phrase+i}><b>“{flag.phrase}”</b><small>{flag.category.replaceAll("_"," ")}</small><p>{flag.explanation}</p></div>)}
          </article>
        </div>
        <div className="warning result-warning"><strong>Interpret with care</strong><p>{result.caveat}</p></div>
        <div className="gemini-row"><div><strong>✦ Ask Gemini for contextual analysis</strong><small>Optional AI commentary, not independent verification.</small></div><button className="outline" disabled={geminiBusy} onClick={analyzeWithGemini}>{geminiBusy ? "Thinking..." : "Analyze with Gemini"}</button></div>
        {gemini && <article className="panel gemini-result"><div className="eyebrow">GEMINI COMMENTARY</div><p>{gemini.analysis}</p><small>{gemini.caveat}</small></article>}
      </section>}
    </main>
    <footer><div className="brand"><span className="logo">◉</span><span>truth<span>lens</span></span></div><p>Built for careful reading, not automatic verdicts.</p><small>LEARNING PROJECT · 0.1</small></footer>
  </div>;
}
