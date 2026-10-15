import { useEffect, useMemo, useState } from "react";
import { supabase } from "./lib/supabase.js";
import {
  Activity, AlertCircle, ArrowUpRight, Boxes, Check, ChevronDown, CircleHelp, Clock3,
  Database, Download, FileBarChart2, FlaskConical, GitCompareArrows, Home, LoaderCircle,
  LogOut, MessageSquareText, PlugZap, RefreshCw, Rocket, Send, Sparkles, Trophy, Upload,
} from "lucide-react";

const API = window.KATNYO_API_URL || `${location.protocol === "https:" ? "https:" : "http:"}//${location.hostname || "localhost"}:8000`;
const sessionId = sessionStorage.getItem("katnyo-session-id") || (() => {
  const id = crypto.randomUUID?.() || [...crypto.getRandomValues(new Uint8Array(16))].map((value) => value.toString(16).padStart(2, "0")).join("");
  sessionStorage.setItem("katnyo-session-id", id);
  return id;
})();
const nav = [
  ["Overview", Home], ["Train models", FlaskConical], ["Leaderboard", Trophy],
  ["Experiments", GitCompareArrows], ["Prediction", Rocket], ["Datasets", Database],
  ["Model registry", Boxes], ["AI data chat", MessageSquareText], ["Integrations", PlugZap],
];

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (supabase) {
    const { data } = await supabase.auth.getSession();
    if (data.session?.access_token) headers.set("Authorization", `Bearer ${data.session.access_token}`);
  }
  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response;
}

function prettyDate(value) {
  if (!value) return "Not available";
  return new Date(value).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

export default function App() {
  return supabase ? <AccountGate /> : <WorkspaceApp />;
}

function AccountGate() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    supabase.auth.getSession().then(({ data, error: sessionError }) => {
      if (!active) return;
      setSession(data.session);
      setError(sessionError?.message || "");
      setLoading(false);
    }).catch((sessionError) => {
      if (active) { setError(sessionError.message); setLoading(false); }
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setError("");
      setLoading(false);
    });
    return () => { active = false; subscription.unsubscribe(); };
  }, []);

  if (loading) return <div className="auth-screen"><span className="auth-mark"><Activity size={20} /></span><span className="auth-loading">KATNYO AI</span></div>;
  if (!session) return <AccountForm error={error} setError={setError} />;
  return <WorkspaceApp key={session.user.id} authUser={session.user} onSignOut={() => supabase.auth.signOut()} />;
}

function AccountForm({ error, setError }) {
  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function submit(event) {
    event.preventDefault(); setBusy(true); setError(""); setMessage("");
    try {
      const result = mode === "signin"
        ? await supabase.auth.signInWithPassword({ email, password })
        : await supabase.auth.signUp({ email, password });
      if (result.error) throw result.error;
      if (mode === "signup" && !result.data.session) setMessage("Check your email to confirm your account, then sign in.");
    } catch (authError) { setError(authError.message || "Authentication failed."); }
    finally { setBusy(false); }
  }

  return <main className="auth-screen">
    <section className="auth-panel" aria-labelledby="auth-title">
      <div className="auth-brand"><span className="auth-mark"><Activity size={20} /></span><span>KATNYO AI<small>STRUCTURED DATA LAB</small></span></div>
      <div className="auth-copy"><span className="auth-eyebrow">PRIVATE WORKSPACE</span><h1 id="auth-title">{mode === "signin" ? "Welcome back." : "Create your workspace."}</h1><p>{mode === "signin" ? "Sign in to access your datasets, experiments, and models." : "Your datasets, experiments, and models stay tied to your account."}</p></div>
      <form className="auth-form" onSubmit={submit}>
        <label>Email address<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
        <label>Password<input type="password" autoComplete={mode === "signin" ? "current-password" : "new-password"} minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
        {error && <p className="auth-message error" role="alert">{error}</p>}
        {message && <p className="auth-message" role="status">{message}</p>}
        <button className="auth-submit" type="submit" disabled={busy}>{busy ? <LoaderCircle className="spin" size={16} /> : null}{mode === "signin" ? "Sign in" : "Create account"}</button>
      </form>
      <p className="auth-switch">{mode === "signin" ? "New to KATNYO AI?" : "Already have an account?"}<button type="button" onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setError(""); setMessage(""); }}>{mode === "signin" ? "Create account" : "Sign in"}</button></p>
      <p className="auth-footnote">Your files and model artifacts are isolated to your account.</p>
    </section>
  </main>;
}

function WorkspaceApp({ authUser = null, onSignOut = null }) {
  const [page, setPage] = useState("Overview");
  const storageKey = authUser ? `katnyo-dataset:${authUser.id}` : "katnyo-dataset";
  const [datasetId, setDatasetId] = useState(localStorage.getItem(storageKey) || "");
  const [profile, setProfile] = useState(null);
  const [runs, setRuns] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);
  const [registry, setRegistry] = useState([]);
  const [integrations, setIntegrations] = useState(null);
  const [providerStatuses, setProviderStatuses] = useState({});
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [uploading, setUploading] = useState(false);
  const [target, setTarget] = useState("");
  const [revisionId, setRevisionId] = useState("");
  const [trials, setTrials] = useState(4);
  const [maxRows, setMaxRows] = useState(30000);
  const [revisions, setRevisions] = useState([]);
  const [compareIds, setCompareIds] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [selectedRun, setSelectedRun] = useState("");
  const [predictionModel, setPredictionModel] = useState("");
  const [predictionRows, setPredictionRows] = useState("[]");
  const [predictionResult, setPredictionResult] = useState(null);
  const [modelName, setModelName] = useState("");
  const [modelStage, setModelStage] = useState("staging");
  const [provider, setProvider] = useState("local");
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState([]);

  const targets = useMemo(() => {
    if (!profile) return [];
    return [...new Set([profile.target, ...profile.featureTypes.numeric, ...profile.featureTypes.categorical])].filter(Boolean);
  }, [profile]);

  async function refreshWorkspace(id = datasetId) {
    const [profileResponse, runResponse, boardResponse, registryResponse, integrationResponse, revisionResponse, jobsResponse] = await Promise.all([
      api(`/profile?dataset_id=${encodeURIComponent(id)}`).then((res) => res.json()),
      api(`/runs?dataset_id=${encodeURIComponent(id)}`).then((res) => res.json()),
      api(`/leaderboard?dataset_id=${encodeURIComponent(id)}`).then((res) => res.json()),
      api("/registry").then((res) => res.json()),
      api("/integrations").then((res) => res.json()),
      api(`/datasets/${encodeURIComponent(id)}/revisions`).then((res) => res.json()),
      api(`/train/jobs?dataset_id=${encodeURIComponent(id)}`).then((res) => res.json()),
    ]);
    setProfile(profileResponse.profile);
    setRuns(runResponse.runs);
    setLeaderboard(boardResponse.models);
    setRegistry(registryResponse.models);
    setIntegrations(integrationResponse);
    setRevisions(revisionResponse.revisions);
    setJob((current) => current || jobsResponse.jobs[0] || null);
    setTarget((current) => current || profileResponse.profile.target);
    setSelectedRun((current) => current || runResponse.runs[0]?.runId || "");
    setPredictionModel((current) => current || (boardResponse.models[0] ? `${boardResponse.models[0].runId}::${boardResponse.models[0].name}` : ""));
    setModelName((current) => current || runResponse.runs[0]?.bestModel?.name || "");
  }

  useEffect(() => {
    api("/integrations").then((res) => res.json()).then((value) => { setIntegrations(value); setProvider(value.chatProvider || "local"); }).catch((err) => setError(err.message));
    api(`/sessions/${encodeURIComponent(sessionId)}/providers`).then((res) => res.json()).then((value) => setProviderStatuses(value.providers || {})).catch(() => {});
    if (datasetId) refreshWorkspace(datasetId).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return undefined;
    setBusy(true);
    const timer = window.setTimeout(() => pollJob(job.jobId).catch((err) => { setBusy(false); setError(err.message); }), 1100);
    return () => window.clearTimeout(timer);
  }, [job?.jobId, job?.status]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(""), 3600);
    return () => clearTimeout(timer);
  }, [notice]);

  async function uploadFile(file) {
    if (!file) return;
    setUploading(true); setError("");
    const form = new FormData(); form.append("file", file);
    try {
      const response = await api("/upload", { method: "POST", body: form });
      const result = await response.json();
      setDatasetId(result.datasetId); localStorage.setItem(storageKey, result.datasetId);
      setRevisionId(""); setJob(null); setSelectedRun(""); setPredictionModel(""); setPredictionResult(null);
      setCompareIds([]); setComparison(null); setChat([]); setModelName("");
      setProfile(result.profile); setTarget(result.profile.target); setPage("Overview");
      await refreshWorkspace(result.datasetId);
      setNotice(`${file.name} is ready`);
    } catch (err) { setError(err.message); }
    finally { setUploading(false); }
  }

  async function startTraining() {
    setBusy(true); setError(""); setPage("Train models");
    try {
      const response = await api("/train/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ datasetId, revisionId: revisionId || undefined, target, optunaTrials: Number(trials), maxRows: Number(maxRows) }) });
      const firstJob = await response.json(); setJob(firstJob);
      setJob(firstJob);
    } catch (err) { setError(err.message); setBusy(false); }
  }

  async function pollJob(id) {
    const response = await api(`/train/jobs/${id}`); const current = await response.json(); setJob(current);
    if (current.status === "completed") {
      setBusy(false); setNotice("Training run completed");
      await refreshWorkspace(datasetId);
    } else if (current.status === "failed") {
      setBusy(false); setError(current.error || "Training failed");
    } else {
      window.setTimeout(() => pollJob(id).catch((err) => { setBusy(false); setError(err.message); }), 1100);
    }
  }

  async function loadRevisions() {
    const response = await api(`/datasets/${encodeURIComponent(datasetId)}/revisions`);
    setRevisions((await response.json()).revisions);
  }

  async function compareRuns() {
    const params = new URLSearchParams(); compareIds.forEach((id) => params.append("runIds", id));
    try { const response = await api(`/experiments/compare?${params}`); setComparison((await response.json()).runs); }
    catch (err) { setError(err.message); }
  }

  async function makePrediction(event) {
    event.preventDefault(); setError("");
    try {
      const [runId, modelName] = predictionModel.split("::");
      const response = await api(`/predict/${encodeURIComponent(runId)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ modelName, rows: JSON.parse(predictionRows) }) });
      setPredictionResult(await response.json());
    } catch (err) { setError(err.message); }
  }

  async function registerModel(event) {
    event.preventDefault(); setError("");
    try {
      const response = await api("/registry", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ runId: selectedRun, name: modelName, stage: modelStage }) });
      const created = await response.json(); setRegistry((items) => [created, ...items]); setNotice("Model added to registry");
    } catch (err) { setError(err.message); }
  }

  async function askQuestion(event) {
    event.preventDefault(); if (!question.trim()) return;
    const current = question; setQuestion(""); setChat((items) => [...items, { role: "user", text: current }]);
    try {
      const response = await api("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ datasetId, sessionId, question: current, provider }) });
      const answer = await response.json(); setChat((items) => [...items, { role: "assistant", text: answer.answer, provider: answer.provider }]);
    } catch (err) { setError(err.message); }
  }

  async function downloadReport(runId) {
    try {
      const response = await api(`/reports/${encodeURIComponent(runId)}`); const blob = await response.blob();
      const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `katnyo-report-${runId}.md`; link.click(); URL.revokeObjectURL(link.href);
    } catch (err) { setError(err.message); }
  }

  function openPage(name) {
    setPage(name); setError("");
    if (name === "Datasets" && datasetId) loadRevisions().catch((err) => setError(err.message));
  }

  const featureCount = profile ? profile.featureTypes.numeric.length + profile.featureTypes.categorical.length + profile.featureTypes.datetime.length : 0;

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Activity size={19} /></span><span>KATNYO<span className="brand-ai"> AI</span><small>STRUCTURED DATA LAB</small></span></div>
      <div className="workspace-label">WORKSPACE <button title="Workspace details"><CircleHelp size={14} /></button></div>
      <div className="workspace-select"><span className="workspace-dot" /> <span className="workspace-name">{authUser?.email || "Personal workspace"}</span></div>
      <nav className="nav-list" aria-label="Main navigation">
        {nav.map(([name, Icon]) => <button key={name} className={`nav-item ${page === name ? "active" : ""}`} onClick={() => openPage(name)}><Icon size={17} /><span>{name}</span>{name === "Train models" && busy && <i className="nav-pulse" />}</button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="connection"><span className="connection-light" /> API connection <span>LIVE</span></div>
        <div className="user-chip"><div className="avatar">{authUser?.email?.[0]?.toUpperCase() || "K"}</div><div><strong>{authUser?.email || "Local workspace"}</strong><small>{authUser ? "Private workspace" : "Local beta"}</small></div>{onSignOut && <button className="icon-button signout-button" title="Sign out" onClick={onSignOut}><LogOut size={15} /></button>}</div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="breadcrumbs">KATNYO AI <span>/</span> <strong>{page}</strong></div><div className="top-actions"><span className="api-status"><span /> API {integrations ? "CONNECTED" : "CONNECTING"}</span><button className="icon-button" title="Refresh workspace" onClick={() => datasetId && refreshWorkspace().catch((err) => setError(err.message))}><RefreshCw size={16} /></button><a className="top-upload" href="/" title="Manage provider session"><PlugZap size={15} /> Provider session</a><button className="top-upload" onClick={() => document.getElementById("file-upload").click()}><Upload size={15} /> Upload data</button><input id="file-upload" type="file" accept=".csv,text/csv" hidden onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; uploadFile(file); }} /></div></header>
      <div className="content">
        {!supabase && <div className="workspace-notice" role="status"><span className="workspace-notice-icon"><Database size={16} /></span><div className="workspace-notice-copy"><strong>LOCAL DEMO · SIGN-IN & PRIVATE STORAGE OFF</strong><span>Supabase is not configured, so customer accounts and per-user cloud storage are unavailable.</span></div><button type="button" onClick={() => openPage("Integrations")}>Account setup <ArrowUpRight size={14} /></button></div>}
        <div className="page-heading"><div><div className="eyebrow">AUTOMATED MACHINE LEARNING <span className="beta-tag">BETA</span></div><h1>{page}</h1><p>{datasetId && profile ? <><span className="dataset-indicator" /> {profile.filename} <span className="dot-sep">·</span> {profile.problemType} <span className="dot-sep">·</span> target <b>{profile.target}</b></> : "Structured data modeling workspace"}</p></div>{page === "Overview" && datasetId && <button className="button-primary" onClick={() => openPage("Train models")}><Sparkles size={16} /> Start a run</button>}</div>
        {error && <div className="alert error-alert"><AlertCircle size={17} /><span>{error}</span><button onClick={() => setError("")}>Dismiss</button></div>}
        {notice && <div className="alert success-alert"><Check size={17} /><span>{notice}</span></div>}
        {uploading && <div className="loading-line"><LoaderCircle className="spin" size={16} /> Profiling dataset…</div>}
        {!datasetId && page !== "Integrations" ? <EmptyWorkspace onUpload={uploadFile} uploading={uploading} /> : <>
          {page === "Overview" && <Overview profile={profile} runs={runs} leaderboard={leaderboard} job={job} onDownload={downloadReport} onNavigate={openPage} />}
          {page === "Train models" && <TrainView profile={profile} revisions={revisions} revisionId={revisionId} setRevisionId={setRevisionId} target={target} setTarget={setTarget} targets={targets} trials={trials} setTrials={setTrials} maxRows={maxRows} setMaxRows={setMaxRows} busy={busy} job={job} onTrain={startTraining} />}
          {page === "Leaderboard" && <Leaderboard models={leaderboard} onDownload={downloadReport} />}
          {page === "Experiments" && <Experiments runs={runs} selected={compareIds} setSelected={setCompareIds} comparison={comparison} onCompare={compareRuns} />}
          {page === "Prediction" && <PredictionView models={leaderboard} predictionModel={predictionModel} setPredictionModel={setPredictionModel} profile={profile} rows={predictionRows} setRows={setPredictionRows} result={predictionResult} onSubmit={makePrediction} />}
          {page === "Datasets" && <Datasets profile={profile} revisions={revisions} onReload={loadRevisions} />}
          {page === "Model registry" && <Registry registry={registry} models={leaderboard} selectedRun={selectedRun} setSelectedRun={setSelectedRun} modelName={modelName} setModelName={setModelName} modelStage={modelStage} setModelStage={setModelStage} onRegister={registerModel} />}
          {page === "AI data chat" && <Chat chat={chat} question={question} setQuestion={setQuestion} provider={provider} setProvider={setProvider} providerStatuses={providerStatuses} onSubmit={askQuestion} />}
          {page === "Integrations" && <Integrations integrations={integrations} providerStatuses={providerStatuses} />}
        </>}
        <footer className="page-footer"><span>KATNYO AI <span className="dot-sep">·</span> local beta</span><span>{datasetId ? `${featureCount} input features` : "No active dataset"}</span></footer>
      </div>
    </main>
  </div>;
}

function EmptyWorkspace({ onUpload, uploading }) {
  return <label className="dropzone"><input type="file" accept=".csv,text/csv" hidden onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; onUpload(file); }} /><span className="drop-icon"><Database size={22} /></span><strong>{uploading ? "Profiling your dataset…" : "Upload a dataset to begin"}</strong><span>CSV files up to your configured server limit</span><span className="button-primary"><Upload size={15} /> Choose CSV</span></label>;
}

function Overview({ profile, runs, leaderboard, job, onDownload, onNavigate }) {
  const best = runs[0]?.bestModel;
  return <>
    <section className="metric-grid">
      <Metric label="Dataset rows" value={profile?.rows?.toLocaleString() || "—"} sub={profile?.rowCountEstimated ? "sampled estimate" : "exact count"} icon={Database} />
      <Metric label="Columns" value={profile?.columns ?? "—"} sub={`${profile?.featureTypes?.numeric?.length || 0} numeric · ${profile?.featureTypes?.categorical?.length || 0} categorical`} icon={Activity} />
      <Metric label="Training runs" value={runs.length} sub={runs.length ? "saved to experiment history" : "No runs yet"} icon={FlaskConical} />
      <Metric label="Missing values" value={profile ? `${profile.missingPercent}%` : "—"} sub={profile ? `${profile.missingCells.toLocaleString()} sampled cells` : "Upload a dataset"} icon={FileBarChart2} />
    </section>
    {job && <section className="panel job-panel"><div className="section-heading"><div><span className="eyebrow">BACKGROUND TRAINING</span><h2>{job.status === "completed" ? "Latest job completed" : job.status === "failed" ? "Latest job failed" : "Training in progress"}</h2></div><span className={`status-pill ${job.status}`}>{job.status}</span></div><div className="progress-track"><i style={{ width: `${job.progress || 0}%` }} /></div><div className="job-meta"><span>{job.progress || 0}%</span><span>{job.runId || job.error || job.logs?.at(-1) || "Waiting for a worker"}</span></div></section>}
    <div className="overview-columns"><section className="panel"><div className="section-heading"><div><span className="eyebrow">MODEL PERFORMANCE</span><h2>{best ? "Best current run" : "Ready for your first run"}</h2></div>{runs.length > 0 && <button className="text-button" onClick={() => onNavigate("Leaderboard")}>View leaderboard <ArrowUpRight size={14} /></button>}</div>
      {best ? <div className="best-model"><div className="best-icon"><Trophy size={19} /></div><div className="best-name"><strong>{best.name}</strong><span>{runs[0].problemType} · {best.primaryMetric}</span></div><div className="best-score"><strong>{best.primaryValue}</strong><span>{best.primaryMetric}</span></div><button className="icon-button" title="Download report" onClick={() => onDownload(runs[0].runId)}><Download size={16} /></button></div> : <div className="empty-inline"><FlaskConical size={18} /> Upload a dataset and train a model to see results here.</div>}
      <div className="subsection-title">Recent experiments</div>{runs.slice(0, 4).map((run) => <div className="run-row" key={run.runId}><span className="run-mark"><Activity size={14} /></span><div><strong>{run.bestModel.name}</strong><small>{run.runId} · {prettyDate(run.createdAt)}</small></div><span className="run-value">{run.bestModel.primaryValue}</span></div>)}{!runs.length && <p className="muted-copy">Your training history will appear here.</p>}
    </section><section className="panel side-panel"><div className="section-heading"><div><span className="eyebrow">DATA PROFILE</span><h2>Dataset snapshot</h2></div><button className="icon-button" title="View dataset revisions" onClick={() => onNavigate("Datasets")}><ArrowUpRight size={16} /></button></div><div className="profile-line"><span>Detected task</span><strong className="capitalize">{profile?.problemType || "—"}</strong></div><div className="profile-line"><span>Prediction target</span><strong>{profile?.target || "—"}</strong></div><div className="profile-line"><span>Input features</span><strong>{profile ? profile.featureTypes.numeric.length + profile.featureTypes.categorical.length + profile.featureTypes.datetime.length : "—"}</strong></div><div className="profile-line"><span>Dataset version</span><strong className="mono">{profile?.filename || "—"}</strong></div><div className="quality-meter"><div><span>Data completeness</span><strong>{profile ? `${100 - profile.missingPercent}%` : "—"}</strong></div><div className="progress-track"><i style={{ width: `${profile ? 100 - profile.missingPercent : 0}%` }} /></div></div></section></div>
    <section className="panel quick-actions"><div className="section-heading"><div><span className="eyebrow">WORKSPACE</span><h2>Continue analysis</h2></div></div><div className="action-grid">{[[Trophy,"Compare model quality","Leaderboard"],[GitCompareArrows,"Review experiment runs","Experiments"],[Rocket,"Try sample predictions","Prediction"],[MessageSquareText,"Ask about this dataset","AI data chat"]].map(([Icon,label,page]) => <button key={page} className="action-item" onClick={() => onNavigate(page)}><Icon size={17} /><span>{label}</span><ArrowUpRight size={14} /></button>)}</div>{leaderboard.length === 0 && <span className="muted-copy">No model scores yet</span>}</section>
  </>;
}

function Metric({ label, value, sub, icon: Icon }) { return <div className="metric-card"><div className="metric-top"><span>{label}</span><Icon size={16} /></div><strong>{value}</strong><small>{sub}</small></div>; }

function TrainView({ profile, revisions, revisionId, setRevisionId, target, setTarget, targets, trials, setTrials, maxRows, setMaxRows, busy, job, onTrain }) {
  return <div className="train-layout"><section className="panel train-panel"><div className="section-heading"><div><span className="eyebrow">NEW EXPERIMENT</span><h2>Configure training</h2><p>Choose the dataset version, target, and search budget.</p></div><FlaskConical className="section-icon" size={20} /></div><label className="field-label">Dataset version<select value={revisionId} onChange={(e) => setRevisionId(e.target.value)}><option value="">Latest revision</option>{revisions.map((item) => <option key={item.revisionId} value={item.revisionId}>{item.revisionId} · {prettyDate(item.createdAt)}</option>)}</select></label><label className="field-label">Prediction target<select value={target} onChange={(e) => setTarget(e.target.value)}>{targets.map((item) => <option key={item}>{item}</option>)}</select></label><div className="field-row"><label className="field-label">Optuna trials<input type="number" min="0" max="100" value={trials} onChange={(e) => setTrials(e.target.value)} /></label><label className="field-label">Maximum rows<input type="number" min="200" value={maxRows} onChange={(e) => setMaxRows(e.target.value)} /></label></div><div className="train-context"><span><Database size={15} /> {profile?.filename}</span><span>{profile?.problemType} · {profile?.rows?.toLocaleString()} rows</span></div><button className="button-primary full-button" disabled={busy || !profile} onClick={onTrain}>{busy ? <><LoaderCircle className="spin" size={16} /> Training…</> : <><Sparkles size={16} /> Start background training</>}</button></section>
    <section className="panel job-details"><span className="eyebrow">LIVE JOB STATUS</span><div className="job-title"><h2>{job?.status === "running" ? "Training models" : job?.status === "completed" ? "Run complete" : job?.status === "failed" ? "Training error" : "No active job"}</h2>{job && <span className={`status-pill ${job.status}`}>{job.status}</span>}</div><div className="progress-track"><i style={{ width: `${job?.progress || 0}%` }} /></div><div className="job-meta"><span>{job?.progress || 0}%</span><span>{job?.runId || "Job progress will appear here"}</span></div><div className="log-list">{(job?.logs || ["Select a dataset and start a run to see training progress."]).slice(-9).map((line, index) => <div className="log-entry" key={`${index}-${line}`}><Clock3 size={13} /><span>{line}</span></div>)}</div></section></div>;
}

function Leaderboard({ models, onDownload }) {
  return <section className="panel table-panel"><div className="section-heading"><div><span className="eyebrow">ALL EVALUATED CANDIDATES</span><h2>Model leaderboard</h2><p>Sorted by the normalized primary score across saved runs.</p></div><span className="count-tag">{models.length} models</span></div><div className="table-wrap"><table><thead><tr><th>Rank</th><th>Model</th><th>Run</th><th>Type</th><th>Primary metric</th><th>Score</th><th>Metrics</th><th /></tr></thead><tbody>{models.map((model, index) => <tr key={`${model.runId}-${model.name}`}><td><span className={`rank-number ${index === 0 ? "rank-first" : ""}`}>{String(index + 1).padStart(2, "0")}</span></td><td><strong>{model.name}</strong></td><td className="mono">{model.runId}</td><td><span className="type-tag">{model.type}</span></td><td>{model.primaryMetric}</td><td><strong>{model.primaryValue}</strong></td><td className="metric-detail">{Object.entries(model.metrics || {}).map(([name, value]) => `${name} ${value}`).join(" · ")}</td><td><button className="icon-button" title="Download run report" onClick={() => onDownload(model.runId)}><Download size={15} /></button></td></tr>)}</tbody></table>{models.length === 0 && <div className="empty-inline">No trained models yet. Start a background training job to populate the leaderboard.</div>}</div></section>;
}

function Experiments({ runs, selected, setSelected, comparison, onCompare }) {
  const toggle = (id) => setSelected((items) => items.includes(id) ? items.filter((entry) => entry !== id) : [...items, id]);
  return <div className="experiment-layout"><section className="panel"><div className="section-heading"><div><span className="eyebrow">RUN HISTORY</span><h2>Select runs to compare</h2><p>Choose two or more runs to compare their best models and scores.</p></div><GitCompareArrows size={20} className="section-icon" /></div><div className="run-select-list">{runs.map((run) => <label className="run-select-row" key={run.runId}><input type="checkbox" checked={selected.includes(run.runId)} onChange={() => toggle(run.runId)} /><span className="run-mark"><Activity size={14} /></span><span className="run-info"><strong>{run.bestModel.name}</strong><small>{run.runId} · {prettyDate(run.createdAt)}</small></span><span className="run-value">{run.bestModel.primaryMetric} {run.bestModel.primaryValue}</span></label>)}{runs.length === 0 && <div className="empty-inline">No runs are saved yet.</div>}</div><button className="button-primary" disabled={selected.length < 2} onClick={onCompare}><GitCompareArrows size={16} /> Compare {selected.length || ""} runs</button></section>
    {comparison && <section className="panel table-panel comparison-panel"><div className="section-heading"><div><span className="eyebrow">SIDE BY SIDE</span><h2>Experiment comparison</h2></div></div><div className="table-wrap"><table><thead><tr><th>Run</th><th>Target</th><th>Problem</th><th>Best model</th><th>Metric</th><th>Value</th></tr></thead><tbody>{comparison.map((run) => <tr key={run.runId}><td className="mono">{run.runId}</td><td>{run.target}</td><td>{run.problemType}</td><td>{run.bestModel.name}</td><td>{run.bestModel.primaryMetric}</td><td><strong>{run.bestModel.primaryValue}</strong></td></tr>)}</tbody></table></div></section>}
  </div>;
}

function PredictionView({ models, predictionModel, setPredictionModel, profile, rows, setRows, result, onSubmit }) {
  const example = { ...(profile?.featureTypes?.numeric || []).reduce((all, col) => ({ ...all, [col]: 0 }), {}), ...(profile?.featureTypes?.categorical || []).reduce((all, col) => ({ ...all, [col]: "example" }), {}), ...(profile?.featureTypes?.datetime || []).reduce((all, col) => ({ ...all, [col]: "2025-01-01" }), {}) };
  return <div className="playground-layout"><section className="panel playground-input"><div className="section-heading"><div><span className="eyebrow">MODEL INFERENCE</span><h2>Prediction playground</h2><p>Send one or more JSON feature rows to a saved model.</p></div><Rocket className="section-icon" size={20} /></div><label className="field-label">Trained model<select value={predictionModel} onChange={(e) => setPredictionModel(e.target.value)}>{models.map((model) => <option key={`${model.runId}-${model.name}`} value={`${model.runId}::${model.name}`}>{model.runId} · {model.name}</option>)}</select></label><div className="editor-heading"><label className="field-label">Input rows <span>JSON array</span></label><button className="text-button" onClick={() => setRows(JSON.stringify([example], null, 2))}>Load feature template</button></div><textarea className="code-input" spellCheck="false" value={rows} onChange={(e) => setRows(e.target.value)} placeholder={'[{"feature": 12}]'} /><button className="button-primary" disabled={!predictionModel || !models.length} onClick={onSubmit}><Rocket size={15} /> Run prediction</button></section><section className="panel prediction-output"><span className="eyebrow">OUTPUT</span><h2>Model response</h2>{result ? <pre className="result-json">{JSON.stringify(result, null, 2)}</pre> : <div className="prediction-empty"><Activity size={19} /><span>Prediction output will appear here</span></div>}</section></div>;
}

function Datasets({ profile, revisions, onReload }) {
  return <section className="panel table-panel"><div className="section-heading"><div><span className="eyebrow">DATA LINEAGE</span><h2>Dataset versions</h2><p>{profile?.filename} · {revisions.length} saved revision{revisions.length === 1 ? "" : "s"}</p></div><button className="icon-button" title="Refresh revisions" onClick={onReload}><RefreshCw size={16} /></button></div><div className="table-wrap"><table><thead><tr><th>Revision</th><th>Created</th><th>Size</th><th>Target profile</th><th>Profile rows</th></tr></thead><tbody>{revisions.map((revision) => <tr key={revision.revisionId}><td className="mono">{revision.revisionId}</td><td>{prettyDate(revision.createdAt)}</td><td>{(revision.sizeBytes / 1024).toFixed(1)} KB</td><td>{revision.profile?.target || "—"}</td><td>{revision.profile?.sampleRows?.toLocaleString() || "—"}</td></tr>)}</tbody></table>{revisions.length === 0 && <div className="empty-inline">Revision history will appear here after upload.</div>}</div><p className="footnote">Upload an updated CSV with the same filename to add a revision. Select a revision on the training form before training.</p></section>;
}

function Registry({ registry, models, selectedRun, setSelectedRun, modelName, setModelName, modelStage, setModelStage, onRegister }) {
  const uniqueRuns = [...new Map(models.map((item) => [item.runId, item])).values()];
  return <div className="registry-layout"><section className="panel register-panel"><div className="section-heading"><div><span className="eyebrow">MODEL LIFECYCLE</span><h2>Register a model</h2></div><Boxes className="section-icon" size={20} /></div><form onSubmit={onRegister}><label className="field-label">Training run<select value={selectedRun} onChange={(e) => { setSelectedRun(e.target.value); setModelName(uniqueRuns.find((run) => run.runId === e.target.value)?.name || ""); }}>{uniqueRuns.map((run) => <option value={run.runId} key={run.runId}>{run.runId} · {run.name}</option>)}</select></label><label className="field-label">Registry name<input value={modelName} onChange={(e) => setModelName(e.target.value)} required placeholder="Demand model v1" /></label><label className="field-label">Lifecycle stage<select value={modelStage} onChange={(e) => setModelStage(e.target.value)}><option value="staging">Staging</option><option value="production">Production</option><option value="archived">Archived</option></select></label><button className="button-primary" disabled={!uniqueRuns.length}><Boxes size={15} /> Add to registry</button></form></section><section className="panel registry-list"><div className="section-heading"><div><span className="eyebrow">REGISTERED ARTIFACTS</span><h2>Model registry <span className="count-tag">{registry.length}</span></h2></div></div>{registry.map((item) => <div className="registry-row" key={item.modelId}><span className="registry-icon"><Boxes size={16} /></span><div><strong>{item.name}</strong><small>{item.modelId} · {item.runId}</small></div><span className={`stage-tag ${item.stage}`}>{item.stage}</span><span className="registry-metric">{Object.entries(item.metrics || {}).map(([key, value]) => `${key}: ${value}`).join(" · ")}</span></div>)}{registry.length === 0 && <div className="empty-inline">Registered models will appear here.</div>}</section></div>;
}

function Chat({ chat, question, setQuestion, provider, setProvider, providerStatuses, onSubmit }) {
  const options = [["local","Local profile"],["openai","OpenAI"],["azure-openai","Azure OpenAI"],["gemini","Google Gemini"],["anthropic","Anthropic Claude"],["huggingface","Hugging Face"],["cohere","Cohere"]];
  return <section className="panel chat-panel"><div className="section-heading"><div><span className="eyebrow">DATASET ASSISTANT</span><h2>Ask your dataset</h2><p>Questions use the uploaded profile; configured LLMs receive profile metadata only.</p></div><Sparkles className="section-icon" size={20} /></div><label className="chat-provider">AI provider<select value={provider} onChange={(e) => setProvider(e.target.value)}>{options.map(([id,label]) => { const status = providerStatuses[id]; const state = id === "local" ? "" : status?.verified ? " · verified" : status?.connected ? " · saved" : " · not linked"; return <option key={id} value={id}>{label}{state}</option>; })}</select><a className="button-quiet" href="/">Manage provider keys</a></label><div className="chat-thread">{chat.length === 0 && <div className="chat-intro"><span className="chat-orb"><MessageSquareText size={20} /></span><strong>What would you like to know?</strong><span>Try “Which columns are incomplete?” or “What target did you detect?”</span></div>}{chat.map((message, index) => <div className={`chat-message ${message.role}`} key={index}><span className="chat-avatar">{message.role === "user" ? "Y" : "K"}</span><div><strong>{message.role === "user" ? "You" : `KATNYO AI${message.provider ? ` · ${message.provider}` : ""}`}</strong><p>{message.text}</p></div></div>)}</div><form className="chat-form" onSubmit={onSubmit}><input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question about this dataset…" /><button className="button-primary" type="submit" disabled={!question.trim()} title="Send question"><Send size={16} /></button></form></section>;
}

function Integrations({ integrations, providerStatuses }) {
  if (!integrations) return <div className="loading-line"><LoaderCircle className="spin" size={16} /> Loading integration status…</div>;
  const aiProviders = [["OpenAI", "openai"], ["Azure OpenAI", "azure-openai"], ["Google Gemini", "gemini"], ["Anthropic Claude", "anthropic"], ["Hugging Face Inference", "huggingface"], ["Cohere", "cohere"]];
  const accounts = integrations.accounts || {};
  const groups = [
    ["Accounts & storage", [["Supabase Auth", accounts.supabaseConfigured ? "Enabled · sign in at this console" : "Off · add project URL and publishable key, then rebuild", accounts.supabaseConfigured], ["Private CSV storage", accounts.storageCredentialsConfigured ? "Project configured · private bucket and policies required" : "Off · create a private bucket and apply user-scoped policies", accounts.storageCredentialsConfigured]]],
    ["AI & ML APIs", aiProviders.map(([name, key]) => { const status = providerStatuses[key]; return [name, status?.verified ? "Session key verified" : status?.connected ? "Key saved · not verified" : "Connect in provider session", Boolean(status?.verified)]; })],
    ["Tracking & optimization", [["MLflow", integrations.tracking.mlflow ? "Installed and enabled" : "Not installed", integrations.tracking.mlflow], ["Optuna", integrations.automl.optuna ? "Installed and enabled" : "Not installed", integrations.automl.optuna], ["Weights & Biases", integrations.tracking.wandbCredentialsConfigured && integrations.tracking.wandbPackageInstalled ? "Logger ready" : integrations.tracking.wandbCredentialsConfigured ? "Credentials set · install optional logger" : "No credentials", integrations.tracking.wandbCredentialsConfigured && integrations.tracking.wandbPackageInstalled], ["Neptune", integrations.tracking.neptuneCredentialsConfigured && integrations.tracking.neptunePackageInstalled ? "Logger ready" : integrations.tracking.neptuneCredentialsConfigured ? "Credentials set · install optional logger" : "No credentials", integrations.tracking.neptuneCredentialsConfigured && integrations.tracking.neptunePackageInstalled], ["Ray Tune", integrations.automl.rayTuneConfigured ? "Flag set · adapter not implemented" : "Not configured", false], ["H2O AutoML", integrations.automl.h2oAutoMLConfigured ? "Flag set · adapter not implemented" : "Not configured", false]]],
    ["Cloud ML & forecasting", Object.entries(integrations.cloud || {}).map(([key, ready]) => [key.replace(/(CredentialsConfigured|ProjectConfigured)/, "").replace(/([A-Z])/g, " $1").trim(), ready ? key === "vertexAIProjectConfigured" ? "Project and location set · adapter not implemented" : "Credentials set · adapter not implemented" : "Not configured", false])],
  ];
  return <div className="integration-grid">{groups.map(([title, items]) => <section className="panel integration-panel" key={title}><div className="section-heading"><div><span className="eyebrow">PROVIDER CONNECTIONS</span><h2>{title}</h2></div><PlugZap className="section-icon" size={20} /></div>{items.map(([name, status, active]) => <div className="integration-row" key={name}><span className={`integration-state ${active ? "ready" : "pending"}`}>{active ? <Check size={13} /> : <span />}</span><strong>{name}</strong><small>{status}</small></div>)}{title === "Accounts & storage" && <p className="integration-help">Set SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, VITE_SUPABASE_URL, and VITE_SUPABASE_PUBLISHABLE_KEY in the root .env file. Create the private katnyo-user-data bucket, run supabase/storage_policies.sql in the Supabase SQL editor, then rebuild with docker compose up --build -d. Sign up and sign in are available here after configuration.</p>}</section>)}</div>;
}
