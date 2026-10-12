import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Camera, ScanLine, ArrowUpRight, Download, CircleCheck, CircleX, SlidersHorizontal, ShieldCheck, Activity, Play, Square, Check, AlertTriangle } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import { classify, modelBase, csv } from './inspection.mjs';
import './style.css';

type Prediction = {className:string; probability:number};
type Reading = {verdict:string;confidence:number;label:string};
type RecordRow = Reading & {time:number;modelUrl:string};
const historyKey = 'partlens-history-v1';
function readHistory(): RecordRow[] { try { const data = JSON.parse(localStorage.getItem(historyKey) || '[]'); return Array.isArray(data) ? data.filter(r => Number.isFinite(r.time) && Number.isFinite(r.confidence) && ['GO','NO-GO','REVIEW'].includes(r.verdict) && typeof r.label === 'string' && typeof r.modelUrl === 'string').slice(-1000) : []; } catch { return []; } }
function App() {
  const [modelUrl,setModelUrl] = useState(new URLSearchParams(location.search).get('model') || 'https://teachablemachine.withgoogle.com/models/fAuwZbUNp/');
  const [loadedUrl,setLoadedUrl] = useState('');
  const [classes,setClasses] = useState<string[]>([]);
  const [correct,setCorrect] = useState(''); const [incorrect,setIncorrect] = useState('');
  const [threshold,setThreshold] = useState(90); const [loading,setLoading] = useState(false); const [mirror,setMirror] = useState(true);
  const [camera,setCamera] = useState(false); const [cameraBusy,setCameraBusy] = useState(false);
  const [predictions,setPredictions] = useState<Prediction[]>([]); const [error,setError] = useState('');
  const [auto,setAuto] = useState(false); const [interval,setIntervalValue] = useState(3);
  const [records,setRecords] = useState<RecordRow[]>(readHistory); const [storageWarning,setStorageWarning] = useState('');
  const video = useRef<HTMLVideoElement>(null); const model = useRef<tf.LayersModel|null>(null);
  const stream = useRef<MediaStream|null>(null); const labels = useRef<string[]>([]); const size = useRef(224);
  const lastPrediction = useRef(0); const lastRecord = useRef(0); const cameraStarting = useRef(false);
  const mounted = useRef(true);
  const reading: Reading = classify(predictions,correct,incorrect,threshold/100);
  const active = camera && classes.length > 0 && predictions.length > 0;
  const current = useRef({reading,active,loadedUrl}); current.current = {reading,active,loadedUrl};
  useEffect(() => { return () => { mounted.current = false; stream.current?.getTracks().forEach(t=>t.stop()); model.current?.dispose(); }; }, []);
  useEffect(() => { try { localStorage.setItem(historyKey,JSON.stringify(records)); } catch { setStorageWarning('Browser storage is unavailable. Export CSV to keep this session.'); } }, [records]);
  function record() { const value = current.current; if (!value.active || performance.now()-lastPrediction.current > 2000) return; setRecords(r=>[...r,{...value.reading,time:Date.now(),modelUrl:value.loadedUrl}].slice(-1000)); lastRecord.current = performance.now(); }
  useEffect(() => { if (!auto) return; lastRecord.current = performance.now(); const timer = window.setInterval(()=> { if (performance.now()-lastRecord.current >= interval*1000) record(); },250); return ()=>window.clearInterval(timer); }, [auto,interval]);
  useEffect(() => {
    if (!camera || !classes.length) return;
    let stopped = false; let timer:number;
    async function predict() {
      try {
        const v=video.current; const m=model.current;
        if (!v || !m || v.readyState < 2 || !v.videoWidth) return;
        const canvas=document.createElement('canvas');canvas.width=size.current;canvas.height=size.current;
        const ctx=canvas.getContext('2d');if(!ctx) throw new Error('Canvas is unavailable.');
        const side=Math.min(v.videoWidth,v.videoHeight);
        if(mirror) {ctx.translate(size.current,0);ctx.scale(-1,1);}
        ctx.drawImage(v,(v.videoWidth-side)/2,(v.videoHeight-side)/2,side,side,0,0,size.current,size.current);
        const input = tf.tidy(()=>tf.browser.fromPixels(canvas).toFloat().div(127.5).sub(1).expandDims(0));
        let output:tf.Tensor|tf.Tensor[]|undefined;
        try { output=m.predict(input) as tf.Tensor; const values=await (Array.isArray(output)?output[0]:output).data(); if (!stopped) { lastPrediction.current=performance.now(); setPredictions(labels.current.map((className,i)=>({className,probability:Number(values[i])}))); } }
        finally { input.dispose(); if(output) tf.dispose(output); }
      } catch(e) { if(!stopped) { setError('Inference failed: '+(e as Error).message); setPredictions([]); setAuto(false); stopped=true; } }
      finally { if(!stopped) timer=window.setTimeout(predict,150); }
    }
    void predict(); return ()=>{stopped=true;window.clearTimeout(timer);};
  }, [camera,classes,mirror]);
  async function loadModel() {
    setLoading(true);setError('');setAuto(false);setPredictions([]);setClasses([]);model.current?.dispose();model.current=null;setLoadedUrl('');
    let candidate:tf.LayersModel|null=null;
    try {
      const base=modelBase(modelUrl); const response=await fetch(base+'metadata.json');
      if(!response.ok) throw new Error('Metadata request returned '+response.status);
      const metadata=await response.json();
      if(!Array.isArray(metadata.labels) || metadata.labels.length<2 || !metadata.labels.every((x:unknown)=>typeof x==='string') || new Set(metadata.labels).size!==metadata.labels.length) throw new Error('An image model with at least two unique class labels is required.');
      await tf.ready(); candidate=await tf.loadLayersModel(base+'model.json');
      const shape=candidate.inputs[0].shape; const imageSize=Number(shape[1] || metadata.imageSize || 224);
      if(shape.length!==4 || shape[3]!==3 || shape[2]!==imageSize || candidate.outputs[0].shape.at(-1)!==metadata.labels.length) throw new Error('This export is not a supported Teachable Machine image model.');
      size.current=imageSize; model.current=candidate;candidate=null;labels.current=metadata.labels;
      setClasses(metadata.labels);setLoadedUrl(base);
      setCorrect(metadata.labels.find((s:string)=>/^(correct|good|pass|go|ok)$/i.test(s.trim())) || metadata.labels[0]);
      setIncorrect(metadata.labels.find((s:string)=>/^(incorrect|bad|fail|no[ -]?go|defect)$/i.test(s.trim())) || metadata.labels[1]);
    } catch(e) {candidate?.dispose();setError('Could not load model. Check the exported image-model URL and browser access. '+(e as Error).message);} finally {setLoading(false);}
  }
  function stopCamera() {stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;if(video.current) video.current.srcObject=null;setCamera(false);setPredictions([]);setAuto(false);}
  async function startCamera() {
    if(cameraStarting.current) return; cameraStarting.current=true;setCameraBusy(true);setError('');
    try {
      if(!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires HTTPS or localhost.');
      const media=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}},audio:false});
      if(!mounted.current) {media.getTracks().forEach(t=>t.stop());return;}
      stream.current=media;
      if(video.current) {video.current.srcObject=media;await video.current.play();}
      media.getVideoTracks()[0].addEventListener('ended',stopCamera,{once:true});setCamera(true);
    } catch(e) {stopCamera();setError('Camera unavailable. Allow camera access and close other apps using it. '+(e as Error).message);} finally {cameraStarting.current=false;setCameraBusy(false);}
  }
  const go=records.filter(r=>r.verdict==='GO').length;const noGo=records.filter(r=>r.verdict==='NO-GO').length;
  const recent=records.slice(-40);const first=recent[0]?.time || 0;const span=Math.max(1,(recent.at(-1)?.time || 0)-first);
  const point=(r:RecordRow,i:number)=>`${recent.length===1?300:40+(r.time-first)/span*550},${140-r.confidence*110}`;
  const verdict=active?reading.verdict:'WAITING';
  return <div className="app">
    <aside className="sidebar"><a className="brand" href="./"><span className="brand-icon"><ScanLine size={23}/></span>partlens<span className="brand-dot">®</span></a><div className="workspace-label">WORKSPACE</div><div className="nav-item"><ScanLine size={18}/> Visual inspection <span className="nav-dot"/></div><div className="sidebar-bottom"><ShieldCheck size={20}/><div>Private by design<small>Camera processing stays local</small></div></div></aside>
    <div className="main"><header><div className="breadcrumb">Workspace <span>/</span> Visual inspection</div><div className="session"><span className="dot"/> {active?'Inspection live':'Station ready'} <span className="avatar">OP</span></div></header>
    <main><div className="page-heading"><div><div className="eyebrow">QUALITY CONTROL / STATION 01</div><h1>Every part. A clear decision.</h1><p>Real-time visual inspection, powered by your trained model.</p></div><button className="button secondary" disabled={!records.length} onClick={()=>{const blob=new Blob([csv(records)],{type:'text/csv;charset=utf-8;'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='partlens-inspections.csv';a.click();window.setTimeout(()=>URL.revokeObjectURL(url),1000);}}><Download size={16}/> Export data</button></div>
    {error && <div role="alert" className="alert"><AlertTriangle size={18}/>{error}</div>}{storageWarning && <div role="alert" className="alert">{storageWarning}</div>}
    <div className="top-grid"><section className="card camera-card"><div className="card-heading"><h2><Camera size={17}/> Live camera</h2><span className="tag">{camera?'LIVE FEED':'CAMERA OFF'}</span></div><div className={'camera-view '+(camera?'on':'')}><video ref={video} style={{transform:mirror?'scaleX(-1)':undefined}} playsInline muted aria-label="Live inspection camera"/><div className="camera-grid"/><div className="target"><i/><i/><i/><i/>{!camera && <div className="camera-placeholder"><Camera size={35}/><h3>Your inspection starts here</h3><p>Connect your model, then enable your camera.<br/>Position one part inside the guide.</p><button className="button light" disabled={cameraBusy} onClick={startCamera}><Camera size={16}/>{cameraBusy?'Connecting…':'Enable camera'}</button></div>}</div><div className="feed-caption"><span className="dot"/> {camera?'LOCAL CAMERA · CENTER CROP':'AWAITING CAMERA CONNECTION'}<span>01</span></div></div><div className="camera-footer"><span><ShieldCheck size={15}/> Frames never leave your device</span>{camera && <button className="text-button" onClick={stopCamera}><Square size={12}/> Stop camera</button>}</div></section>
    <section className={'card decision-card '+verdict.toLowerCase()}><div className="card-heading"><h2>Inspection result</h2><span className="tag">{active?'LIVE':'STANDBY'}</span></div><div className="decision-main"><div className="result-icon">{verdict==='GO'?<Check size={45}/>:verdict==='NO-GO'?<CircleX size={45}/>:verdict==='REVIEW'?<AlertTriangle size={39}/>:<ScanLine size={42}/>}</div><div className="eyebrow">{active?'CURRENT CLASSIFICATION':'READY WHEN YOU ARE'}</div><h2 className="verdict" aria-live="polite">{verdict}</h2><p>{verdict==='GO'?'Part meets your model’s criteria':verdict==='NO-GO'?'Part flagged by your model':verdict==='REVIEW'?'Uncertain result — check this part':'Load a model to begin inspection'}</p></div><div className="confidence"><div><span>Model confidence</span><strong>{active?(reading.confidence*100).toFixed(1)+'%':'—'}</strong></div><div className="progress"><span style={{width:active?reading.confidence*100+'%':'0%'}}/></div><small>Decision threshold <b>{threshold}%</b></small></div><button className="button primary record" disabled={!active || correct===incorrect} onClick={record}><CircleCheck size={17}/> Record inspection</button></section></div>
    <div className="stats"><div className="stat"><span>Total observations</span><strong>{records.length}<small>Recorded inspections</small></strong><Activity size={21}/></div><div className="stat"><span>GO results</span><strong>{go}<small className="green">Accepted</small></strong><CircleCheck size={21}/></div><div className="stat"><span>NO-GO results</span><strong>{noGo}<small className="red">Flagged for review</small></strong><CircleX size={21}/></div><div className="stat"><span>Acceptance rate</span><strong>{go+noGo?Math.round(go/(go+noGo)*100)+'%':'—'}<small>Decisive observations only</small></strong><ArrowUpRight size={21}/></div></div>
    <div className="bottom-grid"><section className="card chart-card"><div className="card-heading"><div><h2>Inspection timeline</h2><p>Confidence over time · last 40 observations</p></div><div className="legend"><span className="green">● GO</span><span className="red">● NO-GO</span><span>● Review</span></div></div><div className="chart"><svg viewBox="0 0 620 180" role="img" aria-label={`Inspection confidence timeline with ${recent.length} observations`}><title>Recorded inspection confidence over time</title>{[0,50,100].map(n=><g key={n}><text x="0" y={145-n*1.1}>{n}%</text><line x1="38" x2="605" y1={140-n*1.1} y2={140-n*1.1}/></g>)}<line className="threshold-line" x1="38" x2="605" y1={140-threshold*1.1} y2={140-threshold*1.1}/>{recent.length>1 && <polyline points={recent.map(point).join(' ')} fill="none" stroke="#78998c" strokeWidth="2"/>}{recent.map((r,i)=>{const [x,y]=point(r,i).split(',');return <circle key={i} cx={x} cy={y} r="4" fill={r.verdict==='GO'?'#278162':r.verdict==='NO-GO'?'#cf6559':'#d6a04d'}><title>{new Date(r.time).toLocaleTimeString()} · {r.verdict} · {(r.confidence*100).toFixed(1)}%</title></circle>;})}<text x="38" y="173">{recent.length?new Date(first).toLocaleTimeString():'Session start'}</text><text x="600" y="173" textAnchor="end">{recent.length?new Date(recent.at(-1)!.time).toLocaleTimeString():'Now'}</text></svg>{!recent.length && <div className="empty-chart"><Activity size={20}/><span>Your next inspection starts the story</span><small>Recorded results will appear here.</small></div>}</div><div className="chart-footer"><span><span className="dot"/> Stored in this browser · up to 1,000 observations</span><button className="text-button" disabled={!records.length} onClick={()=>{if(window.confirm('Clear all recorded inspections from this browser? Export CSV first if you need to keep them.'))setRecords([]);}}>Clear history</button></div></section>
    <section className="card settings-card"><div className="card-heading"><h2><SlidersHorizontal size={17}/> Station settings</h2><span className="tag">TEACHABLE MACHINE</span></div><label htmlFor="model-url">Image model URL</label><div className="model-input"><input id="model-url" type="url" placeholder="https://teachablemachine.withgoogle.com/models/…/" value={modelUrl} onChange={e=>setModelUrl(e.target.value)}/><button className="button primary" disabled={loading || !modelUrl.trim()} onClick={loadModel}>{loading?'Loading…':'Load'}</button></div><small className="help">Export → TensorFlow.js → Upload your model → Copy URL.</small>{loadedUrl && <div className="model-ready"><Check size={13}/> Model connected · {classes.length} classes</div>}<div className="class-grid"><label>GO class<select aria-label="GO class" value={correct} disabled={!classes.length} onChange={e=>{setCorrect(e.target.value);setAuto(false);}}>{!classes.length && <option>Not connected</option>}{classes.map(c=><option key={c}>{c}</option>)}</select></label><label>NO-GO class<select aria-label="NO-GO class" value={incorrect} disabled={!classes.length} onChange={e=>{setIncorrect(e.target.value);setAuto(false);}}>{!classes.length && <option>Not connected</option>}{classes.map(c=><option key={c}>{c}</option>)}</select></label></div>{classes.length>0 && correct===incorrect && <small className="red">Choose different GO and NO-GO classes.</small>}<div className="threshold-label"><label htmlFor="threshold">Confidence threshold</label><strong>{threshold}%</strong></div><input id="threshold" type="range" min="50" max="100" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/><div className="auto-row"><div><label htmlFor="mirror">Mirror camera</label><small>Matches Teachable Machine’s default webcam</small></div><input id="mirror" type="checkbox" role="switch" checked={mirror} onChange={e=>{setMirror(e.target.checked);setPredictions([]);}}/></div><div className="auto-row"><div><label htmlFor="auto">Timed recording</label><small>One observation every <select aria-label="Recording interval" value={interval} onChange={e=>setIntervalValue(Number(e.target.value))}>{[1,3,5,10,30].map(n=><option key={n} value={n}>{n}s</option>)}</select></small></div><input id="auto" type="checkbox" role="switch" checked={auto} disabled={!active || correct===incorrect} onChange={e=>setAuto(e.target.checked)}/></div><small className="help">Timed readings may count the same part repeatedly. Use manual recording for one result per part.</small></section></div>
    {records.length>0 && <section className="card log-card"><div className="card-heading"><h2>Recent observations</h2><span className="tag">LAST 5</span></div><div className="table-wrap"><table><thead><tr><th>Time</th><th>Decision</th><th>Class</th><th>Confidence</th></tr></thead><tbody>{records.slice(-5).reverse().map((r,i)=><tr key={i}><td>{new Date(r.time).toLocaleString()}</td><td><span className={'result-badge '+r.verdict.toLowerCase()}>{r.verdict}</span></td><td>{r.label}</td><td>{(r.confidence*100).toFixed(1)}%</td></tr>)}</tbody></table></div></section>}
    <footer><span>PARTLENS <span className="footer-divider">/</span> A clearer view of quality.</span><span><ShieldCheck size={13}/> Local inference · Browser storage</span></footer></main></div></div>;
}
createRoot(document.getElementById('root')!).render(<App/>);
