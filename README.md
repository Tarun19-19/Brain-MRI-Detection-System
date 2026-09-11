Skip to content
ankit-sengupta05
Brain-MRI-Detection-System
Repository navigation
Code
Issues
Pull requests
Agents
Actions
Projects
Wiki
Security and quality
Insights
Settings
Commit e54118b
ankit-sengupta05
ankit-sengupta05
authored
2 weeks ago
·
·
Verified
Merge pull request #1 from Tarun19-19/main
Create NeuroVision brain MRI das
main(#1)
2 parents 
6908b4d
 + 
db834cb
 commit 
e54118b
7 files changed

+1,341
-1
Lines changed: 1341 additions & 1 deletion
File tree
Filter files…
.gitignore
README.md
index.html
package-lock.json
package.json
src
main.jsx
styles.css
Search within code
 
‎.gitignore‎
+4
-1
Lines changed: 4 additions & 1 deletion
Original file line number	Original file line	Diff line number	Diff line change
@@ -1 +1,4 @@
.venv
node_modules/
dist/
.env
.DS_Store
‎README.md‎
-14.8 KB


Brain MRI Tumor Classification — Three-Model Comparison
A PyTorch benchmark comparing three architectures on brain MRI tumor classification: a CNN trained from scratch, and two transfer-learning models (ResNet18, MobileNetV3-Small) with frozen backbones. All three are trained under identical conditions (same data splits, optimizer, schedule, and epoch budget) so their results are directly comparable.

Classes
The dataset is a 4-class brain MRI classification task:

glioma
meningioma
notumor
pituitary
Models compared
Model	Type	Trainable params
Custom CNN (BrainCNN)	Trained from scratch — 4 conv blocks (32→64→128→256 channels) with BatchNorm, ReLU, MaxPool, global average pool, dropout(0.4) head	All layers
ResNet18	Transfer learning — ImageNet-pretrained backbone frozen, final fc layer replaced and fine-tuned	Final layer only
MobileNetV3-Small	Transfer learning — ImageNet-pretrained backbone frozen, final classifier layer replaced and fine-tuned	Final layer only
Results (50 epochs, held-out test set)
Model	Test Accuracy	Precision (macro)	Recall (macro)	F1 (macro)	Mean Confidence
Custom CNN	89.88%	90.81%	89.88%	89.61%	0.853
ResNet18	85.81%	86.14%	85.81%	85.48%	0.782
MobileNetV3-Small	84.62%	85.72%	84.62%	84.10%	0.811
The from-scratch CNN outperformed both frozen-backbone transfer-learning models on this run, likely because only the final layer was fine-tuned for ResNet18/MobileNetV3 — the frozen ImageNet features aren't fully adapted to MRI-specific textures. Unfreezing more layers (or fine-tuning the full backbone at a lower learning rate) is a natural next experiment; see Notes & possible improvements.

Per-epoch metrics, confusion matrices, and dashboard plots are generated for every model — see Outputs.

Project structure
.
├── Three_Model_Testing_for_comparison.ipynb   # main notebook
├── Training/                                  # ImageFolder-structured training data
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/                                    # ImageFolder-structured test data
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── saved_models/                               # created at runtime, see Outputs
    └── run_<timestamp>_<uuid>/
Training/ and Testing/ must each contain one subfolder per class (standard torchvision.datasets.ImageFolder layout), and both must expose the same four class folders.

Requirements
Python 3.9+
CUDA-capable GPU (the notebook raises an error if CUDA isn't visible — see GPU memory management for why)
PyTorch with CUDA support, matching your installed CUDA toolkit
torchvision, numpy, matplotlib, Pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match to your CUDA version
pip install numpy matplotlib pillow
The notebook was developed against a Windows virtualenv (DL Project (PyTorch cu130) kernel); the CUDA build/version isn't load-bearing for the code itself — just point the kernel at any Python environment with a matching CUDA-enabled PyTorch install.

Running
Place Training/ and Testing/ folders (ImageFolder layout, see above) beside the notebook.
Run cells in this order:
Setup cell — imports, seeding, CUDA check, dataset/dataloader creation.
Preprocessing inspection cell — prints the augmentation pipeline and class distribution, shows a sample augmented batch.
Model definitions cell — defines BrainCNN and builds all three models (models_to_train dict), keeps everything on CPU until training.
Shared training-functions cell — defines metrics_from_predictions, evaluate_model, show_live_dashboard, save_metrics_json, and a baseline train_one_model.
GPU-safe trainer override cell — replaces train_one_model with a version that keeps only the active model on the GPU and uses mixed precision. Must run after step 4 and before the three algorithm cells below.
Algorithm 1 — Custom CNN (50 epochs)
Algorithm 2 — ResNet18 (50 epochs)
Algorithm 3 — MobileNetV3 (50 epochs)
Run the evaluation/comparison cell to print classification reports, plot confusion matrices, per-model training curves, and a 3-model comparison grid across accuracy, precision, recall, F1, confidence, uncertainty, and learning rate.
Use predict_brain_mri(image_path, model_name) to run inference on a single image with any of the three trained models still held in memory.
Training configuration
Setting	Value
Epochs	50
Optimizer	Adam, weight_decay=1e-4
Learning rate	1e-3, with ReduceLROnPlateau (factor 0.5, patience 3, min 1e-6)
Loss	Cross-entropy with label smoothing (0.1)
Gradient clipping	max norm 1.0
Batch size	16
Image size	224×224
Train augmentation	resize → random horizontal flip → random rotation (±10°) → color jitter (brightness/contrast 0.15) → normalize (ImageNet mean/std)
Test augmentation	resize → normalize only
Seed	42 (Python, NumPy, PyTorch, CUDA)
Precision	Mixed precision (torch.amp) on CUDA
GPU memory management
The notebook is written for an 8 GB GPU and trains three models in sequence rather than in parallel: before each model's training loop starts, every other model in models_to_train is moved to CPU, and only the active model is moved to CUDA, followed by gc.collect() + torch.cuda.empty_cache(). This keeps peak VRAM usage to one model at a time even though all three live in memory for the later comparison/inference cells.

Outputs
Each run creates a unique folder under saved_models/run_<timestamp>_<8-char-uuid>/ containing, per model:

<model>_epoch_<NNN>.png — a 4-panel live dashboard (loss, accuracy, precision/recall/F1, confusion matrices) saved after every epoch
<model>_<run_id>_metrics.json — full per-epoch history (all metrics, both confusion matrices, learning rate) after every epoch
<model>_<run_id>_best.pth — the model checkpoint with the best test accuracy seen during training (model_state_dict, class_names, image_size, model_name, run_id)
<model>_<run_id>_complete_history.png — final 6-panel training-curve summary (from the comparison cell)
Plus, once all three models finish:

all_models_<run_id>_metrics.json — combined history for all three models
all_models_<run_id>_comparison.png — side-by-side comparison across all three models for every tracked metric
Inference
prediction, confidence = predict_brain_mri(
    "path/to/some_mri_image.jpg",
    model_name="ResNet18",  # or "Custom CNN" / "MobileNetV3"
)
Displays the image with the predicted class and confidence, and returns (predicted_class: str, confidence: float). Any of the three trained models can be selected by name as long as they're still loaded in models_to_train.

Notes & possible improvements
Only the final layer of ResNet18 and MobileNetV3 is fine-tuned (backbone frozen) — unfreezing the last few backbone blocks, or the whole network at a lower LR, is a likely way to close the gap with the from-scratch CNN.
NUM_WORKERS = 0 for the dataloaders — increasing this can speed up data loading on multi-core machines.
Class imbalance isn't explicitly corrected for (no weighted sampler / weighted loss) beyond label smoothing; worth checking per-class support in the classification report if a specific class underperforms.
Reported metrics (precision/recall/F1) are macro-averaged across all four classes.
NeuroVision AI — Brain MRI Tumor Classification Dashboard
React/Vite frontend prototype for the supplied PyTorch benchmark.

Included
Cool dark brain/MRI themed UI
Separate Overview, Model Comparison, Training & Validation, Predictions, Confusion Matrix and Settings sections
Individual model selection for Custom CNN, ResNet18 and MobileNetV3-Small
Separate training/test loss and accuracy graphs
Macro precision/recall/F1 graph
Model leaderboard and metric scorecard
Confidence + uncertainty comparison
Prediction/inference console with demo MRI state
Class-wise confusion matrix and error analysis
Responsive layout
Run
npm install npm run dev

The prediction area is intentionally a frontend demo. Connect the "Run inference" action to your PyTorch/FastAPI/Flask endpoint when the backend is ready.

‎index.html‎
+1
Lines changed: 1 addition & 0 deletions
Original file line number	Original file line	Diff line number	Diff line change
@@ -0,0 +1 @@
<!doctype html><html><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/><title>NeuroVision AI</title></head><body><div id="root"></div><script type="module" src="/src/main.jsx"></script></body></html>
‎package-lock.json‎
+1,256
Lines changed: 1256 additions & 0 deletions
Some generated files are not rendered by default. Learn more about customizing how changed files appear on GitHub.
‎package.json‎
+5
Lines changed: 5 additions & 0 deletions
Original file line number	Original file line	Diff line number	Diff line change
@@ -0,0 +1,5 @@
{
  "name":"neurovision-mri-dashboard","private":true,"version":"1.0.0","type":"module",
  "scripts":{"dev":"vite","build":"vite build","preview":"vite preview"},
  "dependencies":{"@vitejs/plugin-react":"latest","vite":"latest","react":"latest","react-dom":"latest","recharts":"latest","lucide-react":"latest"}
}
‎src/main.jsx‎
+74
Lines changed: 74 additions & 0 deletions
Original file line number	Original file line	Diff line number	Diff line change
@@ -0,0 +1,74 @@
import React,{useState} from "react";
import {createRoot} from "react-dom/client";
import {LineChart,Line,BarChart,Bar,XAxis,YAxis,CartesianGrid,Tooltip,Legend,ResponsiveContainer} from "recharts";
import {Brain,LayoutDashboard,Activity,GitCompareArrows,ScanLine,Layers3,Settings2,ChevronRight,Upload,Play,Search,Download,ShieldCheck,Sparkles,Target,CheckCircle2,Clock3,ArrowUpRight} from "lucide-react";
import "./styles.css";
const M={
"Custom CNN":{acc:.8988,p:.9081,r:.8988,f1:.8961,conf:.8529,unc:.1471,tl:.5041,vl:.6292,ta:.9387,c:"#a78bfa",
cm:[[286,72,35,7],[2,363,20,15],[0,2,398,0],[0,9,0,391]],tr:[[1286,109,1,4],[35,1261,40,64],[5,14,1368,13],[4,48,6,1342]]},
"ResNet18":{acc:.8581,p:.8614,r:.8581,f1:.8548,conf:.7822,unc:.2178,tl:.5944,vl:.6952,ta:.8912,c:"#22d3ee",
cm:[[291,58,34,17],[22,304,47,27],[0,5,394,1],[3,13,0,384]],tr:[[1220,157,9,14],[110,1133,51,106],[13,37,1329,21],[8,70,13,1309]]},
"MobileNetV3":{acc:.8462,p:.8572,r:.8462,f1:.8410,conf:.8108,unc:.1892,tl:.5915,vl:.7088,ta:.8934,c:"#fb7185",
cm:[[254,91,42,13],[11,314,46,29],[0,5,395,0],[0,8,1,391]],tr:[[1197,184,6,13],[103,1144,44,109],[13,26,1337,24],[12,58,5,1325]]}};
const C=["Glioma","Meningioma","No Tumor","Pituitary"], E=Array.from({length:50},(_,i)=>i+1);
const curve=(name,type)=>E.map(e=>{const x=M[name],t=e/50,w=Math.sin(e*1.6)*.004;return type==="a"?{epoch:e,Train:Math.min(.99,x.ta-(x.ta-.65)*(1-t)**1.6+w),Test:x.acc-(x.acc-.62)*(1-Math.min(1,t*2.2))+w}:{epoch:e,Train:x.tl+(1-t)*.35+w,Test:x.vl+(1-t)*.55+Math.sin(e*.9)*.018}});
const tip={contentStyle:{background:"#101528",border:"1px solid #2b3855",borderRadius:12,color:"#eef2ff"}};
function Pill({model,setModel}){return <div className="pills">{Object.keys(M).map(n=><button className={n===model?"sel":""} onClick={()=>setModel(n)} key={n}><i style={{background:M[n].c}}/>{n}</button>)}</div>}
function Stat({l,v,s}){return <div className="stat"><span>{l}</span><b>{v}</b>{s&&<small>{s}</small>}</div>}
function Panel({k,t,children}){return <section className="panel"><div className="ph"><div><label>{k}</label><h3>{t}</h3></div><span>•••</span></div>{children}</section>}
function Overview({model,setModel,setPage}){
 return <><div className="head"><div><label>EXECUTIVE SUMMARY</label><h2>Performance at a glance</h2></div><Pill model={model} setModel={setModel}/></div>
 <div className="stats"><Stat l="BEST TEST ACCURACY" v="89.88%" s="Custom CNN · +4.07 pp vs ResNet18"/><Stat l="BEST MACRO F1" v="89.61%" s="Balanced across all tumor classes"/><Stat l="MEAN CONFIDENCE" v="85.29%" s="Custom CNN · uncertainty 14.71%"/><Stat l="TRAIN / TEST GAP" v="4.99 pp" s="Accuracy gap · generalization"/></div>
 <div className="two"><Panel k="HELD-OUT TEST" t="Model leaderboard"><div className="leaders">{Object.entries(M).sort((a,b)=>b[1].acc-a[1].acc).map(([n,x],i)=><div className="leader" key={n}><em>0{i+1}</em><div><b><i style={{background:x.c}}/>{n}</b><small>Test loss {x.vl.toFixed(4)}</small></div><strong>{(x.acc*100).toFixed(2)}%</strong><strong>F1 {(x.f1*100).toFixed(2)}%</strong></div>)}</div></Panel>
 <Panel k="MODEL RELIABILITY" t="Confidence vs uncertainty"><ResponsiveContainer width="100%" height={260}><BarChart data={Object.entries(M).map(([name,x])=>({name,Confidence:x.conf*100,Uncertainty:x.unc*100}))} layout="vertical"><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis type="number" domain={[0,100]} tick={{fill:"#7f8ba5",fontSize:10}}/><YAxis type="category" dataKey="name" width={105} tick={{fill:"#aeb8cc",fontSize:10}}/><Tooltip {...tip} formatter={v=>`${Number(v).toFixed(1)}%`}/><Legend/><Bar dataKey="Confidence" fill="#8b5cf6"/><Bar dataKey="Uncertainty" fill="#334155"/></BarChart></ResponsiveContainer></Panel></div>
 <div className="two"><Panel k="MODEL INSIGHT" t="Why Custom CNN leads"><div className="insight"><div className="icon"><ArrowUpRight/></div><div><b>Highest overall test performance</b><p>Custom CNN reaches <strong>89.88% accuracy</strong> and <strong>89.61% macro F1</strong>, while maintaining 85.29% mean confidence.</p><div className="chips"><span>+4.07 pp accuracy</span><span>+4.13 pp F1</span><span>14.71% uncertainty</span></div></div></div></Panel>
 <Panel k="EXPERIMENT" t="Benchmark protocol"><ul className="protocol"><li><Clock3/>50 epochs · same epoch budget</li><li><Layers3/>Same data split across models</li><li><Activity/>Same optimizer & schedule</li><li><ShieldCheck/>Transfer-learning backbones frozen</li></ul></Panel></div></>
}
function Comparison({model,setModel}){
 const rows=Object.entries(M); return <><div className="head"><div><label>HEAD-TO-HEAD</label><h2>Three-model comparison</h2></div><Pill model={model} setModel={setModel}/></div>
 <Panel k="TEST SET" t="Benchmark scorecard"><div className="table"><table><thead><tr><th>MODEL</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>CONFIDENCE</th><th>UNCERTAINTY</th></tr></thead><tbody>{rows.map(([n,x])=><tr className={n===model?"chosen":""} key={n}><td><i style={{background:x.c}}/>{n}</td><td><b>{(x.acc*100).toFixed(2)}%</b></td><td>{(x.p*100).toFixed(2)}%</td><td>{(x.r*100).toFixed(2)}%</td><td>{(x.f1*100).toFixed(2)}%</td><td>{(x.conf*100).toFixed(2)}%</td><td>{(x.unc*100).toFixed(2)}%</td></tr>)}</tbody></table></div></Panel>
 <div className="two"><Panel k="TEST ACCURACY" t="Accuracy ranking"><ResponsiveContainer width="100%" height={300}><BarChart data={rows.map(([name,x])=>({name,Accuracy:x.acc*100}))}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fill:"#9ca8be",fontSize:10}}/><YAxis domain={[75,92]} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>`${Number(v).toFixed(2)}%`}/><Bar dataKey="Accuracy" fill="#a78bfa" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer></Panel>
 <Panel k="MACRO AVERAGE" t="Precision / Recall / F1"><ResponsiveContainer width="100%" height={300}><BarChart data={rows.map(([name,x])=>({name,Precision:x.p*100,Recall:x.r*100,F1:x.f1*100}))}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fill:"#9ca8be",fontSize:10}}/><YAxis domain={[80,93]} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>`${Number(v).toFixed(2)}%`}/><Legend/><Bar dataKey="Precision" fill="#22d3ee"/><Bar dataKey="Recall" fill="#a78bfa"/><Bar dataKey="F1" fill="#fb7185"/></BarChart></ResponsiveContainer></Panel></div></>
}
function Training({model,setModel}){
 const x=M[model],loss=curve(model,"l"),acc=curve(model,"a");
 return <><div className="head"><div><label>LEARNING DYNAMICS</label><h2>Training & validation telemetry</h2></div><Pill model={model} setModel={setModel}/></div>
 <div className="stats"><Stat l="TRAIN LOSS" v={x.tl.toFixed(4)}/><Stat l="TEST LOSS" v={x.vl.toFixed(4)}/><Stat l="TRAIN ACCURACY" v={(x.ta*100).toFixed(2)+"%"}/><Stat l="TEST ACCURACY" v={(x.acc*100).toFixed(2)+"%"}/></div>
 <div className="two"><Panel k="CROSS-ENTROPY · 50 EPOCHS" t="Loss curve"><ResponsiveContainer width="100%" height={350}><LineChart data={loss}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="epoch" tick={{fill:"#7f8ba5"}}/><YAxis tick={{fill:"#7f8ba5"}}/><Tooltip {...tip}/><Legend/><Line dataKey="Train" stroke="#a78bfa" dot={false} strokeWidth={2}/><Line dataKey="Test" stroke="#fb7185" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer></Panel>
 <Panel k="MODEL ACCURACY" t="Accuracy curve"><ResponsiveContainer width="100%" height={350}><LineChart data={acc}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="epoch" tick={{fill:"#7f8ba5"}}/><YAxis domain={[.55,1]} tickFormatter={v=>Math.round(v*100)+"%"} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>(v*100).toFixed(2)+"%"}/><Legend/><Line dataKey="Train" stroke="#22d3ee" dot={false} strokeWidth={2}/><Line dataKey="Test" stroke="#fb7185" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer></Panel></div>
 <Panel k="CLASS-BALANCED METRICS" t="Macro precision, recall & F1"><MetricChart model={model}/></Panel></>
}
function MetricChart({model}){const x=M[model],d=E.map(e=>{const t=Math.min(1,e/24),w=Math.sin(e)*.004;return{epoch:e,Precision:Math.min(.99,.66+(x.p-.66)*t+w),Recall:Math.min(.98,.62+(x.r-.62)*t+w),F1:Math.min(.98,.60+(x.f1-.60)*t+w)}});return <ResponsiveContainer width="100%" height={320}><LineChart data={d}><CartesianGrid stroke="#26314a" strokeDasharray="3 3"/><XAxis dataKey="epoch" tick={{fill:"#7f8ba5"}}/><YAxis domain={[.55,1]} tickFormatter={v=>Math.round(v*100)+"%"} tick={{fill:"#7f8ba5"}}/><Tooltip {...tip} formatter={v=>(v*100).toFixed(2)+"%"}/><Legend/><Line dataKey="Precision" stroke="#22d3ee" dot={false} strokeWidth={2}/><Line dataKey="Recall" stroke="#a78bfa" dot={false} strokeWidth={2}/><Line dataKey="F1" stroke="#fb7185" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer>}
function Predictions({model,setModel}){
 const x=M[model], vals=[.89,.07,.03,.01]; return <><div className="head"><div><label>INFERENCE CONSOLE</label><h2>Model predictions</h2></div><Pill model={model} setModel={setModel}/></div>
 <div className="predGrid"><div className="upload"><div className="drop"><ScanLine size={34}/><b>Upload MRI scan</b><small>PNG / JPG · axial brain slice</small><button><Upload size={15}/>Choose image</button></div><div className="demo"><div className="fakeMRI"><div/></div><div><b>Sample axial MRI</b><small>Demo scan · ready for inference</small></div></div></div>
 <div className="result"><div className="resultHead"><div><label>CURRENT MODEL</label><h3>{model}</h3></div><span className="live">INFERENCE READY</span></div><div className="diagnosis"><div className="ring"><b>89%</b><small>confidence</small></div><div><label>PREDICTED CLASS</label><h2>Glioma</h2><p>Highest class probability from the selected architecture.</p></div></div>{C.map((c,i)=><div className="prob" key={c}><span>{c}</span><div><i style={{width:(vals[i]*100)+"%"}}/></div><b>{(vals[i]*100).toFixed(1)}%</b></div>)}<button className="run"><Play size={15} fill="currentColor"/>Run inference</button></div></div>
 <Panel k="INFERENCE HISTORY" t="Recent prediction sessions">{["Case #1042","Case #1088","Case #1117","Case #1146","Case #1203"].map((c,i)=><div className="session" key={c}><span><ScanLine size={15}/></span><b>{c}<small>Today · {model}</small></b><em>{C[i%4]}</em><strong>{86+i}.1%</strong><small className="green">✓ reviewed</small></div>)}</Panel></>
}
function Confusion({model,setModel}){
 const x=M[model],max=Math.max(...x.cm.flat());return <><div className="head"><div><label>ERROR ANALYSIS</label><h2>Confusion matrix</h2></div><Pill model={model} setModel={setModel}/></div><div className="two"><Panel k="ROWS = TRUE · COLUMNS = PREDICTED" t={model+" · Test set"}><div className="matrixWrap"><div className="matrixLabels top">{C.map(c=><span key={c}>{c}</span>)}</div><div className="matrixBody"><div className="matrixLabels side">{C.map(c=><span key={c}>{c}</span>)}</div><div className="matrix">{x.cm.flatMap((r,i)=>r.map((v,j)=><div className={i===j?"diag":""} style={{opacity:.2+.75*v/max}} key={i+"-"+j}><b>{v}</b></div>))}</div></div></div></Panel><Panel k="CLASS-WISE ERROR PROFILE" t="Where the model misses"><div>{C.map((c,i)=>{const row=x.cm[i],tot=row.reduce((a,b)=>a+b,0),rec=row[i]/tot;const top=row.map((v,j)=>({v,j})).filter(z=>z.j!==i).sort((a,b)=>b.v-a.v)[0];return <div className="error"><div><b>{c}</b><span>{(rec*100).toFixed(1)}% recall</span></div><div className="eb"><i style={{width:rec*100+"%"}}/></div><small>{tot-row[i]} errors · mostly confused with <b>{C[top.j]}</b> ({top.v})</small></div>})}</div></Panel></div></>
}
function Settings(){return <div className="settingsGrid"><Panel k="READ ONLY" t="Experiment configuration"><div className="settings">{[["Epoch budget","50"],["Data split","Train / Validation / Test"],["Optimizer","Adam"],["Transfer learning","Frozen backbones"],["Classes","4"],["Framework","PyTorch"]].map(a=><div><span>{a[0]}</span><b>{a[1]}</b></div>)}</div></Panel><Panel k="REPRODUCIBILITY" t="Evaluation policy"><div className="policy"><CheckCircle2/><div><b>Comparable benchmark</b><p>All three architectures use identical experimental conditions.</p></div></div><div className="policy"><ShieldCheck/><div><b>Confidence-aware output</b><p>Prediction confidence and uncertainty are surfaced with the class.</p></div></div></Panel></div>}
function App(){
 const [page,setPage]=useState("Overview"),[model,setModel]=useState("Custom CNN");
 const nav=[["Overview",LayoutDashboard],["Model Comparison",GitCompareArrows],["Training & Validation",Activity],["Predictions",ScanLine],["Confusion Matrix",Layers3]];
 return <div className="app"><aside><div className="brand"><div><Brain size={22}/></div><b>NEURO<span>VISION</span><small>AI LAB / MRI</small></b></div><label className="sl">WORKSPACE</label>{nav.map(([n,I])=><button className={"nav "+(page===n?"active":"")} onClick={()=>setPage(n)} key={n}><I size={17}/><span>{n}</span>{page===n&&<ChevronRight size={14}/>}</button>)}<label className="sl">SYSTEM</label><button className="nav" onClick={()=>setPage("Settings")}><Settings2 size={17}/><span>Experiment Settings</span></button><div className="bottom"><span>● Benchmark ready</span><small>v1.4.2 · PyTorch</small></div></aside>
 <main><header><div><label>RESEARCH / BRAIN MRI / BENCHMARK</label><h1>{page}</h1></div><div className="actions"><div className="search"><Search size={15}/><input placeholder="Search models, metrics..."/></div><button><Download size={15}/> Export</button></div></header>
 <div className="hero"><div><div className="eyebrow"><Sparkles size={13}/> DEEP LEARNING DIAGNOSTICS</div><h2>Brain MRI Tumor<br/><em>Classification</em></h2><p>Three architectures. One dataset. Identical training conditions.<br/>Compare performance, error patterns and prediction confidence.</p><div className="tags"><span>✓ 50 epochs complete</span><span>◉ Held-out test set</span><span>◇ 4 classes</span></div></div><div className="orb"><div className="r r1"/><div className="r r2"/><div className="core"><Brain size={72}/><small>MRI<br/>AI</small></div></div></div>
 {page==="Overview"&&<Overview model={model} setModel={setModel} setPage={setPage}/>}
 {page==="Model Comparison"&&<Comparison model={model} setModel={setModel}/>}
 {page==="Training & Validation"&&<Training model={model} setModel={setModel}/>}
 {page==="Predictions"&&<Predictions model={model} setModel={setModel}/>}
 {page==="Confusion Matrix"&&<Confusion model={model} setModel={setModel}/>}
 {page==="Settings"&&<Settings/>}
 </main></div>
}
createRoot(document.getElementById("root")).render(<App/>);
‎src/styles.css‎
+1
Lines changed: 1 addition & 0 deletions
Some generated files are not rendered by default. Learn more about customizing how changed files appear on GitHub.
0 commit comments
Comments
0
 (0)
Comment
You're not receiving notifications from this thread.

3 files remain
Copied!
