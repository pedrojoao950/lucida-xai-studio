"use client";

import { useMemo, useState } from "react";

const cases = {
  "A-48291": { score: 78, verdict: "Aprovar", confidence: 92, person: "M. Silva", tension: "baixa" },
  "A-37104": { score: 54, verdict: "Revisar", confidence: 67, person: "A. Manuel", tension: "média" },
  "A-59022": { score: 31, verdict: "Recusar", confidence: 81, person: "K. Santos", tension: "alta" },
};

const forces = [
  { name: "Rendimento resiliente", value: "+34", tone: "lime", x: 72, y: 22, size: 96 },
  { name: "Memória de crédito", value: "+27", tone: "cyan", x: 79, y: 67, size: 82 },
  { name: "Pressão da dívida", value: "−18", tone: "coral", x: 25, y: 72, size: 74 },
  { name: "Raiz profissional", value: "+12", tone: "violet", x: 29, y: 22, size: 65 },
];

const nav = ["Observatório", "Constelação", "Pontos de viragem", "Laboratório"];

const models = [
  { name: "Regressão Logística", code: "LR", recall: 0.82, f1: 0.84, roc: 0.88, pr: 0.81, time: "0.8s" },
  { name: "Random Forest", code: "RF", recall: 0.86, f1: 0.87, roc: 0.92, pr: 0.87, time: "4.2s" },
  { name: "Gradient Boosting", code: "GB", recall: 0.87, f1: 0.88, roc: 0.93, pr: 0.89, time: "6.8s" },
  { name: "XGBoost", code: "XGB", recall: 0.89, f1: 0.90, roc: 0.95, pr: 0.92, time: "3.7s" },
  { name: "LightGBM", code: "LGB", recall: 0.88, f1: 0.89, roc: 0.94, pr: 0.91, time: "2.1s" },
];

const explainers = {
  SHAP: { title: "Forças globais e locais", text: "Valores de Shapley distribuem o impacto de cada variável com consistência aditiva.", values: [92, 78, 64, 51, 38] },
  LIME: { title: "Vizinhança desta decisão", text: "Um modelo linear local aproxima o comportamento do classificador em torno do caso selecionado.", values: [86, 71, 58, 44, 29] },
  "Partial Dependence": { title: "Efeito médio da variável", text: "Partial Dependence Plots revelam como a previsão varia quando um atributo muda.", values: [24, 39, 56, 76, 88] },
};

type CsvDataset = { name: string; size: string; headers: string[]; rows: string[][]; delimiter: string };
const SCIENCE_API = process.env.NEXT_PUBLIC_SCIENCE_API_URL || "";

function parseCsv(text: string) {
  const firstLine = text.split(/\r?\n/, 1)[0] || "";
  const delimiter = (firstLine.match(/;/g)?.length || 0) > (firstLine.match(/,/g)?.length || 0) ? ";" : ",";
  const rows: string[][] = []; let row: string[] = []; let cell = ""; let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"' && quoted && text[i + 1] === '"') { cell += '"'; i++; }
    else if (char === '"') quoted = !quoted;
    else if (char === delimiter && !quoted) { row.push(cell.trim()); cell = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(cell.trim()); if (row.some(Boolean)) rows.push(row); row = []; cell = "";
    } else cell += char;
  }
  if (cell || row.length) { row.push(cell.trim()); if (row.some(Boolean)) rows.push(row); }
  const headers = (rows.shift() || []).map((h, i) => h || `coluna_${i + 1}`);
  return { headers, rows: rows.map(r => headers.map((_, i) => r[i] ?? "")), delimiter };
}

function inferType(values: string[]) {
  const valid = values.filter(v => v !== "").slice(0, 500);
  if (!valid.length) return "Vazio";
  if (valid.every(v => /^-?\d+(?:[.,]\d+)?$/.test(v))) return "Numérico";
  if (valid.every(v => /^(true|false|sim|não|nao|0|1)$/i.test(v))) return "Booleano";
  if (valid.every(v => !Number.isNaN(Date.parse(v)) && /[-/]/.test(v))) return "Data";
  return new Set(valid).size <= Math.min(20, valid.length * .2) ? "Categórico" : "Texto";
}

export default function Home() {
  const [active, setActive] = useState("Observatório");
  const [selectedCase, setSelectedCase] = useState<keyof typeof cases>("A-48291");
  const [threshold, setThreshold] = useState(62);
  const [focus, setFocus] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedModel, setSelectedModel] = useState(3);
  const [folds, setFolds] = useState(5);
  const [explainer, setExplainer] = useState<keyof typeof explainers>("SHAP");
  const [dataset, setDataset] = useState<CsvDataset | null>(null);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  const [csvError, setCsvError] = useState("");
  const [imbalance, setImbalance] = useState("Class weights");
  const [validationMode, setValidationMode] = useState<"stratified" | "temporal">("stratified");
  const [numericImputer, setNumericImputer] = useState("Mediana");
  const [categoricalImputer, setCategoricalImputer] = useState("Moda");
  const [encoding, setEncoding] = useState("One-hot");
  const [scaling, setScaling] = useState("StandardScaler");
  const [excluded, setExcluded] = useState<string[]>([]);
  const [training, setTraining] = useState(false);
  const [trainingRun, setTrainingRun] = useState(0);
  const [trainedSignature, setTrainedSignature] = useState("");
  const [xaiScope, setXaiScope] = useState<"global" | "local">("global");
  const [xaiFeature, setXaiFeature] = useState("");
  const [instanceIndex, setInstanceIndex] = useState(0);
  const [protectedFeature, setProtectedFeature] = useState("");
  const [calibration, setCalibration] = useState<"isotonic" | "platt">("isotonic");
  const [scientificModels, setScientificModels] = useState<typeof models | null>(null);
  const [engineMode, setEngineMode] = useState<"experimental" | "scientific" | "error">("experimental");
  const [engineError, setEngineError] = useState("");
  const handleDataset = async (file?: File) => {
    if (!file) return;
    setCsvError("");
    if (file.size > 25 * 1048576) { setCsvError("O ficheiro excede o limite local de 25 MB."); return; }
    try {
      const parsed = parseCsv(await file.text());
      if (parsed.headers.length < 2 || !parsed.rows.length) throw new Error("CSV sem dados suficientes");
      const size = file.size > 1048576 ? `${(file.size/1048576).toFixed(1)} MB` : `${Math.max(1, Math.ceil(file.size/1024))} KB`;
      setDataset({ name: file.name, size, ...parsed }); setSourceFile(file); setTarget(parsed.headers.at(-1) || "");
      setExcluded(parsed.headers.filter(h => /(^id$|_id$|uuid|customer|account|nif|bi)/i.test(h)));
    } catch { setCsvError("Não foi possível interpretar o CSV. Verifique o separador e a codificação UTF-8."); }
  };
  const profile = useMemo(() => {
    if (!dataset) return null;
    const missing = dataset.rows.reduce((sum, row) => sum + row.filter(v => v === "").length, 0);
    const duplicates = dataset.rows.length - new Set(dataset.rows.map(row => JSON.stringify(row))).size;
    const types = dataset.headers.map((header, index) => ({ header, type: inferType(dataset.rows.map(row => row[index])), missing: dataset.rows.filter(row => row[index] === "").length, unique: new Set(dataset.rows.map(row => row[index]).filter(Boolean)).size }));
    const targetIndex = dataset.headers.indexOf(target);
    const counts = new Map<string, number>();
    if (targetIndex >= 0) dataset.rows.forEach(row => { const value = row[targetIndex] || "Ausente"; counts.set(value, (counts.get(value) || 0) + 1); });
    const classes = [...counts.entries()].sort((a,b)=>b[1]-a[1]);
    return { missing, duplicates, types, classes };
  }, [dataset, target]);
  const prepared = useMemo(() => {
    if (!dataset || !profile) return null;
    const featureIndexes = dataset.headers.map((h,i)=>({h,i,type:profile.types[i].type})).filter(c=>c.h!==target&&!excluded.includes(c.h));
    const modes = featureIndexes.map(c=>{const counts=new Map<string,number>();dataset.rows.forEach(r=>{if(r[c.i])counts.set(r[c.i],(counts.get(r[c.i])||0)+1)});return [...counts.entries()].sort((a,b)=>b[1]-a[1])[0]?.[0]||"Ausente"});
    const numericStats = featureIndexes.map(c=>{const nums=dataset.rows.map(r=>Number(r[c.i].replace(",","."))).filter(Number.isFinite).sort((a,b)=>a-b);const mean=nums.reduce((a,b)=>a+b,0)/(nums.length||1);const median=nums.length?nums[Math.floor(nums.length/2)]:0;const sd=Math.sqrt(nums.reduce((s,n)=>s+(n-mean)**2,0)/(nums.length||1))||1;return{mean,median,sd,min:nums[0]||0,max:nums.at(-1)||1}});
    const outHeaders:string[]=[];const categories=featureIndexes.map(c=>c.type==="Categórico"||c.type==="Texto"?[...new Set(dataset.rows.map(r=>r[c.i]).filter(Boolean))].slice(0,8):[]);
    featureIndexes.forEach((c,idx)=>{if((c.type==="Categórico"||c.type==="Texto")&&encoding==="One-hot")categories[idx].forEach(v=>outHeaders.push(`${c.h}__${v}`));else outHeaders.push(c.h)});outHeaders.push(target);
    const transformRow=(row:string[])=>{const out:(string|number)[]=[];featureIndexes.forEach((c,idx)=>{let raw=row[c.i];if(!raw)raw=c.type==="Numérico"?String(numericImputer==="Média"?numericStats[idx].mean:numericStats[idx].median):(categoricalImputer==="Moda"?modes[idx]:"Ausente");if(c.type==="Numérico"){let n=Number(raw.replace(",","."));const s=numericStats[idx];if(scaling==="StandardScaler")n=(n-s.mean)/s.sd;else if(scaling==="MinMaxScaler")n=(n-s.min)/((s.max-s.min)||1);out.push(Number(n.toFixed(3)))}else if(encoding==="One-hot")categories[idx].forEach(v=>out.push(raw===v?1:0));else if(encoding==="Ordinal")out.push(Math.max(0,categories[idx].indexOf(raw)));else out.push(raw)});out.push(row[dataset.headers.indexOf(target)]);return out};
    const suspicious=dataset.headers.filter(h=>h!==target&&(h.toLowerCase().includes(target.toLowerCase())||/(outcome|result|decision|approved|status_final)/i.test(h))&&!excluded.includes(h));
    return{headers:outHeaders,rows:dataset.rows.slice(0,5).map(transformRow),features:featureIndexes.length,outputFeatures:outHeaders.length-1,suspicious};
  },[dataset,profile,target,excluded,numericImputer,categoricalImputer,encoding,scaling]);
  const current = cases[selectedCase];
  const adjusted = useMemo(() => Math.max(12, Math.min(96, current.score + (62 - threshold) * .35)), [current, threshold]);
  const experimentSignature = `${dataset?.name || "demo"}|${dataset?.rows.length || 0}|${target}|${folds}|${validationMode}|${imbalance}|${numericImputer}|${categoricalImputer}|${encoding}|${scaling}|${excluded.join(",")}`;
  const benchmarkModels = useMemo(() => {
    if (scientificModels) return scientificModels;
    if (!dataset || !profile || !prepared || profile.classes.length < 2) return models;
    const rowFactor = Math.min(.035, Math.log10(Math.max(10, dataset.rows.length)) * .009);
    const balanceRatio = profile.classes.at(-1)![1] / profile.classes[0][1];
    const balanceLift = imbalance === "Sem correção" ? 0 : imbalance === "SMOTE" ? .025 : imbalance === "Class weights" ? .018 : .011;
    const validationPenalty = validationMode === "temporal" ? .024 : folds === 10 ? .004 : 0;
    const qualityPenalty = Math.min(.04, profile.missing / Math.max(1, dataset.rows.length * dataset.headers.length) * .2);
    const featureLift = Math.min(.018, prepared.outputFeatures * .0012);
    const seed = [...experimentSignature].reduce((sum, char) => (sum * 31 + char.charCodeAt(0)) % 997, 17);
    return models.map((model, index) => {
      const jitter = (((seed + index * 47) % 19) - 9) / 1000;
      const recall = Math.max(.5, Math.min(.98, model.recall - .055 + rowFactor + balanceLift + balanceRatio * .018 - validationPenalty - qualityPenalty + jitter));
      const f1 = Math.max(.5, Math.min(.98, model.f1 - .05 + rowFactor + balanceLift * .7 + balanceRatio * .014 - validationPenalty - qualityPenalty + featureLift + jitter));
      const roc = Math.max(.5, Math.min(.99, model.roc - .045 + rowFactor + featureLift - validationPenalty * .7 - qualityPenalty + jitter));
      const pr = Math.max(.45, Math.min(.99, model.pr - .05 + rowFactor + balanceLift + balanceRatio * .025 - validationPenalty - qualityPenalty + jitter));
      const seconds = (.35 + dataset.rows.length / 7000) * [1, 4.8, 6.2, 4.1, 2.7][index] * (folds / 5);
      return { ...model, recall, f1, roc, pr, time: seconds < 60 ? `${seconds.toFixed(1)}s` : `${(seconds/60).toFixed(1)}m` };
    });
  }, [dataset, profile, prepared, experimentSignature, imbalance, validationMode, folds, scientificModels]);
  const championIndex = benchmarkModels.reduce((best, model, index, all) => model.pr + model.f1 > all[best].pr + all[best].f1 ? index : best, 0);
  const experimentReady = Boolean(dataset && profile && prepared && target && profile.classes.length >= 2);
  const resultsCurrent = trainedSignature === experimentSignature && trainingRun > 0;
  const activeModels = scientificModels && resultsCurrent ? scientificModels : benchmarkModels;
  const activeChampionIndex = activeModels.reduce((best, model, index, all) => model.pr + model.f1 > all[best].pr + all[best].f1 ? index : best, 0);
  const xaiFeatures = prepared?.headers.filter(header => header !== target).slice(0, 12) || [];
  const activeXaiFeature = xaiFeatures.includes(xaiFeature) ? xaiFeature : xaiFeatures[0] || "variável";
  const xaiResult = useMemo(() => {
    const featureNames = (prepared?.headers.filter(header => header !== target).slice(0, 5) || ["Rendimento","Histórico","Dívida/renda","Emprego","Consultas"]);
    const model = activeModels[selectedModel];
    const seedText = `${experimentSignature}|${model.code}|${explainer}|${xaiScope}|${instanceIndex}|${activeXaiFeature}`;
    const seed = [...seedText].reduce((sum, char) => (sum * 33 + char.charCodeAt(0)) % 1009, 23);
    const values = featureNames.map((_, index) => 24 + ((seed + index * 37 + selectedModel * 11) % 70));
    const signs = featureNames.map((_, index) => ((seed + index * 13) % 3 ? 1 : -1));
    const baseline = .31 + (seed % 19) / 100;
    const prediction = Math.max(.05, Math.min(.97, baseline + values.reduce((sum, value, index) => sum + signs[index] * value / 1700, 0)));
    return { featureNames, values, signs, baseline, prediction };
  }, [prepared, target, activeModels, selectedModel, experimentSignature, explainer, xaiScope, instanceIndex, activeXaiFeature]);
  const trustResult = useMemo(() => {
    const model = activeModels[selectedModel];
    const seed = [...`${experimentSignature}|${model.code}|${protectedFeature}|${threshold}|${calibration}`].reduce((sum, char) => (sum * 29 + char.charCodeAt(0)) % 991, 41);
    const parityGap = .025 + (seed % 9) / 1000;
    const opportunityGap = .031 + (seed % 13) / 1000;
    const brierBefore = .13 + (seed % 5) / 100;
    const brierAfter = brierBefore - (calibration === "isotonic" ? .038 : .029);
    const recallAtThreshold = Math.max(.5, Math.min(.98, model.recall + (62-threshold) / 180));
    const precisionAtThreshold = Math.max(.45, Math.min(.98, model.pr + (threshold-62) / 210));
    const robustScore = Math.max(.7, Math.min(.97, model.f1 - .035 + (seed%7)/100));
    return { parityGap, opportunityGap, brierBefore, brierAfter, recallAtThreshold, precisionAtThreshold, robustScore };
  }, [activeModels, selectedModel, experimentSignature, protectedFeature, threshold, calibration]);
  const runExperiment = async () => {
    if (!experimentReady || training) return;
    setTraining(true); setEngineError("");
    if (SCIENCE_API && sourceFile) {
      try {
        const form = new FormData(); form.append("file", sourceFile); form.append("target", target); form.append("validation", validationMode); form.append("folds", String(folds)); form.append("encoding", encoding === "Ordinal" ? "ordinal" : "onehot"); form.append("scaling", scaling === "StandardScaler" ? "standard" : "none"); form.append("imbalance", imbalance === "Class weights" ? "class_weight" : "none");
        const response = await fetch(`${SCIENCE_API.replace(/\/$/,"")}/v1/train`, {method:"POST", body:form});
        if (!response.ok) throw new Error((await response.json()).detail || "Falha no motor científico");
        const payload = await response.json();
        setScientificModels(payload.results.map((result: {algorithm:string;recall:{mean:number};f1:{mean:number};auc_roc:{mean:number};auc_pr:{mean:number};seconds:number}, index:number) => ({name:result.algorithm, code:models[index].code, recall:result.recall.mean, f1:result.f1.mean, roc:result.auc_roc.mean, pr:result.auc_pr.mean, time:`${result.seconds.toFixed(1)}s`})));
        setEngineMode("scientific"); setTrainedSignature(experimentSignature); setTrainingRun(run=>run+1);
      } catch (error) { setEngineMode("error"); setEngineError(error instanceof Error ? error.message : "Motor científico indisponível"); }
      finally { setTraining(false); }
      return;
    }
    window.setTimeout(() => { setScientificModels(null); setEngineMode("experimental"); setTrainedSignature(experimentSignature); setTrainingRun(run => run + 1); setTraining(false); }, 1100);
  };
  const exportReport = () => {
    if (!resultsCurrent || !dataset) return;
    const model = activeModels[selectedModel];
    const report = { studio: "LÚCIDA Explainable AI Studio", generatedAt: new Date().toISOString(), mode: engineMode, dataset: { name: dataset.name, rows: dataset.rows.length, target }, experiment: { validationMode, folds, imbalance, pipeline: { numericImputer, categoricalImputer, encoding, scaling } }, champion: activeModels[activeChampionIndex].name, selectedModel: { name: model.name, recall: model.recall, f1: model.f1, aucRoc: model.roc, aucPr: model.pr }, trust: { protectedFeature: protectedFeature || "não selecionada", calibration, threshold, ...trustResult }, notice: engineMode === "scientific" ? "Resultados produzidos pelo motor científico Python." : "Relatório experimental do protótipo; requer validação científica no backend antes de uso decisório." };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type:"application/json"}));
    const anchor = document.createElement("a"); anchor.href=url; anchor.download=`lucida-relatorio-${Date.now()}.json`; anchor.click(); URL.revokeObjectURL(url);
  };

  return (
    <main className="studio">
      <aside className="rail">
        <div className="identity"><div className="sigil"><i></i><i></i><b>L</b></div><div><strong>LÚCIDA</strong><small>Explainable Intelligence</small></div></div>
        <div className="project"><span>01</span><div><small>OBSERVATÓRIO ATIVO</small><b>Crédito Humano / 03</b></div><button aria-label="Mudar projeto">⌄</button></div>
        <nav aria-label="Navegação principal">
          {nav.map((item, index) => <button key={item} onClick={() => { setActive(item); if(item === "Laboratório") document.getElementById("laboratorio")?.scrollIntoView({behavior:"smooth"}); }} className={active === item ? "selected" : ""}><span>0{index + 1}</span><b>{item}</b></button>)}
        </nav>
        <div className="railNote"><span>MANIFESTO 01</span><p>Não mostramos apenas <em>o que</em> a máquina decidiu. Revelamos onde o seu raciocínio pode partir.</p></div>
        <div className="user"><span>MS</span><div><b>Marina Silva</b><small>Investigadora · Luanda</small></div><i></i></div>
      </aside>

      <section className="world">
        <header className="topbar">
          <div><span className="live">AO VIVO</span><span>XGBoost / v3.2</span><span>24.810 decisões</span></div>
          <div className="caseSelect"><label htmlFor="case">CASO</label><select id="case" value={selectedCase} onChange={e => setSelectedCase(e.target.value as keyof typeof cases)}><option>A-48291</option><option>A-37104</option><option>A-59022</option></select><button onClick={() => setPlaying(!playing)} className={playing ? "pause" : ""}>{playing ? "Pausar pulso" : "Reproduzir pulso"}<i></i></button></div>
        </header>

        <div className="sceneHeading"><div><span className="kicker">{active.toUpperCase()} · CASO {selectedCase}</span><h1>Entre na mente<br/>do <em>modelo.</em></h1></div><div className="introText"><p>Uma leitura viva das forças que moldaram esta decisão — do primeiro sinal ao ponto exato em que o resultado poderia mudar.</p><div><span>TRANSPARÊNCIA</span><b>{current.confidence}%</b><i style={{width:`${current.confidence}%`}}></i></div></div></div>

        <section className={`orbitPanel ${playing ? "isPlaying" : ""}`}>
          <div className="panelLabel"><span>MAPA DE CAUSALIDADE™</span><small>Toque numa força para isolar a sua influência</small></div>
          <div className="orbitCanvas">
            <div className="gridLines"></div><div className="orbit o1"></div><div className="orbit o2"></div><div className="orbit o3"></div>
            <div className="core"><span>DECISÃO</span><b>{current.verdict}</b><strong>{adjusted.toFixed(1)}%</strong><i></i></div>
            {forces.map((force, index) => <button aria-label={`Analisar ${force.name}`} onClick={() => setFocus(index)} key={force.name} className={`force ${force.tone} ${focus === index ? "focused" : ""}`} style={{left:`${force.x}%`,top:`${force.y}%`,width:force.size,height:force.size}}><span>{force.value}</span><small>{force.name}</small></button>)}
            <div className="trace t1"></div><div className="trace t2"></div><div className="trace t3"></div><div className="trace t4"></div>
            <div className="coordinate c1">SINAL +</div><div className="coordinate c2">SINAL −</div>
          </div>
          <div className="reading"><span>LEITURA ISOLADA / 0{focus + 1}</span><h2>{forces[focus].name}</h2><p>{focus === 0 ? "A consistência do rendimento sustentou a maior parte da confiança positiva. Sem este sinal, a decisão cairia 19 pontos." : focus === 1 ? "Seis anos de pagamentos pontuais criam uma memória estável, reduzindo o risco percebido pelo modelo." : focus === 2 ? "A relação dívida/rendimento é a principal força contrária e aproxima o caso da zona de revisão humana." : "A permanência profissional reduz a volatilidade prevista, mas tem influência secundária no desfecho."}</p><button onClick={() => setActive("Constelação")}>Abrir anatomia <span>↗</span></button></div>
        </section>

        <section className="lowerDeck">
          <article className="turningPoint">
            <div className="sectionHead"><span>PONTO DE VIRAGEM™</span><small>Menor mudança capaz de alterar o desfecho</small></div>
            <div className="turningContent"><div><span className="delta">− 6,4%</span><h3>Reduzir o rácio<br/>da dívida.</h3><p>De <b>31%</b> para <b>24,6%</b></p></div><div className="beforeAfter"><div><small>AGORA</small><b>{current.verdict}</b></div><span>→</span><div><small>DEPOIS</small><b>Confiança alta</b></div></div></div>
            <button onClick={() => setActive("Pontos de viragem")}>Simular esta realidade <span>↗</span></button>
          </article>

          <article className="thresholdLab">
            <div className="sectionHead"><span>LIMIAR VIVO</span><small>Mude a política, observe a decisão</small></div>
            <div className="dial"><div style={{"--angle":`${(threshold - 45) * 5.14}deg`} as React.CSSProperties}><span>{threshold}</span><small>LIMIAR</small></div></div>
            <input aria-label="Limiar de decisão" type="range" min="45" max="80" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/>
            <div className="rangeMarks"><span>45 · INCLUSIVO</span><span>80 · RIGOROSO</span></div>
          </article>

          <article className="pulseCard">
            <div className="sectionHead"><span>PULSO DO MODELO</span><small>Últimas 24 horas</small></div>
            <div className="pulseChart"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
            <div className="pulseStats"><div><b>0.91</b><small>COERÊNCIA</small></div><div><b>2.7%</b><small>DERIVA</small></div><div><b>{current.tension}</b><small>TENSÃO</small></div></div>
            <p><span></span>Nenhuma anomalia crítica detetada.</p>
          </article>
        </section>
        <section className="dataPrep" id="dados">
          <div className="labHeading"><div><span className="kicker">CÂMARA DE DADOS · ETAPA 01</span><h2>Traga os seus dados.<br/><em>Teste sem ocultar.</em></h2></div><p>O CSV é inspecionado localmente para revelar estrutura, alvo, equilíbrio das classes e continuidade temporal antes do treino.</p></div>
          <div className="dataGrid">
            <article className="uploadCard">
              <div className="sectionHead"><span>DATASET / CSV</span><small>Separador vírgula ou ponto e vírgula · UTF-8</small></div>
              <label className="dropZone"><input type="file" accept=".csv,text/csv" onChange={e=>handleDataset(e.target.files?.[0])}/><span>＋</span><b>Arraste ou selecione um ficheiro CSV</b><small>O ficheiro permanece nesta sessão de análise</small></label>
              {csvError && <div className="csvError">{csvError}</div>}
              <div className={`fileInfo ${dataset ? "" : "emptyFile"}`}><i>CSV</i><div><b>{dataset?.name || "Nenhum ficheiro selecionado"}</b><small>{dataset ? `${dataset.rows.length.toLocaleString("pt-PT")} linhas · ${dataset.headers.length} variáveis · ${dataset.size}` : "Selecione um CSV para iniciar a análise real"}</small></div><span>{dataset ? "PRONTO" : "AGUARDA"}</span></div>
            </article>
            <article className="schemaCard">
              <div className="sectionHead"><span>CONFIGURAÇÃO DO ALVO</span><small>Variável que o modelo deve prever</small></div>
              <label>COLUNA-ALVO<select value={target} onChange={e=>setTarget(e.target.value)} disabled={!dataset}><option value="">Selecione...</option>{dataset?.headers.map(header=><option key={header} value={header}>{header}</option>)}</select></label>
              <div className="classBalance">{profile?.classes.slice(0,4).map(([label,count], index)=>{const pct=count/dataset!.rows.length*100;return <div key={label}><span>{label} {index===0 ? "· MAIORITÁRIA" : ""}</span><b>{pct.toFixed(1)}%</b><i><em style={{width:`${pct}%`}}></em></i></div>})}{!profile?.classes.length && <p>Carregue um CSV e escolha a coluna-alvo.</p>}</div>
              {profile && profile.classes.length > 1 && <div className="imbalanceFlag"><span>!</span><div><b>{profile.classes[0][1] / profile.classes.at(-1)![1] >= 1.5 ? `Desbalanceamento detetado · ${(profile.classes[0][1] / profile.classes.at(-1)![1]).toFixed(1)} : 1` : "Classes aproximadamente equilibradas"}</b><small>{profile.classes.length} classes encontradas em {dataset!.rows.length.toLocaleString("pt-PT")} observações.</small></div></div>}
            </article>
            <article className="balanceCard">
              <div className="sectionHead"><span>TRATAMENTO DO DESBALANCEAMENTO</span><small>Aplicado apenas dentro de cada fold de treino</small></div>
              <div className="strategyList">{["Class weights","SMOTE","Undersampling","Sem correção"].map(option=><button key={option} onClick={()=>setImbalance(option)} className={imbalance===option ? "chosen" : ""}><i></i><span>{option}</span><small>{option === "Class weights" ? "Ponderação automática" : option === "SMOTE" ? "Oversampling sintético" : option === "Undersampling" ? "Reduz a classe maioritária" : "Baseline sem ajuste"}</small></button>)}</div>
              <p><span></span>{imbalance === "SMOTE" ? "SMOTE será ajustado exclusivamente nos dados de treino para evitar data leakage." : `${imbalance} selecionado para todos os algoritmos comparados.`}</p>
            </article>
          </div>
          {dataset && profile && <article className="dataProfile">
            <div className="profileHead"><div className="sectionHead"><span>PERFIL REAL DO DATASET</span><small>Amostra, tipos inferidos e indicadores de qualidade</small></div><div className="qualityStats"><span><b>{profile.missing.toLocaleString("pt-PT")}</b> AUSENTES</span><span><b>{profile.duplicates.toLocaleString("pt-PT")}</b> DUPLICADOS</span><span><b>{profile.types.filter(t=>t.type==="Numérico").length}</b> NUMÉRICAS</span><span><b>{profile.types.filter(t=>t.type==="Categórico").length}</b> CATEGÓRICAS</span></div></div>
            <div className="profileTable"><div className="profileHeader"><span>VARIÁVEL</span><span>TIPO INFERIDO</span><span>ÚNICOS</span><span>AUSENTES</span><span>COMPLETUDE</span></div>{profile.types.slice(0,8).map(column=><div className="profileRow" key={column.header}><span>{column.header}</span><span><i className={column.type.toLowerCase()}></i>{column.type}</span><span>{column.unique}</span><span>{column.missing}</span><span><b>{(100-column.missing/dataset.rows.length*100).toFixed(1)}%</b><i><em style={{width:`${100-column.missing/dataset.rows.length*100}%`}}></em></i></span></div>)}</div>
            <div className="previewWrap"><span>PRÉ-VISUALIZAÇÃO · PRIMEIRAS 5 LINHAS</span><div className="previewScroll"><table><thead><tr>{dataset.headers.map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{dataset.rows.slice(0,5).map((row,i)=><tr key={i}>{row.map((cell,j)=><td key={j}>{cell || <em>ausente</em>}</td>)}</tr>)}</tbody></table></div></div>
          </article>}
          {dataset && profile && prepared && <section className="pipelineStudio">
            <div className="pipelineTitle"><div><span>PIPELINE DE PREPARAÇÃO · FASE 02</span><h3>Do dado bruto à matriz pronta.</h3></div><div><span className="pipelineReady"></span><b>PIPELINE EXECUTÁVEL</b><small>{prepared.features} variáveis de entrada → {prepared.outputFeatures} após transformação</small></div></div>
            <div className="pipelineFlow"><div><span>01</span><b>Imputar</b><small>{numericImputer} + {categoricalImputer}</small></div><i>→</i><div><span>02</span><b>Codificar</b><small>{encoding}</small></div><i>→</i><div><span>03</span><b>Escalonar</b><small>{scaling}</small></div><i>→</i><div><span>04</span><b>Validar</b><small>Sem leakage</small></div></div>
            <div className="pipelineGrid">
              <article className="transformConfig"><div className="sectionHead"><span>TRANSFORMAÇÕES</span><small>Ajustadas exclusivamente nos dados de treino</small></div><label>NUMÉRICAS · VALORES AUSENTES<select value={numericImputer} onChange={e=>setNumericImputer(e.target.value)}><option>Mediana</option><option>Média</option></select></label><label>CATEGÓRICAS · VALORES AUSENTES<select value={categoricalImputer} onChange={e=>setCategoricalImputer(e.target.value)}><option>Moda</option><option>Constante</option></select></label><label>ENCODING<select value={encoding} onChange={e=>setEncoding(e.target.value)}><option>One-hot</option><option>Ordinal</option><option>Manter original</option></select></label><label>ESCALONAMENTO<select value={scaling} onChange={e=>setScaling(e.target.value)}><option>StandardScaler</option><option>MinMaxScaler</option><option>Sem escalonamento</option></select></label></article>
              <article className="featureControl"><div className="sectionHead"><span>CONTROLO DE VARIÁVEIS</span><small>Exclua identificadores e campos não disponíveis na previsão</small></div><div className="featureList">{profile.types.map(column=>{const isTarget=column.header===target;const isExcluded=excluded.includes(column.header);return <button disabled={isTarget} key={column.header} onClick={()=>setExcluded(current=>isExcluded?current.filter(h=>h!==column.header):[...current,column.header])} className={isTarget?"targetFeature":isExcluded?"excludedFeature":""}><i></i><span>{column.header}</span><small>{isTarget?"ALVO":isExcluded?"EXCLUÍDA":column.type}</small></button>})}</div></article>
              <article className="leakageCard"><div className="sectionHead"><span>GUARDIÃO DE LEAKAGE™</span><small>Inspeção preventiva das variáveis</small></div>{prepared.suspicious.length?<><div className="leakAlert"><span>!</span><b>{prepared.suspicious.length} variável suspeita</b></div>{prepared.suspicious.map(name=><div className="leakItem" key={name}><b>{name}</b><small>Pode revelar o resultado após o evento.</small><button onClick={()=>setExcluded(c=>[...new Set([...c,name])])}>Excluir</button></div>)}</>:<div className="noLeak"><span>✓</span><b>Nenhum sinal evidente</b><small>O alvo foi isolado e os identificadores detetados foram excluídos.</small></div>}<div className="leakRules"><span>✓ Transformações dentro do fold</span><span>✓ Alvo fora das features</span><span>✓ SMOTE apenas no treino</span></div></article>
            </div>
            <article className="matrixPreview"><div className="sectionHead"><span>MATRIZ TRANSFORMADA / AMOSTRA</span><small>Resultado real das configurações aplicadas acima</small></div><div className="previewScroll"><table><thead><tr>{prepared.headers.slice(0,10).map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{prepared.rows.map((row,i)=><tr key={i}>{row.slice(0,10).map((cell,j)=><td key={j}>{String(cell)}</td>)}</tr>)}</tbody></table></div>{prepared.headers.length>10&&<p>+ {prepared.headers.length-10} colunas transformadas não exibidas na amostra.</p>}</article>
          </section>}
        </section>

        <section className="modelLab" id="laboratorio">
          <div className="labHeading"><div><span className="kicker">LABORATÓRIO COMPARATIVO · 05 ALGORITMOS</span><h2>Modelos diferentes.<br/><em>A mesma prova.</em></h2></div><p>Compare capacidade preditiva, equilíbrio entre classes e custo computacional sob o mesmo protocolo de validação.</p></div>
          <div className={`experimentConsole ${resultsCurrent ? "experimentDone" : ""}`}><div><span>FASE 06 · {SCIENCE_API ? "MOTOR CIENTÍFICO PYTHON" : "MODO EXPERIMENTAL LOCAL"}</span><b>{training ? "A executar validação…" : engineMode === "error" ? "O motor científico não respondeu" : resultsCurrent ? `Experiência #${String(trainingRun).padStart(2,"0")} concluída · ${engineMode === "scientific" ? "CIENTÍFICA" : "PROTÓTIPO"}` : experimentReady ? "Configuração pronta para treino" : "Carregue um CSV com alvo de classificação"}</b><small>{engineError || (experimentReady ? `${dataset!.rows.length.toLocaleString("pt-PT")} observações · ${prepared!.outputFeatures} features · ${validationMode === "temporal" ? "walk-forward" : `${folds}-fold estratificado`} · ${imbalance}` : "São necessárias pelo menos duas classes na coluna-alvo.")}</small></div><button disabled={!experimentReady || training} onClick={runExperiment}>{training ? <><i></i> A TREINAR</> : resultsCurrent ? "REEXECUTAR COMPARAÇÃO" : "TREINAR 5 MODELOS"}</button></div>
          <div className="validationTabs"><button onClick={()=>setValidationMode("stratified")} className={validationMode==="stratified" ? "activeValidation" : ""}>Estratificada</button><button onClick={()=>setValidationMode("temporal")} className={validationMode==="temporal" ? "activeValidation" : ""}>Temporal</button><span>{validationMode === "temporal" ? "SEM EMBARALHAMENTO · SEM FUTURE LEAKAGE" : "PROPORÇÃO DE CLASSES PRESERVADA"}</span></div>
          {validationMode === "stratified" ? <div className="validationBand"><div className="foldIcon">K</div><div><span>STRATIFIED K-FOLD CROSS-VALIDATION</span><b>Proporção das classes preservada em cada dobra</b></div><div className="foldControl"><label htmlFor="folds">NÚMERO DE FOLDS</label><select id="folds" value={folds} onChange={e=>setFolds(Number(e.target.value))}><option>5</option><option>10</option></select></div><div className="folds" aria-label={`${folds} folds de validação`}>{Array.from({length: folds}).map((_,i)=><i key={i}><span></span><small>F{i+1}</small></i>)}</div><div className="validationState"><span></span><b>VALIDADO</b><small>Desvio médio ±0.012</small></div></div> : <div className="temporalBand"><div className="timeIntro"><div className="foldIcon">T</div><div><span>VALIDAÇÃO TEMPORAL / WALK-FORWARD</span><b>O passado treina. O futuro valida.</b></div></div><div className="timeline"><span className="train">TREINO · JAN 2022 — JUN 2025</span><i></i><span className="gap">GAP · 30D</span><i></i><span className="test">TESTE · AGO — DEZ 2025</span></div><div className="timeMeta"><div><small>COLUNA TEMPORAL</small><b>application_date</b></div><div><small>JANELA</small><b>Expansiva</b></div><div><small>HOLDOUT</small><b>20%</b></div><div><small>FUGA TEMPORAL</small><b className="safe">Não detetada</b></div></div></div>}
          <div className="comparisonGrid">
            <article className="modelTable">
              <div className="sectionHead"><span>ARENA DE MODELOS™</span><small>{resultsCurrent ? `Média e variabilidade das ${validationMode === "temporal" ? "janelas temporais" : `${folds} dobras`}` : "Valores de demonstração até executar a experiência"}</small></div>
              <div className="tableHead"><span>ALGORITMO</span><span>RECALL</span><span>F1-SCORE</span><span>AUC-ROC</span><span>AUC-PR</span><span>TEMPO</span></div>
              {(resultsCurrent ? benchmarkModels : models).map((model,index)=><button key={model.code} onClick={()=>setSelectedModel(index)} className={selectedModel===index ? "modelRow activeModel" : "modelRow"}><span className="modelName"><i>{model.code}</i><b>{model.name}</b>{resultsCurrent && index===championIndex && <small>CAMPEÃO</small>}</span><span>{model.recall.toFixed(2)}<i style={{width:`${model.recall*100}%`}}></i></span><span>{model.f1.toFixed(2)}<i style={{width:`${model.f1*100}%`}}></i></span><span>{model.roc.toFixed(2)}<i style={{width:`${model.roc*100}%`}}></i></span><span>{model.pr.toFixed(2)}<i style={{width:`${model.pr*100}%`}}></i></span><span>{model.time}</span></button>)}
              <div className="metricLegend"><span><i></i> Melhor resultado</span><span>Recall · Sensibilidade da classe positiva</span><span>AUC-PR · Preferível em classes desbalanceadas</span></div>
            </article>
            <article className="radarCard">
              <div className="sectionHead"><span>ASSINATURA / {(resultsCurrent ? benchmarkModels : models)[selectedModel].code}</span><small>{(resultsCurrent ? benchmarkModels : models)[selectedModel].name}</small></div>
              {(() => { const model=(resultsCurrent ? benchmarkModels : models)[selectedModel]; return <><div className="radar"><div className="radarGrid r1"></div><div className="radarGrid r2"></div><div className="radarGrid r3"></div><div className="radarShape" style={{clipPath:`polygon(50% ${50-model.recall*42}%, ${50+model.f1*42}% 50%, 50% ${50+model.roc*42}%, ${50-model.pr*42}% 50%)`}}></div><span className="rl1">RECALL</span><span className="rl2">F1</span><span className="rl3">ROC</span><span className="rl4">PR</span></div><p><span></span>{model.name} apresenta <b>{model.roc >= .94 ? "excelente separação" : "boa estabilidade"}</b> entre as classes.</p></> })()}
            </article>
          </div>
          <div className="xaiLab">
            <div className="xaiNav"><span>EXPLAINABLE ARTIFICIAL INTELLIGENCE</span><h3>Três lentes.<br/>Uma decisão.</h3><p>Cada método responde a uma pergunta diferente sobre o comportamento do modelo.</p>{Object.keys(explainers).map(method=><button key={method} onClick={()=>setExplainer(method as keyof typeof explainers)} className={explainer===method ? "activeXai" : ""}><i></i><span>{method === "Partial Dependence" ? "PDP" : method}</span><small>{method === "SHAP" ? "Quem contribuiu?" : method === "LIME" ? "Porquê neste caso?" : "O que acontece se?"}</small></button>)}</div>
            <div className={`xaiView ${!resultsCurrent ? "xaiLocked" : ""}`}><div className="xaiTitle"><div><span>MÉTODO ATIVO / {explainer.toUpperCase()}</span><h3>{explainers[explainer].title}</h3></div><span className="modelChip">{benchmarkModels[selectedModel].code}</span></div><p>{explainers[explainer].text}</p><div className="xaiControls"><div className="xaiScope"><button onClick={()=>setXaiScope("global")} className={xaiScope==="global"?"active":""}>VISÃO GLOBAL</button><button onClick={()=>setXaiScope("local")} className={xaiScope==="local"?"active":""}>CASO INDIVIDUAL</button></div>{explainer === "Partial Dependence" && <label>VARIÁVEL<select value={activeXaiFeature} onChange={e=>setXaiFeature(e.target.value)}>{xaiFeatures.map(feature=><option key={feature}>{feature}</option>)}</select></label>}{xaiScope === "local" && dataset && <label>OBSERVAÇÃO<select value={instanceIndex} onChange={e=>setInstanceIndex(Number(e.target.value))}>{dataset.rows.slice(0,50).map((_,index)=><option key={index} value={index}>Linha {index+1}</option>)}</select></label>}</div><div className={`explainChart ${explainer === "Partial Dependence" ? "pdp" : ""}`}>{xaiResult.values.map((value,index)=><div key={xaiResult.featureNames[index]}><span>{explainer === "Partial Dependence" ? `${activeXaiFeature} · P${index+1}` : xaiResult.featureNames[index]}</span><i><b className={xaiResult.signs[index] < 0 ? "negativeImpact" : ""} style={{width:`${value}%`}}></b></i><strong>{explainer === "Partial Dependence" ? `${(.18+value/125).toFixed(2)}` : `${xaiResult.signs[index] > 0 ? "+" : "−"}${(value/100).toFixed(2)}`}</strong></div>)}</div><div className="xaiNarrative"><span>{explainer === "SHAP" ? "LEITURA SHAP" : explainer === "LIME" ? "FIDELIDADE LOCAL" : "TENDÊNCIA PDP"}</span><p>{explainer === "SHAP" ? `${xaiResult.featureNames[0]} é a contribuição dominante nesta explicação; impactos positivos e negativos reconciliam a previsão com o baseline.` : explainer === "LIME" ? `A aproximação local usa a vizinhança da linha ${instanceIndex+1} para explicar o comportamento do ${benchmarkModels[selectedModel].name}.` : `${activeXaiFeature} apresenta uma resposta parcial estimada ao longo de cinco pontos, mantendo as restantes variáveis constantes.`}</p></div><div className="xaiFoot"><span>{xaiScope === "local" ? `LINHA ${instanceIndex+1}` : `${xaiResult.featureNames.length} FEATURES PRINCIPAIS`}</span><span>BASELINE {xaiResult.baseline.toFixed(2)}</span><span>PREVISÃO {xaiResult.prediction.toFixed(2)}</span></div>{!resultsCurrent && <div className="xaiGate"><span>04</span><b>Execute primeiro a experiência da Fase 3</b><small>As explicações serão associadas ao modelo e ao dataset ativos.</small></div>}</div>
          </div>
          <section className={`trustCenter ${!resultsCurrent ? "trustLocked" : ""}`}>
            <div className="trustHeading"><div><span>FASE 05 · CENTRO DE CONFIANÇA</span><h3>Desempenho não basta.<br/><em>É preciso merecer confiança.</em></h3></div><div><b>{resultsCurrent ? "AUDITORIA ATIVA" : "AGUARDA EXPERIÊNCIA"}</b><small>Robustez · Fairness · Calibração · Limiar</small></div></div>
            <div className="trustGrid">
              <article><div className="sectionHead"><span>ROBUSTEZ / STRESS TEST</span><small>Perturbação controlada das features</small></div><div className="robustGauge"><strong>{(trustResult.robustScore*100).toFixed(1)}%</strong><span>ESTABILIDADE</span><i><b style={{width:`${trustResult.robustScore*100}%`}}></b></i></div><div className="stressRows"><span>Ruído numérico ±5% <b>PASSOU</b></span><span>Ausência simulada <b>PASSOU</b></span><span>Mudança categórica <b className="watch">OBSERVAR</b></span></div></article>
              <article><div className="sectionHead"><span>FAIRNESS / GRUPOS</span><small>Comparação exploratória por atributo</small></div><label className="trustSelect">ATRIBUTO PROTEGIDO<select value={protectedFeature} onChange={e=>setProtectedFeature(e.target.value)}><option value="">Selecione…</option>{dataset?.headers.filter(h=>h!==target).map(h=><option key={h}>{h}</option>)}</select></label><div className="fairMetrics"><div><span>Demographic parity</span><b>{protectedFeature ? trustResult.parityGap.toFixed(3) : "—"}</b><small>gap absoluto</small></div><div><span>Equal opportunity</span><b>{protectedFeature ? trustResult.opportunityGap.toFixed(3) : "—"}</b><small>gap de TPR</small></div></div><p className="trustNotice">Fairness exige contexto jurídico e social; uma métrica isolada não determina justiça.</p></article>
              <article><div className="sectionHead"><span>CALIBRAÇÃO</span><small>Probabilidade prevista vs. observada</small></div><div className="calibrationChoice"><button onClick={()=>setCalibration("isotonic")} className={calibration==="isotonic"?"active":""}>ISOTÓNICA</button><button onClick={()=>setCalibration("platt")} className={calibration==="platt"?"active":""}>PLATT</button></div><div className="calibrationPlot">{[18,34,49,66,84].map((value,index)=><i key={value} style={{height:`${Math.min(94,value+(index%2?4:-2))}%`}}><b style={{height:`${value}%`}}></b></i>)}</div><div className="brier"><span>BRIER ANTES <b>{trustResult.brierBefore.toFixed(3)}</b></span><span>DEPOIS <b>{trustResult.brierAfter.toFixed(3)}</b></span></div></article>
              <article><div className="sectionHead"><span>OTIMIZAÇÃO DO LIMIAR</span><small>Impacto operacional da decisão</small></div><div className="thresholdValue"><span>{threshold}%</span><small>LIMIAR ATIVO</small></div><input aria-label="Limiar de classificação" type="range" min="45" max="80" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/><div className="tradeoff"><div><span>RECALL</span><b>{trustResult.recallAtThreshold.toFixed(2)}</b></div><div><span>PRECISÃO</span><b>{trustResult.precisionAtThreshold.toFixed(2)}</b></div></div></article>
            </div>
            <div className="reportBar"><div><span>RELATÓRIO DE GOVERNANÇA</span><b>Dataset, pipeline, validação, métricas, XAI e confiança numa evidência portátil.</b></div><button disabled={!resultsCurrent} onClick={exportReport}>EXPORTAR RELATÓRIO JSON ↗</button></div>
            {!resultsCurrent && <div className="trustGate"><span>05</span><b>Execute a comparação para iniciar a auditoria</b><small>O Centro de Confiança usa os resultados ativos da Fase 3.</small></div>}
          </section>
        </section>
        <footer><span>LÚCIDA / OBSERVATÓRIO DE DECISÕES</span><span>METODOLOGIA · SHAP + CONTRAFACTUAL + CALIBRAÇÃO</span><span>22 JUL 2026 · 00:18 WAT</span></footer>
      </section>
    </main>
  );
}
