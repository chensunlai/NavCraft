const $ = (q, root = document) => root.querySelector(q);
const $$ = (q, root = document) => [...root.querySelectorAll(q)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value, digits = 0) => Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
const colors = ['#16836c', '#6783a1', '#bd9565', '#9b77a6'];
const titles = ['初步探索与建档','找到床','找到浴缸','找到餐桌','卧室中床对面的电视','备餐区水槽','独立卫生间洗手盆','浴室双台盆','客餐厅电视','衣帽间 → 浴缸','走廊 → 餐桌 → 床','浴室 → 卧室 → 茶几'];
const reasonText = {submitted:'已提交',request_budget:'请求预算耗尽',action_budget:'动作预算耗尽',unsubmitted:'未提交',output_limit:'输出达到上限',timeout:'超时',model_error:'模型接口异常',no_action_output:'无可执行输出'};
const state = {index:null,model:'sol',round:0,detail:null,frame:0,playing:false,timer:null,speed:1,roundFilter:'all',tab:'events',eventFilter:'all',search:'',memory:'after',kind:'all',request:0,showPath:true,showEntities:true};
const cache = new Map();
function model() {return state.index.models.find(m => m.id === state.model) ?? state.index.models[0];}
function outcome(row) {return row.kind === 'explore' || row.success === null ? 'neutral' : row.success ? 'good' : 'bad';}
function label(row) {return outcome(row) === 'neutral' ? '探索建档' : row.success ? '成功' : '失败';}
function toast(text) {$('#toast').textContent=text;$('#toast').classList.add('show');setTimeout(()=>$('#toast').classList.remove('show'),2000);}
function stop() {state.playing=false;clearTimeout(state.timer);const b=$('#play-btn');if(b)b.textContent='▶ 播放';}
function route(modelId, round=0) {stop();location.hash=round ? `${modelId}/${round}` : 'overview';}
function sidebar() {
  $('#model-select').value=state.model;
  $('#overview-btn').classList.toggle('active', !state.round);
  const rows=model().rounds.filter(r=>state.roundFilter!=='failure'||outcome(r)==='bad');
  $('#round-list').innerHTML=rows.map(r=>`<button class="round-item ${state.round===r.number?'active':''}" data-round="${r.number}" title="${esc(r.instruction)}"><span class="round-number">${String(r.number).padStart(2,'0')}</span><span class="round-title">${esc(titles[r.number-1]||r.id)}</span><span class="small-dot ${outcome(r)}"></span></button>`).join('')||'<div class="empty">没有失败任务</div>';
  $$('.round-item').forEach(b=>b.onclick=()=>route(state.model,+b.dataset.round));
  $$('.sidebar-filter button').forEach(b=>b.classList.toggle('active',b.dataset.roundFilter===state.roundFilter));
}
function overview() {
  const models=state.index.models;
  $('#breadcrumbs').textContent='连续导航 / 实验总览';
  $('#main').innerHTML=`<div class="page-heading"><div><div class="eyebrow">CONTINUOUS NAVIGATION · MEMORY</div><h1>一次探索，十二轮导航交互</h1><p class="intro">四个模型在同一场景中完成连续任务。从空记忆开始，记录空间、寻找目标，再从不同位置回访。点击任意结果，查看它实际看到了什么、怎么走、记住了什么。</p></div><button class="outline-btn" id="open-first">查看逐轮回放 ↗</button></div>
  <section class="model-grid">${models.map((m,i)=>`<article class="model-card" data-model="${esc(m.id)}"><div class="model-name"><span class="small-dot" style="background:${colors[i]}"></span>${esc(shortName(m))}<span class="effort">${esc(m.effort)}</span></div><div class="score"><strong>${number(m.successes/m.navigation_total*100,1)}<span style="font-size:18px">%</span></strong><span>${m.successes} / ${m.navigation_total} 成功</span></div><div class="progress-track"><div class="progress-fill" style="width:${m.successes/m.navigation_total*100}%;background:${colors[i]}"></div></div><div class="model-meta"><span>首轮 / 最终实体</span><b>${m.first_entities} → ${m.final_entities}</b></div><div class="model-meta"><span>记忆读取 / 编辑调用</span><b>${m.memory_reads} / ${m.memory_writes}</b></div></article>`).join('')}</section>
  <section class="panel"><div class="panel-head"><div><h2>逐轮结果对比</h2><p>第 1 轮为探索，不计入导航成功率。其余 11 轮均要求提交，并满足到达与指定路线条件。</p></div><div class="legend"><span><i style="background:var(--accent)"></i>成功</span><span><i style="background:var(--red)"></i>失败</span><span><i style="background:#a6aeaa"></i>探索</span></div></div><div class="table-wrap"><table class="heat-table"><thead><tr><th>任务 / 轮次</th>${models.map(m=>`<th>${esc(shortName(m))}</th>`).join('')}</tr></thead><tbody>${models[0].rounds.map((r,j)=>`<tr><td><span>${String(r.number).padStart(2,'0')}</span>${esc(titles[j]||r.id)}${r.number>=10?' <small style="color:#a5ada8">↪ 传送</small>':''}</td>${models.map(m=>{const row=m.rounds.find(x=>x.number===r.number);return `<td><button class="result-cell ${outcome(row)}" data-cell="${m.id}/${r.number}" title="${esc(reasonText[row.reason]||row.reason)} · ${row.actions} 动作">${label(row)} ↗</button></td>`;}).join('')}</tr>`).join('')}</tbody></table></div></section>
  <div class="overview-bottom"><section class="panel"><div class="panel-head"><div><h2>记忆是否持续增长</h2><p>每轮结束时的实体数；数量不等于内容正确，也不证明记忆被有效使用。</p></div></div><div class="chart">${memoryChart(models)}</div><div class="legend chart-legend">${models.map((m,i)=>`<span><i style="background:${colors[i]}"></i>${esc(shortName(m))}</span>`).join('')}</div></section><section class="panel"><div class="panel-head"><h2>这轮实验看到了什么</h2></div><div class="insight"><div class="insight-item"><span class="insight-number">01</span><div><h3>会建档，不代表会持续维护</h3><p>Sol、Astra、Qwen 都在首轮建立了 11 个实体。Sol 后续持续补充；Qwen 的新增与修订集中在少数轮次。</p></div></div><div class="insight-item"><span class="insight-number">02</span><div><h3>到达、提交、路线分别检查</h3><p>部分失败轮次结束时已在目标范围内，但未提交。含途经要求的任务还要检查真实穿门方向与顺序。</p></div></div><div class="insight-item"><span class="insight-number">03</span><div><h3>记忆的质量仍需看内容</h3><p>可以逐条查看实体的坐标、房间归属、连接与证据。可打开的证据不等于物体识别或空间关系正确。</p></div></div></div></section></div><p class="footnote">开发检查 · 单场景、每模型一次连续任务链，12 轮并非 12 个独立样本。SuperTools 使用私有 RGB-D 几何辅助，模型持续获得地图位姿；本页的路径叠图用于离线核验。记忆编辑次数包含失败调用，详情中另列实际成功的操作。</p>`;
  $('#open-first').onclick=()=>route(state.model,1);
  $$('.model-card').forEach(e=>e.onclick=()=>route(e.dataset.model,1));
  $$('[data-cell]').forEach(e=>e.onclick=()=>{location.hash=e.dataset.cell;});
}
function shortName(m) {return ({sol:'Sol',astra:'Astra',deepseek:'DeepSeek 1017',qwen27:'Qwen 27B'})[m.id]||m.label;}
function memoryChart(models) {
  const w=620,h=215,left=32,right=18,top=16,bottom=30;
  const maximum=Math.max(5,...models.flatMap(m=>m.rounds.map(r=>r.entities||0)));
  const ceiling=Math.ceil(maximum/5)*5;const x=n=>left+(n-1)*(w-left-right)/11,y=n=>h-bottom-n/ceiling*(h-top-bottom);
  let out=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="四个模型每轮结束时的实体数量">`;
  for(let n=0;n<=ceiling;n+=5)out+=`<line x1="${left}" x2="${w-right}" y1="${y(n)}" y2="${y(n)}" stroke="#edf0ee"/><text x="22" y="${y(n)+3}" text-anchor="end" fill="#9aa29e" font-size="10">${n}</text>`;
  for(let n=1;n<=12;n++)out+=`<text x="${x(n)}" y="${h-9}" text-anchor="middle" fill="#9aa29e" font-size="10">${n}</text>`;
  models.forEach((m,i)=>{out+=`<polyline points="${m.rounds.map(r=>`${x(r.number)},${y(r.entities||0)}`).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2"/>`;m.rounds.forEach(r=>out+=`<circle cx="${x(r.number)}" cy="${y(r.entities||0)}" r="3" fill="${colors[i]}"><title>${esc(shortName(m))} · 第${r.number}轮 · ${r.entities}个实体</title></circle>`);});return out+'</svg>';
}
async function detailPage() {
  const ticket=++state.request;
  const summary=model().rounds.find(r=>r.number===state.round);
  if(!summary){route(state.model);return;}
  $('#main').innerHTML='<div class="loading">正在读取本轮观察与交互记录…</div>';
  try {
    if(!cache.has(summary.detail)){const response=await fetch(summary.detail);if(!response.ok)throw Error(`HTTP ${response.status}`);cache.set(summary.detail,await response.json());}
    if(ticket!==state.request)return;
    state.detail=cache.get(summary.detail);state.frame=0;state.tab='events';state.eventFilter='all';state.search='';state.memory='after';state.kind='all';
    const d=state.detail,s=d.summary||summary,frames=d.frames||[];
    $('#breadcrumbs').textContent=`连续导航 / ${shortName(model())} / 第 ${state.round} 轮`;
    $('#main').innerHTML=`<div class="page-heading detail-heading"><div><div class="eyebrow">${esc(shortName(model()))} · ${esc(model().effort.toUpperCase())} · ROUND ${String(state.round).padStart(2,'0')}</div><span class="result-badge ${outcome(summary)}">${label(summary)}${summary.kind==='navigation'?' · '+esc(reasonText[s.reason]||s.reason):''}</span><h1>${esc(d.instruction||summary.instruction)}</h1><p class="intro">${s.teleported?'本轮开始前显式传送；保留此前会话和空间记忆。':'沿用上轮的环境位置、会话和空间记忆。'}${state.round===1?' 初始记忆为空。':''}</p></div><div class="round-switch"><button id="prev-round" ${state.round===1?'disabled':''}>← 上一轮</button><button id="next-round" ${state.round===12?'disabled':''}>下一轮 →</button></div></div>
      <div class="metric-strip">${metric('实际动作',s.actions,'步')}${metric('模型请求',s.requests,'次')}${metric('实际路程',number(s.path_length,2),'m')}${metric('终点至成功区域',s.navigation_error===undefined?'—':number(s.navigation_error,3),'m')}${metric('结束时实体',s.memory?.entities??summary.entities,'个')}${metric('工具错误',s.memory?.tool_errors??summary.tool_errors??0,'次')}</div>
      <div class="media-grid"><section class="media-panel"><div class="media-title"><span>第一视角回放</span><small>已保存的真实 RGB · 点击放大</small></div><div class="observation-stage"><img id="current-image" alt="模型观察画面"><span class="frame-label" id="frame-label"></span></div><div class="playback"><input id="frame-slider" aria-label="观察帧进度" type="range" min="0" max="${Math.max(0,frames.length-1)}" value="0"><div class="playback-row"><button class="step-btn" id="prev-frame" aria-label="上一帧">‹</button><button id="play-btn" class="play-btn">▶ 播放</button><button class="step-btn" id="next-frame" aria-label="下一帧">›</button><select id="speed" class="speed-select" aria-label="播放速度">${[.5,1,2,4,8].map(v=>`<option value="${v}" ${v===state.speed?'selected':''}>${v}×</option>`).join('')}</select><span class="frame-count" id="frame-count"></span></div></div><div class="pose-row" id="pose-row"></div></section><section class="media-panel"><div class="media-title"><span>地图与空间记忆</span><small>墙体 + 当前帧位姿</small></div><div class="map-stage"><canvas id="map-canvas" aria-label="墙体、位姿、轨迹与记忆实体"></canvas><span class="map-note">本轮完整轨迹：私有核验；实体：本轮结束时的模型记录</span></div><div class="map-options"><label><input id="show-path" type="checkbox" checked>轨迹</label><label><input id="show-entities" type="checkbox" checked>本轮结束记忆</label><span style="margin-left:auto">红色箭头为当前图片拍摄朝向</span></div><div class="pose-row"><span><b>●</b> 房间</span><span><b style="color:#9c7757">●</b> 物体</span><span><b style="color:#d19a42">━</b> 门</span><span>空白不代表可通行</span></div></section></div>
      <section class="lower-section"><div class="tab-bar"><button data-tab="events" class="active">执行过程<span class="tab-count">${d.events?.length||0}</span></button><button data-tab="memory">空间记忆<span class="tab-count">${d.memory_after?.length||0}</span></button><button data-tab="verdict">判定详情</button></div><div id="tab-content"></div></section>`;
    $('#prev-round').onclick=()=>route(state.model,state.round-1);$('#next-round').onclick=()=>route(state.model,state.round+1);
    $('#frame-slider').oninput=e=>{stop();setFrame(+e.target.value);};$('#prev-frame').onclick=()=>{stop();setFrame(state.frame-1);};$('#next-frame').onclick=()=>{stop();setFrame(state.frame+1);};
    $('#play-btn').onclick=()=>{if(state.playing)stop();else{if(state.frame>=frames.length-1)state.frame=0;state.playing=true;$('#play-btn').textContent='❚❚ 暂停';play();}};
    $('#speed').onchange=e=>{state.speed=+e.target.value;};$('#current-image').onclick=()=>{const f=frames[state.frame];if(f)openImage(f.url,f.id);};
    $('#show-path').onchange=e=>{state.showPath=e.target.checked;drawMap();};$('#show-entities').onchange=e=>{state.showEntities=e.target.checked;drawMap();};
    $('#show-path').checked=state.showPath;$('#show-entities').checked=state.showEntities;
    $$('[data-tab]').forEach(b=>b.onclick=()=>{state.tab=b.dataset.tab;renderTab();});
    setFrame(0);renderTab();
  }catch(error){if(ticket===state.request)$('#main').innerHTML=`<div class="empty">本轮数据读取失败：${esc(error.message)}</div>`;}
}
function metric(name,value,unit) {return `<div class="metric"><label>${name}</label><strong>${esc(value??'—')}<small>${unit}</small></strong></div>`;}
function setFrame(index) {
  const frames=state.detail?.frames||[];if(!frames.length){$('#frame-count').textContent='没有已保存的观察帧';return;}
  state.frame=Math.max(0,Math.min(index,frames.length-1));const frame=frames[state.frame];
  $('#current-image').src=frame.url;$('#frame-slider').value=state.frame;$('#frame-label').textContent=`${frame.id} · 动作 ${frame.step??'—'}`;$('#frame-count').textContent=`${state.frame+1} / ${frames.length} 帧`;
  $('#pose-row').innerHTML=`<span>帧 <b>${esc(frame.id)}</b></span><span>地图位置 <b>${frame.pose?.pos?.map(n=>number(n,4)).join(', ')||'—'}</b></span><span>拍摄朝向 <b>${number(frame.pose?.heading,1)}°</b></span>`;
  $('#prev-frame').disabled=state.frame===0;$('#next-frame').disabled=state.frame===frames.length-1;
  for(const f of frames.slice(state.frame+1,state.frame+3)){const img=new Image();img.src=f.url;}
  drawMap();
}
function play() {if(!state.playing)return;setFrame(state.frame);if(state.frame>=state.detail.frames.length-1){stop();return;}state.timer=setTimeout(()=>{state.frame++;play();},650/state.speed);}
function drawMap() {
  const canvas=$('#map-canvas');if(!canvas||!state.index.map)return;
  const width=canvas.clientWidth||500,height=canvas.clientHeight||375,ratio=window.devicePixelRatio||1;canvas.width=width*ratio;canvas.height=height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.fillStyle='#fcfdfc';ctx.fillRect(0,0,width,height);
  const map=state.index.map,all=map.walls.flat(),uv=p=>[(p[0]-map.origin[0])/map.size_m,(p[1]-map.origin[1])/map.size_m];
  const points=all.map(uv);const minX=Math.min(...points.map(p=>p[0]))-.025,maxX=Math.max(...points.map(p=>p[0]))+.025,minY=Math.min(...points.map(p=>p[1]))-.025,maxY=Math.max(...points.map(p=>p[1]))+.025;
  const scale=Math.min((width-35)/(maxX-minX),(height-35)/(maxY-minY));const ox=(width-(maxX-minX)*scale)/2,oy=(height-(maxY-minY)*scale)/2;
  const pixel=p=>[ox+(p[0]-minX)*scale,oy+(p[1]-minY)*scale];
  ctx.lineCap='round';ctx.lineWidth=1;ctx.strokeStyle='#edf0ed';
  for(let v=Math.ceil(minX*map.size_m)/map.size_m;v<maxX;v+=1/map.size_m){const [x]=pixel([v,minY]);ctx.beginPath();ctx.moveTo(x,oy);ctx.lineTo(x,height-oy);ctx.stroke();}
  for(let v=Math.ceil(minY*map.size_m)/map.size_m;v<maxY;v+=1/map.size_m){const [,y]=pixel([minX,v]);ctx.beginPath();ctx.moveTo(ox,y);ctx.lineTo(width-ox,y);ctx.stroke();}
  if(state.showPath){const path=state.detail.trajectory||[];ctx.strokeStyle='#a8c9bd';ctx.lineWidth=2;ctx.beginPath();path.forEach((p,i)=>{const [x,y]=pixel(uv([p[0],p[2]]));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  ctx.strokeStyle='#515e58';ctx.lineWidth=2;for(const segment of map.walls){ctx.beginPath();segment.forEach((p,i)=>{const [x,y]=pixel(uv(p));i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  if(state.showEntities){const entities=(state.memory==='before'&&state.tab==='memory'?state.detail.memory_before:state.detail.memory_after)||[];
    for(const entity of entities){if(entity.type==='door'&&entity.segment){ctx.strokeStyle='#d19a42';ctx.lineWidth=3;ctx.beginPath();entity.segment.forEach((p,i)=>{const [x,y]=pixel(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
      const point=entity.point||entity.anchor||(entity.segment?.length?[(entity.segment[0][0]+entity.segment[1][0])/2,(entity.segment[0][1]+entity.segment[1][1])/2]:null);if(!point)continue;
      const [x,y]=pixel(point);ctx.fillStyle=entity.type==='room'?'#468a71':entity.type==='door'?'#bd8e43':'#9c7757';ctx.beginPath();ctx.arc(x,y,entity.type==='room'?4:2.6,0,2*Math.PI);ctx.fill();ctx.font='9px sans-serif';ctx.fillText(entity.id,x+5,y-4);
    }
  }
  const frame=state.detail.frames[state.frame];if(frame?.pose?.pos){const [x,y]=pixel(frame.pose.pos),angle=frame.pose.heading*Math.PI/180;ctx.save();ctx.translate(x,y);ctx.rotate(angle);ctx.fillStyle='#c4554f';ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(0,-11);ctx.lineTo(6,7);ctx.lineTo(0,4);ctx.lineTo(-6,7);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();}
  ctx.font='9px sans-serif';ctx.fillStyle='#9ba69f';ctx.fillText('地图上方 ↑',width-67,17);
}
function renderTab() {
  $$('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.tab));
  if(state.tab==='events')renderEvents();else if(state.tab==='memory')renderMemory();else renderVerdict();drawMap();
}
function renderEvents() {
  const d=state.detail;
  $('#tab-content').innerHTML=`<div class="stream-toolbar"><div class="pill-group">${[['all','全部'],['assistant','模型文字'],['tool','工具调用'],['error','仅错误']].map(([v,l])=>`<button class="pill ${state.eventFilter===v?'active':''}" data-event-filter="${v}">${l}</button>`).join('')}</div><input id="event-search" class="stream-search" placeholder="搜索工具、内容或帧号" aria-label="搜索交互记录" value="${esc(state.search)}"></div><div id="event-stream"></div>`;
  $$('[data-event-filter]').forEach(b=>b.onclick=()=>{state.eventFilter=b.dataset.eventFilter;renderEvents();});$('#event-search').oninput=e=>{state.search=e.target.value;eventStream();};eventStream();
}
function eventStream() {
  const rows=(state.detail.events||[]).filter(e=>((state.eventFilter==='all')||(state.eventFilter==='error'?e.error:e.type===state.eventFilter))&&(!state.search||JSON.stringify(e).toLowerCase().includes(state.search.toLowerCase())));
  $('#event-stream').innerHTML=rows.map(e=>{
    if(e.type==='assistant')return `<article class="event assistant"><div class="assistant-title"><span class="assistant-avatar">↟</span>${esc(shortName(model()))}<span class="event-time">${timeLabel(e.time)}</span></div><p class="event-text">${esc(e.text)}</p></article>`;
    if(e.type==='notice')return `<div class="notice">${esc(e.text)}</div>`;
    const result=typeof e.result==='string'?e.result:JSON.stringify(e.result,null,2);
    return `<details class="event" ${e.error?'open':''}><summary class="tool-summary"><span class="tool-name">${esc(e.name)}</span>${e.error?'<span class="tool-error">工具错误</span>':''}<span class="event-time">${timeLabel(e.time)}</span></summary><div class="event-body"><div class="code-label">输入</div><pre>${esc(JSON.stringify(e.args,null,2))}</pre><div class="code-label">实际回执</div><pre>${esc(result)}</pre>${e.frame||e.step!==undefined?`<button class="jump-btn" data-jump="${esc(e.frame||'')}" data-step="${e.step??''}">定位到这次观察 ↗</button>`:''}${e.images?.length?`<div class="event-pictures">${e.images.map(im=>`<div class="event-picture"><img loading="lazy" src="${esc(im.url)}" data-image="${esc(im.url)}" data-caption="${esc(im.label||e.name)}" alt="${esc(im.label||e.name)}"><small>${esc(im.label||e.name)}</small></div>`).join('')}</div>`:''}</div></details>`;
  }).join('')||'<div class="empty">没有匹配的交互记录</div>';
  bindImages($('#event-stream'));
  $$('[data-jump]').forEach(b=>b.onclick=()=>{const frames=state.detail.frames;let index=frames.findIndex(f=>f.id===b.dataset.jump);if(index<0&&b.dataset.step!=='')index=frames.findIndex(f=>f.step>=+b.dataset.step);if(index>=0){stop();setFrame(index);$('.media-grid').scrollIntoView({behavior:'smooth',block:'start'});}});
}
function renderMemory() {
  const d=state.detail,entities=(state.memory==='before'?d.memory_before:d.memory_after)||[];
  $('#tab-content').innerHTML=`<div class="memory-top"><div class="pill-group"><button class="pill ${state.memory==='before'?'active':''}" data-memory="before">本轮开始 · ${d.memory_before?.length||0} 个</button><button class="pill ${state.memory==='after'?'active':''}" data-memory="after">本轮结束 · ${d.memory_after?.length||0} 个</button></div><div class="pill-group">${[['all','全部'],['room','房间'],['door','门'],['object','物体']].map(([v,l])=>`<button class="pill ${state.kind===v?'active':''}" data-kind="${v}">${l}</button>`).join('')}</div></div><p class="memory-help">这里展示模型写入的实体记录。坐标为全局归一化地图坐标；证据按钮可打开原记录关联的图片。新增、修订标记对比本轮开始与结束状态，不表示独立核验通过。</p><div class="entity-grid">${entities.filter(e=>state.kind==='all'||e.type===state.kind).map(e=>entityCard(e)).join('')||'<div class="empty">没有对应的实体记录</div>'}</div>`;
  $$('[data-memory]').forEach(b=>b.onclick=()=>{state.memory=b.dataset.memory;renderMemory();drawMap();});$$('[data-kind]').forEach(b=>b.onclick=()=>{state.kind=b.dataset.kind;renderMemory();});bindImages($('#tab-content'));
}
function entityCard(e) {
  const changes=state.detail.memory_changes||{},mark=(changes.created||[]).includes(e.id)?'新增':(changes.revised||[]).includes(e.id)?'修订':'';
  const location=e.point||e.anchor||e.segment;
  const fields=[`类型 <span>${({room:'房间',door:'门',object:'物体'})[e.type]||esc(e.type)}</span>`,location?`位置 <span>${esc(JSON.stringify(location))}</span>`:'位置 <span>未定位</span>'];
  if(e.room)fields.push(`所属房间 <span>${esc(e.room)}</span>`);
  if(e.connects)fields.push(`连接 <span>${e.connects.map(x=>esc(x||'未知')).join(' ↔ ')}</span>`);
  if(e.passage)fields.push(`通行状态 <span>${esc(e.passage)}</span>`);
  if(e.category)fields.push(`类别 <span>${esc(e.category)}</span>`);
  const evidence=(e.evidence||[]).map(id=>{const f=state.detail.evidence?.[id]||state.detail.frames.find(f=>f.id===id);return `<button class="evidence-btn" ${f?`data-image="${esc(f.url||f)}" data-caption="${esc(e.name+' · '+id)}"`:'disabled'}>${esc(id)} ↗</button>`;}).join('');
  return `<article class="entity-card"><div class="entity-head"><span class="entity-id">${esc(e.id)}</span>${esc(e.name)}${mark&&state.memory==='after'?`<span class="entity-change">${mark}</span>`:''}</div><p>${esc(e.description||'没有文字描述')}</p><div class="entity-fields">${fields.join('<br>')}</div><div class="entity-evidence">${evidence}</div></article>`;
}
function renderVerdict() {
  const s=state.detail.summary,route=s.route;const status=x=>x===undefined||x===null?'不适用':x?'满足':'不满足';
  $('#tab-content').innerHTML=`${s.answer?`<div class="answer-box"><strong>模型提交的回答</strong><br>${esc(s.answer)}</div>`:''}<section class="panel" style="margin-top:0"><div class="panel-head"><div><h2>私有评估结果</h2><p>以下结果供离线检查，没有作为成功反馈发给模型。</p></div></div><table><tbody><tr><td>结束原因</td><td>${esc(reasonText[s.reason]||s.reason)}</td></tr><tr><td>主动提交</td><td>${status(s.submitted)}</td></tr><tr><td>结束时进入目标范围</td><td>${status(s.arrived)}</td></tr><tr><td>到达阈值</td><td>${s.success_distance!==undefined?`${number(s.success_distance,1)} m`:'探索轮，无到达判据'}</td></tr><tr><td>终点测地距离</td><td>${s.navigation_error!==undefined?`${number(s.navigation_error,4)} m`:'—'}</td></tr><tr><td>指定路线方向与顺序</td><td>${route?.events?.length?status(route.passed):'本轮无额外路线约束'}</td></tr><tr><td>SPL</td><td>${number(s.spl,3)}</td></tr></tbody></table></section>${route?.events?.length?`<section class="panel"><div class="panel-head"><h2>路线事件核验</h2></div><table><thead><tr><th>事件</th><th>类型</th><th>是否按顺序发生</th><th>微步路径索引</th></tr></thead><tbody>${route.events.map(e=>`<tr><td>${esc(e.id)}</td><td>${esc(e.kind)}</td><td>${status(e.passed)}</td><td>${e.trajectory_index??'—'}</td></tr>`).join('')}</tbody></table></section>`:''}<p class="memory-help">记忆文件中的证据引用只验证帧号和图片是否存在，尚未逐个核验物体身份、位置和连接。导航成功也不等于整个场景地图构建正确。</p>`;
}
function timeLabel(value) {if(!value)return '';const date=new Date(value);return Number.isNaN(+date)?'':date.toLocaleTimeString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'});}
function openImage(url,caption) {$('#large-image').src=url;$('#large-caption').textContent=caption||'';$('#image-dialog').showModal();}
function bindImages(root) {$$('[data-image]',root).forEach(e=>e.onclick=()=>openImage(e.dataset.image,e.dataset.caption));}
function navigate() {
  stop();const [id,n]=location.hash.slice(1).split('/');
  if(id&&state.index.models.some(m=>m.id===id)){state.model=id;state.round=Math.min(12,Math.max(1,+n||1));}else state.round=0;
  sidebar();document.body.classList.remove('sidebar-open');if(state.round)detailPage();else{state.request++;overview();}window.scrollTo(0,0);
}
$('#overview-btn').onclick=()=>route(state.model);$('#model-select').onchange=e=>{state.model=e.target.value;if(state.round)route(state.model,state.round);else{sidebar();overview();}};
$$('[data-round-filter]').forEach(b=>b.onclick=()=>{state.roundFilter=b.dataset.roundFilter;sidebar();});
$('#menu-btn').onclick=()=>document.body.classList.toggle('sidebar-open');$('#mobile-shade').onclick=()=>document.body.classList.remove('sidebar-open');
$('#close-image').onclick=()=>$('#image-dialog').close();$('#image-dialog').onclick=e=>{if(e.target===$('#image-dialog'))$('#image-dialog').close();};
window.addEventListener('hashchange',navigate);window.addEventListener('resize',()=>{if(state.round)drawMap();});
window.addEventListener('keydown',e=>{if(!state.round||!state.detail||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||$('#image-dialog').open)return;if(e.key==='ArrowLeft'){e.preventDefault();stop();setFrame(state.frame-1);}else if(e.key==='ArrowRight'){e.preventDefault();stop();setFrame(state.frame+1);}else if(e.key===' '){e.preventDefault();$('#play-btn').click();}});
try {const response=await fetch('data/index.json');if(!response.ok)throw Error(`HTTP ${response.status}`);state.index=await response.json();state.index.models.sort((a,b)=>['sol','astra','qwen27','deepseek'].indexOf(a.id)-['sol','astra','qwen27','deepseek'].indexOf(b.id));$('#model-select').innerHTML=state.index.models.map(m=>`<option value="${esc(m.id)}">${esc(shortName(m))} · ${esc(m.effort)}</option>`).join('');navigate();}catch(error){$('#main').innerHTML=`<div class="empty">无法加载实验：${esc(error.message)}。请从网页服务地址打开。</div>`;}
