const $ = (selector, parent = document) => parent.querySelector(selector);
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[character]);
const json = value => JSON.stringify(value, null, 2);
const text = value => typeof value === "string" ? value : value == null ? "—" : json(value);
const asList = value => Array.isArray(value) ? value : value ? [value] : [];
const formNames = {qa: "QA", decision: "Agent 决策", execution: "Agent 执行"};
const statusNames = {implemented: "已有真实样例", planned: "待构建", deferred: "暂缓接入"};
const entryNames = {ask: "追加问答 · 保留原导航目标", continue: "原生续接 · 不新增用户消息", prompt: "新导航请求"};
const reasonNames = {correct:"独立评分通过", incorrect:"答案与标签不符", target_not_created:"目标实例未新增，或原有实例被覆盖", duplicate_target_memory:"同一实例重复登记", unrelated_memory_lost:"原有实例记忆丢失", not_submitted:"未提交", answer_format:"回答格式无效"};
const state = {catalog: null, examples: null, group: "all", form: "all", status: "all", query: "", focused: "5a"};
const selectedExamples = new Map();

function badge(value, type = "") {
  return `<span class="badge ${escapeHTML(type)}">${escapeHTML(value)}</span>`;
}

function paragraph(value) {
  return `<p>${escapeHTML(text(value)).replace(/\n/g, "<br>")}</p>`;
}

function translatedPrompt(original, chinese, sourceLabel = "英文原文（实际模型输入）") {
  const translation = chinese || original;
  return paragraph(translation) + (chinese && chinese !== original
    ? `<details class="original-prompt"><summary>${escapeHTML(sourceLabel)}</summary>${paragraph(original)}</details>` : "");
}

function list(value) {
  const rows = asList(value);
  return rows.length > 1 ? `<ul>${rows.map(row => `<li>${escapeHTML(text(row))}</li>`).join("")}</ul>` : paragraph(rows[0]);
}

function imageFigure(item) {
  if (!item?.src || !/^assets\/[a-zA-Z0-9_.-]+$/.test(item.src)) return "";
  const review = item.scope === "review";
  return `<figure class="image-figure"><button class="image-button" data-image="${escapeHTML(item.src)}" data-caption="${escapeHTML(item.label)}" aria-label="放大：${escapeHTML(item.label)}"><img src="${escapeHTML(item.src)}" alt="${escapeHTML(item.label)}" loading="lazy" width="${Number(item.width) || 640}" height="${Number(item.height) || 480}"></button><figcaption>${review ? '<span class="scope-label">检查者可见</span>' : ""}${escapeHTML(item.label)}</figcaption></figure>`;
}

function memoryTable(entities) {
  if (!entities?.length) return '<p class="review-note">此切点的正式空间记忆为空。</p>';
  return `<table class="memory-table"><thead><tr><th>ID / 类型</th><th>名称 / 地图点</th><th>描述与证据</th></tr></thead><tbody>${entities.map(entity => `<tr><td><code>${escapeHTML(entity.id)}</code><br>${escapeHTML(entity.type)}</td><td>${escapeHTML(entity.name || entity.category || "—")}<br><code>${escapeHTML(entity.point ? entity.point.map(value => Number(value).toFixed(4)).join(", ") : "位置未知")}</code></td><td>${escapeHTML(entity.description || "未写描述")}<br><span class="review-note">证据：${escapeHTML((entity.evidence || []).map(frame => typeof frame === "string" ? frame : frame.frame).join(" / ") || "无")}</span></td></tr>`).join("")}</tbody></table>`;
}

function limitsText(sample) {
  const limits = sample.limits || {}, parts = [];
  if (limits.max_turns) parts.push(`最多 ${limits.max_turns} 轮决策`);
  if (limits.max_requests) parts.push(`${limits.max_requests} 次有效请求`);
  if (sample.form !== "qa" && limits.max_actions) parts.push(`${limits.max_actions} 个原子动作`);
  if (limits.max_path_length) parts.push(`${limits.max_path_length} m 实际路程`);
  if (sample.form === "qa") parts.push("禁止移动／新观察／修改正式记忆");
  return parts.join(" · ");
}

function teacherBlock(sample) {
  if (!sample.teacher?.length) return '<div class="empty-teacher">本条没有教师作答记录。任务素材和参考重放不等于模型示范。</div>';
  return `<div class="teacher-area"><h5>真实教师执行 <span class="review-note">可见正文、实际工具调用与独立评分</span></h5>${sample.teacher.map((attempt, index) => {
    const tools = attempt.tools || [];
    const outputs = asList(attempt.visible_outputs).filter(Boolean);
    return `<section class="attempt"><div class="attempt-heading"><span class="attempt-model">${escapeHTML(attempt.model)} · ${escapeHTML(attempt.effort)} · 尝试 ${index + 1}</span><span class="${attempt.success ? "outcome-pass" : "outcome-fail"}">${attempt.success ? "通过" : "未通过"}</span></div><div class="attempt-body"><div class="example-meta"><span>${escapeHTML(attempt.requests ?? "—")} 次请求</span><span>${escapeHTML(attempt.actions ?? "—")} 个动作</span><span>${Number(attempt.path_length || 0).toFixed(3)} m</span><span>${escapeHTML(reasonNames[attempt.reason] || attempt.reason || attempt.status)}</span></div>${attempt.answer ? `<b>实际提交</b><pre class="teacher-answer">${escapeHTML(text(attempt.answer))}</pre>` : ""}${outputs.length ? `<details class="mini-details"><summary>查看模型可见正文（${outputs.length} 段）</summary><div>${outputs.map(output => `<pre class="teacher-answer">${escapeHTML(output)}</pre>`).join("")}</div></details>` : ""}<details class="mini-details"><summary>按执行顺序查看 ${tools.length} 次工具调用</summary><div>${tools.map((tool, toolIndex) => `<details class="tool-call"><summary>${toolIndex + 1}. ${escapeHTML(tool.name)}${tool.is_error ? " · 工具报错" : ""}</summary><pre>${escapeHTML(json(tool.arguments))}</pre>${tool.reply ? `<span class="review-note">实际回执</span><pre>${escapeHTML(tool.reply)}</pre>` : ""}</details>`).join("")}</div></details>${attempt.images?.length ? `<details class="mini-details"><summary>教师真实行动后的画面（${attempt.images.length} 张）</summary><div class="route-frames">${attempt.images.map(imageFigure).join("")}</div></details>` : ""}${attempt.final_memory ? `<details class="mini-details"><summary>执行结束时的实际记忆（${attempt.final_memory.length} 条）</summary><div>${memoryTable(attempt.final_memory)}</div></details>` : ""}<details class="mini-details"><summary>评分细节与来源记录</summary><pre class="json">${escapeHTML(json({reason:attempt.reason, metrics:attempt.metrics, source:attempt.run}))}</pre></details></div></section>`;
  }).join("")}</div>`;
}

function realExample(sample) {
  const publicImages = (sample.images || []).filter(image => image.scope !== "review");
  const reviewImages = (sample.images || []).filter(image => image.scope === "review");
  const label = sample.label || {};
  const reference = sample.reference || {};
  const referenceFrames = reference.images || reference.keyframes || reference.frames || [];
  const referenceMap = reference.image || reference.map || reference.route_map;
  const question = sample.question || sample.active_request;
  const questionChinese = sample.question ? sample.question_zh : sample.active_request_zh;
  return `<div class="example-meta"><span>${escapeHTML(sample.scene)}</span><span>${escapeHTML(sample.split)}</span><span>${escapeHTML(sample.tool_tier)} 工具</span><span>切点 ${escapeHTML(sample.frame)}</span></div>${sample.entry === "continue" ? '<div class="no-new-user">没有新增“记忆题”提示：模型接着上一条真实工具回执继续执行。下方是切点保留的原导航请求。</div>' : ""}<div class="question"><b>${sample.entry === "continue" ? "原导航请求 · 不是额外干预" : "实际追加给模型的题目"}</b>${translatedPrompt(question, questionChinese)}</div>${sample.entry === "ask" && sample.active_request ? `<details class="mini-details"><summary>问答期间保留的原导航请求与目标图</summary><div>${translatedPrompt(sample.active_request, sample.active_request_zh)}<p class="review-note">本次 submit 仅提交问答答案，原导航目标保留。</p></div></details>` : ""}<div class="image-grid ${publicImages.length === 1 ? "single" : ""}">${publicImages.map(imageFigure).join("")}</div><div class="example-meta"><span>公开位姿：${escapeHTML(text(sample.pose?.pos || []))}</span><span>朝向 ${escapeHTML(sample.pose?.heading)}°</span></div><details class="mini-details"><summary>切点时模型可读取的记忆（${sample.memory?.length || 0} 条）</summary><div>${memoryTable(sample.memory)}</div></details>${sample.quality_note ? `<div class="pilot-note">素材质量提示：${escapeHTML(sample.quality_note)}</div>` : ""}<div class="entry-note">本条预算：${escapeHTML(limitsText(sample))}</div><details class="mini-details"><summary>展开标准答案、核验图与评分依据（仅检查者可见）</summary><div><div class="answer-card"><h5>预期答案／行为</h5><pre>${escapeHTML(text(label.answer))}</pre>${paragraph(label.summary)}<p class="review-note">这些私有标签用于检查与评分，不进入模型输入。</p></div>${reviewImages.length ? `<div class="image-grid ${reviewImages.length === 1 ? "single" : ""}">${reviewImages.map(imageFigure).join("")}</div>` : ""}${referenceMap?.src ? `<div class="image-grid single">${imageFigure(referenceMap)}</div>` : ""}${referenceFrames.length ? `<h5>已实际重放的参考动作（脚本／程序辅助）</h5><div class="route-frames">${referenceFrames.map(imageFigure).join("")}</div>` : ""}${Object.keys(reference).length ? `<details class="mini-details"><summary>参考路线与独立重放记录</summary><pre class="json">${escapeHTML(json(Object.fromEntries(Object.entries(reference).filter(([key]) => !["image","images","keyframes","frames","map","route_map"].includes(key)))))}</pre></details>` : ""}</div></details>${teacherBlock(sample)}<details class="mini-details"><summary>样本来源、前缀与数据边界</summary><pre class="json">${escapeHTML(json({sample_id:sample.id, source_group:sample.group_id, entry:sample.entry, scripted_prefix:sample.prefix, tool_tier:sample.tool_tier, permissions:sample.permissions, limits:sample.limits, budget_note:sample.budget_note, note:"参考轨迹与脚本前缀不承担教师输出的训练loss；实际教师回答和调用才是候选监督。"}))}</pre></details>`;
}

function exampleSection(task) {
  const samples = state.examples.tasks[task.id] || [];
  if (!samples.length) return `<section class="example-section"><h4 class="section-label">输出形式示意</h4><span class="illustration-label">设计例子 · 尚无当前格式的真实样例</span><div class="illustration-grid"><div><h5>${task.entry === "continue" ? "继承的导航请求／触发情形" : "提问／指令示意"}</h5>${translatedPrompt(task.instruction, task.instruction_zh, "英文原文（设计示例）")}</div><div><h5>预期回答／实际操作示意</h5><pre>${escapeHTML(text(task.answer_example))}</pre></div></div></section>`;
  const first = task.id === "5a" ? samples.find(sample => sample.teacher_selected && sample.label?.answer !== "none") || samples[0] : samples[0];
  let selected = selectedExamples.get(task.id) || first.id;
  if (!samples.some(sample => sample.id === selected)) selected = samples[0].id;
  return `<section class="example-section" data-example-section="${escapeHTML(task.id)}"><div class="example-header"><h4>真实样例<span>${samples.length} 条 · 优先显示已做教师验收的样本</span></h4><select class="example-select" data-task="${escapeHTML(task.id)}" aria-label="选择 ${escapeHTML(task.id)} 的真实样例">${samples.map((sample, index) => `<option value="${escapeHTML(sample.id)}"${sample.id === selected ? " selected" : ""}>${index + 1}. ${escapeHTML(sample.scene)}${sample.teacher?.length ? " · 有教师记录" : ""} · ${escapeHTML(sample.id.slice(-6))}</option>`).join("")}</select></div><div class="example-content">${realExample(samples.find(sample => sample.id === selected))}</div></section>`;
}

function taskBody(task) {
  return `<div class="flow-grid"><div class="info-block"><h4>INPUT / 模型得到什么</h4>${paragraph(task.input)}</div><div class="info-block output"><h4>OUTPUT / 要求模型产出什么</h4>${paragraph(task.expected)}</div></div><div class="entry-note">${escapeHTML(entryNames[task.entry] || task.entry)}${task.entry === "continue" ? "：不把私有评分目标写成新提示词，不指定必须采用哪条调用序列。" : ""}</div><div class="scoring"><h4>怎样判对？</h4>${list(task.scoring)}</div>${task.pilot ? `<div class="pilot-note">当前试验实现：${escapeHTML(text(task.pilot))}</div>` : ""}<div class="detail-grid"><div class="subtle-block"><h4>构造素材与标签依赖</h4>${list(task.dependencies || task.source)}</div><div class="subtle-block"><h4>与相近任务的区别／容易混淆的地方</h4>${list(task.contrasts || task.pitfalls)}</div></div>${task.status === "implemented" ? `<details class="mini-details"><summary>预期输出格式示意（非教师轨迹）</summary><pre class="json">${escapeHTML(text(task.answer_example))}</pre></details>` : ""}${exampleSection(task)}<div class="task-footer"><a href="#task-${escapeHTML(task.id)}">${escapeHTML(task.id)} ↗</a> · ${escapeHTML(formNames[task.form])} · ${escapeHTML(statusNames[task.status])}</div>`;
}

function matches(task) {
  return (state.group === "all" || String(task.group) === state.group)
    && (state.form === "all" || task.form === state.form)
    && (state.status === "all" || task.status === state.status)
    && (!state.query || JSON.stringify(task).toLocaleLowerCase().includes(state.query));
}

function renderNavigation() {
  const layers = [...new Set(state.catalog.groups.map(group => group.layer))];
  $("#navigation").innerHTML = `<a href="#all" class="nav-all ${state.group === "all" ? "active" : ""}"><span>▦</span>全部任务<small>${state.catalog.tasks.length}</small></a>${layers.map(layer => `<div class="nav-heading">${escapeHTML(layer)}</div>${state.catalog.groups.filter(group => group.layer === layer).map(group => `<a href="#group-${escapeHTML(group.id)}" class="nav-item ${state.group === String(group.id) ? "active" : ""}"><span class="nav-id">${escapeHTML(group.id)}</span><span>${escapeHTML(group.title)}</span><small>${state.catalog.tasks.filter(task => String(task.group) === String(group.id)).length}</small></a>`).join("")}`).join("")}`;
}

function render() {
  renderNavigation();
  const tasks = state.catalog.tasks.filter(matches);
  $("#results-count").textContent = `显示 ${tasks.length} / ${state.catalog.tasks.length} 个题型 · 点击题型展开输入、输出、评分与例子`;
  const currentGroup = state.catalog.groups.find(group => String(group.id) === state.group);
  $("#breadcrumb").textContent = currentGroup ? `${currentGroup.layer} / ${currentGroup.id}. ${currentGroup.title}` : "全部能力 / 任务总览";
  $("#task-list").innerHTML = tasks.length ? state.catalog.groups.map(group => {
    const members = tasks.filter(task => String(task.group) === String(group.id));
    if (!members.length) return "";
    return `<section class="group-section" id="group-${escapeHTML(group.id)}"><header class="group-header"><div class="group-number">${escapeHTML(group.id)}</div><div><h2>${escapeHTML(group.title)}</h2><p>${escapeHTML(group.summary || group.description || "")}</p></div><span class="group-count">${members.length} 个题型</span></header>${members.map(task => `<details class="task-card" id="task-${escapeHTML(task.id)}" data-task="${escapeHTML(task.id)}"${task.id === state.focused ? " open" : ""}><summary class="task-summary"><span class="task-code">${escapeHTML(task.id)}</span><div class="task-name"><h3>${escapeHTML(task.title)}</h3><p>${escapeHTML(text(task.expected))}</p></div><div class="summary-tags">${badge(formNames[task.form], task.form)}${badge(statusNames[task.status], task.status)}</div><span class="chevron">›</span></summary><div class="task-body">${task.id === state.focused ? taskBody(task) : ""}</div></details>`).join("")}</section>`;
  }).join("") : '<div class="empty-state"><h2>没有匹配的任务</h2><p>试试其他关键词，或切回全部形式／全部设计。</p></div>';
  document.querySelectorAll(".task-card").forEach(card => card.addEventListener("toggle", () => {
    if (card.open && !$(".task-body", card).childElementCount) {
      const task = state.catalog.tasks.find(task => task.id === card.dataset.task);
      $(".task-body", card).innerHTML = taskBody(task);
    }
  }));
}

function route() {
  const hash = decodeURIComponent(location.hash.slice(1));
  state.group = hash.startsWith("group-") ? hash.slice(6) : "all";
  const taskId = hash.startsWith("task-") ? hash.slice(5) : null;
  if (taskId && state.catalog.tasks.some(task => task.id === taskId)) {
    state.focused = taskId;
    state.group = String(state.catalog.tasks.find(task => task.id === taskId).group);
    state.form = "all";
    state.status = "all";
    state.query = "";
    $("#search").value = "";
    $("#status-filter").value = "all";
    document.querySelectorAll("[data-form]").forEach(button => button.classList.toggle("active", button.dataset.form === "all"));
  }
  render();
  $("#sidebar").classList.remove("open");
  if (taskId) requestAnimationFrame(() => $("#task-" + CSS.escape(taskId))?.scrollIntoView({block:"start"}));
  else window.scrollTo({top:0});
}

async function load() {
  const responses = await Promise.all(["catalog.json", "examples.json"].map(async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
  }));
  [state.catalog, state.examples] = responses;
  const atomic = state.catalog.tasks.filter(task => !task.id.startsWith("C")).length;
  $("#stats").innerHTML = [[atomic,"原子任务"],[state.catalog.tasks.length - atomic,"串联任务"],[Object.keys(state.examples.tasks).length,"已构建题型"],[state.examples.counts.samples,"真实样例"]].map(([value,label]) => `<div class="stat"><strong>${value}</strong><span>${label}</span></div>`).join("");
  route();
  $("#search").addEventListener("input", event => { state.query = event.target.value.trim().toLocaleLowerCase(); render(); });
  $("#status-filter").addEventListener("change", event => { state.status = event.target.value; render(); });
  $("#form-filters").addEventListener("click", event => {
    const button = event.target.closest("[data-form]");
    if (!button) return;
    state.form = button.dataset.form;
    $("#form-filters").querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
    render();
  });
  $("#task-list").addEventListener("change", event => {
    if (!event.target.matches(".example-select")) return;
    const task = event.target.dataset.task;
    selectedExamples.set(task, event.target.value);
    $(".example-content", event.target.closest(".example-section")).innerHTML = realExample(state.examples.tasks[task].find(sample => sample.id === event.target.value));
  });
  $("#task-list").addEventListener("click", event => {
    const button = event.target.closest("[data-image]");
    if (!button) return;
    $("#large-image").src = button.dataset.image;
    $("#large-image").alt = button.dataset.caption;
    $("#image-caption").textContent = button.dataset.caption;
    $("#image-dialog").showModal();
  });
  $("#close-image").addEventListener("click", () => $("#image-dialog").close());
  $("#image-dialog").addEventListener("click", event => { if (event.target === event.currentTarget) event.currentTarget.close(); });
  $("#collapse-all").addEventListener("click", () => document.querySelectorAll(".task-card[open]").forEach(card => {card.open = false;}));
  $("#menu-toggle").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#about-toggle").addEventListener("click", () => {
    $("#reading-guide").hidden = !$("#reading-guide").hidden;
    if (!$("#reading-guide").hidden) $("#reading-guide").scrollIntoView({block:"start", behavior:"smooth"});
  });
  window.addEventListener("hashchange", route);
  document.addEventListener("keydown", event => {
    if (event.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) {
      event.preventDefault();
      if (matchMedia("(max-width:760px)").matches) $("#sidebar").classList.add("open");
      $("#search").focus();
    }
    if (event.key === "Escape") $("#sidebar").classList.remove("open");
  });
}

load().catch(error => {
  $("#load-error").hidden = false;
  $("#load-error").textContent = `任务加载失败：${error.message}。请通过提供的网址访问，不要直接双击 HTML 文件。`;
});
