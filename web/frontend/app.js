const I18N = window.OPBenchI18n || {};
const SECTION_IDS = ['pipeline', 'examples', 'results'];

function readPreferredLanguage() {
  try {
    return window.localStorage.getItem('opbench-language') === 'zh' ? 'zh' : 'en';
  } catch {
    return 'en';
  }
}

function ui(key) {
  return I18N[state.language]?.ui?.[key] ?? I18N.en?.ui?.[key] ?? key;
}

function trustedUiHtml(key) {
  return ui(key);
}

function sectionFromHash() {
  const section = window.location.hash.slice(1);
  return SECTION_IDS.includes(section) ? section : null;
}

const state = {
  summary: null,
  results: null,
  examples: {},
  activeSection: sectionFromHash() || 'pipeline',
  exampleCategory: 'irrelevance',
  activeRQ: 'rq1',
  selectedModel: 'GPT-4o-mini',
  deltaMetric: 'OP-Bench',
  language: readPreferredLanguage(),
  loadError: null,
  pipelineIndex: 0,
  pipelineVisibleCount: 1,
  pipelineAnimating: false,
  pipelinePlaying: false,
  pipelineTimer: null,
  pipelineRunId: 0,
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function getJson(path) {
  try {
    const response = await fetch(`./api/${path}`);
    if (response.ok) return response.json();
  } catch {
    // GitHub Pages has no server-side API. Fall back to the static payloads.
  }
  return getStaticJson(path);
}

async function getStaticJson(path) {
  const [resource, queryString = ''] = path.split('?');
  const dataFile = resource === 'examples' ? 'qa_cases.json' : 'research_results.json';
  const response = await fetch(`./data/${dataFile}`);
  if (!response.ok) {
    const error = new Error(`Unable to load static demo data: ${dataFile}`);
    error.statusCode = response.status;
    throw error;
  }
  const data = await response.json();

  if (resource === 'summary') {
    return {
      meta: data.meta,
      overview: data.overview,
      workflow: data.workflow,
    };
  }
  if (resource === 'results') {
    return {
      meta: data.meta,
      rq1: data.rq1,
      rq2: data.rq2,
      rq3: data.rq3,
      rq4: data.rq4,
    };
  }
  if (resource === 'examples') {
    const category = new URLSearchParams(queryString).get('category') || 'irrelevance';
    if (!data.categories?.[category]) {
      const error = new Error(`Unknown static example category: ${category}`);
      error.statusCode = 404;
      throw error;
    }
    return { meta: data.meta, ...data.categories[category] };
  }

  const error = new Error(`Unknown demo resource: ${resource}`);
  error.statusCode = 404;
  throw error;
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function zhData(section, key) {
  return state.language === 'zh' ? I18N.zh?.[section]?.[key] : null;
}

function localizeOverviewCard(card, index) {
  const translation = zhData('overview', 'cards')?.[index];
  return translation ? { ...card, ...translation } : card;
}

function localizeOverviewCategory(category) {
  const translation = zhData('overview', 'categories')?.[category.id];
  if (!translation) return category;
  return {
    ...category,
    ...translation,
    subcategories: category.subcategories.map((subcategory, index) => ({
      ...subcategory,
      ...(translation.subcategories?.[index] || {}),
    })),
  };
}

function localizeWorkflowStep(step) {
  return { ...step, ...(zhData('workflow', step.id) || {}) };
}

function localizeResult(result, id) {
  const translation = zhData('results', id);
  if (!translation) return result;
  const localized = { ...result, ...translation };
  if (translation.evidence) {
    localized.evidence = result.evidence.map((item) => ({
      ...item,
      ...(translation.evidence[item.id] || {}),
    }));
  }
  if (translation.insights) {
    localized.insights = result.insights.map((item, index) => ({
      ...item,
      ...(translation.insights[index] || {}),
    }));
  }
  return localized;
}

function localizeQAItem(item) {
  const translation = zhData('qa', 'items')?.[item.id];
  if (!translation) return item;
  const prompts = Array.isArray(item.prompts) ? item.prompts : [];
  return {
    ...item,
    ...translation,
    prompts: prompts.map((prompt, index) => {
      const promptTranslation = translation.prompts?.[index] || {};
      return {
        ...prompt,
        ...promptTranslation,
        high_answer: { ...prompt.high_answer, ...(promptTranslation.high_answer || {}) },
        low_answer: { ...prompt.low_answer, ...(promptTranslation.low_answer || {}) },
      };
    }),
  };
}

function localizeExampleCategory(category, categoryId = category.id) {
  const translation = zhData('qa', 'categories')?.[categoryId];
  if (!translation) return category;
  return {
    ...category,
    ...translation,
    meta: { ...category.meta, ...(zhData('qa', 'meta') || {}) },
    groups: category.groups.map((group) => {
      const groupTranslation = translation.groups?.[group.id] || {};
      return {
        ...group,
        ...groupTranslation,
        items: group.items.map(localizeQAItem),
      };
    }),
  };
}

function applyLanguage() {
  const isChinese = state.language === 'zh';
  document.documentElement.lang = isChinese ? 'zh-CN' : 'en';
  document.title = ui('pageTitle');

  document.querySelectorAll('[data-i18n]').forEach((element) => {
    const key = element.dataset.i18n;
    const value = ui(key);
    const attribute = element.dataset.i18nAttr;
    if (attribute) {
      element.setAttribute(attribute, value);
    } else {
      element.textContent = value;
    }
  });

  document.querySelectorAll('[data-i18n-html]').forEach((element) => {
    element.innerHTML = trustedUiHtml(element.dataset.i18nHtml);
  });

  const toggle = $('#language-toggle');
  if (toggle) {
    const label = isChinese ? ui('switchToEnglish') : ui('switchToChinese');
    toggle.textContent = label;
    toggle.setAttribute('aria-label', label);
    toggle.setAttribute('aria-pressed', String(isChinese));
    toggle.title = label;
  }
}

function toggleLanguage() {
  state.language = state.language === 'zh' ? 'en' : 'zh';
  try {
    window.localStorage.setItem('opbench-language', state.language);
  } catch {
    // Language selection still works when browser storage is unavailable.
  }
  applyLanguage();
  if (state.summary && state.results) {
    renderAll();
  } else if (state.loadError) {
    renderLoadError();
  }
}

function bindLanguageToggle() {
  const toggle = $('#language-toggle');
  if (!toggle || toggle.dataset.bound === 'true') return;
  toggle.addEventListener('click', toggleLanguage);
  toggle.dataset.bound = 'true';
}

function showError(container, error) {
  if (!container) return;
  const detail = error?.statusCode ? `${ui('requestFailed')}：${error.statusCode}` : ui('requestFailed');
  container.innerHTML = `<div class="error-state">${escapeHtml(ui('dataLoadError'))} ${escapeHtml(detail)}</div>`;
}

function renderLoadError() {
  if (!state.loadError) return;
  showError($('#overview-cards'), state.loadError);
  showError($('#example-content'), state.loadError);
  showError($('#result-content'), state.loadError);
}

async function loadDemo() {
  try {
    const categories = ['irrelevance', 'repetition', 'sycophancy'];
    const [summary, results, ...examplePayloads] = await Promise.all([
      getJson('summary'),
      getJson('results'),
      ...categories.map((category) => getJson(`examples?category=${category}`)),
    ]);
    state.summary = summary;
    state.results = results;
    state.loadError = null;
    categories.forEach((category, index) => {
      state.examples[category] = examplePayloads[index];
    });
    renderAll();
    observeReveals();
  } catch (error) {
    state.loadError = error;
    renderLoadError();
  }
}

function renderAll() {
  renderOverview();
  renderSectionTabs();
  renderPipeline();
  renderExampleTabs();
  renderExamples();
  renderResultTabs();
  renderResults();
  bindGlobalEvents();
  setActiveSection(state.activeSection, {
    scroll: Boolean(sectionFromHash()),
    updateHash: false,
  });
}

function renderOverview() {
  const cards = state.summary.overview.cards;
  $('#overview-cards').innerHTML = cards
    .map((card, index) => {
      const localizedCard = localizeOverviewCard(card, index);
      return `
        <article class="stat-card reveal">
          <span class="stat-value">${escapeHtml(localizedCard.value)}</span>
          <span class="stat-label">${escapeHtml(localizedCard.label)}</span>
          <span class="stat-detail">${escapeHtml(localizedCard.detail)}</span>
        </article>`;
    })
    .join('');

  $('#category-overview').innerHTML = state.summary.overview.categories
    .map((rawCategory) => {
      const category = localizeOverviewCategory(rawCategory);
      return `
        <article class="category-card reveal" data-color="${escapeHtml(category.color)}">
          <div class="category-card-head">
            <span class="category-name"><i class="category-dot"></i>${escapeHtml(category.label)}</span>
            <span class="category-percent">${escapeHtml(category.percent)}%</span>
          </div>
          <p class="category-description">${escapeHtml(category.description)}</p>
          <div class="subcategories">
            ${category.subcategories
              .map((sub) => `<span class="subtag">${escapeHtml(sub.label)} <b>${escapeHtml(sub.count)}</b></span>`)
              .join('')}
            </div>
        </article>`;
    })
    .join('');
}

function renderSectionTabs() {
  const tabs = [
    ['pipeline', '01', ui('sectionPipelineLabel'), ui('sectionPipelineDescription')],
    ['examples', '02', ui('sectionExamplesLabel'), ui('sectionExamplesDescription')],
    ['results', '03', ui('sectionResultsLabel'), ui('sectionResultsDescription')],
  ];
  $('#section-tabs').innerHTML = tabs
    .map(
      ([id, number, label, description]) => `
        <button class="section-tab" type="button" role="tab" aria-selected="${id === state.activeSection}" aria-controls="${id}" data-section-tab="${id}">
          <span class="section-tab-number">${number}</span>
          <span class="section-tab-copy"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(description)}</small></span>
          <span class="section-tab-arrow" aria-hidden="true">↗</span>
        </button>`,
    )
    .join('');
}

let sectionNavigationBound = false;

function bindSectionNavigation() {
  if (sectionNavigationBound) return;
  sectionNavigationBound = true;

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!target || typeof target.closest !== 'function') return;

    const sectionButton = target.closest('[data-section-tab]');
    if (sectionButton) {
      setActiveSection(sectionButton.dataset.sectionTab);
      return;
    }

    const link = target.closest('a[href^="#"]');
    if (!link) return;
    const section = link.getAttribute('href').slice(1);
    if (!SECTION_IDS.includes(section)) return;
    event.preventDefault();
    setActiveSection(section);
  });
}

function setActiveSection(section, { scroll = true, updateHash = true } = {}) {
  if (!SECTION_IDS.includes(section)) return;
  state.activeSection = section;

  document.querySelectorAll('[data-explorer-page]').forEach((page) => {
    const isActive = page.dataset.explorerPage === section;
    page.hidden = !isActive;
    page.classList.toggle('is-active', isActive);
    page.setAttribute('aria-hidden', String(!isActive));
  });

  document.querySelectorAll('[data-section-tab]').forEach((tab) => {
    tab.setAttribute('aria-selected', String(tab.dataset.sectionTab === section));
  });

  if (updateHash && window.location.hash !== `#${section}`) {
    window.history.replaceState(null, '', `#${section}`);
  }

  if (scroll) {
    document.querySelector(`[data-explorer-page="${section}"]`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  }
  observeReveals();
}

function renderPipeline() {
  const steps = state.summary.workflow.map(localizeWorkflowStep);
  const total = steps.length;
  state.pipelineVisibleCount = Math.min(Math.max(state.pipelineVisibleCount, 1), total);
  state.pipelineIndex = Math.min(state.pipelineIndex, state.pipelineVisibleCount - 1);

  $('#pipeline-stage').innerHTML = steps
    .map((step, index) => {
      const isVisible = index < state.pipelineVisibleCount;
      const isCurrent = index === state.pipelineIndex;
      const stepColor = ['var(--coral)', 'var(--violet-bright)', 'var(--mint)'][index % 3];
      const classes = ['flow-step', isVisible ? 'is-visible' : 'is-hidden'];
      if (isCurrent) classes.push('is-current');
      const controlsDisabled = !isVisible || state.pipelinePlaying || state.pipelineAnimating;
      const previousDisabled = index === 0 || controlsDisabled;
      const nextDisabled = index === total - 1 || controlsDisabled;
      const controls = `
        <div class="step-controls">
          <button class="step-previous-button" type="button" data-pipeline-previous="${index}" ${previousDisabled ? 'disabled' : ''}>${escapeHtml(ui('previous'))}</button>
          <button class="step-next-button" type="button" data-pipeline-next="${index}" ${nextDisabled ? 'disabled' : ''}>${escapeHtml(ui('next'))}</button>
          ${index === total - 1 ? `<button class="step-top-button" type="button" data-pipeline-top>${escapeHtml(ui('backToHeading'))}</button>` : ''}
        </div>`;
      const card = `
        <article class="flow-step ${classes.join(' ')}" style="--step-color: ${stepColor}" data-pipeline-card="${index}" aria-hidden="${String(!isVisible)}">
          <div class="step-index">${escapeHtml(step.number)}</div>
          <div class="step-content">
            <p class="step-eyebrow">${escapeHtml(step.eyebrow)}</p>
            <h3 class="stage-title">${escapeHtml(step.title)}</h3>
            <p class="stage-description">${escapeHtml(step.description)}</p>
            <div class="stage-details">
              <span class="stage-details-label">${escapeHtml(ui('whatHappens'))}</span>
              <ul>${step.details.map((detail) => `<li>${escapeHtml(detail)}</li>`).join('')}</ul>
              <div class="stage-output">
                <span class="stage-output-label">${escapeHtml(ui('output'))}</span>
                ${escapeHtml(step.output)}
              </div>
            </div>
            ${controls}
          </div>
        </article>`;
      if (index === total - 1) return card;
      const arrowVisible = index + 1 < state.pipelineVisibleCount;
      return `${card}
        <div class="flow-arrow ${arrowVisible ? 'is-visible' : 'is-hidden'}" data-pipeline-arrow="${index}" aria-hidden="${String(!arrowVisible)}">
          <span class="arrow-line"></span>
          <span class="arrow-head" aria-hidden="true">↓</span>
          <span class="arrow-label">${escapeHtml(ui('nextStage'))} · ${escapeHtml(steps[index + 1].eyebrow)}</span>
        </div>`;
    })
    .join('');
  updatePipelineControls();
}

function updatePipelineControls() {
  const total = state.summary?.workflow?.length || 0;
  const busy = state.pipelinePlaying || state.pipelineAnimating;
  const playButton = $('#play-pipeline');
  const resetButton = $('#reset-pipeline');

  if (playButton) {
    playButton.innerHTML = state.pipelinePlaying
      ? `<span class="play-icon">Ⅱ</span> ${escapeHtml(ui('pauseConstruction'))}`
      : `<span class="play-icon">▶</span> ${escapeHtml(ui('playConstruction'))}`;
    playButton.disabled = !total;
  }
  if (resetButton) resetButton.disabled = state.pipelineAnimating;

  document.querySelectorAll('[data-pipeline-previous]').forEach((button) => {
    const index = Number(button.dataset.pipelinePrevious);
    button.disabled = index === 0 || index >= state.pipelineVisibleCount || busy;
  });
  document.querySelectorAll('[data-pipeline-next]').forEach((button) => {
    const index = Number(button.dataset.pipelineNext);
    button.disabled = index >= total - 1 || index >= state.pipelineVisibleCount || busy;
  });
  document.querySelectorAll('[data-pipeline-top]').forEach((button) => {
    button.disabled = state.pipelineAnimating;
  });
}

function revealElement(element) {
  if (!element) return;
  element.classList.remove('is-hidden');
  element.classList.add('is-new');
  window.requestAnimationFrame(() => {
    element.classList.add('is-visible');
  });
}

function setPipelineFocus(index, { scroll = true } = {}) {
  if (index < 0 || index >= state.pipelineVisibleCount) return;
  state.pipelineIndex = index;
  document.querySelectorAll('[data-pipeline-card]').forEach((card) => {
    card.classList.toggle('is-current', Number(card.dataset.pipelineCard) === index);
  });
  updatePipelineControls();
  if (scroll) {
    document.querySelector(`[data-pipeline-card="${index}"]`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  }
}

async function revealNextPipelineStep(index = state.pipelineVisibleCount, runId = state.pipelineRunId) {
  const total = state.summary?.workflow?.length || 0;
  if (!total || index < 0 || index >= total || state.pipelineAnimating) return false;
  if (index < state.pipelineVisibleCount) {
    setPipelineFocus(index);
    return true;
  }
  if (index !== state.pipelineVisibleCount) return false;

  state.pipelineAnimating = true;
  updatePipelineControls();
  const arrow = document.querySelector(`[data-pipeline-arrow="${index - 1}"]`);
  const card = document.querySelector(`[data-pipeline-card="${index}"]`);
  if (!card) {
    state.pipelineAnimating = false;
    updatePipelineControls();
    return false;
  }

  if (arrow) {
    revealElement(arrow);
    await sleep(330);
  }
  if (runId !== state.pipelineRunId) return false;

  revealElement(card);
  state.pipelineVisibleCount = index + 1;
  state.pipelineIndex = index;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  state.pipelineAnimating = false;
  updatePipelineControls();
  return true;
}

function stopPipelinePlayback() {
  if (state.pipelineTimer !== null) window.clearTimeout(state.pipelineTimer);
  state.pipelineTimer = null;
  state.pipelinePlaying = false;
  state.pipelineAnimating = false;
  state.pipelineRunId += 1;
}

function schedulePipelinePlayback(runId) {
  const total = state.summary?.workflow?.length || 0;
  if (!state.pipelinePlaying || runId !== state.pipelineRunId || state.pipelineVisibleCount >= total) {
    if (runId === state.pipelineRunId) {
      state.pipelinePlaying = false;
      state.pipelineTimer = null;
      updatePipelineControls();
    }
    return;
  }
  state.pipelineTimer = window.setTimeout(async () => {
    state.pipelineTimer = null;
    const revealed = await revealNextPipelineStep(state.pipelineVisibleCount, runId);
    if (runId !== state.pipelineRunId || !state.pipelinePlaying) return;
    if (!revealed || state.pipelineVisibleCount >= total) {
      state.pipelinePlaying = false;
      updatePipelineControls();
      return;
    }
    schedulePipelinePlayback(runId);
  }, 2500);
}

function togglePipelinePlayback() {
  if (state.pipelinePlaying) {
    stopPipelinePlayback();
    renderPipeline();
    return;
  }
  resetPipeline();
  state.pipelinePlaying = true;
  const runId = state.pipelineRunId;
  updatePipelineControls();
  schedulePipelinePlayback(runId);
}

function resetPipeline() {
  stopPipelinePlayback();
  state.pipelineVisibleCount = 1;
  state.pipelineIndex = 0;
  if (state.summary?.workflow?.length) renderPipeline();
}

function handlePipelineNext(index) {
  if (state.pipelinePlaying || state.pipelineAnimating) return;
  const nextIndex = index + 1;
  const total = state.summary?.workflow?.length || 0;
  if (nextIndex >= total) return;
  if (nextIndex < state.pipelineVisibleCount) {
    setPipelineFocus(nextIndex);
    return;
  }
  revealNextPipelineStep(nextIndex);
}

function handlePipelinePrevious(index) {
  if (state.pipelinePlaying || state.pipelineAnimating) return;
  setPipelineFocus(index - 1);
}

function renderExampleTabs() {
  const categories = Object.entries(state.examples);
  $('#example-tabs').innerHTML = categories
    .map(([id, rawCategory]) => {
      const category = localizeExampleCategory(rawCategory, id);
      return `
        <button class="pill-tab" type="button" role="tab" aria-selected="${id === state.exampleCategory}" data-example-category="${escapeHtml(id)}">
          ${escapeHtml(category.label)}
        </button>`;
    })
    .join('');
}

function renderExpandableText(text, className, limit = 520) {
  const value = String(text ?? '').trim();
  if (!value) return '';
  if (value.length <= limit) return `<p class="${className}">${escapeHtml(value)}</p>`;
  return `
    <details class="text-disclosure">
      <summary class="${className}">
        <span>${escapeHtml(value.slice(0, limit).trimEnd())}…</span>
        <small>${escapeHtml(ui('showFull'))}</small>
      </summary>
      <p class="${className}">${escapeHtml(value)}</p>
    </details>`;
}

function renderMemory(memory) {
  const values = Array.isArray(memory) ? memory : [memory];
  const cleaned = values.map((value) => String(value ?? '').trim()).filter(Boolean);
  const noRelevantContext = ui('noRelevantContext');
  const isEmpty = cleaned.length === 1 && (cleaned[0].toUpperCase() === 'NO RELEVANT CONTEXT' || cleaned[0] === noRelevantContext);
  return `
    <div class="memory-block">
      <div class="qa-block-head">
        <span class="qa-block-label">${escapeHtml(ui('retrievedMemory'))}</span>
        <span class="qa-block-meta">${isEmpty ? escapeHtml(ui('noneAvailable')) : `${cleaned.length} ${escapeHtml(ui('items'))}`}</span>
      </div>
      ${isEmpty
        ? `<p class="memory-empty">${escapeHtml(noRelevantContext)}</p>`
        : `<ul class="memory-list">${cleaned
            .map((value) => `<li>${renderExpandableText(value, 'memory-text', 260)}</li>`)
            .join('')}</ul>`}
    </div>`;
}

function renderAnswerCard(answer, variant, reason) {
  const score = Number(answer?.score);
  const scoreText = Number.isFinite(score) ? score.toFixed(2) : '—';
  const isReference = answer?.score_label === 'reference';
  const label = variant === 'high' ? ui('highScoreAnswer') : ui('lowScoreAnswer');
  const source = isReference ? ui('rubricReference') : ui('appendixExample');
  const reasonLabel = variant === 'high' ? ui('whyScoresHigh') : ui('whyScoresLow');
  return `
    <article class="answer-card answer-${variant}">
      <div class="answer-card-head">
        <span class="answer-label">${escapeHtml(label)}</span>
        <span class="answer-score">${scoreText}<small>${escapeHtml(source)}</small></span>
      </div>
      ${renderExpandableText(answer?.text, 'answer-text')}
      <div class="answer-reason">
        <span>${escapeHtml(reasonLabel)}</span>
        <p>${escapeHtml(reason || '')}</p>
      </div>
    </article>`;
}

function renderPromptCase(prompt) {
  return `
    <div class="qa-prompt">
      <div class="qa-prompt-head">
        <span class="qa-prompt-label">${escapeHtml(prompt.label || ui('question'))}</span>
        <p class="qa-question">“${escapeHtml(prompt.question)}”</p>
      </div>
      ${renderMemory(prompt.memory)}
      <div class="answer-compare">
        ${renderAnswerCard(prompt.high_answer, 'high', prompt.high_reason)}
        ${renderAnswerCard(prompt.low_answer, 'low', prompt.low_reason)}
      </div>
    </div>`;
}

function renderQACard(item) {
  const prompts = Array.isArray(item.prompts) ? item.prompts : [];
  return `
    <article class="qa-case ${prompts.length > 1 ? 'is-repetition' : ''}">
      <header class="qa-case-head">
        <div class="qa-case-identity">
          <div class="example-meta">${escapeHtml(item.persona || ui('benchmarkPrompt'))} <span class="risk-chip">${escapeHtml(item.task || '')}</span></div>
          <strong>${escapeHtml(item.method || ui('memoryAugmentedExample'))}</strong>
        </div>
        <span class="qa-source">${escapeHtml(item.source || '')}</span>
      </header>
      <div class="qa-case-body">
        ${prompts.map((prompt) => renderPromptCase(prompt)).join('')}
      </div>
    </article>`;
}

function renderExampleCard(item) {
  const subtype = item.type ? `<span class="risk-chip">${escapeHtml(item.type)}</span>` : '';
  const topic = item.topic ? `<span>${escapeHtml(item.topic)}</span>` : '';
  return `
    <article class="example-card">
      <div class="example-meta">${escapeHtml(item.persona || ui('benchmarkPrompt'))} ${topic} ${subtype}</div>
      <p class="example-question">“${escapeHtml(item.question)}”</p>
      ${item.explanation ? `<p class="example-explanation">${escapeHtml(item.explanation)}</p>` : ''}
    </article>`;
}

function renderDiversityCard(item) {
  const prompts = item.prompts || item.items || [];
  return `
    <article class="example-card">
      <div class="example-meta">${escapeHtml(item.persona || ui('benchmarkPrompt'))} · ${escapeHtml(ui('relatedPromptSet'))}</div>
      ${prompts
        .map(
          (pair) => `
            <div class="diversity-pair">
              <span class="diversity-topic">${escapeHtml(pair.topic || pair.label || ui('openEndedTopic'))}</span>
              <p>“${escapeHtml(pair.question)}”</p>
            </div>`,
        )
        .join('')}
    </article>`;
}

function renderExamples() {
  const rawCategory = state.examples[state.exampleCategory];
  const category = rawCategory ? localizeExampleCategory(rawCategory, state.exampleCategory) : null;
  if (!category) return;
  const sourceNote = category.meta?.note || '';
  $('#example-content').innerHTML = `
    <p class="example-category-intro">${escapeHtml(category.description)}</p>
    ${sourceNote ? `<p class="example-source-note">${escapeHtml(sourceNote)}</p>` : ''}
    <div class="example-groups" data-count="${category.groups.length}">
      ${category.groups
        .map(
          (group) => `
            <article class="example-group reveal">
              <div class="example-group-head">
                <span class="example-group-label">${escapeHtml(group.label)}</span>
                <h3>${escapeHtml(group.definition)}</h3>
                <p class="example-definition">${escapeHtml(group.signal)}</p>
              </div>
              <div class="example-items">
                ${group.items
                  .map((item) => {
                    if (item.prompts) return renderQACard(item);
                    return item.items ? renderDiversityCard(item) : renderExampleCard(item);
                  })
                  .join('')}
              </div>
            </article>`,
        )
        .join('')}
    </div>`;
  observeReveals();
}

function renderResultTabs() {
  const tabs = [
    ['rq1', ui('rq1Tab')],
    ['rq2', ui('rq2Tab')],
    ['rq3', ui('rq3Tab')],
    ['rq4', ui('rq4Tab')],
  ];
  $('#result-tabs').innerHTML = tabs
    .map(
      ([id, label]) => `
        <button class="result-tab" type="button" role="tab" aria-selected="${id === state.activeRQ}" data-result-tab="${id}">${escapeHtml(label)}</button>`,
    )
    .join('');
}

function resultHeading(result, includeModel = false) {
  const models = Object.keys(result.models || {});
  return `
    <div class="result-panel-heading">
      <div>
        <p class="result-kicker">${escapeHtml(result.kicker)}</p>
        <h3>${escapeHtml(result.title)}</h3>
      </div>
      ${includeModel
        ? `<div class="result-control">
            <label for="model-select">${escapeHtml(ui('model'))}</label>
            <select class="model-select" id="model-select">
              ${models.map((model) => `<option ${model === state.selectedModel ? 'selected' : ''}>${escapeHtml(model)}</option>`).join('')}
            </select>
          </div>`
        : ''}
    </div>
    <p class="result-note">${escapeHtml(result.summary)}</p>`;
}

function renderRQ1(result) {
  const rows = result.models[state.selectedModel] || result.models[Object.keys(result.models)[0]];
  const columns = result.columns;
  return `
    <div class="result-panel">
      ${resultHeading(result, true)}
      <div class="bar-chart">
        <div class="bar-chart-head">
          <span>${escapeHtml(ui('averageScore'))}</span>
          <span class="bar-legend"><i class="legend-dot"></i> ${escapeHtml(ui('higherLessOP'))}</span>
        </div>
        ${rows
          .map(
            (row) => `
              <div class="bar-row ${row.method === 'BASE' ? 'is-base' : ''}">
                <span class="bar-label">${escapeHtml(row.method)}</span>
                <span class="bar-track"><i class="bar-fill" style="width: ${row.values[6]}%"></i></span>
                <span class="bar-value">${row.values[6].toFixed(2)}</span>
                ${row.drop ? `<span class="drop-label">−${row.drop.toFixed(1)}% ${escapeHtml(ui('deltaVsBase'))}</span>` : ''}
              </div>`,
          )
          .join('')}
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>${escapeHtml(ui('method'))}</th>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join('')}<th>${escapeHtml(ui('deltaVsBase'))}</th></tr></thead>
          <tbody>
            ${rows
              .map(
                (row) => `
                  <tr>
                    <td class="method-cell">${escapeHtml(row.method)}</td>
                    ${row.values.map((value) => `<td>${value.toFixed(2)}</td>`).join('')}
                    <td class="drop-cell">${row.drop ? `−${row.drop.toFixed(1)}%` : '—'}</td>
                  </tr>`,
              )
              .join('')}
          </tbody>
        </table>
      </div>
      <p class="latency-footnote">${escapeHtml(state.language === 'zh' ? `${ui('sourceTable2')} ${ui('shownFor')}${state.selectedModel}。` : `${ui('sourceTable2')} ${ui('shownFor')} ${state.selectedModel}.`)}</p>
    </div>`;
}

function evidenceVisual(id) {
  if (id === 'attention') {
    return `
      <div class="evidence-visual">
        <span class="attention-bar"></span><span class="attention-bar memory"></span>
        <span class="attention-label query">${escapeHtml(ui('query'))} · 1×</span>
        <span class="attention-label memory-label">${escapeHtml(ui('memory'))} · &gt;2×</span>
      </div>`;
  }
  if (id === 'retrieval') {
    return `
      <div class="retrieval-track">
        <span class="retrieval-node low">${trustedUiHtml('unrelatedQuery')}</span>
        <span class="retrieval-arrow">→</span>
        <span class="retrieval-node">${trustedUiHtml('memoryReturned')}</span>
        <span class="retrieval-arrow">→</span>
        <span class="retrieval-node low">${trustedUiHtml('misleadingContext')}</span>
      </div>`;
  }
  return `
    <div class="collapse-plot">
      <div class="collapse-column wide">
        <i class="plot-dot"></i><i class="plot-dot"></i><i class="plot-dot"></i><i class="plot-dot"></i>
        <span class="collapse-column-label">${escapeHtml(ui('withoutMemory'))}</span>
      </div>
      <div class="collapse-column">
        <i class="plot-dot muted"></i><i class="plot-dot"></i><i class="plot-dot"></i><i class="plot-dot"></i>
        <span class="collapse-column-label">${escapeHtml(ui('withMemory'))}</span>
      </div>
    </div>`;
}

function renderRQ2(result) {
  return `
    <div class="result-panel">
      ${resultHeading(result)}
      <div class="evidence-grid">
        ${result.evidence
          .map(
            (item) => `
              <article class="evidence-card reveal">
                <div class="evidence-card-top">
                  <span class="evidence-label">${escapeHtml(item.label)}</span>
                  <div><span class="evidence-stat">${escapeHtml(item.stat)}</span><span class="evidence-unit">${escapeHtml(item.unit)}</span></div>
                </div>
                <h3>${escapeHtml(item.title)}</h3>
                <p class="evidence-description">${escapeHtml(item.description)}</p>
                ${evidenceVisual(item.id)}
              </article>`,
          )
          .join('')}
      </div>
      <p class="latency-footnote">${escapeHtml(ui('sourceMechanism'))}</p>
    </div>`;
}

function formatDelta(value) {
  if (value > 0) return `+${value.toFixed(1)}%`;
  if (value < 0) return `−${Math.abs(value).toFixed(1)}%`;
  return '0.0%';
}

function renderRQ3(result) {
  const metric = state.deltaMetric;
  const values = result.deltas[metric];
  return `
    <div class="result-panel">
      ${resultHeading(result)}
      <div class="delta-toolbar">
        <div class="delta-toggle" role="tablist" aria-label="${escapeHtml(ui('deltaMetric'))}">
          ${['OP-Bench', 'LoCoMo']
            .map((item) => `<button type="button" role="tab" aria-selected="${item === metric}" data-delta-metric="${item}">${item}</button>`)
            .join('')}
        </div>
        <span class="delta-caption">${escapeHtml(ui('correspondingBaseline'))}</span>
      </div>
      <div class="delta-table">
        <div class="delta-grid">
          <div class="delta-cell delta-head">${escapeHtml(ui('memory'))}</div>
          ${result.methods.map((method) => `<div class="delta-cell delta-head">${escapeHtml(method)}</div>`).join('')}
          ${result.deltaLabels
            .map(
              (label) => `
                <div class="delta-cell delta-method">${escapeHtml(label)}</div>
                ${values[label]
                  .map((value) => `<div class="delta-cell ${value >= 0 ? 'delta-positive' : 'delta-negative'}">${formatDelta(value)}</div>`)
                  .join('')}`,
            )
            .join('')}
        </div>
      </div>
      <div class="insight-grid">
        ${result.insights
          .map(
            (item) => `
              <article class="insight-card">
                <span class="insight-value">${escapeHtml(item.value)}</span>
                <span class="insight-label">${escapeHtml(item.label)}</span>
                <span class="insight-detail">${escapeHtml(item.detail)}</span>
              </article>`,
          )
          .join('')}
      </div>
      <p class="latency-footnote">${escapeHtml(ui('sourceFigure6'))}</p>
    </div>`;
}

function renderRQ4(result) {
  const totals = result.latency.map((row) => row.values.reduce((sum, value) => sum + value, 0));
  const maxTotal = Math.max(...totals);
  return `
    <div class="result-panel">
      ${resultHeading(result)}
      <p class="result-note">${escapeHtml(result.note)}</p>
      <div class="latency-list">
        <div class="latency-legend">
          <span><i class="latency-swatch latency-retrieval"></i> ${escapeHtml(ui('retrieval'))}</span>
          <span><i class="latency-swatch latency-post"></i> ${escapeHtml(ui('postProcessing'))}</span>
          <span><i class="latency-swatch latency-response"></i> ${escapeHtml(ui('response'))}</span>
        </div>
        ${result.latency
          .map((row, index) => {
            const total = totals[index];
            return `
              <div class="latency-row">
                <span class="latency-name">${escapeHtml(row.method)}</span>
                <span class="latency-bar">
                  <i class="latency-segment latency-retrieval" style="width: ${(row.values[0] / maxTotal) * 100}%"></i>
                  <i class="latency-segment latency-post" style="width: ${(row.values[1] / maxTotal) * 100}%"></i>
                  <i class="latency-segment latency-response" style="width: ${(row.values[2] / maxTotal) * 100}%"></i>
                </span>
                <span class="latency-total">${total.toFixed(0)} ${escapeHtml(ui('milliseconds'))}</span>
              </div>`;
          })
          .join('')}
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>${escapeHtml(ui('method'))}</th>${result.columns.map((column) => `<th>${escapeHtml(column)} (${escapeHtml(ui('averageUnit'))})</th>`).join('')}</tr></thead>
          <tbody>
            ${result.latency
              .map((row, index) => {
                const total = totals[index];
                return `<tr><td class="method-cell">${escapeHtml(row.method)}</td>${row.values.map((value) => `<td>${value.toFixed(2)}</td>`).join('')}<td>${total.toFixed(2)}</td></tr>`;
              })
              .join('')}
          </tbody>
        </table>
      </div>
      <p class="latency-footnote">${escapeHtml(ui('sourceAppendixTable11'))}</p>
    </div>`;
}

function renderResults() {
  const rawResult = state.results[state.activeRQ];
  const result = rawResult ? localizeResult(rawResult, state.activeRQ) : null;
  if (!result) return;
  if (state.activeRQ === 'rq1') $('#result-content').innerHTML = renderRQ1(result);
  if (state.activeRQ === 'rq2') $('#result-content').innerHTML = renderRQ2(result);
  if (state.activeRQ === 'rq3') $('#result-content').innerHTML = renderRQ3(result);
  if (state.activeRQ === 'rq4') $('#result-content').innerHTML = renderRQ4(result);
  observeReveals();
}

let globalEventsBound = false;

function bindGlobalEvents() {
  if (globalEventsBound) return;
  globalEventsBound = true;
  $('#play-pipeline').addEventListener('click', togglePipelinePlayback);
  $('#reset-pipeline').addEventListener('click', resetPipeline);
  $('#pipeline-stage').addEventListener('click', (event) => {
    const button = event.target.closest('[data-pipeline-next], [data-pipeline-previous], [data-pipeline-top]');
    if (!button) return;
    if (button.hasAttribute('data-pipeline-next')) {
      handlePipelineNext(Number(button.dataset.pipelineNext));
      return;
    }
    if (button.hasAttribute('data-pipeline-previous')) {
      handlePipelinePrevious(Number(button.dataset.pipelinePrevious));
      return;
    }
    document.querySelector('#pipeline-heading')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  $('#example-tabs').addEventListener('click', (event) => {
    const button = event.target.closest('[data-example-category]');
    if (!button) return;
    state.exampleCategory = button.dataset.exampleCategory;
    renderExampleTabs();
    renderExamples();
  });
  $('#result-tabs').addEventListener('click', (event) => {
    const button = event.target.closest('[data-result-tab]');
    if (!button) return;
    state.activeRQ = button.dataset.resultTab;
    renderResultTabs();
    renderResults();
  });
  $('#result-content').addEventListener('change', (event) => {
    if (event.target.id === 'model-select') {
      state.selectedModel = event.target.value;
      renderResults();
    }
  });
  $('#result-content').addEventListener('click', (event) => {
    const button = event.target.closest('[data-delta-metric]');
    if (!button) return;
    state.deltaMetric = button.dataset.deltaMetric;
    renderResults();
  });
}

window.addEventListener('hashchange', () => {
  const section = sectionFromHash();
  if (section) setActiveSection(section, { updateHash: false });
});

function observeReveals() {
  const items = document.querySelectorAll('.reveal:not([data-observed])');
  if (!('IntersectionObserver' in window)) {
    items.forEach((item) => item.classList.add('is-visible'));
    return;
  }
  const observer = new IntersectionObserver(
    (entries, currentObserver) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        currentObserver.unobserve(entry.target);
      });
    },
    { threshold: 0.12 },
  );
  items.forEach((item) => {
    item.dataset.observed = 'true';
    observer.observe(item);
  });
}

bindLanguageToggle();
applyLanguage();
bindSectionNavigation();
setActiveSection(state.activeSection, { scroll: false, updateHash: false });
loadDemo();
