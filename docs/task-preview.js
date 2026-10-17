// Isaac Lab-style task catalog and authentic, same-origin MuJoCo WASM previews.
// All selections reuse Command Builder's compatible method catalog; no new
// simulation, fabricated policy footage, or inferred flight success is created.
'use strict';

(function(){
  const labels={tracking:'Tracking',racing:'Racing',static_oa:'Static OA',dynamic_oa:'Dynamic OA'};
  const outcomes={
    SUCCESS:'Success',
    METHOD_FAILURE:'Method failure',
    COLLISION:'Collision',
    OUT_OF_BOUNDS:'Out of bounds',
    TIMEOUT:'Task timeout',
    NUMERICAL_FAILURE:'Numerical failure',
    MISSED_GATE:'Missed gate',
  };
  const nativeCatalogUrl='../results/command-builder-native-full-20261017/preview-catalog.json';
  const $=id=>document.getElementById(id);

  function exactlySelected(state,cases){
    if(state.method==='policy'||state.mode==='train')return null;
    return cases.find(item=>
      item.mode===state.mode && item.method===state.method && item.task===state.task &&
      item.controller===state.controller &&
      (state.method!=='planner'||item.planner===state.planner) &&
      (state.mode!=='play'||!['static_oa','dynamic_oa'].includes(state.task)||item.scene===state.scene) &&
      (state.mode!=='benchmark'||!['static_oa','dynamic_oa'].includes(state.task)||item.suite===state.suite)
    )||null;
  }

  function taskExample(state,cases){
    // There are no policy/checkpoint records in this *native* catalog. Show a
    // labelled task-only example with the selected sensor and map where possible.
    const candidates=cases.filter(item=>item.mode==='play'&&item.task===state.task&&item.recordings.length);
    const wantedScene=state.mode==='play'&&['static_oa','dynamic_oa'].includes(state.task)?state.scene:null;
    const preferredSensor=state.sensor;
    const score=item=>
      (wantedScene&&item.scene===wantedScene?100:0)+
      (item.sensor===preferredSensor?40:0)+
      (item.recordings.some(x=>x.event==='SUCCESS')?15:0)+
      (item.controller==='sampling_mpc'?5:0)+
      ((item.scene==='S01'||item.scene==='D01')?3:0);
    return candidates.sort((a,b)=>score(b)-score(a)||a.id-b.id)[0]||null;
  }

  function caseForSelection(state,cases){
    const exact=exactlySelected(state,cases);
    if(exact)return {recordedCase:exact,exact:true};
    if(state.method!=='policy'&&state.mode!=='train')return {recordedCase:null,exact:false};
    return {recordedCase:taskExample(state,cases),exact:false};
  }

  function defaultRecording(item){
    if(!item||!item.recordings.length)return -1;
    const success=item.recordings.findIndex(record=>record.event==='SUCCESS');
    return success>=0?success:0;
  }

  function resultLabel(item){
    if(!item||!item.recordings.length)return 'No recorded flight';
    const tally={};
    for(const row of item.recordings)tally[row.event]=(tally[row.event]||0)+1;
    return Object.entries(tally).map(([name,count])=>`${count} ${outcomes[name]||name}`).join(' · ');
  }

  // Node tests use these pure matching helpers without instantiating a browser.
  globalThis.TaskPreviewCatalog={exactlySelected,taskExample,caseForSelection,defaultRecording,resultLabel};
  if(typeof document==='undefined')return;

  const builder=globalThis.CommandBuilder;
  if(!builder||typeof builder.getSelection!=='function')return;
  const shareableKeys=['mode','method','task','sensor','algorithm','backward','checkpoint','planner','controller','scene','suite'];
  function shareSelection(state){
    const url=new URL(location.href);
    // Keep the supported workflow shareable without putting local checkpoint
    // paths, unused planner settings, private files or recording paths into
    // the page address.
    const relevant=new Set(['mode','task']);
    if(state.mode!=='train')relevant.add('method');
    if(state.method==='policy'||state.mode==='train'){
      relevant.add('sensor');relevant.add('algorithm');
      if(state.mode==='train')relevant.add('backward');
      if(state.mode!=='train'&&state.checkpoint!=='path')relevant.add('checkpoint');
    }else{
      relevant.add('controller');
      if(state.method==='planner')relevant.add('planner');
    }
    if(state.mode==='play'&&['static_oa','dynamic_oa'].includes(state.task))relevant.add('scene');
    if(state.mode==='benchmark'&&['static_oa','dynamic_oa'].includes(state.task))relevant.add('suite');
    for(const key of shareableKeys){
      if(relevant.has(key)&&state[key]!==undefined)url.searchParams.set(key,String(state[key]));
      else url.searchParams.delete(key);
    }
    history.replaceState(null,'',url);
  }
  const frame=$('preview-frame');
  let catalog=null;
  let selection=builder.getSelection();
  let chosenCase=-1;
  let chosenRecording=-1;
  let lastEmbeddedRecord='';
  let scheduledLoad=null;
  let viewerReady=false;
  let nextViewerMessage=null;
  let expectedRecord='';
  let pageLimit=12;

  const loadStages={
    fetch:['Reading flight recording','Fetching recorded poses, timestamps and sensor state',12],
    wasm:['Starting MuJoCo','Initializing the shared native WebAssembly runtime',30],
    assets:['Loading scene assets','Fetching original MJCF, airframe meshes and scene geometry',46],
    scene:['Compiling 3D environment','Preparing native obstacle geometry; dense scenes take longer',66],
    geometry:['Building flight view','Preparing model geometry and trajectory overlays',86],
  };

  function loading(stage,record,detail=''){
    const overlay=$('preview-loading');
    const ready=stage==='ready';
    if(stage==='cleared'||ready){overlay.hidden=true;return;}
    overlay.hidden=false;
    $('preview-retry').hidden=stage!=='error';
    const steps=loadStages[stage]||['Loading recorded flight','Waiting for MuJoCo to prepare the selected environment',8];
    $('preview-loading-title').textContent=stage==='error'?'Unable to load this recording':steps[0];
    $('preview-loading-detail').textContent=stage==='error'?detail:steps[1];
    $('preview-loading-progress').style.width=(stage==='error'?100:steps[2])+'%';
    $('preview-loading-hint').textContent=stage==='error'
      ?'The native record is unchanged. You can retry or open it in the full viewer.'
      :'Task data remains unchanged while the browser prepares the visualization.';
    overlay.dataset.stage=stage;
    $('preview-status').textContent=stage==='error'
      ?'Viewer error: '+detail
      :`Loading ${record?.split('/').at(-2)||'flight'} · ${steps[0]}`;
  }

  function previewMessage(){
    const record=expectedRecord;
    if(!record)return null;
    const item=catalog?.cases.find(c=>c.id===chosenCase);
    const flight=item?.recordings[chosenRecording];
    if(!flight||flight.browser_record!==record)return null;
    return {
      type:'dp-wasm-load',record,
      autoplay:true,outcome:flight.event,reason:flight.failure_detail||'',
      loop:flight.event==='SUCCESS',
    };
  }

  function sendViewerMessage(){
    if(viewerReady&&nextViewerMessage&&frame.contentWindow){
      frame.contentWindow.postMessage(nextViewerMessage,location.origin);
      nextViewerMessage=null;
    }
  }
  window.addEventListener('message',event=>{
    if(event.origin!==location.origin||event.source!==frame.contentWindow)return;
    if(event.data?.type==='dp-wasm-ready'){
      viewerReady=true;
      sendViewerMessage();
    }else if(event.data?.type==='dp-wasm-progress'){
      if(event.data.record!==expectedRecord)return;
      if(event.data.stage==='ready'){
        loading('ready',expectedRecord);
        const item=catalog?.cases.find(c=>c.id===chosenCase);
        const flight=item?.recordings[chosenRecording];
        if(flight) $('preview-status').textContent=`Ready · ${flight.scene} · ${event.data.frames} recorded frames · ${Number(event.data.elapsed_ms/1000).toFixed(1)} s load`;
      }else loading(event.data.stage,event.data.record,event.data.message||'');
    }
  });
  // Do not navigate this iframe for each Task selection. The 10 MB official
  // MuJoCo WASM runtime stays instantiated while new recorded MJCF/qpos files
  // are sent over a same-origin message channel.
  frame.src=new URL('viewer/dist/index.html?embed=1',location.href).href;

  function showLink(id,url){
    const link=$(id);link.hidden=!url;
    if(url)link.href=url;
    else link.removeAttribute('href');
  }

  function resetPreview(note){
    if(scheduledLoad!==null)clearTimeout(scheduledLoad);
    scheduledLoad=null;
    frame.hidden=true;lastEmbeddedRecord='';expectedRecord='';
    $('preview-loading').hidden=true;
    nextViewerMessage={type:'dp-wasm-clear'};
    sendViewerMessage();
    $('preview-placeholder').hidden=false;
    $('preview-placeholder').querySelector('strong').textContent=note;
    $('preview-placeholder').querySelector('p').textContent='Choose a recorded native method or a task example.';
    $('preview-scene-field').hidden=true;
    $('preview-method').textContent='—';$('preview-event').textContent='—';
    $('preview-event').removeAttribute('data-event');
    $('preview-duration').textContent='—';$('preview-scene-name').textContent='—';
    $('preview-match').textContent='No recording';
    for(const id of ['preview-open','preview-rscope','preview-report'])showLink(id,null);
  }

  function viewerUrl(record){
    const url=new URL('viewer/dist/index.html',location.href);
    url.searchParams.set('record',record.browser_record);
    url.searchParams.set('embed','1');
    url.searchParams.set('autoplay','1');
    url.searchParams.set('outcome',record.event);
    if(record.failure_detail)url.searchParams.set('reason',record.failure_detail);
    // A failed flight should stop on its true terminal frame, visibly showing
    // the planner failure. Loop only a genuinely successful task preview.
    if(record.event==='SUCCESS')url.searchParams.set('loop','1');
    return url.href;
  }

  function playbackNote(item,record,exact){
    if(!exact){
      return `Task illustration only: this is ${item.planner?item.planner.toUpperCase()+' + ':''}${item.controller} in ${record.scene}. It is NOT a recording of the selected PPO/BPTT/SHAC policy or checkpoint.`;
    }
    if(record.event==='SUCCESS'){
      return `Completed this actual recorded task at ${record.elapsed_seconds.toFixed(2)} s. The embedded viewer loops this successful flight.`;
    }
    if(record.event==='METHOD_FAILURE'){
      const detail=record.failure_detail||'Method failed to provide a valid control decision';
      return `Flight ended early at ${record.elapsed_seconds.toFixed(2)} s: ${detail}. This is the real end of the recorded episode, not a collision or missing video. No continuation was synthesized.`;
    }
    return `Physical flight ended at ${record.elapsed_seconds.toFixed(2)} s with ${outcomes[record.event]||record.event}. The replay covers that event; it does not imply task success.`;
  }

  function renderRecording(item,index,exact){
    const record=item.recordings[index];
    if(!record){resetPreview('No recorded episode');return;}
    const url=viewerUrl(record);
    $('preview-method').textContent=`${labels[item.task]} · ${item.planner?item.planner.toUpperCase()+' + ':''}${item.controller}`;
    $('preview-event').textContent=outcomes[record.event]||record.event;
    $('preview-event').dataset.event=record.event;
    $('preview-duration').textContent=`${record.elapsed_seconds.toFixed(2)} s / ${record.task_limit_seconds.toFixed(0)} s limit`;
    $('preview-scene-name').textContent=record.scene;
    $('preview-match').textContent=exact?'Exact selected method':'Task example · different method';
    $('preview-status').textContent=`${item.mode.toUpperCase()} · ${record.scene} · Actual ${record.event} · ${item.recordings.length} recorded episode${item.recordings.length===1?'':'s'}`;
    const note=$('preview-note');
    note.textContent=playbackNote(item,record,exact);
    note.dataset.kind=exact&&record.event==='SUCCESS'?'success':exact&&record.event!=='SUCCESS'?'failure':'neutral';
    const fullUrl=new URL(url);
    fullUrl.searchParams.delete('embed');fullUrl.searchParams.delete('autoplay');fullUrl.searchParams.delete('loop');
    showLink('preview-open',fullUrl.href);
    showLink('preview-rscope',record.rscope_record);
    showLink('preview-report',record.report);
    $('preview-placeholder').hidden=true;frame.hidden=false;
    if(lastEmbeddedRecord!==url){
      expectedRecord=record.browser_record;
      loading('pending',expectedRecord);
      lastEmbeddedRecord=url;
      if(scheduledLoad!==null)clearTimeout(scheduledLoad);
      scheduledLoad=setTimeout(()=>{
        nextViewerMessage=previewMessage();
        sendViewerMessage();
        scheduledLoad=null;
      },90);
    }
  }

  function syncPreview(){
    for(const card of document.querySelectorAll('.task-card'))
      card.setAttribute('aria-pressed',String(card.dataset.task===selection.task));
    if(!catalog){resetPreview('Loading recordings…');return;}
    const {recordedCase:item,exact}=caseForSelection(selection,catalog.cases);
    if(!item||!item.recordings.length){
      resetPreview('No matching recording');
      $('preview-status').textContent='This combination is compatible but has no matching saved Episode. Preview is unavailable until a real Play/Benchmark recording exists.';
      $('preview-note').textContent='No other controller or scene will be represented as this method.';
      return;
    }
    if(chosenCase!==item.id){chosenCase=item.id;chosenRecording=defaultRecording(item);}
    const recordSelect=$('preview-scene');
    const entries=item.recordings.map((record,index)=>{
      const description=`${record.scene} · ${outcomes[record.event]||record.event} · ${record.elapsed_seconds.toFixed(2)} s`;
      return new Option(description,String(index));
    });
    recordSelect.replaceChildren(...entries);
    chosenRecording=Math.min(Math.max(0,chosenRecording),item.recordings.length-1);
    recordSelect.value=String(chosenRecording);
    $('preview-scene-field').hidden=item.recordings.length<=1;
    renderRecording(item,chosenRecording,exact);
  }

  function renderGallery(){
    const list=$('gallery-list');list.replaceChildren();
    if(!catalog)return;
    const text=$('gallery-search').value.toLowerCase().trim();
    const task=$('gallery-task').value;
    const outcome=$('gallery-outcome').value;
    const matches=catalog.cases.filter(item=>{
      if(task&&item.task!==task)return false;
      if(outcome&&!item.recordings.some(record=>record.event===outcome))return false;
      const terms=[item.mode,item.method,item.task,item.planner,item.controller,item.scene,item.suite,
        ...item.recordings.flatMap(r=>[r.scene,r.event,r.failure_detail])];
      return !text||terms.join(' ').toLowerCase().includes(text);
    });
    $('gallery-count').textContent=`${matches.length} combinations · ${catalog.summary.recorded_episodes} recorded episodes`;
    for(const item of matches.slice(0,pageLimit)){
      const button=document.createElement('button');button.type='button';button.className='recording-card';
      button.setAttribute('aria-label',`Preview ${item.task}, ${item.planner||''} ${item.controller}, ${item.scene||item.suite||''}`);
      const id=document.createElement('span');id.className='recording-id';
      id.textContent=`#${String(item.id).padStart(3,'0')}  /  ${item.mode.toUpperCase()}`;
      const title=document.createElement('strong');
      title.textContent=`${item.planner?item.planner.toUpperCase()+' + ':''}${item.controller}`;
      const line=document.createElement('span');
      line.textContent=`${labels[item.task]} · ${item.scene||item.suite||'single course'} · ${item.mode}`;
      const summary=document.createElement('span');summary.className='result';
      summary.textContent=resultLabel(item);
      if(item.recordings.some(row=>row.event!=='SUCCESS'))summary.classList.add('fail');
      button.append(id,title,line,summary);
      button.addEventListener('click',()=>{
        chosenCase=-1;chosenRecording=-1;
        const update={mode:item.mode,method:item.method,task:item.task,controller:item.controller};
        if(item.planner)update.planner=item.planner;
        if(item.scene)update.scene=item.scene;
        if(item.suite)update.suite=item.suite;
        builder.applySelection(update);
        $('preview-heading').scrollIntoView({behavior:'smooth',block:'start'});
      });
      list.appendChild(button);
    }
    $('gallery-more').hidden=matches.length<=pageLimit;
    $('gallery-more').textContent=`Show more (${Math.max(0,matches.length-pageLimit)} remaining)`;
  }

  document.addEventListener('drone-playground:selection',event=>{
    selection=event.detail;
    shareSelection(selection);
    syncPreview();
  });
  for(const card of document.querySelectorAll('.task-card'))card.addEventListener('click',()=>{
    const task=card.dataset.task;
    const partial={task};
    if(selection.mode!=='train'&&selection.method!=='policy')
      partial.method=['static_oa','dynamic_oa'].includes(task)?'planner':'controller';
    builder.applySelection(partial);
  });
  $('preview-scene').addEventListener('change',event=>{
    chosenRecording=Number(event.target.value);
    const chosen=catalog?.cases.find(item=>item.id===chosenCase);
    if(chosen)renderRecording(chosen,chosenRecording,caseForSelection(selection,catalog.cases).exact);
  });
  for(const id of ['gallery-search','gallery-task','gallery-outcome'])
    $(id).addEventListener(id==='gallery-search'?'input':'change',()=>{pageLimit=12;renderGallery();});
  $('gallery-more').addEventListener('click',()=>{pageLimit+=12;renderGallery();});
  $('preview-retry').addEventListener('click',()=>{
    const message=previewMessage();
    if(!message)return;
    loading('pending',expectedRecord);
    nextViewerMessage=message;
    sendViewerMessage();
  });

  const incoming=new URL(location.href).searchParams;
  const restored={};
  const allowed={
    mode:['train','play','benchmark'],
    method:['policy','controller','planner'],
    task:Object.keys(labels),
    sensor:['state','depth','lidar'],
    algorithm:['bptt','shac','ppo'],
    backward:['native','lotf','point_mass_lag'],
    checkpoint:['latest','best'],
    planner:['ego','super'],
    controller:Object.keys(builder.controllerCapabilities),
  };
  for(const key of shareableKeys){
    const value=incoming.get(key);
    if(value&&(allowed[key]?allowed[key].includes(value):/^[a-zA-Z0-9_-]+$/.test(value)))
      restored[key]=value;
  }
  if(Object.keys(restored).length)builder.applySelection(restored);

  (async()=>{
    try{
      if(location.protocol==='file:')throw new Error('Start a local HTTP server to browse saved MuJoCo recordings.');
      const response=await fetch(new URL(nativeCatalogUrl,location.href),{cache:'no-store'});
      if(!response.ok)throw new Error(`Recording index HTTP ${response.status}. Generate it with: python tools/build_task_preview_catalog.py`);
      const loaded=await response.json();
      if(loaded.format!=='drone-playground-task-preview-v1'||!Array.isArray(loaded.cases))
        throw new Error('Unsupported recording catalog format');
      catalog=loaded;
      $('hero-native-count').textContent=String(catalog.summary.native_methods);
      $('hero-episode-count').textContent=String(catalog.summary.recorded_episodes);
      syncPreview();renderGallery();
    }catch(error){
      resetPreview('Preview recordings unavailable');
      $('preview-status').textContent=error.message;
      $('preview-note').textContent='Command generation still works without the optional local recordings. This catalog is generated from real results, not bundled with GitHub.';
      $('gallery-count').textContent='No local recordings found';
    }
  })();
})();
