// Real Chrome acceptance: the Command Builder selects the actual saved WASM
// recording, preserves physical failure statuses, and plays without simulating.
import {mkdirSync} from 'node:fs';
import {chromium} from 'playwright-core';

mkdirSync('../../tmp', {recursive:true});
const browser=await chromium.launch({
  executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',
  headless:true,
  args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
});
const waitFor=async(condition,page,timeout=90000)=>page.waitForFunction(condition,null,{timeout});

try{
  const page=await browser.newPage({viewport:{width:1450,height:1050}});
  const errors=[];
  const wasmRequests=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  page.on('request',request=>{if(request.url().endsWith('.wasm'))wasmRequests.push(request.url());});
  page.on('response',response=>{
    if(response.status()>=400)errors.push(`HTTP ${response.status()} ${response.url()}`);
  });
  const url=process.env.BUILDER_URL||'http://127.0.0.1:4176/docs/command-builder.html';
  await page.goto(url,{waitUntil:'domcontentloaded'});
  await waitFor(()=>document.querySelector('#gallery-count')?.textContent?.includes('256 recorded episodes'),page);
  const summary=await page.locator('#gallery-count').textContent();
  const size=await page.evaluate(()=>CommandBuilder.combinations().length);
  if(size!==550)throw Error(`Builder catalog changed: ${size}`);
  const firstTitle=await page.locator('#gallery-list .recording-card strong').first().textContent();
  if(firstTitle!=='Sampling MPC')throw Error(`Gallery displays a raw controller key: ${firstTitle}`);
  await page.locator('#gallery-search').fill('Full-State NMPC');
  if(!await page.locator('#gallery-list .recording-card').count())
    throw Error('Readable controller labels are not searchable');
  await page.locator('#gallery-search').fill('');

  // Initial Train/Policy selection deliberately displays a labelled task
  // example, not a misattributed policy checkpoint.
  await waitFor(()=>document.querySelector('#preview-note')?.textContent?.includes('NOT a recording'),page);
  const sample=await page.frameLocator('#preview-frame').locator('#status');
  // The embedded viewer intentionally hides its URL/status bar, but retains
  // status in DOM for asynchronous loading and diagnostics.
  await sample.waitFor({state:'attached',timeout:90000});
  await page.frameLocator('#preview-frame').locator('#status').filter({hasText:'Crazyflie mesh geoms'}).waitFor({state:'attached',timeout:90000});
  await page.frameLocator('#preview-frame').locator('#obstacles').filter({hasText:'Translucent'}).waitFor({timeout:90000});
  const firstBefore=Number(await page.frameLocator('#preview-frame').locator('#timeline').inputValue());
  await page.waitForTimeout(950);
  const firstAfter=Number(await page.frameLocator('#preview-frame').locator('#timeline').inputValue());
  if(firstAfter<=firstBefore)throw Error(`Autoplay not advancing: ${firstBefore} -> ${firstAfter}`);

  // Match one specific native command; the historic S06 flight *failed* at
  // 6.4 seconds with NO_SOLUTION. Task Preview must not claim a collision.
  await page.locator('#mode').selectOption('play');
  await page.locator('#method').selectOption('planner');
  await page.locator('#task').selectOption('static_oa');
  await page.locator('#planner').selectOption('ego');
  await page.locator('#controller').selectOption('sampling_mpc');
  await page.locator('#scene').selectOption('S06');
  await waitFor(()=>document.querySelector('#preview-note')?.textContent?.includes('NO_SOLUTION'),page);
  const failureNote=await page.locator('#preview-note').textContent();
  const outcome=await page.locator('#preview-event').textContent();
  if(outcome!=='Method failure'||!failureNote.includes('not a collision'))
    throw Error(`Wrong physical outcome: ${outcome}: ${failureNote}`);
  const persistentFrame=await page.locator('#preview-frame').getAttribute('src');
  if(!persistentFrame.includes('embed=1'))throw Error('Preview did not keep a persistent iframe');
  const frame=page.frameLocator('#preview-frame');
  await frame.locator('#status').filter({hasText:'321 frames'}).waitFor({state:'attached',timeout:90000});
  await page.waitForFunction(()=>
    document.querySelector('#preview-frame')?.contentWindow?.location?.search?.includes('S06-0000'),
    null,{timeout:10000},
  );
  await frame.locator('#timeline').fill('320');
  await frame.locator('#end-banner').filter({hasText:'NO_SOLUTION'}).waitFor({timeout:15000});
  const lastTime=await frame.locator('#time').textContent();
  if(!lastTime.includes('6.400'))throw Error(`Wrong S06 physical duration ${lastTime}`);
  await page.screenshot({path:'../../tmp/task-preview-s06-real-failure.png',fullPage:true});

  // Benchmark core has a separate recorded episode per scene. Selecting S06
  // must not silently retain the earlier successful S02 clip.
  await page.locator('#mode').selectOption('benchmark');
  await waitFor(()=>document.querySelector('#preview-scene')?.options?.length===4,page);
  const defaultCoreScene=await page.locator('#preview-scene').inputValue();
  await page.locator('#preview-scene').selectOption('3');
  await waitFor(()=>document.querySelector('#preview-scene-name')?.textContent==='S06',page);
  if(await page.locator('#preview-event').textContent()!=='Method failure')
    throw Error('Benchmark preview confused core scene status');
  await page.frameLocator('#preview-frame').locator('#status').filter({hasText:'321 frames'}).waitFor({state:'attached',timeout:90000});

  // Selecting a card in the available-combinations gallery also updates the
  // real Command Builder method, not just a detached video iframe.
  await page.locator('#gallery-search').fill('racing_mpcc');
  await page.locator('#gallery-task').selectOption('racing');
  await waitFor(()=>document.querySelectorAll('#gallery-list .recording-card').length===2,page);
  await page.locator('#gallery-list .recording-card').first().click();
  await waitFor(()=>document.querySelector('#task')?.value==='racing'&&
    document.querySelector('#controller')?.value==='racing_mpcc',page);
  await page.frameLocator('#preview-frame').locator('#status').filter({hasText:'350 frames'}).waitFor({state:'attached',timeout:90000});
  const afterFrame=await page.locator('#preview-frame').getAttribute('src');
  if(afterFrame!==persistentFrame)throw Error('Preview unnecessarily reloaded the WebAssembly iframe');
  const exact=await page.locator('#preview-match').textContent();
  if(!exact.includes('Exact'))throw Error('Native MPCC preview did not match exact method');
  const methodLabel=await page.locator('#preview-method').textContent();
  if(!methodLabel.includes('Racing MPCC')||methodLabel.includes('racing_mpcc'))
    throw Error(`Preview shows raw controller key: ${methodLabel}`);
  if((await page.locator('#preview-open').textContent()).includes('↗'))
    throw Error('Viewer link duplicates its decorative external-link arrow');
  if(wasmRequests.length!==1)
    throw Error(`Expected a single persistent WASM request across selections; received ${wasmRequests.length}`);
  const shared=new URL(page.url());
  if(shared.searchParams.get('controller')!=='racing_mpcc'||shared.searchParams.has('scene')||shared.searchParams.has('planner'))
    throw Error('Shareable URL contains mismatched or unused controller/scene selections');
  await page.screenshot({path:'../../tmp/task-preview-racing-mpcc.png',fullPage:true});

  if(errors.length)throw Error(errors.join('\n'));
  console.log(JSON.stringify({url:page.url(),builderCases:size,catalog:summary,autoplay:[firstBefore,firstAfter],S06:{outcome,time:lastTime},benchmarkScene:'S06',racingPreview:exact,wasmRequests:wasmRequests.length,errors}));
}finally{
  await browser.close();
}
