// End-to-end acceptance of the exported site under a GitHub Pages project path.
// Start a local HTTP server at the export's parent directory before running.
import {chromium} from 'playwright-core';

const base=process.env.PAGES_PREVIEW_URL||'http://127.0.0.1:4198/drone_playground/';
const browser=await chromium.launch({
  executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',
  headless:true,
  args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
});
const errors=[];
const wasm=[];
try{
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  page.on('response',response=>{
    if(response.status()>=400)errors.push(`HTTP ${response.status()} ${response.url()}`);
  });
  page.on('request',request=>{if(request.url().endsWith('.wasm'))wasm.push(request.url());});
  await page.goto(base,{waitUntil:'domcontentloaded'});
  await page.waitForURL(new URL('docs/command-builder.html',base).href,{timeout:15000});
  await page.waitForFunction(()=>document.querySelector('#gallery-count')?.textContent?.includes('24 combinations · 30 recorded episodes'),null,{timeout:20000});
  const commands=await page.evaluate(()=>CommandBuilder.combinations().length);
  if(commands!==550)throw Error(`Published command builder lost combinations: ${commands}`);
  await page.evaluate(()=>CommandBuilder.applySelection({
    mode:'play',method:'planner',task:'static_oa',planner:'ego',
    controller:'sampling_mpc',scene:'S06',
  }));
  await page.waitForFunction(()=>document.querySelector('#preview-note')?.textContent?.includes('NO_SOLUTION'),null,{timeout:20000});
  await page.waitForFunction(()=>document.querySelector('#preview-status')?.textContent?.includes('Ready'),null,{timeout:40000});
  const failure=await page.locator('#preview-event').textContent();
  if(failure!=='Method failure')throw Error(`Published S06 event changed: ${failure}`);
  const iframe=page.frameLocator('#preview-frame');
  await iframe.locator('#status').filter({hasText:'321 frames'}).waitFor({state:'attached',timeout:15000});
  await iframe.locator('#timeline').fill('320');
  await iframe.locator('#end-banner').filter({hasText:'NO_SOLUTION'}).waitFor({state:'visible',timeout:15000});
  const publishedUrl=await page.locator('#preview-open').getAttribute('href');
  if(!new URL(publishedUrl).searchParams.get('record')?.startsWith('/drone_playground/results/'))
    throw Error(`Replay escaped Pages base path: ${publishedUrl}`);
  await page.evaluate(()=>CommandBuilder.applySelection({
    mode:'play',method:'planner',task:'dynamic_oa',planner:'ego',
    controller:'sampling_mpc',scene:'D01',
  }));
  await page.waitForFunction(()=>document.querySelector('#preview-scene-name')?.textContent==='D01'&&
    document.querySelector('#preview-status')?.textContent?.includes('Ready'),null,{timeout:50000});
  await iframe.locator('#status').filter({hasText:'350 frames'}).waitFor({state:'attached',timeout:15000});
  const first=Number(await iframe.locator('#timeline').inputValue());
  await page.waitForTimeout(750);
  const second=Number(await iframe.locator('#timeline').inputValue());
  if(second<=first)throw Error(`Public D01 playback stalled: ${first} -> ${second}`);
  await page.setViewportSize({width:390,height:844});
  const mobile=await page.evaluate(()=>({page:document.documentElement.scrollWidth,width:innerWidth}));
  if(mobile.page>mobile.width+2)throw Error(`Public mobile overflow: ${JSON.stringify(mobile)}`);
  if(wasm.length!==1||errors.length)throw Error(JSON.stringify({wasm:wasm.length,errors}));
  console.log(JSON.stringify({commands,publicCases:24,episodes:30,S06:failure,D01:'success playback',wasmRequests:wasm.length,mobile,errors}));
}finally{await browser.close();}
