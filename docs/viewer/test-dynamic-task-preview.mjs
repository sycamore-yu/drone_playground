// Regression of the reported dynamic_oa / EGO + Sampling MPC / D01 stall.
// Real 6.6 MB native flight record, original MJCF and persistent MuJoCo WASM.
import {mkdirSync} from 'node:fs';
import {chromium} from 'playwright-core';

mkdirSync('../../tmp',{recursive:true});
const browser=await chromium.launch({
  executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',
  headless:true,
  args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
});
const url='http://127.0.0.1:4176/docs/command-builder.html?utm_source=chatgpt.com&mode=play&task=dynamic_oa&method=planner&controller=sampling_mpc&planner=ego&scene=D01';
const errors=[],wasmRequests=[];

try{
  const page=await browser.newPage({viewport:{width:1480,height:1000}});
  page.on('pageerror',error=>errors.push(error.message));
  page.on('console',message=>{
    if(message.type()==='error')errors.push(message.text());
  });
  page.on('request',request=>{
    if(request.url().endsWith('.wasm'))wasmRequests.push(request.url());
  });
  const t0=Date.now();
  await page.goto(url,{waitUntil:'domcontentloaded'});
  await page.locator('#preview-loading').waitFor({state:'visible',timeout:10000});
  const startingLabel=await page.locator('#preview-loading-title').textContent();
  await page.waitForFunction(()=>
    document.querySelector('#preview-loading')?.hidden===true&&
    document.querySelector('#preview-status')?.textContent?.includes('Ready')&&
    document.querySelector('#preview-scene-name')?.textContent==='D01',
    null,{timeout:40000});
  const firstLoadMs=Date.now()-t0;
  const iframe=page.frameLocator('#preview-frame');
  await iframe.locator('#status').filter({hasText:'350 frames'}).waitFor({state:'attached',timeout:10000});
  const before=Number(await iframe.locator('#timeline').inputValue());
  await page.waitForTimeout(1350);
  const after=Number(await iframe.locator('#timeline').inputValue());
  if(after<=before)throw Error(`D01 did not advance automatically: ${before} -> ${after}`);
  await page.screenshot({path:'../../tmp/task-preview-d01-final-desktop.png',fullPage:true});

  // Scene changes must immediately mask stale old frames and show progress.
  await page.evaluate(()=>CommandBuilder.applySelection({mode:'play',method:'planner',task:'static_oa',planner:'ego',controller:'sampling_mpc',scene:'S06'}));
  await page.locator('#preview-loading').waitFor({state:'visible',timeout:3000});
  await page.waitForFunction(()=>
    document.querySelector('#preview-loading')?.hidden===true&&
    document.querySelector('#preview-scene-name')?.textContent==='S06'&&
    document.querySelector('#preview-status')?.textContent?.includes('Ready'),
    null,{timeout:40000});
  await iframe.locator('#status').filter({hasText:'321 frames'}).waitFor({state:'attached',timeout:10000});
  if(!String(await page.locator('#preview-note').textContent()).includes('NO_SOLUTION'))
    throw Error('S06 lost its physical planner-failure explanation');

  const t1=Date.now();
  await page.evaluate(()=>CommandBuilder.applySelection({mode:'play',method:'planner',task:'dynamic_oa',planner:'ego',controller:'sampling_mpc',scene:'D01'}));
  await page.locator('#preview-loading').waitFor({state:'visible',timeout:3000});
  const pendingLabel=await page.locator('#preview-loading-title').textContent();
  await page.waitForFunction(()=>
    document.querySelector('#preview-loading')?.hidden===true&&
    document.querySelector('#preview-scene-name')?.textContent==='D01'&&
    document.querySelector('#preview-status')?.textContent?.includes('Ready'),
    null,{timeout:40000});
  const returnLoadMs=Date.now()-t1;
  const timeOne=Number(await iframe.locator('#timeline').inputValue());
  await page.waitForTimeout(900);
  const timeTwo=Number(await iframe.locator('#timeline').inputValue());
  if(timeTwo<=timeOne)throw Error('D01 stalled after switching back from S06');

  // Quickly choosing several scenes should load the latest selection rather
  // than showing stale data; errors must be user-visible and offer Retry.
  await page.evaluate(async()=>{
    for(const scene of ['D02','D03','D01']){
      CommandBuilder.applySelection({mode:'play',method:'planner',task:'dynamic_oa',planner:'ego',controller:'sampling_mpc',scene});
      await new Promise(resolve=>setTimeout(resolve,70));
    }
  });
  await page.waitForFunction(()=>
    document.querySelector('#preview-loading')?.hidden===true&&
    document.querySelector('#preview-scene-name')?.textContent==='D01'&&
    document.querySelector('#preview-status')?.textContent?.includes('Ready'),
    null,{timeout:50000});
  const record=await page.evaluate(()=>document.querySelector('#preview-frame')?.contentWindow?.document?.querySelector('#record')?.value);
  if(!record?.includes('D01-0000'))throw Error(`Stale scene after rapid switching: ${record}`);
  if(wasmRequests.length!==1)throw Error(`WASM was loaded ${wasmRequests.length} times, expected one`);

  // Responsive and accessible: the environment selector should not overflow
  // on phone widths, and the error/preview heading stay available.
  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(450);
  const sizes=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,window:innerWidth,header:document.querySelector('.site-header')?.getBoundingClientRect().width}));
  if(sizes.scroll>sizes.window+3)throw Error(`Mobile horizontal overflow: ${JSON.stringify(sizes)}`);
  await page.screenshot({path:'../../tmp/task-preview-d01-final-mobile.png',fullPage:true});
  if(errors.length)throw Error(errors.join('\n'));
  console.log(JSON.stringify({startingLabel,pendingLabel,firstLoadMs,returnLoadMs,frames:350,autoplay:[before,after],afterSwitch:[timeOne,timeTwo],wasmRequests:wasmRequests.length,mobile:sizes,errors}));
}finally{
  await browser.close();
}
