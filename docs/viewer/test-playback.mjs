// Headless integration check against the SSH-forwardable static HTTP server.
import {mkdirSync} from 'node:fs';
import {chromium} from 'playwright-core';

mkdirSync('../../tmp', {recursive:true});
const browser = await chromium.launch({
  executablePath:process.env.CHROME_PATH||'/usr/bin/google-chrome',
  headless:true,
  args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'],
});
try{
  const page = await browser.newPage({viewport:{width:1200,height:760}});
  const errors=[];
  page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
  page.on('pageerror',exception=>errors.push(exception.message));
  const base=process.env.VIEWER_URL||'http://127.0.0.1:4175/docs/viewer/dist/index.html';
  const data=process.env.VIEWER_RECORD||'/results/mpcc-full-benchmark-8/replays/racing-0000/browser_replay.json';
  await page.goto(`${base}?record=${encodeURIComponent(data)}`, {waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>/\d+ frames ·/.test(document.querySelector('#status')?.textContent||'') || /Cannot load recording/.test(document.querySelector('#status')?.textContent||''),{timeout:90000});
  const before=await page.locator('#status').textContent();
  if(before.includes('Cannot load'))throw new Error(before);
  await page.locator('#timeline').fill('100');
  const time=await page.locator('#time').textContent();
  if(!/\d+\.\d+.*s/.test(time))throw new Error(`Invalid replay seek: ${time}`);
  const beforePlay=Number(await page.locator('#timeline').inputValue());
  await page.locator('#play').click();
  await page.waitForTimeout(1100);
  const afterPlay=Number(await page.locator('#timeline').inputValue());
  if(afterPlay<=beforePlay+5)
    throw new Error(`Play did not advance the recording: ${beforePlay} -> ${afterPlay}`);
  if(!(await page.locator('#status').textContent()).includes('Crazyflie mesh geoms'))
    throw new Error('The native Crazyflie 2.x visual model did not load');
  await page.screenshot({path:process.env.VIEWER_SCREENSHOT||'../../tmp/mujoco-wasm-browser-smoke.png'});
  await page.locator('#play').click();
  await page.locator('#timeline').fill('0');
  await page.locator('#trail').click();
  if(await page.locator('#trail').textContent()!=='Flight path: Off')
    throw new Error('Flight path visibility control did not work');
  await page.screenshot({path:'../../tmp/mujoco-wasm-crazyflie-focus.png'});
  await page.locator('#view').click();
  if(await page.locator('#view').textContent()!=='Camera: Course')
    throw new Error('Camera view did not switch to course');
  console.log(JSON.stringify({before,afterSeek:time,beforePlay,afterPlay,errors,viewer:page.url()}));
  if(errors.length)throw new Error(errors.join('\n'));
}finally{await browser.close();}
