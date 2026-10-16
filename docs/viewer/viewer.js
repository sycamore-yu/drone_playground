// Play recorded Crazyflow/MuJoCo poses through the official MuJoCo WASM model.

import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';

const $ = id => document.getElementById(id);
const viewport = $('scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x121f2b);
const camera = new THREE.PerspectiveCamera(50, 1, .02, 500);
camera.up.set(0, 0, 1);
camera.position.set(5, -5, 4);
const renderer = new THREE.WebGLRenderer({antialias:true, powerPreference:'high-performance'});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = .25;
controls.maxDistance = 100;
const ambient = new THREE.AmbientLight(0xffffff, 1.8);
scene.add(ambient);
const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
keyLight.position.set(4, 2, 10);
scene.add(keyLight);

let engine = null, model = null, data = null, frames = [];
let meshes = [], pathLine = null, frameIndex = 0, playing = false, follow = true;
let closeCamera = true;
let wallThen = performance.now();
let playheadTime = 0;

function status(message){ $('status').textContent = message; }
function disposeRecord(){
  playing=false; $('play').textContent='Play';
  playheadTime=0;
  for(const mesh of meshes){scene.remove(mesh);mesh.geometry?.dispose();mesh.material?.dispose();}
  meshes=[];
  if(pathLine){scene.remove(pathLine);pathLine.geometry.dispose();pathLine.material.dispose();pathLine=null;}
  data?.delete(); model?.delete(); model=null; data=null; frames=[];
}

function shape(type, size, id){
  const kind = engine.mjtGeom;
  if(type===kind.mjGEOM_PLANE.value)return new THREE.PlaneGeometry(Math.max(1,size[0]*2),Math.max(1,size[1]*2));
  if(type===kind.mjGEOM_SPHERE.value)return new THREE.SphereGeometry(size[0],14,10);
  if(type===kind.mjGEOM_CAPSULE.value){const result=new THREE.CapsuleGeometry(size[0],2*size[1],8,10);result.rotateX(Math.PI/2);return result;}
  if(type===kind.mjGEOM_ELLIPSOID.value){const result=new THREE.SphereGeometry(1,14,10);result.scale(...size);return result;}
  if(type===kind.mjGEOM_CYLINDER.value){const result=new THREE.CylinderGeometry(size[0],size[0],2*size[1],16);result.rotateX(Math.PI/2);return result;}
  if(type===kind.mjGEOM_BOX.value)return new THREE.BoxGeometry(...size.map(v=>2*v));
  if(type===kind.mjGEOM_MESH.value){
    const meshId=model.geom_dataid[id];
    const vertexStart=model.mesh_vertadr[meshId],vertexCount=model.mesh_vertnum[meshId];
    const faceStart=model.mesh_faceadr[meshId],faceCount=model.mesh_facenum[meshId];
    const vertices=Float32Array.from(model.mesh_vert.subarray(vertexStart*3,(vertexStart+vertexCount)*3));
    const triangles=Array.from(model.mesh_face.subarray(faceStart*3,(faceStart+faceCount)*3));
    const result=new THREE.BufferGeometry();
    result.setAttribute('position',new THREE.BufferAttribute(vertices,3));
    result.setIndex(triangles);
    result.computeVertexNormals();
    return result;
  }
  return new THREE.SphereGeometry(Math.max(size[0],.02));
}

function materialFor(i){
  const rgba=Array.from(model.geom_rgba.subarray(i*4,i*4+4));
  const matId=model.geom_matid[i];
  if(matId>=0 && rgba[3]!==0){
    const diffuse=Array.from(model.mat_rgba.subarray(matId*4,matId*4+4));
    if(diffuse[3]>0)rgba.splice(0,4,...diffuse);
  }
  return new THREE.MeshStandardMaterial({
    color:new THREE.Color(rgba[0],rgba[1],rgba[2]),
    opacity:rgba[3],transparent:rgba[3]<1,side:THREE.DoubleSide,
    roughness:.75,metalness:.04,
  });
}

function buildScene(){
  let crazyflieMeshCount = 0;
  const names = model.names;
  const nameAddresses = model.name_geomadr;
  const decoder = new TextDecoder();
  for(let i=0;i<model.ngeom;i++){
    if(model.geom_rgba[i*4+3]<.015)continue;
    if(model.geom_group[i]>=4)continue;
    if(names && nameAddresses){
      const start=nameAddresses[i];
      let end=start;
      while(end<names.length && names[end]!==0)end++;
      // RScope writes a capsule for every recorded path segment. Our single
      // toggleable Line shows the same trajectory without a forest of tubes.
      if(decoder.decode(names.subarray(start,end)).startsWith('replay_actual_'))continue;
    }
    const size=Array.from(model.geom_size.subarray(i*3,i*3+3));
    const mesh=new THREE.Mesh(shape(model.geom_type[i],size,i),materialFor(i));
    mesh.userData.geomId=i;
    if(model.geom_group[i]===2 && model.geom_type[i]===engine.mjtGeom.mjGEOM_MESH.value)
      crazyflieMeshCount++;
    scene.add(mesh);meshes.push(mesh);
  }
  const positions=frames.map(f=>f.qpos.slice(0,3));
  const geometry=new THREE.BufferGeometry().setFromPoints(positions.map(p=>new THREE.Vector3(...p)));
  pathLine=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0x49cdf5,opacity:.85,transparent:true}));
  scene.add(pathLine);
  return crazyflieMeshCount;
}

function nearestFrame(target){
  let lo=0,hi=frames.length-1;
  while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].time<target)lo=mid+1;else hi=mid;}
  return Math.abs(frames[lo].time-target)<Math.abs(frames[Math.max(0,lo-1)].time-target)?lo:Math.max(0,lo-1);
}

function displayFrame(index){
  if(!model)return;
  frameIndex=Math.max(0,Math.min(frames.length-1,index));
  const frame=frames[frameIndex];
  data.time=frame.time;
  data.qpos.set(frame.qpos);
  data.mocap_pos.set(frame.mocap_pos.flat());
  data.mocap_quat.set(frame.mocap_quat.flat());
  engine.mj_forward(model,data);
  const matrix=new THREE.Matrix4();
  for(const mesh of meshes){
    const i=mesh.userData.geomId;
    mesh.position.set(data.geom_xpos[3*i],data.geom_xpos[3*i+1],data.geom_xpos[3*i+2]);
    const R=data.geom_xmat.subarray(9*i,9*i+9);
    matrix.set(R[0],R[1],R[2],0,R[3],R[4],R[5],0,R[6],R[7],R[8],0,0,0,0,1);
    mesh.quaternion.setFromRotationMatrix(matrix);
  }
  if(follow){
    const previous=controls.target.clone();
    controls.target.set(...frame.qpos.slice(0,3));
    camera.position.add(controls.target.clone().sub(previous));
  }
  $('timeline').value=String(frameIndex);
  $('time').textContent=`${frame.time.toFixed(3)} / ${frames.at(-1).time.toFixed(3)} s`;
}

function frameCamera(){
  if(!frames.length)return;
  const center=frames[frameIndex].qpos.slice(0,3);
  controls.target.set(...center);
  // The STL Crazyflie is about 8 cm across. A multi-metre orbit makes even
  // the genuine mesh look like a dot; keep 1:1 physical geometry and move the
  // camera instead of exaggerating the mesh's scale.
  const offset=closeCamera?[.40,-.55,.32]:[3.2,-3.2,2.1];
  camera.position.set(center[0]+offset[0],center[1]+offset[1],center[2]+offset[2]);
  controls.update();
}

async function load(){
  const address=$('record').value.trim();
  if(!address)return status('Specify a browser_replay.json URL');
  $('load').disabled=true;
  disposeRecord();
  status('Loading recorded poses and MJCF...');
  try{
    const url=new URL(address,location.href);
    if(url.origin!==location.origin)throw new Error('Record must be served by the same SSH-forwarded HTTP server');
    const response=await fetch(url);
    if(!response.ok)throw new Error(`HTTP ${response.status} loading ${url.pathname}`);
    const parsed=await response.json();
    if(parsed.format!=='drone-playground-mujoco-replay-v1'||!Array.isArray(parsed.frames)||parsed.frames.length<2)
      throw new Error('Incompatible recorded MuJoCo episode');
    if(!engine){
      status('Initializing MuJoCo WebAssembly...');
      engine=await loadMujoco({locateFile:path=>path.endsWith('.wasm')?wasmUrl:path});
      engine.FS.mkdir('/working');
      engine.FS.mount(engine.MEMFS,{root:'.'},'/working');
    }
    for(const file of [parsed.model,...parsed.assets]){
      if(!/^[A-Za-z0-9_.-]+$/.test(file))throw new Error(`Unsupported model filename: ${file}`);
      const assetResponse=await fetch(new URL(file,url));
      if(!assetResponse.ok)throw new Error(`Missing MJCF asset ${file}`);
      const bytes=new Uint8Array(await assetResponse.arrayBuffer());
      engine.FS.writeFile(`/working/${file}`,bytes);
    }
    model=engine.MjModel.mj_loadXML(`/working/${parsed.model}`);
    data=new engine.MjData(model);
    frames=parsed.frames;
    for(const frame of frames){
      if(frame.qpos.length!==model.nq||frame.mocap_pos.length!==model.nmocap)
        throw new Error('Recorded MuJoCo dimensions differ from loaded MJCF');
    }
    const airframeMeshes=buildScene();
    displayFrame(0);
    frameCamera();
    playheadTime=frames[0].time;
    $('timeline').max=String(frames.length-1);
    $('timeline').disabled=false;$('play').disabled=false;
    status(`${frames.length} frames · ${airframeMeshes} Crazyflie mesh geoms · ${model.nmocap} mocap bodies`);
    const updated=new URL(location.href);updated.searchParams.set('record',address);history.replaceState(null,'',updated);
  }catch(error){
    status(`Cannot load recording: ${error.message}`);console.error(error);disposeRecord();
  }finally{$('load').disabled=false;}
}

$('load').addEventListener('click',load);
$('record').addEventListener('keydown',event=>{if(event.key==='Enter')load();});
$('play').addEventListener('click',()=>{
  if(!frames.length)return;
  if(!playing && frameIndex===frames.length-1){
    playheadTime=frames[0].time;
    displayFrame(0);
  }
  playing=!playing;$('play').textContent=playing?'Pause':'Play';wallThen=performance.now();
});
$('timeline').addEventListener('input',event=>{
  playing=false;$('play').textContent='Play';
  displayFrame(Number(event.target.value));playheadTime=frames[frameIndex].time;
});
$('follow').addEventListener('click',()=>{follow=!follow;$('follow').textContent=`Follow drone: ${follow?'On':'Off'}`;});
$('view').addEventListener('click',()=>{
  closeCamera=!closeCamera;
  $('view').textContent=`Camera: ${closeCamera?'Aircraft':'Course'}`;
  frameCamera();
});
$('trail').addEventListener('click',()=>{
  if(!pathLine)return;
  pathLine.visible=!pathLine.visible;
  $('trail').textContent=`Flight path: ${pathLine.visible?'On':'Off'}`;
});
window.addEventListener('resize',resize);
function resize(){const {width,height}=viewport.getBoundingClientRect();camera.aspect=width/Math.max(height,1);camera.updateProjectionMatrix();renderer.setSize(width,height);}
new ResizeObserver(resize).observe(viewport);
function animate(now){
  if(playing&&frames.length){
    const dt=Math.min(.3,Math.max(0,(now-wallThen)/1000))*Number($('speed').value);
    // Advance in continuous recorded time. Nearest-frame rounding must not
    // discard sub-frame time or playback stalls at every sampled frame.
    playheadTime+=dt;
    if(playheadTime>=frames.at(-1).time){
      playheadTime=frames.at(-1).time;
      displayFrame(frames.length-1);playing=false;$('play').textContent='Play';
    }else displayFrame(nearestFrame(playheadTime));
  }
  wallThen=now;
  controls.update();renderer.render(scene,camera);
  requestAnimationFrame(animate);
}
resize();requestAnimationFrame(animate);
const initial=new URL(location.href).searchParams.get('record');
if(initial){$('record').value=initial;load();}
