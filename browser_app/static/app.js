'use strict';
const $ = id => document.getElementById(id);
const state = {config:null, frame:0, selection:null, selectionFrame:null, job:null, rows:new Map(), playing:false, drag:null, epoch:0, seen:0, polling:false, frameReady:false};
const canvas = $('overlay'), ctx = canvas.getContext('2d'), image = $('frameImage'), video = $('video');
const active = job => job && ['queued','loading','running','stopping'].includes(job.status);
let noticeTimer, pollTimer, videoCallback;

function notify(message){ $('notice').textContent=message; $('notice').hidden=false; clearTimeout(noticeTimer); noticeTimer=setTimeout(()=>$('notice').hidden=true,8000); }
async function api(path, options){ const response=await fetch(path,options); let data; try{data=await response.json();}catch{throw new Error('The server returned an unreadable response.');} if(!response.ok)throw new Error(data.error||`Request failed (${response.status}).`); return data; }
function request(method, body){return {method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};}
function updateLabels(){
  const c=state.config;if(!c)return;
  $('frameNumber').value=state.frame+1;$('timeline').value=state.frame;$('timecode').textContent=`${(state.frame/c.fps).toFixed(2)} s`;
  const row=state.rows.get(state.frame);
  $('score').textContent=row?.score==null?'—':row.score.toFixed(3);
  $('frameState').textContent=row ? (row.seed?'Initialization frame':'Tracked frame') : 'Original frame';
  $('previous').disabled=state.frame===0;$('next').disabled=state.frame===c.count-1;
}
function drawBox(box, color, label){
  const rect=canvas.getBoundingClientRect();const scale=canvas.width/Math.max(rect.width,1);const width=Math.max(2*scale,1);
  ctx.lineWidth=width;ctx.strokeStyle='#000';ctx.strokeRect(box[0],box[1],box[2],box[3]);
  ctx.lineWidth=Math.max(scale,1);ctx.strokeStyle=color;ctx.strokeRect(box[0],box[1],box[2],box[3]);
  if(label){ctx.font=`${Math.max(12*scale,12)}px monospace`;const tw=ctx.measureText(label).width;const th=17*scale;const y=box[1]>=th?box[1]-th:box[1]+box[3];ctx.fillStyle='#000';ctx.fillRect(box[0],y,tw+8*scale,th);ctx.fillStyle=color;ctx.fillText(label,box[0]+4*scale,y+12*scale);}
}
function render(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const row=state.rows.get(state.frame);
  if(row&&!row.seed)drawBox(row.box,'#60f4ad',`LoRAT ${row.score.toFixed(3)}`);
  if(row?.seed)drawBox(row.box,'#ffca58','INITIAL TARGET');
  if(state.selection&&state.selectionFrame===state.frame)drawBox(state.selection,'#ffca58','YOUR TARGET');
  if(state.drag){const d=state.drag;drawBox([Math.min(d.start[0],d.end[0]),Math.min(d.start[1],d.end[1]),Math.abs(d.end[0]-d.start[0]),Math.abs(d.end[1]-d.start[1])],'#ffca58',null);}
  updateLabels();
}
async function showFrame(index){
  if(!state.config)return;
  pause();state.frame=Math.max(0,Math.min(state.config.count-1,Math.round(index)));state.frameReady=false;
  const epoch=++state.epoch;image.hidden=false;video.hidden=true;canvas.classList.remove('playing');
  $('frameState').textContent='Loading frame…';updateLabels();render();
  const nextImage=new Image();nextImage.src=`/api/frames/${state.frame}.jpg`;
  try{await nextImage.decode();if(epoch!==state.epoch)return;image.src=nextImage.src;await image.decode();if(epoch!==state.epoch)return;state.frameReady=true;render();}
  catch{if(epoch===state.epoch)notify('Could not load this frame. Check the server connection.');}
}
function pause(){
  state.playing=false;video.pause();$('play').textContent='Play';
  if(videoCallback!=null&&video.cancelVideoFrameCallback)video.cancelVideoFrameCallback(videoCallback);videoCallback=null;
}
function frameAtTime(t){return Math.max(0,Math.min(state.config.count-1,Math.floor(t*state.config.fps+0.0001)));}
function playbackTick(_now, metadata){
  if(!state.playing)return;
  state.frame=frameAtTime(metadata?.mediaTime??video.currentTime);render();
  videoCallback=video.requestVideoFrameCallback?video.requestVideoFrameCallback(playbackTick):requestAnimationFrame(playbackTick);
}
async function togglePlay(){
  if(state.playing){const frame=state.frame;pause();await showFrame(frame);return;}
  if(!state.config?.video){notify('The playback video is still being prepared. Frame stepping is available.');return;}
  if(state.frame>=state.config.count-1)state.frame=0;
  try{video.currentTime=state.frame/state.config.fps;image.hidden=true;video.hidden=false;state.playing=true;state.frameReady=false;canvas.classList.add('playing');$('play').textContent='Pause';await video.play();playbackTick();}
  catch(error){pause();image.hidden=false;video.hidden=true;notify(error.message);}
}
function point(event){const r=canvas.getBoundingClientRect();return [Math.min(canvas.width,Math.max(0,(event.clientX-r.left)*canvas.width/r.width)),Math.min(canvas.height,Math.max(0,(event.clientY-r.top)*canvas.height/r.height))];}
function selectionUI(){
  const box=state.selection;
  ['coordX','coordY','coordW','coordH'].forEach((id,i)=>$(id).textContent=box?box[i].toFixed(1):'—');
  $('selectionFrame').textContent=box?`Selected on frame ${state.selectionFrame+1}`:'No target selected';
  $('targetPlaceholder').hidden=!!box;$('clear').disabled=!box||active(state.job);
  $('track').disabled=!box||active(state.job);
  $('selectionHelp').textContent=active(state.job)?'Wait for this run or stop it to select another target.':box?`Track forward from frame ${state.selectionFrame+1}.`:'Select a target to begin.';
  const crop=$('targetCrop'), c=crop.getContext('2d');
  if(!box)c.clearRect(0,0,crop.width,crop.height);
  if(box&&state.frameReady&&state.frame===state.selectionFrame){c.clearRect(0,0,crop.width,crop.height);const ratio=Math.min(crop.width/box[2],crop.height/box[3]);const w=box[2]*ratio,h=box[3]*ratio;c.drawImage(image,box[0],box[1],box[2],box[3],(crop.width-w)/2,(crop.height-h)/2,w,h);}
}
canvas.addEventListener('pointerdown',event=>{
  if(event.button!==0||state.playing||!state.frameReady||active(state.job))return;
  const p=point(event);state.drag={start:p,end:p};canvas.setPointerCapture(event.pointerId);render();
});
canvas.addEventListener('pointermove',event=>{if(!state.drag)return;state.drag.end=point(event);render();});
canvas.addEventListener('pointerup',event=>{
  if(!state.drag)return;const d=state.drag;d.end=point(event);state.drag=null;
  const box=[Math.min(d.start[0],d.end[0]),Math.min(d.start[1],d.end[1]),Math.abs(d.end[0]-d.start[0]),Math.abs(d.end[1]-d.start[1])];
  if(box[2]>=2&&box[3]>=2){state.selection=box;state.selectionFrame=state.frame;$('endFrame').min=state.frame+1;if(Number($('endFrame').value)<state.frame+1)$('endFrame').value=state.config.count;selectionUI();}
  else notify('Draw a box at least two pixels wide and high.');render();
});
canvas.addEventListener('pointercancel',()=>{state.drag=null;render();});
$('clear').onclick=()=>{state.selection=null;state.selectionFrame=null;selectionUI();render();};
$('previous').onclick=()=>showFrame(state.frame-1);$('next').onclick=()=>showFrame(state.frame+1);$('play').onclick=togglePlay;
video.addEventListener('ended',()=>{const end=state.config.count-1;pause();showFrame(end);});
$('frameNumber').onfocus=pause;
function goToFrame(){const n=Number($('frameNumber').value);if(Number.isFinite(n))showFrame(n-1);}
$('frameNumber').onchange=goToFrame;
$('frameNumber').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();goToFrame();}});
$('timeline').oninput=()=>showFrame(Number($('timeline').value));
$('zoom').onchange=()=>{$('stage').style.width=`${Number($('zoom').value)*100}%`;render();};
window.addEventListener('resize',render);
document.addEventListener('keydown',event=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON','A'].includes(event.target.tagName)||event.ctrlKey||event.metaKey||event.altKey)return;
  if(event.key==='ArrowLeft'){event.preventDefault();showFrame(state.frame-1);}
  if(event.key==='ArrowRight'){event.preventDefault();showFrame(state.frame+1);}
  if(event.code==='Space'){event.preventDefault();togglePlay();}
});
function jobUI(){
  const job=state.job;if(!job)return;const busy=active(job);
  $('runStatus').textContent=job.status.toUpperCase();$('runStatus').className=`status ${busy?'active':job.status==='failed'?'failed':''}`;
  const text={queued:'Waiting for the tracker.',loading:'Loading LoRAT g-378 on the GPU…',running:'Tracking forward from your selected frame.',stopping:'Stopping after the current frame.',complete:'Tracking complete. Play or scrub to inspect the result.',canceled:'Run stopped. Processed frames are available.',failed:job.error||'Tracking failed.'};
  $('runMessage').textContent=text[job.status]||job.status;
  $('processed').textContent=`${job.completed_frames} / ${job.total_frames}`;$('progress').max=job.total_frames;$('progress').value=job.completed_frames;
  $('stop').disabled=!busy||job.status==='stopping';$('history').disabled=busy;
  for(const [id,kind] of [['csvDownload','csv'],['jsonDownload','json']]){$(id).href=`/api/jobs/${job.id}/${kind}`;$(id).setAttribute('aria-disabled',String(job.completed_frames===0));}
  $('modelState').textContent=job.status==='loading'?'Loading model on GPU 1':job.status==='failed'?'Run failed. See the message above.':'Model stays loaded between runs';
  const coverage=$('coverage');coverage.replaceChildren();if(job.completed_frames){const segment=document.createElement('span');segment.style.left=`${job.start_frame/state.config.count*100}%`;segment.style.width=`${job.completed_frames/state.config.count*100}%`;coverage.append(segment);}
  selectionUI();render();
}
async function history(){
  const jobs=await api('/api/jobs');const selected=state.job?.id||'';const select=$('history');select.replaceChildren(new Option('Select a saved run',''));
  for(const job of jobs)select.add(new Option(`Frame ${job.start_frame+1} → ${job.end_frame+1} · ${job.status}`,job.id));select.value=selected;return jobs;
}
async function poll(){
  if(!state.job||state.polling)return;state.polling=true;const id=state.job.id;
  try{const job=await api(`/api/jobs/${id}?since=${state.seen}`);if(state.job?.id!==id)return;
    for(const row of job.rows)state.rows.set(row.frame,row);state.seen=job.completed_frames;state.job=job;jobUI();
    $('connection').textContent='Connected to comech-1176';$('connectionDot').classList.add('ready');
    if($('follow').checked&&active(job)&&job.rows.length&&!state.playing)await showFrame(job.rows.at(-1).frame);
    if(active(job))pollTimer=setTimeout(poll,800);else await history();
  }catch(error){notify(error.message);$('connection').textContent='Connection interrupted';$('connectionDot').classList.remove('ready');pollTimer=setTimeout(poll,2000);}
  finally{state.polling=false;}
}
$('track').onclick=async()=>{
  if(!state.selection||active(state.job))return;const end=Number($('endFrame').value)-1;
  if(!Number.isInteger(end)||end<state.selectionFrame||end>=state.config.count){notify('Choose an end frame at or after the selected frame.');return;}
  $('track').disabled=true;
  try{const job=await api('/api/jobs',request('POST',{start_frame:state.selectionFrame,end_frame:end,box:state.selection}));clearTimeout(pollTimer);state.job=job;state.rows.clear();state.seen=0;pause();await showFrame(state.selectionFrame);jobUI();await history();poll();}
  catch(error){notify(error.message);selectionUI();}
};
$('stop').onclick=async()=>{if(!state.job)return;try{await api(`/api/jobs/${state.job.id}/cancel`,request('POST',{}));await poll();}catch(error){notify(error.message);}};
$('history').onchange=async()=>{
  const id=$('history').value;if(!id||active(state.job))return;
  try{clearTimeout(pollTimer);const job=await api(`/api/jobs/${id}`);state.job=job;state.rows=new Map(job.rows.map(row=>[row.frame,row]));state.seen=job.completed_frames;state.selection=job.initial_box;state.selectionFrame=job.start_frame;$('endFrame').min=job.start_frame+1;$('endFrame').value=job.end_frame+1;await showFrame(job.start_frame);jobUI();if(active(job))poll();}
  catch(error){notify(error.message);}
};
async function boot(){
  try{const c=await api('/api/config');state.config=c;canvas.width=c.width;canvas.height=c.height;$('stage').style.aspectRatio=`${c.width}/${c.height}`;
    $('sequence').textContent=c.sequence;$('resolution').textContent=`${c.width} × ${c.height} · ${c.fps} fps`;$('frameTotal').textContent=`/ ${c.count}`;$('rangeLabel').textContent=`${c.count} original frames`;
    $('timeline').max=c.count-1;for(const id of ['frameNumber','endFrame'])$(id).max=c.count;$('endFrame').value=c.count;
    if(c.video)video.src=c.video;video.style.height=`${Math.ceil(c.height/2)*2/c.height*100}%`;$('stage').style.overflow='hidden';$('connection').textContent='Connected to comech-1176';$('connectionDot').classList.add('ready');$('modelState').textContent=c.model_state==='Ready'?'Model is loaded and ready':'Model loads on the first run';
    await showFrame(0);selectionUI();const jobs=await history();const running=jobs.find(active);
    if(running){state.job=running;state.selection=running.initial_box;state.selectionFrame=running.start_frame;$('endFrame').min=running.start_frame+1;$('endFrame').value=running.end_frame+1;await showFrame(running.start_frame);jobUI();poll();}
  }catch(error){$('connection').textContent='Server unavailable';notify(error.message);}
}
boot();
