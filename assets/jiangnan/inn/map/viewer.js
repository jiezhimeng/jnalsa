/* Full-resolution panorama coordinates; this viewer never sends device input. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('map'), ctx = canvas.getContext('2d');
  const picture = new Image(), pixels = document.createElement('canvas');
  const pixelCtx = pixels.getContext('2d', {willReadFrequently: true});
  const instance = new URLSearchParams(location.search).get('instance') || '';
  const storageKey = 'yingtian-inn-point-v1:' + instance;
  let width = 0, height = 0, zoom = 1, panX = 0, panY = 0, ready = false;
  let selection = null, observation = null, pointer = null, fitted = true;
  const minZoom = () => Math.min(width / picture.width, height / picture.height) * 0.75;
  const toMap = (x, y) => [(x - panX) / zoom, (y - panY) / zoom];
  const toScreen = p => [p[0] * zoom + panX, p[1] * zoom + panY];
  const pointText = p => `${Math.round(p[0])}, ${Math.round(p[1])}`;
  function covered(p) {
    const x = Math.floor(p[0]), y = Math.floor(p[1]);
    if (x < 0 || y < 0 || x >= picture.width || y >= picture.height) return false;
    return pixelCtx.getImageData(x, y, 1, 1).data.slice(0, 3).some(v => v > 12);
  }
  function draw() {
    if (!ready) return;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(picture, panX, panY, picture.width * zoom, picture.height * zoom);
    const loc = observation && observation.matched && observation.location;
    if (loc) {
      ctx.beginPath();
      loc.viewport.forEach((p, i) => { const [x, y] = toScreen(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.closePath();ctx.fillStyle = '#69d8df20';ctx.fill();ctx.strokeStyle = '#69d8df';ctx.lineWidth = 2;ctx.stroke();
      marker(loc.center, '#ffb958', false);
    }
    if (selection) marker(selection, '#ff8b92', true);
    $('zoom').textContent = Math.round(zoom * 100) + '%';
  }
  function marker(p, color, label) {
    const [x, y] = toScreen(p);
    ctx.strokeStyle = color;ctx.fillStyle = color;ctx.lineWidth = 2;
    ctx.beginPath();ctx.arc(x, y, 6, 0, Math.PI * 2);ctx.stroke();
    ctx.beginPath();ctx.moveTo(x-13,y);ctx.lineTo(x+13,y);ctx.moveTo(x,y-13);ctx.lineTo(x,y+13);ctx.stroke();
    if (label) {
      const text = '(' + pointText(p) + ')';ctx.font = '13px monospace';
      ctx.fillStyle = '#162029e8';ctx.fillRect(x+12,y-20,ctx.measureText(text).width+12,23);
      ctx.fillStyle = color;ctx.fillText(text,x+18,y-4);
    }
  }
  function fit() {
    if (!ready) return;
    zoom = Math.min(width / picture.width, height / picture.height) * 0.95;
    panX = (width-picture.width*zoom)/2;panY = (height-picture.height*zoom)/2;fitted = true;draw();
  }
  function zoomAt(value, x, y) {
    if (!ready) return;
    const p = toMap(x,y);zoom = Math.max(minZoom(),Math.min(4,value));
    panX = x-p[0]*zoom;panY = y-p[1]*zoom;fitted = false;draw();
  }
  function select(p) {
    if (!covered(p)) { $('hover').textContent = '未采集区域';return; }
    selection = p.map(Math.round);$('selected').value = pointText(selection);$('copy').disabled = false;
    try { localStorage.setItem(storageKey,JSON.stringify(selection)); } catch (_) { /* Private browsing is supported. */ }
    draw();
  }
  function resize() {
    const rect = $('map-wrap').getBoundingClientRect();
    const previous = ready ? toMap(width/2,height/2) : null;
    width = rect.width;height = rect.height;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width*ratio);canvas.height = Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);
    if (fitted) fit();else if (previous) { panX=width/2-previous[0]*zoom;panY=height/2-previous[1]*zoom;draw(); }
  }
  function eventPoint(e) { const r=canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top]; }
  canvas.addEventListener('wheel',e=>{e.preventDefault();const [x,y]=eventPoint(e);zoomAt(zoom*Math.exp(-e.deltaY*0.0015),x,y);},{passive:false});
  canvas.addEventListener('pointerdown',e=>{
    if (!ready || e.button !== 0) return;
    const [x,y]=eventPoint(e);pointer={id:e.pointerId,x,y,panX,panY,moved:false};canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    if (!ready) return;
    const [x,y]=eventPoint(e), p=toMap(x,y);
    $('hover').textContent=covered(p)?pointText(p):'未采集区域';
    if (pointer && pointer.id===e.pointerId) {
      const dx=x-pointer.x,dy=y-pointer.y;
      if (pointer.moved || Math.hypot(dx,dy)>5) {
        pointer.moved=true;fitted=false;panX=pointer.panX+dx;panY=pointer.panY+dy;draw();
      }
    }
  });
  canvas.addEventListener('pointerup',e=>{
    if (!pointer || pointer.id!==e.pointerId) return;
    if (!pointer.moved) select(toMap(...eventPoint(e)));
    pointer=null;
  });
  canvas.addEventListener('pointercancel',()=>{pointer=null;});
  canvas.addEventListener('lostpointercapture',()=>{pointer=null;});
  canvas.addEventListener('keydown',e=>{
    if (!ready) return;
    const keys={ArrowLeft:[50,0],ArrowRight:[-50,0],ArrowUp:[0,50],ArrowDown:[0,-50]};
    if (keys[e.key]) { e.preventDefault();const d=keys[e.key];panX+=d[0];panY+=d[1];fitted=false;draw(); }
    else if(e.key==='Enter'){e.preventDefault();select(toMap(width/2,height/2));}
  });
  $('fit').onclick=fit;
  $('zoom-in').onclick=()=>zoomAt(zoom*1.3,width/2,height/2);
  $('zoom-out').onclick=()=>zoomAt(zoom/1.3,width/2,height/2);
  $('actual').onclick=()=>zoomAt(1,width/2,height/2);
  $('clear').onclick=()=>{selection=null;$('selected').value='';$('copy').disabled=true;try{localStorage.removeItem(storageKey);}catch(_){}draw();};
  $('copy').onclick=async()=>{
    if(!selection)return;
    try{await navigator.clipboard.writeText(pointText(selection));$('copy').textContent='已复制';}
    catch(_){$('selected').focus();$('selected').select();$('copy').textContent='请按 Ctrl+C';}
    setTimeout(()=>{$('copy').textContent='复制坐标';},1600);
  };
  $('locate').onclick=()=>{
    if(!observation || !observation.matched)return;
    const loc=observation.location;
    const xs=loc.viewport.map(p=>p[0]),ys=loc.viewport.map(p=>p[1]);
    zoom=Math.min(width/(Math.max(...xs)-Math.min(...xs)),height/(Math.max(...ys)-Math.min(...ys)))*0.85;
    panX=width/2-loc.center[0]*zoom;panY=height/2-loc.center[1]*zoom;fitted=false;draw();
  };
  async function refreshObservation(){
    if(!/^[0-9a-f]{16}$/.test(instance) || document.hidden)return;
    try{
      const response=await fetch('state-'+instance+'.json',{cache:'no-store'});
      if(response.status===404)return;
      if(!response.ok)throw new Error('读取失败');
      const value=await response.json(),loc=value.location;
      if(value.matched && (!loc || !Array.isArray(loc.viewport) || loc.viewport.length!==4 ||
          !Array.isArray(loc.center) || loc.center.length!==2 ||
          ![...loc.viewport.flat(),...loc.center].every(Number.isFinite)))throw new Error('定位记录格式错误');
      observation=value;$('locate').disabled=!value.matched;
      $('observation').textContent=value.matched
        ? `最近识别：${value.observed_at} · 截图中心 (${pointText(loc.center)}) · ${loc.inliers} 个匹配特征 · 误差 ${loc.error_px} px`
        : `最近识别：${value.observed_at} · ${value.reason || '未确认位置'}。`;
      draw();
    }catch(_){$('observation').textContent='定位记录暂时不可读，稍后重试。';observation=null;$('locate').disabled=true;draw();}
  }
  picture.onload=()=>{
    ready=true;pixels.width=picture.width;pixels.height=picture.height;pixelCtx.drawImage(picture,0,0);
    $('loading').hidden=true;
    try{const p=JSON.parse(localStorage.getItem(storageKey));if(Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&covered(p)){selection=p;$('selected').value=pointText(p);$('copy').disabled=false;}}catch(_){}
    resize();refreshObservation();
  };
  picture.onerror=()=>{$('loading').textContent='全景图加载失败，请检查 panorama.png。';};
  picture.src='panorama.png';new ResizeObserver(resize).observe($('map-wrap'));
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshObservation();});
  setInterval(refreshObservation,5000);
})();
