/* Display the route and current position from the existing configuration UI. */
(() => {
  'use strict';
  const svg = document.getElementById('map');
  if (!svg) return;
  const ns = 'http://www.w3.org/2000/svg', points = new Map(), edges = new Set(), positions = new Map();
  const key = (x, y) => `${Number(x).toFixed(1)},${Number(y).toFixed(1)}`;
  svg.querySelectorAll('#points .label').forEach(label => {
    const id = label.textContent.match(/P\d+/)?.[0];
    const node = label.parentElement.querySelector('circle.node');
    if (!id || !node) return;
    const p = {x:Number(node.getAttribute('cx')), y:Number(node.getAttribute('cy'))};
    points.set(id, p); positions.set(key(p.x, p.y), id);
  });
  svg.querySelectorAll('#routes line').forEach(line => {
    const a = positions.get(key(line.getAttribute('x1'), line.getAttribute('y1')));
    const b = positions.get(key(line.getAttribute('x2'), line.getAttribute('y2')));
    if (a && b) { edges.add(`${a}:${b}`); edges.add(`${b}:${a}`); }
  });
  function element(name, attrs, parent) {
    const e = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k,v)); parent.append(e); return e;
  }
  const layer = element('g', {id:'custom-route-overlay', 'pointer-events':'none'}, svg);
  const pointLayer = svg.querySelector('#points');
  if (pointLayer) pointLayer.before(layer);
  const path = element('path', {id:'custom-route-path', fill:'none', stroke:'#d886ff', 'stroke-width':5, opacity:.9, 'vector-effect':'non-scaling-stroke'}, layer);
  const current = element('g', {id:'canal-current-point', display:'none', 'pointer-events':'none'}, svg);
  const halo = element('circle', {fill:'#ff9f1a', 'fill-opacity':.25, stroke:'#ffb347', 'stroke-width':3, 'vector-effect':'non-scaling-stroke'}, current);
  const label = element('text', {fill:'#fff', stroke:'#101419', 'stroke-width':3, 'paint-order':'stroke', 'font-weight':'bold', 'vector-effect':'non-scaling-stroke'}, current);
  let state = {route:'', point:''}, lastRoute = null;
  function resizeMarker() {
    const p = points.get(state.point.trim().toUpperCase());
    if (!p) return;
    const scale = svg.viewBox.baseVal.width / (svg.getBoundingClientRect().width || 1000);
    halo.setAttribute('cx',p.x); halo.setAttribute('cy',p.y); halo.setAttribute('r',11*scale);
    label.setAttribute('x',p.x+16*scale); label.setAttribute('y',p.y-14*scale);
    label.setAttribute('font-size',14*scale);
  }
  function update(next) {
    state = {route:String(next.route || ''), point:String(next.point || '')};
    if (state.route !== lastRoute) {
      lastRoute = state.route;
      const route = state.route.trim().toUpperCase().split(/\s*[-－—]\s*/);
      const valid = route.length >= 2 && route.every((id,i) => points.has(id) && (!i || edges.has(`${route[i-1]}:${id}`)));
      path.setAttribute('d',valid ? route.map((id,i) => {const p=points.get(id); return `${i?'L':'M'}${p.x},${p.y}`;}).join('') : '');
    }
    const id = state.point.trim().toUpperCase();
    current.setAttribute('display',points.has(id) ? '' : 'none');
    label.textContent = `当前位置 ${id}`;
    resizeMarker();
  }
  new MutationObserver(resizeMarker).observe(svg, {attributes:true, attributeFilter:['viewBox']});
  window.addEventListener('resize',resizeMarker);
  window.addEventListener('message',event => {
    if (event.source === window.parent && event.origin === location.origin && event.data?.type === 'canal-map-state') update(event.data);
  });
  const params = new URLSearchParams(location.search);
  update({route:params.get('route'), point:params.get('point')});
  // Read the existing fields directly so edits are reflected without reopening the page.
  if (window.parent !== window) {
    try {
      const doc = window.parent.document;
      const prefix = 'JiangnanYangzhouGrandCanal_yangzhou_grand_canal_';
      const readFields = () => {
        const route = doc.querySelector(`[name="${prefix}CustomRoute"]`);
        const point = doc.querySelector(`[name="${prefix}LastPoint"]`);
        update({route:route ? route.value : state.route, point:point ? point.value : state.point});
      };
      const onEdit = event => {
        if ([prefix+'CustomRoute',prefix+'LastPoint'].includes(event.target.name)) readFields();
      };
      doc.addEventListener('input',onEdit); doc.addEventListener('change',onEdit);
      window.addEventListener('pagehide',() => {doc.removeEventListener('input',onEdit); doc.removeEventListener('change',onEdit);}, {once:true});
      readFields();
    } catch (_) { /* URL state remains available when opened outside the same-origin UI. */ }
  }
})();
