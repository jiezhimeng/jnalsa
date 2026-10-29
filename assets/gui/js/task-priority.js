(() => {
    const root = document.getElementById(container_id);
    if (!root) return;
    const list = root.querySelector('.task-priority-list');
    const feedback = root.querySelector('.task-priority-feedback');
    const order = () => Array.from(list.children, row => row.dataset.task);
    const renumber = () => Array.from(list.children).forEach((row, index) => {
        row.querySelector('.task-priority-rank').textContent = index + 1;
    });
    const restore = names => {
        const rows = new Map(Array.from(list.children, row => [row.dataset.task, row]));
        if (names.length !== rows.size || new Set(names).size !== rows.size ||
            names.some(name => !rows.has(name))) return;
        names.forEach(name => list.appendChild(rows.get(name)));
        renumber();
    };
    let savedOrder = order();
    let saving = false;
    let drag = null;
    const submit = () => {
        renumber();
        const next = order();
        if (JSON.stringify(next) === JSON.stringify(savedOrder)) return;
        saving = true;
        root.setAttribute('aria-busy', 'true');
        feedback.textContent = labels.saving;
        feedback.classList.remove('save-failed');
        WebIO.pushData(next, callback_id);
    };
    root.finishSave = (success, names) => {
        savedOrder = names;
        restore(savedOrder);
        saving = false;
        root.setAttribute('aria-busy', 'false');
        feedback.textContent = success ? labels.saved : labels.failed;
        feedback.classList.toggle('save-failed', !success);
    };
    const finishDrag = cancelled => {
        if (!drag) return;
        const current = drag;
        drag = null;
        current.row.classList.remove('dragging');
        if (!current.native && current.handle.hasPointerCapture(current.pointerId)) {
            current.handle.releasePointerCapture(current.pointerId);
        }
        if (cancelled) restore(current.before);
        else submit();
    };
    list.addEventListener('pointerdown', event => {
        const handle = event.target.closest('.task-priority-handle');
        if (!handle || saving || drag || event.button !== 0) return;
        if (event.pointerType === 'mouse') return;
        event.preventDefault();
        handle.focus({preventScroll: true});
        const row = handle.closest('.task-priority-item');
        drag = {row, handle, pointerId: event.pointerId, before: order()};
        handle.setPointerCapture(event.pointerId);
        row.classList.add('dragging');
    });
    const moveRow = event => {
        const scroller = root.closest('#pywebio-scope-groups') || root.parentElement;
        const bounds = scroller.getBoundingClientRect();
        if (event.clientY < bounds.top + 48) scroller.scrollTop -= 18;
        if (event.clientY > bounds.bottom - 48) scroller.scrollTop += 18;
        const hit = document.elementFromPoint(event.clientX, event.clientY);
        const target = hit && hit.closest('.task-priority-item');
        if (!target || target === drag.row || target.parentElement !== list) return;
        const box = target.getBoundingClientRect();
        list.insertBefore(drag.row, event.clientY < box.top + box.height / 2
            ? target : target.nextSibling);
        renumber();
    };
    list.addEventListener('dragstart', event => {
        const handle = event.target.closest('.task-priority-handle');
        if (!handle || saving || drag) {
            event.preventDefault();
            return;
        }
        const row = handle.closest('.task-priority-item');
        drag = {row, handle, native: true, before: order()};
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', row.dataset.task);
        row.classList.add('dragging');
    });
    list.addEventListener('dragover', event => {
        if (!drag || !drag.native) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
        moveRow(event);
    });
    list.addEventListener('drop', event => {
        if (!drag || !drag.native) return;
        event.preventDefault();
        moveRow(event);
        finishDrag(false);
    });
    list.addEventListener('dragend', () => finishDrag(true));
    list.addEventListener('pointermove', event => {
        if (!drag || drag.native || event.pointerId !== drag.pointerId) return;
        event.preventDefault();
        moveRow(event);
    });
    list.addEventListener('pointerup', event => {
        if (drag && event.pointerId === drag.pointerId) finishDrag(false);
    });
    list.addEventListener('pointercancel', () => { if (drag && !drag.native) finishDrag(true); });
    list.addEventListener('lostpointercapture', () => { if (drag && !drag.native) finishDrag(true); });
    list.addEventListener('keydown', event => {
        if (event.key === 'Escape' && drag) {
            finishDrag(true);
            return;
        }
        const handle = event.target.closest('.task-priority-handle');
        if (!handle || saving || drag || !['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const row = handle.closest('.task-priority-item');
        if (event.key === 'ArrowUp' && row.previousElementSibling) {
            list.insertBefore(row, row.previousElementSibling);
        } else if (event.key === 'ArrowDown' && row.nextElementSibling) {
            list.insertBefore(row.nextElementSibling, row);
        } else if (event.key === 'Home') list.prepend(row);
        else if (event.key === 'End') list.appendChild(row);
        handle.focus({preventScroll: true});
        row.scrollIntoView({block: 'nearest'});
        submit();
    });
})();
