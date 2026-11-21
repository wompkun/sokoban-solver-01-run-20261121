// Sokoban solver entry for make.fun bounty sokoban-solver-01-run-20261121.
// Push-optimal per room: the generator lays a 1-wide hallway with 1-4 rooms
// off it (one door each, boxes never leave their room), so the level optimum
// is the sum of the rooms' optima. Each room is solved with A* over pushes
// (Hungarian-assignment heuristic, dead-square + 2x2 freeze pruning, player
// region normalisation, transposition table, fixed expansion budget), then
// the rooms' move strings are joined with hallway walks and the whole
// answer is verified by replay before it is written.
// Node 22 type stripping: erasable TypeScript only, node builtins only.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
/** Fixed expansion budget per room: deterministic, never the clock. */
const ROOM_BUDGET = 500_000;
/** Shared deterministic cap on expansions across the whole run (all 5
 * instances): bounds the worst-case wall time without ever reading a clock.
 * Practice uses ~3,000 expansions total; this is 300x headroom. */
const RUN_EXPANSIONS: { used: number; cap: number } = { used: 0, cap: 1_000_000 };
/** State-count guard per room: bounds memory (512 MB) if a search explodes. */
const ROOM_STATE_CAP = 1_500_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

interface Room {
  door: number;
  entry: number;
  cells: number[];
  boxes: number[];
  goals: number[];
}

interface RoomSolution {
  moves: string;
  pushes: number;
  finalBoxes: number[];
  finalPlayer: number;
}

/** Parse XSB text. Rows shorter than the widest are padded with wall. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

/** Split the level into rooms. The hallway is the player's row; each floor
 * cell in the rows beside it is a door, and its room is the flood fill from
 * that door that never enters the hallway row. */
function findRooms(lv: Level): Room[] {
  const { w, h, wall } = lv;
  const hall = Math.floor(lv.player / w);
  const rooms: Room[] = [];
  for (const dr of [-1, 1]) {
    const y = hall + dr;
    if (y < 0 || y >= h) continue;
    for (let x = 0; x < w; x++) {
      const door = y * w + x;
      if (wall[door]) continue;
      const seen = new Uint8Array(w * h);
      const cells: number[] = [];
      const stack = [door];
      seen[door] = 1;
      while (stack.length > 0) {
        const c = stack.pop()!;
        cells.push(c);
        for (let d = 0; d < 4; d++) {
          const q = c + DY[d]! * w + DX[d]!;
          if (q < 0 || q >= w * h) continue;
          if (Math.floor(q / w) === hall) continue;
          if (wall[q] || seen[q]) continue;
          seen[q] = 1;
          stack.push(q);
        }
      }
      const inRoom = (c: number): boolean => seen[c] === 1;
      cells.sort((a, b) => a - b);
      const key = cells.join(',');
      // A wide-open wall row can expose the same room through several
      // neighbouring floor cells; keep one door per room.
      if (rooms.some((r) => r.cells.join(',') === key)) continue;
      rooms.push({
        door,
        entry: hall * w + x,
        cells,
        boxes: lv.boxes.filter(inRoom),
        goals: cells.filter((c) => lv.goal[c] === 1),
      });
    }
  }
  rooms.sort((a, b) => a.door % w - (b.door % w) || a.door - b.door);
  return rooms.filter((r) => r.boxes.length > 0);
}

/** BFS distances from one start over floor cells, ignoring boxes. */
function bfsDist(w: number, h: number, floor: Uint8Array, from: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    for (let d = 0; d < 4; d++) {
      const q = c + DY[d]! * w + DX[d]!;
      if (q < 0 || q >= w * h || !floor[q] || dist[q] >= 0) continue;
      dist[q] = dist[c]! + 1;
      queue.push(q);
    }
  }
  return dist;
}

/** BFS walk path (move letters) from `from` to `to` over floor cells with
 * `blocked` cells treated as walls; null when unreachable. */
function walkPath(
  w: number,
  h: number,
  floor: Uint8Array,
  blocked: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    for (let d = 0; d < 4; d++) {
      const q = c + DY[d]! * w + DX[d]!;
      if (q < 0 || q >= w * h || !floor[q] || blocked[q] || prev[q] >= 0) continue;
      prev[q] = c;
      if (q === to) {
        let moves = '';
        let cur = q;
        while (cur !== from) {
          const p = prev[cur]!;
          const dd = cur - p;
          const d = dd === 1 ? 3 : dd === -1 ? 2 : dd === w ? 1 : 0;
          moves = CH[d]! + moves;
          cur = p;
        }
        return moves;
      }
      queue.push(q);
    }
  }
  return null;
}

/** Player's reachable region from `from` around walls and `blocked` boxes.
 * Returns the region mask and its canonical representative (lowest cell). */
function regionOf(
  w: number,
  h: number,
  floor: Uint8Array,
  blocked: Uint8Array,
  from: number,
): { mask: Uint8Array; rep: number } {
  const mask = new Uint8Array(w * h);
  const queue = [from];
  mask[from] = 1;
  let rep = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi]!;
    if (c < rep) rep = c;
    for (let d = 0; d < 4; d++) {
      const q = c + DY[d]! * w + DX[d]!;
      if (q < 0 || q >= w * h || !floor[q] || blocked[q] || mask[q]) continue;
      mask[q] = 1;
      queue.push(q);
    }
  }
  return { mask, rep };
}

/** Dead squares: floor cells (non-goals) from which no goal is reachable in
 * the push graph (edge b -> b+d iff b+d and b-d are floor). Computed by
 * reverse BFS from the goals: from x, predecessor y = x-d needs x-2d floor. */
function deadSquares(w: number, h: number, floor: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(w * h);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let d = 0; d < 4; d++) {
      const y = x - (DY[d]! * w + DX[d]!);
      const beyond = x - 2 * (DY[d]! * w + DX[d]!);
      if (y < 0 || y >= w * h || !floor[y] || live[y]) continue;
      if (beyond < 0 || beyond >= w * h || !floor[beyond]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  const dead = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (floor[i] && !live[i]) dead[i] = 1;
  return dead;
}

/** All permutations of [0..n-1] in fixed order (n <= 4). */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === 1) {
      out.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(k - 1);
      const j = k % 2 === 1 ? 0 : i;
      [a[j], a[k - 1]] = [a[k - 1]!, a[j]!];
    }
  };
  gen(n);
  return out;
}

/** Pack sorted box cells (each < 255) into one number; missing boxes are 255. */
function packBoxes(boxes: number[]): number {
  let p = 0;
  for (let i = 0; i < 4; i++) p |= (boxes[i] ?? 255) << (8 * i);
  return p >>> 0;
}

function unpackBoxes(packed: number, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push((packed >>> (8 * i)) & 0xff);
  return out;
}

/** Push-optimal A* for one room. `floor` covers the room cells plus the
 * entry cell; the player starts on the entry cell. Returns null when the
 * budget is exhausted or the room is deadlocked from the start. */
function solveRoom(
  w: number,
  h: number,
  floor: Uint8Array,
  boxes0: number[],
  goals: number[],
  startPlayer: number,
  budget: number,
): RoomSolution | null {
  const n = boxes0.length;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  const dead = deadSquares(w, h, floor, goals);
  for (const b of boxes0) if (dead[b] && !goalSet[b]) return null;
  // BFS distance from each goal over the room floor, ignoring boxes.
  const distFromGoal = goals.map((g) => bfsDist(w, h, floor, g));
  const perms = permutations(n);
  const heuristic = (boxes: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let s = 0;
      let ok = true;
      for (let i = 0; i < n; i++) {
        const d = distFromGoal[perm[i]!]![boxes[i]!]!;
        if (d < 0) {
          ok = false;
          break;
        }
        s += d;
      }
      if (ok && s < best) best = s;
    }
    return best;
  };
  const isGoal = (boxes: number[]): boolean => boxes.every((b) => goalSet[b] === 1);
  const blockedOf = (boxes: number[]): Uint8Array => {
    const b = new Uint8Array(w * h);
    for (const c of boxes) b[c] = 1;
    return b;
  };
  // 2x2 freeze: four boxes in a square, not all on goals, can never move.
  const frozen = (boxes: number[]): boolean => {
    const occ = blockedOf(boxes);
    for (let y = 0; y + 1 < h; y++) {
      for (let x = 0; x + 1 < w; x++) {
        const c = y * w + x;
        if (!floor[c] || !floor[c + 1] || !floor[c + w] || !floor[c + w + 1]) continue;
        if (occ[c] && occ[c + 1] && occ[c + w] && occ[c + w + 1]) {
          if (!(goalSet[c] && goalSet[c + 1] && goalSet[c + w] && goalSet[c + w + 1])) return true;
        }
      }
    }
    return false;
  };

  // State storage (parallel arrays, index = state id).
  const sBoxes: number[] = [];
  const sPlayer: number[] = [];
  const sRep: number[] = [];
  const sParent: number[] = [];
  const sPushBox: number[] = [];
  const sPushDir: number[] = [];
  const sG: number[] = [];
  const sF: number[] = [];
  const sCnt: number[] = [];
  const best = new Map<number, number>();
  const heap: number[] = [];
  const less = (a: number, b: number): boolean =>
    sF[a]! !== sF[b]! ? sF[a]! < sF[b]! : sG[a]! !== sG[b]! ? sG[a]! > sG[b]! : sCnt[a]! < sCnt[b]!;
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let counter = 0;
  const addState = (
    boxes: number[],
    player: number,
    rep: number,
    parent: number,
    pushBox: number,
    pushDir: number,
    g: number,
  ): number => {
    const id = sBoxes.length;
    const hv = heuristic(boxes);
    sBoxes.push(packBoxes(boxes));
    sPlayer.push(player);
    sRep.push(rep);
    sParent.push(parent);
    sPushBox.push(pushBox);
    sPushDir.push(pushDir);
    sG.push(g);
    sF.push(g + hv);
    sCnt.push(counter++);
    best.set(packBoxes(boxes) * 256 + rep, id);
    heapPush(id);
    return id;
  };

  const startRep = regionOf(w, h, floor, blockedOf(boxes0), startPlayer).rep;
  const root = addState(boxes0, startPlayer, startRep, -1, -1, -1, 0);
  if (isGoal(boxes0)) return { moves: '', pushes: 0, finalBoxes: boxes0.slice(), finalPlayer: startPlayer };
  let expanded = 0;
  while (heap.length > 0) {
    const s = heapPop();
    const boxes = unpackBoxes(sBoxes[s]!, n);
    if (best.get(sBoxes[s]! * 256 + sRep[s]!) !== s) continue;
    if (isGoal(boxes)) {
      // Reconstruct: for each push on the path, BFS the player's walk to the
      // stance cell with the pre-push boxes as obstacles, then the push.
      const path: number[] = [];
      for (let t = s; t !== root; t = sParent[t]!) path.push(t);
      path.reverse();
      let moves = '';
      let player = startPlayer;
      let curBoxes = boxes0.slice();
      for (const t of path) {
        const box = sPushBox[t]!;
        const d = sPushDir[t]!;
        const stance = box - (DY[d]! * w + DX[d]!);
        const walk = walkPath(w, h, floor, blockedOf(curBoxes), player, stance);
        if (walk === null) return null;
        moves += walk + CH[d]!;
        player = box;
        curBoxes = unpackBoxes(sBoxes[t]!, n);
      }
      return { moves, pushes: sG[s]!, finalBoxes: boxes, finalPlayer: player };
    }
    if (expanded >= budget || RUN_EXPANSIONS.used >= RUN_EXPANSIONS.cap) return null;
    if (sBoxes.length > ROOM_STATE_CAP) return null;
    expanded++;
    RUN_EXPANSIONS.used++;
    const { mask: region } = regionOf(w, h, floor, blockedOf(boxes), sPlayer[s]!);
    const occ = blockedOf(boxes);
    for (let bi = 0; bi < n; bi++) {
      const box = boxes[bi]!;
      for (let d = 0; d < 4; d++) {
        const dd = DY[d]! * w + DX[d]!;
        const dest = box + dd;
        const stance = box - dd;
        if (dest < 0 || dest >= w * h || !floor[dest] || occ[dest]) continue;
        if (stance < 0 || stance >= w * h || !region[stance]) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b) => a - b);
        if (nb.some((b) => dead[b] && !goalSet[b])) continue;
        if (frozen(nb)) continue;
        if (heuristic(nb) === Infinity) continue;
        const nPacked = packBoxes(nb);
        const nRep = regionOf(w, h, floor, blockedOf(nb), box).rep;
        const key = nPacked * 256 + nRep;
        const seen = best.get(key);
        if (seen !== undefined && sG[seen]! <= sG[s]! + 1) continue;
        addState(nb, box, nRep, s, box, d, sG[s]! + 1);
      }
    }
  }
  return null;
}

/** Replay moves on the level exactly like the kit checker; count pushes. */
function replay(lv: Level, moves: string): { ok: boolean; pushes: number } {
  const box = new Uint8Array(lv.w * lv.h);
  for (const b of lv.boxes) box[b] = 1;
  let player = lv.player;
  let pushes = 0;
  for (let k = 0; k < moves.length; k++) {
    const c = moves[k]!.toLowerCase();
    const d = c === 'u' ? -lv.w : c === 'd' ? lv.w : c === 'l' ? -1 : 1;
    const next = player + d;
    if (lv.wall[next]) return { ok: false, pushes };
    if (box[next]) {
      const beyond = next + d;
      if (lv.wall[beyond] || box[beyond]) return { ok: false, pushes };
      box[next] = 0;
      box[beyond] = 1;
      pushes++;
    }
    player = next;
  }
  for (let i = 0; i < lv.w * lv.h; i++) if (box[i] && !lv.goal[i]) return { ok: false, pushes };
  return { ok: true, pushes };
}

/** Solve one level: decompose into rooms, solve each, join with walks. */
function solveLevel(text: string): string {
  const lv = parseLevel(text);
  const floorAll = new Uint8Array(lv.w * lv.h);
  for (let i = 0; i < lv.w * lv.h; i++) floorAll[i] = lv.wall[i] ? 0 : 1;
  const rooms = findRooms(lv);
  // No doors found: treat the whole level as one room.
  const jobs: Room[] =
    rooms.length > 0
      ? rooms
      : [
          {
            door: -1,
            entry: lv.player,
            cells: [],
            boxes: lv.boxes.slice(),
            goals: (() => {
              const g: number[] = [];
              for (let i = 0; i < lv.w * lv.h; i++) if (lv.goal[i]) g.push(i);
              return g;
            })(),
          },
        ];
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes.slice();
  for (const room of jobs) {
    const roomFloor = new Uint8Array(lv.w * lv.h);
    if (room.cells.length > 0) {
      for (const c of room.cells) roomFloor[c] = 1;
      roomFloor[room.entry] = 1;
    } else {
      roomFloor.set(floorAll);
    }
    // Walk to the room's entry with the current boxes as obstacles.
    const blocked = new Uint8Array(lv.w * lv.h);
    for (const b of boxes) blocked[b] = 1;
    const walk = walkPath(lv.w, lv.h, floorAll, blocked, player, room.entry);
    if (walk === null) return '';
    moves += walk;
    player = room.entry;
    // The room's boxes as they stand NOW (an earlier room's solution may
    // have moved a box that this room's parse-time list still shows stale).
    const roomCellSet = new Set(room.cells);
    const curRoomBoxes =
      room.cells.length > 0 ? boxes.filter((b) => roomCellSet.has(b)) : boxes.slice();
    const solved = solveRoom(lv.w, lv.h, roomFloor, curRoomBoxes, room.goals, room.entry, ROOM_BUDGET);
    if (solved === null) return '';
    moves += solved.moves;
    player = solved.finalPlayer;
    boxes = boxes.filter((b) => !roomCellSet.has(b)).concat(solved.finalBoxes);
    boxes.sort((a, b) => a - b);
  }
  const check = replay(lv, moves);
  return check.ok ? moves : '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
