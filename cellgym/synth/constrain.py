"""Grammar-constrained decoding for the plan model.

    g = PlanGrammar(netlist)
    g.allowed(prefix_ids) -> set[int]          # tokens that may follow the target prefix
    ids = sample_constrained(model, prompt_ids, netlist, n=4, temperature=0.8)

The grammar is a token-level state machine over tokens.VOCAB (target side only,
no <bos>/<sep>). It enforces:

    col <i> sd n=<net>[*] p=<net>[*] \\n      i = 0,1,2,... in order; net in the netlist's
    col <i> g  n=<in> p=<in> \\n               device nets or '-'; '*' never on '-';
                                              g columns carry one input (same on n= and p=)
    route <net> li1|met1 <track> <endpoint>{2,} \\n
    pin <net> <landing> [<track>] \\n
    <eos>

Endpoints/landings are `gate:<c>` (column c is a g column of that net) or
`nmos:<c>` / `pmos:<c>` (column c is a contacted sd column whose n=/p= is that
net), so the column index is always < the number of columns. Phases are ordered
cols -> routes -> pins. The model picks the number of columns: after each
completed col line a `col`, `route` or `pin` line may follow, but route/pin are
only offered once every input has a g column and some output has a contacted sd
column (otherwise the plan could never be finished). <eos> is allowed only when
every input and at least one output has a pin line.

Not enforced: device coverage/matching, diffusion sharing consistency, which
net is contacted, track existence at the cell's height, route connectivity,
overlaps, split gate columns (n=A p=B; opt in with split_gates=True), `st=`.
"""

from __future__ import annotations

import numpy as np

from cellgym.baselines.common import outputs_of
from cellgym.synth import tokens as T

V = len(T.VOCAB)
_ID = T.TOKEN_ID
MAX_COLS = T.MAX_NUM + 1
_TRACK_IDS = [_ID[t] for t in T.TRACKS]
_NUM_IDS = [_ID[str(i)] for i in range(MAX_COLS)]
_KIND_TOK = {"gate": _ID["gate:"], "nmos": _ID["nmos:"], "pmos": _ID["pmos:"]}


class State:
    __slots__ = ("phase", "sub", "cols", "kind", "net", "stars", "rnet", "used", "ekind",
                 "pinned", "pnet", "pkind", "pcol")

    def __init__(self):
        self.phase = 0              # 0 cols, 1 routes, 2 pins
        self.sub = "L"              # position inside a line
        self.cols = ()              # (kind, n, p, n_star, p_star) per column
        self.kind = None            # current col kind
        self.net = None             # n= net of the column being read
        self.stars = [False, False]
        self.rnet = None            # route/pin net
        self.used = ()              # (kind, col) endpoints used on this route
        self.ekind = None
        self.pinned = frozenset()
        self.pnet = self.pkind = self.pcol = None

    def clone(self) -> "State":
        s = State.__new__(State)
        for a in State.__slots__:
            setattr(s, a, getattr(self, a))
        s.stars = list(self.stars)
        return s


class PlanGrammar:
    def __init__(self, netlist: dict, split_gates: bool = False):
        self.inputs = [x for x in netlist["inputs"] if x in _ID]
        self.outputs = [x for x in outputs_of(netlist) if x in _ID]
        sd = {d[k] for d in netlist["devices"] for k in ("drain", "source")}
        self.sd_nets = sorted(x for x in sd if x in _ID)
        self.all_nets = sorted(set(self.sd_nets) | set(self.inputs) | set(self.outputs))
        self.split_gates = split_gates
        self._memo: dict = {}

    # -- column queries ------------------------------------------------------------
    @staticmethod
    def _elig(cols, net, kind):
        if kind == "gate":
            return [i for i, c in enumerate(cols) if c[0] == "g" and c[1] == net]
        j = 1 if kind == "nmos" else 2
        return [i for i, c in enumerate(cols) if c[0] == "sd" and c[j] == net and c[j + 2]]

    def _route_nets(self, cols):
        return [n for n in self.all_nets
                if sum(len(self._elig(cols, n, k)) for k in ("gate", "nmos", "pmos")) >= 2]

    def _can_leave(self, cols):
        return (bool(cols)
                and all(self._elig(cols, n, "gate") for n in self.inputs)
                and any(self._elig(cols, n, k) for n in self.outputs for k in ("nmos", "pmos")))

    def _pin_nets(self, s):
        out = [n for n in self.inputs if n not in s.pinned]
        out += [n for n in self.outputs if n not in s.pinned
                and any(self._elig(s.cols, n, k) for k in ("nmos", "pmos"))]
        return out

    def _done(self, s):
        return (all(n in s.pinned for n in self.inputs)
                and any(n in s.pinned for n in self.outputs))

    # -- allowed tokens --------------------------------------------------------------
    def allowed_ids(self, s: State, left: int | None = None) -> tuple[int, ...]:
        """Allowed next ids. left = tokens still available (incl. this one): near the end
        no new col/route line is opened, so the plan can still be finished."""
        sub = s.sub
        if sub == "L" and left is not None:
            return self._dynamic(s, left)
        if sub in ("L", "R_NET", "R_EKIND", "R_ENUM", "R_AFTER", "P_NET", "P_NUM", "CN_NET",
                   "CP_NET", "C_NUM", "CN_AFTER", "CP_AFTER", "P_KIND"):
            return self._dynamic(s)
        key = (sub, s.kind)
        r = self._memo.get(key)
        if r is None:
            r = self._memo[key] = self._static(s)
        return r

    def _static(self, s):
        sub = s.sub
        if sub == "C_KIND":
            return (_ID["sd"], _ID["g"])
        if sub == "CN_EQ":
            return (_ID["n="],)
        if sub == "CP_EQ":
            return (_ID["p="],)
        if sub == "R_LAYER":
            return (_ID["li1"], _ID["met1"])
        if sub == "R_TRACK":
            return tuple(_TRACK_IDS)
        if sub == "P_AFTER":
            return (T.NL, *_TRACK_IDS)
        if sub in ("P_END", "CP_END"):
            return (T.NL,)
        return ()

    def _reserve(self, s):
        # tokens needed to pin what is still unpinned, plus <eos>
        r = 1 + 5 * sum(n not in s.pinned for n in self.inputs)
        return r + (0 if any(n in s.pinned for n in self.outputs) else 5)

    def _dynamic(self, s, left=None):
        sub, cols = s.sub, s.cols
        if sub == "L":
            out = []
            room = 10**9 if left is None else left - self._reserve(s)
            if s.phase == 0:
                if len(cols) < MAX_COLS and room >= 10:
                    out.append(_ID["col"])
                if self._can_leave(cols):
                    out.append(_ID["pin"])
                    if self._route_nets(cols) and room >= 40:
                        out.append(_ID["route"])
                elif not out:
                    # the plan cannot be completed in the budget: stop cleanly (budget-only
                    # escape, never taken without `left`) rather than truncate mid-line
                    out.append(_ID["col"] if left is None and len(cols) < MAX_COLS else T.EOS)
            elif s.phase == 1:
                out.append(_ID["pin"])
                if self._route_nets(cols) and room >= 40:
                    out.append(_ID["route"])
            else:
                if self._pin_nets(s):
                    out.append(_ID["pin"])
                if self._done(s):
                    out.append(T.EOS)
            return tuple(out)
        if sub == "C_NUM":
            return (_NUM_IDS[len(cols)],)
        if sub == "CN_NET":
            if s.kind == "g":
                return tuple(_ID[x] for x in self.inputs)
            return tuple(_ID[x] for x in self.sd_nets) + (_ID["-"],)
        if sub == "CP_NET":
            if s.kind == "g":
                if self.split_gates:
                    return tuple(_ID[x] for x in self.inputs)
                return (_ID[s.net],)
            return tuple(_ID[x] for x in self.sd_nets) + (_ID["-"],)
        if sub in ("CN_AFTER", "CP_AFTER"):
            nxt = _ID["p="] if sub == "CN_AFTER" else T.NL
            last = s.net if sub == "CN_AFTER" else s.pnet
            if s.kind == "sd" and last != "-":
                return (_ID["*"], nxt)
            return (nxt,)
        if sub == "R_NET":
            return tuple(_ID[x] for x in self._route_nets(cols))
        if sub == "R_EKIND":
            out = []
            for k in ("gate", "nmos", "pmos"):
                if any((k, i) not in s.used for i in self._elig(cols, s.rnet, k)):
                    out.append(_KIND_TOK[k])
            return tuple(out)
        if sub == "R_ENUM":
            return tuple(_NUM_IDS[i] for i in self._elig(cols, s.rnet, s.ekind)
                         if (s.ekind, i) not in s.used)
        if sub == "R_AFTER":
            out = [_KIND_TOK[k] for k in ("gate", "nmos", "pmos")
                   if any((k, i) not in s.used for i in self._elig(cols, s.rnet, k))]
            if len(s.used) >= 2:
                out.append(T.NL)
            return tuple(out)
        if sub == "P_NET":
            return tuple(_ID[x] for x in self._pin_nets(s))
        if sub == "P_KIND":
            return (_KIND_TOK["gate"],) if s.rnet in self.inputs else \
                tuple(_KIND_TOK[k] for k in ("nmos", "pmos") if self._elig(cols, s.rnet, k))
        if sub == "P_NUM":
            return tuple(_NUM_IDS[i] for i in self._elig(cols, s.rnet, s.pkind))
        raise AssertionError(sub)

    # -- transitions -----------------------------------------------------------------
    def step(self, s: State, tok: int) -> State | None:
        """New state after tok, or None if tok is not allowed."""
        sub = s.sub
        if tok not in self.allowed_ids(s):
            return None
        t = s.clone()
        w = T.VOCAB[tok]
        if tok == T.EOS:
            t.sub = "DONE"
        elif sub == "L":
            t.sub = {"col": "C_NUM", "route": "R_NET", "pin": "P_NET"}[w]
            if w == "route":
                t.phase = max(t.phase, 1)
            elif w == "pin":
                t.phase = 2
        elif sub == "C_NUM":
            t.sub = "C_KIND"
        elif sub == "C_KIND":
            t.kind, t.sub, t.stars = w, "CN_EQ", [False, False]
        elif sub == "CN_EQ":
            t.sub = "CN_NET"
        elif sub == "CN_NET":
            t.net, t.sub = w, "CN_AFTER"
        elif sub == "CN_AFTER":
            if w == "*":
                t.stars[0], t.sub = True, "CP_EQ"
            else:
                t.sub = "CP_NET"
        elif sub == "CP_EQ":
            t.sub = "CP_NET"
        elif sub == "CP_NET":
            t.pnet, t.sub = w, "CP_AFTER"
        elif sub == "CP_AFTER":
            if w == "*":
                t.stars[1], t.sub = True, "CP_END"
            else:
                self._end_col(t)
        elif sub == "CP_END":
            self._end_col(t)
        elif sub == "R_NET":
            t.rnet, t.used, t.sub = w, (), "R_LAYER"
        elif sub == "R_LAYER":
            t.sub = "R_TRACK"
        elif sub == "R_TRACK":
            t.sub = "R_EKIND"
        elif sub in ("R_EKIND", "R_AFTER"):
            if tok == T.NL:
                t.sub = "L"
            else:
                t.ekind, t.sub = w[:-1], "R_ENUM"
        elif sub == "R_ENUM":
            t.used = s.used + ((s.ekind, int(w)),)
            t.sub = "R_AFTER"
        elif sub == "P_NET":
            t.rnet, t.sub = w, "P_KIND"
        elif sub == "P_KIND":
            t.pkind, t.sub = w[:-1], "P_NUM"
        elif sub == "P_NUM":
            t.sub = "P_AFTER"
        elif sub == "P_AFTER":
            if tok == T.NL:
                t.pinned = s.pinned | {s.rnet}
                t.sub = "L"
            else:
                t.sub = "P_END"
        elif sub == "P_END":
            t.pinned = s.pinned | {s.rnet}
            t.sub = "L"
        else:
            raise AssertionError(sub)
        return t

    def _end_col(self, t: State):
        n, p = t.net, t.pnet
        t.cols = t.cols + ((t.kind, n, p, t.stars[0], t.stars[1]),)
        t.sub = "L"

    # -- public API ------------------------------------------------------------------
    def initial(self) -> State:
        return State()

    def walk(self, prefix) -> State | None:
        s = self.initial()
        for tok in prefix:
            s = self.step(s, self._to_id(tok))
            if s is None:
                return None
        return s

    @staticmethod
    def _to_id(tok) -> int:
        return tok if isinstance(tok, (int, np.integer)) else _ID[tok]

    def allowed(self, prefix_tokens) -> set[int]:
        """Token ids that may follow the prefix (ids or token strings); empty if the prefix
        is itself invalid or complete."""
        s = self.walk(prefix_tokens)
        if s is None or s.sub == "DONE":
            return set()
        return set(self.allowed_ids(s))

    def accepts(self, tokens) -> bool:
        """True if the full token sequence (ending in <eos>) is in the language."""
        s = self.walk(tokens)
        return s is not None and s.sub == "DONE"

    def mask(self, s: State, left: int | None = None) -> np.ndarray:
        m = np.zeros(V, dtype=bool)
        m[list(self.allowed_ids(s, left))] = True
        return m


# -- constrained sampling -------------------------------------------------------------------

def sample_constrained(model, prompt_ids: list[int], netlist: dict, n: int = 1,
                       temperature: float = 1.0, max_new: int = 320, top_k: int | None = None,
                       grammar: PlanGrammar | None = None) -> list[list[int]]:
    """Like model.sample, but every step's logits are masked to the grammar.

    Returns n lists of generated ids, cut at and including EOS (or max_new tokens).
    """
    import torch
    import torch.nn.functional as F

    g = grammar or PlanGrammar(netlist)
    with torch.no_grad():
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        max_new = min(max_new, model.cfg.max_len - len(prompt_ids))
        caches = [[] for _ in model.blocks]
        x = torch.tensor([prompt_ids] * n, device=dev)
        states = [g.initial() for _ in range(n)]
        out = [[] for _ in range(n)]
        done = [False] * n
        pos = 0
        for step_i in range(max_new):
            logits = model(x, caches, start=pos)[:, -1].float()
            pos += x.shape[1]
            m = np.ones((n, V), dtype=bool)
            left = max_new - step_i
            for i in range(n):
                if not done[i]:
                    m[i] = g.mask(states[i], left)
            logits = logits.masked_fill(~torch.from_numpy(m).to(dev), float("-inf"))
            if top_k:
                kth = logits.topk(min(top_k, V), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if temperature <= 0 or top_k == 1:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(1)
            nl = nxt.tolist()
            for i in range(n):
                if done[i]:
                    nl[i] = T.PAD
                    continue
                out[i].append(nl[i])
                states[i] = g.step(states[i], nl[i])
                if nl[i] == T.EOS:
                    done[i] = True
            if all(done):
                break
            x = torch.tensor(nl, device=dev)[:, None]
        if was_training:
            model.train()
    return out
