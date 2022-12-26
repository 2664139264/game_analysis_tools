import numpy as np
from itertools import product

def sym(m:np.ndarray) -> np.ndarray:
    assert len(m.shape) == 2
    shape = (m.shape[0] * m.shape[1],) * 2
    sym_m = np.zeros(shape)
    for a,b,c,d in product(range(m.shape[0]), range(m.shape[1]), range(m.shape[0]), range(m.shape[1])):
        sym_m[a*m.shape[1]+b, c*m.shape[1]+d] = (m[a,d] - m[c,b]) / 2
    return sym_m

def grad(r:np.ndarray) -> np.ndarray:
    assert len(r.shape) == 1
    grad_r = np.zeros((r.shape[0], r.shape[0]))
    for a, b in product(range(r.shape[0]), range(r.shape[0])):
        grad_r[a, b] = r[a] - r[b]
    return grad_r

def div(m:np.ndarray) -> np.ndarray:
    assert len(m.shape) == 2
    div_m = np.zeros(m.shape[0])
    for a, b in product(range(m.shape[0]), range(m.shape[1])):
        div_m[a] += m[a, b]
    
    div_m /= m.shape[1]
    return div_m

def rot(m:np.ndarray) -> np.ndarray:
    assert len(m.shape) == 2 and m.shape[0] == m.shape[1]
    rot_m = np.zeros_like(m)
    for a, b, c in product(range(m.shape[0]), range(m.shape[0]), range(m.shape[0])):
        rot_m[a,b] += m[a,b] + m[b,c] + m[c,a]
    
    rot_m /= m.shape[0]
    return rot_m           

if __name__ == '__main__':
    m = np.array([[1,-2],[-4,8]], dtype=np.float32)
    sym_m = sym(m)
    print(sym_m)
    print(rot(sym_m))
    print(grad(div(sym_m)))
    print(rot(sym_m) + grad(div(sym_m)) - sym_m)