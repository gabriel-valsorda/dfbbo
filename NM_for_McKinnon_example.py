import numpy as np  
import matplotlib.pyplot as plt

def Nelden_Mead(fct, simplex, diam_tol = 1e-3, max_eval = 250):
    # parameters
    de = 2
    doc = 0.5
    dic = 0.5
    gamma = 0.5
    
    # evaluate simplex
    f = np.array([fct(s) for s in simplex])
    # dimension
    n = len(simplex[0])
    neval = len(f)

    while neval <= max_eval:
# Diameter tolerance
        diam = max(np.linalg.norm(simplex[i]-simplex[j])
                   for i in range(n+1)
                   for j in range(n+1))
        if diam < diam_tol:
            print("NM stopped because of small simplex radius")
            break

# 1) SORTING
        # Sort simplex [y0,...,yn]
        idx = np.argsort(f)
        simplex = simplex[idx]
        # Sort f(simplex)
        f = f[idx]
        fbest = f[0]

        xc = 1/n * np.sum(simplex[:-1], axis=0)
        
# 2) Reflect
        xr = xc + (xc - simplex[-1])
        fr = fct(xr)
        
        if fbest <= fr < f[-2]:
            simplex[-1] = xr
            f[-1] = fr
            continue 

# 3) Expand
        if fr < fbest:
            xe = xc + de*(xc-f[-1])
            fe = fct(xe)
            if fe < fr:
                simplex[-1] = xe
                f[-1] = fe
            else:
                simplex[-1] = xr
                f[-1] = fr
            continue

# 4) outside contraction
        if f[-2] <= fr < f[-1]:
            xoc = xc + doc*(xc-f[-1])
            foc = fct(xoc)
            if foc < fr:
                simplex[-1] = xoc
                f[-1] = foc
            else:
                simplex[-1] = xr
                f[-1] = fr
            continue
        
# 5) inside contraction
        if fr >= f[-1]:
            xic = xc - dic*(xc-f[-1])
            fic = fct(xic)
            if fic < f[-1]:
                simplex[-1] = xic
                f[-1] = fic
                continue
# 6) Shrink
        for i in range(1, n+1):
            simplex[i] = simplex[0] + gamma*(simplex[i]-simplex[0])
            f[i] = fct(simplex[i])
        
        neval += n
    print(f"NM reached {neval} function evaluations")
    return f, simplex

def McKinnon_fct(x):
    x1, x2 = x
    if x1 >= 0:
        return 360*x1**2 + x2**2 + x2
    else:
        return 6*x1**2 + x2**2 + x2
def McKinnon_fct_np(X1, X2):
    return np.where(X1 <= 0, 360*X1**2 + X2 + X2**2, 6*X1**2 + X2 + X2**2)

Lambda = (1+np.sqrt(33)) / 8
Mu = (1-np.sqrt(33)) / 8

Initial_McKinnon_simplex = np.array([[0,0], [Lambda, Mu], [1,1]])
# f, simplex = Nelden_Mead(McKinnon_fct, Initial_McKinnon_simplex)

# print(f, simplex)

# b) 
def pts_generator_unit_cercle(center, rayon=1):
    h, k = center
    x = np.random.uniform(h-rayon, h+rayon, size=20)

    pts = np.zeros((20,2))
    for i in range(len(x)):
        sign = np.random.default_rng().choice([-1, 1])
        pts[i,0] = x[i]
        pts[i,1] = sign*np.sqrt(rayon**2 - (x[i]-h)**2) + k**2
    return pts


pts = pts_generator_unit_cercle([1,1])

fct, simplex = Nelden_Mead(McKinnon_fct, pts)
print(fct)

plt.figure()

x1 = np.linspace(-0.25, 2, 100)
x2 = np.linspace(-0.5, 2.5, 100)
X1, X2 = np.meshgrid(x1,x2)
contours = plt.contour(X1, X2, McKinnon_fct_np(X1, X2), levels=50)


plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('scaled')
plt.grid()
plt.xticks()
plt.yticks()
plt.show()