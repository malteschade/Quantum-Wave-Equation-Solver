def hamiltonian(m, k):
    k1 = np.linspace(-0.9, 0, int((len(k)-1)/2)+2)
    k1 = raised_cosine(k1, 2e10, 1e10) 
    k2 = [3e10]*len(k)
    k = np.concatenate([k2])
    
    m1 = np.linspace(-0.9, 0, int(len(m)/2)+2)
    m1 = raised_cosine(m1, 2000, 1000) 
    m2 = [3000]*len(m)
    m = np.concatenate([m2])
    
    
    order = 1
    if order == 1:
        #FD (Order 1)
        D_u = (
            np.diag(np.full(len(m), -1), k=0) + 
            np.diag(np.full(len(m)-1, 1), k=-1)
            ).astype(float) / order
        
    elif order == 2:
        #FD (Order 2)
        D_u = (
            np.diag(np.full(len(m), -(3/2)), k=0) +
            np.diag(np.full(len(m)-1, 2), k=1) +
            np.diag(np.full(len(m)-2, -1/2), k=2)
        ).astype(float) / order
    elif order == 3:
        # FD (Order 3)
        D_u = (
            np.diag(np.full(len(m), -11/6), k=0) +
            np.diag(np.full(len(m)-1, 3), k=1) +
            np.diag(np.full(len(m)-2, -3/2), k=2) +
            np.diag(np.full(len(m)-3, 1/3), k=3)
        ).astype(float) / order
    elif order == 4:
        # FD (Order 4)
        D_u = (
            np.diag(np.full(len(m), -25/12), k=0) +
            np.diag(np.full(len(m)-1, 4), k=1) +
            np.diag(np.full(len(m)-2, -3), k=2) +
            np.diag(np.full(len(m)-3, 4/3), k=3) +
            np.diag(np.full(len(m)-4, -1/4), k=4)
        ).astype(float) / order
    
    n = len(D_u)
    Z = np.zeros((n,n))
    Z1 = np.zeros((n,n-1))
    I = np.identity(n)
    I1 = np.identity(n-1)
    ISM = np.diag(1/np.sqrt(m[:-1]))
    SE = np.diag(np.sqrt(k[:-1]))
    D = D_u[:,:-1]
    
    # Dirichlet 2 side
    D[0,0] = 1
    D[-1,-1] = 1
    
    U = SE @ D @ ISM

    IT = Z1.copy()
    IT[:-1,:] = I1
    T = np.block([[U, Z1],[Z1, IT]]) # 0 reihen an den falschen positionen? Eine in der mitte?
    INV_T = inv(T.T @ T) @ T.T 

    U1 = Z.copy()
    U1[:,:-1] = U
    H = np.block([[Z, -1j*U1],[1j*U1.T, Z]])
    
    K2 = -U.T @ U
    
    H1 = H 
    T = T
    INV_T = INV_T
    
    
    Z2 = np.zeros((n-1,n-1))
    DV = np.block([[Z2, I1],[K2, Z2]])
    return H1, T, INV_T, DV


Z = np.zeros((n,n))
U1 = Z.copy()
U1[:,:-1] = U
H = np.block([[Z, -1j*U1],[1j*U1.T, Z]])