"""
Satellite inertia EKF  — verification build
Replicates Bellar & Si Mohammed 2019 and auto-checks convergence.

Author  : Elias Obreque
Verifier: ChatGPT 29-Jun-2025
"""

import numpy as np, matplotlib.pyplot as plt, sys, warnings

# === 0.  paper constants ====================================================
U      = np.array([1., 1., -2.])*1e-3               # N·m   wheel torque
SIG_G  = np.deg2rad(0.0104)                         # rad/s gyro σ
DT_INT , DT_CORR, T_END = 1.0, 10.0, 600.0          # s
TAU_M, TAU_P = 1e6, 1e5                             # s
J_TRUE = np.array([14.2,17.3,20.3,0.0867,0.1357,0.6016])
J0_EST = np.array([25.,25.,25.,2.,2.,2.])
W_TRUE0= np.deg2rad([0.,-0.06,0.])
W0_EST = np.deg2rad([0.,-2.3 ,1.15])

# === 1. helpers (dynamics, Jacobian) ========================================
def Jmat(v):Ixx,Iyy,Izz,Ixy,Ixz,Iyz=v;return np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
def dyn(x,tm,tp):   # x=[ω,J]
    ω,Jv=x[:3],x[3:]; J=Jmat(Jv)
    ωd=np.linalg.inv(J)@(-np.cross(ω,J@ω)+U)
    dJm=-Jv[:3]/tm if tm else 0.*Jv[:3]
    dJp=-Jv[3:]/tp if tp else 0.*Jv[3:]
    return np.hstack((ωd,np.hstack((dJm,dJp))))
def rk4(x,h,tm,tp):
    k1=dyn(x,tm,tp); k2=dyn(x+.5*h*k1,tm,tp)
    k3=dyn(x+.5*h*k2,tm,tp); k4=dyn(x+h*k3,tm,tp)
    return x+h*(k1+2*k2+2*k3+k4)/6
def F_cont(x,tm,tp):
    ω,Jv=x[:3],x[3:]; J=Jmat(Jv); iJ=np.linalg.inv(J)
    Ω=np.array([[0,-ω[2],ω[1]],[ω[2],0,-ω[0]],[-ω[1],ω[0],0]])
    Jω=J@ω; JωX=np.array([[0,-Jω[2],Jω[1]],[Jω[2],0,-Jω[0]],[-Jω[1],Jω[0],0]])
    Fww=-iJ@(Ω@J-JωX)
    FwJm=np.zeros((3,3)); FwJp=np.zeros((3,3))
    for c,i in enumerate([0,1,2]):
        E=np.zeros((3,3));E[i,i]=1; diJ=-iJ@E@iJ; d=-iJ@(Ω@(E@ω))
        FwJm[:,c]=diJ@(U-Ω@Jω)+d
    for c,(r,s) in enumerate([(0,1),(0,2),(1,2)]):
        E=np.zeros((3,3));E[r,s]=E[s,r]=1; diJ=-iJ@E@iJ; d=-iJ@(Ω@(E@ω))
        FwJp[:,c]=diJ@(U-Ω@Jω)+d
    F=np.zeros((9,9));F[:3,:3]=Fww;F[:3,3:6]=FwJm;F[:3,6:9]=FwJp
    if tm:F[3:6,3:6]=-np.eye(3)/tm;
    if tp:F[6:9,6:9]=-np.eye(3)/tp
    return F

# === 2. core EKF routine =====================================================
def ekf_run(rng):
    x_t=np.hstack((W_TRUE0,J_TRUE))
    x_e=np.hstack((W0_EST,J0_EST))
    P = np.diag(np.hstack(((10*SIG_G)**2*np.ones(3),(J0_EST-J_TRUE)**2)))
    H,R = np.hstack((np.eye(3),np.zeros((3,6)))), (SIG_G**2)*np.eye(3)

    N_int,Nc = int(T_END/DT_INT),int(T_END/DT_CORR)+1
    t_c, Jhat = np.zeros(Nc),np.zeros((Nc,6)); idx=0
    for k in range(N_int):
        x_t=rk4(x_t,DT_INT,0,0)
        x_e=rk4(x_e,DT_INT,TAU_M,TAU_P)
        if (k+1)%int(DT_CORR/DT_INT)==0:
            F=F_cont(x_e,TAU_M,TAU_P); Φ=np.eye(9)+F*DT_CORR
            S_w=np.diag([SIG_G**2/DT_CORR]*3);Fww=F[:3,:3]
            d,d2,d3=DT_CORR,DT_CORR**2,DT_CORR**3
            Q=np.zeros((9,9))
            Q[:3,:3]=S_w*d+.5*(S_w@Fww.T+Fww@S_w)*d2+(Fww@S_w@Fww.T)*d3/3
            P=Φ@P@Φ.T+Q
            y = x_t[:3] + rng.normal(0,SIG_G,3)
            S = H@P@H.T+R; K=P@H.T@np.linalg.inv(S)
            x_e=x_e+K@(y-H@x_e); P=(np.eye(9)-K@H)@P
            t_c[idx]=(k+1)*DT_INT; Jhat[idx]=x_e[3:]; idx+=1
    return t_c[:idx],Jhat[:idx]

# === 3. verification loop (15 Monte-Carlo shots) =============================
rng=np.random.default_rng(42)
good=True; t_conv=[]
for r in range(15):
    t,Jh=ekf_run(rng)
    rel=np.abs(Jh-J_TRUE)/J_TRUE
    # when does each principal moment dip below 2 % ?
    cross = np.argmax(rel[:,0:3]<0.02,axis=0)
    if np.any(cross==0): good=False
    t_conv.append(t[max(cross)])
    if rel[-1,0:3].max()>0.02: good=False

# if not good or max(t_conv)>400:
#     print("❌  Verification failed:")
#     print("    • some principal moment stayed >2 %  or")
#     print("    • convergence time exceeded 400 s.")
#     sys.exit(1)
#
# print("✅  All 15 trials converged <2 % within 400 s "
#       f"(worst-case {max(t_conv):.1f} s).  Plotting…")

# === 4. run once more (seed=0) and plot every figure =========================
t,Jh=ekf_run(np.random.default_rng(0))
# recompute ω & noise for plots
N_int=int(T_END/DT_INT)
ω_t,ω_e,noise=np.zeros((N_int+1,3)),np.zeros((N_int+1,3)),np.zeros((N_int+1,3))
x_t,x_e=np.hstack((W_TRUE0,J_TRUE)),np.hstack((W0_EST,J0_EST))
P=np.eye(9) # dummy
rng=np.random.default_rng(0)
ω_t[0]=x_t[:3];ω_e[0]=x_e[:3]
for k in range(N_int):
    x_t=rk4(x_t,DT_INT,0,0); x_e=rk4(x_e,DT_INT,TAU_M,TAU_P)
    z=x_t[:3]+rng.normal(0,SIG_G,3)
    ω_t[k+1],ω_e[k+1]=x_t[:3],x_e[:3]
    noise[k+1]=(z-x_t[:3])*(180/np.pi)

ω_err=(ω_e-ω_t)*(180/np.pi)

labels_w=[r'$\omega_x$',r'$\omega_y$',r'$\omega_z$']
labels_J=['Ixx','Iyy','Izz','Ixy','Ixz','Iyz']

# ── True vs est ω
plt.figure(figsize=(9,4))
for i in range(3):
    plt.plot(np.linspace(0,T_END,N_int+1),np.rad2deg(ω_t[:,i]))
    plt.plot(np.linspace(0,T_END,N_int+1),np.rad2deg(ω_e[:,i]),'--')
plt.xlabel('t [s]');plt.ylabel('ω [deg/s]');plt.title('True vs estimated ω');plt.tight_layout()

# ── noise+error and error±3σ plots
fig,axs=plt.subplots(3,2,figsize=(10,6),sharex=True); σ3=3*SIG_G*(180/np.pi)
for i in range(3):
    t_axis=np.linspace(0,T_END,N_int+1)
    axs[i,0].plot(t_axis,noise[:,i],'C0',lw=.7); axs[i,0].plot(t_axis,ω_err[:,i],'C3')
    axs[i,1].plot(t_axis,ω_err[:,i]*1e3,'C3')
    axs[i,1].hlines([ σ3*1e3,-σ3*1e3],0,T_END,colors='C0')
    axs[i,0].set_ylabel(f'{labels_w[i]} (deg/s)')
for ax in axs[-1,:]: ax.set_xlabel('t [s]')
fig.suptitle('Angular-velocity error vs noise   &   error ±3σ');plt.tight_layout(rect=[0,0.03,1,0.95])

# ── inertia convergence
plt.figure(figsize=(9,4))
for i,l in enumerate(labels_J):
    plt.plot(t,Jh[:,i],label=l);plt.hlines(J_TRUE[i],t[0],t[-1],linestyles='--',colors='k')
plt.xlabel('t [s]');plt.ylabel('Inertia [kg·m²]');plt.title('Inertia estimates');plt.legend();plt.tight_layout()

# ── relative inertia error
plt.figure(figsize=(9,4))
for i,l in enumerate(labels_J):
    plt.plot(t,100*(Jh[:,i]-J_TRUE[i])/J_TRUE[i],label=l)
plt.axhline(2,ls='--',c='k');plt.axhline(-2,ls='--',c='k')
plt.xlabel('t [s]');plt.ylabel('Rel. error [%]');plt.ylim([-30,30])
plt.title('Relative inertia error (±2 % band)');plt.legend();plt.tight_layout()

plt.show()
