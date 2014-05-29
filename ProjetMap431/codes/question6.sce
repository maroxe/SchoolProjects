exec('calcul2.sce',-1) 

mu = 0.075
u_rb = resoudre2(mu)
u_mu = resoudre(mu)


plot2d(mailles, [u_rb u_mu], 1:2)
legend(["$u_\mu$"; "$u_{rb}$"])
f1=scf(1);
scf(f1);

plot(mailles, abs(u_mu-u_rb), 'b')
