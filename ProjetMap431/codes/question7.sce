exec('calcul2.sce',-1) 

//Méthode de Monte Carlo
M = 1000

// Uniforme
E_MC = zeros(N)
for i=1:M
    mu = 0.99 * rand() + 0.01
    E_MC = E_MC + resoudre2(mu)
end
E_MC = E_MC/M
U_mu_bar = resoudre2( (1+0.01)/2)

a=gca();
a.font_size=4;
a.thickness=2;
plot2d(mailles, [E_MC U_mu_bar], 1:2)
legend(["$E_{MC}$"; "$U_{\mu}$"])



//Log unifomre
E_MC = zeros(N)
for i=1:M
    mu = 10**(rand()*2-2)
    E_MC = E_MC + resoudre2(mu)
end
E_MC = E_MC/M
U_mu_bar = resoudre2(0.99/log(100))

f1=scf(1);
scf(f1);
a=gca();
a.font_size=4;
a.thickness=2;

plot2d(mailles, [E_MC U_mu_bar], 1:2)
legend(["$E_{MC}$"; "$U_{\mu}$"])
