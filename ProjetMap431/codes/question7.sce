exec('calcul2.sce',-1) 

//Méthode de Monte Carlo

function Y = aleatoire()
    //Y = 0.99 * rand() + 0.01
    Y = 10**(rand()*2-2)
endfunction
  

M = 1000
E_MC = zeros(N)
for i=1:M
    mu = aleatoire()
    E_MC = E_MC + resoudre2(mu)
end

E_MC = E_MC/M

U_mu_bar = resoudre2(0.99/log(100))

plot(mailles, E_MC, "r")
plot(mailles, U_mu_bar)

