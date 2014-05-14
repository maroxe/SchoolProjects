exec('calcul2.sce',-1) 

mu = 0.075
u_rb = resoudre2(mu)
u_mu = resoudre(mu)


plot(mailles, u_rb, 'r')
plot(mailles, u_mu, 'g')

plot(mailles, abs(u_mu-u_rb), 'b')
