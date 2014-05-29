exec('calcul.sce',-1) 

xset("color",6)


for i=omega_2'
    rect = [i(1) 0.03 (i(2)-i(1)) 0.06]
    xrect(rect)
end

xset("color",1)

mu = [0.01, 0.1, 1]
colors = ['r', 'g', 'b']
solutions = zeros(N, 3)
for i=1:3
    A = calc_A(mu(i))
    B = calc_B()
    solutions(:, i) = inv(A) * B
end

plot2d(mailles, solutions, 1:3)
legend(["$\mu=0.01$";"$\mu=0.1$";"$\mu=1$"]);
