exec('calcul.sce',-1) 

N0 = 3
tab_mu = [0.05, 0.2, 1]

// Calcul de A0, A1 et B
A0 = calc_A(0)
A1 = calc_A(1) - A0
B = calc_B()

// Famille des solutions Xi
solutions = zeros(N, N0)
for j=1:N0
    solutions(:, j) = resoudre(tab_mu(j))
end

// Evaluation de A0_RB, A1_RB et B_RB
A0_RB = zeros(N0, N0)
A1_RB = zeros(N0, N0)
for i=1:N0
    for j=1:i
        a_rb = solutions(:,i)' * (A0 * solutions(:,j))
        A0_RB(i, j) = a_rb
        A0_RB(j, i)  = a_rb
        
        a_rb = solutions(:,i)' *  (A1 * solutions(:,j))
        A1_RB(i, j) = a_rb
        A1_RB(j, i)  = a_rb
    end
end

B_RB = zeros(N0)
for i=1:N0
    B_RB(i) = B' * solutions(:,i)
end

// Resolution du petit système linéaire
function U=resoudre2(mu)
    U = solutions * ( inv(A0_RB + mu * A1_RB) * B_RB )
endfunction

