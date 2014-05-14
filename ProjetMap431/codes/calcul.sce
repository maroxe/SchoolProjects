clear
N = 100
h = 1/(N+1)
omega_2 = [0.19, 0.21; 0.38, 0.42; 0.58, 0.62; 0.79, 0.81]
mailles = [1:N]*h

// Calcul de l'intégrale
function y=int_k(a, b, mu)
    if a > b then
        [b, a] = (a, b)
    end
    k = 1
    for i=omega_2'
        if i(1) <= a & i(2) >= b then
            k = mu
            break
        end
    end
    y = k * (b-a)
endfunction

// Calcul du coefficient de A
function y=coeff_a(i, j, mu)
    y = 0
    if i == j then
         y = 2/3 * h +  2/(h**2) * int_k(i*h, (i+1)*h, mu)
    end
    if abs(i-j) == 1 then 
        y = h / 4 - 1/(h**2) * int_k(i*h, j*h, mu)
    end
    
endfunction

// Calcul de la matrice A
function A=calc_A(mu)
    A = zeros(N, N)
    for i=1:N
        for j=1:N
            A(i, j) = coeff_a(i, j, mu)
        end
    end
endfunction

// Calcul de la matrice B
function B=calc_B()
    B = zeros(N)
    for i=1:N
        B(i) = h  * sin(2*%pi*i*h) * (sinc(%pi*h))**2
    end
endfunction

// Calcul de la solution
function X=resoudre(mu)
    A = calc_A(mu)
    B = calc_B()
    X = inv(A) * B
endfunction
