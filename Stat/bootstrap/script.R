# setwd("D:\\Ecole\\Projet MAP\\map432")
rm(list=ls())

library(numDeriv); 
library(Matrix); 

#helper function
integrate_matrix <- function(A, steps=1000) {
	res=0
	for (i in seq(0,1,(1/steps)))
		res=res+A(i)
	res=res/steps
	return(res)
}
	
z <- read.table("data.txt", header=T)
# calcul de x = log(1/d)
z$x <- sapply(strsplit(as.character(z$d), split = "/"), function(x) -log(as.numeric(x[1]) / as.numeric(x[2]), 10))
print(z)

k <- length(z$d)
n <- 2 * k
message(sprintf("n = %d", n))

f <- function(x, theta)
theta[1] + (theta[2]-theta[1])/(1+exp(theta[3]*(x-theta[4])))

Mn_mai <- function(theta)
(1/n) *  sum( (z$mai1 - f(z$x, theta))**2 + (z$mai2 - f(z$x, theta))**2 )

Mn_juin <- function(theta)
(1/n) *  sum( (z$juin1 - f(z$x, theta))**2 + (z$juin2 - f(z$x, theta))**2 )

theta_mai <- nlm(Mn_mai, p = c(0,2,2.5,3.5), hessian=TRUE)$estimate
theta_juin <- nlm(Mn_juin, p = c(0,2,2.5,3.5), hessian=TRUE)$estimate
theta <- c(theta_mai, theta_juin)

message("theta = ")
print(matrix(c(theta_mai, theta_juin), nrow=4, ncol=2, byrow=F))

A = matrix(rep(0, 3*8), nrow=3, ncol=8, byrow=T)   
for (i in 1:3) { 
	A[i, i] <- 1
	A[i, i+4] <- -1
}
message("A = ")
print(A)

sigma2 =  (Mn_mai(theta_mai) + Mn_juin(theta_juin))/2
message( sprintf("Sigma^2 = %e", sigma2) )

grad_f <- function(x, theta) {
	gradient <- grad( (function(t)  f(x, t) ) , theta )
}
H_theta_mois = function(theta_mois)
					integrate_matrix(function(x){
						gradient <-grad_f(x, theta_mois);
						return ( gradient %*% t(gradient) );
					})
					
H_theta = bdiag( H_theta_mois(theta_mai), H_theta_mois(theta_juin))
message("H(theta) = ")
print(H_theta)

V_theta = A  %*% solve(H_theta)  %*% t(A)
A_theta = A %*% theta
T_n =  as.numeric(t(A_theta) %*% solve(V_theta) %*% A_theta) * (n/sigma2)


alpha <- 0.05
beta <- qchisq(1-alpha, 3)
message("Tn = ", T_n, " beta = ", beta)


# Calcul de rho

theta_tilde <- nlm(function(t) Mn_mai(t[1:4]) + Mn_juin(c(t[1:3], t[5])), p = c(0,2,2.5,3.5, 3.5), hessian=TRUE)$estimate
theta_tilde <- c( theta_tilde[1:4], theta_tilde[1:3], theta_tilde[5] )
theta_tilde <- matrix(theta_tilde, nrow=4, ncol=2, byrow=F)
colnames(theta_tilde) <- c("mai", "juin")
message("theta tilde = ")
print(theta_tilde)

rho <- 10**(theta_tilde[4,"juin"] - theta_tilde[4,"mai"])
message("rho = ", rho)


# Partie 3, bootstrap -----------------------------------------------------

# mai
epsilon_tilde_mai <- c( z$mai1 - f(z$x, theta_tilde[,1]), z$mai2 - f(z$x, theta_tilde[,1]))
# juin
epsilon_tilde_juin <- c( z$juin1 - f(z$x, theta_tilde[,1]), z$juin2 - f(z$x, theta_tilde[,1]))

epsilon_tilde <- cbind(epsilon_tilde_mai, epsilon_tilde_juin)
epsilon_tilde <- epsilon_tilde  - mean(epsilon_tilde)
print(epsilon_tilde)

Mn <- function(theta, Y) (1./n) *  sum( (Y - f(c(z$x, z$x), theta))**2 )
B <- 1000
rho_b <- rho
for( i in 1:B ) {
	Y_mai <- f(z$x, theta_tilde[,"mai"]) + sample(epsilon_tilde, n, replace=T)
	Y_juin <- f(z$x, theta_tilde[,"juin"]) + sample(epsilon_tilde, n, replace=T)
	theta_b <- nlm(function(t) 
		Mn(t[1:4], Y_mai) + Mn(c(t[1:3], t[5]), Y_juin), 
		p = c(0,2,2.5,3.5, 3.5),
		hessian=TRUE)$estimate
	theta_b <- c( theta_b[1:4], theta_b[1:3], theta_b[5] )
	theta_b <- matrix(theta_b, nrow=4, ncol=2, byrow=F)
	colnames(theta_b) <- c("mai", "juin")
	rho_b <- c(rho_b, 10**(theta_b[4,"juin"] - theta_b[4,"mai"]))
}

FB <- function(alpha) quantile(rho_b, probs=alpha)

message(sprintf("Intervalle de confiance a 5 = [ %f, %f ]\n", FB(alpha/2), FB(1-alpha/2)))

# Calcul de la variance de rho
nu_1 <- solve(H_theta_mois(theta_tilde[,"juin"]))[4,4]
nu_2 <- solve(H_theta_mois(theta_tilde[,"mai"]))[4,4]

s_chapeau <- log(10) * (10**(theta_tilde[4,"juin"] - theta_tilde[4,"mai"])) * sqrt(sigma2 *(nu_1 + nu_2) )

message(sprintf("s^ = %f", s_chapeau))


