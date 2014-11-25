#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>

using namespace std;

int main()
{
  int k = 5;
  double p = 0.2;

  // Number of independant simulations
  int M = 1e6;

  // generator of X_i
  random_device rd;
  mt19937 gen(rd());
  map<int, bernoulli_distribution> alea;
  // generator of X_i conditional to  X_{i-1} = 1
  alea[1] = std::bernoulli_distribution(p);
  // generator of X_i conditional to  X_{i-1} = -1
  alea[-1] = std::bernoulli_distribution(1 - p);
  
  double mean = 0;
  for (int m = 0; m < M; m++) {
    int S = 0;
    int X0 = 1;
    for (int i = 0; i < k && S >= 0; i++) {
      X0 = 2*alea[X0](gen)-1;
      S += X0;
    }
    mean += S;
  }
  
  mean /= M;
  cout << mean << endl;
  
return 0;
}
