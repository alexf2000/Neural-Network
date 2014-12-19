#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main() {
   //training set for XOR - if in1 == in2, out = 0. if in1 != in2, out = 1

   cout << "topology: 2 4 1" << endl;
   for(int i = 2000, i >= 0; i--) {
      int n1 = (int)(2.0 * rand() / double(RAND_MAX));
      int n2 = (int)(2.0 * rand() / double(RAND_MAX));

      int target = n1 ^ n2; 
      cout << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
      cout << "out: " << target << ".0" << endl;
   }

}