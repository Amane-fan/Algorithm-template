#include <bits/stdc++.h>
using namespace std;

int main() {

    system("g++ -std=c++20 main.cpp -o main");
    system("g++ -std=c++20 main__Good.cpp -o main__Good");
    system("g++ -std=c++20 main__Generator.cpp -o main__Generator");

    int t = 0;
    while (true) {
        cout << "test: " << t++ << endl;
        system("./main__Generator > main.in");
        system("./main < main.in > main.out");
        system("./main__Good < main.in > main__Good.out");

        // linux使用diff，windows使用fc
        if (system("diff main.out main__Good.out")) {
            cout << "WA" << endl;
            return 0;
        }
    }

    return 0;
}