//Simple example of basic use of flame math library...
//This code is provided under MIT Licence.

#include <iostream>
#include "flame/flame.h"

using namespace flame;

int main() 
{
    Vec3 a(1,2,3);        //vector declaration
    Vec3 b(4,5,6);

    auto c = a + b;      //operations are same as default math in C++
    float d = dot(a, b); //Dot-Product operation

    std::cout << "Dot: " << d << std::endl;
    std::cout << "Result: " << c.x << ", " << c.y << ", " << c.z << std::endl;

    return 0;
}
