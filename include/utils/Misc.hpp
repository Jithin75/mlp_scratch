#ifndef MISC_HPP
#define MISC_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>

namespace utils
{
    class Misc
    {
        public:
        static std::vector<std::vector<double>> getData(std::string path);
    };
}
#endif // MISC_HPP