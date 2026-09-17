- 创建 ./include/pch.hpp
- 写入如下代码:
`#pragma once`

`#include <bits/stdc++.h>`

- `cd ./include && g++ -std=c++20 pch.hpp -o pch.hpp.gch`

- 后续编译其他代码时，加入 `-include "./include/pch.hpp"` 参数即可