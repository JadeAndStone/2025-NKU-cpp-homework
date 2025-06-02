@echo off
REM 启用 MSVC 编译环境
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM 获取传入的源文件路径（支持空格）
set "SRC=%~1"

REM 编译命令（折行时确保 ^ 后无空格）
cl /EHsc /std:c++17 ^
    /I "E:\libtorch-win-shared-with-deps-2.7.0+cu128\libtorch\include" ^
    /I "E:\libtorch-win-shared-with-deps-2.7.0+cu128\libtorch\include\torch\csrc\api\include" ^
    "%SRC%" ^
    /Fe:main.exe ^
    /link /LIBPATH:"E:\libtorch-win-shared-with-deps-2.7.0+cu128\libtorch\lib" ^
    torch.lib torch_cpu.lib torch_cuda.lib c10.lib

REM 编译成功后，新开一个命令行窗口执行main.exe，且窗口保持打开
start cmd /k main.exe
